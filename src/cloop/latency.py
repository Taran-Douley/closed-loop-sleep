"""Measure the two latencies the pipeline assumes (ClosedLoop.eeg_latency_s / audio_latency_s).

Audio: play command -> sound at the ear. Headphones on a table with the microphone
pressed to one ear cup (or a cable from the headphone jack to the mic input), quiet room:

    uv run --extra audio cloop-audio-latency --n 40

EEG: something happens at the head -> the sample showing it reaches the software. Wear
the headband with the microphone close to it and tap the headband firmly next to the
sensor (AF7 on a Muse) with a fingernail or pen each time you're prompted. The mic hears
the tap; the EEG shows it as a spike.

    uv run --extra audio cloop-eeg-latency --board muse_s_athena --channel AF7

Both print the median (use it in `cloop-train --eeg-latency/--audio-latency`), the
spread (jitter, which costs accuracy on its own: Phase 0) and the 95th percentile (how
late the worst clicks are). Raw recordings go to logs/latency-*/ so the numbers can be
re-derived.

Run on Windows Python, not WSL: WSL can't see the Bluetooth adapter or audio devices.
Timing is on two clocks: the audio stream's (PortAudio) and the wall clock (BrainFlow
packet arrival). `clock_offset` maps one to the other from the audio callbacks; the
mapping can only err late by the smallest callback delay (~1 ms), which makes EEG
latency read that much too low.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

Blocks = list[tuple[float, np.ndarray, float]]  # (ADC time of first sample, samples, wall time at callback)


def _flatten(blocks: Blocks, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples and their ADC times (per block, so a dropped block leaves a gap, not an error)."""
    x = np.concatenate([b[1] for b in blocks])
    t = np.concatenate([b[0] + np.arange(len(b[1])) / fs for b in blocks])
    return x, t


def _envelope(x: np.ndarray, fs: float, hp_hz: float) -> np.ndarray:
    return np.abs(sosfilt(butter(2, hp_hz, btype="high", fs=fs, output="sos"), x))


def click_onsets(blocks: Blocks, fs: float, t_cmds: np.ndarray, max_s: float = 0.5, rel: float = 0.1,
                 snr: float = 10.0) -> np.ndarray:
    """Stream-clock time each commanded click was first heard (NaN if not found).
    Onset = first sample reaching `rel` x the click's peak, searched from the command on;
    a click must stand `snr` x above the 50 ms before the command."""
    x, t = _flatten(blocks, fs)
    env = _envelope(x, fs, 100.0)
    out = np.full(len(t_cmds), np.nan)
    for i, tc in enumerate(t_cmds):
        pre = env[(t >= tc - 0.05) & (t < tc)]
        win = np.flatnonzero((t >= tc) & (t < tc + max_s))
        if len(win) == 0 or len(pre) == 0:
            continue
        peak = env[win].max()
        if peak < snr * max(np.sqrt(np.mean(pre**2)), 1e-12):
            continue
        out[i] = t[win[np.argmax(env[win] >= rel * peak)]]
    return out


def tap_onsets(blocks: Blocks, fs: float, refractory_s: float = 1.0, snr: float = 15.0, rel: float = 0.1) -> np.ndarray:
    """Stream-clock times of taps heard by the microphone (unprompted: no command times)."""
    x, t = _flatten(blocks, fs)
    env = _envelope(x, fs, 200.0)
    noise = np.median(env) / 0.6745  # robust noise level (MAD of a ~zero-mean signal)
    loud = np.flatnonzero(env > snr * noise)
    out, i = [], 0
    while i < len(loud):
        start = loud[i]
        seg = slice(max(start - int(0.01 * fs), 0), start + int(0.05 * fs))  # the tap's first 50 ms
        peak = env[seg].max()
        first = seg.start + np.argmax(env[seg] >= rel * peak)
        out.append(t[first])
        i = np.searchsorted(loud, start + int(refractory_s * fs))
    return np.array(out)


def clock_offset(blocks: Blocks, fs: float) -> float:
    """wall time - stream time. A callback runs after its block's last sample was captured,
    so the smallest (wall at callback - ADC time of last sample) is the best estimate."""
    return min(b[2] - (b[0] + (len(b[1]) - 1) / fs) for b in blocks)


def artifact_latency(tap_wall: np.ndarray, eeg: np.ndarray, arrival: np.ndarray, fs: float,
                     max_s: float = 1.0, z: float = 8.0) -> np.ndarray:
    """For each tap (wall time), wall time from the tap to the arrival of the first EEG sample
    that shows it (a jump in the signal > z robust SDs), NaN if none arrives within max_s.
    arrival: wall time each EEG sample reached the software (its packet's arrival)."""
    d = np.abs(np.diff(eeg, prepend=eeg[0]))
    scale = np.median(d) / 0.6745 + 1e-12
    spike = d > z * scale
    out = np.full(len(tap_wall), np.nan)
    for i, tw in enumerate(tap_wall):
        # the artefact can't have arrived before the tap; samples acquired well before it
        # can still arrive after it, so look among samples that arrived in the window and
        # take the first spike among those acquired no earlier than ~0.2 s before the tap
        cand = np.flatnonzero((arrival >= tw) & (arrival < tw + max_s))
        if len(cand) == 0:
            continue
        lo = max(cand[0] - int(0.2 * fs), 0)
        hits = np.flatnonzero(spike[lo : cand[-1] + 1]) + lo
        hits = hits[arrival[hits] >= tw]
        if len(hits):
            out[i] = arrival[hits[0]] - tw
    return out


def summary(lat_s: np.ndarray) -> dict:
    ok = lat_s[np.isfinite(lat_s)]
    if len(ok) == 0:
        return {"n": 0, "missed": len(lat_s)}
    return {"n": len(ok), "missed": int(len(lat_s) - len(ok)), "median_ms": 1000 * np.median(ok),
            "sd_ms": 1000 * np.std(ok), "p5_ms": 1000 * np.percentile(ok, 5), "p95_ms": 1000 * np.percentile(ok, 95),
            "max_ms": 1000 * ok.max()}


def _print_summary(what: str, s: dict, flag: str) -> None:
    if s["n"] == 0:
        print(f"{what}: nothing detected ({s['missed']} attempts). Louder clicks/taps, quieter room, mic closer?")
        return
    print(f"{what}: median {s['median_ms']:.1f} ms, SD {s['sd_ms']:.1f} ms, 5-95% {s['p5_ms']:.1f}-{s['p95_ms']:.1f} ms, "
          f"max {s['max_ms']:.1f} ms  ({s['n']} measured, {s['missed']} missed)")
    print(f"  -> uv run cloop-train {flag} {s['median_ms'] / 1000:.3f}")


def _out_dir(kind: str, logs: str) -> Path:
    out = Path(logs) / f"latency-{kind}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    return out


def audio_main():
    from .audio import ClickPlayer

    ap = argparse.ArgumentParser(description="Measure play command -> sound latency.")
    ap.add_argument("--n", type=int, default=40, help="clicks to play")
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--block", type=int, default=64, help="audio block size (samples); smaller = lower latency")
    ap.add_argument("--volume", type=float, default=0.1, help="fraction of full scale (headphones off your head!)")
    ap.add_argument("--device", default=None, help="sounddevice device name/index (default: system default)")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    p = ClickPlayer(args.fs, args.block, args.volume, record=True, device=args.device)
    print(f"{p.host_api}, {args.fs} Hz, block {args.block}; reported output latency "
          f"{1000 * p.stream.latency[1]:.1f} ms. Playing {args.n} clicks ...")
    rng = np.random.default_rng()
    t_cmds = []
    try:
        time.sleep(1.0)
        for _ in range(args.n):
            t_cmds.append(p.play())
            time.sleep(rng.uniform(0.6, 1.0))
        time.sleep(0.6)
    finally:
        p.close()
    t_cmds = np.array(t_cmds)
    lat = click_onsets(p.blocks, args.fs, t_cmds) - t_cmds
    out = _out_dir("audio", args.logs)
    x, t = _flatten(p.blocks, args.fs)
    np.savez_compressed(out / "audio.npz", mic=x, mic_t=t, t_cmds=t_cmds, latency_s=lat, fs=args.fs, block=args.block)
    _print_summary("audio latency (command -> sound)", summary(lat), "--audio-latency")
    print(f"  raw data: {out}")


def eeg_main():
    from brainflow.board_shim import BoardShim

    from .audio import ClickPlayer
    from .live import BOARDS, open_board

    ap = argparse.ArgumentParser(description="Measure EEG latency by tapping the headband while a microphone listens.")
    ap.add_argument("--board", choices=[b for b in BOARDS if b not in ("playback", "synthetic")], default="muse_s_athena")
    ap.add_argument("--channel", default=None, help="EEG channel nearest the tap (default: first)")
    ap.add_argument("--taps", type=int, default=20)
    ap.add_argument("--every", type=float, default=3.0, help="seconds between prompts")
    ap.add_argument("--serial-port")
    ap.add_argument("--mac")
    ap.add_argument("--fs", type=int, default=48000, help="microphone sample rate")
    ap.add_argument("--device", default=None, help="sounddevice device for the microphone")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    board, data_id = open_board(args)
    fs = BoardShim.get_sampling_rate(data_id)
    names, rows = BoardShim.get_eeg_names(data_id), BoardShim.get_eeg_channels(data_id)
    ch = rows[names.index(args.channel)] if args.channel else rows[0]
    ts_row = BoardShim.get_timestamp_channel(data_id)
    mic = ClickPlayer(args.fs, 256, volume=0.0, record=True, device=args.device)  # recording only, plays nothing
    eeg, arrival, bf_ts = [], [], []
    board.start_stream()
    print(f"{names[rows.index(ch)]} at {fs} Hz; mic via {mic.host_api}. Settling for 5 s, then tap the headband "
          f"next to the sensor at each TAP ({args.taps} taps).")
    try:
        start, next_prompt, n = time.time(), time.time() + 5.0, 0
        while n < args.taps or time.time() < next_prompt:
            if time.time() >= next_prompt and n < args.taps:
                n += 1
                print(f"  TAP {n}/{args.taps}", flush=True)
                next_prompt += args.every
            data = board.get_board_data()
            if data.shape[1]:
                now = time.time()
                eeg.append(data[ch])
                arrival.append(np.full(data.shape[1], now))
                bf_ts.append(data[ts_row])
            time.sleep(0.002)
    finally:
        board.stop_stream()
        board.release_session()
        mic.close()
    eeg, arrival, bf_ts = np.concatenate(eeg), np.concatenate(arrival), np.concatenate(bf_ts)
    off = clock_offset(mic.blocks, args.fs)
    taps = tap_onsets(mic.blocks, args.fs) + off
    taps = taps[taps > start + 2.0]  # ignore anything while settling
    lat = artifact_latency(taps, eeg, arrival, fs)
    lat_bf = artifact_latency(taps, eeg, bf_ts, fs)
    out = _out_dir("eeg", args.logs)
    x, t = _flatten(mic.blocks, args.fs)
    np.savez_compressed(out / "eeg.npz", mic=x, mic_t=t, clock_offset=off, eeg=eeg, arrival=arrival, brainflow_ts=bf_ts,
                        fs=fs, taps_wall=taps, latency_s=lat, latency_brainflow_s=lat_bf)
    print(f"heard {len(taps)} taps")
    _print_summary("EEG latency (tap -> sample reaches the software)", summary(lat), "--eeg-latency")
    s = summary(lat_bf)
    if s["n"]:
        print(f"  (by BrainFlow's own timestamps: median {s['median_ms']:.1f} ms; the difference is time spent "
              f"waiting to be polled)")
    print(f"  raw data: {out}")
