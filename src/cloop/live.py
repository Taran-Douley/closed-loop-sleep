"""Phase 1 runner: stream EEG through BrainFlow into the pipeline and LOG what it would do.

No sound is played. Every decision is logged so nights can be scored offline later.

    uv run cloop-live --board muse_s_athena --channel AF7             # Muse S Athena (native BLE; Windows Python)
    uv run cloop-live --board muse_s --channel AF7                    # older Muse S (BLED / native BLE)
    uv run cloop-live --board cyton --serial-port /dev/ttyUSB0 --channel Fp1
    uv run cloop-live --board synthetic --channel Fz --minutes 2      # smoke test
    uv run cloop-export-playback --subject 5 --out night.csv          # Sleep-EDF night -> Muse-format file
                                                                      # (BrainFlow 5.23 can't seek in playback: trim with --start-min/--minutes)
    uv run cloop-live --board playback --file night.csv --channel AF7 # replay it as if live

Writes to logs/<start time>/: clicks.csv (one row per click the pipeline would have
played, with wall-clock issue time and how much slack it had), packets.csv (arrival
times, for EEG latency/jitter), stage.csv (P(N2/N3) per second) and raw.npz (the
resampled signal that was processed).
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .pipeline import ClosedLoop
from .stream import Resampler, StreamingLoop

BOARDS = {"muse_s_athena": "MUSE_S_ATHENA_BOARD", "muse_s": "MUSE_S_BOARD", "muse_2": "MUSE_2_BOARD", "muse_s_bled": "MUSE_S_BLED_BOARD",
          "cyton": "CYTON_BOARD", "synthetic": "SYNTHETIC_BOARD", "playback": "PLAYBACK_FILE_BOARD"}


def open_board(args):
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port or ""
    params.mac_address = args.mac or ""
    board_id = getattr(BoardIds, BOARDS[args.board]).value
    data_id = board_id
    if args.board == "playback":
        params.file = args.file
        params.master_board = getattr(BoardIds, BOARDS[args.master]).value
        data_id = params.master_board
    BoardShim.disable_board_logger()
    board = BoardShim(board_id, params)
    board.prepare_session()
    return board, data_id


def main():
    from brainflow.board_shim import BoardShim

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=BOARDS, default="synthetic")
    ap.add_argument("--channel", default=None, help="EEG channel name (default: first EEG channel)")
    ap.add_argument("--minus", default=None, help="optional reference channel to subtract (bipolar)")
    ap.add_argument("--model", default="models/closed_loop.pkl")
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long (default: until Ctrl+C)")
    ap.add_argument("--serial-port")
    ap.add_argument("--mac")
    ap.add_argument("--file", help="playback: BrainFlow data file")
    ap.add_argument("--master", choices=[b for b in BOARDS if b != "playback"], default="muse_s", help="playback: board the file was recorded with")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    loop = ClosedLoop.load(args.model)
    board, data_id = open_board(args)
    fs = BoardShim.get_sampling_rate(data_id)
    names = BoardShim.get_eeg_names(data_id)
    rows = BoardShim.get_eeg_channels(data_id)
    ch = rows[names.index(args.channel)] if args.channel else rows[0]
    ref = rows[names.index(args.minus)] if args.minus else None
    ts_row = BoardShim.get_timestamp_channel(data_id)

    out = Path(args.logs) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    resampler = Resampler(fs)
    live = StreamingLoop(loop, 100.0, signal_delay_s=resampler.delay_s, extra_wait_s=resampler.wait_s)
    f_clicks = open(out / "clicks.csv", "w", newline="")
    f_packets = open(out / "packets.csv", "w", newline="")
    w_clicks, w_packets = csv.writer(f_clicks), csv.writer(f_packets)
    w_clicks.writerow(["trigger_idx", "available_idx", "delay_s", "heard_s", "issue_s", "confidence",
                       "issue_wall", "decided_wall", "slack_s"])
    w_packets.writerow(["arrival_wall", "n_samples", "first_ts", "last_ts", "process_ms"])
    raw = []
    t0_wall = None  # wall-clock time of acquisition sample 0 (from the first packet's timestamp)
    print(f"streaming {args.board} {names[rows.index(ch)]}{' - ' + args.minus if ref else ''} at {fs} Hz -> {out}  (log only, no sound)")
    board.start_stream()
    started = time.time()
    n_clicks = n_late = 0
    last_arrival = None
    try:
        while args.minutes is None or time.time() - started < 60 * args.minutes:
            data = board.get_board_data()
            if data.shape[1] == 0:
                time.sleep(0.005)
                continue
            arrival = time.time()
            if last_arrival is not None and arrival - last_arrival > 1.0:
                # a stall (BLE dropout, laptop sleep): the backlog arrives at once and every
                # click in it is late. Logged, not played; the slack column shows it.
                print(f"  {datetime.now():%H:%M:%S}  WARNING: no data for {arrival - last_arrival:.1f} s", flush=True)
            last_arrival = arrival
            x = data[ch] - (data[ref] if ref is not None else 0.0)
            if t0_wall is None:
                t0_wall = data[ts_row, 0] if args.board != "playback" else arrival
            y = resampler.push(x)
            clicks = live.push(y)
            decided = time.time()
            raw.append(y)
            w_packets.writerow([f"{arrival:.4f}", data.shape[1], f"{data[ts_row, 0]:.4f}", f"{data[ts_row, -1]:.4f}",
                                f"{1000 * (decided - arrival):.2f}"])
            for c in clicks:
                issue_wall = t0_wall + c.issue_s
                w_clicks.writerow([c.trigger_idx, c.available_idx, f"{c.delay_s:.3f}", f"{c.heard_s:.3f}", f"{c.issue_s:.3f}",
                                   f"{c.confidence:.3f}", f"{issue_wall:.4f}", f"{decided:.4f}", f"{issue_wall - decided:.4f}"])
                n_clicks += 1
                n_late += issue_wall < decided
            if clicks:
                f_clicks.flush()
                print(f"  {datetime.now():%H:%M:%S}  {n_clicks} clicks so far, P(N2/N3) {live.prob_log[-1]:.2f}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()
        f_clicks.close()
        f_packets.close()
        np.savez_compressed(out / "raw.npz", eeg_100hz=np.concatenate(raw) if raw else np.empty(0), fs=100.0,
                            device_fs=fs, t0_wall=t0_wall if t0_wall is not None else np.nan,
                            signal_delay_s=resampler.delay_s)
        np.savetxt(out / "stage.csv", np.array(live.prob_log), fmt="%.4f", header="P(N2/N3) per second", comments="")
        print(f"stopped: {live.n / 100 / 60:.1f} min processed, {n_clicks} clicks logged ({n_late} would have been late) in {out}")


def export_playback():
    """Write a Sleep-EDF night as a BrainFlow file for the playback board (Muse S layout:
    Fpz-Cz upsampled to 256 Hz on AF7; other channels zero)."""
    from brainflow.board_shim import BoardIds, BoardShim
    from brainflow.data_filter import DataFilter
    from scipy.signal import resample_poly

    from .data import load_subject

    ap = argparse.ArgumentParser(description=export_playback.__doc__)
    ap.add_argument("--subject", type=int, default=5)
    ap.add_argument("--night", type=int, default=1)
    ap.add_argument("--master", choices=[b for b in BOARDS if b != "playback"], default="muse_s")
    ap.add_argument("--channel", default="AF7")
    ap.add_argument("--start-min", type=float, default=0.0, help="skip this much of the night (the file starts 30 min before sleep)")
    ap.add_argument("--minutes", type=float, default=None, help="keep this much after --start-min")
    ap.add_argument("--out", default="night.csv")
    args = ap.parse_args()
    rec = load_subject(args.subject, night=args.night)
    a = int(args.start_min * 60 * rec.sfreq)
    b = len(rec.eeg) if args.minutes is None else a + int(args.minutes * 60 * rec.sfreq)
    rec.eeg, rec.stages = rec.eeg[a:b], rec.stages[a:b]
    board_id = getattr(BoardIds, BOARDS[args.master]).value
    fs = BoardShim.get_sampling_rate(board_id)
    x = resample_poly(rec.eeg, int(fs), int(rec.sfreq))  # zero-phase, offline: fine for making test data
    data = np.zeros((BoardShim.get_num_rows(board_id), len(x)))
    names, rows = BoardShim.get_eeg_names(board_id), BoardShim.get_eeg_channels(board_id)
    data[rows[names.index(args.channel)]] = x
    data[BoardShim.get_timestamp_channel(board_id)] = time.time() + np.arange(len(x)) / fs
    DataFilter.write_file(data, args.out, "w")
    np.save(Path(args.out).with_suffix(".stages.npy"), rec.stages)
    print(f"wrote {args.out}: subject {args.subject} night {args.night}, {len(x) / fs / 3600:.1f} h at {fs} Hz on {args.channel}")


if __name__ == "__main__":
    main()


def report():
    """Score a log directory offline: targeting, agreement with the offline pipeline, timing."""
    import pandas as pd
    from scipy.signal import butter, hilbert, sosfiltfilt

    ap = argparse.ArgumentParser(description=report.__doc__)
    ap.add_argument("logdir")
    ap.add_argument("--model", default="models/closed_loop.pkl")
    args = ap.parse_args()
    d = Path(args.logdir)
    z = np.load(d / "raw.npz")
    eeg, fs = z["eeg_100hz"], float(z["fs"])
    clicks = pd.read_csv(d / "clicks.csv")
    packets = pd.read_csv(d / "packets.csv")
    print(f"{d}: {len(eeg) / fs / 60:.1f} min processed, {len(clicks)} clicks")

    # targeting, against the phase of the signal that was actually processed (offline truth).
    # That signal lags the brain by the resampler's delay; heard_s is in brain time.
    lag = float(z["signal_delay_s"]) if "signal_delay_s" in z else 0.0
    phase = np.angle(hilbert(sosfiltfilt(butter(2, (0.5, 2.0), btype="band", fs=fs, output="sos"), eeg)))
    if len(clicks):
        idx = np.clip(np.round((clicks.heard_s.to_numpy() + lag) * fs).astype(int), 0, len(eeg) - 1)
        err = np.angle(np.exp(1j * phase[idx]))
        print(f"  targeting: {100 * np.mean(np.abs(err) <= np.pi / 4):.1f}% within ±45° of the up-state, "
              f"mean error {np.degrees(np.angle(np.mean(np.exp(1j * err)))):+.0f}°")
        print(f"  slack before play time: min {1000 * clicks.slack_s.min():.0f} ms, median {1000 * clicks.slack_s.median():.0f} ms "
              f"({int((clicks.slack_s < 0).sum())} late)")
    # the offline pipeline on the same signal should pick the same waves
    off = ClosedLoop.load(args.model).plan(eeg, fs)
    if len(clicks) or len(off.times_s):
        same = np.intersect1d(clicks.trigger_idx.to_numpy(), off.triggers.trigger_idx)
        print(f"  offline pipeline on the same signal: {len(off.times_s)} clicks, {len(same)} on the same trigger samples")
    gaps = np.diff(packets.arrival_wall.to_numpy())
    print(f"  packets: {len(packets)}, median {packets.n_samples.median():.0f} samples, inter-arrival p50 {1000 * np.median(gaps):.1f} ms / "
          f"p99 {1000 * np.quantile(gaps, 0.99):.1f} ms / max {1000 * gaps.max():.0f} ms")
    print(f"  processing per packet: p50 {packets.process_ms.median():.2f} ms, p99 {packets.process_ms.quantile(0.99):.2f} ms")
