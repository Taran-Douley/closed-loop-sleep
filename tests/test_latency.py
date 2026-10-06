"""The latency tools' analysis recovers known latencies from synthetic recordings."""

import numpy as np

from cloop.audio import pink_click
from cloop.latency import artifact_latency, click_onsets, clock_offset, summary, tap_onsets

FS = 48000
BLOCK = 64


def _blocks(x, t0, wall_offset=0.0, cb_delay=lambda n: 0.002):
    """Split a recording into audio callback blocks: (ADC time, samples, wall time at callback)."""
    out = []
    for k, i in enumerate(range(0, len(x), BLOCK)):
        seg = x[i : i + BLOCK]
        adc = t0 + i / FS
        out.append((adc, seg, adc + (len(seg) - 1) / FS + wall_offset + cb_delay(k)))
    return out


def test_click_onsets_recover_the_latency():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-3, 30 * FS)
    t0 = 100.0  # stream clock at the first sample
    t_cmds = t0 + 1.0 + 0.8 * np.arange(30)
    true = rng.uniform(0.005, 0.030, len(t_cmds))
    click = 0.5 * pink_click(FS)
    for k, (tc, lat) in enumerate(zip(t_cmds, true)):
        if k == 7:
            continue  # this click never plays
        i = int(round((tc + lat - t0) * FS))
        x[i : i + len(click)] += click
    got = click_onsets(_blocks(x, t0), FS, t_cmds) - t_cmds
    assert np.isnan(got[7])
    ok = np.arange(len(t_cmds)) != 7
    assert np.all(np.abs(got[ok] - true[ok]) < 0.0015)  # within the click's 1 ms onset ramp
    s = summary(got)
    assert s["n"] == 29 and s["missed"] == 1


def test_eeg_latency_from_taps():
    rng = np.random.default_rng(1)
    offset = 1.7e9  # wall time - stream time
    t0 = 50.0
    dur = 60.0
    mic = rng.normal(0, 1e-3, int(dur * FS))
    taps = t0 + 3.0 + 2.5 * np.arange(20) + rng.uniform(0, 0.3, 20)  # stream clock
    for tp in taps:
        i = int(round((tp - t0) * FS))
        n = int(0.02 * FS)
        mic[i : i + n] += rng.normal(0, 0.3, n) * np.exp(-np.arange(n) / (0.004 * FS))
    blocks = _blocks(mic, t0, offset, cb_delay=lambda k: 0.0005 + 0.004 * ((k * 7919) % 13) / 13)

    heard = tap_onsets(blocks, FS)
    assert len(heard) == len(taps) and np.all(np.abs(heard - taps) < 0.001)
    off = clock_offset(blocks, FS)
    assert abs(off - (offset + 0.0005)) < 1e-4  # best estimate errs late by the smallest callback delay

    # 256 Hz EEG on the wall clock; the tap shows as a jump at the first sample acquired after it;
    # 12-sample packets arrive 40 ms after their last sample, plus up to 10 ms of BLE jitter
    fs = 256.0
    acq = offset + t0 + np.arange(int(dur * fs)) / fs
    eeg = rng.normal(0, 5, len(acq))
    tap_wall = taps + offset
    first = np.searchsorted(acq, tap_wall)
    for f in first:
        eeg[f : f + 20] += 300 * np.exp(-np.arange(20) / 5)
    arrival = np.empty(len(acq))
    for i in range(0, len(acq), 12):
        arrival[i : i + 12] = acq[min(i + 11, len(acq) - 1)] + 0.040 + rng.uniform(0, 0.010)
    true = arrival[first] - tap_wall
    got = artifact_latency(heard + off, eeg, arrival, fs)
    assert np.all(np.abs(got - true) < 0.002)
    assert 0.040 < np.median(got) < 0.040 + 12 / fs + 0.010


def test_click_player_starts_the_click_in_the_next_block():
    import threading
    from types import SimpleNamespace

    from cloop.audio import ClickPlayer

    p = ClickPlayer.__new__(ClickPlayer)  # no audio device here: drive the callback by hand
    p.record, p.click, p._pending, p._pos, p._lock = False, np.arange(1, 151, dtype=float), False, None, threading.Lock()
    p.stream = SimpleNamespace(time=12.5)
    out, t = np.zeros((BLOCK, 1), np.float32), SimpleNamespace()
    blocks = []
    for k in range(5):
        if k == 1:
            assert p.play() == 12.5
        p._callback(out, BLOCK, t, None)
        blocks.append(out[:, 0].copy())
    played = np.concatenate(blocks)
    assert not played[:BLOCK].any()  # nothing before the command
    assert np.array_equal(played[BLOCK : BLOCK + 150], p.click)  # whole click, across blocks, from the next block
    assert not played[BLOCK + 150 :].any()
