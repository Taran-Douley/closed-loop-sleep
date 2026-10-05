"""Streaming pipeline == offline pipeline; the resampler is causal and nearly delay-free."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from cloop.stream import Resampler, StreamingLoop
from test_pipeline import _fitted


def _stream(loop, eeg, fs, packet):
    live = StreamingLoop(loop, fs)
    out = []
    for i in range(0, len(eeg), packet):
        out += live.push(eeg[i : i + packet])
    return out


def test_streaming_matches_offline_exactly():
    loop, eeg, fs, _ = _fitted()
    off = loop.plan(eeg, fs)
    on = _stream(loop, eeg, fs, loop.detector.chunk)
    assert len(on) == len(off.times_s) > 30
    assert np.array_equal([c.heard_s for c in on], off.times_s)
    assert np.array_equal([c.available_idx for c in on], off.triggers.available_idx)


def test_any_packet_size_gives_the_same_clicks():
    loop, eeg, fs, _ = _fitted()
    off = loop.plan(eeg, fs)
    for packet in (1, 12, 4096):  # sample by sample, a Muse BLE packet, a burst after a stall
        on = _stream(loop, eeg, fs, packet)
        assert np.array_equal([c.heard_s for c in on], off.times_s), packet


def test_resampler_is_causal_and_nearly_delay_free():
    fs_in = 256.0
    t = np.arange(int(120 * fs_in)) / fs_in
    x = 50 * np.sin(2 * np.pi * 1.0 * t) + 10 * np.sin(2 * np.pi * 12 * t)
    rs = Resampler(fs_in)
    out = np.concatenate([rs.push(x[i : i + 12]) for i in range(0, len(x), 12)])
    assert abs(len(out) - 120 * 100) <= 1
    ref = 50 * np.sin(2 * np.pi * 1.0 * np.arange(len(out)) / 100)
    lag = np.argmax(np.correlate(out[2000:4000], ref[2000:4000], "full")) - 1999
    assert abs(lag) <= 1  # <= 10 ms at 100 Hz
    cut = 6000
    x2 = x.copy()
    x2[int(cut / 100 * fs_in) + 1 :] = 0
    rs2 = Resampler(fs_in)
    out2 = np.concatenate([rs2.push(x2[i : i + 12]) for i in range(0, len(x2), 12)])
    assert np.allclose(out[:cut], out2[:cut])


def test_resampler_delay_is_compensated():
    from scipy.signal import resample_poly

    loop, eeg, fs, _ = _fitted()
    off = loop.plan(eeg, fs)
    x = resample_poly(eeg, 256, 100)  # as a 256 Hz headband would record it
    rs = Resampler(256.0)
    assert 0.005 < rs.delay_s < 0.015
    y = np.concatenate([rs.push(x[i : i + 12]) for i in range(0, len(x), 12)])
    t_off = {int(k): t for k, t in zip(off.triggers.trigger_idx, off.times_s)}
    runs = {}
    for name, kw in (("raw", {}), ("comp", dict(signal_delay_s=rs.delay_s, extra_wait_s=rs.wait_s))):
        live = StreamingLoop(loop, fs, **kw)
        clicks = [c for i in range(0, len(y), 5) for c in live.push(y[i : i + 5])]
        lead = np.array([c.heard_s + kw.get("signal_delay_s", 0) - c.available_idx / fs for c in clicks])
        assert lead.min() >= loop.latency_s + kw.get("signal_delay_s", 0) + kw.get("extra_wait_s", 0) - 1e-9
        runs[name] = {c.trigger_idx: c for c in clicks}
    # The chosen delays differ between the runs (the larger budget moves the latency floor, and
    # the timing model's stretch is anchored to it), so compare bookkeeping, not delays: a
    # compensated click is heard delay_s earlier than its chosen delay alone would put it.
    shared = runs["raw"].keys() & runs["comp"].keys()
    assert len(shared) > 0.8 * len(off.times_s)
    for name, shift in (("raw", 0.0), ("comp", rs.delay_s)):
        for c in runs[name].values():
            assert np.isclose(c.heard_s, c.available_idx / fs + c.delay_s - shift)

    def err(name):  # click time vs offline on the original signal, matching triggers within 2 samples
        cs = runs[name].values()
        return np.array([c.heard_s - t_off[k] for c in cs for k in range(c.trigger_idx - 2, c.trigger_idx + 3) if k in t_off])

    assert abs(np.mean(err("comp"))) < abs(np.mean(err("raw")))
