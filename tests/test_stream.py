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
