"""The default pipeline end to end: trains, respects latency, and never looks ahead."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from cloop.detector import DetectorConfig, detect
from cloop.evaluate import true_phase
from cloop.pipeline import TIMING_WINDOW_S, ClosedLoop
from cloop.predict import features, targets
from cloop.stage import stage_features
from test_synthetic import make_rec


def _fitted():
    # "sleep" = 1 Hz slow waves, "wake" = noise with 10 Hz alpha; alternate 10-minute blocks
    fs = 100.0
    sleep = make_rec(minutes=10).eeg
    t = np.arange(len(sleep)) / fs
    wake = 15 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(1).normal(0, 5, len(t))
    eeg = np.concatenate([wake, sleep, wake, sleep])
    is_sleep = np.repeat([False, True, False, True], 600)
    trg = detect(eeg, fs, DetectorConfig(gate="oracle"), np.repeat(np.where(is_sleep, 3, 0), int(fs)))
    rec = make_rec(minutes=40)
    rec.eeg = eeg
    X = features(eeg, fs, trg, window_s=TIMING_WINDOW_S)
    loop = ClosedLoop().fit(stage_features(eeg, fs), is_sleep, X, targets(true_phase(rec), fs, trg))
    return loop, eeg, fs, is_sleep


def test_pipeline_clicks_in_sleep_only_and_respects_latency():
    loop, eeg, fs, is_sleep = _fitted()
    c = loop.plan(eeg, fs)
    assert len(c.times_s) > 30  # identical waves: the adaptive "deepest 5%" threshold passes few
    assert is_sleep[(c.times_s).astype(int)].mean() > 0.95
    lead = c.times_s - c.triggers.available_idx / fs
    assert lead.min() >= loop.latency_s - 1e-9
    assert np.allclose(c.times_s - c.issue_s, loop.audio_latency_s)


def test_pipeline_is_causal():
    loop, eeg, fs, _ = _fitted()
    c = loop.plan(eeg, fs)
    cut = int(1500 * fs)
    tampered = eeg.copy()
    tampered[cut:] = np.random.default_rng(2).normal(0, 300, len(eeg) - cut)
    c2 = loop.plan(tampered, fs)
    before = c.triggers.available_idx < cut
    before2 = c2.triggers.available_idx < cut
    assert np.allclose(c.times_s[before], c2.times_s[before2])
