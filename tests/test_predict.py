"""The timing model must only ever see data the live system would have had."""

import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from cloop.detector import DetectorConfig, detect
from cloop.predict import DELAYS, TimingModel, crop, features, targets
from cloop.evaluate import true_phase
from test_synthetic import make_rec


def test_features_ignore_the_future():
    rec = make_rec(minutes=10)
    trg = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=None))
    X = features(rec.eeg, rec.sfreq, trg)
    for i in (5, 50, 100):
        cut = trg.available_idx[i] + 1
        tampered = rec.eeg.copy()
        tampered[cut:] = np.random.default_rng(i).normal(0, 500, len(tampered) - cut)
        assert np.allclose(features(tampered, rec.sfreq, trg)[i], X[i])


def test_model_learns_the_delay_on_clean_waves():
    rec = make_rec(freq=0.8, minutes=30)
    trg = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=None))
    X, Y = features(rec.eeg, rec.sfreq, trg), targets(true_phase(rec), rec.sfreq, trg)
    delay, conf = TimingModel("ridge").fit(X[:200], Y[:200]).predict(X[200:])
    # Not simply half a period (625 ms): the causal filter makes 0.8 Hz troughs look
    # ~90 ms early, and the model should learn that correction by itself.
    best_in_hindsight = DELAYS[Y[200:].mean(0).argmax()]
    assert abs(np.median(delay) - best_in_hindsight) <= 0.025
    assert np.median(conf) > 0.9
    assert DELAYS.min() <= delay.min() and delay.max() <= DELAYS.max()


def test_crop_matches_a_shorter_window():
    rec = make_rec(minutes=10)
    trg = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=None))
    long = features(rec.eeg, rec.sfreq, trg, window_s=8.0)
    assert np.allclose(crop(long, 2.0), features(rec.eeg, rec.sfreq, trg))


def test_gru_learns_the_delay_on_clean_waves():
    pytest.importorskip("torch")
    rec = make_rec(freq=0.8, minutes=30)
    trg = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=None))
    X, Y = features(rec.eeg, rec.sfreq, trg), targets(true_phase(rec), rec.sfreq, trg)
    delay, conf = TimingModel("gru").fit(X[:200], Y[:200]).predict(X[200:])
    best_in_hindsight = DELAYS[Y[200:].mean(0).argmax()]
    assert abs(np.median(delay) - best_in_hindsight) <= 0.05
