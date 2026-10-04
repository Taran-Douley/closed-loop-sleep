"""Sanity check on a synthetic 1 Hz slow oscillation with known phase."""

import numpy as np

from cloop.data import Recording
from cloop.detector import DetectorConfig, detect
from cloop.evaluate import score, true_phase
from cloop.timing import SCENARIOS, Strategy, schedule


def make_rec(freq=1.0, minutes=20, fs=100.0, noise_uv=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(minutes * 60 * fs)) / fs
    eeg = 60 * np.cos(2 * np.pi * freq * t) + rng.normal(0, noise_uv, len(t))
    return Recording(0, eeg, fs, np.full(len(t), 3, dtype=np.int8))


def run(rec, strat, lat=SCENARIOS[0]):
    trg = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=None))
    click1, _, late = schedule(trg, rec.sfreq, strat, lat, np.random.default_rng(0))
    return trg, score(click1, true_phase(rec), rec), late


def test_one_trigger_per_lockout():
    rec = make_rec()
    trg, _, _ = run(rec, Strategy("adaptive"))
    gaps = np.diff(trg.trough_idx) / rec.sfreq
    assert len(trg.trough_idx) > 200
    assert gaps.min() >= DetectorConfig().lockout_s


def test_detected_troughs_are_troughs():
    rec = make_rec()
    trg, _, _ = run(rec, Strategy("adaptive"))
    ph = true_phase(rec)[trg.trough_idx]
    assert np.mean(np.abs(np.abs(ph) - np.pi) < np.radians(20)) > 0.9


def test_adaptive_hits_up_state_across_frequencies():
    for f in (0.7, 1.0, 1.4):
        _, s, _ = run(make_rec(freq=f), Strategy("adaptive", filter_correction=True))
        assert abs(s["mean_err_deg"]) < 30, (f, s["mean_err_deg"])
        assert s["R"] > 0.9


def test_uncompensated_latency_shifts_phase():
    rec = make_rec()
    _, ideal, _ = run(rec, Strategy("adaptive"), SCENARIOS[0])
    _, bt, _ = run(rec, Strategy("adaptive"), SCENARIOS[3])
    # 240 ms at 1 Hz ~ 86 degrees late
    assert bt["mean_err_deg"] - ideal["mean_err_deg"] > 60


def test_uncorrected_adaptive_is_early_for_slow_waves():
    # documents *why* filter_correction exists
    _, s, _ = run(make_rec(freq=0.7), Strategy("adaptive"))
    assert s["mean_err_deg"] < -30


def test_adaptive_threshold_is_amplitude_invariant():
    # same signal at 1x and 3x amplitude (different person / headband / montage)
    base = make_rec(minutes=30)
    rng = np.random.default_rng(1)
    base.eeg *= 1 + 0.5 * rng.standard_normal(len(base.eeg) // 100).repeat(100)[: len(base.eeg)].clip(-1, 1)
    cfg = DetectorConfig(gate="none", threshold="adaptive", adapt_floor_uv=-1.0)
    small = detect(base.eeg, base.sfreq, cfg)
    big = detect(base.eeg * 3, base.sfreq, cfg)
    assert len(small.trough_idx) > 100
    assert abs(len(small.trough_idx) - len(big.trough_idx)) / len(small.trough_idx) < 0.05
    # whereas a fixed threshold picks very different numbers of waves
    fixed = DetectorConfig(gate="none", threshold="fixed", trough_uv=-90.0)
    n_small, n_big = (len(detect(x, base.sfreq, fixed).trough_idx) for x in (base.eeg, base.eeg * 3))
    assert n_big > 3 * n_small


def test_gate_rejects_slow_eye_movements():
    # drowsy-wake-like signal: big 0.3 Hz eye-movement swings + some 1 Hz delta
    fs, minutes = 100.0, 20
    t = np.arange(int(minutes * 60 * fs)) / fs
    rng = np.random.default_rng(0)
    eye = 120 * np.sin(2 * np.pi * 0.3 * t)
    eeg = eye + 30 * np.cos(2 * np.pi * 1.0 * t) + rng.normal(0, 1, len(t))
    assert len(detect(eeg, fs, DetectorConfig(eye_ratio_max=None, sustain_s=0, trigger_rise=None)).trough_idx) > 50  # old gate fooled
    assert len(detect(eeg, fs, DetectorConfig(trigger_rise=None)).trough_idx) == 0


def test_gate_accepts_steady_deep_sleep():
    rec = make_rec(minutes=20, noise_uv=1.0)
    trg = detect(rec.eeg, rec.sfreq, DetectorConfig(threshold="fixed"))  # isolate the gate
    assert len(trg.trough_idx) > 200
    assert trg.trough_idx[0] / rec.sfreq >= DetectorConfig().sustain_s * DetectorConfig().sustain_frac


def test_rising_flank_trigger_fires_after_the_trough_at_the_requested_level():
    from scipy.signal import butter, sosfilt

    rec = make_rec(minutes=10)
    base = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=None))
    late = detect(rec.eeg, rec.sfreq, DetectorConfig(gate="none", threshold="fixed", trigger_rise=0.0))
    sw = sosfilt(butter(2, (0.5, 2.0), btype="band", fs=rec.sfreq, output="sos"), rec.eeg)
    assert len(late.trough_idx) > 0.9 * len(base.trough_idx)
    assert np.all(late.trigger_idx > late.trough_idx)
    assert np.all(sw[late.trigger_idx] >= 0) and np.all(sw[late.trigger_idx - 1] < 0)  # upward zero crossing
    assert np.all(late.available_idx >= late.trigger_idx)


def test_relative_gate_is_causal_and_never_stricter_than_fixed():
    from cloop.detector import _relative_limit

    rng = np.random.default_rng(0)
    fs = 100.0
    v = rng.gamma(2.0, 2.0, int(3600 * fs))
    ok = rng.random(len(v)) < 0.7
    cfg = DetectorConfig(rel_min_history_s=60)
    lim = _relative_limit(v, ok, 2.5, 2.0, fs, cfg)
    assert np.all(lim >= 2.5)
    assert np.all(lim[: int(61 * fs)] == 2.5)  # fixed until there is enough history
    cut = int(1800 * fs)
    tampered = v.copy()
    tampered[cut:] *= 100
    assert np.array_equal(_relative_limit(tampered, ok, 2.5, 2.0, fs, cfg)[: cut + 1], lim[: cut + 1])
