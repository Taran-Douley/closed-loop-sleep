"""Learned, causal sleep-stage gate: "is this N2/N3?" once a second.

Phase 0.6 showed the hand-made gate can't separate wake from sleep in older people on a
single frontal channel: their wake has slow activity and their sleep has little delta,
so fixed band-power ratios fail and person-relative ones let wake in. Here a small
gradient-boosted classifier learns that boundary from the Sleep-EDF hypnograms.

Features, one row per second, using only samples up to the end of that second:
  * log band power (slow 0.16-0.5, delta 0.5-2 and 2-4, theta, alpha, sigma, beta),
    each smoothed with causal EMAs at 10 s and 60 s;
  * the same values minus this person's running median so far tonight, so the model can
    read "more delta than usual for *you*" -- the person-adaptation that fixed ratios lack.
The prediction for second s is applied to the samples of second s + 1 (causal), then the
usual sustained-sleep rule smooths it.
"""

from bisect import insort
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS = [(0.16, 0.5), (0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0)]
TAUS_S = (10.0, 60.0)
MEDIAN_EVERY_S = 10  # refresh the running medians this often (cheap, and plenty)


def stage_features(eeg: np.ndarray, fs: float) -> np.ndarray:
    """(n_seconds, 2 * len(BANDS) * len(TAUS_S)) causal features; row s uses samples < (s + 1) * fs."""
    step = int(fs)
    ends = np.arange(step - 1, len(eeg), step)
    cols = []
    for lo, hi in BANDS:
        p = sosfilt(butter(2, [lo, hi], btype="band", fs=fs, output="sos"), eeg) ** 2
        for tau in TAUS_S:
            a = 1.0 / (tau * fs)
            cols.append(np.log10(lfilter([a], [1, a - 1], p)[ends] + 1e-6))
    raw = np.stack(cols, axis=1)
    return np.hstack([raw, raw - _running_median(raw)])


def _running_median(F: np.ndarray) -> np.ndarray:
    """Per-column median of rows seen so far (refreshed every MEDIAN_EVERY_S rows), causal."""
    out = np.empty_like(F)
    ranked = [[] for _ in range(F.shape[1])]
    med = F[0].copy()
    for s in range(len(F)):
        for j in range(F.shape[1]):
            insort(ranked[j], F[s, j])
        if s % MEDIAN_EVERY_S == 0:
            med = np.array([r[len(r) // 2] for r in ranked])
        out[s] = med
    return out


def gate_from_prob(prob: np.ndarray, n_samples: int, fs: float, threshold: float) -> np.ndarray:
    """Per-sample gate: second s's prediction applies to second s + 1."""
    step = int(fs)
    per = np.repeat(prob >= threshold, step)[: max(n_samples - step, 0)]
    gate = np.zeros(n_samples, bool)
    gate[step : step + len(per)] = per
    return gate


@dataclass
class StageGate:
    """Binary N2/N3 classifier on stage_features."""

    max_iter: int = 300
    seed: int = 0
    clf: HistGradientBoostingClassifier = field(init=False, repr=False)

    def fit(self, F: np.ndarray, is_n2n3: np.ndarray) -> "StageGate":
        self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                                  l2_regularization=1.0, early_stopping=True, random_state=self.seed)
        self.clf.fit(F, is_n2n3)
        return self

    def prob(self, F: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(F)[:, 1] if len(F) else np.empty(0)


def stage_labels(stages: np.ndarray, fs: float) -> np.ndarray:
    """Stage at the last sample of each second (aligned with stage_features rows)."""
    step = int(fs)
    return stages[np.arange(step - 1, len(stages), step)]
