"""Learned timing: predict when to click from the signal leading up to the trough.

Phase 0 showed trough *detection* is accurate but *extrapolating* to the next up-state
is not. So instead of a fixed delay, a small model looks at the last couple of
seconds the live system has actually received and predicts, for each candidate delay,
how well a click there would land (cos of phase error, 1 = on the up-state peak).
We click at the best predicted delay.

Predicting the whole score-vs-delay curve (not a single delay) does two things:
  * it optimises the metric we care about directly, and copes with waves whose
    future is genuinely ambiguous (the curve is flat; a point estimate would be noise);
  * the height of the curve is a confidence: we can choose to *skip* waves the model
    can't predict, trading stimulation dose for accuracy.

Everything here is causal: features use only samples up to the packet that confirmed
the trough. The model is tiny (ridge regression or a ~10k-weight MLP), so inference on
a laptop or Raspberry Pi is microseconds.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .detector import DetectorConfig, Triggers

DELAYS = np.arange(0.15, 0.901, 0.025)  # candidate click delays after the trough, seconds
FLANK_DELAYS = np.arange(0.0, 0.601, 0.025)  # ... after a rising-flank trigger is available
WINDOW_S = 2.0
DECIMATE = 4  # 100 Hz -> 25 Hz for the model input


def features(eeg: np.ndarray, fs: float, trg: Triggers, cfg: DetectorConfig = DetectorConfig(),
             window_s: float = WINDOW_S) -> np.ndarray:
    """One row per trigger, using only data available when the trigger fired:
    [slow-wave band, window_s seconds at 25 Hz | delta band, same | 3 scalars].
    A shorter window is the tail of each sequence (see `crop`)."""
    sos = lambda lo, hi: butter(2, [lo, hi], btype="band", fs=fs, output="sos")  # noqa: E731
    sw = sosfilt(sos(*cfg.sw_band), eeg)
    delta = sosfilt(sos(0.5, 4.0), eeg)
    w = int(window_s * fs)
    rows = []
    for trough, zc, avail in zip(trg.trough_idx, trg.zc_idx, trg.available_idx):
        a = min(avail + 1, len(eeg))
        scale = max(abs(sw[trough]), 1e-6)  # normalise by this wave's depth -> amplitude invariant
        seg_sw = np.zeros(w)
        seg_d = np.zeros(w)
        seg = slice(max(a - w, 0), a)
        seg_sw[w - (seg.stop - seg.start) :] = sw[seg] / scale
        seg_d[w - (seg.stop - seg.start) :] = delta[seg] / scale
        rows.append(np.concatenate([
            seg_sw[::DECIMATE],
            seg_d[::DECIMATE],
            [np.log(scale), (trough - zc) / fs, (a - 1 - trough) / fs],
        ]))
    return np.array(rows) if rows else np.empty((0, 2 * w // DECIMATE + 3))


def crop(X: np.ndarray, window_s: float, fs: float = 100.0) -> np.ndarray:
    """Cut features made with a longer window down to the last `window_s` seconds."""
    n = (X.shape[1] - 3) // 2
    k = int(window_s * fs) // DECIMATE
    return np.hstack([X[:, n - k : n], X[:, 2 * n - k : 2 * n], X[:, -3:]])


def anchor(trg: Triggers) -> tuple[np.ndarray, np.ndarray]:
    """(sample each delay counts from, candidate delays): the trough for trough triggers,
    the moment the trigger is available for rising-flank triggers."""
    if trg.trigger_idx is None:
        return trg.trough_idx, DELAYS
    return trg.available_idx, FLANK_DELAYS


def targets(phase: np.ndarray, fs: float, trg: Triggers) -> np.ndarray:
    """cos(phase error) for a click at each candidate delay (ground truth, offline only)."""
    start, delays = anchor(trg)
    idx = start[:, None] + np.round(delays * fs).astype(int)[None, :]
    return np.cos(phase[np.clip(idx, 0, len(phase) - 1)])


@dataclass
class TimingModel:
    kind: str = "ridge"  # "ridge" | "mlp" | "gru" (needs the `gru` extra: torch)
    seed: int = 0
    delays: np.ndarray = field(default_factory=lambda: DELAYS)  # FLANK_DELAYS for rising-flank triggers

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TimingModel":
        if self.kind == "ridge":
            reg = Ridge(alpha=10.0)
        elif self.kind == "mlp":
            reg = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                               max_iter=300, random_state=self.seed)
        else:
            from .gru import GRURegressor

            self.pipe = GRURegressor(seed=self.seed).fit(X, Y)  # scales per channel, not per column
            return self
        self.pipe = make_pipeline(StandardScaler(), reg).fit(X, Y)
        return self

    def predict(self, X: np.ndarray, min_delay: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Returns (delay in seconds, confidence = predicted cos at that delay), choosing
        only among delays >= min_delay (e.g. the expected EEG + audio latency)."""
        if len(X) == 0:
            return np.empty(0), np.empty(0)
        curve = np.where(self.delays[None, :] >= min_delay - 1e-9, self.pipe.predict(X), -np.inf)
        best = curve.argmax(axis=1)
        return self.delays[best], curve[np.arange(len(curve)), best]
