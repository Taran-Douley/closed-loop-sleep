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
from scipy.interpolate import CubicSpline
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
    alpha: float = 10.0  # ridge strength
    target: str = "cos"  # "cos" | "soft": exp(4 (cos - 1)), which rewards near-misses less than cos does
    stretch: float = 1.0  # delays above the latency floor are scaled by this (see fit_stretch)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TimingModel":
        """Y: cos(phase error) at each delay (`targets`), whatever `target` is."""
        if self.target == "soft":
            Y = np.exp(4 * (Y - 1))
        if self.kind == "ridge":
            reg = Ridge(alpha=self.alpha)
        elif self.kind == "mlp":
            reg = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                               max_iter=300, random_state=self.seed)
        else:
            from .gru import GRURegressor

            self.pipe = GRURegressor(seed=self.seed).fit(X, Y)  # scales per channel, not per column
            return self
        self.pipe = make_pipeline(StandardScaler(), reg).fit(X, Y)
        return self

    @property
    def window_s(self) -> float:
        """Signal window the model was trained on (from its input width; 100 Hz features)."""
        n = getattr(self.pipe, "n_features_in_", None)
        return WINDOW_S if n is None else (n - 3) // 2 * DECIMATE / 100.0

    def predict(self, X: np.ndarray, min_delay: float = 0.0, step: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (delay in seconds, confidence = predicted cos at that delay), choosing
        only among delays >= min_delay (e.g. the expected EEG + audio latency).

        step: choose on a finer grid than the model was trained on, by cubic-spline
        interpolation of the predicted curve. For ridge this is what training on the finer
        grid would give (the model is linear and the targets are smooth over 25 ms)."""
        if len(X) == 0:
            return np.empty(0), np.empty(0)
        curve, delays = self.pipe.predict(X), self.delays
        if step is not None:
            delays = np.arange(self.delays[0], self.delays[-1] + 1e-9, step)
            curve = curve @ _spline_matrix(self.delays, delays).T
        allowed = delays >= min_delay - 1e-9
        curve = np.where(allowed[None, :], curve, -np.inf)
        best = curve.argmax(axis=1)
        delay = delays[best]
        if self.stretch != 1.0:
            floor = delays[allowed.argmax()]
            delay = np.clip(floor + self.stretch * (delay - floor), floor, delays[-1])
        return delay, curve[np.arange(len(curve)), best]

    def fit_stretch(self, X: np.ndarray, Y: np.ndarray, min_delay: float, step: float | None = None) -> "TimingModel":
        """Ridge shrinks every wave's curve toward the average one, whose peak is near the
        latency floor, so for slower waves it picks too short a delay (clicks ~30° early).
        Scale delays above the floor by the factor that maximises clicks within ±45° on
        (X, Y), the training data. Y as in `fit`; its cos is spline-interpolated between
        grid points to score delays off the grid."""
        self.stretch = 1.0
        d, _ = self.predict(X, min_delay, step)
        grid = self.delays if step is None else np.arange(self.delays[0], self.delays[-1] + 1e-9, step)
        floor = grid[(grid >= min_delay - 1e-9).argmax()]
        best_k, best_hit = 1.0, -1.0
        for k in np.arange(0.9, 1.81, 0.05):
            dk = np.clip(floor + k * (d - floor), floor, self.delays[-1])
            hit = np.mean(_interp_rows(Y, self.delays, dk) >= np.cos(np.pi / 4))
            if hit > best_hit + 1e-12:
                best_k, best_hit = float(k), hit
        self.stretch = best_k
        return self


def _spline_matrix(x: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """(len(x_new), len(x)) matrix taking values at x to their cubic spline at x_new."""
    key = (x.tobytes(), x_new.tobytes())
    if key not in _SPLINES:
        _SPLINES[key] = CubicSpline(x, np.eye(len(x)), axis=0)(x_new)
    return _SPLINES[key]


_SPLINES: dict = {}


def _interp_rows(Y: np.ndarray, x: np.ndarray, x_row: np.ndarray) -> np.ndarray:
    """Each row of Y (values at x) cubic-spline interpolated at its own point x_row[i]."""
    c = CubicSpline(x, Y, axis=1).c  # (4, len(x) - 1, len(Y)): cubic coefficients per interval
    j = np.clip(np.searchsorted(x, x_row, side="right") - 1, 0, len(x) - 2)
    dx = x_row - x[j]
    rows = np.arange(len(Y))
    return ((c[0, j, rows] * dx + c[1, j, rows]) * dx + c[2, j, rows]) * dx + c[3, j, rows]
