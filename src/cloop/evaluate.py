"""Score sound times against ground-truth slow-oscillation phase.

Ground truth uses a zero-phase (forward-backward) filter + Hilbert transform,
which needs the future and so is impossible live -- that's the point: it tells
us what phase the brain was *actually* in when the sound played.

Phase convention: 0 = up-state peak (the target), +/-pi = trough.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .data import Recording


def true_phase(rec: Recording, band=(0.5, 2.0)) -> np.ndarray:
    sos = butter(2, band, btype="band", fs=rec.sfreq, output="sos")
    return np.angle(hilbert(sosfiltfilt(sos, rec.eeg)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def score(times: np.ndarray, phase: np.ndarray, rec: Recording, target: float = 0.0) -> dict:
    idx = np.clip(np.round(times * rec.sfreq).astype(int), 0, len(phase) - 1)
    err = wrap(phase[idx] - target)
    stage = rec.stages[idx]
    if len(err) == 0:
        return {"n": 0}
    z = np.mean(np.exp(1j * err))
    return {
        "n": len(err),
        "mean_err_deg": float(np.degrees(np.angle(z))),
        "R": float(np.abs(z)),  # phase-locking: 1 = perfectly consistent, 0 = random
        "mean_cos": float(np.mean(np.cos(err))),  # 1 = always on target, 0 = chance
        "pct_within_45": float(100 * np.mean(np.abs(err) <= np.pi / 4)),  # chance = 25%
        "pct_N2N3": float(100 * np.mean(np.isin(stage, (2, 3)))),
        "pct_N3": float(100 * np.mean(stage == 3)),
        "pct_wake_REM": float(100 * np.mean(np.isin(stage, (0, 4)))),
        "errors": err,
    }
