"""Turn detector triggers into sound times, including hardware latency.

Time frame: seconds of *acquisition* time, i.e. sample n happened at n / fs.

Latency model (the numbers are assumptions until measured in Phase 1):
  * EEG latency: a packet acquired at t arrives at t + L_eeg (BLE + buffering).
    If the device has no hardware timestamps, the software can only estimate
    acquisition time as arrival - mean(L_eeg), so EEG jitter becomes timing error.
  * Audio latency: a play command issued at t is heard at t + L_audio.
  * Compensation: subtract the *mean* latencies when scheduling. Jitter (the SD)
    cannot be compensated, which is why wired audio matters.
  * A stimulus is "late" if its compensated play time had already passed when
    the triggering packet arrived; it is then played immediately.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfreqz

from .detector import Triggers

SECOND_CLICK_S = 1.075  # Ngo et al. 2013: second click ~1.075 s after the first


@dataclass(frozen=True)
class Latency:
    name: str
    eeg_mean: float = 0.0
    eeg_sd: float = 0.0
    audio_mean: float = 0.0
    audio_sd: float = 0.0
    compensate: bool = True


SCENARIOS = [
    Latency("ideal (no latency)"),
    Latency("BLE EEG + wired audio", 0.040, 0.010, 0.015, 0.003),
    Latency("BLE EEG + BT earbuds, compensated", 0.040, 0.010, 0.200, 0.050),
    Latency("BLE EEG + BT earbuds, uncompensated", 0.040, 0.010, 0.200, 0.050, compensate=False),
]


@dataclass(frozen=True)
class Strategy:
    """fixed: sound at trough + delay.
    adaptive: estimate the wave's quarter period from (trough - last zero crossing)
    and aim two quarter periods after the trough (= the next up-state peak), + offset."""

    kind: str  # "fixed" | "adaptive" | "predicted" (per-trigger delay from a timing model)
    delay: float = 0.5  # fixed: seconds after trough
    offset: float = 0.0  # adaptive: constant correction, seconds
    # adaptive: undo the causal filter's frequency-dependent phase shift. A band-pass
    # leads below its centre frequency and lags above it, so the detected trough is
    # early for slow (~0.7 Hz) waves and late for fast ones.
    filter_correction: bool = False
    sw_band: tuple[float, float] = (0.5, 2.0)

    def label(self) -> str:
        if self.kind == "fixed":
            return f"fixed {self.delay * 1000:.0f} ms"
        if self.kind == "predicted":
            return "predicted"
        corr = " + filter corr" if self.filter_correction else ""
        return f"adaptive{corr} {self.offset * 1000:+.0f} ms"

    def target_times(self, trg: Triggers, fs: float) -> np.ndarray:
        trough = trg.trough_idx / fs
        if self.kind == "fixed":
            return trough + self.delay
        if self.kind == "predicted":
            return trough + trg.pred_delay
        quarter = np.clip((trg.trough_idx - trg.zc_idx) / fs, 0.125, 0.5)  # 0.5-2 Hz waves
        target = trough + 2 * quarter + self.offset
        if self.filter_correction:
            f_est = 1.0 / (4 * quarter)
            _, h = sosfreqz(butter(2, self.sw_band, btype="band", fs=fs, output="sos"), worN=f_est, fs=fs)
            target += np.angle(h) / (2 * np.pi * f_est)  # positive angle = filter output leads
        return target


def schedule(trg: Triggers, fs: float, strat: Strategy, lat: Latency, rng: np.random.Generator):
    """Returns (click1_times, click2_times, late_mask) in acquisition seconds."""
    n = len(trg.trough_idx)
    target = strat.target_times(trg, fs)
    l_eeg = np.clip(rng.normal(lat.eeg_mean, lat.eeg_sd, n), 0, None)
    l_audio1 = np.clip(rng.normal(lat.audio_mean, lat.audio_sd, n), 0, None)
    l_audio2 = np.clip(rng.normal(lat.audio_mean, lat.audio_sd, n), 0, None)
    est_eeg = lat.eeg_mean if lat.compensate else 0.0
    est_audio = lat.audio_mean if lat.compensate else 0.0

    arrival = trg.available_idx / fs + l_eeg
    # software's mapping from acquisition time to wall time is off by the EEG jitter
    wanted_issue = target + (l_eeg - est_eeg) - est_audio
    issue = np.maximum(wanted_issue, arrival)
    late = wanted_issue < arrival
    click1 = issue + l_audio1
    click2 = issue + SECOND_CLICK_S + l_audio2
    return click1, click2, late
