"""The default closed-loop pipeline (Phase 0.7 + timing accuracy), as one object to train and run.

    learned stage gate (p >= 0.8) -> adaptive threshold, rising-flank trigger (r = +0.5)
    -> ridge timing model (6 s window, soft targets, stretch; delay >= expected latency, 5 ms grid) -> optional confidence cut

Held-out Sleep-EDF, all ages, 55 ms latency: ~62% of clicks within ±45° of the up-state,
~41 on-target clicks per hour of N2+N3, < 1% of clicks in wake/REM (README).

    uv run cloop-train                      # fit on all 153 nights, save models/closed_loop.pkl
    loop = ClosedLoop.load("models/closed_loop.pkl")
    clicks = loop.plan(eeg_uv, fs)          # offline: every click time for a recording

`plan` processes a whole recording at once; everything in it is causal, so the same
click times come out of a streaming implementation (Phase 1).
"""

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .detector import DetectorConfig, Triggers, detect
from .predict import FLANK_DELAYS, TimingModel, features
from .stage import StageGate, gate_from_prob, stage_features


TIMING_WINDOW_S = 6.0  # signal window for the timing model (2 s until the window sweep; README)


def fit_timing(X: np.ndarray, Y: np.ndarray, latency_s: float, step: float | None) -> TimingModel:
    """The default timing model: ridge (alpha 1) on soft targets, delays stretched above the
    latency floor by a factor fitted on the same data (~1.3). Held-out subjects, all real
    N2/N3 waves, 55 ms: +1.9 points within ±45° over ridge on cos targets (README)."""
    return TimingModel("ridge", delays=FLANK_DELAYS, alpha=1.0, target="soft").fit(X, Y).fit_stretch(X, Y, latency_s, step)


@dataclass
class Clicks:
    times_s: np.ndarray  # when the sound should be *heard*, in acquisition time
    issue_s: np.ndarray  # when to send the play command (times_s - expected audio latency)
    confidence: np.ndarray  # predicted cos(phase error) at the chosen delay
    triggers: Triggers


@dataclass
class ClosedLoop:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    stage_threshold: float = 0.8  # learned gate opens at P(N2/N3) >= this
    eeg_latency_s: float = 0.040  # acquisition -> trigger reaches software
    audio_latency_s: float = 0.015  # play command -> sound
    keep: float = 1.0  # click on this fraction of waves (most confident first); 0.5 = +12 points, half the dose
    delay_step_s: float = 0.005  # choose the click delay on this grid (the timing model is trained on 25 ms)
    stage_gate: StageGate | None = None
    timing: TimingModel | None = None
    conf_cut: float = -np.inf

    def fit(self, stage_F: np.ndarray, is_n2n3: np.ndarray, X: np.ndarray, Y: np.ndarray, seed: int = 0) -> "ClosedLoop":
        """stage_F / is_n2n3: per-second stage features and labels (stage.py).
        X / Y: timing features and score-vs-FLANK_DELAYS targets of rising-flank triggers
        in real N2/N3 (hypnogram-gated), from the training nights."""
        self.stage_gate = StageGate(seed=seed).fit(stage_F, is_n2n3)
        self.timing = fit_timing(X, Y, self.latency_s, self.delay_step_s)
        conf = self.timing.predict(X, self.latency_s, self.delay_step_s)[1]
        self.conf_cut = np.quantile(conf, 1 - self.keep) if self.keep < 1 else -np.inf
        return self

    @property
    def latency_s(self) -> float:
        return self.eeg_latency_s + self.audio_latency_s

    def plan(self, eeg: np.ndarray, fs: float) -> Clicks:
        prob = self.stage_gate.prob(stage_features(eeg, fs))
        gate = gate_from_prob(prob, len(eeg), fs, self.stage_threshold)
        cfg = DetectorConfig(**{**self.detector.__dict__, "gate": "external"})
        trg = detect(eeg, fs, cfg, gate_on=gate)
        delay, conf = self.timing.predict(features(eeg, fs, trg, window_s=self.timing.window_s), self.latency_s, self.delay_step_s)
        keep = conf >= self.conf_cut
        trg = trg.subset(keep)
        heard = trg.available_idx / fs + delay[keep]
        return Clicks(heard, heard - self.audio_latency_s, conf[keep], trg)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "ClosedLoop":
        return pickle.loads(Path(path).read_bytes())


def main():
    from .data import ALL_RECORDINGS
    from .phase07 import FIXED, prepare, prepare_window

    ap = argparse.ArgumentParser(description="Fit the default pipeline on all Sleep-EDF nights.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--out", default="models/closed_loop.pkl")
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        nights = list(ex.map(prepare, ALL_RECORDINGS))  # cached by cloop-phase07
        X = list(ex.map(prepare_window, [(r, TIMING_WINDOW_S) for r in ALL_RECORDINGS]))
    oracle = list(FIXED).index("hypnogram gate (oracle)")
    loop = ClosedLoop(keep=args.keep).fit(
        np.concatenate([n["F_train"] for n in nights]), np.concatenate([n["y_train"] for n in nights]),
        np.concatenate([x[oracle] for x in X]), np.concatenate([n["fixed"][oracle]["Y"] for n in nights]))
    loop.save(args.out)
    print(f"fitted on {len(nights)} nights; saved {args.out}")


if __name__ == "__main__":
    main()
