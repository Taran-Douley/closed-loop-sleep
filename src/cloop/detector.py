"""Causal, chunked slow-wave trough detector -- the part that would run live.

Everything here only uses samples that have already arrived, and data is fed in
chunks to mimic packetised streaming (a Muse sends ~12 samples per BLE packet).

The detector emits *triggers* (a confirmed slow-wave trough). Turning a trigger
into a sound time (delay strategy, latency compensation) lives in timing.py so
that many timing variants can be evaluated from one detection pass.
"""

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter, sosfilt


@dataclass
class DetectorConfig:
    sw_band: tuple[float, float] = (0.5, 2.0)  # slow-wave band for trough detection
    trough_uv: float = -40.0  # fixed mode: trough must go below this (montage dependent!)

    # Adaptive threshold: target this person's deepest waves instead of a fixed uV level.
    # The threshold is the `adapt_pct` percentile of the minima of gated negative
    # half-waves seen so far, so 5 = "only the deepest 5% of waves". Amplitude varies a
    # lot between people, montages and headbands; this makes the setting portable.
    # The memory must be long (a whole night): a ~10 min window tracks sleep *stage*
    # rather than the *person*, relaxes in N2 and stimulates unpredictable small waves.
    threshold: str = "adaptive"  # "fixed" | "adaptive"
    adapt_pct: float = 5.0
    adapt_window: int = 20000  # ~ a whole night of gated half-waves
    adapt_min_history: int = 50  # use trough_uv until we've seen this many
    adapt_floor_uv: float = -25.0  # never accept troughs shallower than this
    lockout_s: float = 3.6  # after a trigger: 2 clicks (~1.1 s apart) + 2.5 s pause
    chunk: int = 4  # samples per packet delivered to the detector
    # Trigger point. None = at the trough (confirmed one sample later). A number r = later,
    # on the rising flank, at the first sample >= r * |trough depth| (0 = upward zero
    # crossing, -0.5 = halfway back up). Less to extrapolate, less latency slack.
    # Portiloop's rule fires around r = +0.25 for typical waves. Default since Phase 0.7
    # (+20 points over the trough in Phase 0.5); Phase 0-0.5 scripts pin None.
    trigger_rise: float | None = 0.5
    trigger_max_wait_s: float = 1.0  # give up if the rise doesn't happen in time

    # Causal sleep gate: only stimulate while delta dominates and nothing looks like arousal.
    # "external": a per-sample gate passed to detect(), e.g. from the learned stage
    # classifier (stage.py); the sustained-sleep rule below is still applied to it.
    gate: str = "causal"  # "causal" | "external" | "oracle" (use hypnogram N2/N3) | "none"
    gate_tau_s: float = 10.0  # EMA time constant for band powers
    delta_ratio_min: float = 0.80  # power(0.5-4 Hz) / power(0.5-30 Hz)
    hf_rms_max_uv: float = 2.5  # RMS in 16-30 Hz band (arousal / muscle veto)
    # Eye-movement veto. Frontal sensors (Fpz, Muse AF7/AF8) pick up slow eye movements
    # in drowsy wake and around waking up; they look like big slow waves to the delta
    # test above but carry relatively more power below 0.5 Hz.
    eye_ratio_max: float | None = 0.5  # power(0.16-0.5 Hz) / power(0.5-4 Hz)
    # Sustained sleep: the instantaneous gate flickers. Require it to have been on for
    # `sustain_frac` of the last `sustain_s` seconds. Kills brief wake false alarms
    # without dropping deep sleep every time the gate blinks off for a few seconds.
    sustain_s: float = 120.0
    sustain_frac: float = 0.8
    # Person-relative gate (Phase 0.6). The fixed muscle (uV) and eye-movement (ratio)
    # limits above were tuned on 25-34 year olds and lock many older sleepers out
    # entirely: their sleep has more < 0.5 Hz power and a higher high-frequency floor.
    # With gate_relative, each limit becomes max(fixed limit, k x this person's running
    # median), the median taken over seconds that pass the delta test so far tonight.
    # Never stricter than the fixed gate, so it can only open it for people whose
    # baseline is high.
    gate_relative: bool = False
    hf_rel: float = 2.0
    eye_rel: float = 2.0
    rel_min_history_s: float = 600.0  # use the fixed limits until we have this much


@dataclass
class Triggers:
    trough_idx: np.ndarray  # sample index of detected trough (in causal-filtered signal)
    zc_idx: np.ndarray  # last downward zero crossing before the trough
    available_idx: np.ndarray  # index of the last sample in the packet that confirmed it
    gate_on: np.ndarray = field(repr=False)  # per-sample gate state (for diagnostics)
    threshold_uv: np.ndarray = field(repr=False)  # threshold in force at each trigger
    pred_delay: np.ndarray | None = field(default=None, repr=False)  # set by a timing model (predict.py)
    trigger_idx: np.ndarray | None = field(default=None, repr=False)  # rising-flank trigger (trigger_rise), else None

    def subset(self, keep: np.ndarray) -> "Triggers":
        pred = None if self.pred_delay is None else self.pred_delay[keep]
        trig = None if self.trigger_idx is None else self.trigger_idx[keep]
        return Triggers(self.trough_idx[keep], self.zc_idx[keep], self.available_idx[keep], None, self.threshold_uv[keep], pred, trig)


def _bp(lo, hi, fs, order=2):
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def _relative_limit(v: np.ndarray, ok: np.ndarray, fixed: float, k: float, fs: float, cfg: DetectorConfig) -> np.ndarray:
    """Per-sample limit max(fixed, k x running median of v over past seconds where `ok`).
    Causal: the limit in force during second s uses only seconds before s."""
    step = int(fs)
    secs = range(step - 1, len(v), step)  # last sample of each second
    ranked: list[float] = []
    lim = np.full(len(v), fixed, dtype=float)
    for i, n in enumerate(secs):
        if len(ranked) >= cfg.rel_min_history_s:
            lim[n + 1 : n + 1 + step] = max(fixed, k * ranked[len(ranked) // 2])
        if ok[n]:
            insort(ranked, float(v[n]))
    return lim


class TroughTracker:
    """Per-sample trough / rising-flank trigger logic, shared by the offline detect()
    and the streaming pipeline (stream.py) so the two can't drift apart.

    step(n, x, on) takes sample n of the causal slow-wave signal and the gate state, and
    returns (trough_idx, zc_idx, threshold, confirm_idx) when a trigger fires: confirm_idx
    is the sample that confirmed it (trough + 1, or the rising-flank sample)."""

    def __init__(self, cfg: DetectorConfig, fs: float):
        self.cfg = cfg
        self.lockout = int(cfg.lockout_s * fs)
        self.max_wait = int(cfg.trigger_max_wait_s * fs)
        self.pending = None  # (trough, zc, threshold, depth) waiting for the rising-flank trigger
        self.prev = 0.0
        self.zc = -1
        self.armed = False  # signal has crossed below threshold, waiting for the minimum
        self.spent = False  # already handled this negative half-wave
        self.last_trigger = -self.lockout
        self.thr = cfg.trough_uv
        self.history: deque[float] = deque(maxlen=cfg.adapt_window)  # arrival order, for eviction
        self.ranked: list[float] = []  # same values kept sorted, for O(log n) percentiles
        self.cur_min = 0.0  # minimum of the current negative half-wave

    def step(self, n: int, x: float, on: bool):
        cfg, prev, out = self.cfg, self.prev, None
        if prev >= 0 > x:
            self.zc = n
            self.spent = False
            self.cur_min = 0.0
        elif prev < 0 <= x and cfg.threshold == "adaptive":
            # half-wave finished: learn its depth (only from sleep we'd stimulate in)
            if on:
                if len(self.history) == self.history.maxlen:
                    del self.ranked[bisect_left(self.ranked, self.history[0])]
                self.history.append(self.cur_min)
                insort(self.ranked, self.cur_min)
                if len(self.ranked) >= cfg.adapt_min_history:
                    self.thr = min(self.ranked[int(cfg.adapt_pct / 100 * (len(self.ranked) - 1))], cfg.adapt_floor_uv)
        if x < 0:
            self.cur_min = min(self.cur_min, x)
        if self.pending is not None:
            t0, z0, th0, depth = self.pending
            if x >= cfg.trigger_rise * depth:
                out = (t0, z0, th0, n)
                self.last_trigger = t0
                self.pending = None
            elif n - t0 > self.max_wait or not on:
                self.pending = None
        if self.armed and x > prev:
            # first rising sample after going below threshold -> trough was n-1
            self.armed = False
            self.spent = True
            if on and n - 1 - self.last_trigger >= self.lockout and self.zc >= 0:
                if cfg.trigger_rise is None:
                    out = (n - 1, self.zc, self.thr, n)
                    self.last_trigger = n - 1
                else:
                    self.pending = (n - 1, self.zc, self.thr, -prev)
        elif x < self.thr and not self.spent:
            self.armed = True
        self.prev = x
        return out


def detect(eeg: np.ndarray, fs: float, cfg: DetectorConfig, stages: np.ndarray | None = None,
           gate_on: np.ndarray | None = None) -> Triggers:
    # A causal IIR filter gives identical output whether run packet-by-packet (with
    # carried state, as live code would) or over the whole array, so filter once.
    # Packetisation only matters for *when* a trigger becomes available (below).
    sw, delta, broad, hf = (sosfilt(_bp(lo, hi, fs), eeg) for lo, hi in (cfg.sw_band, (0.5, 4.0), (0.5, 30.0), (16.0, 30.0)))

    if cfg.gate == "causal":
        alpha = 1.0 / (cfg.gate_tau_s * fs)
        ema = lambda v: lfilter([alpha], [1, alpha - 1], v)  # noqa: E731 -- causal exponential moving average
        p_delta, p_broad, p_hf = ema(delta**2), ema(broad**2), ema(hf**2)
        delta_ok = p_delta / (p_broad + 1e-9) > cfg.delta_ratio_min
        hf_rms = np.sqrt(p_hf)
        hf_max = _relative_limit(hf_rms, delta_ok, cfg.hf_rms_max_uv, cfg.hf_rel, fs, cfg) if cfg.gate_relative else cfg.hf_rms_max_uv
        gate_on = delta_ok & (hf_rms < hf_max)
        if cfg.eye_ratio_max is not None:
            p_eye = ema(sosfilt(_bp(0.16, 0.5, fs), eeg) ** 2)
            eye = p_eye / (p_delta + 1e-9)
            eye_max = _relative_limit(eye, delta_ok, cfg.eye_ratio_max, cfg.eye_rel, fs, cfg) if cfg.gate_relative else cfg.eye_ratio_max
            gate_on &= eye < eye_max
        gate_on[: int(cfg.gate_tau_s * fs)] = False  # EMA warm-up
    elif cfg.gate == "external":
        gate_on = np.asarray(gate_on, bool).copy()
    elif cfg.gate == "oracle":
        gate_on = np.isin(stages, (2, 3))
    else:
        gate_on = np.ones(len(eeg), dtype=bool)
    if cfg.gate in ("causal", "external") and cfg.sustain_s > 0:
        w = int(cfg.sustain_s * fs)
        cs = np.concatenate([[0], np.cumsum(gate_on)])
        n = np.arange(1, len(gate_on) + 1)
        frac_on = (cs[n] - cs[np.maximum(n - w, 0)]) / w  # causal: last w samples only
        gate_on &= frac_on >= cfg.sustain_frac

    tracker = TroughTracker(cfg, fs)
    events = [e for n, (x, on) in enumerate(zip(sw.tolist(), gate_on.tolist())) if (e := tracker.step(n, x, on))]
    troughs, zcs, thresholds, confirmed = (list(c) for c in zip(*events)) if events else ([], [], [], [])

    troughs, confirmed = np.array(troughs, int), np.array(confirmed, int)
    avail = (confirmed // cfg.chunk) * cfg.chunk + cfg.chunk - 1  # end of that packet
    fired = confirmed if cfg.trigger_rise is not None else None
    return Triggers(troughs, np.array(zcs, int), avail, gate_on, np.array(thresholds), trigger_idx=fired)


def portiloop_so(eeg: np.ndarray, fs: float, gate_on: np.ndarray | None = None, scale: float = 1.0,
                 refractory_s: float = 3.6) -> Triggers:
    """Port of Portiloop's rule-based SlowOscillationDetector (portiloop-software,
    custom_detectors.py) as a baseline. Fixed Massimini-style criteria on a 0.16-4 Hz
    17-tap FIR (after their 0.5-30 Hz FIR + DC removal): since the last downward zero
    crossing, negative peak > 40 uV, peak-to-peak > 75 uV, negative time 125-1500 ms,
    positive time < 1000 ms. Fires at the first sample meeting all of them, i.e. on the
    up-swing after the trough. `scale` multiplies the uV thresholds (montage differences).

    Returned triggers use the *detection* sample as `trough_idx`, because Portiloop's
    delay counts from detection. A refractory period (our 2-click lockout by default)
    keeps the dose comparable; Portiloop leaves this to the operator.
    """
    from scipy.signal import firwin

    x = lfilter(firwin(20, [0.5, 30.0], pass_zero=False, window="hamming", fs=fs), [1.0], eeg)
    x = x - lfilter([0.005], [1, -0.995], x)  # their DC tracker
    x = lfilter(firwin(17, [0.16, 4.0], fs=fs, pass_zero="bandpass"), [1.0], x)
    th_pap, th_neg = 75 * scale, 40 * scale
    hits = []
    mx, mn, down, up, prev = -1.0, 1000.0, 0, 0, None
    last = -int(refractory_s * fs)
    on = np.ones(len(x), bool) if gate_on is None else gate_on
    for n, (v, g) in enumerate(zip(x.tolist(), on.tolist())):
        mx, mn = max(mx, v), min(mn, v)
        if v >= 0:
            up += 1
        else:
            down += 1
        prev = prev or v
        if prev * v <= 0 < prev:  # downward zero crossing starts a new segment
            mx, mn, down, up = -1.0, 1000.0, 0, 0
        prev = v
        tne, tpo = 1000 * down / fs, 1000 * up / fs
        if (mx - mn > th_pap and -mn > th_neg and 125 < tne < 1500 and tpo < 1000
                and g and n - last >= refractory_s * fs):
            hits.append(n)
            last = n
    hits = np.array(hits, int)
    return Triggers(hits, hits, hits, None, np.full(len(hits), -th_neg))
