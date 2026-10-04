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
    # Portiloop's rule fires around r = +0.25 for typical waves.
    trigger_rise: float | None = None
    trigger_max_wait_s: float = 1.0  # give up if the rise doesn't happen in time

    # Causal sleep gate: only stimulate while delta dominates and nothing looks like arousal.
    gate: str = "causal"  # "causal" | "oracle" (use hypnogram N2/N3) | "none"
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


def detect(eeg: np.ndarray, fs: float, cfg: DetectorConfig, stages: np.ndarray | None = None) -> Triggers:
    # A causal IIR filter gives identical output whether run packet-by-packet (with
    # carried state, as live code would) or over the whole array, so filter once.
    # Packetisation only matters for *when* a trigger becomes available (below).
    sw, delta, broad, hf = (sosfilt(_bp(lo, hi, fs), eeg) for lo, hi in (cfg.sw_band, (0.5, 4.0), (0.5, 30.0), (16.0, 30.0)))

    if cfg.gate == "causal":
        alpha = 1.0 / (cfg.gate_tau_s * fs)
        ema = lambda v: lfilter([alpha], [1, alpha - 1], v)  # noqa: E731 -- causal exponential moving average
        p_delta, p_broad, p_hf = ema(delta**2), ema(broad**2), ema(hf**2)
        gate_on = (p_delta / (p_broad + 1e-9) > cfg.delta_ratio_min) & (np.sqrt(p_hf) < cfg.hf_rms_max_uv)
        if cfg.eye_ratio_max is not None:
            p_eye = ema(sosfilt(_bp(0.16, 0.5, fs), eeg) ** 2)
            gate_on &= p_eye / (p_delta + 1e-9) < cfg.eye_ratio_max
        gate_on[: int(cfg.gate_tau_s * fs)] = False  # EMA warm-up
        if cfg.sustain_s > 0:
            w = int(cfg.sustain_s * fs)
            cs = np.concatenate([[0], np.cumsum(gate_on)])
            n = np.arange(1, len(gate_on) + 1)
            frac_on = (cs[n] - cs[np.maximum(n - w, 0)]) / w  # causal: last w samples only
            gate_on &= frac_on >= cfg.sustain_frac
    elif cfg.gate == "oracle":
        gate_on = np.isin(stages, (2, 3))
    else:
        gate_on = np.ones(len(eeg), dtype=bool)

    lockout = int(cfg.lockout_s * fs)
    troughs, zcs, fired = [], [], []
    pending = None  # (trough, zc, threshold, depth) waiting for the rising-flank trigger
    max_wait = int(cfg.trigger_max_wait_s * fs)
    prev = 0.0
    zc = -1
    armed = False  # signal has crossed below threshold, waiting for the minimum
    spent = False  # already handled this negative half-wave
    last_trigger = -lockout
    thr = cfg.trough_uv
    history: deque[float] = deque(maxlen=cfg.adapt_window)  # arrival order, for eviction
    ranked: list[float] = []  # same values kept sorted, for O(log n) percentiles
    cur_min = 0.0  # minimum of the current negative half-wave
    thresholds = []

    for n, (x, on) in enumerate(zip(sw.tolist(), gate_on.tolist())):
        if prev >= 0 > x:
            zc = n
            spent = False
            cur_min = 0.0
        elif prev < 0 <= x and cfg.threshold == "adaptive":
            # half-wave finished: learn its depth (only from sleep we'd stimulate in)
            if on:
                if len(history) == history.maxlen:
                    del ranked[bisect_left(ranked, history[0])]
                history.append(cur_min)
                insort(ranked, cur_min)
                if len(ranked) >= cfg.adapt_min_history:
                    thr = min(ranked[int(cfg.adapt_pct / 100 * (len(ranked) - 1))], cfg.adapt_floor_uv)
        if x < 0:
            cur_min = min(cur_min, x)
        if pending is not None:
            t0, z0, th0, depth = pending
            if x >= cfg.trigger_rise * depth:
                troughs.append(t0)
                zcs.append(z0)
                thresholds.append(th0)
                fired.append(n)
                last_trigger = t0
                pending = None
            elif n - t0 > max_wait or not on:
                pending = None
        if armed and x > prev:
            # first rising sample after going below threshold -> trough was n-1
            armed = False
            spent = True
            if on and n - 1 - last_trigger >= lockout and zc >= 0:
                if cfg.trigger_rise is None:
                    troughs.append(n - 1)
                    zcs.append(zc)
                    thresholds.append(thr)
                    last_trigger = n - 1
                else:
                    pending = (n - 1, zc, thr, -prev)
        elif x < thr and not spent:
            armed = True
        prev = x

    troughs = np.array(troughs, int)
    fired = np.array(fired, int) if cfg.trigger_rise is not None else None
    confirmed = troughs + 1 if fired is None else fired  # trough: needed one rising sample to confirm the minimum
    avail = (confirmed // cfg.chunk) * cfg.chunk + cfg.chunk - 1  # end of that packet
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
