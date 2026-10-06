"""Streaming version of the default pipeline: feed packets as they arrive, get clicks.

    live = StreamingLoop(ClosedLoop.load("models/closed_loop.pkl"), fs=100)
    for packet in device:                  # microvolts, any packet size
        for click in live.push(packet):
            schedule(click.issue_s)        # acquisition-time seconds; map to wall clock

Gives exactly the clicks of ClosedLoop.plan() on the same signal, whatever the packet
size (tested), because it shares the per-sample logic (TroughTracker, RunningMedian),
carries every filter's state, and works through each packet in `detector.chunk`-sample
steps. Delays count from when a trigger's data was *acquired*; packets that arrive late
(BLE buffering, a stall) eat into the latency budget, and the runner logs the slack.

The models were trained at 100 Hz; devices at other rates (Muse 256 Hz, OpenBCI 250 Hz)
go through `Resampler` first, and StreamingLoop compensates its delay.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, group_delay, lfilter, lfilter_zi, sosfilt

from .detector import DetectorConfig, TroughTracker
from .pipeline import ClosedLoop
from .predict import DECIMATE
from .stage import BANDS, TAUS_S, RunningMedian


@dataclass
class Click:
    trigger_idx: int  # sample that fired the trigger
    available_idx: int  # last sample of the packet it arrived in
    delay_s: float  # chosen delay after available_idx
    heard_s: float  # when the sound should be heard (acquisition time, seconds)
    issue_s: float  # when to send the play command
    confidence: float


class StreamingLoop:
    """signal_delay_s: how far the input lags the brain (e.g. Resampler.delay_s); clicks are
    moved that much earlier. extra_wait_s: worst-case extra time before a sample is
    available (e.g. Resampler.wait_s). Both come out of the latency budget, so the chosen
    delay always leaves the same slack for EEG + audio latency."""

    def __init__(self, loop: ClosedLoop, fs: float = 100.0, signal_delay_s: float = 0.0, extra_wait_s: float = 0.0):
        if fs != 100.0:
            raise ValueError("models are trained at 100 Hz; wrap the device stream in Resampler")
        self.loop, self.fs, self.step = loop, fs, int(fs)
        self.signal_delay_s, self.extra_wait_s = signal_delay_s, extra_wait_s
        self.cfg = DetectorConfig(**{**loop.detector.__dict__, "gate": "external"})
        bp = lambda lo, hi: butter(2, [lo, hi], btype="band", fs=fs, output="sos")  # noqa: E731
        self.sw_sos, self.delta_sos = bp(*self.cfg.sw_band), bp(0.5, 4.0)
        self.sw_zi, self.delta_zi = np.zeros((2, 2)), np.zeros((2, 2))
        self.band_sos = [bp(lo, hi) for lo, hi in BANDS]
        self.band_zi = [np.zeros((2, 2)) for _ in BANDS]
        self.ema_a = [1.0 / (tau * fs) for tau in TAUS_S]
        self.ema_zi = [[np.zeros(1) for _ in TAUS_S] for _ in BANDS]
        window_s = loop.timing.window_s
        self.w = int(window_s * fs)
        if self.cfg.trigger_max_wait_s >= window_s:
            raise ValueError("the trough must still be in the feature window when the trigger fires")
        self.buf_sw, self.buf_delta = np.zeros(self.w), np.zeros(self.w)
        self.sus_w = int(self.cfg.sustain_s * fs)
        self.sus_buf = np.zeros(max(self.sus_w, 1), bool)
        self.sus_count = 0
        self.gate_now = False  # stage gate for the current second (decided at the end of the previous one)
        self.median = RunningMedian(len(BANDS) * len(TAUS_S))
        self.tracker = TroughTracker(self.cfg, fs)
        self.n = 0  # index of the next sample
        self.waiting = []  # triggers whose chunk isn't complete yet
        self.prob_log: list[float] = []  # P(N2/N3) per second, for logging

    def push(self, x) -> list[Click]:
        x = np.asarray(x, dtype=float)
        # Work in pieces that end on chunk boundaries; a trigger is released when its
        # chunk is complete (what the simulation assumed), with buffers as of that moment.
        c = self.cfg.chunk
        clicks, i = [], 0
        while i < len(x):
            j = min(i + c - self.n % c, len(x))
            self._push(x[i:j])
            if self.n % c == 0:
                clicks += [k for e in self.waiting if (k := self._click(*e))]
                self.waiting = []
            i = j
        return clicks

    def _push(self, x: np.ndarray) -> None:
        m, n0 = len(x), self.n
        if m == 0:
            return []
        sw, self.sw_zi = sosfilt(self.sw_sos, x, zi=self.sw_zi)
        delta, self.delta_zi = sosfilt(self.delta_sos, x, zi=self.delta_zi)
        emas = []
        for b, sos in enumerate(self.band_sos):
            p, self.band_zi[b] = sosfilt(sos, x, zi=self.band_zi[b])
            p = p**2
            for t, a in enumerate(self.ema_a):
                e, self.ema_zi[b][t] = lfilter([a], [1, a - 1], p, zi=self.ema_zi[b][t])
                emas.append(e)

        sw_list = sw.tolist()
        for i in range(m):
            n = n0 + i
            g = self.gate_now
            if self.sus_w > 0:
                k = n % self.sus_w
                self.sus_count += int(g) - int(self.sus_buf[k])
                self.sus_buf[k] = g
                on = g and self.sus_count / self.sus_w >= self.cfg.sustain_frac
            else:
                on = g
            if e := self.tracker.step(n, sw_list[i], on):
                self.waiting.append(e)
            if n % self.step == self.step - 1:  # end of a second: decide the gate for the next one
                raw = np.log10(np.array([e[i] for e in emas]) + 1e-6)
                row = np.concatenate([raw, raw - self.median.update(raw)])
                prob = float(self.loop.stage_gate.prob(row[None, :])[0])
                self.prob_log.append(prob)
                self.gate_now = prob >= self.loop.stage_threshold
        self.n += m
        self.buf_sw = np.concatenate([self.buf_sw, sw])[-self.w :]
        self.buf_delta = np.concatenate([self.buf_delta, delta])[-self.w :]

    def _click(self, trough, zc, thr, confirm) -> Click | None:
        avail = self.n - 1  # end of this chunk (== offline's available_idx when chunks are aligned)
        first = avail + 1 - self.w  # sample index of buf[0]
        scale = max(abs(self.buf_sw[trough - first]), 1e-6)
        row = np.concatenate([self.buf_sw[::DECIMATE] / scale, self.buf_delta[::DECIMATE] / scale,
                              [np.log(scale), (trough - zc) / self.fs, (avail - trough) / self.fs]])
        delay, conf = self.loop.timing.predict(row[None, :], self.loop.latency_s + self.signal_delay_s + self.extra_wait_s,
                                               self.loop.delay_step_s)
        if conf[0] < self.loop.conf_cut:
            return None
        heard = avail / self.fs + delay[0] - self.signal_delay_s
        return Click(confirm, avail, float(delay[0]), heard, heard - self.loop.audio_latency_s, float(conf[0]))


class Resampler:
    """Causal resampling of a device stream to the models' 100 Hz.

    Anti-alias with a 4th-order Butterworth low-pass at 40 Hz, then linear interpolation at
    the output times. The filter delays slow waves by a constant `delay_s` (~9.5 ms, flat
    across 0.5-4 Hz), which StreamingLoop subtracts from click times; an output sample also
    needs the next input sample, so it is available up to `wait_s` (one input sample) late.
    Pass both to StreamingLoop(signal_delay_s=..., extra_wait_s=...).
    """

    def __init__(self, fs_in: float, fs_out: float = 100.0, cutoff: float = 40.0):
        self.fs_in, self.fs_out = fs_in, fs_out
        self.b, self.a = butter(4, cutoff, fs=fs_in)
        self.delay_s = float(group_delay((self.b, self.a), w=[1.0], fs=fs_in)[1][0] / fs_in)
        self.wait_s = 1.0 / fs_in
        self.zi = None
        self.t_next = 0.0  # time (s) of the next output sample
        self.last = None  # (time, value) of the last filtered input sample
        self.k = 0  # input samples seen

    def push(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.empty(0)
        if self.zi is None:  # start the filter at the first value rather than ramping from 0
            self.zi = lfilter_zi(self.b, self.a) * x[0]
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        t = (self.k + np.arange(len(x))) / self.fs_in
        if self.last is not None:
            t, y = np.concatenate([[self.last[0]], t]), np.concatenate([[self.last[1]], y])
        self.k += len(x)
        self.last = (t[-1], y[-1])
        out_t = np.arange(self.t_next, t[-1] + 1e-12, 1.0 / self.fs_out)
        out_t = out_t[out_t >= t[0]]
        self.t_next = out_t[-1] + 1.0 / self.fs_out if len(out_t) else self.t_next
        return np.interp(out_t, t, y)
