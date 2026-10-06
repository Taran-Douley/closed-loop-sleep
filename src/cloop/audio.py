"""Click output with low, measurable latency (Phase 2's player; Phase 1 measures it).

`ClickPlayer` keeps a PortAudio stream running and starts the click at the beginning of
the next audio block after `play()`, so command-to-sound latency is the device's output
latency plus at most one block. With `record=True` the stream is full duplex and keeps
each input block with its ADC time on the same clock that `play()` returns, which is
what the latency tools use (cloop.latency).

Needs the `audio` extra (sounddevice; on Windows its wheel includes PortAudio). Prefer
the WASAPI host API on Windows: its stream times are accurate, MME's are not.
"""

import threading
import time

import numpy as np


def pink_click(fs: float, dur_s: float = 0.05, ramp_s: float = 0.001, seed: int = 0) -> np.ndarray:
    """A 50 ms pink-noise burst (the usual CLAS stimulus), peak-normalised to 1."""
    n = int(dur_s * fs)
    spec = np.fft.rfft(np.random.default_rng(seed).normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    r = max(int(ramp_s * fs), 1)
    x[:r] *= np.linspace(0, 1, r)
    x[-r:] *= np.linspace(1, 0, r)
    return x / np.abs(x).max()


class ClickPlayer:
    def __init__(self, fs: int = 48000, block: int = 64, volume: float = 0.1, click: np.ndarray | None = None,
                 record: bool = False, device=None):
        """volume: peak amplitude as a fraction of digital full scale. It is NOT a sound
        level; calibrate the level at the ear before anyone sleeps with it."""
        import sounddevice as sd

        if not 0 <= volume <= 1:
            raise ValueError("volume is a fraction of full scale, 0..1")
        self.fs, self.record = fs, record
        self.click = (pink_click(fs) if click is None else np.asarray(click, float)) * volume
        self._pending, self._pos = False, None
        self._lock = threading.Lock()
        self.blocks: list[tuple[float, np.ndarray, float]] = []  # (ADC time of first sample, samples, wall time at callback)
        cls = sd.Stream if record else sd.OutputStream
        self.stream = cls(samplerate=fs, blocksize=block, channels=1, dtype="float32", latency="low",
                          device=device, callback=self._callback)
        self.stream.start()

    def _callback(self, *args):
        if self.record:
            indata, outdata, frames, t, _ = args
            self.blocks.append((t.inputBufferAdcTime, indata[:, 0].copy(), time.time()))
        else:
            outdata, frames, t, _ = args
        outdata.fill(0)
        with self._lock:
            if self._pending:
                self._pending, self._pos = False, 0
            if self._pos is not None:
                chunk = self.click[self._pos : self._pos + frames]
                outdata[: len(chunk), 0] = chunk
                self._pos = self._pos + frames if self._pos + frames < len(self.click) else None

    def play(self) -> float:
        """Start the click as soon as possible; returns the command time on the stream clock."""
        with self._lock:
            self._pending = True
            return self.stream.time

    @property
    def host_api(self) -> str:
        import sounddevice as sd

        dev = sd.query_devices(self.stream.device[1] if self.record else self.stream.device)
        return sd.query_hostapis(dev["hostapi"])["name"]

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()
