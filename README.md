# closed-loop-sleep

Open-source closed-loop acoustic stimulation (CLAS) for slow-wave sleep: short pink-noise
clicks timed to land on the up-state of slow oscillations (Ngo et al., 2013).

> **Not a medical device.** This is a research/hobby project. It makes no claim to improve
> sleep, memory or health. Don't use it in place of medical care.

## Phase 0 — offline phase-targeting (this repo, today)

Before buying hardware, answer: *using only data a live system would have, how often would
clicks actually land on the up-state?*

Pipeline (all causal, i.e. no peeking at the future):

1. **Slow-wave filter** — 0.5–2 Hz Butterworth band-pass, causal.
2. **Sleep gate** — stimulate only when
   - delta (0.5–4 Hz) dominates and 16–30 Hz power is low (arousal/muscle veto),
   - power below 0.5 Hz is not large relative to delta (**eye-movement veto** — frontal
     sensors see slow eye movements in drowsy wake, which otherwise look like slow waves),
   - and the above has held for ≥ 80% of the last 2 minutes (**sustained sleep**).
3. **Trough detection** — the signal dips below a threshold and we trigger at the first
   rising sample. The threshold **adapts to the person**: the 5th percentile of all
   negative half-wave depths seen so far tonight ("only the deepest 5% of waves").
4. **Timing** — when to click after the trough:
   - *fixed delay* (the classic lab approach)
   - *adaptive*: aim half of this wave's estimated period later (± filter-phase correction)
   - *timing model* (`predict.py`): ridge regression on the last 2 s of causal signal predicts
     how well a click would land at each candidate delay (150–900 ms); click at the best one.
     Its confidence can also be used to **skip waves it can't predict**.
5. **Two clicks then a 2.5 s pause** (Ngo protocol), per trigger.
6. **Latency model** — EEG Bluetooth delay, audio delay, their jitter, optional compensation.

Ground truth: zero-phase filter + Hilbert phase computed offline on the full recording.

Evaluation: **20 random half/half subject splits** (as in the Portiloop paper). Every
parameter and model is fitted on the tune half and scored on the held-out half; we report
mean ± SD across splits.

Data: [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/), Fpz-Cz channel
(frontal, similar to consumer headband placement), subjects 0–19 (one night each). Files are
fetched from PhysioNet's AWS open-data mirror (much faster than physionet.org).

### Run

```bash
uv sync
uv run pytest                                    # synthetic sanity + causality checks
uv run cloop-phase0 --subjects 0-19 --splits 20  # ~1 GB download on first run, ~15 min
```

Outputs land in `results/`: `results.csv`, `thresholds.csv`, `jitter.csv`, `tuning.png`,
`phase_polar.png`, `jitter.png`.

### Metrics

- **% within ±45°** of the up-state peak (chance = 25%)
- **R** — phase-locking (0 random, 1 perfectly consistent) and **mean error** (degrees; − = early)
- **clicks/hour of N2+N3** — stimulation dose, and its spread across people
- **% in N2/N3**, **% in wake/REM** — is the gate stimulating at the right time?
- **% late** — the click was scheduled for a moment that had already passed when the data arrived

### Results (20 subjects, 20 splits, held-out subjects, mean ± SD across splits)

**Timing** (all waves the detector picks, ~44 clicks/hour of N2+N3, BLE EEG + wired audio):

| method | % within ±45° (chance 25%) |
|---|---|
| fixed delay (415 ms) | 41.5 ± 1.8 |
| adaptive (period estimate) | 37.9 ± 1.2 |
| adaptive + filter correction | 35.9 ± 1.3 |
| **timing model (ridge)** | **43.3 ± 1.9** — better than fixed in 20/20 splits (+1.8 ± 0.9) |
| timing model (small MLP) | 41.6 ± 1.9 — no better than fixed |

**What to stimulate** (no latency; dose = clicks per hour of N2+N3):

| selection | % within ±45° | worst subject | dose | dose range across people |
|---|---|---|---|---|
| fixed −40 µV | 39.1 ± 1.2 | 31 | 94 | 22–245 |
| fixed −80 µV | 49.4 ± 5.4 | 26 | 10 | 0.1–54 (some get none) |
| adaptive p5 (default) | 41.6 ± 1.8 | 32 | 44 | 15–94 |
| adaptive p2 | 44.3 ± 2.3 | 32 | 22 | 7–49 |
| **model top 50% confidence** | **51.8 ± 2.0** | **39** | 22 | 7–55 |
| **model top 25% confidence** | **55.9 ± 2.2** | **41** | 11 | 3–32 |

At the same dose, skipping the waves the model is unsure about beats skipping shallow
waves: +7.5 ± 1.9 points over adaptive p2, better in 20/20 splits.

**Sleep gate**: 99.3% of clicks in N2/N3, **0.6% in wake/REM** (was 15% before the
eye-movement veto and sustained-sleep rule). It even beats the expert-hypnogram "oracle"
gate on accuracy (41.6% vs 37.5%), because it stimulates in cleaner, more stable stretches.

**Latency**: uncompensated Bluetooth audio (~200 ms) drops targeting to chance (24%).
Compensating the mean recovers it; the remaining cost is jitter: 41.6% at 0 ms SD →
36.2% at 100 ms → 32.4% at 150 ms. Use wired audio (Portiloop measured 4 ms).

**Second click** (fixed 1.075 s after the first): ~26%, i.e. untargeted — consistent with
the Ngo protocol, where only the first click is phase-aimed.

### What we learned

1. **Trough detection is easy; prediction is hard.** Causal troughs land within ~3° of the
   true trough, but phase-locking decays from 0.8 at the trough to ~0.15 half a second later.
   The hindsight ceiling (best delay per wave) is ~95%, so there is a lot of headroom.
2. **Amplitude thresholds must adapt to the person.** A fixed µV level gives one person 25×
   the dose of another, and a short (10 min) memory tracks sleep stage instead of the person.
3. **Clean sine-wave logic (period estimate, filter-phase correction) loses to a fixed delay
   on real EEG.** A learned model wins, and learns the filter correction by itself.
4. **Knowing when not to click is the biggest win** — model confidence > amplitude.
5. **The small MLP doesn't beat ridge** with ~2k training waves. More data (MASS, more
   Sleep-EDF nights) is the obvious next step before trying bigger models.

### Caveats

- 20 subjects, one night each, 100 Hz, one frontal *bipolar* channel (Muse is referenced
  differently — expect to re-tune, and the adaptive threshold exists for exactly this).
- Gate settings (eye-movement ratio 0.5, 120 s sustain) were chosen on subjects 0–9 and
  checked on 10–19, but those subjects also appear in the splits.
- Ground-truth phase comes from a 0.5–2 Hz band, which includes faster delta than "true"
  slow oscillations (0.5–1.25 Hz).
- Latency numbers are assumptions until measured in Phase 1.
- Confidence cut-offs are set from in-sample predictions on the tune half, so realised dose
  can differ slightly from the nominal 50% / 25%.

## Phase 0.5 — all of Sleep-EDF, learned models, and where to trigger

Same pipeline, now on **all 153 Sleep Cassette nights (78 subjects, ages 25–101)**. Splits are
by subject, so both nights of a person are always on the same side.

```bash
uv sync --extra gru                                  # CPU-only PyTorch, for the GRU
uv run cloop-timing --splits 10                      # learning curves, ~2.5 h (features cached in data/cache/)
uv run cloop-baseline --subjects 0-82 --splits 20    # trigger point + Portiloop comparison, ~40 min
```

Outputs: `results/timing_learning_curve.csv`, `learning_curve.png`, `timing_run_log.txt`,
`baseline_portiloop.txt` (subjects 0–19), `baseline_portiloop_all_ages.txt`.

### 1. Where on the wave to trigger matters more than any model

Instead of triggering at the trough and predicting ~400 ms ahead, keep the same adaptive
threshold and gate but fire later, on the rising flank, once the signal has climbed back to
`r × |trough depth|` (`DetectorConfig(trigger_rise=r)`). Fixed delay tuned on held-out
splits, measured from when the trigger packet is available.

All 76 subjects with data (20 splits, held-out subjects, no latency unless stated):

| trigger | % within ±45° | at 50 ms latency | at 100 ms | clicks/h N2+N3 |
|---|---|---|---|---|
| trough (Phase 0) | 35.5 ± 1.6 | 36.3 | 36.3 | 27.5 |
| upward zero crossing (r = 0) | 40.9 ± 1.8 | 41.7 | 41.7 | 27.2 |
| r = +0.25 | 45.2 ± 1.9 | 46.0 | 46.0 | 26.8 |
| **r = +0.5** | **55.6 ± 1.9** | **56.6** | **55.3** | 22.5 |

Young subjects (0–19) only: trough 41.2 → **r = +0.5: 61.7 ± 1.5**. The trade-off is latency
slack: at r = +0.5 the best delay is ~50 ms, so up to ~75 ms of EEG + audio latency is free
and it degrades after that (48% at 150 ms). Part of the gain is selection: ~17% of waves never
climb that far within 1 s and are skipped.

### 2. Portiloop's slow-oscillation rule, ported (`detector.portiloop_so`)

[Portiloop](https://github.com/Portiloop/portiloop-software) ships fixed Massimini-style
criteria (negative peak > 40 µV, peak-to-peak > 75 µV, 0.16–4 Hz FIR), an operator-set delay
and no sleep gate. It fires on the up-swing, i.e. it *detects* the up-state rather than
predicting it — the best delay is 0 ms.

| | % within ±45° | at 50 ms | at 100 ms | clicks/h N2+N3 (per-night range) | % in wake/REM |
|---|---|---|---|---|---|
| Portiloop rule, no gate | 61.2 ± 0.9 | 53.7 | 44.0 | 370 (54–2107) | **63.4** |
| Portiloop rule + our causal gate | 64.0 ± 1.5 | 50.9 | 37.6 | 48 (0–261) | 6.4 |
| ours, r = +0.5 | 55.6 ± 1.9 | 56.6 | 55.3 | 22.5 (0–80) | 4.8 |

- Without a gate, most of its clicks land in wake/REM on this frontal channel (eye movements).
- With zero latency (Portiloop's on-device, wired setup, ~4 ms) it is the most accurate, but its
  fixed µV thresholds hand the dose to people with big waves. Halving the thresholds to spread
  the dose drops it to 41%.
- Above ~40 ms of latency (any Bluetooth EEG), triggering at r = +0.5 with a ~50 ms delay wins.

### 3. More data for the learned timing model (trough trigger)

Learning curve on held-out subjects, 10 splits, % within ±45° (all clicks):

| training subjects | 5 | 10 | 20 | 38 |
|---|---|---|---|---|
| fixed delay | 34.7 | 35.1 | 35.2 | 35.2 |
| **ridge** | 38.2 | 39.3 | 39.9 | **40.1** (+5.0 ± 0.8, 10/10 splits) |
| GRU | 35.4 | 38.3 | 39.4 | 39.9 |
| MLP | 36.0 | 37.1 | 38.6 | 39.2 |
| ridge, top 25% confidence | 49.1 | 52.9 | 53.5 | 54.1 |

- Gains flatten after ~10 subjects; with ~10k training waves the GRU catches up with ridge
  but doesn't beat it. Bigger models aren't the bottleneck.
- 4–8 s of context instead of 2 s: no change for all clicks, +2 points for confidence-selected
  clicks (ridge top 25%: 56.1).
- Hindsight ceiling (best delay per wave): 91%.

### 4. Age breaks the Phase 0 settings

Ages 25–34 are subjects 0–19; most of the rest are 50–101.

| age | nights | nights with no clicks | clicks/h N2+N3 | % in N2/N3 | fixed 415 ms, % within ±45° |
|---|---|---|---|---|---|
| ≤ 40 | 39 | 0 | 43.9 | 98.9 | 39.0 |
| 41–60 | 43 | 0 | 27.0 | 88.4 | 34.8 |
| 61–80 | 39 | 4 | 25.8 | 96.7 | 27.1 |
| 81+ | 32 | 8 | 12.2 | 65.8 | 25.6 |

- **The gate**: on the 12 empty nights (ages 67–96) it never opens. The eye-movement veto
  (relatively more < 0.5 Hz power in older sleep) and the fixed 2.5 µV muscle veto reject most
  N2/N3, and the 2-minute sustain rule turns partial passes into zero. Both were tuned on young
  subjects; they need to become relative/adaptive like the trough threshold.
- **The waves**: even with a perfect (hypnogram) gate, the best fixed delay moves from 425 ms
  (≤ 40) to 600 ms (81+) and accuracy at that delay falls by about two thirds. Slow waves in old
  age are slower and smaller, and phase is harder to call.

### What changed

1. **Trigger on the rising flank (r ≈ +0.5), not the trough.** +20 points, and it keeps
   ~75 ms of latency slack. This should be the default going into Phase 1.
2. Use ridge for timing/confidence; the GRU isn't worth the dependency yet.
3. Next modelling step: the timing model and confidence on top of the rising-flank trigger,
   and an age/person-adaptive gate.

## Layout

```
src/cloop/data.py       Sleep-EDF loading, per-sample hypnogram
src/cloop/detector.py   causal gate + trough detector (the bit that runs live), Portiloop SO rule port
src/cloop/timing.py     trough -> click time, latency model
src/cloop/predict.py    learned timing model (causal features, score-vs-delay curve)
src/cloop/evaluate.py   ground-truth phase + scoring
src/cloop/phase0.py     tune / held-out evaluation / plots
src/cloop/gru.py        small GRU timing model (optional `gru` extra)
src/cloop/phase05.py    learning curves on all Sleep-EDF nights (cloop-timing)
src/cloop/baseline.py   trigger point + Portiloop comparison (cloop-baseline)
tests/                  synthetic-signal sanity tests, causality test for the model
```

## Roadmap

- ~~**Phase 0.5** — more training data for the timing model; try a small GRU~~ (done, above).
- **Phase 0.6** — timing model on the rising-flank trigger; relative (person-adaptive) gate
  thresholds so older sleepers aren't locked out; re-check on MASS if access is granted.
- **Phase 1** — live streaming from Muse S / OpenBCI via BrainFlow, log only, no sound.
  *Measure* real EEG + audio latency (loopback the audio into an EEG channel or a mic).
- **Phase 2** — stimulation with randomised sham vs real nights, hard volume cap.
- **Phase 3** — YASA analysis, publish code + methods + results (including null results).

## References

- Ngo et al. (2013). Auditory closed-loop stimulation of the sleep slow oscillation enhances memory. *Neuron*.
- Valenchon et al. (2022). The Portiloop: a deep learning-based open science tool for closed-loop brain stimulation. *PLOS ONE*.
- Kemp et al. (2000). Sleep-EDF. PhysioNet.
