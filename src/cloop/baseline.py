"""Baseline: Portiloop's rule-based slow-oscillation detector vs ours, on the same nights.

Portiloop (Valenchon et al. 2022; github.com/Portiloop/portiloop-software) ships a
fixed-threshold SO detector with a constant, operator-set delay and no sleep gate.
We give it every advantage we can: its delay is tuned on the training subjects of
each split (as ours is), and we also run it behind our causal gate and the hypnogram.

    uv run cloop-baseline --subjects 0-19 --splits 20
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import ALL_RECORDINGS, load_subject
from .detector import DetectorConfig, detect, portiloop_so
from .evaluate import true_phase
from .phase0 import parse_range

DELAYS = np.arange(0.0, 0.901, 0.025)  # after the trigger (Portiloop: detection; ours: trough)
COS45 = np.cos(np.pi / 4)
RISES = (-0.5, 0.0, 0.25, 0.5)  # rising-flank trigger points for our detector (see DetectorConfig.trigger_rise)


def run_night(rec_id):
    sid, night = rec_id
    rec = load_subject(sid, night=night)
    phase = true_phase(rec)
    ours = detect(rec.eeg, rec.sfreq, DetectorConfig(), rec.stages)
    variants = {"ours (adaptive p5 + causal gate)": ours}
    for r in RISES:
        variants[f"ours, trigger at rise {r:+g} x depth"] = detect(rec.eeg, rec.sfreq, DetectorConfig(trigger_rise=r), rec.stages)
    n2n3 = np.isin(rec.stages, (2, 3))
    for scale in (1.0, 0.5):
        tag = f"Portiloop SO rule, thresholds x{scale:g}"
        variants[f"{tag}, no gate"] = portiloop_so(rec.eeg, rec.sfreq, scale=scale)
        variants[f"{tag}, our causal gate"] = portiloop_so(rec.eeg, rec.sfreq, ours.gate_on, scale=scale)
        variants[f"{tag}, hypnogram gate"] = portiloop_so(rec.eeg, rec.sfreq, n2n3, scale=scale)
    out = {"subject": sid, "night": night, "n2n3_h": n2n3.sum() / rec.sfreq / 3600, "v": {}}
    for name, t in variants.items():
        # delays count from when the trigger is *available* (packet end; detection sample for Portiloop)
        t0 = t.available_idx
        idx = t0[:, None] + np.round(DELAYS * rec.sfreq).astype(int)[None, :]
        out["v"][name] = {"cos": np.cos(phase[np.clip(idx, 0, len(phase) - 1)]), "stage": rec.stages[t0]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="0-19")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ids = set(parse_range(args.subjects))
    recs = [r for r in ALL_RECORDINGS if r[0] in ids]
    with ProcessPoolExecutor(args.workers) as ex:
        nights = list(ex.map(run_night, recs))
    subjects = np.array(sorted({n["subject"] for n in nights}))
    names = list(nights[0]["v"])
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.splits):
        perm = rng.permutation(subjects)
        tune_ids = set(perm[: len(perm) // 2].tolist())
        for name in names:
            tr = [n for n in nights if n["subject"] in tune_ids]
            te = [n for n in nights if n["subject"] not in tune_ids]
            C_tr = np.concatenate([n["v"][name]["cos"] for n in tr])
            best = int(C_tr.mean(0).argmax()) if len(C_tr) else 0
            C_te = np.concatenate([n["v"][name]["cos"] for n in te])
            st = np.concatenate([n["v"][name]["stage"] for n in te])
            rates = [len(n["v"][name]["stage"]) / n["n2n3_h"] for n in te]
            rows.append({"split": k, "detector": name, "delay_ms": 1000 * DELAYS[best],
                         "pct_within_45": 100 * (C_te[:, best] >= COS45).mean() if len(C_te) else np.nan,
                         "per_hour_N2N3": len(C_te) / sum(n["n2n3_h"] for n in te),
                         "min_night_per_hour": min(rates), "max_night_per_hour": max(rates),
                         "pct_N2N3": 100 * np.isin(st, (2, 3)).mean(), "pct_wake_REM": 100 * np.isin(st, (0, 4)).mean()})
    df = pd.DataFrame(rows)
    cols = ["pct_within_45", "delay_ms", "per_hour_N2N3", "min_night_per_hour", "max_night_per_hour", "pct_N2N3", "pct_wake_REM"]
    g = df.groupby("detector", sort=False)[cols]
    summ = g.mean().round(1).astype(str) + " ± " + g.std().round(1).astype(str)
    print(f"\n{len(nights)} nights, {len(subjects)} subjects; held-out subjects, mean ± SD across {args.splits} splits")
    print("(click 1, no latency; each detector's fixed delay tuned on the tune half)\n")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(summ.to_string())

    # Latency headroom: a trigger-based click can't play before the trigger arrives, so
    # the click lands at max(tuned delay, latency). Pooled over all nights.
    print("\nLatency headroom: % within ±45° if the click lands at max(tuned delay, L) after the trigger")
    lat_ms = (0, 25, 50, 75, 100, 150, 200)
    print(f"{'detector':<55}" + "".join(f"{f'L={L}ms':>9}" for L in lat_ms))
    for name in names:
        C = np.concatenate([n["v"][name]["cos"] for n in nights])
        tuned = df[df.detector == name].delay_ms.median()
        cells = [100 * (C[:, int(round(max(tuned, L) / 25))] >= COS45).mean() for L in lat_ms]
        print(f"{name:<55}" + "".join(f"{c:>9.1f}" for c in cells))


if __name__ == "__main__":
    main()
