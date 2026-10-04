"""Phase 0.6: rising-flank trigger + timing model, and a person-relative sleep gate.

Phase 0.5 found that (a) firing on the rising flank (r = +0.5 x trough depth) beats
firing at the trough by ~20 points, and (b) the fixed gate locks many older sleepers
out. Here, on all 153 Sleep-EDF nights, with the rising-flank trigger throughout:

  * gate variants: fixed (Phase 0), person-relative at a few strengths, hypnogram;
  * timing: best fixed delay vs ridge timing model vs ridge + confidence selection;
  * latency: none, and 55 ms (BLE EEG + wired audio) -- a click can't play before
    trigger + latency, so delays below that are off the table;
  * everything broken down by age band.

Delays are measured from when the trigger packet is available. Subject-level splits.

    uv run cloop-phase06 --splits 20
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ALL_RECORDINGS, DATA_DIR, ages, load_subject
from .detector import DetectorConfig, detect
from .evaluate import true_phase
from .predict import TimingModel, features

CACHE = DATA_DIR / "cache"
CACHE_VERSION = 1
DELAYS = np.arange(0.0, 0.601, 0.025)  # after the trigger is available
LATENCIES = (0.0, 0.055)
COS45 = np.cos(np.pi / 4)
AGE_BANDS = [0, 40, 60, 80, 110]
R = 0.5

VARIANTS = {
    "fixed gate (Phase 0)": DetectorConfig(trigger_rise=R),
    "relative, muscle only, k=2": DetectorConfig(trigger_rise=R, gate_relative=True, hf_rel=2.0, eye_rel=0.0),
    "relative k=1.25": DetectorConfig(trigger_rise=R, gate_relative=True, hf_rel=1.25, eye_rel=1.25),
    "relative k=1.5": DetectorConfig(trigger_rise=R, gate_relative=True, hf_rel=1.5, eye_rel=1.5),
    "relative k=2": DetectorConfig(trigger_rise=R, gate_relative=True, hf_rel=2.0, eye_rel=2.0),
    "hypnogram gate (oracle)": DetectorConfig(trigger_rise=R, gate="oracle"),
}


def prepare(rec_id):
    sid, night = rec_id
    path = CACHE / f"p06_SC4{sid:02d}{night}_v{CACHE_VERSION}.npz"
    if not path.exists():
        rec = load_subject(sid, night=night)
        phase = true_phase(rec)
        sleep, wake_rem = np.isin(rec.stages, (2, 3)), np.isin(rec.stages, (0, 4))
        arrays = {"n2n3_h": sleep.sum() / rec.sfreq / 3600}
        for i, cfg in enumerate(VARIANTS.values()):
            t = detect(rec.eeg, rec.sfreq, cfg, rec.stages)
            idx = t.available_idx[:, None] + np.round(DELAYS * rec.sfreq).astype(int)[None, :]
            arrays[f"X{i}"] = features(rec.eeg, rec.sfreq, t).astype(np.float32)
            arrays[f"Y{i}"] = np.cos(phase[np.clip(idx, 0, len(phase) - 1)]).astype(np.float32)
            arrays[f"stage{i}"] = rec.stages[t.available_idx]
            arrays[f"open_sleep{i}"] = t.gate_on[sleep].mean() if sleep.any() else np.nan
            arrays[f"open_wake_rem{i}"] = t.gate_on[wake_rem].mean() if wake_rem.any() else np.nan
        CACHE.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(path)
    z = np.load(path)
    return {"subject": sid, "night": night, "n2n3_h": float(z["n2n3_h"]),
            "v": [{k: z[f"{k}{i}"] for k in ("X", "Y", "stage", "open_sleep", "open_wake_rem")} for i in range(len(VARIANTS))]}


def choose(curve: np.ndarray, latency: float) -> np.ndarray:
    """Best allowed delay per row: can't click before trigger + latency."""
    allowed = DELAYS >= latency - 1e-9
    c = np.where(allowed[None, :], curve, -np.inf)
    return c.argmax(1)


def run_split(k, nights, subjects, age, rng):
    perm = rng.permutation(subjects)
    tune_ids = set(perm[: len(perm) // 2].tolist())
    rows = []
    for vi, vname in enumerate(VARIANTS):
        tr = [n for n in nights if n["subject"] in tune_ids]
        te = [n for n in nights if n["subject"] not in tune_ids]
        X_tr = np.concatenate([n["v"][vi]["X"] for n in tr])
        Y_tr = np.concatenate([n["v"][vi]["Y"] for n in tr])
        model = TimingModel("ridge").fit(X_tr, Y_tr)
        conf_tr = model.pipe.predict(X_tr)
        for lat in LATENCIES:
            fixed_col = choose(Y_tr.mean(0, keepdims=True), lat)[0]
            tr_best = conf_tr[np.arange(len(conf_tr)), choose(conf_tr, lat)]
            cut50 = np.quantile(tr_best, 0.5)
            for n in te:
                v = n["v"][vi]
                base = {"split": k, "variant": vname, "latency_ms": 1000 * lat, "subject": n["subject"], "night": n["night"],
                        "age": age[n["subject"]], "n2n3_h": n["n2n3_h"], "open_sleep": float(v["open_sleep"]),
                        "open_wake_rem": float(v["open_wake_rem"])}
                Y, st = v["Y"], v["stage"]
                if len(Y):
                    curve = model.pipe.predict(v["X"])
                    col = choose(curve, lat)
                    conf = curve[np.arange(len(curve)), col]
                    hit_fixed = Y[:, fixed_col] >= COS45
                    hit_model = Y[np.arange(len(Y)), col] >= COS45
                    sel = conf >= cut50
                else:
                    hit_fixed = hit_model = sel = np.zeros(0, bool)
                for method, hit, keep in (("fixed delay", hit_fixed, np.ones(len(hit_fixed), bool)),
                                          ("ridge", hit_model, np.ones(len(hit_model), bool)),
                                          ("ridge, top 50% conf", hit_model, sel)):
                    rows.append({**base, "method": method, "clicks": int(keep.sum()), "hits": int(hit[keep].sum()),
                                 "in_N2N3": int(np.isin(st[keep], (2, 3)).sum()),
                                 "in_wake_REM": int(np.isin(st[keep], (0, 4)).sum())})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Pool clicks over held-out nights within each split, then mean ± SD across splits."""
    def per_split(g):
        c = g.clicks.sum()
        return pd.Series({
            "pct_within_45": 100 * g.hits.sum() / c if c else np.nan,
            "clicks_per_h": c / g.n2n3_h.sum(),
            "pct_N2N3": 100 * g.in_N2N3.sum() / c if c else np.nan,
            "pct_wake_REM": 100 * g.in_wake_REM.sum() / c if c else np.nan,
            "pct_nights_no_clicks": 100 * (g.clicks == 0).mean(),
            "gate_open_pct_N2N3": 100 * g.open_sleep.mean(),
        })
    keys = [c for c in ("band", "variant", "latency_ms", "method") if c in df]
    s = df.groupby(keys + ["split"], sort=False, observed=True).apply(per_split, include_groups=False)
    g = s.groupby(keys, sort=False, observed=True)
    return (g.mean().round(1).astype(str) + " ± " + g.std().round(1).astype(str)).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-subject", type=int, default=82)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    recs = [r for r in ALL_RECORDINGS if r[0] <= args.max_subject]
    print(f"preparing {len(recs)} nights x {len(VARIANTS)} detector variants ...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        nights = list(ex.map(prepare, recs))
    age = ages()
    subjects = np.array(sorted({n["subject"] for n in nights}))
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.splits):
        rows += run_split(k, nights, subjects, age, rng)
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase06_nights.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 40):
        print(f"\n=== {len(nights)} nights, {len(subjects)} subjects; held-out subjects, mean ± SD across {args.splits} splits ===")
        print("trigger at +0.5 x trough depth; delays from trigger availability\n")
        print(summarise(df).to_string(index=False))
        df["band"] = pd.cut(df.age, AGE_BANDS)
        sub = df[(df.latency_ms == 55) & (df.method != "ridge, top 50% conf")]
        print("\nBy age band (55 ms latency):\n")
        print(summarise(sub).sort_values(["band", "variant"], kind="stable").to_string(index=False))
    print(f"\nwrote {out}/phase06_nights.csv")


if __name__ == "__main__":
    main()
