"""Phase 0.5: does the learned timing model get better with more data?

Phase 0 trained on ~10 nights. Here we use all 153 Sleep-EDF cassette nights
(78 subjects, ages 25-101) and ask, on held-out subjects:

  * learning curve: % of clicks within ±45° as the number of training subjects grows,
    for ridge, the small MLP and a small GRU;
  * context: does seeing more than 2 s before the trough help?
  * selection: does confidence-based skipping (top 50% / 25%) also improve?

Splits are by *subject* (both nights of a person stay on the same side). Each split
holds out half the subjects; training sets are nested subsets of the other half, so
every training size is scored on the same test subjects. Everything is evaluated with
the default detector + causal gate and no latency, from the per-trigger score-vs-delay
matrix: a click at delay d lands within ±45° iff cos(phase error) >= cos(45°).

    uv sync --extra gru
    uv run cloop-timing --splits 10
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ALL_RECORDINGS, DATA_DIR, load_subject
from .detector import DetectorConfig, detect
from .evaluate import true_phase
from .predict import DELAYS, TimingModel, crop, features, targets

CACHE = DATA_DIR / "cache"
CACHE_VERSION = 1  # bump when the detector or features change
MAX_WINDOW_S = 8.0
WINDOWS = (2.0, 4.0, 8.0)
SIZES = (5, 10, 20, 39)  # training subjects (39 = the whole tune half)
KEEP = (1.0, 0.5, 0.25)
COS45 = np.cos(np.pi / 4)


def prepare(rec_id: tuple[int, int]) -> dict:
    """Default detector + causal gate on one night; cache features and targets."""
    sid, night = rec_id
    path = CACHE / f"SC4{sid:02d}{night}_v{CACHE_VERSION}.npz"
    if not path.exists():
        rec = load_subject(sid, night=night)
        trg = detect(rec.eeg, rec.sfreq, DetectorConfig(trigger_rise=None), rec.stages)
        X = features(rec.eeg, rec.sfreq, trg, window_s=MAX_WINDOW_S)
        Y = targets(true_phase(rec), rec.sfreq, trg)
        n2n3_h = np.isin(rec.stages, (2, 3)).sum() / rec.sfreq / 3600
        CACHE.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, X=X.astype(np.float32), Y=Y.astype(np.float32), n2n3_h=n2n3_h,
                            stage=rec.stages[trg.trough_idx])
        tmp.rename(path)
    z = np.load(path)
    return {"subject": sid, "night": night, "X": z["X"], "Y": z["Y"], "n2n3_h": float(z["n2n3_h"]), "stage": z["stage"]}


def stack(nights, key, window=None):
    arr = np.concatenate([n[key] for n in nights])
    return crop(arr, window) if window else arr


def evaluate(name, Y_test, delay_idx, conf, conf_tune, subj_test):
    """% within ±45° on all clicks and on the most confident fractions."""
    hit = Y_test[np.arange(len(Y_test)), delay_idx] >= COS45
    rows = []
    for keep in KEEP:
        sel = np.ones(len(hit), bool) if keep == 1.0 else conf >= np.quantile(conf_tune, 1 - keep)
        per = [100 * hit[sel & (subj_test == s)].mean() for s in np.unique(subj_test) if (sel & (subj_test == s)).any()]
        rows.append({"model": name, "keep": keep, "pct_within_45": 100 * hit[sel].mean(), "n_clicks": int(sel.sum()),
                     "worst_subject": np.min(per), "median_subject": np.median(per)})
    return rows


def run_split(k, nights, subjects, rng, models, skip_windows):
    perm = rng.permutation(subjects)
    half = len(perm) // 2
    train_pool, test_ids = perm[:half], set(perm[half:].tolist())
    test = [n for n in nights if n["subject"] in test_ids]
    Y_test = stack(test, "Y")
    subj_test = np.concatenate([np.full(len(n["Y"]), n["subject"]) for n in test])
    rows = []

    def add(rs, size, window):
        for r in rs:
            rows.append({"split": k, "train_subjects": size, "window_s": window, **r})

    runs = [(size, 2.0) for size in sorted({min(s, half) for s in SIZES})]
    if not skip_windows:
        runs += [(min(SIZES[-1], half), w) for w in WINDOWS if w != 2.0]
    for size, window in runs:
        ids = set(train_pool[:size].tolist())
        train = [n for n in nights if n["subject"] in ids]
        X_tr, Y_tr = stack(train, "X", window), stack(train, "Y")
        X_te = stack(test, "X", window)
        n_train = len(Y_tr)
        # baseline: the single best fixed delay on the training subjects
        best = int(Y_tr.mean(0).argmax())
        fixed = evaluate("fixed delay", Y_test, np.full(len(Y_test), best), np.zeros(len(Y_test)), np.zeros(1), subj_test)
        add([dict(r, n_train_waves=n_train, delay_ms=1000 * DELAYS[best]) for r in fixed[:1]], size, window)
        for kind in models:
            if window != 2.0 and kind == "mlp":
                continue
            m = TimingModel(kind, seed=k).fit(X_tr, Y_tr)
            curve = m.pipe.predict(X_te)
            idx = curve.argmax(1)
            conf = curve[np.arange(len(curve)), idx]
            conf_tune = m.pipe.predict(X_tr).max(1)
            add([dict(r, n_train_waves=n_train) for r in evaluate(kind, Y_test, idx, conf, conf_tune, subj_test)], size, window)
        print(f"    split {k + 1}: {size} subjects, {window:.0f} s window done", flush=True)
    # hindsight ceiling: best delay for every wave
    rows.append({"split": k, "model": "hindsight (best delay per wave)", "keep": 1.0, "train_subjects": np.nan,
                 "window_s": np.nan, "pct_within_45": 100 * (Y_test.max(1) >= COS45).mean()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4, help="parallel nights while preparing (~1 GB RAM each)")
    ap.add_argument("--models", default="ridge,mlp,gru")
    ap.add_argument("--max-subject", type=int, default=82, help="use subjects 0..N only (quick runs)")
    ap.add_argument("--skip-windows", action="store_true", help="skip the 4 s / 8 s context comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="redraw learning_curve.png from the saved CSV")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    models = args.models.split(",")
    if args.plot_only:
        return plot(pd.read_csv(out / "timing_learning_curve.csv"), models, out)

    recs = [r for r in ALL_RECORDINGS if r[0] <= args.max_subject]
    print(f"preparing {len(recs)} nights ...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        nights = list(ex.map(prepare, recs))
    nights = [n for n in nights if len(n["Y"])]
    subjects = np.array(sorted({n["subject"] for n in nights}))
    n_waves = sum(len(n["Y"]) for n in nights)
    in_sleep = 100 * np.mean(np.concatenate([np.isin(n["stage"], (2, 3)) for n in nights]))
    print(f"{len(nights)} nights, {len(subjects)} subjects, {n_waves} triggers ({in_sleep:.1f}% in N2/N3), "
          f"{n_waves / sum(n['n2n3_h'] for n in nights):.0f} per hour of N2+N3", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.splits):
        rows += run_split(k, nights, subjects, rng, models, args.skip_windows)
    df = pd.DataFrame(rows)
    df.to_csv(out / "timing_learning_curve.csv", index=False)

    # --- summary -----------------------------------------------------------------
    cols = ["pct_within_45", "worst_subject", "median_subject"]
    g = df.groupby(["window_s", "train_subjects", "model", "keep"], sort=True, dropna=False)
    summ = g[cols].agg(lambda v: f"{v.mean():.1f} ± {v.std():.1f}").join(g["n_train_waves"].mean().round().rename("train_waves"))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(f"\n=== held-out subjects, mean ± SD across {args.splits} splits ===")
        print(summ.to_string())

    # paired gain over fixed delay (all clicks, 2 s window)
    a = df[(df.keep == 1.0) & (df.window_s == 2.0)].pivot_table(index=["split", "train_subjects"], columns="model", values="pct_within_45")
    print("\nGain over fixed delay (all clicks, 2 s window), points, mean ± SD [splits better]:")
    for size, sub in a.groupby(level="train_subjects"):
        parts = [f"{m}: {(sub[m] - sub['fixed delay']).mean():+.1f} ± {(sub[m] - sub['fixed delay']).std():.1f} "
                 f"[{int((sub[m] > sub['fixed delay']).sum())}/{len(sub)}]" for m in models]
        print(f"  {size:>2} subjects  " + "   ".join(parts))

    plot(df, models, out)
    print(f"\nwrote {out}/timing_learning_curve.csv, learning_curve.png")


def plot(df, models, out):
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, len(KEEP), figsize=(4.2 * len(KEEP), 3.8), sharey=False)
    lc = df[df.window_s == 2.0]
    for ax, keep in zip(axes, KEEP):
        for i, m in enumerate(["fixed delay", *models]):
            s = lc[(lc.model == m) & (lc.keep == keep)].groupby("train_subjects")["pct_within_45"].agg(["mean", "std"])
            if len(s):
                ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, color=f"C{i}", label=m)
        sizes = sorted(lc.train_subjects.dropna().unique().astype(int))
        ax.set(xscale="log", xticks=sizes, xticklabels=sizes, xlabel="training subjects (≈2 nights each)",
               title="all clicks" if keep == 1.0 else f"top {keep:.0%} confidence")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(25, color="grey", ls="--", lw=0.8)
    axes[0].set_ylabel("% clicks within ±45° (held-out subjects)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Timing model learning curve (2 s context, no latency)")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=120)


if __name__ == "__main__":
    main()
