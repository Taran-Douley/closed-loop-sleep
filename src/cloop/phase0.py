"""Phase 0: how accurately could a real-time system hit the slow-wave up-state?

Repeated random subject splits (as in the Portiloop paper): each split tunes timing
parameters on half the subjects and scores the other, held-out half. We report the
mean and spread across splits, so a lucky split can't flatter the result.

    uv run cloop-phase0 --subjects 0-19 --splits 20
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_subject
from .detector import DetectorConfig, detect
from .evaluate import score, true_phase
from .predict import TimingModel, features, targets
from .timing import SCENARIOS, Latency, Strategy, schedule

DEFAULT = "adaptive p5, whole night (default)"
DETECTORS = {
    "fixed -40 uV": DetectorConfig(threshold="fixed", trough_uv=-40),
    "fixed -60 uV": DetectorConfig(threshold="fixed", trough_uv=-60),
    "fixed -80 uV": DetectorConfig(threshold="fixed", trough_uv=-80),
    "adaptive p10, 10-min memory": DetectorConfig(adapt_pct=10, adapt_window=500),
    "adaptive p10, whole night": DetectorConfig(adapt_pct=10),
    DEFAULT: DetectorConfig(adapt_pct=5),
    "adaptive p2, whole night": DetectorConfig(adapt_pct=2),
    "oracle gate": DetectorConfig(gate="oracle"),  # default threshold, hypnogram gate
}
FAMILIES = {
    "fixed delay": [Strategy("fixed", delay=d) for d in np.arange(0.3, 0.701, 0.025)],
    "adaptive": [Strategy("adaptive", offset=o) for o in np.arange(-0.3, 0.301, 0.025)],
    "adaptive + filter corr": [Strategy("adaptive", offset=o, filter_correction=True) for o in np.arange(-0.3, 0.301, 0.025)],
}
JITTER_SDS = np.arange(0, 0.151, 0.025)
MODELS = ("ridge", "mlp")
KEEP = (0.5, 0.25)  # model-confidence selective stimulation: fraction of waves to click on


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def prepare(sid: int) -> dict:
    """Load one subject and run every detector config once (detection doesn't depend on the split)."""
    rec = load_subject(sid)
    trg = {}
    for name, cfg in DETECTORS.items():
        t = detect(rec.eeg, rec.sfreq, cfg, rec.stages)
        trg[name] = replace(t, gate_on=None)  # per-sample array not needed downstream
    phase = true_phase(rec).astype(np.float32)
    # timing-model inputs (causal) and training targets (offline truth), default detector only
    X = features(rec.eeg, rec.sfreq, trg[DEFAULT])
    Y = targets(phase, rec.sfreq, trg[DEFAULT])
    n2n3_h = np.isin(rec.stages, (2, 3)).sum() / rec.sfreq / 3600
    hours = rec.hours
    rec.eeg = np.empty(0)  # keep memory low: scoring needs only stages + phase
    return {"rec": rec, "phase": phase, "trg": trg, "X": X, "Y": Y, "n2n3_h": n2n3_h, "hours": hours}


def pooled(subjects, detector, strat, lat, which=0):
    """Score sound events pooled across subjects."""
    errs, parts = [], []
    for s in subjects:
        rng = np.random.default_rng(s["rec"].subject)  # same latency draws in every split
        clicks = schedule(s["trg"][detector], s["rec"].sfreq, strat, lat, rng)
        sc = score(clicks[which], s["phase"], s["rec"])
        if sc["n"]:
            errs.append(sc["errors"])
            parts.append((sc, clicks[2].mean()))
    n = sum(p[0]["n"] for p in parts)
    if n == 0:
        return {"n": 0, "per_hour_N2N3": 0.0, "errors": np.empty(0)}
    err = np.concatenate(errs)
    z = np.mean(np.exp(1j * err))
    w = lambda k: sum(p[0][k] * p[0]["n"] for p in parts) / n  # noqa: E731
    return {
        "n": n,
        "per_hour_N2N3": n / sum(s["n2n3_h"] for s in subjects),
        "mean_err_deg": float(np.degrees(np.angle(z))),
        "R": float(np.abs(z)),
        "mean_cos": float(np.mean(np.cos(err))),
        "pct_within_45": float(100 * np.mean(np.abs(err) <= np.pi / 4)),
        "pct_N2N3": w("pct_N2N3"),
        "pct_wake_REM": w("pct_wake_REM"),
        "pct_late": 100 * sum(p[1] * p[0]["n"] for p in parts) / n,
        "errors": err,
    }


def tune(subjects, detector, cands):
    vals = [pooled(subjects, detector, c, SCENARIOS[0]).get("mean_cos", -np.inf) for c in cands]
    return cands[int(np.argmax(vals))], vals


def param_ms(s: Strategy) -> float:
    if s.kind == "predicted":
        return np.nan
    return 1000 * (s.delay if s.kind == "fixed" else s.offset)


def with_models(tune_s, test_s, seed):
    """Train timing models on the tune subjects only; attach their per-trigger delays to
    copies of the test subjects as extra 'detectors' (all clicks, and confidence-selected)."""
    X = np.concatenate([s["X"] for s in tune_s])
    Y = np.concatenate([s["Y"] for s in tune_s])
    out = [dict(s, trg=dict(s["trg"])) for s in test_s]
    for kind in MODELS:
        model = TimingModel(kind, seed=seed).fit(X, Y)
        conf_tune = model.predict(X)[1]
        cut = {keep: np.quantile(conf_tune, 1 - keep) for keep in KEEP}
        for s in out:
            delay, conf = model.predict(s["X"])
            base = replace(s["trg"][DEFAULT], pred_delay=delay)
            s["trg"][f"model: {kind}"] = base
            for keep in KEEP:
                s["trg"][f"model: {kind}, top {keep:.0%} conf"] = base.subset(conf >= cut[keep])
    return out


def run_split(k, tune_s, test_s):
    """One split: returns (main rows, threshold rows, jitter rows, tuning curves, polar errors)."""
    main, thr, jit, curves, polar = [], [], [], {}, {}
    no_err = lambda r: {k: v for k, v in r.items() if k != "errors"}  # noqa: E731

    # strategies x latency x gate, using the default detector
    best = {}
    for fam, cands in FAMILIES.items():
        best[fam], curves[fam] = tune(tune_s, DEFAULT, cands)
    test_m = with_models(tune_s, test_s, seed=k)
    runs = [(DEFAULT, "causal", fam, strat) for fam, strat in best.items()]
    runs += [(f"model: {m}", "causal", f"model: {m}", Strategy("predicted")) for m in MODELS]
    runs += [("oracle gate", "oracle", fam, strat) for fam, strat in best.items()]
    for det, gate, fam, strat in runs:
        for lat in SCENARIOS:
            for which in (0, 1):
                r = pooled(test_m, det, strat, lat, which)
                if gate == "causal" and which == 0 and fam == "fixed delay":
                    polar[lat.name] = r["errors"]
                main.append({"split": k, "gate": gate, "family": fam, "param_ms": param_ms(strat), "latency": lat.name,
                             "click": which + 1, **no_err(r)})

    # what to stimulate: thresholds (each with its own tuned fixed delay) vs model selection
    selections = [(d, None) for d in DETECTORS if d != "oracle gate"]
    selections += [(f"model: {m}, top {keep:.0%} conf", Strategy("predicted")) for m in MODELS for keep in KEEP]
    for det, strat in selections:
        if strat is None:
            strat, _ = tune(tune_s, det, FAMILIES["fixed delay"])
        r = pooled(test_m, det, strat, SCENARIOS[0])
        rates = [len(s["trg"][det].trough_idx) / s["n2n3_h"] for s in test_m]
        per = [pooled([s], det, strat, SCENARIOS[0]).get("pct_within_45", np.nan) for s in test_m]
        thr.append({"split": k, "threshold": det, "param_ms": param_ms(strat), "pct_within_45": r.get("pct_within_45", np.nan),
                    "worst_subject_within_45": np.nanmin(per), "best_subject_within_45": np.nanmax(per),
                    "per_hour_N2N3": r["per_hour_N2N3"], "min_subject_per_hour": min(rates), "max_subject_per_hour": max(rates),
                    "subjects_with_no_clicks": sum(x == 0 for x in rates)})

    # audio jitter tolerance (mean latency compensated), fixed-delay strategy
    for sd in JITTER_SDS:
        r = pooled(test_s, DEFAULT, best["fixed delay"], Latency("j", 0.04, 0.01, 0.2, sd))
        jit.append({"split": k, "audio_sd_ms": sd * 1000, "pct_within_45": r["pct_within_45"]})
    return main, thr, jit, curves, polar


def mean_sd(df, by, cols):
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.1f} ± {b:.1f}" for a, b in zip(m[c], s[c])]
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="0-19", help="Sleep-EDF SC subject ids, e.g. 0-19 or 0,2,5-7")
    ap.add_argument("--splits", type=int, default=20, help="number of random half/half subject splits")
    ap.add_argument("--workers", type=int, default=3, help="parallel subjects (each needs ~1 GB RAM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    ids = parse_range(args.subjects)
    print(f"preparing {len(ids)} subjects x {len(DETECTORS)} detector configs ...")
    with ProcessPoolExecutor(args.workers) as ex:
        subjects = {s["rec"].subject: s for s in ex.map(prepare, ids)}
    for sid, s in subjects.items():
        print(f"  subject {sid}: {s['hours']:.1f} h, {s['n2n3_h']:.1f} h N2/N3, {len(s['trg'][DEFAULT].trough_idx)} triggers (default detector)")

    rng = np.random.default_rng(args.seed)
    main_rows, thr_rows, jit_rows, all_curves = [], [], [], []
    for k in range(args.splits):
        perm = rng.permutation(ids)
        half = len(ids) // 2
        tune_s = [subjects[i] for i in perm[:half]]
        test_s = [subjects[i] for i in perm[half:]]
        m, t, j, curves, p = run_split(k, tune_s, test_s)
        main_rows += m
        thr_rows += t
        jit_rows += j
        all_curves.append(curves)
        if k == 0:
            polar = p
        print(f"  split {k + 1}/{args.splits} done")

    main_df, thr_df, jit_df = pd.DataFrame(main_rows), pd.DataFrame(thr_rows), pd.DataFrame(jit_rows)
    main_df.to_csv(out / "results.csv", index=False)
    thr_df.to_csv(out / "thresholds.csv", index=False)
    jit_df.to_csv(out / "jitter.csv", index=False)

    # --- plots -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (fam, cands) in enumerate(FAMILIES.items()):
        x = [param_ms(c) for c in cands]
        for j, curves in enumerate(all_curves):
            ax.plot(x, curves[fam], color=f"C{i}", alpha=0.25, lw=1, label=fam if j == 0 else None)
    ax.set(xlabel="delay after trough (fixed) / offset (adaptive), ms", ylabel="mean cos(phase error)",
           title=f"Tuning curves, {len(all_curves)} splits (tune subjects, no latency)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tuning.png", dpi=120)

    js = jit_df.groupby("audio_sd_ms")["pct_within_45"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(js.index, js["mean"], marker="o")
    ax.fill_between(js.index, js["mean"] - js["std"], js["mean"] + js["std"], alpha=0.25)
    ax.axhline(25, color="grey", ls="--", label="chance (25%)")
    ax.set(xlabel="audio latency jitter (SD, ms) -- mean latency compensated", ylabel="% clicks within ±45° of up-state", ylim=(0, 100))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "jitter.png", dpi=120)

    fig, axes = plt.subplots(1, len(SCENARIOS), subplot_kw={"projection": "polar"}, figsize=(4 * len(SCENARIOS), 4.3))
    for ax, lat in zip(axes, SCENARIOS):
        ax.hist(polar[lat.name], bins=36, density=True)
        ax.set_title(lat.name.replace(", ", ",\n"), fontsize=9)
        ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["up-state\n(target)", "", "trough", ""])
    fig.suptitle("Phase at click 1 -- held-out subjects of split 1, causal gate, fixed delay")
    fig.tight_layout()
    fig.savefig(out / "phase_polar.png", dpi=120)

    # --- summary -----------------------------------------------------------------
    cols = ["pct_within_45", "mean_err_deg", "R", "param_ms", "per_hour_N2N3", "pct_late", "pct_N2N3", "pct_wake_REM"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40):
        print(f"\n=== mean ± SD across {args.splits} splits, held-out subjects ===")
        print("\nStrategy x latency (causal gate, click 1):")
        c1 = main_df[(main_df.gate == "causal") & (main_df.click == 1)]
        print(mean_sd(c1, ["family", "latency"], cols[:6]).to_string(index=False))
        print("\nGate (fixed delay, no latency, click 1):")
        g = main_df[(main_df.family == "fixed delay") & (main_df.latency == SCENARIOS[0].name) & (main_df.click == 1)]
        print(mean_sd(g, ["gate"], ["pct_within_45", "per_hour_N2N3", "pct_N2N3", "pct_wake_REM"]).to_string(index=False))
        print("\nSecond click (fixed delay, causal gate):")
        c2 = main_df[(main_df.family == "fixed delay") & (main_df.gate == "causal") & (main_df.click == 2)]
        print(mean_sd(c2, ["latency"], ["pct_within_45", "mean_err_deg"]).to_string(index=False))
        print("\nWhat to stimulate: trough thresholds (fixed delay tuned per threshold) vs timing-model confidence (causal gate, no latency):")
        print(mean_sd(thr_df, ["threshold"], ["pct_within_45", "worst_subject_within_45", "best_subject_within_45",
                                               "per_hour_N2N3", "min_subject_per_hour", "max_subject_per_hour",
                                               "subjects_with_no_clicks", "param_ms"]).to_string(index=False))
        print("\nAudio jitter (% within ±45°):  " + ", ".join(f"{i:.0f} ms: {r['mean']:.1f}±{r['std']:.1f}" for i, r in js.iterrows()))
    print(f"\nwrote {out}/results.csv, thresholds.csv, jitter.csv, tuning.png, jitter.png, phase_polar.png")


if __name__ == "__main__":
    main()
