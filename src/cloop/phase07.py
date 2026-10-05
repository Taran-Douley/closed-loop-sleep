"""Phase 0.7: learned causal sleep-stage gate, on the default pipeline.

Default pipeline from here on: rising-flank trigger (r = +0.5) + ridge timing model,
55 ms latency (BLE EEG + wired audio). Gates compared on held-out subjects:

  * fixed (Phase 0) and person-relative k = 1.25 (Phase 0.6) hand-made gates;
  * learned stage gate (stage.py) at a few probability thresholds, retrained on the
    tune subjects of every split;
  * hypnogram gate (oracle) for reference.

One ridge timing model per split, trained on the tune subjects' hypnogram-gated triggers
(i.e. all their real N2/N3 slow waves), is shared by every gate so only the gate differs.

    uv run cloop-phase07 --splits 10
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ALL_RECORDINGS, DATA_DIR, ages, load_subject
from .detector import DetectorConfig, detect
from .evaluate import true_phase
from .phase06 import AGE_BANDS, COS45, summarise
from .predict import FLANK_DELAYS, TimingModel, features, targets
from .stage import StageGate, gate_from_prob, stage_features, stage_labels

CACHE = DATA_DIR / "cache"
CACHE_VERSION = 1
LATENCY = 0.055
THRESHOLDS = (0.5, 0.8, 0.9)
FIXED = {
    "fixed gate (Phase 0)": DetectorConfig(),
    "relative gate k=1.25 (Phase 0.6)": DetectorConfig(gate_relative=True, hf_rel=1.25, eye_rel=1.25),
    "hypnogram gate (oracle)": DetectorConfig(gate="oracle"),
}
TRAIN_EVERY_S = 5  # subsample seconds when training the stage gate
EXACT_GRID = np.arange(0.0, FLANK_DELAYS[-1] + 1e-9, 0.005)  # --delay-step scoring resolution


def _path(rec_id, what):
    return CACHE / f"p07_{what}_SC4{rec_id[0]:02d}{rec_id[1]}_v{CACHE_VERSION}.npz"


def _trigger_arrays(eeg, fs, stages, phase, trg, exact=False):
    out = {"X": features(eeg, fs, trg).astype(np.float32), "Y": targets(phase, fs, trg).astype(np.float32),
           "stage": stages[trg.available_idx]}
    if exact:
        out["P"] = exact_phase(phase, fs, trg.available_idx)
    return out


def exact_phase(phase, fs, available_idx):
    """True phase of a click at each EXACT_GRID delay after the trigger is available,
    interpolated between samples (for scoring delays that aren't whole samples)."""
    z = np.exp(1j * phase.astype(float))
    pos = np.clip(available_idx[:, None] + EXACT_GRID[None, :] * fs, 0, len(z) - 1.001)
    i = pos.astype(int)
    f = pos - i
    return np.angle((1 - f) * z[i] + f * z[i + 1]).astype(np.float32)


def prepare_exact(rec_id):
    """Exact-time phase for the fixed gates' triggers (cached beside the night)."""
    path = _path(rec_id, "exact")
    if not path.exists():
        z = np.load(_path(rec_id, "night"))
        eeg, stages, fs = z["eeg"].astype(float), z["stages"], float(z["fs"])
        arrays = {f"P{i}": exact_phase(z["phase"], fs, detect(eeg, fs, cfg, stages).available_idx)
                  for i, cfg in enumerate(FIXED.values())}
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(path)
    z = np.load(path)
    return [z[f"P{i}"] for i in range(len(FIXED))]


def prepare(rec_id):
    """Split-independent work, cached: signal, stage features, fixed-gate triggers."""
    path = _path(rec_id, "night")
    if not path.exists():
        rec = load_subject(rec_id[0], night=rec_id[1])
        phase = true_phase(rec).astype(np.float32)
        arrays = {"eeg": rec.eeg.astype(np.float32), "stages": rec.stages, "phase": phase, "fs": rec.sfreq,
                  "F": stage_features(rec.eeg, rec.sfreq).astype(np.float32), "labels": stage_labels(rec.stages, rec.sfreq)}
        for i, cfg in enumerate(FIXED.values()):
            t = detect(rec.eeg, rec.sfreq, cfg, rec.stages)
            for k, v in _trigger_arrays(rec.eeg, rec.sfreq, rec.stages, phase, t).items():
                arrays[f"{k}{i}"] = v
            arrays[f"gate{i}"] = _gate_stats(t.gate_on, rec.stages)
        CACHE.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(path)
    z = np.load(path)
    lab = z["labels"]
    keep = np.arange(0, len(lab), TRAIN_EVERY_S)
    keep = keep[lab[keep] >= 0]
    return {"rec": rec_id, "n2n3_h": float(np.isin(z["stages"], (2, 3)).sum() / z["fs"] / 3600),
            "F_train": z["F"][keep], "y_train": np.isin(lab[keep], (2, 3)),
            "fixed": [{k: z[f"{k}{i}"] for k in ("X", "Y", "stage", "gate")} for i in range(len(FIXED))]}


def _gate_stats(gate_on, stages):
    sleep, wr = np.isin(stages, (2, 3)), np.isin(stages, (0, 4))
    return np.array([gate_on[sleep].mean() if sleep.any() else np.nan, gate_on[wr].mean() if wr.any() else np.nan])


def learned_triggers(args):
    """Gate one held-out night with this split's stage probabilities, at every threshold."""
    rec_id, prob, exact = args
    z = np.load(_path(rec_id, "night"))
    eeg, stages, phase, fs = z["eeg"].astype(float), z["stages"], z["phase"], float(z["fs"])
    out = []
    for th in THRESHOLDS:
        g = gate_from_prob(prob, len(eeg), fs, th)
        t = detect(eeg, fs, DetectorConfig(gate="external"), stages, gate_on=g)
        out.append({**_trigger_arrays(eeg, fs, stages, phase, t, exact), "gate": _gate_stats(t.gate_on, stages)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-subject", type=int, default=82)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--delay-step", type=float, default=None,
                    help="choose delays on this grid (s, a multiple of 0.005) and score at the exact click time; "
                         "default: the 25 ms training grid, scored on it (the published Phase 0.7 numbers)")
    ap.add_argument("--timing", choices=("ridge", "pipeline"), default="ridge",
                    help="ridge: the Phase 0.7 timing model; pipeline: the current default (pipeline.fit_timing)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    step, exact = args.delay_step, args.delay_step is not None
    if exact:
        assert abs(step / 0.005 - round(step / 0.005)) < 1e-6, "--delay-step must be a multiple of 0.005"
    suffix = (f"_step{round(1000 * step)}ms" if exact else "") + ("_pipeline" if args.timing == "pipeline" else "")

    recs = [r for r in ALL_RECORDINGS if r[0] <= args.max_subject]
    print(f"preparing {len(recs)} nights ...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        nights = list(ex.map(prepare, recs))
        if exact:
            for n, ps in zip(nights, ex.map(prepare_exact, recs)):
                for v, p in zip(n["fixed"], ps):
                    assert len(p) == len(v["X"]), n["rec"]
                    v["P"] = p
    age = ages()
    subjects = np.array(sorted({n["rec"][0] for n in nights}))
    rng = np.random.default_rng(args.seed)
    oracle = list(FIXED).index("hypnogram gate (oracle)")
    rows = []
    for k in range(args.splits):
        perm = rng.permutation(subjects)
        tune_ids = set(perm[: len(perm) // 2].tolist())
        tr = [n for n in nights if n["rec"][0] in tune_ids]
        te = [n for n in nights if n["rec"][0] not in tune_ids]

        gate = StageGate(seed=k).fit(np.concatenate([n["F_train"] for n in tr]), np.concatenate([n["y_train"] for n in tr]))
        X_tr = np.concatenate([n["fixed"][oracle]["X"] for n in tr])
        Y_tr = np.concatenate([n["fixed"][oracle]["Y"] for n in tr])
        if args.timing == "pipeline":
            from .pipeline import fit_timing

            timing = fit_timing(X_tr, Y_tr, LATENCY, step)
            print(f"  split {k + 1}: stretch {timing.stretch:.2f}", flush=True)
        else:
            timing = TimingModel("ridge", delays=FLANK_DELAYS).fit(X_tr, Y_tr)
        cut50 = np.quantile(timing.predict(X_tr, LATENCY, step)[1], 0.5)

        # stage probabilities in the parent; workers are spawned, not forked, because
        # HistGradientBoosting's OpenMP threads deadlock forked children
        probs = [gate.prob(np.load(_path(n["rec"], "night"))["F"]) for n in te]
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            learned = list(ex.map(learned_triggers, [(n["rec"], p, exact) for n, p in zip(te, probs)]))
        for n, lv in zip(te, learned):
            variants = dict(zip(FIXED, n["fixed"]))
            variants.update({f"learned gate p>={th}": v for th, v in zip(THRESHOLDS, lv)})
            for name, v in variants.items():
                Y, st = v["Y"], v["stage"]
                if len(Y):
                    delay, conf = timing.predict(v["X"], LATENCY, step)
                    if exact:
                        hit = np.abs(v["P"][np.arange(len(Y)), np.round(delay / 0.005).astype(int)]) <= np.pi / 4
                    else:
                        col = np.round(delay / 0.025).astype(int)
                        hit = Y[np.arange(len(Y)), col] >= COS45
                else:
                    hit = conf = np.zeros(0)
                base = {"split": k, "variant": name, "latency_ms": 1000 * LATENCY, "subject": n["rec"][0], "night": n["rec"][1],
                        "age": age[n["rec"][0]], "n2n3_h": n["n2n3_h"], "open_sleep": v["gate"][0], "open_wake_rem": v["gate"][1]}
                for method, keep in (("ridge", np.ones(len(hit), bool)), ("ridge, top 50% conf", conf >= cut50)):
                    rows.append({**base, "method": method, "clicks": int(keep.sum()), "hits": int(hit[keep].sum()),
                                 "in_N2N3": int(np.isin(st[keep], (2, 3)).sum()), "in_wake_REM": int(np.isin(st[keep], (0, 4)).sum())})
        print(f"  split {k + 1}/{args.splits} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"phase07_nights{suffix}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 40):
        print(f"\n=== {len(nights)} nights, {len(subjects)} subjects; held-out subjects, mean ± SD across {args.splits} splits ===")
        grid = f"delays every {1000 * step:.0f} ms, scored at the exact click time" if exact else "delays on the 25 ms grid"
        print(f"rising-flank trigger, ridge timing, 55 ms latency, {grid}\n")
        print(summarise(df.drop(columns="latency_ms")).to_string(index=False))
        df["band"] = pd.cut(df.age, AGE_BANDS)
        print("\nBy age band (ridge, all clicks):\n")
        print(summarise(df[df.method == "ridge"].drop(columns=["latency_ms", "method"])).sort_values(["band", "variant"], kind="stable").to_string(index=False))
    print(f"\nwrote {out}/phase07_nights{suffix}.csv")


if __name__ == "__main__":
    main()
