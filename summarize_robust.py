"""
Improvement #5: aggregate the robust runs into mean +- 95% CI tables and
paired significance tests.

Reads results/<tag>/robust/{cecr,baselines}_seed*.pkl and writes
results/<tag>/robust/summary.md.

  * CI: Student-t over seeds (each seed = independently trained models +
    an independently drawn set of test conversations)
  * paired tests, per noise level, pooling the per-conversation outcomes of
    every seed (conversations are identical across models within a seed):
      - turns   : Wilcoxon signed-rank (failures count as MAX_ROUNDS)
      - SR@10   : exact McNemar test on the discordant pairs

  python summarize_robust.py --tag movielens
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np
from scipy import stats

from crs_bench import MAX_ROUNDS

METRICS = ("SR@5", "SR@10", "SR@20", "AT")


def load(tag):
    runs = {}                                   # seed -> {model: {noise: result}}
    for path in glob.glob(f"results/{tag}/robust/*_seed*.pkl"):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        with open(path, "rb") as f:
            runs.setdefault(seed, {}).update(pickle.load(f))
    return runs


def ci(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return xs.mean(), float("nan")
    return xs.mean(), stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))


def turns(rounds):
    return np.array([r if r is not None else MAX_ROUNDS for r in rounds], dtype=float)


def succ(rounds, k=10):
    return np.array([r is not None and r <= k for r in rounds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="movielens")
    ap.add_argument("--ref", default="CECR+policy", help="model the paired tests compare against")
    args = ap.parse_args()
    runs = load(args.tag)
    if not runs:
        raise SystemExit(f"no robust runs under results/{args.tag}/robust/")
    # a seed counts only once both the CECR and the baseline runs exist for it
    seeds = sorted(s for s in runs if "CECR" in runs[s] and "EAR" in runs[s])
    # only models present for every seed, so CIs are over the same seeds
    models = sorted(set.intersection(*(set(runs[s]) for s in seeds)))
    noises = sorted(runs[seeds[0]][models[0]])

    lines = [f"# Robust evaluation: {args.tag}", "",
             f"Seeds: {seeds} ({len(seeds)}).  Mean +- 95% CI (t over seeds).  "
             f"Answer noise = probability each attribute answer is flipped.", ""]
    for nz in noises:
        lines += [f"## Answer noise {nz:g}", "",
                  "| Model | " + " | ".join(METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for m in models:
            cells = []
            for k in METRICS:
                mu, h = ci([runs[s][m][nz][k] for s in seeds])
                cells.append(f"{mu:.3f} +- {h:.3f}" if k != "AT" else f"{mu:.2f} +- {h:.2f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        if args.ref in models:
            lines += [f"Paired tests vs **{args.ref}** (pooled over seeds):", "",
                      "| Model | mean turns diff | Wilcoxon p | SR@10 diff | McNemar p |",
                      "|---|---|---|---|---|"]
            for m in models:
                if m == args.ref:
                    continue
                ta = np.concatenate([turns(runs[s][args.ref][nz]["rounds"]) for s in seeds])
                tb = np.concatenate([turns(runs[s][m][nz]["rounds"]) for s in seeds])
                sa = np.concatenate([succ(runs[s][args.ref][nz]["rounds"]) for s in seeds])
                sb = np.concatenate([succ(runs[s][m][nz]["rounds"]) for s in seeds])
                p_w = stats.wilcoxon(ta, tb).pvalue if np.any(ta != tb) else 1.0
                b, c = int(np.sum(sa & ~sb)), int(np.sum(~sa & sb))
                p_m = stats.binomtest(b, b + c).pvalue if b + c else 1.0
                lines.append(f"| {m} | {tb.mean() - ta.mean():+.2f} | {p_w:.2g} | "
                             f"{sb.mean() - sa.mean():+.3f} | {p_m:.2g} |")
            lines += ["", "(positive turns diff = the model needs more turns than the reference)", ""]

    out = f"results/{args.tag}/robust/summary.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
