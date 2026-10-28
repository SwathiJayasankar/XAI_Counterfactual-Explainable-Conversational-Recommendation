"""
Renders, from OUR runs, exactly the four artefacts in 'papper output/', in the
paper's format: Table IV, Fig. 3, Fig. 4, Fig. 6 (Yelp). Writes them to
reimplementation/figures/ and the numbers to reimplementation/REIMPLEMENTATION_RESULTS.md.

  python make_paper_figs.py
"""
import csv
import glob
import os
import pickle
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figs import SURFACE

ROOT = "reimplementation"
OUT = f"{ROOT}/figures"
DATASETS = [("yelp", "Yelp"), ("douban", "Douban-Book"), ("movielens", "MovieLens")]
MODELS = ["CECR", "FPAN", "EAR", "UNICORN", "KBRD"]          # the paper's legend order

ABL = ["- GNNs", "- PreRi", "- CouRe", "- RecPe", "CECR (CCR)"]
ABL_LBL_PAPER = ["GNNs", "PreRi", "ConRe", "RecPe", "CCR"]   # Fig. 4 prints "ConRe"

# ---------------- the paper's figure styling (read off the published figures) ----------------
P_FIG3 = dict(CECR=("tab:blue", "-"), FPAN=("tab:orange", "--"), EAR=("tab:green", ":"),
              UNICORN=("tab:red", "--"), KBRD=("tab:purple", "-."))
P_FIG4_BARS = ["#e48085", "#b5973f", "#6cad55", "#5bb79b", "#58b5dd"]
P_FIG6_TIME = dict(EAR="#ee7592", UNICORN="#7a8bf3", FPAN="#fbd77c", KBRD="#9d78f3",
                   CECR="#98f6b4")
P_FIG6_LAM = [0, 25, 50, 75, 100]
P_FIG6_K = [1, 5, 10, 20, 50]


def load(tag):
    """Merge results/<tag>/cecr.pkl and baselines.pkl into {model: metrics}."""
    perf, abl, expl, extra = {}, {}, {}, {}
    p = f"results/{tag}/cecr.pkl"
    if not os.path.exists(p) and os.path.exists(f"results/{tag}/cecr_table2.csv"):
        with open(f"results/{tag}/cecr_table2.csv") as f:      # run still in progress
            for r in csv.DictReader(f):
                abl[r["Model"]] = {k: float(r[k]) for k in r if k != "Model"}
        if "CECR (CCR)" in abl:
            perf["CECR"] = abl["CECR (CCR)"]
        if os.path.exists(f"results/{tag}/fig6bc_topk.csv"):
            with open(f"results/{tag}/fig6bc_topk.csv") as f:
                extra["topk"] = [(int(r["K"]), float(r["AT"]), float(r["SR@20"]))
                                 for r in csv.DictReader(f)]
    if os.path.exists(p):
        with open(p, "rb") as f:
            c = pickle.load(f)
        abl = c["results"]
        if "CECR (CCR)" in abl:
            perf["CECR"] = abl["CECR (CCR)"]
        expl.update(c.get("expl", {}))
        extra = dict(lam=c.get("lam", []), topk=c.get("topk", []))
    p = f"results/{tag}/baselines.pkl"
    if os.path.exists(p):
        with open(p, "rb") as f:
            b = pickle.load(f)
        for m in MODELS[1:]:
            if m in b:
                perf[m] = b[m]
        expl.update(b.get("table4", {}))
    return perf, abl, expl, extra


def save(fig, path, face=SURFACE):
    fig.savefig(path, dpi=160, facecolor=face, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
#  Paper-format renderers
# ============================================================================
def _improv(vals, lower_better=False):
    """Paper's Improv. %: CECR relative to the best baseline."""
    best = min(vals[:-1]) if lower_better else max(vals[:-1])
    if best == 0:
        return float("nan")
    return 100 * ((best - vals[-1]) if lower_better else (vals[-1] - best)) / best


def draw_table(fname, title, subtitle, groups, metrics, row_names, cols, lower_better, fmt,
               footnote="The best results are bolded, while the best baselines are underlined."):
    """A TKDE-style grid table: one column group per dataset, an 'Improv. %' last row.
    cols: list (per group*metric) of value lists aligned with row_names; None = not run."""
    n_m = len(metrics)
    n_c = len(groups) * n_m
    n_r = len(row_names) + 1                       # + Improv. %
    w0, wc, h = 1.25, 0.78, 0.30
    W, H = w0 + n_c * wc, (n_r + 2) * h
    fig = plt.figure(figsize=(W + 0.3, H + 1.25), facecolor="white")
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.66])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=13, family="serif")
    fig.text(0.5, 0.86, subtitle, ha="center", va="center", fontsize=11, family="serif",
             variant="small-caps")
    kw = dict(ha="center", va="center", family="serif", fontsize=9.5)
    # header rows
    y1, y2 = H - h / 2, H - 1.5 * h
    ax.text(w0 / 2, y1, "Dataset", **kw)
    ax.text(w0 - 0.08, y2 + 0.07, "Metrics", ha="right", va="center", family="serif", fontsize=8.5)
    ax.text(0.08, y2 - 0.08, "Model", ha="left", va="center", family="serif", fontsize=8.5)
    ax.plot([0, w0], [H - h, H - 2 * h], color="k", lw=0.6)
    for g, gname in enumerate(groups):
        x0 = w0 + g * n_m * wc
        ax.text(x0 + n_m * wc / 2, y1, gname, **kw)
        for k, m in enumerate(metrics):
            ax.text(x0 + (k + 0.5) * wc, y2, m, **kw)
    # body
    for c in range(n_c):
        vals = cols[c]
        lb = lower_better[c % n_m]
        done = [v for v in vals if v is not None]
        base = [v for v in vals[:-1] if v is not None]
        best = (min(done) if lb else max(done)) if done else None
        bbest = (min(base) if lb else max(base)) if base else None
        x = w0 + (c + 0.5) * wc
        for r, v in enumerate(vals):
            y = H - (r + 2.5) * h
            if v is None:
                ax.text(x, y, "n/a", color="#888", **kw)
                continue
            s = fmt(v)
            bold = v == best and len(done) > 1
            ax.text(x, y, s, fontweight="bold" if bold else "normal", **kw)
            if v == bbest and len(base) > 1 and r < len(vals) - 1:
                tw = 0.068 * len(s)                    # ~text width in data units (inches)
                ax.plot([x - tw / 2, x + tw / 2], [y - 0.11, y - 0.11], color="k", lw=0.7)
        y = H - (n_r + 1.5) * h
        if None in vals:
            ax.text(x, y, "n/a", color="#888", **kw)
        else:
            ax.text(x, y, f"{_improv(vals, lb):.2f}", **kw)
    for r, name in enumerate(row_names + ["Improv. %"]):
        ax.text(w0 / 2, H - (r + 2.5) * h, name, **kw)
    # grid
    for yy in [0, H - 2 * h] + [H - (r + 2) * h for r in range(1, n_r)] + [H]:
        lw = 0.9 if yy in (0, H, H - 2 * h) else 0.5
        ax.plot([0, W], [yy, yy], color="k", lw=lw)
    ax.plot([w0, W], [H - h, H - h], color="k", lw=0.5)
    for xx in [0, w0, W] + [w0 + g * n_m * wc for g in range(1, len(groups))]:
        ax.plot([xx, xx], [0, H], color="k", lw=0.9)
    for c in range(1, n_c):
        if c % n_m:
            xx = w0 + c * wc
            ax.plot([xx, xx], [0, H - h], color="k", lw=0.5)
    fig.text(0.04, 0.05, footnote, ha="left", va="center", family="serif", fontsize=10)
    save(fig, fname, face="white")


def table4_paper(all_expl):
    groups = ["Yelp", "MovieLens"]
    metrics = ["Pr%", "Re%", "PN%", "PS%"]
    keys = ["Precision", "Recall", "PN", "PS"]
    rows = ["CPR", "CAEC", "SAUR", "Random", "CECR"]
    cols = []
    for tag in ("yelp", "movielens"):
        ex = all_expl.get(tag, {})
        for k in keys:
            cols.append([100 * ex[r][k] if r in ex else None for r in rows])
    # Improv. % compares CECR with the best *explainable* baseline, Random excluded
    draw_table(f"{OUT}/table4.png", "TABLE IV", "Explainability Comparison", groups,
               metrics, rows, cols, [False] * 4, lambda v: f"{v:.2f}")


def fig3_paper(all_perf):
    """Paper layout: Yelp and MovieLens on top, Douban-Book centred below."""
    fig = plt.figure(figsize=(9.6, 6.6), facecolor="white")
    pos = {"yelp": [0.06, 0.56, 0.40, 0.38], "movielens": [0.56, 0.56, 0.40, 0.38],
           "douban": [0.31, 0.06, 0.40, 0.38]}
    for tag, title in DATASETS:
        ax = fig.add_axes(pos[tag])
        perf = all_perf[tag]
        x = np.arange(1, 11)
        for m in MODELS:
            c, ls = P_FIG3[m]
            if m in perf and "curve" in perf[m]:
                ax.plot(x, perf[m]["curve"][:10], color=c, ls=ls, marker="o", ms=4, lw=1.2,
                        label=m)
            else:
                ax.plot([], [], color=c, ls=ls, marker="o", ms=4, lw=1.2, label=f"{m} (n/a)")
        ax.set_title(f"SR on {title}", fontsize=12)
        ax.set_xlabel("Conversation Rounds", fontsize=11)
        ax.set_ylabel("Success Rate", fontsize=11)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_ylim(-0.04, 1.0)
        ax.grid(True, color="#d0d0d0", lw=0.8)
        ax.legend(loc="lower right", fontsize=9, facecolor="#e0e0e0", framealpha=1,
                  edgecolor="#e0e0e0")
    save(fig, f"{OUT}/fig3.png", face="white")


def fig4_paper(all_abl):
    tags = [t for t in ("movielens", "yelp") if all(a in all_abl.get(t, {}) for a in ABL)]
    if not tags:
        return
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(len(tags), 2, figsize=(7.6, 3.0 * len(tags)), squeeze=False)
        for row, tag in zip(axes, tags):
            title = dict(DATASETS)[tag]
            abl = all_abl[tag]
            sr = [abl[a]["SR@20"] for a in ABL]
            at = [abl[a]["AT"] for a in ABL]
            x = np.arange(len(ABL))
            row[0].bar(x, sr, 0.8, color=P_FIG4_BARS)
            for xi, v in zip(x, sr):
                row[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
            row[0].set_xticks(x, ABL_LBL_PAPER, fontsize=8)
            row[0].set_ylim(0, 1.0)
            row[0].set_ylabel("Success Rate@20", fontsize=10)
            row[0].set_title(f"SR@20 on {title}", fontsize=11)
            row[0].tick_params(axis="y", labelsize=8)
            row[1].plot(x, at, color="#e8423f", marker="o", ms=4, lw=1.2)
            row[1].set_xticks(x, ABL_LBL_PAPER, fontsize=10)
            row[1].set_ylabel("Average Turn", fontsize=10)
            row[1].set_title(f"AT on {title}", fontsize=11)
        fig.tight_layout()
        save(fig, f"{OUT}/fig4.png", face="white")


def fig6_paper(extra, perf):
    fig, ax = plt.subplots(2, 2, figsize=(8.2, 7.0), facecolor="white")
    grid = dict(ls="--", color="#999", lw=0.6)
    # (a) lambda
    a = ax[0, 0]
    lam = {int(r[0]): r for r in extra.get("lam", [])}
    L = [l for l in P_FIG6_LAM if l in lam]
    if L:
        a.plot(L, [np.log1p(lam[l][2]) for l in L], color="#e8423f", alpha=0.7, lw=3,
               marker="o", ms=4, label="complexity")
        a.plot(L, [np.log1p(max(lam[l][1], 0)) for l in L], color="#2f2ff0", alpha=0.7, lw=3,
               marker="o", ms=4, label="strength")
        a.legend(loc="lower right", fontsize=9)
    else:
        a.text(0.5, 0.5, "not run", transform=a.transAxes, ha="center", color="#888")
    a.set_xticks(P_FIG6_LAM); a.set_xlabel("Lambda"); a.set_ylabel("Log Value")
    a.set_title("(a) Impacts of $\\lambda$ for counterfactual explanation.", fontsize=10, y=-0.3)
    # (b), (c) Top-K: categorical x axis, like the paper
    tk = {int(r[0]): r for r in extra.get("topk", [])}
    K = [k for k in P_FIG6_K if k in tk]
    b, c = ax[0, 1], ax[1, 0]
    if K:
        b.plot(range(len(K)), [tk[k][1] for k in K], color="#fbb829", alpha=0.7, lw=3,
               marker="o", ms=4, label="AT")
        c.plot(range(len(K)), [tk[k][2] for k in K], color="#2e9d2e", alpha=0.7, lw=3,
               marker="o", ms=4, label="SR@20")
        for p in (b, c):
            p.set_xticks(range(len(K)), [str(k) for k in K])
            p.legend(loc="upper right" if p is b else "upper left", fontsize=9)
    b.set_xlabel("Top-K"); b.set_ylabel("Round Number")
    b.set_title("(b) Impacts of Top-$K$ for AT.", fontsize=10, y=-0.3)
    c.set_xlabel("Top-K"); c.set_ylabel("Accuracy Number")
    c.set_title("(c) Impacts of Top-$K$ for SR@20.", fontsize=10, y=-0.3)
    # (d) response time per conversation
    d = ax[1, 1]
    order = ["EAR", "UNICORN", "FPAN", "KBRD", "CECR"]
    present = [m for m in order if m in perf]
    vals = [perf[m]["sec_per_conv"] for m in present]
    d.bar(present, vals, 0.8, color=[P_FIG6_TIME[m] for m in present])
    for i, v in enumerate(vals):
        d.text(i, v * 1.01, f"{v:.2f}" if v >= 0.1 else f"{v:.3f}", ha="center", va="bottom",
               fontsize=8)
    if vals:
        d.set_ylim(0, max(vals) * 1.12)
    d.set_axisbelow(True)
    d.set_xlabel("Different Models"); d.set_ylabel("Response Time (seconds)")
    d.set_title("(d) Time complexity comparison.", fontsize=10, y=-0.3)
    for p in ax.flat:
        p.grid(True, **grid)
        p.tick_params(labelsize=8)
    fig.tight_layout(h_pad=3.0)
    save(fig, f"{OUT}/fig6.png", face="white")


# ============================================================================
#  Markdown report
# ============================================================================
def pct(x):
    return f"{100 * x:.2f}"


def report(all_perf, all_abl, all_expl, yelp_extra):
    """Only the four artefacts in 'papper output/': Table IV, Fig. 3, Fig. 4, Fig. 6."""
    L = ["# CECR reimplementation - Table IV, Fig. 3, Fig. 4, Fig. 6", "",
         "Our own runs, drawn in the same format as the paper (`papper output/`). "
         "What each one means and why it differs from the paper: "
         "[`EXPLANATION.md`](EXPLANATION.md).", ""]
    L += ["## Table IV - Explainability comparison", "", "![Table IV](figures/table4.png)", "",
          "| Model | Yelp Pr% | Re% | PN% | PS% | MovieLens Pr% | Re% | PN% | PS% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ("CPR", "CAEC", "SAUR", "Random", "CECR"):
        cells = []
        for tag in ("yelp", "movielens"):
            ex = all_expl.get(tag, {})
            cells += ([pct(ex[m][k]) for k in ("Precision", "Recall", "PN", "PS")]
                      if m in ex else ["n/a"] * 4)
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Fig. 3 - SR by conversation round", "", "![Fig 3](figures/fig3.png)", "",
          "| Dataset | Model | " + " | ".join(f"R{r}" for r in range(1, 11)) + " |",
          "|---|---|" + "---|" * 10]
    for tag, title in DATASETS:
        for m in MODELS:
            if m in all_perf[tag] and "curve" in all_perf[tag][m]:
                L.append(f"| {title} | {m} | " + " | ".join(
                    f"{v:.3f}" for v in all_perf[tag][m]["curve"][:10]) + " |")
    L += ["", "## Fig. 4 - Ablation study (MovieLens, Yelp)", "", "![Fig 4](figures/fig4.png)", "",
          "| Dataset | Metric | " + " | ".join(ABL_LBL_PAPER) + " |", "|---|---|" + "---|" * 5]
    for tag in ("movielens", "yelp"):
        abl = all_abl.get(tag, {})
        if all(a in abl for a in ABL):
            L.append(f"| {dict(DATASETS)[tag]} | SR@20 | " +
                     " | ".join(f"{abl[a]['SR@20']:.3f}" for a in ABL) + " |")
            L.append(f"| {dict(DATASETS)[tag]} | AT | " +
                     " | ".join(f"{abl[a]['AT']:.2f}" for a in ABL) + " |")
    L += ["", "## Fig. 6 - Parameters and time complexity (Yelp)", "", "![Fig 6](figures/fig6.png)", ""]
    lam = {int(r[0]): r for r in yelp_extra.get("lam", [])}
    if lam:
        ls = [l for l in P_FIG6_LAM if l in lam]
        L += ["| (a) lambda | " + " | ".join(map(str, ls)) + " |", "|---" * (len(ls) + 1) + "|",
              "| complexity | " + " | ".join(f"{lam[l][2]:.2f}" for l in ls) + " |",
              "| strength | " + " | ".join(f"{lam[l][1]:.3f}" for l in ls) + " |", ""]
    tk = yelp_extra.get("topk", [])
    if tk:
        L += ["| Top-K | " + " | ".join(str(r[0]) for r in tk) + " |", "|---" * (len(tk) + 1) + "|",
              "| (b) AT | " + " | ".join(f"{r[1]:.2f}" for r in tk) + " |",
              "| (c) SR@20 | " + " | ".join(f"{r[2]:.3f}" for r in tk) + " |", ""]
    perf = all_perf["yelp"]
    order = [m for m in ("EAR", "UNICORN", "FPAN", "KBRD", "CECR") if m in perf]
    if order:
        L += ["| (d) Model | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|",
              "| seconds / conversation | " + " | ".join(
                  f"{perf[m]['sec_per_conv']:.3f}" for m in order) + " |", ""]
    return L


def main():
    os.makedirs(OUT, exist_ok=True)
    all_perf, all_abl, all_expl, extras = {}, {}, {}, {}
    for tag, _ in DATASETS:
        all_perf[tag], all_abl[tag], all_expl[tag], extras[tag] = load(tag)
    table4_paper(all_expl)
    fig3_paper(all_perf)
    fig4_paper(all_abl)
    fig6_paper(extras["yelp"], all_perf["yelp"])
    keep = {"table4.png", "fig3.png", "fig4.png", "fig6.png"}
    for f in os.listdir(OUT):                                  # only the paper's four artefacts
        path = f"{OUT}/{f}"
        if f not in keep:
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    with open(f"{ROOT}/REIMPLEMENTATION_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report(all_perf, all_abl, all_expl, extras["yelp"])) + "\n")
    for tag, _ in DATASETS:
        if os.path.isdir(f"results/{tag}"):
            shutil.copytree(f"results/{tag}", f"{ROOT}/raw_results/{tag}", dirs_exist_ok=True)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    for lg in glob.glob("logs/*.txt") + glob.glob("logs_train_*.txt"):
        shutil.copy(lg, f"{ROOT}/logs/")
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
