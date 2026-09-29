"""
Reproduces the paper's OWN figures and tables, using our replicated results.

Every panel here mirrors a specific artefact in the published PDF, with the same
chart type, title, axis labels, tick ranges and value annotations, so our output
can be laid directly beside the paper's:

  Fig. 3     "SR on MovieLens"          line plot, x "Conversation Rounds",
                                        y "Success Rate"
  Fig. 4     "SR@20 on MovieLens"       bar per ablation, value printed on each,
                                        y "Success Rate@20"
             "AT on MovieLens"          line+markers, y "Average Turn"
                                        ablation order: GNNs, PreRi, CouRe, RecPe, CCR
  Fig. 6(a)  "Impacts of lambda ..."    x "Lambda" (0,25,50,75,100),
                                        y "Log Value", lines: complexity, strength
  Table II   recommendation performance  SR@5 / SR@10 / SR@20 / AT
  Table IV   explainability              Pr% / Re% / PN% / PS%

Paper reference values (MovieLens column), transcribed from the published PDF:
  Table II : EAR .844/.871/.943/4.31   FPAN .851/.888/.935/4.11
             UNICORN .843/.865/.937/4.17  KBRD .841/.869/.941/4.21
             CECR .891/.949/.982/3.77
  Fig. 4   : SR@20  GNNs .801  PreRi .599  CouRe .613  RecPe .701  CCR .982
  Table IV : CPR 29.31/55.13/92.35/94.25   CAEC 28.14/46.67/85.55/89.97
             SAUR 26.48/56.74/91.51/93.14  Random 0.33/0.51/2.22/5.14
             CECR 35.14/59.91/96.42/97.12
The paper's per-ablation Average Turn values appear only inside Fig. 4's plot and
are never tabulated, so they are not transcribed or invented here.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out_full"

# validated categorical palette (dataviz validator: all checks pass, light mode)
C_OURS, C_PAPER, C_3, C_RED, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7"
BAR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

ABL_ORDER = ["- GNNs", "- PreRi", "- CouRe", "- RecPe", "CECR (CCR)"]
ABL_LABEL = {"- GNNs": "GNNs", "- PreRi": "PreRi", "- CouRe": "CouRe",
             "- RecPe": "RecPe", "CECR (CCR)": "CCR"}

# --- the paper's published Table II, per dataset -----------------------------
PAPER_BY_DATASET = {
    "movielens": {
        "title": "MovieLens",
        "CECR": {"SR@5": .891, "SR@10": .949, "SR@20": .982, "AT": 3.77},
        "base": {
            "EAR":     {"SR@5": .844, "SR@10": .871, "SR@20": .943, "AT": 4.31},
            "FPAN":    {"SR@5": .851, "SR@10": .888, "SR@20": .935, "AT": 4.11},
            "UNICORN": {"SR@5": .843, "SR@10": .865, "SR@20": .937, "AT": 4.17},
            "KBRD":    {"SR@5": .841, "SR@10": .869, "SR@20": .941, "AT": 4.21},
        },
    },
    "douban": {
        "title": "Douban-Book",
        "CECR": {"SR@5": .866, "SR@10": .922, "SR@20": .973, "AT": 4.22},
        "base": {
            "EAR":     {"SR@5": .812, "SR@10": .833, "SR@20": .866, "AT": 4.82},
            "FPAN":    {"SR@5": .833, "SR@10": .867, "SR@20": .918, "AT": 4.51},
            "UNICORN": {"SR@5": .836, "SR@10": .869, "SR@20": .914, "AT": 4.89},
            "KBRD":    {"SR@5": .821, "SR@10": .848, "SR@20": .915, "AT": 4.66},
        },
    },
}
DATASET = "movielens"
PAPER_T2 = PAPER_BY_DATASET["movielens"]["CECR"]
PAPER_BASE = PAPER_BY_DATASET["movielens"]["base"]


def set_dataset(name):
    """Point the paper-reference constants at the right Table II column."""
    global DATASET, PAPER_T2, PAPER_BASE
    DATASET = name
    PAPER_T2 = PAPER_BY_DATASET[name]["CECR"]
    PAPER_BASE = PAPER_BY_DATASET[name]["base"]
PAPER_ABL20 = {"- GNNs": .801, "- PreRi": .599, "- CouRe": .613,
               "- RecPe": .701, "CECR (CCR)": .982}
PAPER_T4 = {
    "CPR":    dict(Pr=29.31, Re=55.13, PN=92.35, PS=94.25),
    "CAEC":   dict(Pr=28.14, Re=46.67, PN=85.55, PS=89.97),
    "SAUR":   dict(Pr=26.48, Re=56.74, PN=91.51, PS=93.14),
    "Random": dict(Pr=0.33,  Re=0.51,  PN=2.22,  PS=5.14),
    "CECR":   dict(Pr=35.14, Re=59.91, PN=96.42, PS=97.12),
}


def style(ax, ylabel=None, title=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", pad=9)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ===========================================================================
# Fig. 3  --  "SR on MovieLens"
# ===========================================================================
def fig3_comparison(results):
    """"SR on MovieLens" with ONLY our full model (CCR) and the paper's CECR.

    The paper publishes a per-round curve in its Fig. 3 but tabulates only
    SR@5 / SR@10 / SR@20 (Table II), so the paper's series is drawn as those
    three published operating points, joined by a dashed line purely as a
    visual guide -- the dashes are not published per-round data.
    """
    ours = results.get("CECR (CCR)")
    if ours is None:
        return
    curve = ours["curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.patch.set_facecolor(SURFACE)

    ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=5,
            linewidth=2, color=C_OURS, label="CCR (ours)", zorder=3)

    pr, pv = [5, 10, 20], [PAPER_T2["SR@5"], PAPER_T2["SR@10"], PAPER_T2["SR@20"]]
    ax.plot(pr, pv, linestyle="--", linewidth=1.6, color=C_PAPER, alpha=0.75, zorder=3)
    ax.plot(pr, pv, marker="*", markersize=15, linestyle="none", color=C_PAPER,
            label="CECR (paper)", zorder=4)
    for x, y in zip(pr, pv):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=INK2)
    for x in pr:
        ax.annotate(f"{curve[x-1]:.3f}", (x, curve[x - 1]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=INK2)

    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_ylim(0, 1.08)
    style(ax, xlabel="Conversation Rounds", ylabel="Success Rate",
          title=f"SR on {PAPER_BY_DATASET[DATASET]['title']}")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_comparison.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig3(results):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.patch.set_facecolor(SURFACE)
    markers = ["o", "s", "^", "D", "v"]
    for (name, res), c, mk in zip(results.items(), BAR_COLORS, markers):
        curve = res["curve"]
        ax.plot(range(1, len(curve) + 1), curve, marker=mk, markersize=5,
                linewidth=2, color=c, label=ABL_LABEL.get(name, name), zorder=3)
    for r, key in ((5, "SR@5"), (10, "SR@10"), (20, "SR@20")):
        ax.plot(r, PAPER_T2[key], marker="*", markersize=15, color=INK,
                linestyle="none", zorder=5,
                label="CECR (paper)" if r == 5 else None)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_ylim(0, 1.05)
    style(ax, xlabel="Conversation Rounds", ylabel="Success Rate",
          title=f"SR on {PAPER_BY_DATASET[DATASET]['title']}")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_sr_on_movielens.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ===========================================================================
# Fig. 4  --  paper layout: "SR@20 on MovieLens" bars + "AT on MovieLens" line
# ===========================================================================
def fig4(rows):
    have = {r["Model"]: r for r in rows}
    present = [k for k in ABL_ORDER if k in have]
    if not present:
        return
    labels = [ABL_LABEL[k] for k in present]
    sr20 = [float(have[k]["SR@20"]) for k in present]
    at = [float(have[k]["AT"]) for k in present]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    b = ax.bar(labels, sr20, 0.6, color=BAR_COLORS[:len(labels)], zorder=3)
    ax.bar_label(b, fmt="%.3f", fontsize=9, color=INK2, padding=2)
    ax.set_ylim(0, 1.1)
    style(ax, ylabel="Success Rate@20", title=f"SR@20 on {PAPER_BY_DATASET[DATASET]['title']}")

    ax = axes[1]
    ax.plot(labels, at, marker="o", markersize=8, linewidth=2, color=C_RED, zorder=3)
    for x, y in zip(labels, at):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, max(at) * 1.3)
    style(ax, ylabel="Average Turn", title=f"AT on {PAPER_BY_DATASET[DATASET]['title']}")

    fig.suptitle(f"Ablation Study for our CECR on {PAPER_BY_DATASET[DATASET]['title']}",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_ablation_movielens.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig4_vs_paper(rows):
    """Same ablation, ours overlaid on the paper's published Fig. 4 values."""
    have = {r["Model"]: r for r in rows}
    present = [k for k in ABL_ORDER if k in have]
    if not present:
        return
    labels = [ABL_LABEL[k] for k in present]
    ours = [float(have[k]["SR@20"]) for k in present]
    pap = [PAPER_ABL20[k] for k in present]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    x = range(len(labels)); w = 0.38
    b1 = ax.bar([i - w / 2 - .01 for i in x], ours, w, label="Ours", color=C_OURS, zorder=3)
    b2 = ax.bar([i + w / 2 + .01 for i in x], pap, w, label="Paper (Fig. 4)",
                color=C_PAPER, zorder=3)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, color=INK2, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    style(ax, ylabel="Success Rate@20", title=f"Ablation on {PAPER_BY_DATASET[DATASET]['title']}: ours vs. paper")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cmp_ablation.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ===========================================================================
# Fig. 6(a)  --  "Impacts of lambda for counterfactual explanation"
# ===========================================================================
def fig6a(rows):
    if not rows:
        return
    lam = [float(r["lambda"]) for r in rows]
    es = [float(r["ES"]) for r in rows]
    ec = [float(r["EC"]) for r in rows]
    les = [math.log(1 + max(v, 0)) for v in es]
    lec = [math.log(1 + max(v, 0)) for v in ec]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.plot(lam, lec, marker="o", markersize=7, linewidth=2, color=C_RED,
            label="complexity", zorder=3)
    ax.plot(lam, les, marker="s", markersize=7, linewidth=2, color=C_OURS,
            label="strength", zorder=3)
    ax.set_xticks([0, 25, 50, 75, 100])
    style(ax, xlabel="Lambda", ylabel="Log Value",
          title="Impacts of λ for counterfactual explanation")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6a_lambda.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    fig.patch.set_facecolor(SURFACE)
    axes[0].plot(lam, es, marker="s", markersize=7, linewidth=2, color=C_OURS, zorder=3)
    style(axes[0], xlabel="Lambda", ylabel="Explanation Strength (ES)",
          title="ES vs λ (raw values)")
    axes[1].plot(lam, ec, marker="o", markersize=7, linewidth=2, color=C_RED, zorder=3)
    style(axes[1], xlabel="Lambda", ylabel="Explanation Complexity (EC)",
          title="EC vs λ (raw values)")
    for ax, ys in zip(axes, (es, ec)):
        for xx, yy in zip(lam, ys):
            ax.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=INK2)
        lo = min(ys) * .8 if min(ys) > 0 else 0.0
        hi = max(ys) * 1.25
        ax.set_ylim(lo, hi if hi > lo else lo + 1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6a_lambda_raw.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ===========================================================================
# Rendered tables in the paper's own layout
# ===========================================================================
def _table_fig(col_labels, row_labels, cell_text, title, bold_rows, path,
               width=9.0, note=None):
    h = 0.42 * (len(row_labels) + 1) + (0.45 if note else 0) + 0.5
    fig, ax = plt.subplots(figsize=(width, h))
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_position([0.02, 0.02, 0.96, 0.86])
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels,
                   loc="center", cellLoc="center", rowLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.45)
    ncol = len(col_labels)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(color=INK, fontweight="bold")
        elif c == -1:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor("#f7f6f4")
        else:
            cell.set_text_props(color=INK2)
        if r - 1 in bold_rows and r > 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor("#eaf2fc" if c != -1 else "#dfeafa")
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=14)
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8, color=INK2)
    fig.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def table2(rows):
    """Paper Table II layout: models x (SR@5, SR@10, SR@20, AT), MovieLens."""
    ours = next((r for r in rows if r["Model"].startswith("CECR")), None)
    if not ours:
        return
    order = ["EAR", "FPAN", "UNICORN", "KBRD"]
    row_labels, cells = [], []
    for m in order:
        row_labels.append(m)
        v = PAPER_BASE[m]
        cells.append([f"{v['SR@5']:.3f}", f"{v['SR@10']:.3f}",
                      f"{v['SR@20']:.3f}", f"{v['AT']:.2f}"])
    row_labels.append("CECR (paper)")
    cells.append([f"{PAPER_T2['SR@5']:.3f}", f"{PAPER_T2['SR@10']:.3f}",
                  f"{PAPER_T2['SR@20']:.3f}", f"{PAPER_T2['AT']:.2f}"])
    row_labels.append("CECR (ours)")
    cells.append([f"{float(ours['SR@5']):.3f}", f"{float(ours['SR@10']):.3f}",
                  f"{float(ours['SR@20']):.3f}", f"{float(ours['AT']):.2f}"])
    # how close our replication lands, as % of the paper's reported value
    rep = []
    for k in ("SR@5", "SR@10", "SR@20"):
        rep.append(f"{100*float(ours[k])/PAPER_T2[k]:.1f}%")
    rep.append(f"{100*PAPER_T2['AT']/max(float(ours['AT']),1e-9):.1f}%")
    row_labels.append("ours / paper")
    cells.append(rep)

    _table_fig(["SR@5", "SR@10", "SR@20", "AT"], row_labels, cells,
               f"Table II — Recommendation performance on {PAPER_BY_DATASET[DATASET]['title']}",
               bold_rows={len(row_labels) - 2, len(row_labels) - 3},
               path=f"{OUT}/table2_performance.png", width=8.2,
               note="Baseline and CECR(paper) rows transcribed from the published Table II; "
                    "CECR(ours) is this replication. AT ratio is paper/ours (lower AT is better).")


def table4(rows):
    """Paper Table IV layout: models x (Pr%, Re%, PN%, PS%), MovieLens."""
    have = {r["Model"]: r for r in rows}
    if "CECR" not in have:
        return
    row_labels, cells = [], []
    for m in ["CPR", "CAEC", "SAUR", "Random"]:
        v = PAPER_T4[m]
        row_labels.append(f"{m} (paper)")
        cells.append([f"{v['Pr']:.2f}", f"{v['Re']:.2f}", f"{v['PN']:.2f}", f"{v['PS']:.2f}"])
    v = PAPER_T4["CECR"]
    row_labels.append("CECR (paper)")
    cells.append([f"{v['Pr']:.2f}", f"{v['Re']:.2f}", f"{v['PN']:.2f}", f"{v['PS']:.2f}"])
    for who, lbl in (("Random", "Random (ours)"), ("CECR", "CECR (ours)")):
        r = have[who]
        row_labels.append(lbl)
        cells.append([f"{100*float(r['Precision']):.2f}", f"{100*float(r['Recall']):.2f}",
                      f"{100*float(r['PN']):.2f}", f"{100*float(r['PS']):.2f}"])

    _table_fig(["Pr%", "Re%", "PN%", "PS%"], row_labels, cells,
               f"Table IV — Explainability on {PAPER_BY_DATASET[DATASET]['title']}",
               bold_rows={len(row_labels) - 1, 4},
               path=f"{OUT}/table4_explainability.png", width=8.2,
               note="Rows marked (paper) are transcribed from the published Table IV; "
                    "(ours) rows are this replication, converted to the paper's % scale.")


# ===========================================================================
def cmp_performance(rows):
    ours = next((r for r in rows if r["Model"].startswith("CECR")), None)
    if not ours:
        return
    metrics = ["SR@5", "SR@10", "SR@20"]
    o = [float(ours[m]) for m in metrics]
    p = [PAPER_T2[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    fig.patch.set_facecolor(SURFACE)
    x = range(len(metrics)); w = 0.38
    b1 = ax.bar([i - w/2 - .01 for i in x], o, w, label="Our replication",
                color=C_OURS, zorder=3)
    b2 = ax.bar([i + w/2 + .01 for i in x], p, w, label="Paper (published)",
                color=C_PAPER, zorder=3)
    ax.bar_label(b1, fmt="%.3f", fontsize=9, color=INK2, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, color=INK2, padding=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    style(ax, ylabel="Success Rate",
          title=f"Recommendation performance on {PAPER_BY_DATASET[DATASET]['title']} (Table II)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cmp_performance.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def cmp_avg_turns(rows):
    ours = next((r for r in rows if r["Model"].startswith("CECR")), None)
    if not ours:
        return
    names = ["Ours", "CECR\n(paper)", "EAR", "FPAN", "UNICORN", "KBRD"]
    vals = [float(ours["AT"]), PAPER_T2["AT"]] + \
           [PAPER_BASE[k]["AT"] for k in ("EAR", "FPAN", "UNICORN", "KBRD")]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    fig.patch.set_facecolor(SURFACE)
    b = ax.bar(names, vals, 0.62, color=[C_OURS, C_PAPER] + [C_3]*4, zorder=3)
    ax.bar_label(b, fmt="%.2f", fontsize=9, color=INK2, padding=2)
    ax.set_ylim(0, max(vals)*1.25)
    style(ax, ylabel="Average Turn (lower is better)",
          title="Average conversation length vs. the paper's models")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cmp_avg_turns.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def cmp_explainability(rows):
    have = {r["Model"]: r for r in rows}
    if "CECR" not in have:
        return
    metrics = [("Precision", "Pr"), ("Recall", "Re"), ("PN", "PN"), ("PS", "PS")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, who in zip(axes, ["CECR", "Random"]):
        o = [100*float(have[who][m]) for m, _ in metrics]
        p = [PAPER_T4[who][k] for _, k in metrics]
        x = range(len(metrics)); w = 0.38
        b1 = ax.bar([i - w/2 - .01 for i in x], o, w, label="Ours", color=C_OURS, zorder=3)
        b2 = ax.bar([i + w/2 + .01 for i in x], p, w, label="Paper", color=C_PAPER, zorder=3)
        ax.bar_label(b1, fmt="%.1f", fontsize=8, color=INK2, padding=2)
        ax.bar_label(b2, fmt="%.1f", fontsize=8, color=INK2, padding=2)
        ax.set_xticks(list(x)); ax.set_xticklabels([k for _, k in metrics], fontsize=9)
        ax.set_ylim(0, 118)
        style(ax, ylabel="%" if who == "CECR" else None, title=f"{who} explanations")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2, loc="upper left")
    fig.suptitle(f"Explainability on {PAPER_BY_DATASET[DATASET]['title']} (Table IV): ours vs. paper",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cmp_explainability.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movielens", choices=list(PAPER_BY_DATASET))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    set_dataset(a.dataset)
    if a.out:
        OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    t2 = read_csv(f"{OUT}/table2_performance.csv")
    t4 = read_csv(f"{OUT}/table4_explainability.csv")
    t6 = read_csv(f"{OUT}/fig6a_data.csv")

    fig4(t2); fig6a(t6)
    table2(t2)
    cmp_performance(t2); cmp_avg_turns(t2)
    if DATASET == "movielens":
        # the paper only publishes Fig. 4 ablations and Table IV for
        # MovieLens/Yelp, so there is nothing to compare against on Douban-Book
        fig4_vs_paper(t2); table4(t4); cmp_explainability(t4)

    import pickle
    p = f"{OUT}/results.pkl"
    if os.path.exists(p):
        with open(p, "rb") as f:
            res = pickle.load(f)["results"]
        fig3(res)
        fig3_comparison(res)
    print(f"figures + tables written to {OUT}/")


if __name__ == "__main__":
    main()
