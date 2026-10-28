# Explanation of the reimplementation results

Paper: *Counterfactual Explainable Conversational Recommendation* (CECR), Yu et al.,
IEEE TKDE 36(6), 2024. The paper's figures are in `papper output/`; ours, drawn in
the same format from our own runs, are in `reimplementation/figures/`. All numbers
are in `REIMPLEMENTATION_RESULTS.md`.

| Paper (`papper output/`) | Ours (`figures/`) | What it answers |
|---|---|---|
| `table output.png` | `table4.png` | Are CECR's explanations better than other explainable systems? |
| `fig 3.png` | `fig3.png` | How fast does each system find the item, round by round? |
| `fig 4.png` | `fig4.png` | Which part of CECR matters (ablation)? |
| `fig 6.png` | `fig6.png` | Effect of λ and Top-K, and speed of each model (Yelp) |

## How every number was produced

* **Same test for every model.** CECR and the four baselines (EAR, FPAN, UNICORN,
  KBRD) talk to the same simulated users about the same 500 held-out
  (user, liked item) pairs per dataset. Every conversation opens with the same
  attribute for every model. Each model gets at most 20 rounds and Top-K = 10.
  The simulated user accepts a recommendation if the target item is in the list,
  and answers "yes" to an attribute question if the target item has that
  attribute.
* **Paper's settings**: 70/20/10 split, embedding size 128, Adam, lr 1e-3,
  Ω = 1e-5, λ = 100, Top-K = 10.
* **Our metrics**:
  * **SR@t** (success rate): share of conversations where the target item was
    recommended within t rounds.
  * **AT** (average turns): failed conversations count as 20.
  * **Pr / Re**: how many of the explanation's attributes the user actually likes.
  * **PN** (necessity): remove the explanation's attributes; does the item drop
    out of the Top-10?
  * **PS** (sufficiency): keep only the explanation's attributes; does the item
    stay in the Top-10?

## Table IV — explainability

**What it shows.** Each of the five explainers (CPR, CAEC, SAUR, Random, CECR)
explains the same 1,000 recommended items (100 conversation states × Top-10).

**What reproduces.**
* **Random is far below every real explainer**, as in the paper (Yelp PN 5.8%,
  PS 6.6%).
* **CECR has the best Recall on both datasets**, by a wide margin (Yelp 81.05%,
  MovieLens 73.06%; next best 54.84% and 61.51%). The paper also has CECR best on
  Recall (+12.66% / +5.59%).
* **CECR's PN and PS are high**: Yelp PN 94.09% vs the paper's 94.11%, PS 94.49%
  vs 96.39%.

**What does not.** The paper has CECR best on all four metrics. In ours, CECR is
best only on Recall. SAUR has the best Precision, and CPR / CAEC have the best
PN / PS. The **Improv. %** row is negative wherever a baseline wins.

**Why CPR has PN = 100% (and CAEC PS = 100% on Yelp).** These are real outputs,
but they are guaranteed by how those two models work, not a sign of good
explanations:
* **CPR**: its explanation is the attributes the user already said "yes" to.
  Those same attributes are what put the item in the Top-10 in CPR's scorer, and
  every candidate item must contain them. Removing them therefore always drops
  the item out of the list, so PN is almost always 100%.
* **CAEC**: its explanation is the categories that its own scorer weighs most.
  Keeping only those categories therefore always keeps the item in the list,
  so PS = 100%.
* **Cause**: each model is tested with its own scorer.

The paper's CPR value (92.35%) shows its test could fail, but the paper does not
say how it scored this.

**Why absolute Precision/Recall differ from the paper.** The paper defines
"attributes the user likes" without saying where that comes from. We use the
user's own held-out liked items. An attribute counts as liked when it appears in
them more often than its base rate. A different source gives different absolute
values.

## Fig. 3 — success rate per round

**What it shows.** SR at rounds 1–10 for all five models, on Yelp, MovieLens and
Douban-Book. The layout, line styles and colours are the same as the paper's.

**What reproduces.**
* **Shape**: SR climbs steeply over the first ~5 rounds and then flattens, as in
  the paper.
* **MovieLens is the closest dataset to the paper**: CECR, FPAN, EAR and KBRD
  overlap and reach 0.80–0.88 by round 10. The paper has 0.85–0.95.

**What does not.**
1. **CECR is not the top line.**
   * MovieLens: it ties the best early (round 4: 0.644, the highest) but ends
     below KBRD and FPAN at round 10.
   * Yelp: EAR and KBRD are above it.
   * Douban-Book: EAR and KBRD are above it.
2. **Absolute levels are lower than the paper on Yelp and Douban-Book.** CECR
   reaches 0.50 at round 10 on both, vs the paper's ~0.90. The reasons are
   below.
3. **UNICORN (and FPAN on Yelp/Douban) start at 0 and rise late.** Their trained
   policies keep asking questions for 4–6 rounds before recommending anything.
   For UNICORN this persisted even after we doubled its training (4,000 episodes
   on MovieLens gave 0.098 / 0.564 / 0.808, the same as the first run). This
   points to our reimplementation of its reinforcement-learning setup, not to
   the data.

**Why the levels are lower.**
* **Yelp is a different copy.** The paper's Yelp (27,675 users / 70,311 items)
  is not available, so we used a smaller version with 4,613 users, 13,915
  businesses and 245 category/city attributes. One confirmed attribute leaves
  far more candidates than MovieLens's 18 genres, so every model needs more
  rounds.
* **Reduced CPU training budget.** CECR was trained for 25/25 epochs on
  MovieLens, 10/20 on Yelp and 5/15 on Douban-Book. Baselines got 5–20 epochs.
  The paper trained to convergence on a GPU. Douban-Book (9,432 users, 21,676
  books) had the smallest budget and shows the biggest gap.
* **Baselines are our own reimplementations**, not the authors' code (the
  original repositories target other datasets and library versions).

## Fig. 4 — ablation

**What it shows.** Each bar is CECR with one part removed:
* **GNNs**: no graph aggregation.
* **PreRi**: no preference refinement.
* **ConRe**: no counterfactual training samples.
* **RecPe**: random ask/recommend decision instead of the learned rule.

**CCR** is the full model. The left panels show SR@20 and the right panels AT,
on MovieLens and Yelp.

**What reproduces.**
* **The full model (CCR) has the lowest AT on both datasets** (5.26 MovieLens,
  10.99 Yelp), as in the paper.
* **CCR has the best or tied-best SR@20 on Yelp** (0.660).
* **Removing the GNN or preference refinement clearly hurts** (MovieLens SR@20
  0.952 → 0.554 and 0.882).

**What does not.**
* **Which part matters most differs from the paper.** In the paper the biggest
  drops come from PreRi and ConRe (0.599 / 0.613 on MovieLens). In ours the
  biggest drop is GNNs.
* **Removing ConRe changes almost nothing in ours** (MovieLens 0.954 vs 0.952;
  Yelp 0.656 vs 0.660).
* **So the counterfactual training samples, the paper's main idea for accuracy,
  show no measurable benefit in our runs.** Their benefit is in the
  explanations (Table IV Recall), not in SR.

## Fig. 6 — parameters and time (Yelp)

**(a) λ.** λ weights explanation strength against complexity in CECR's
counterfactual search. Both strength and complexity (number of attributes
changed) rise as λ grows and then level off by λ = 50–100. That is the paper's
point, so it reproduces. The order of the two lines differs:
* **Ours**: strength is above complexity. Complexity saturates at about 2.6
  attributes, because a Yelp business has only a few attributes to change.
* **Paper**: complexity is the higher line.

The y-axis is log(1 + value), so λ = 0 plots at 0.

**(b)(c) Top-K.** Recommending more items per round lowers AT (17.1 → 5.9) and
raises SR@20 (0.25 → 0.88) as K goes 1 → 50. This is the same monotone trend as
the paper, which also has AT falling from ~9 to ~2 and SR@20 rising from 0.13
to ~1.0. Our absolute values are lower for the same reasons as Fig. 3.

**(d) Response time.** Seconds per conversation:

| EAR | UNICORN | FPAN | KBRD | CECR |
|---|---|---|---|---|
| 0.21 | 0.10 | 0.11 | 0.08 | 0.47 |

In the paper, CECR is the fastest (0.74 s vs 0.88–1.44 s). In ours it is the
slowest. Two caveats:
* **Timings were measured under shared load.** Up to 8 runs shared the CPU while
  these were measured, so the absolute values are inflated and only roughly
  comparable.
* **Ours are wall-clock times on a CPU, not the paper's hardware.**

CECR's per-round work (graph-aggregated preference, gated refinement and
ranking network) is heavier than the baselines' dot-product scoring, so CECR
being slower on a CPU is expected.

## Bottom line

* **Reproduced.**
  * The full evaluation protocol, and every table and figure in the paper's
    format with no gaps.
  * The qualitative trends: SR rises fast and then flattens; the full CECR has
    the lowest AT in the ablation; GNNs and preference refinement matter; λ and
    Top-K behave as the paper describes.
  * CECR's explanations beat Random by far and have the best Recall.
* **Not reproduced.**
  * CECR beating all four baselines.
  * The large ConRe effect in the ablation.
  * CECR being best on all four explainability metrics.
  * CECR being the fastest.
* **Main reasons.**
  * A different, smaller Yelp copy.
  * A reduced CPU-only training budget, smallest on Douban-Book.
  * Reimplemented, not original, baselines.
  * Two metric details the paper leaves unspecified: where "liked attributes"
    come from, and how PN/PS is scored for the baselines.

Closer numbers would need the paper's own Yelp data and full GPU training. They
would not come from changing any of the values above.
