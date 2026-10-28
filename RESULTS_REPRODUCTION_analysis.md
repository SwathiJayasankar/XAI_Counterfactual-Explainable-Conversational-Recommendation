## What was re-implemented

The paper ("Counterfactual Explainable Conversational Recommendation", Yu et al.,
IEEE TKDE 36(6), 2024) reports five research questions. Every artefact was
re-run on this machine:

| Paper artefact | What it measures | Models we ran | Datasets |
|---|---|---|---|
| Table II, Fig. 3 | SR@5/10/20, AT, SR by round | CECR, EAR, FPAN, UNICORN, KBRD | Yelp, Douban-Book, MovieLens |
| Fig. 4 | ablation SR@20 + AT | CECR minus GNNs / PreRi / CouRe / RecPe | MovieLens, Yelp |
| Table IV | Precision, Recall, PN, PS of explanations | CECR, CPR, CAEC, SAUR, Random | Yelp, MovieLens |
| Fig. 5 | case study | CECR | MovieLens (`raw_results/movielens/case_study.txt`) |
| Fig. 6 (a)-(c) | λ, Top-K vs AT, Top-K vs SR@20 | CECR | Yelp |
| Fig. 6 (d) | response time | CECR, EAR, FPAN, UNICORN, KBRD | Yelp |

Code: `cecr_torch.py`/`train_cecr.py`/`eval_cecr.py` (CECR),
`baselines.py`/`run_baselines.py` (all 7 comparison models), `crs_bench.py` (the
shared conversation harness and metrics), `make_paper_figs.py` (this report).

## How our results compare with the paper

**1. Absolute numbers are lower than published for every model, baselines
included.** On MovieLens the SR@20 values are close to the paper (EAR 0.940 vs
0.943, KBRD 0.960 vs 0.941, CECR 0.952 vs 0.982). SR@5 and AT are worse
across the board: our conversations need roughly 1–2 more turns. On Yelp
everything is lower still (CECR SR@20 0.660 vs 0.961). Our Yelp copy has 245
category/city attributes and ~14k businesses, so a single confirmed attribute
leaves a much larger candidate pool than MovieLens's 18 genres. The reduced
CPU training budget adds to that. Because the gap appears for the baselines
too, it comes mostly from the setting (data copy, budget, simulator), not
from one model being broken.

**2. CECR's headline advantage over the baselines does not reproduce.** The
paper reports CECR beating the strongest baseline on SR@10 by 5.75% (Yelp),
6.10% (Douban) and 6.87% (MovieLens). On identical conversations:
* **MovieLens**: CECR (SR@10 0.830) beats EAR (0.806) but trails FPAN (0.846)
  and KBRD (0.878). By AT, KBRD (5.14) and FPAN (5.23) are level with CECR
  (5.26).
* **Yelp** (so far): CECR (SR@10 0.502) trails EAR (0.566).
  FPAN/UNICORN/KBRD are still running.

This is a result of the reproduction, not something we tuned away. CECR uses
the paper's own fixed settings (λ = 100, Eq. 20's dot-product ask/recommend
rule). One baseline knob, KBRD's recommend threshold, was tuned on validation
conversations, as the paper permits ("we tune the baselines' parameters in the
same range as ours").

**3. Ablation (Fig. 4).** The paper's qualitative point holds: the full model
is the best or tied-best variant on SR@20 and has the lowest AT on both
datasets. Removing the GNN aggregation hurts most in our runs (MovieLens SR@20
0.952 → 0.554; Yelp 0.660 → 0.394). Removing preference refinement (PreRi)
is clearly harmful (MovieLens 0.882, AT 5.26 → 7.35). The paper's biggest
effects, however, are from PreRi and CouRe (0.599 / 0.613 on MovieLens).
Removing counterfactual augmentation (CouRe) changes almost nothing in our
runs (MovieLens 0.954 vs 0.952; Yelp 0.656 vs 0.660), and RecPe (random
ask/recommend) mainly costs turns. So the counterfactual-sample component,
the paper's main novelty for recommendation accuracy, shows no measurable
benefit here.

**4. Explainability (Table IV).** As in the paper, Random is far below every
real explainer, and CECR's explanations have the best **Recall** on MovieLens
(73.1% vs 53.5–61.5% for CPR/CAEC/SAUR). Precision is close between methods
(51–56%; CECR 53.3%, SAUR 56.3%). On PN/PS (necessity/sufficiency) CPR and CAEC
score higher than CECR (CPR PN 100% vs CECR 86.0%). That is partly
structural: their explanations are the confirmed attributes, which the
FM-style scorers depend on directly. The paper's claim that CECR is best on
all four metrics therefore holds only for Recall in our runs. Absolute
Precision/Recall differ from the paper because the paper does not say where
the "user's liked attributes" come from (see Deviation 4).

**5. λ and Top-K (Fig. 6 a-c), response time (Fig. 6 d).** Increasing λ raises
both explanation strength and complexity, then saturates, and increasing K
lowers AT and raises SR@20 monotonically. Both match the paper. For response
time, all models answer in milliseconds per turn on our CPU. The paper's
ordering (CECR fastest) is checked in the Fig. 6(d) table once all Yelp
baselines finish.

## Verdict

* **Reproduced:** the evaluation protocol and every artefact. Also the
  qualitative trends: SR rises fast in the first ~5 rounds then flattens;
  PreRi and GNNs matter; λ and Top-K behave as described; CECR's explanations
  beat Random by a wide margin and have the best recall.
* **Not reproduced:** CECR's consistent lead over EAR/FPAN/UNICORN/KBRD, the
  large CouRe effect in the ablation, and CECR's lead on all four
  explainability metrics.
* **Caveats:** reduced CPU budget, a smaller Yelp copy, and re-implemented (not
  original) baselines. A different result here does not prove the paper
  wrong, but its gains are not robust to an independent, apples-to-apples
  re-run.

## Run status

* Rows marked *running* are being computed now (Douban-Book: CECR training
  then all baselines; Yelp: FPAN, UNICORN, KBRD and Table IV; Yelp CECR Table IV
  and λ). Re-running `python make_paper_figs.py` refreshes this folder.
* The **MovieLens UNICORN row (SR@5 0.098) is the first, under-trained run**:
  1,500 DQN episodes at lr 1e-4 only reached ~0.77 success on its own training
  conversations, so it asked far too long before recommending. It is being
  re-run with lr 1e-3 (inside the paper's tuning range) and 4,000 episodes.
  Treat that row as provisional until it is replaced.
