# CECR: improvements beyond the paper

Three improvements on top of the reproduction (`RESULTS_REPRODUCTION_analysis.md`),
all run on the shared harness (`crs_bench.py`), so every number is comparable
with the CECR and baseline numbers already reported. CPU only.

| # | Improvement | Code | Logs / results |
|---|---|---|---|
| 3 | Ask-vs-recommend decision | `decision.py`, `run_decision.py` | `logs/decision_*.txt`, `results/<ds>/decision_results.json` |
| 1 | Amortised counterfactual explanations | `cf_amortized.py` | `logs/cf_amortized_*.txt`, `results/<ds>/cf_amortized_lam*.json` |
| 5 | Robust evaluation (seeds, answer noise, CIs, paired tests) | `robust_cecr.py`, `summarize_robust.py`, `run_robust.sh`, `run_baselines.py --robust` | `results/movielens/robust/summary.md` |

---

## Improvement #3: the ask-vs-recommend decision

**Problem found.** The paper's Eq. 20 compares `max e_i . e_overall` with
`max e_a . e_overall`. In our replica it **never asks after the opening
attribute**: 0 % ask decisions on MovieLens, 1 % on Yelp. Item scores are
always larger (median margin 1.32 on MovieLens). CECR therefore behaves as
"recommend 10 items every turn". Rescaling to cosine similarity changes
nothing (still 0 % / 1 %).

**What we tried.**

1. *Learned policy* (EAR-style: imitation of Eq. 20, then REINFORCE, 3000
   episodes, model selection on validation). It converges to "never ask" on
   both datasets, the same collapse we see in the retrained EAR/FPAN policies
   (Improvement #5). Result: no change (Yelp SR@10 0.502 -> 0.510).
2. *Entropy questions with a validated threshold*: ask the
   preference-weighted max-entropy attribute while more than T candidates
   remain, otherwise recommend. T is chosen on validation conversations
   (MovieLens T = 500, Yelp T = 200). Asking CECR's own "most preferred"
   attribute instead is strictly worse (it asks about what is already known).

**Results (500 test conversations, same pairs as Table II):**

| Dataset | Model | SR@5 | SR@10 | SR@20 | AT |
|---|---|---|---|---|---|
| MovieLens | CECR, Eq. 20 (paper) | 0.696 | 0.830 | 0.952 | 5.26 |
| MovieLens | **CECR + entropy questions** | **0.730** | **0.898** | **0.982** | **4.63** |
| MovieLens | KBRD (best baseline) | 0.682 | 0.878 | 0.960 | 5.14 |
| MovieLens | FPAN | 0.692 | 0.846 | 0.958 | 5.23 |
| MovieLens | CECR as published in the paper | 0.891 | 0.949 | 0.982 | 3.77 |
| Yelp | CECR, Eq. 20 (paper) | 0.374 | 0.502 | 0.660 | 10.99 |
| Yelp | **CECR + entropy questions** | 0.410 | 0.708 | 0.952 | 7.95 |
| Yelp | KBRD (best baseline) | **0.466** | **0.710** | **0.960** | **7.39** |
| Yelp | EAR | 0.462 | 0.566 | 0.670 | 9.98 |

* MovieLens: CECR now beats every baseline on all four metrics, and SR@20
  equals the paper's 0.982.
* Yelp: +0.21 SR@10, +0.29 SR@20, 3 fewer turns. This closes most of the gap
  to KBRD, which remains slightly ahead on SR@5 and AT.

---

## Improvement #1: amortised counterfactual explanations

**Problem.** Eqs. 17-18 run a separate gradient-descent search for Delta for
every explained item (60 Adam steps). That makes CECR the slowest model per
turn (Fig. 6(d) in our reproduction).

**Method.** A small MLP (`DeltaNet`) predicts Delta in one forward pass from
`(e_overall, e_item, e_overall*e_item, attribute mask, s_item, s_item - s_K)`.
It is trained on the paper's own Eq. 17 objective over 30,000 (state, Top-K
item) samples from **training-split** conversations, with the CECR model
frozen. Optional: warm-start the paper's optimiser from it for 5 steps.
Evaluated on the same 100 Table IV states as before.

**Finding about the paper.** At the paper's lambda = 100, the ranking term
outweighs the sparsity penalty (at most ~2 per attribute) by about 50x. The
optimum is then "change every attribute of the item", so the "minimal" part
of the minimal counterfactual is effectively switched off (EC ~ number of
attributes the item has). We therefore report lambda = 100 (paper) and
lambda = 3 (genuinely sparse explanations).

| Dataset | lambda | Explainer | Pr | Re | PN | PS | EC | ms / expl |
|---|---|---|---|---|---|---|---|---|
| MovieLens | 100 | paper optimiser | 0.533 | 0.731 | 0.860 | 0.949 | 1.98 | 140 |
| MovieLens | 100 | DeltaNet | 0.530 | 0.730 | 0.857 | 0.954 | 2.00 | **3.9** |
| MovieLens | 3 | paper optimiser | 0.510 | 0.629 | 0.812 | 0.842 | 1.14 | 259 |
| MovieLens | 3 | **DeltaNet** | **0.543** | **0.715** | **0.890** | **0.897** | 1.39 | **5.4** |
| Yelp | 100 | paper optimiser | 0.438 | 0.811 | 0.941 | 0.945 | 3.35 | 194 |
| Yelp | 100 | DeltaNet | 0.463 | 0.771 | 0.924 | 0.915 | 2.98 | **5.8** |
| Yelp | 100 | DeltaNet + 5 steps | 0.452 | 0.794 | 0.935 | 0.926 | 3.17 | 18 |
| Yelp | 3 | paper optimiser | 0.528 | 0.606 | 0.817 | 0.717 | 1.87 | 188 |
| Yelp | 3 | DeltaNet | 0.527 | 0.646 | 0.832 | 0.758 | 2.10 | **3.8** |
| Yelp | 3 | DeltaNet + 5 steps | 0.472 | **0.731** | **0.885** | **0.829** | 2.74 | 16 |

* 34-49x faster per explanation in every setting.
* With sparse explanations (lambda = 3), one network beats the per-item
  optimiser on MovieLens on all four metrics, and on Yelp on Re/PN/PS.
* At lambda = 100 it matches the optimiser on MovieLens and costs 2-4 points
  of Re/PN/PS on Yelp. 5 refinement steps recover most of that.
* Caveat: items whose explanation comes out empty are skipped by the Table IV
  metric (as in the original protocol), so coverage `n` differs slightly
  between explainers. It is reported in the logs.

---

## Improvement #5: robust evaluation (MovieLens)

**Protocol.** 5 seeds (0-4). Each seed retrains CECR and every baseline,
draws its own 500 test conversations, and runs at answer-noise rates 0 / 0.1
/ 0.2 (each attribute answer flipped with that probability). Mean +- 95 % CI
uses a t-distribution over seeds. Paired tests pool the per-conversation
outcomes of all seeds: Wilcoxon on turns, exact McNemar on SR@10. Full tables
are in `results/movielens/robust/summary.md`.

**Noise 0 (truthful user):**

| Model | SR@5 | SR@10 | SR@20 | AT |
|---|---|---|---|---|
| CECR (paper's Eq. 20) | 0.688 +- 0.021 | 0.847 +- 0.024 | 0.948 +- 0.005 | 5.20 +- 0.23 |
| **CECR + entropy questions** | **0.717 +- 0.021** | **0.895 +- 0.015** | **0.976 +- 0.005** | **4.79 +- 0.24** |
| KBRD | 0.673 +- 0.012 | 0.866 +- 0.010 | 0.953 +- 0.008 | 5.27 +- 0.21 |
| FPAN | 0.689 +- 0.011 | 0.850 +- 0.014 | 0.947 +- 0.011 | 5.25 +- 0.24 |
| EAR | 0.643 +- 0.027 | 0.789 +- 0.017 | 0.920 +- 0.015 | 6.13 +- 0.30 |
| UNICORN | 0.087 +- 0.021 | 0.510 +- 0.040 | 0.755 +- 0.039 | 11.79 +- 0.51 |

CECR + entropy questions is significantly better than every baseline on SR@10
(McNemar p = 1.6e-5 vs KBRD, 2.5e-11 vs FPAN) and better than the paper's
Eq. 20 on SR@10 (p = 2.4e-24). It needs fewer turns than KBRD and EAR
(Wilcoxon p <= 1.5e-5); against FPAN the turn difference (-0.46) is not
significant (p = 0.08).

**With noisy answers** the picture changes:

| Model | SR@20, noise 0 | 0.1 | 0.2 |
|---|---|---|---|
| CECR (Eq. 20) | 0.948 | 0.940 | 0.936 |
| CECR + entropy questions | 0.976 | 0.900 | 0.832 |
| KBRD | 0.953 | 0.866 | 0.806 |
| FPAN / EAR | 0.947 / 0.920 | unchanged | unchanged |
| UNICORN | 0.755 | 0.380 | 0.174 |

* Every model that asks (CECR + questions, KBRD, UNICORN) degrades with noise.
  Models that (almost) never ask (CECR with Eq. 20, and EAR/FPAN, whose
  REINFORCE policies also collapse to "always recommend") are barely or not
  affected.
* At noise 0.2, CECR + entropy questions is still level with KBRD (p = 1),
  but below the never-ask models.
* So the paper's (and our) truthful-user simulator flatters question-asking.
  The candidate filter treats every "no" as a hard constraint, so a single
  wrong answer can remove the target for good. Soft candidate filtering is
  the natural next fix.

---

## Side finding relevant to CouRe (counterfactual augmentation)

`train_cecr.py` produces only **0-3 counterfactual samples per epoch**
(`cf_samples` in `logs_train_full.txt`). That is far too few to change the
model, which explains why the CouRe ablation shows ~0 effect in our
reproduction, unlike the paper's 0.982 -> 0.613.

## Reproduce

```bash
python run_decision.py --data data/prepared.pkl      --model data/cecr_full.pt      --tag movielens
python run_decision.py --data data/prepared_yelp.pkl --model data/cecr_yelp_full.pt --tag yelp
python cf_amortized.py --data data/prepared.pkl      --model data/cecr_full.pt      --tag movielens --lam 3   # and --lam 100
python cf_amortized.py --data data/prepared_yelp.pkl --model data/cecr_yelp_full.pt --tag yelp      --lam 3   # and --lam 100
bash run_robust.sh                                   # several hours on CPU
python summarize_robust.py --tag movielens --ref CECR+ask
```
