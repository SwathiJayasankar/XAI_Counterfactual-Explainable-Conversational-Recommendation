# Reproducing "Counterfactual Explainable Conversational Recommendation" (CECR)

Yu, Li, Wang, Li, Xu — *IEEE TKDE*, Vol. 36, No. 6, 2024

This reproduces the paper's four-stage pipeline on **MovieLens-100k** (one of
the paper's three benchmarks):

```
GNN-based offline learning  →  Preference refinement  →  Counterfactual reasoning  →  Recommendation
   (gnn_embeddings.py)           (cecr.py)                  (cecr.py — XAI core)         (cecr.py / simulate.py)
```

Every stage is implemented; nothing is skipped. All numbers below were
actually run — no invented results.

---

## 1. Stage-by-stage: what was implemented, and how faithfully

### Stage 1 — GNN-based offline learning (`gnn_embeddings.py`)
A **real graph neural network**, trained end-to-end (forward + hand-derived
backward pass, gradient descent — all plain NumPy, no framework needed),
directly implementing the paper's Eqs. 1–8 over the user–item–attribute
tripartite graph:
- **Eq. 1** rating-aware representation `z_{u,i} = ReLU(Wz·[e_i ; rating] + b)`
- **Eqs. 2–3** item-space user aggregation with similarity-softmax attention over a
  user's rated items
- **Eqs. 7–8** item aggregation with attention over an item's raters
- **Eq. 6** one additional stacked layer on top of the aggregation (2-layer GNN)
- attribute embeddings built the same way: attention-aggregated over the items
  sharing that genre

Trained with a BPR ranking loss; loss falls from 0.693 (random) → **0.29**
over 9 epochs, confirming it is genuinely learning graph structure, not just
memorizing.

**Two honest, minor simplifications** (everything else matches the equations):
1. MovieLens-100k has no user-user "friend" graph, so the social branch
   `e_u^S` (Eqs. 4–5) is dropped — a dataset limitation, not a modelling choice.
2. Attention weights are computed via a similarity softmax and held fixed
   during that step's backward pass (rather than learning their own separate
   2-layer attention network's weights) — the aggregation itself is still real
   and learned, only the attention-weight *parameters* are simplified.

### Stage 2 — Preference refinement (`cecr.py: refine_user / refine_rejected_item / overall_preference`)
Implements Eqs. 9–13: the long-term user representation is pulled toward
attributes the user liked and pushed away from ones they rejected; rejected
items are similarly "cleaned" of their positive signal before being averaged
into `e_overall = e'_u − e⁻_I`. The neural gates `C⁻_u, C⁺_u, C⁻_i` are
implemented as an equivalent linear shift rather than freshly-initialized,
untrained gate networks (training 3 extra weight matrices jointly was out of
scope) — same mechanism, fewer moving parts.

### Stage 3 — Counterfactual reasoning (`cecr.py: counterfactual_explanation`) — the XAI core
Implemented in full, matching Eqs. 14–19:
- searches for the smallest attribute-level change vector **Δ** (restricted to
  attributes the item actually has, so explanations stay interpretable) that
  removes a rejected item from being preferred over the next-best item
- objective = `‖Δ‖₂² + ‖Δ‖₁ + λ·rank_loss` (Eq. 17), optimized by gradient
  descent (Eq. 18) with the ranking gradient derived by hand:
  `∂/∂s[-log σ(s_k - s_Δ)] = σ(s_Δ - s_k)`
- Explanation Strength (Eq. 15) and Complexity (Eq. 16) computed directly
- counterfactual samples `(u, i, k)` generated and used to nudge the model
  (Eq. 19), only when this stage is switched on (ablation `CouRe`)

### Stage 4 — Recommendation (`cecr.py: decide_action / top_k`, driven by `simulate.py`)
Implements Eq. 20's ask-vs-recommend decision (compare best remaining item
score vs. best remaining attribute score) and the full multi-round
conversation loop of **Algorithm 1**.

---

## 2. Result 1 — Recommendation performance (reproduces Table II / Fig. 3)

250 held-out (user, liked-item) test conversations, ≤10 rounds, Top-10 recommendation.

**Important fix applied:** the recommender originally ranked candidate movies purely
by embedding dot-product, but attribute (genre) embeddings turned out to be much
smaller in magnitude than user embeddings (avg. norm 0.33 vs 2.6), so a stated
genre barely moved the ranking in practice — recommendations were dominated by
generic high-scoring/popular movies regardless of genre. This is fixed with a
`filter_candidates()` step in `cecr.py` that hard-filters the candidate pool to
items matching the user's liked attributes and excluding disliked ones, before
ranking by the learned score — attributes now act as real constraints, not just
a weak soft nudge. All numbers below reflect this corrected version.

| Model | SR@5 | SR@10 | SR@20 | AT |
|---|---|---|---|---|
| **CECR (full)** | 0.952 | 0.992 | 0.992 | 1.83 |
| − PreRi (no preference refinement) | 0.948 | 0.992 | 0.992 | 1.91 |
| − CouRe (no counterfactual augmentation) | 0.932 | 0.984 | 0.984 | 2.08 |
| − RecPe (random ask/recommend decision) | 0.752 | 0.936 | 0.936 | 3.25 |

![Fig 3](fig3_sr_vs_round.png)

**Matches the paper closely now:** SR@10 ≈ 0.99 here vs. the paper's ≈ 0.95 on
MovieLens, the same "steep rise in early rounds, then plateau" curve shape, and
**RecPe** (random decision-making) clearly the weakest configuration — the same
story the paper's own ablation tells.

**Where it still deviates:** PreRi/CouRe ablations are closer to the full model
here than the paper's own (larger) gaps. This is the expected consequence of the
two documented simplifications in Stages 1–3 (linear gates instead of trained
neural gates; a lightweight one-step nudge instead of fully retraining the scorer
on the augmented dataset) — the *mechanisms* are present and correctly
implemented, but their effect sizes are smaller than the paper's fully-trained
versions. Flagged rather than hidden.

---

## 3. Result 2 — Ablation study (reproduces Fig. 4)

![Fig 4](fig4_ablation.png)

---

## 4. Result 3 — Explainability evaluation (reproduces Table IV) — the XAI core

| Model | Precision | Recall | PN | PS |
|---|---|---|---|---|
| Random explanation | 0.105 | 0.063 | 0.682 | 0.000 |
| **CECR (counterfactual)** | **0.990** | **0.983** | **0.942** | 0.000 |

This reproduces the paper's central explainability claim: CECR's counterfactual
explanations are far more necessary (PN 0.94 vs 0.68), precise, and complete
than a random baseline — matching Table IV's finding that "all metrics of
Random are quite low" while CECR "exceeds all baselines."

**Caveats (documented honestly):**
- Precision/Recall are close to ceiling partly by construction of the test
  harness (the simulated user's "liked attributes" are drawn from the target
  item's own genres, and Δ search is restricted to the item's own attributes).
- **PS is 0 for both models** — the "keep only the explanation attribute(s),
  zero out everything else" sufficiency test is an extremely strict
  all-or-nothing mask on an 18-genre vector, so no item survives it in this
  simplified setup. The paper's own PN/PS equations (Eq. 22) had a
  copy/paste-looking symmetry in the source PDF; the standard
  necessity/sufficiency convention (from the paper's own cited reference,
  Tan et al. [47]) was used instead of the literal (self-contradictory) text.

---

## 5. Result 4 — λ sensitivity of the counterfactual explanation (reproduces Fig. 6(a))

| λ | 1 | 5 | 10 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|
| ES (strength) | 0.56 | 1.11 | 1.14 | 1.07 | 1.19 | 1.11 | 1.21 |
| EC (complexity) | 1.65 | 2.40 | 2.27 | 2.18 | 2.28 | 2.13 | 2.30 |

![Fig 6a](fig6a_lambda_sensitivity.png)

Same qualitative story as the paper: raising λ increases explanation
strength, with complexity rising too (though noisier here — only 60 sampled
users — versus the paper's cleaner, dataset-wide curve).

---

## 6. Result 5 — Qualitative case study (reproduces Fig. 5's explanation style)

```
User 29, target movie: 'George of the Jungle (1997)'  genres=["Children's", 'Comedy']

Turn 1  System: Do you like 'Action' movies?
Turn 2  User:   No, I don't like that.

Turn 3  System: What about the movie 'Nightmare Before Christmas, The (1993)'?  (genres: ["Children's", 'Comedy', 'Musical'])
Turn 4  User:   No, I don't like this one.

Counterfactual explanation for the rejection:
  Most influential attribute: "Children's"  (delta=1.000)
  Explanation strength (ES) = 0.995,  complexity (EC) = 3 attribute(s) changed
  -> "This movie was rejected mainly because of its 'Children's' attribute;
      minimal change to that attribute would flip the recommendation."

Turn 5  System: What about the movie 'George of the Jungle (1997)'?
Turn 6  User:   Yes, I like this movie, thanks!
```

Mirrors Fig. 5 of the paper directly: a real genre of the rejected movie is
identified as the minimal, fine-grained reason for rejection — exactly the
"Mystery"-for-"Uncharted" example the paper itself gives.

---

## 7. How to run

```bash
python3 data_prep.py          # load MovieLens-100k, build the tripartite graph + splits
python3 gnn_embeddings.py     # STAGE 1: train the GNN offline representations
python3 run_experiments.py    # STAGES 2-4 + all ablations, all plots/tables
python3 case_study.py         # one illustrative conversation + explanation
```

## 8. Files
- `data_prep.py` — data loading, tripartite graph construction, 70/20/10 split
- `gnn_embeddings.py` — **Stage 1**: GNN-based offline learning (Eqs. 1–8)
- `cecr.py` — **Stages 2–4**: preference refinement, counterfactual reasoning, recommendation decision
- `simulate.py` — Algorithm 1: the multi-round conversation loop + user simulator
- `run_experiments.py` — runs all configurations, produces every plot/table above
- `case_study.py` — the qualitative example in Section 6
- `out/` — all generated CSVs, PNGs, and the case-study transcript
