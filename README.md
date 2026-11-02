# Counterfactual Explainable Conversational Recommendation: Reproduction and Extensions

This repository reproduces **CECR** in PyTorch and adds three extensions that address gaps we found in the original method.

> **Original paper.** D. Yu, Q. Li, X. Wang, Q. Li, G. Xu, *"Counterfactual Explainable Conversational Recommendation"*,
> **IEEE Transactions on Knowledge and Data Engineering**, vol. 36, no. 6, 2024. DOI: [10.1109/TKDE.2023.3322403](https://doi.org/10.1109/TKDE.2023.3322403)

CECR is a conversational recommender (CRS). In each turn it either asks the user about an attribute or recommends a Top-K list. It explains each recommendation with a **minimal counterfactual**: the smallest change to the item's attributes that would push the item out of the Top-K.

| | |
|---|---|
| **Phase 1: reproduction** | The whole CECR pipeline (GNN encoder → preference refinement → counterfactual reasoning → recommendation), 7 baselines (EAR, FPAN, UNICORN, KBRD, CPR, CAEC, SAUR) and every table and figure in the paper (Table II and IV, Fig. 3–6). All models run on the same conversation harness and datasets: MovieLens-100k, Yelp and Douban-Book. |
| **Phase 2: extensions** | **(1)** Entropy-based ask-vs-recommend decision. **(2)** Amortised counterfactual explanations (*DeltaNet*). **(3)** Robust evaluation: 5 seeds, noisy users, confidence intervals and paired significance tests. |

---

## Contents

1. [Research gaps and extensions](#1-research-gaps-and-extensions)
2. [Headline results](#2-headline-results)
3. [What changed: new and modified files](#3-what-changed-new-and-modified-files)
4. [Repository layout](#4-repository-layout)
5. [Installation](#5-installation)
6. [Reproducing the experiments](#6-reproducing-the-experiments)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Deviations from the paper and limitations](#8-deviations-from-the-paper-and-limitations)
9. [Further documentation](#9-further-documentation)
10. [References](#10-references)

---

## 1. Research gaps and extensions

| # | Gap in the original CECR | Our extension | Code |
|---|---|---|---|
| **1** | **The ask-vs-recommend rule (Eq. 20) almost never asks.** It compares `max e_i·e_overall` with `max e_a·e_overall`. Item and attribute embeddings come from different GNN branches and have different norms, so item scores always win. On MovieLens CECR asked a question in **0 %** of decisions after the first turn (1 % on Yelp), so the "conversational" recommender just recommends 10 items every turn. | **Entropy questions with a validated threshold.** While more than *T* candidates remain, ask the attribute with the highest preference-weighted entropy; otherwise recommend. *T* is picked on validation conversations, never on test. We also tried a learned REINFORCE policy (EAR-style); it collapses to "never ask". | `decision.py`, `run_decision.py` |
| **2** | **Counterfactual explanations are slow.** Eqs. 17–18 run a separate gradient-descent search (60 Adam steps) for every explained item, which makes CECR the slowest model per turn (Fig. 6d). At the paper's λ = 100 the ranking term also outweighs the sparsity penalty by ~50×, so the explanations stop being minimal. | **Amortised counterfactuals (*DeltaNet*).** A small MLP predicts Δ in one forward pass. It is trained on the paper's own Eq. 17 objective over training-split conversation states, with the CECR model frozen. It can optionally warm-start 5 refinement steps. Results are reported at λ = 100 (paper) and λ = 3 (sparse explanations). | `cf_amortized.py` |
| **3** | **A single evaluation run with a truthful user.** The paper reports one run with no variance or significance tests, and its simulated user always answers correctly. That favours models that ask many questions. | **Robust evaluation.** 5 seeds; each seed retrains CECR and every baseline and draws new test conversations. Answer noise is 0 / 0.1 / 0.2. We report mean ± 95 % CI (Student-t), Wilcoxon signed-rank tests on turns and exact McNemar tests on SR@10. | `robust_cecr.py`, `summarize_robust.py`, `run_robust.sh`, `run_baselines.py --robust` |

---

## 2. Headline results

All numbers come from **our own runs** unless a row says *(paper)*. Every model was evaluated on the same 500 held-out test conversations.

### Extension 1: ask-vs-recommend decision

| Dataset | Model | SR@5 | SR@10 | SR@20 | AT ↓ |
|---|---|---|---|---|---|
| MovieLens | CECR *(published in the paper)* | 0.891 | 0.949 | 0.982 | 3.77 |
| MovieLens | CECR, Eq. 20 (our reproduction) | 0.696 | 0.830 | 0.952 | 5.26 |
| MovieLens | **CECR + entropy questions (ours)** | **0.730** | **0.898** | **0.982** | **4.63** |
| MovieLens | KBRD (best baseline, our run) | 0.682 | 0.878 | 0.960 | 5.14 |
| Yelp | CECR, Eq. 20 (our reproduction) | 0.374 | 0.502 | 0.660 | 10.99 |
| Yelp | **CECR + entropy questions (ours)** | 0.410 | 0.708 | 0.952 | 7.95 |
| Yelp | KBRD (best baseline, our run) | **0.466** | **0.710** | **0.960** | **7.39** |

On MovieLens, CECR now beats every baseline on all four metrics. On Yelp, SR@10 rises by +0.21 and SR@20 by +0.29, and conversations are 3 turns shorter.

### Extension 2: amortised counterfactual explanations

| Dataset | λ | Explainer | Pr | Re | PN | PS | ms / explanation |
|---|---|---|---|---|---|---|---|
| MovieLens | 100 | paper optimiser | 0.533 | 0.731 | 0.860 | 0.949 | 140 |
| MovieLens | 100 | DeltaNet | 0.530 | 0.730 | 0.857 | 0.954 | **3.9** |
| MovieLens | 3 | paper optimiser | 0.510 | 0.629 | 0.812 | 0.842 | 259 |
| MovieLens | 3 | **DeltaNet** | **0.543** | **0.715** | **0.890** | **0.897** | **5.4** |
| Yelp | 100 | paper optimiser | 0.438 | 0.811 | 0.941 | 0.945 | 194 |
| Yelp | 100 | DeltaNet + 5 steps | 0.452 | 0.794 | 0.935 | 0.926 | 18 |

DeltaNet is **34–49× faster** per explanation. At λ = 3 it is as good as or better than the per-item optimiser.

### Extension 3: robust evaluation (MovieLens, 5 seeds, mean ± 95 % CI)

| Model | SR@10, noise 0 | SR@20, noise 0 | SR@20, noise 0.1 | SR@20, noise 0.2 |
|---|---|---|---|---|
| CECR (Eq. 20) | 0.847 ± 0.024 | 0.948 ± 0.005 | 0.940 | 0.936 |
| **CECR + entropy questions** | **0.895 ± 0.015** | **0.976 ± 0.005** | 0.900 | 0.832 |
| KBRD | 0.866 ± 0.010 | 0.953 ± 0.008 | 0.866 | 0.806 |
| FPAN | 0.850 ± 0.014 | 0.947 ± 0.011 | 0.947 | 0.947 |

With a truthful user, CECR + entropy questions is significantly better than every baseline on SR@10 (McNemar p = 1.6e-5 vs KBRD, p = 2.4e-24 vs Eq. 20). With noisy users, every model that asks questions gets worse. This limitation is discussed in [`IMPROVEMENTS_RESULTS.md`](IMPROVEMENTS_RESULTS.md).

Full tables: [`IMPROVEMENTS_RESULTS.md`](IMPROVEMENTS_RESULTS.md) and [`results/movielens/robust/summary.md`](results/movielens/robust/summary.md).

---

## 3. What changed: new and modified files

### New files (Phase 2)

| File | Purpose |
|---|---|
| `decision.py` | `CECRDecisionAgent`: replaces CECR's Eq. 20 decision with (a) Eq. 20 on cosine similarity, (b) a learned REINFORCE policy, or (c) the entropy-question threshold rule. CECR's ranking, attribute scoring and explanations are unchanged. |
| `run_decision.py` | Trains and validates each decision rule (policy training, threshold *T* search on validation) and evaluates on the Table II test conversations. |
| `cf_amortized.py` | `DeltaNet`: trains the amortised counterfactual explainer on the Eq. 17 objective and compares it with the paper's per-item optimiser on the Table IV protocol (Pr, Re, PN, PS, EC, latency). |
| `robust_cecr.py` | Evaluates CECR (Eq. 20, learned policy, entropy questions) for one seed at every answer-noise level. |
| `summarize_robust.py` | Combines all seeds into mean ± 95 % CI tables and Wilcoxon / McNemar paired tests → `results/<ds>/robust/summary.md`. |
| `run_robust.sh` | Runs the whole robust study: 5 seeds × all models × 3 noise levels. |
| `IMPROVEMENTS_RESULTS.md` | Full write-up of the three extensions. |

### Modified files

| File | Change |
|---|---|
| `crs_bench.py` | `run_conversation` / `evaluate` take a new `answer_noise` argument: each attribute answer is flipped with that probability. The default is 0, the paper's truthful simulator. `evaluate` now also returns per-conversation success rounds (`out["rounds"]`), which the paired tests need. |
| `run_baselines.py` | New `--robust` and `--noise-levels` flags: test pairs are drawn per seed, every baseline is evaluated at each noise level, and results go to `results/<ds>/robust/baselines_seed<k>.pkl`. Default behaviour is unchanged. |

The diff of the Phase 2 code alone: `git diff 7e0c409 71d60d9 -- '*.py' '*.sh'`.

---

## 4. Repository layout

```
.
├── cecr_torch.py            # CECR model in PyTorch (Eqs. 1-20), user simulator, counterfactual search
├── train_cecr.py            # two-stage training (GNN pre-training, then conversational stage + CF augmentation)
├── eval_cecr.py             # Table II/IV, Fig. 3-6 for CECR on one dataset
├── crs_bench.py             # shared multi-round conversation harness and metrics (all models)
├── baselines.py             # EAR, FPAN, UNICORN, KBRD, CPR, CAEC, SAUR re-implementations
├── run_baselines.py         # trains and evaluates the baselines
├── data_prep.py             # MovieLens-100k  -> data/prepared.pkl
├── prep_yelp.py             # Yelp (HIN)      -> data/prepared_yelp.pkl
├── prep_douban.py           # Douban-Book     -> data/prepared_douban.pkl
├── make_paper_figs.py       # draws the reproduction figures in the paper's format
│
├── decision.py  run_decision.py                        # Extension 1
├── cf_amortized.py                                     # Extension 2
├── robust_cecr.py  summarize_robust.py  run_robust.sh  # Extension 3
│
├── data/                    # raw datasets, prepared splits, trained checkpoints (*.pt)
├── results/<dataset>/       # result tables (CSV/JSON/PKL) per dataset, incl. robust/
├── logs/                    # stdout of every run reported here
├── reimplementation/        # Phase 1 report: figures in paper format + explanation
├── papper output/           # screenshots of the paper's published results
│
├── cecr.py  gnn_embeddings.py  simulate.py  ranker.py  run_experiments.py   # earlier NumPy prototype (superseded)
└── demo_cecr.py  demo_conversation.py  interactive_demo.py  case_study.py   # demos and qualitative case study
```

---

## 5. Installation

Tested with **Python 3.11** on **CPU only** (Windows 11 and Linux). No GPU is needed.

```bash
git clone https://github.com/SwathiJayasankar/XAI_Counterfactual-Explainable-Conversational-Recommendation.git
cd XAI_Counterfactual-Explainable-Conversational-Recommendation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

| Dependency | Tested version |
|---|---|
| torch | 2.9.1 (CPU) |
| numpy | 2.4.0 |
| pandas | 2.3.3 |
| scipy | 1.16.3 |
| matplotlib | 3.10.8 |

The `.sh` scripts need a POSIX shell (Linux, macOS or Git Bash on Windows). Each command in them can also be run on its own.

### Data

The raw data are already in `data/`:

| Dataset | Source | Attributes |
|---|---|---|
| MovieLens-100k | GroupLens (`data/u.*`) | 18 genres |
| Yelp | [HIN-Datasets](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding) (`data/yelp/`) | business category + city |
| Douban-Book | [HIN-Datasets](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding) (`data/douban_book/`) | author, publisher, year |

`data/prepared.pkl`, `data/prepared_yelp.pkl` and the trained checkpoints (`data/cecr_*.pt`) are committed, so you can **skip straight to the extensions**. `data/prepared_douban.pkl` (67 MB) is not committed; rebuild it with `python prep_douban.py`.

---

## 6. Reproducing the experiments

### 6.1 Quick start: Phase 2 extensions with the committed checkpoints

```bash
# Extension 1: ask-vs-recommend decision           (~minutes per dataset)
python run_decision.py --data data/prepared.pkl      --model data/cecr_full.pt      --tag movielens
python run_decision.py --data data/prepared_yelp.pkl --model data/cecr_yelp_full.pt --tag yelp

# Extension 2: amortised counterfactual explanations
python cf_amortized.py --data data/prepared.pkl      --model data/cecr_full.pt      --tag movielens --lam 100
python cf_amortized.py --data data/prepared.pkl      --model data/cecr_full.pt      --tag movielens --lam 3
python cf_amortized.py --data data/prepared_yelp.pkl --model data/cecr_yelp_full.pt --tag yelp      --lam 100
python cf_amortized.py --data data/prepared_yelp.pkl --model data/cecr_yelp_full.pt --tag yelp      --lam 3

# Extension 3: robust evaluation (5 seeds x all models x 3 noise levels; several hours on CPU)
bash run_robust.sh
python summarize_robust.py --tag movielens --ref CECR+ask
```

| Experiment | Output |
|---|---|
| Extension 1 | `results/<ds>/decision_results.json`, `logs/decision_*.txt` |
| Extension 2 | `results/<ds>/cf_amortized_lam{3,100}.json`, `logs/cf_amortized_*.txt` |
| Extension 3 | `results/movielens/robust/summary.md`, `logs/robust_*.txt` |

### 6.2 Full Phase 1 reproduction (original method and baselines)

`run_all.sh` rebuilds everything from raw data: data preparation, CECR training, the CouRe ablation, CECR evaluation, all 7 baselines and the figures.

```bash
bash run_all.sh
```

Or step by step for one dataset (MovieLens):

```bash
python data_prep.py
python train_cecr.py --data data/prepared.pkl --stage1-epochs 25 --stage2-epochs 25 --out data/cecr_full.pt
python train_cecr.py --data data/prepared.pkl --stage2-epochs 25 --no-counterfactual \
       --init-from data/cecr_full.pt --out data/cecr_nocf.pt              # CouRe ablation
python eval_cecr.py  --data data/prepared.pkl --model data/cecr_full.pt \
       --model-noucf data/cecr_nocf.pt --out results/movielens --skip-lambda
python run_baselines.py --data data/prepared.pkl --tag movielens --expl
python make_paper_figs.py                                                 # -> reimplementation/figures/
```

The paper's hyper-parameters are used throughout: embedding size 128, Adam (lr 1e-3), Ω = 1e-5, λ = 100, Top-K = 10, max 20 turns, 70/20/10 split. Seeds are fixed, so reruns give the same numbers up to CPU thread non-determinism.

---

## 7. Evaluation protocol

Every model (CECR, its variants and all baselines) runs on the same harness, `crs_bench.py`:

* **Test conversations.** 500 held-out (user, liked item) pairs per dataset. The opening attribute is drawn from a per-conversation RNG, so it is the same for every model.
* **Simulated user.** Answers "yes" iff the target item has the attribute that was asked about. With `answer_noise > 0`, each answer is flipped with that probability. A recommendation succeeds iff the target is in the Top-K.
* **Recommendation metrics.** SR@5/10/20: success within *t* turns. AT: average turns, with failures counted as 20.
* **Explanation metrics (Table IV).** Precision and Recall against the user's liked attributes. PN (necessity): removing the explanation's attributes drops the item from the Top-10. PS (sufficiency): keeping only those attributes keeps it in the Top-10. EC is explanation complexity.
* **Model selection.** Thresholds, policies and baseline knobs are tuned on validation conversations only.

---

## 8. Deviations from the paper and limitations

* **Compute.** CPU only, with fewer training epochs than the paper (MovieLens 25+25, Yelp 10+20, Douban 5+15).
* **Yelp copy.** We use the HIN-repo Yelp (16k users / 14k businesses), not the EAR/CPR release in the paper's Table III. It has 245 attributes, so each answer leaves a much larger candidate pool than on MovieLens.
* **Baselines** are our own re-implementations, trained on the same splits through the same harness.
* **Absolute numbers are lower than published for every model, baselines included.** So the gap comes mainly from the setting, not from one broken model. The paper's results are never presented as ours: published numbers are always labelled *(paper)*.
* **Limitations of the extensions.** Entropy questions make the model more sensitive to wrong answers, because every "no" is a hard filter. Soft candidate filtering is the natural next step. DeltaNet loses 2–4 points of Re/PN/PS on Yelp at λ = 100 unless refined for 5 steps. The robust study was run on MovieLens only.

---

## 9. Further documentation

| Document | Contents |
|---|---|
| [`IMPROVEMENTS_RESULTS.md`](IMPROVEMENTS_RESULTS.md) | Phase 2: full results, analysis and limitations of the three extensions |
| [`results/movielens/robust/summary.md`](results/movielens/robust/summary.md) | Robust study: all noise levels, CIs and paired tests |
| [`RESULTS_REPRODUCTION_analysis.md`](RESULTS_REPRODUCTION_analysis.md) | Phase 1: what reproduces and what does not |
| [`reimplementation/REIMPLEMENTATION_RESULTS.md`](reimplementation/REIMPLEMENTATION_RESULTS.md) | Phase 1: Table IV, Fig. 3, 4, 6 next to the paper's |
| [`reimplementation/EXPLANATION.md`](reimplementation/EXPLANATION.md) | Phase 1: explanation of each reproduced figure |
| [`RESULTS_FINAL.md`](RESULTS_FINAL.md) | Earlier MovieLens replication report |

---

## 10. References

1. D. Yu, Q. Li, X. Wang, Q. Li, G. Xu. *Counterfactual Explainable Conversational Recommendation.* IEEE TKDE 36(6), 2024. **(original paper)**
2. W. Lei, X. He, Y. Miao, Q. Wu, R. Hong, M.-Y. Kan, T.-S. Chua. *Estimation–Action–Reflection: Towards Deep Interaction Between Conversational and Recommender Systems.* WSDM 2020. (EAR; uses attribute entropy for question selection)
3. Y. Sun, Y. Zhang. *Conversational Recommender System.* SIGIR 2018. (entropy-based conversational state)
4. J. Tan, S. Xu, Y. Ge, Y. Li, X. Chen, Y. Zhang. *Counterfactual Explainable Recommendation.* CIKM 2021. (the counterfactual objective CECR builds on)
5. S. Verma, K. Hines, J. P. Dickerson. *Amortized Generation of Sequential Algorithmic Recourses for Black-box Models.* AAAI 2022. (amortised counterfactual generation)
6. S. Zhang, K. Balog. *Evaluating Conversational Recommender Systems via User Simulation.* KDD 2020. (realistic user simulation)
7. R. Dror, G. Baumer, S. Shlomov, R. Reichart. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing.* ACL 2018. (significance testing)
