## How the comparison was made fair

* **One harness for every model** (`crs_bench.py`). CECR, EAR, FPAN, UNICORN and
  KBRD all converse with the same simulated users, on the same 500 held-out
  (user, liked-item) test pairs per dataset. Each conversation opens with the same
  attribute for every model, drawn from a per-conversation seed. Every model has
  20 rounds, Top-K = 10, and uses the same candidate filtering and the same SR/AT
  code. AT counts a failed conversation as 20 turns.
* **Same splits and preprocessing as the paper**: 70/20/10, users with fewer than
  10 interactions removed, and attributes on fewer than 20 items removed.
* **Same hyper-parameters for every model**: embedding 128, Adam, Ω = 1e-5,
  lr = 1e-3, Top-K = 10, λ = 100 for CECR. The paper says baselines were tuned
  "in the same range"; the only tuned baseline knob is KBRD's recommend
  threshold, chosen on validation conversations by AT.
* **Table IV**: all explainable models (CPR, CAEC, SAUR, Random, CECR) explain
  the Top-10 lists of the same 100 conversation states. Each state is the
  opening attribute plus two max-entropy questions. Each model ranks with its own
  scorer, and PN/PS are computed by rebuilding the item without, or with only,
  the explanation attributes inside that model's own scorer.

## Declared deviations from the paper

1. **Yelp copy.** The paper's Yelp (27,675 users / 70,311 items / 1.37M
   interactions) is the EAR/CPR release, built from yelp.com/dataset. It isn't
   available here, and its size doesn't fit a CPU-only budget alongside seven
   baselines. We use the Yelp copy from the HIN repository the paper cites for its
   other two datasets: after the paper's filtering, 4,613 users / 13,915
   businesses / 168,977 interactions, with 245 category + city attributes.
   Douban-Book and MovieLens match Table III.
2. **Reduced training budget (CPU only).** CECR GNN/conversational epochs are
   25/25 on MovieLens, 10/20 on Yelp and 5/15 on Douban-Book. Baselines get
   5–20 epochs of embedding training, 1,000 REINFORCE episodes (EAR, FPAN) and
   1,500 DQN episodes (UNICORN).
3. **Baselines are faithful re-implementations, not the authors' code.** The
   original repositories target other datasets, GPUs and 2019–2021 library
   versions. Simplifications (see `baselines.py` docstring):
   * CAEC's BERT encoder and SAUR's search-query memory read the conversation
     state, because none of the datasets has dialogue text or search logs.
   * EAR/FPAN's policy chooses ask-vs-recommend; the attribute is the
     preference-weighted max-entropy one, which is EAR's own pre-training
     strategy.
   * KBRD (a dialogue model with no question policy) asks the max-entropy
     attribute.
4. **Precision/Recall ground truth.** The paper defines p(a)_{u,i} = 1 if
   "user u has positive feedback on attribute a of item i", without saying
   where that feedback comes from. Using the attributes confirmed inside the
   same conversation would make CPR's path explanation score 100% by
   construction. We use the user's held-out ratings instead (val+test liked
   items, excluding the target): a is liked when it appears among them more
   often than its base rate.
5. **PS direction.** The printed Eq. 22 repeats PN's "item leaves Top-K"
   condition for PS. We use the standard sufficiency definition from the
   paper's ref. [47]: the item stays in the Top-K when only the explanation is
   kept.
6. **Fig. 6(d)** reports wall-clock seconds per conversation on our CPU.
   Absolute values are not comparable with the paper's hardware; only the
   ordering is.

## Reproduce

```bash
bash run_all.sh      # every step, in order; logs/ and results/<dataset>/
```
