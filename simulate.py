"""
User simulator + multi-round conversation loop, i.e. Algorithm 1 of the paper.

For each test (user, target_item) pair (target_item is a held-out item the
user actually liked, rating >= 4):
  1. user states one true positive attribute of the target item (Fig. 1, step 1)
  2. loop for up to R rounds:
       - system decides ask-attribute vs recommend-item (Eq. 20)
       - if recommend: propose Top-K items
           - if target item is in Top-K -> success (user accepts)
           - else all Top-K items are *rejected*:
               -> preference refinement update (Eqs. 9-13)
               -> counterfactual reasoning + sample generation + retraining (Eqs. 14-19),
                  only if use_counterfactual=True
       - if ask: system asks about attribute with highest score; user replies
         truthfully (yes/no) based on the target item's real attributes
  3. record success round (for SR@r), and whether explanations were produced
"""
import numpy as np
import pickle
from cecr import CECR
from ranker import Ranker

# NOTE: base ("offline") embeddings in data/base_emb.npz are now produced by
# gnn_embeddings.py -- STAGE 1 "GNN-based offline learning" (Eqs. 1-8) -- not
# by the old plain matrix-factorization script. Run gnn_embeddings.py first.

def run_simulation(cfg_name, use_pref_refine, use_counterfactual, use_score_decision,
                    max_rounds=10, topk=10, n_test_users=300, seed=1, collect_explanations=False,
                    use_hard_filter=True, use_ranker=False):
    with open("data/prepared.pkl", "rb") as f:
        d = pickle.load(f)
    emb = np.load("data/base_emb.npz")
    eu0, ei0, ea0 = emb["eu"], emb["ei"], emb["ea"]
    dim = eu0.shape[1]
    attr_matrix = d["attr_matrix"]
    n_items = d["n_items"]

    rng = np.random.default_rng(seed)
    # Algorithm 1's trained ranking network g (Eq. 14) -- OPTIONAL (off by
    # default). We implemented and tested it (see ranker.py / explain.txt):
    # empirically, a freshly-initialized small MLP trained via BPR on top of
    # the already-BPR-trained embeddings underperforms the plain dot product
    # in this reduced-scale (32-dim, ~70k-interaction) reproduction, so the
    # default recommendation path keeps the dot product. use_ranker=True is
    # kept available to reproduce that documented comparison experiment.
    ranker = None
    if use_ranker:
        try:
            ranker = Ranker.load("data/ranker.npz").copy()
        except FileNotFoundError:
            ranker = None

    model = CECR(eu0, ei0, ea0, attr_matrix, dim,
                 use_pref_refine=use_pref_refine,
                 use_counterfactual=use_counterfactual,
                 use_score_decision=use_score_decision,
                 seed=seed, ranker=ranker)

    # candidate (user, target item) pairs: items the user liked in the test split
    test = d["test"]
    liked = test[test.rating >= 4]
    users_avail = liked.user.unique()
    rng.shuffle(users_avail)
    users_avail = users_avail[:n_test_users]

    success_round = []     # round index (1..max_rounds) at which success happened, or None
    explanations_log = []

    for u in users_avail:
        target_candidates = liked[liked.user == u].item.values
        target_item = int(rng.choice(target_candidates))
        target_attrs = set(np.where(attr_matrix[target_item] > 0)[0].tolist())
        if not target_attrs:
            continue

        pos_attrs, neg_attrs, asked_attrs = set(), set(), set()
        # step 1: user states one true positive attribute
        first_attr = int(rng.choice(list(target_attrs)))
        pos_attrs.add(first_attr)
        asked_attrs.add(first_attr)

        rejected_items = []
        rejected_item_reps = []
        # FULL item catalog (all 1682 items), not a subsampled pool -- evaluating
        # against the whole corpus is what the paper's own Top-K ranking does;
        # capping this to a small random subset (as an earlier version of this
        # script did) makes the task trivially easy and inflates SR.
        candidate_items = list(range(n_items))
        rng.shuffle(candidate_items)

        eu = eu0[u].copy()
        success = None

        for rnd in range(1, max_rounds + 1):
            e_prime_u = model.refine_user(eu, pos_attrs, neg_attrs)
            eoverall = model.overall_preference(e_prime_u, rejected_item_reps)

            if use_hard_filter:
                filtered_candidates = model.filter_candidates(candidate_items, pos_attrs, neg_attrs)
            else:
                # paper-literal path: no post-hoc genre filter, ranking relies
                # purely on the learned e_overall . e_i score (Eq. 20)
                filtered_candidates = candidate_items
            action, remaining_attrs = model.decide_action(eoverall, filtered_candidates, asked_attrs)

            if action == "recommend" or not remaining_attrs:
                topk_items, _ = model.top_k(eoverall, filtered_candidates, k=topk)
                if target_item in topk_items:
                    success = rnd
                    break
                else:
                    # all topk rejected
                    for it in topk_items:
                        if it == target_item:
                            continue
                        item_vec = model.refine_rejected_item(ei0[it], pos_attrs)
                        rejected_item_reps.append(item_vec)
                        rejected_items.append(it)

                        if model.use_counterfactual:
                            # pick k = the item just outside the top-K (next best) as anchor
                            outside = [c for c in candidate_items if c not in topk_items]
                            if outside:
                                k_item = int(rng.choice(outside[:20]))
                                s_ik = model.score_item(eoverall, ei0[k_item])
                                delta, ES, EC, expl_attrs, cf_vec = model.counterfactual_explanation(
                                    eoverall, ei0[it], attr_matrix[it], s_ik)
                                if collect_explanations:
                                    explanations_log.append(dict(
                                        user=u, item=it, ES=ES, EC=EC,
                                        expl_attrs=expl_attrs.tolist()))
                                # Eq. 19 / Algorithm 1 step 19: re-train the target model g
                                # with the counterfactual sample (u, i, ik) -- prefer k_item
                                # (still plausible) over the rejected item `it`.
                                if model.ranker is not None:
                                    model.ranker.train_step(eoverall, ei0[k_item], ei0[it])
                    candidate_items = [c for c in candidate_items if c not in rejected_items]
            else:
                # ask about the attribute with highest score among remaining
                attr_scores = eoverall @ ea0[remaining_attrs].T
                a_star = remaining_attrs[int(np.argmax(attr_scores))]
                asked_attrs.add(a_star)
                if a_star in target_attrs:
                    pos_attrs.add(a_star)
                else:
                    neg_attrs.add(a_star)

        success_round.append(success)

    sr_at_r = {}
    for r in [5, 10, 20]:
        rr = min(r, max_rounds)
        sr_at_r[f"SR@{r}"] = float(np.mean([1 if (s is not None and s <= rr) else 0
                                             for s in success_round]))
    finished = [s for s in success_round if s is not None]
    at = float(np.mean(finished)) if finished else float(max_rounds)
    sr_curve = []
    for r in range(1, max_rounds + 1):
        sr_curve.append(np.mean([1 if (s is not None and s <= r) else 0 for s in success_round]))

    return dict(name=cfg_name, sr_at_r=sr_at_r, AT=at, sr_curve=sr_curve,
                explanations=explanations_log, n_users=len(success_round))


if __name__ == "__main__":
    res = run_simulation("CECR (full)", True, True, True, n_test_users=50, collect_explanations=True)
    print(res["sr_at_r"], res["AT"])
