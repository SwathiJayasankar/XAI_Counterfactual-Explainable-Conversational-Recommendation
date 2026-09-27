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

# NOTE: base ("offline") embeddings in data/base_emb.npz are now produced by
# gnn_embeddings.py -- STAGE 1 "GNN-based offline learning" (Eqs. 1-8) -- not
# by the old plain matrix-factorization script. Run gnn_embeddings.py first.

def run_simulation(cfg_name, use_pref_refine, use_counterfactual, use_score_decision,
                    max_rounds=10, topk=10, n_test_users=300, seed=1, collect_explanations=False):
    with open("data/prepared.pkl", "rb") as f:
        d = pickle.load(f)
    emb = np.load("data/base_emb.npz")
    eu0, ei0, ea0 = emb["eu"], emb["ei"], emb["ea"]
    dim = eu0.shape[1]
    attr_matrix = d["attr_matrix"]
    n_items = d["n_items"]

    rng = np.random.default_rng(seed)
    model = CECR(eu0, ei0, ea0, attr_matrix, dim,
                 use_pref_refine=use_pref_refine,
                 use_counterfactual=use_counterfactual,
                 use_score_decision=use_score_decision,
                 seed=seed)

    # candidate (user, target item) pairs: items the user liked in the test split
    test = d["test"]
    liked = test[test.rating >= 4]
    users_avail = liked.user.unique()
    rng.shuffle(users_avail)
    users_avail = users_avail[:n_test_users]

    success_round = []     # round index (1..max_rounds) at which success happened, or None
    explanations_log = []

    # local (per-run) copies of item/user embeddings that get nudged by
    # counterfactual-sample retraining (Eq. 19) -- only mutated when
    # use_counterfactual=True, so the ablation truly removes this effect.
    ei_local = ei0.copy()

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
        candidate_items = [i for i in range(n_items) if i != target_item]
        rng.shuffle(candidate_items)
        candidate_items = candidate_items[:400] + [target_item]   # cap search space for speed

        eu = eu0[u].copy()
        success = None

        for rnd in range(1, max_rounds + 1):
            e_prime_u = model.refine_user(eu, pos_attrs, neg_attrs)
            eoverall = model.overall_preference(e_prime_u, rejected_item_reps)

            filtered_candidates = model.filter_candidates(candidate_items, pos_attrs, neg_attrs)
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
                        item_vec = model.refine_rejected_item(ei_local[it], pos_attrs)
                        rejected_item_reps.append(item_vec)
                        rejected_items.append(it)

                        if model.use_counterfactual:
                            # pick k = the item just outside the top-K (next best) as anchor
                            outside = [c for c in candidate_items if c not in topk_items]
                            if outside:
                                k_item = int(rng.choice(outside[:20]))
                                s_ik = eoverall @ ei_local[k_item]
                                delta, ES, EC, expl_attrs, cf_vec = model.counterfactual_explanation(
                                    eoverall, ei_local[it], attr_matrix[it], s_ik)
                                if collect_explanations:
                                    explanations_log.append(dict(
                                        user=u, item=it, ES=ES, EC=EC,
                                        expl_attrs=expl_attrs.tolist()))
                                # Eq. 19: augment + retrain with counterfactual sample (u, i, k)
                                s_i = eoverall @ ei_local[it]
                                sig = 1.0 / (1.0 + np.exp(-np.clip(s_ik - s_i, -30, 30)))
                                lr = 0.05
                                grad_common = (1 - sig)
                                ei_local[k_item] += lr * grad_common * eoverall
                                ei_local[it] -= lr * grad_common * eoverall * 0.5
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
