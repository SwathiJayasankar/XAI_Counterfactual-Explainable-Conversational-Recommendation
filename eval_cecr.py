"""
Evaluation of the faithful CECR replica -> every table/figure the paper reports
for MovieLens, written to out_full/.

  RQ1  Table II  + Fig. 3   recommendation performance (SR@5/10/20, AT)
  RQ2  Fig. 4               ablation study (GNNs / PreRi / CouRe / RecPe / CCR)
  RQ3  Table IV             explainability (Precision, Recall, PN, PS)
       Fig. 5               qualitative case study
  RQ4  Fig. 6(a)            lambda sensitivity (ES vs EC)

SR@20 requires conversations of up to 20 rounds -- the earlier NumPy pipeline
capped at 10, which silently made SR@10 and SR@20 identical in every table.
"""
import argparse
import os
import pickle

import numpy as np
import torch

from cecr_torch import (CECRModel, UserSimulator, build_graph_tensors,
                        candidate_set, counterfactual_explanation, run_conversation)

OUT = "out_full"
MAX_ROUNDS = 20
TOPK = 10


def load_model(path, d):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CECRModel(d["n_users"], d["n_items"], d["n_attrs"],
                  dim=ck["dim"], gate_sign=ck.get("gate_sign", 1),
                  use_social=ck.get("use_social", False))
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def embeddings(model, graph, variant="full"):
    """`GNNs` ablation = 'removal of aggregating function, uses the first
    layer's representation' (paper Section V-D)."""
    if variant == "gnns_off":
        return (model.gnn.e_u_in.weight.detach(),
                model.gnn.e_i_in.weight.detach(),
                model.gnn.e_a_in.weight.detach())
    return model.e_u_cache, model.e_i_cache, model.e_a_cache


# --------------------------------------------------------------------------
def eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, seed=7,
                     use_pref_refine=True, use_score_decision=True):
    sim = UserSimulator(attr_matrix)
    rng = np.random.default_rng(seed)
    rounds = []
    for u, tgt in test_pairs:
        r, _ = run_conversation(model, u, tgt, sim, attr_matrix, e_u, e_i, e_a,
                                max_rounds=MAX_ROUNDS, topk=TOPK, rng=rng,
                                use_pref_refine=use_pref_refine,
                                use_score_decision=use_score_decision)
        rounds.append(r)
    out = {}
    for r in (5, 10, 20):
        out[f"SR@{r}"] = float(np.mean([1 if (x is not None and x <= r) else 0 for x in rounds]))
    done = [x for x in rounds if x is not None]
    # AT: unsuccessful conversations count as the full budget (standard CRS convention)
    out["AT"] = float(np.mean([x if x is not None else MAX_ROUNDS for x in rounds]))
    out["curve"] = [float(np.mean([1 if (x is not None and x <= r) else 0 for x in rounds]))
                    for r in range(1, MAX_ROUNDS + 1)]
    out["n"] = len(rounds)
    return out


# --------------------------------------------------------------------------
def eval_explainability(model, d, attr_matrix, e_u, e_i, e_a, test_pairs,
                        lam=100.0, cf_iters=60, seed=7, random_expl=False, n_users=100):
    """
    Eq. 21 (Precision/Recall) and Eq. 22 (PN/PS), evaluated over the actual
    Top-K recommendation list R_{u,K} exactly as the paper defines them.

    Non-circular by construction: the "user's liked attributes" are the ones
    the simulated user actually confirmed during the conversation, not simply
    all attributes of the target item.
    """
    sim = UserSimulator(attr_matrix)
    rng = np.random.default_rng(seed)
    n_items, n_attrs = attr_matrix.shape
    P, R, PN, PS = [], [], [], []

    for u, tgt in test_pairs[:n_users]:
        t_attrs = sim.attrs_of(tgt)
        if not t_attrs:
            continue
        # --- replay a short conversation to obtain a realistic state ---
        first = int(rng.choice(sorted(t_attrs)))
        pos_attrs, neg_attrs, asked = {first}, set(), {first}
        for _ in range(2):
            cands = candidate_set(attr_matrix, pos_attrs, neg_attrs, [], n_items)
            if len(cands) == 0:
                break
            with torch.no_grad():
                eo = model.overall(e_u[u], pos_attrs, neg_attrs, [], e_a, e_i)
            present = np.nonzero(attr_matrix[cands].sum(0) > 0)[0]
            ca = [int(a) for a in present if a not in asked]
            if not ca:
                break
            with torch.no_grad():
                a_star = int(ca[int(torch.argmax(e_a[ca] @ eo))])
            asked.add(a_star)
            (pos_attrs if a_star in t_attrs else neg_attrs).add(a_star)

        cands = candidate_set(attr_matrix, pos_attrs, neg_attrs, [], n_items)
        if len(cands) <= TOPK:
            continue
        with torch.no_grad():
            eo = model.overall(e_u[u], pos_attrs, neg_attrs, [], e_a, e_i)
            scores = model.rank(eo, e_i[cands])
            order = torch.topk(scores, min(TOPK, len(cands))).indices.cpu().numpy()
            topk_items = cands[order]
            kth = float(scores[order[-1]])
            s_ik = torch.as_tensor(kth)

        for it in topk_items:
            it = int(it)
            i_attrs = np.nonzero(attr_matrix[it])[0]
            if len(i_attrs) == 0:
                continue
            if random_expl:
                # Paper's baseline: "Random: randomly selects some aspects as
                # explanations" -- aspects are drawn from the whole attribute
                # set A, NOT from the item's own attributes (drawing from the
                # item's own genres would leak the answer and inflate this
                # baseline far above the paper's reported 0.33% precision).
                k = int(rng.integers(1, 4))
                expl = rng.choice(n_attrs, size=k, replace=False)
            else:
                _, ES, EC, expl, _ = counterfactual_explanation(
                    model, eo, e_i[it], torch.as_tensor(attr_matrix[it]),
                    e_a, s_ik, lam=lam, iters=cf_iters)
                if len(expl) == 0:
                    continue

            # ---- Eq. 21 ----
            liked = pos_attrs & set(i_attrs.tolist())
            ind = np.zeros(n_attrs); ind[np.asarray(expl, dtype=int)] = 1
            p = np.zeros(n_attrs)
            for a in liked:
                p[a] = 1
            num = float((p * ind).sum())
            P.append(num / max(ind.sum(), 1))
            R.append(num / max(p.sum(), 1e-9) if p.sum() > 0 else 0.0)

            # ---- Eq. 22 ----
            # "We set the attribute value of all items to 0 if they are in A_ui":
            # zeroing an attribute the item does NOT carry is a no-op, so only
            # the explanation attributes the item actually has may be removed.
            # (Without this guard a random explanation perturbs the item by
            # attributes it never had, which inflates its PN enormously.)
            expl_owned = [int(a) for a in np.asarray(expl, dtype=int)
                          if attr_matrix[it, int(a)] > 0]
            with torch.no_grad():
                # PN: zero the explanation attributes -> item should LEAVE Top-K
                i_star = e_i[it] - (e_a[torch.as_tensor(expl_owned, dtype=torch.long)].sum(0)
                                    if expl_owned else 0.0)
                PN.append(1.0 if float(model.rank(eo, i_star)) < kth else 0.0)
                # PS: keep ONLY the explanation attributes -> item should STAY
                # (standard sufficiency convention of the paper's own ref. [47];
                #  the paper's printed Eq. 22 repeats PN's condition verbatim,
                #  which is self-contradictory -- documented in explain.txt)
                others = [a for a in i_attrs if a not in set(np.asarray(expl).tolist())]
                i_circ = e_i[it] - (e_a[torch.as_tensor(others, dtype=torch.long)].sum(0)
                                    if others else 0.0)
                PS.append(1.0 if float(model.rank(eo, i_circ)) >= kth else 0.0)

    return dict(Precision=float(np.mean(P)) if P else 0.0,
                Recall=float(np.mean(R)) if R else 0.0,
                PN=float(np.mean(PN)) if PN else 0.0,
                PS=float(np.mean(PS)) if PS else 0.0, n=len(P))


# --------------------------------------------------------------------------
def eval_lambda(model, d, attr_matrix, e_u, e_i, e_a, test_pairs,
                lambdas=(1, 5, 10, 25, 50, 75, 100), n_users=50, seed=7, cf_iters=60):
    """Fig. 6(a): how lambda trades explanation strength off against complexity."""
    sim = UserSimulator(attr_matrix)
    rows = []
    n_items = attr_matrix.shape[0]
    for lam in lambdas:
        ES_l, EC_l = [], []
        rng = np.random.default_rng(seed)
        for u, tgt in test_pairs[:n_users]:
            t_attrs = sim.attrs_of(tgt)
            if not t_attrs:
                continue
            pos_attrs = {int(rng.choice(sorted(t_attrs)))}
            cands = candidate_set(attr_matrix, pos_attrs, set(), [], n_items)
            if len(cands) <= TOPK:
                continue
            with torch.no_grad():
                eo = model.overall(e_u[u], pos_attrs, set(), [], e_a, e_i)
                sc = model.rank(eo, e_i[cands])
                order = torch.topk(sc, TOPK).indices.cpu().numpy()
                it = int(cands[order[0]])
                s_ik = sc[order[-1]].detach()
            _, ES, EC, _, _ = counterfactual_explanation(
                model, eo, e_i[it], torch.as_tensor(attr_matrix[it]), e_a,
                s_ik, lam=float(lam), iters=cf_iters)
            ES_l.append(ES); EC_l.append(EC)
        rows.append((lam, float(np.mean(ES_l)) if ES_l else 0.0,
                     float(np.mean(EC_l)) if EC_l else 0.0))
        print(f"  lambda={lam:>4}  ES={rows[-1][1]:.4f}  EC={rows[-1][2]:.2f}")
    return rows


def case_study(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, titles, genres, seed=3):
    """Fig. 5-style transcript with a counterfactual explanation for a rejection."""
    sim = UserSimulator(attr_matrix)
    rng = np.random.default_rng(seed)
    n_items = attr_matrix.shape[0]
    lines = []
    for u, tgt in test_pairs:
        t_attrs = sim.attrs_of(tgt)
        if len(t_attrs) < 2:
            continue
        first = int(rng.choice(sorted(t_attrs)))
        pos_attrs, neg_attrs, asked, rejected = {first}, set(), {first}, []
        lines = [f"User {u}, target movie: {titles[tgt]!r}  "
                 f"genres={[genres[a] for a in sorted(t_attrs)]}", ""]
        turn = 1
        lines.append(f"Turn {turn}  User:   I'm looking for a '{genres[first]}' movie.")
        ok = False
        for rnd in range(1, 9):
            cands = candidate_set(attr_matrix, pos_attrs, neg_attrs, rejected, n_items)
            if len(cands) == 0:
                break
            with torch.no_grad():
                eo = model.overall(e_u[u], pos_attrs, neg_attrs, rejected, e_a, e_i)
            present = np.nonzero(attr_matrix[cands].sum(0) > 0)[0]
            ca = [int(a) for a in present if a not in asked]
            with torch.no_grad():
                action, _, _ = model.decide_action(eo, cands, ca, e_i, e_a)
            if not ca:
                action = "recommend"
            turn += 1
            if action == "ask":
                with torch.no_grad():
                    a_star = int(ca[int(torch.argmax(e_a[ca] @ eo))])
                asked.add(a_star)
                yes = a_star in t_attrs
                lines.append(f"Turn {turn}  System: Do you like '{genres[a_star]}' movies?")
                turn += 1
                lines.append(f"Turn {turn}  User:   " +
                             ("Yes, I like that." if yes else "No, I don't like that."))
                (pos_attrs if yes else neg_attrs).add(a_star)
            else:
                with torch.no_grad():
                    sc = model.rank(eo, e_i[cands])
                    order = torch.topk(sc, min(TOPK, len(cands))).indices.cpu().numpy()
                top = cands[order]
                if tgt in top:
                    lines.append(f"Turn {turn}  System: What about the movie {titles[tgt]!r}?")
                    turn += 1
                    lines.append(f"Turn {turn}  User:   Yes, I like this movie, thanks!")
                    ok = True
                    break
                it = int(top[0])
                lines.append(f"Turn {turn}  System: What about the movie {titles[it]!r}?  "
                             f"(genres: {[genres[a] for a in np.nonzero(attr_matrix[it])[0]]})")
                turn += 1
                lines.append(f"Turn {turn}  User:   No, I don't like this one.")
                outside = np.setdiff1d(cands, top)
                if len(outside):
                    with torch.no_grad():
                        s_ik = model.rank(eo, e_i[outside]).max()
                    dlt, ES, EC, expl, _ = counterfactual_explanation(
                        model, eo, e_i[it], torch.as_tensor(attr_matrix[it]),
                        e_a, s_ik, lam=100.0, iters=80)
                    if len(expl):
                        mags = np.abs(dlt[torch.as_tensor(expl)].numpy())
                        best = int(expl[int(np.argmax(mags))])
                        lines += ["",
                                  "  [Counterfactual explanation for the rejection]",
                                  f"    Most influential attribute: '{genres[best]}' "
                                  f"(delta={float(dlt[best]):.3f})",
                                  f"    Explanation strength ES = {ES:.3f}, "
                                  f"complexity EC = {EC} attribute(s)",
                                  f"    -> \"This was rejected mainly because of its "
                                  f"'{genres[best]}' attribute; a minimal change there "
                                  f"would flip the decision.\"", ""]
                rejected += [int(x) for x in top]
        if ok and len(lines) > 6:
            return "\n".join(lines)
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/cecr_full.pt")
    ap.add_argument("--model-noucf", default="data/cecr_nocf.pt",
                    help="separately trained model for the CouRe ablation")
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--data", default="data/prepared.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-expl", action="store_true",
                    help="skip Table IV (the paper publishes it only for "
                         "MovieLens and Yelp, so Douban-Book has no reference)")
    ap.add_argument("--skip-lambda", action="store_true", help="skip Fig. 6(a)")
    ap.add_argument("--skip-ablations", action="store_true",
                    help="evaluate only the full model (paper's Fig. 4 ablations "
                         "are published for MovieLens/Yelp only)")
    args = ap.parse_args()

    if args.out:
        OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    with open(args.data, "rb") as f:
        d = pickle.load(f)
    attr_matrix = d["attr_matrix"]
    graph = build_graph_tensors(d)
    model = load_model(args.model, d)

    test = d["test"]
    liked = test[test.rating >= 4]
    rng = np.random.default_rng(7)
    arr = liked[["user", "item"]].values
    sel = rng.choice(len(arr), min(args.n_test, len(arr)), replace=False)
    test_pairs = [(int(u), int(i)) for u, i in arr[sel]]

    e_u, e_i, e_a = embeddings(model, graph)

    # ---------------- RQ1 / RQ2 ----------------
    print("[RQ1/RQ2] recommendation performance + ablations")
    results = {}
    results["CECR (CCR)"] = eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs)
    print("  CCR   ", {k: round(v, 3) for k, v in results["CECR (CCR)"].items() if k != "curve"})

    if not args.skip_ablations:
        results["- PreRi"] = eval_performance(model, d, attr_matrix, e_u, e_i, e_a,
                                              test_pairs, use_pref_refine=False)
        print("  PreRi ", {k: round(v, 3) for k, v in results["- PreRi"].items() if k != "curve"})

        results["- RecPe"] = eval_performance(model, d, attr_matrix, e_u, e_i, e_a,
                                              test_pairs, use_score_decision=False)
        print("  RecPe ", {k: round(v, 3) for k, v in results["- RecPe"].items() if k != "curve"})

        gu, gi, ga = embeddings(model, graph, "gnns_off")
        results["- GNNs"] = eval_performance(model, d, attr_matrix, gu, gi, ga, test_pairs)
        print("  GNNs  ", {k: round(v, 3) for k, v in results["- GNNs"].items() if k != "curve"})

    if not args.skip_ablations and os.path.exists(args.model_noucf):
        m2 = load_model(args.model_noucf, d)
        e_u2, e_i2, e_a2 = embeddings(m2, graph)
        results["- CouRe"] = eval_performance(m2, d, attr_matrix, e_u2, e_i2, e_a2, test_pairs)
        print("  CouRe ", {k: round(v, 3) for k, v in results["- CouRe"].items() if k != "curve"})

    with open(f"{OUT}/table2_performance.csv", "w") as f:
        f.write("Model,SR@5,SR@10,SR@20,AT\n")
        for k, v in results.items():
            f.write(f"{k},{v['SR@5']:.4f},{v['SR@10']:.4f},{v['SR@20']:.4f},{v['AT']:.2f}\n")

    # ---------------- RQ3 ----------------
    ex_cecr = ex_rand = None
    if not args.skip_expl:
      print("[RQ3] explainability (Table IV)")
      ex_cecr = eval_explainability(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, lam=args.lam)
    ex_rand = eval_explainability(model, d, attr_matrix, e_u, e_i, e_a, test_pairs,
                                  lam=args.lam, random_expl=True)
    print("  CECR ", {k: round(v, 4) for k, v in ex_cecr.items()})
    print("  Random", {k: round(v, 4) for k, v in ex_rand.items()})
    with open(f"{OUT}/table4_explainability.csv", "w") as f:
        f.write("Model,Precision,Recall,PN,PS\n")
        f.write(f"Random,{ex_rand['Precision']:.4f},{ex_rand['Recall']:.4f},"
                f"{ex_rand['PN']:.4f},{ex_rand['PS']:.4f}\n")
        f.write(f"CECR,{ex_cecr['Precision']:.4f},{ex_cecr['Recall']:.4f},"
                f"{ex_cecr['PN']:.4f},{ex_cecr['PS']:.4f}\n")

    # ---------------- RQ4 ----------------
    rows = []
    if not args.skip_lambda:
      print("[RQ4] lambda sensitivity (Fig. 6a)")
      rows = eval_lambda(model, d, attr_matrix, e_u, e_i, e_a, test_pairs)
    with open(f"{OUT}/fig6a_data.csv", "w") as f:
        f.write("lambda,ES,EC\n")
        for lam, es, ec in rows:
            f.write(f"{lam},{es:.4f},{ec:.4f}\n")

    # ---------------- case study ----------------
    if os.path.exists("data/u.item") and d.get("name") is None:
        import pandas as pd
        items = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None)
        titles = {i: t for i, t in zip(items[0].values - 1, items[1].values)}
    else:
        # Douban-Book ships ids only, no titles
        titles = {i: f"item #{i}" for i in range(d["n_items"])}
    genres = d["genre_names"]
    txt = case_study(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, titles, genres)
    with open(f"{OUT}/case_study.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    with open(f"{OUT}/results.pkl", "wb") as f:
        pickle.dump(dict(results=results,
                         expl=(dict(CECR=ex_cecr, Random=ex_rand) if ex_cecr else {}),
                         lam=rows), f)
    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
