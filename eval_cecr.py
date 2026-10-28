"""
Evaluation of the faithful CECR replica -> every table/figure the paper reports
for CECR on one dataset, written to results/<dataset>/.

  RQ1  Table II  + Fig. 3   recommendation performance (SR@5/10/20, AT)
  RQ2  Fig. 4               ablation study (GNNs / PreRi / CouRe / RecPe / CCR)
  RQ3  Table IV             explainability (Precision, Recall, PN, PS)
       Fig. 5               qualitative case study
  RQ4  Fig. 6(a)            lambda sensitivity (ES vs EC)
       Fig. 6(b)(c)         Top-K sensitivity (AT, SR@20)
  RQ5  Fig. 6(d)            response time (ms per turn, s per conversation)

Performance and explainability run through crs_bench.py, the harness every
baseline (run_baselines.py) uses, so the comparison is on identical
conversations. SR@20 requires conversations of up to 20 rounds.
"""
import argparse
import os
import pickle

import numpy as np
import torch

from cecr_torch import (CECRModel, UserSimulator, build_graph_tensors,
                        candidate_set, counterfactual_explanation)
from crs_bench import (CECRAgent, evaluate, explanation_states, liked_attr_profile,
                       test_pairs as make_test_pairs)
from crs_bench import eval_explainability as bench_explainability

OUT = "results/movielens"
MAX_ROUNDS = 20
TOPK = 10


def load_model(path, d):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CECRModel(d["n_users"], d["n_items"], d["n_attrs"],
                  dim=ck["dim"], gate_sign=ck.get("gate_sign", 1),
                  use_social=ck.get("use_social", False))
    # strict=False so checkpoints trained before the social branch (Eqs. 4-5)
    # existed still load; those weights are unused when use_social is False.
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {list(unexpected)}")
    if missing and not all(("att_s" in k or "W_S" in k) for k in missing):
        raise RuntimeError(f"checkpoint is missing non-social weights: {list(missing)}")
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
# RQ1 / RQ2 / RQ3 run through the shared harness (crs_bench.py), i.e. the very
# same conversation loop, test pairs and Table IV states as every baseline.
def eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, seed=7,
                     use_pref_refine=True, use_score_decision=True, topk=TOPK):
    agent = CECRAgent(model, e_u, e_i, e_a, attr_matrix,
                      use_pref_refine=use_pref_refine,
                      use_score_decision=use_score_decision)
    return evaluate(agent, test_pairs, attr_matrix, seed=seed,
                    max_rounds=MAX_ROUNDS, topk=topk)


def eval_explainability(model, d, attr_matrix, e_u, e_i, e_a, states, profile,
                        lam=100.0, random_expl=False):
    """Eq. 21 (Precision/Recall) and Eq. 22 (PN/PS); see crs_bench for the
    protocol. `Random` explains with random attributes against CECR's ranker."""
    agent = CECRAgent(model, e_u, e_i, e_a, attr_matrix)
    agent.explain = lambda st, item, kth: CECRAgent.explain(agent, st, item, kth, lam=lam)
    return bench_explainability(agent, states, attr_matrix, profile,
                                rng=np.random.default_rng(7), random_expl=random_expl)


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
def show(name, r):
    print(f"  {name:12s} SR@5={r['SR@5']:.3f} SR@10={r['SR@10']:.3f} "
          f"SR@20={r['SR@20']:.3f} AT={r['AT']:.2f}  ({r['ms_per_turn']:.1f} ms/turn)",
          flush=True)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/cecr_full.pt")
    ap.add_argument("--model-noucf", default="data/cecr_nocf.pt",
                    help="separately trained model for the CouRe ablation")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-expl", type=int, default=100)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--data", default="data/prepared.pkl")
    ap.add_argument("--out", default="results/movielens")
    ap.add_argument("--skip-expl", action="store_true",
                    help="skip Table IV (published for MovieLens and Yelp only)")
    ap.add_argument("--skip-lambda", action="store_true", help="skip Fig. 6(a)")
    ap.add_argument("--skip-ablations", action="store_true",
                    help="evaluate only the full model (Fig. 4 is MovieLens/Yelp only)")
    ap.add_argument("--topk-sweep", action="store_true",
                    help="Fig. 6(b)(c): AT and SR@20 for Top-K in {1,5,10,20,50}")
    args = ap.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    with open(args.data, "rb") as f:
        d = pickle.load(f)
    attr_matrix = d["attr_matrix"]
    graph = build_graph_tensors(d)
    model = load_model(args.model, d)
    test_pairs = make_test_pairs(d, args.n_test)
    e_u, e_i, e_a = embeddings(model, graph)
    out = dict(results={}, expl={}, lam=[], topk=[])

    # ---------------- RQ1 / RQ2 (Table II, Fig. 3, Fig. 4, Fig. 6d) ----------------
    print("[RQ1/RQ2] recommendation performance + ablations", flush=True)
    R = out["results"]
    R["CECR (CCR)"] = eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs)
    show("CCR", R["CECR (CCR)"])
    if not args.skip_ablations:
        R["- PreRi"] = eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs,
                                        use_pref_refine=False)
        show("- PreRi", R["- PreRi"])
        R["- RecPe"] = eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs,
                                        use_score_decision=False)
        show("- RecPe", R["- RecPe"])
        gu, gi, ga = embeddings(model, graph, "gnns_off")
        R["- GNNs"] = eval_performance(model, d, attr_matrix, gu, gi, ga, test_pairs)
        show("- GNNs", R["- GNNs"])
        if os.path.exists(args.model_noucf):
            m2 = load_model(args.model_noucf, d)
            e_u2, e_i2, e_a2 = embeddings(m2, graph)
            R["- CouRe"] = eval_performance(m2, d, attr_matrix, e_u2, e_i2, e_a2, test_pairs)
            show("- CouRe", R["- CouRe"])
    with open(f"{OUT}/cecr_table2.csv", "w") as f:
        f.write("Model,SR@5,SR@10,SR@20,AT,ms_per_turn,sec_per_conv\n")
        for k, v in R.items():
            f.write(f"{k},{v['SR@5']:.4f},{v['SR@10']:.4f},{v['SR@20']:.4f},{v['AT']:.2f},"
                    f"{v['ms_per_turn']:.2f},{v['sec_per_conv']:.4f}\n")

    # ---------------- RQ4: Top-K (Fig. 6b, 6c) ----------------
    if args.topk_sweep:
        print("[RQ4] Top-K sweep (Fig. 6b/6c)", flush=True)
        for k in (1, 5, 10, 20, 50):
            r = eval_performance(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, topk=k)
            out["topk"].append((k, r["AT"], r["SR@20"]))
            print(f"  K={k:>2}  AT={r['AT']:.2f}  SR@20={r['SR@20']:.3f}", flush=True)
        with open(f"{OUT}/fig6bc_topk.csv", "w") as f:
            f.write("K,AT,SR@20\n")
            for k, at, sr in out["topk"]:
                f.write(f"{k},{at:.3f},{sr:.4f}\n")

    # ---------------- RQ3 (Table IV) ----------------
    if not args.skip_expl:
        print("[RQ3] explainability (Table IV)", flush=True)
        states = explanation_states(d, attr_matrix, test_pairs, args.n_expl)
        profile = liked_attr_profile(d, attr_matrix)
        out["expl"]["CECR"] = eval_explainability(model, d, attr_matrix, e_u, e_i, e_a,
                                                  states, profile, lam=args.lam)
        out["expl"]["Random"] = eval_explainability(model, d, attr_matrix, e_u, e_i, e_a,
                                                    states, profile, random_expl=True)
        for k, v in out["expl"].items():
            print(f"  {k:6s}", {kk: round(vv, 4) for kk, vv in v.items()}, flush=True)
        with open(f"{OUT}/cecr_table4.csv", "w") as f:
            f.write("Model,Precision,Recall,PN,PS,n\n")
            for k, v in out["expl"].items():
                f.write(f"{k},{v['Precision']:.4f},{v['Recall']:.4f},{v['PN']:.4f},"
                        f"{v['PS']:.4f},{v['n']}\n")

    # ---------------- RQ4: lambda (Fig. 6a) ----------------
    if not args.skip_lambda:
        print("[RQ4] lambda sensitivity (Fig. 6a)", flush=True)
        out["lam"] = eval_lambda(model, d, attr_matrix, e_u, e_i, e_a, test_pairs,
                                 lambdas=(0, 1, 6, 10, 25, 50, 75, 100))
        with open(f"{OUT}/fig6a_lambda.csv", "w") as f:
            f.write("lambda,ES,EC\n")
            for lam, es, ec in out["lam"]:
                f.write(f"{lam},{es:.4f},{ec:.4f}\n")

    # ---------------- case study (Fig. 5) ----------------
    if os.path.exists("data/u.item") and d.get("name") is None:
        import pandas as pd
        items = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None)
        titles = {i: t for i, t in zip(items[0].values - 1, items[1].values)}
    else:
        titles = {i: f"item #{i}" for i in range(d["n_items"])}   # no titles shipped
    txt = case_study(model, d, attr_matrix, e_u, e_i, e_a, test_pairs, titles,
                     d["genre_names"])
    with open(f"{OUT}/case_study.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    with open(f"{OUT}/cecr.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
