"""
Train + evaluate every baseline the CECR paper compares against, on one
dataset, through the shared harness (crs_bench.py).

  python run_baselines.py --data data/prepared.pkl        --tag movielens --expl
  python run_baselines.py --data data/prepared_yelp.pkl   --tag yelp      --expl
  python run_baselines.py --data data/prepared_douban.pkl --tag douban

Writes results/<tag>/baselines.pkl and results/<tag>/baselines_table2.csv
(+ baselines_table4.csv with --expl; the paper reports Table IV for Yelp and
MovieLens only).
"""
import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np
import torch

from baselines import (AttrFM, CAECAgent, CAECNet, CPRAgent, EARAgent, FPANAgent,
                       FPANGates, KBRD, KBRDAgent, KGEmb, SAURAgent, SAURNet,
                       UnicornAgent, positives, train_caec, train_fm, train_fpan,
                       train_kbrd, train_kg, train_policy_reinforce, train_saur,
                       train_unicorn)
from crs_bench import (evaluate, eval_explainability, explanation_states,
                       liked_attr_profile, test_pairs)

# reduced-epoch budget per dataset (declared in the report)
BUDGET = {
    "movielens": dict(fm=20, kg=20, kbrd=1500, gate_ep=10, rl=1000, dqn=4000),
    "yelp":      dict(fm=10, kg=10, kbrd=2000, gate_ep=10, rl=1000, dqn=3000),
    "douban":    dict(fm=5,  kg=5,  kbrd=2000, gate_ep=8,  rl=1000, dqn=3000),
}


def log(*a):
    print(*a, flush=True)


def show(name, r):
    log(f"  {name:8s} SR@5={r['SR@5']:.3f} SR@10={r['SR@10']:.3f} SR@20={r['SR@20']:.3f} "
        f"AT={r['AT']:.2f}  ({r['ms_per_turn']:.1f} ms/turn, {r['sec_per_conv']:.3f} s/conv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True, choices=list(BUDGET))
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--models", default="EAR,FPAN,UNICORN,KBRD")
    ap.add_argument("--expl", action="store_true", help="also run Table IV (CPR/CAEC/SAUR)")
    ap.add_argument("--n-expl", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="tiny budget, for testing the code")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    B = dict(BUDGET[args.tag])
    if args.smoke:
        B = dict(fm=1, kg=1, kbrd=20, gate_ep=1, rl=30, dqn=30)
        args.n_test, args.n_expl = 30, 10
    out = f"results/{args.tag}"
    os.makedirs(out, exist_ok=True)

    with open(args.data, "rb") as f:
        d = pickle.load(f)
    am = d["attr_matrix"]
    n_users, n_items = d["n_users"], d["n_items"]
    _, _, hist = positives(d)
    pairs = test_pairs(d, args.n_test)
    val_pairs = test_pairs(d, 100, seed=1, split="val")
    tu, ti, _ = positives(d)
    train_pairs = list(zip(tu.tolist(), ti.tolist()))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    log(f"dataset={args.tag} users={n_users} items={n_items} attrs={am.shape[1]} "
        f"test conversations={len(pairs)} budget={B}")

    res_path = f"{out}/baselines.pkl"
    results = {}                      # only what this run computes

    def load_saved():
        if os.path.exists(res_path) and not args.smoke:
            with open(res_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save():
        # re-read before writing, so runs of different models on the same
        # dataset can go in parallel without overwriting each other
        if args.smoke:
            return
        merged = load_saved()
        merged.update(results)
        with open(res_path, "wb") as f:
            pickle.dump(merged, f)

    t_all = time.time()
    need_fm = any(m in models for m in ("EAR", "FPAN")) or args.expl
    fm0 = None
    if need_fm:
        log("[FM] attribute-aware FM (EAR estimation backbone)")
        fm0 = train_fm(AttrFM(n_users, n_items, am), d, am, epochs=B["fm"], seed=args.seed, log=log)

    if "EAR" in models:
        log("[EAR] policy network: max-entropy pre-training + REINFORCE")
        ag = EARAgent(copy.deepcopy(fm0), am, hist, reflection=True)
        train_policy_reinforce(ag, train_pairs, am, episodes=B["rl"], seed=args.seed, log=log)
        results["EAR"] = evaluate(ag, pairs, am); show("EAR", results["EAR"]); save()

    if "FPAN" in models:
        log("[FPAN] gated online-feedback adaptation + EAR policy")
        fm = copy.deepcopy(fm0)
        gates = FPANGates()
        train_fpan(fm, gates, d, am, epochs=B["gate_ep"], seed=args.seed, log=log)
        ag = FPANAgent(fm, gates, am, hist)
        train_policy_reinforce(ag, train_pairs, am, episodes=B["rl"], seed=args.seed, log=log)
        results["FPAN"] = evaluate(ag, pairs, am); show("FPAN", results["FPAN"]); save()

    if "UNICORN" in models:
        log("[UNICORN] TransE graph pre-training + graph-based dueling double DQN")
        kg = KGEmb(n_users, n_items, am.shape[1])
        train_kg(kg, d, am, epochs=B["kg"], seed=args.seed, log=log)
        ag = UnicornAgent(kg, am)
        train_unicorn(ag, train_pairs, am, episodes=B["dqn"], lr=1e-3, seed=args.seed, log=log)
        results["UNICORN"] = evaluate(ag, pairs, am); show("UNICORN", results["UNICORN"]); save()

    if "KBRD" in models:
        log("[KBRD] R-GCN knowledge-graph recommender")
        kb = KBRD(n_items, am, d["genre_names"])
        train_kbrd(kb, d, am, steps=B["kbrd"], seed=args.seed, log=log)
        best = None
        for tau in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8):             # tuned on validation
            r = evaluate(KBRDAgent(kb, am, hist, tau=tau), val_pairs, am)
            log(f"  tau={tau}: val SR@10={r['SR@10']:.3f} AT={r['AT']:.2f}")
            if best is None or r["AT"] < best[1]:
                best = (tau, r["AT"])
        log(f"  chosen tau={best[0]}")
        results["KBRD"] = evaluate(KBRDAgent(kb, am, hist, tau=best[0]), pairs, am)
        results["KBRD"]["tau"] = best[0]
        show("KBRD", results["KBRD"]); save()

    all_res = {**load_saved(), **results}
    with open(f"{out}/baselines_table2.csv", "w") as f:
        f.write("Model,SR@5,SR@10,SR@20,AT,ms_per_turn,sec_per_conv\n")
        for k in ("EAR", "FPAN", "UNICORN", "KBRD"):
            if k in all_res:
                v = all_res[k]
                f.write(f"{k},{v['SR@5']:.4f},{v['SR@10']:.4f},{v['SR@20']:.4f},{v['AT']:.2f},"
                        f"{v['ms_per_turn']:.2f},{v['sec_per_conv']:.4f}\n")

    if args.expl:
        log("[Table IV] explainable CRS baselines")
        states = explanation_states(d, am, pairs, args.n_expl)
        profile = liked_attr_profile(d, am)
        mem = {u: h[:30] for u, h in hist.items()}
        ex = {}
        ex["CPR"] = eval_explainability(CPRAgent(fm0, am), states, am, profile)
        log("  CPR ", ex["CPR"])
        caec = CAECNet(copy.deepcopy(fm0), am.shape[1])
        train_caec(caec, d, am, epochs=B["gate_ep"], seed=args.seed, log=log)
        ex["CAEC"] = eval_explainability(CAECAgent(caec, am), states, am, profile)
        log("  CAEC", ex["CAEC"])
        saur = SAURNet(copy.deepcopy(fm0))
        train_saur(saur, d, am, mem, epochs=B["gate_ep"], seed=args.seed, log=log)
        ex["SAUR"] = eval_explainability(SAURAgent(saur, am, mem), states, am, profile)
        log("  SAUR", ex["SAUR"])
        results["table4"] = ex
        save()
        with open(f"{out}/baselines_table4.csv", "w") as f:
            f.write("Model,Precision,Recall,PN,PS,n\n")
            for k, v in ex.items():
                f.write(f"{k},{v['Precision']:.4f},{v['Recall']:.4f},{v['PN']:.4f},"
                        f"{v['PS']:.4f},{v['n']}\n")
    log(f"done in {time.time()-t_all:.0f}s -> {out}/")


if __name__ == "__main__":
    sys.exit(main())
