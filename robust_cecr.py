"""
Improvement #5: CECR under the robust protocol (seeded test pairs + answer noise),
matching run_baselines.py --robust so the two can be compared pair-for-pair.

For one seed: evaluates CECR with the paper's Eq. 20 decision, with the
learned decision policy (improvement #3, trained with this seed) and with the
entropy-question threshold rule (T chosen on validation) at every noise level. Writes results/<tag>/robust/cecr_seed<seed>.pkl as
{variant: {noise: result}}.

  python robust_cecr.py --data data/prepared.pkl --model data/cecr_full_s1.pt --tag movielens --seed 1
"""
import argparse
import os
import pickle

import numpy as np
import torch

from baselines import positives
from crs_bench import evaluate, test_pairs
from decision import CECRDecisionAgent
from eval_cecr import load_model
from run_decision import evaluate_counting, show, train_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared.pkl")
    ap.add_argument("--model", default="data/cecr_full.pt")
    ap.add_argument("--tag", default="movielens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--noise-levels", default="0,0.1,0.2")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--pick", default="entropy", choices=["pref", "entropy"])
    ap.add_argument("--thresholds", default="10,25,50,100,200,500,1000")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    with open(args.data, "rb") as f:
        d = pickle.load(f)
    am = d["attr_matrix"]
    model = load_model(args.model, d)
    e_u, e_i, e_a = model.e_u_cache, model.e_i_cache, model.e_a_cache

    pairs = test_pairs(d, args.n_test, seed=7 + args.seed)
    val_pairs = test_pairs(d, 100, seed=1, split="val")
    tu, ti, _ = positives(d)
    train_pairs = list(zip(tu.tolist(), ti.tolist()))

    eq20 = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="eq20")
    pol = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="policy", pick=args.pick)
    train_policy(pol, train_pairs, val_pairs, am, args.episodes, pretrain=300, gamma=0.7,
                 lr=1e-3, eval_every=250, seed=args.seed)

    best = None
    for T in [int(x) for x in args.thresholds.split(",")]:
        ag = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="threshold",
                               pick=args.pick, threshold=T)
        v = evaluate_counting(ag, val_pairs, am)
        if best is None or (v["AT"], -v["SR@20"]) < (best[1]["AT"], -best[1]["SR@20"]):
            best = (T, v)
    print(f"  [threshold] chosen T={best[0]} on validation", flush=True)
    thr = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="threshold",
                            pick=args.pick, threshold=best[0])

    results = {}
    for name, ag in (("CECR", eq20), ("CECR+policy", pol), ("CECR+ask", thr)):
        results[name] = {}
        for nz in noise_levels:
            r = evaluate(ag, pairs, am, seed=7 + args.seed, answer_noise=nz)
            r["ask_rate"] = float("nan")
            results[name][nz] = r
            show(f"{name}@{nz}", r)

    out = f"results/{args.tag}/robust"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/cecr_seed{args.seed}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"written to {out}/cecr_seed{args.seed}.pkl")


if __name__ == "__main__":
    main()
