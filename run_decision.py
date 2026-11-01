"""
Improvement #3: compare CECR's ask-vs-recommend rules on the shared harness.

  Eq. 20 (paper)   dot-product comparison, as in eval_cecr.py
  cosine           Eq. 20 on cosine similarities
  policy           learned policy: imitation of Eq. 20, then REINFORCE
  threshold        ask (preference-weighted entropy) while > T candidates
                   remain; T chosen on validation conversations

Only the decision changes; ranking, attribute choice and explanations are
CECR's own. The policy trains on training-split conversations and is selected
on validation conversations, so the test pairs are never seen.

  python run_decision.py --data data/prepared.pkl --model data/cecr_full.pt --tag movielens
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from baselines import ACTIONS, REG, positives
from crs_bench import evaluate, run_conversation, test_pairs
from decision import CECRDecisionAgent
from eval_cecr import load_model


def show(name, r):
    print(f"  {name:<14} SR@5={r['SR@5']:.3f} SR@10={r['SR@10']:.3f} "
          f"SR@20={r['SR@20']:.3f} AT={r['AT']:.2f}  ask={r['ask_rate']:.2f}  "
          f"({r['ms_per_turn']:.1f} ms/turn)", flush=True)


def evaluate_counting(agent, pairs, am):
    """crs_bench.evaluate plus the share of decisions that were questions."""
    n = {"ask": 0, "recommend": 0}
    decide = agent.decide

    def counting(st, cands, cand_attrs, rng):
        a = decide(st, cands, cand_attrs, rng)
        n[a] += 1
        return a

    agent.decide = counting
    try:
        r = evaluate(agent, pairs, am)
    finally:
        del agent.decide
    r["ask_rate"] = n["ask"] / max(sum(n.values()), 1)
    return r


def train_policy(agent, train_pairs, val_pairs, am, episodes, pretrain, gamma, lr,
                 eval_every, seed, ent_bonus=0.01):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(agent.policy.parameters(), lr=lr, weight_decay=REG)

    # --- imitation of Eq. 20, so REINFORCE starts from the paper's behaviour ---
    X, Y = [], []
    agent.mode = "rule"
    rule_decide = agent.decide

    def recording_decide(st, cands, cand_attrs, r):
        a = rule_decide(st, cands, cand_attrs, r)
        X.append(agent.features(st, cands, cand_attrs))
        Y.append(ACTIONS.index(a))
        return a

    agent.decide = recording_decide
    for _ in range(pretrain):
        u, t = train_pairs[int(rng.integers(len(train_pairs)))]
        run_conversation(agent, u, t, am, rng)
    del agent.decide
    Xt, Yt = torch.stack(X), torch.as_tensor(Y)
    for _ in range(300):
        # label smoothing keeps p(ask) away from 0: Eq. 20 almost never asks,
        # and a policy that never samples "ask" can never learn when it helps
        loss = F.cross_entropy(agent.policy(Xt), Yt, label_smoothing=0.3)
        opt.zero_grad(); loss.backward(); opt.step()
    agent.mode = "policy"
    best = evaluate_counting(agent, val_pairs, am)
    best_state = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    print(f"  [imitation] {len(X)} decisions  val SR@20={best['SR@20']:.3f} AT={best['AT']:.2f}",
          flush=True)

    # --- REINFORCE on the harness rewards (crs_bench.REWARDS) ---
    t0 = time.time()
    for e in range(episodes):
        agent.training = True
        u, t = train_pairs[int(rng.integers(len(train_pairs)))]
        run_conversation(agent, u, t, am, rng)
        if agent.logps:
            G, ret = 0.0, []
            for rw in reversed(agent.rewards[:len(agent.logps)]):
                G = rw + gamma * G
                ret.insert(0, G)
            ret = torch.as_tensor(ret, dtype=torch.float32)
            if len(ret) > 1:
                ret = (ret - ret.mean()) / (ret.std() + 1e-6)
            loss = -(torch.stack(agent.logps[:len(ret)]) * ret).sum() \
                - ent_bonus * torch.stack(agent.ents).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if (e + 1) % eval_every == 0:
            agent.training = False
            v = evaluate_counting(agent, val_pairs, am)
            better = (v["AT"], -v["SR@20"]) < (best["AT"], -best["SR@20"])
            if better:
                best = v
                best_state = {k: x.clone() for k, x in agent.policy.state_dict().items()}
            print(f"  [REINFORCE] ep {e+1}/{episodes} val SR@20={v['SR@20']:.3f} "
                  f"AT={v['AT']:.2f} ask={v['ask_rate']:.2f}{'  *' if better else ''} ({time.time()-t0:.0f}s)",
                  flush=True)
    agent.training = False
    agent.policy.load_state_dict(best_state)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared.pkl")
    ap.add_argument("--model", default="data/cecr_full.pt")
    ap.add_argument("--tag", default="movielens")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--pretrain", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick", default="entropy", choices=["pref", "entropy"],
                    help="attribute choice for the learned policy / threshold rule")
    ap.add_argument("--thresholds", default="10,25,50,100,200,500,1000")
    ap.add_argument("--only-threshold", action="store_true",
                    help="run just the threshold rule, merging into existing results")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    with open(args.data, "rb") as f:
        d = pickle.load(f)
    am = d["attr_matrix"]
    model = load_model(args.model, d)
    e_u, e_i, e_a = model.e_u_cache, model.e_i_cache, model.e_a_cache
    print(f"dataset={args.tag}  |e_i|={e_i.norm(dim=1).mean():.2f}  "
          f"|e_a|={e_a.norm(dim=1).mean():.2f}  |e_u|={e_u.norm(dim=1).mean():.2f}")

    pairs = test_pairs(d, args.n_test)
    val_pairs = test_pairs(d, 100, seed=1, split="val")
    tu, ti, _ = positives(d)
    train_pairs = list(zip(tu.tolist(), ti.tolist()))

    out = f"results/{args.tag}"
    os.makedirs(out, exist_ok=True)
    res_path = f"{out}/decision_results.json"
    results = {}
    if args.only_threshold and os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)

    # threshold rule: T picked on validation by AT (ties -> higher SR@20)
    best = None
    for T in [int(x) for x in args.thresholds.split(",")]:
        ag = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="threshold",
                               pick=args.pick, threshold=T)
        v = evaluate_counting(ag, val_pairs, am)
        print(f"  [threshold] T={T:<5} val SR@20={v['SR@20']:.3f} AT={v['AT']:.2f} "
              f"ask={v['ask_rate']:.2f}", flush=True)
        if best is None or (v["AT"], -v["SR@20"]) < (best[1]["AT"], -best[1]["SR@20"]):
            best = (T, v)
    ag = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="threshold",
                           pick=args.pick, threshold=best[0])
    results["threshold"] = evaluate_counting(ag, pairs, am)
    results["threshold"]["T"] = best[0]
    show(f"threshold T={best[0]}", results["threshold"])
    if args.only_threshold:
        with open(res_path, "w") as f:
            json.dump(results, f, indent=1)
        print(f"written to {res_path}")
        return

    for name in ("eq20", "cosine"):
        ag = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision=name)
        results[name] = evaluate_counting(ag, pairs, am)
        show(name, results[name])

    ag = CECRDecisionAgent(model, e_u, e_i, e_a, am, decision="policy", pick=args.pick)
    train_policy(ag, train_pairs, val_pairs, am, args.episodes, args.pretrain,
                 args.gamma, args.lr, args.eval_every, args.seed)
    results["policy"] = evaluate_counting(ag, pairs, am)
    show("policy", results["policy"])

    with open(res_path, "w") as f:
        json.dump(results, f, indent=1)
    torch.save(ag.policy.state_dict(), f"{out}/decision_policy.pt")
    print(f"written to {out}/decision_results.json")


if __name__ == "__main__":
    main()
