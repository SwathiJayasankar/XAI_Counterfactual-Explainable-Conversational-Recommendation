"""
Improvement #1: amortised counterfactual explanations.

The paper (Eqs. 17-18) finds Delta by running gradient descent separately for
every explained item, which is what makes CECR the slowest model per turn.
Here a small network predicts Delta in one forward pass from
(e_overall, e_item, the item's attribute mask, the K-th score), trained on the
very same Eq. 17 objective over training-split conversation states:

    ||Delta||_2^2 + ||Delta||_1 + lambda * softplus(s_{u,i_Delta} - s_{u,i_K})

The CECR model stays frozen; only the Delta network is learned. "refine" mode
warm-starts the paper's optimiser from the predicted Delta for a few steps.

  python cf_amortized.py --data data/prepared.pkl --model data/cecr_full.pt --tag movielens
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import positives
from cecr_torch import candidate_set, counterfactual_explanation
from crs_bench import (TOPK, CECRAgent, eval_explainability, explanation_states,
                       liked_attr_profile, test_pairs)
from eval_cecr import load_model

EXPL_THRESH = 0.05          # |Delta_a| above this counts as part of the explanation


class DeltaNet(nn.Module):
    def __init__(self, dim, n_attrs, hidden=256):
        super().__init__()
        # LayerNorm on the input: user embeddings have norm ~4 and scores are
        # unbounded, which otherwise saturates the tanh from the first step
        # (zero gradient, Delta stuck at +-1 on every attribute)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * dim + n_attrs + 2),
            nn.Linear(3 * dim + n_attrs + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_attrs))
        nn.init.zeros_(self.net[-1].weight)    # start at Delta = 0, like Eq. 18's optimiser
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, eo, ei, mask, s_i, s_k):
        x = torch.cat([eo, ei, eo * ei, mask, s_i.unsqueeze(-1), (s_i - s_k).unsqueeze(-1)], -1)
        return torch.tanh(self.net(x)) * mask          # Delta in [-1, 1], own attributes only


def cf_loss(model, d, eo, ei, e_a, s_k, lam):
    """Eq. 17 per sample (l0 relaxed to l1 as in the paper)."""
    s_d = model.rank(eo, ei - d @ e_a)
    return (d ** 2).sum(-1) + d.abs().sum(-1) + lam * F.softplus(s_d - s_k)


@torch.no_grad()
def build_samples(model, states, am, e_u, e_i, e_a):
    """(e_overall, e_item, mask, s_item, s_K) for every Top-K item of every state."""
    EO, EI, M, SI, SK = [], [], [], [], []
    for st, _, cands in states:
        eo = model.overall(e_u[st.u], st.pos, st.neg, st.rejected, e_a, e_i)
        sc = model.rank(eo, e_i[cands])
        top = torch.topk(sc, TOPK)
        kth = top.values[-1]
        for it, s in zip(cands[top.indices.numpy()], top.values):
            if am[it].sum() == 0:
                continue
            EO.append(eo); EI.append(e_i[it]); M.append(torch.as_tensor(am[it], dtype=torch.float32))
            SI.append(s); SK.append(kth)
    return [torch.stack(x) for x in (EO, EI, M)] + [torch.stack(SI), torch.stack(SK)]


def train_deltanet(model, e_a, samples, lam, epochs=30, bs=256, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    EO, EI, M, SI, SK = samples
    net = DeltaNet(EO.shape[1], M.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(EO)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            d = net(EO[b], EI[b], M[b], SI[b], SK[b])
            loss = cf_loss(model, d, EO[b], EI[b], e_a, SK[b], lam).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 5 == 0:
            log(f"  [DeltaNet] epoch {ep+1}/{epochs}  Eq.17 loss={tot / n:.4f}")
    net.eval()
    return net


def refine(model, eo, ei, mask, e_a, s_k, d0, lam, iters, lr=0.05):
    """The paper's Eq. 18 optimiser, warm-started from the predicted Delta."""
    delta = d0.clone().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(iters):
        d = delta * mask
        loss = cf_loss(model, d, eo, ei, e_a, s_k, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            delta.clamp_(-1.0, 1.0)
    return (delta * mask).detach()


class AmortizedAgent(CECRAgent):
    """CECR agent whose explain() uses DeltaNet (optionally + a few refine steps)."""

    def __init__(self, model, e_u, e_i, e_a, am, net, lam, refine_iters=0):
        super().__init__(model, e_u, e_i, e_a, am)
        self.net, self.lam, self.refine_iters = net, lam, refine_iters
        self.stats = []                       # (ES, EC) per explanation

    def explain(self, st, item, kth_score):
        mask = torch.as_tensor(self.attr_matrix[item], dtype=torch.float32)
        ei, s_k = self.e_i[item], torch.as_tensor(kth_score, dtype=torch.float32)
        with torch.no_grad():
            s_i = self.m.rank(self.eo, ei).squeeze()
            d = self.net(self.eo, ei, mask, s_i, s_k)
        if self.refine_iters:
            d = refine(self.m, self.eo, ei, mask, self.e_a, s_k, d, self.lam, self.refine_iters)
        with torch.no_grad():
            ES = float(s_i - self.m.rank(self.eo, ei - d @ self.e_a).squeeze())
        expl = torch.nonzero(d.abs() > EXPL_THRESH).flatten().tolist()
        self.stats.append((ES, len(expl)))
        return expl


class TimedPaperAgent(CECRAgent):
    """The paper's per-item optimiser (Eqs. 17-18), recording ES / EC."""

    def __init__(self, *a, lam=100.0, iters=60, **k):
        super().__init__(*a, **k)
        self.lam, self.iters, self.stats = lam, iters, []

    def explain(self, st, item, kth_score):
        _, ES, EC, expl, _ = counterfactual_explanation(
            self.m, self.eo, self.e_i[item], torch.as_tensor(self.attr_matrix[item]),
            self.e_a, torch.as_tensor(kth_score), lam=self.lam, iters=self.iters)
        self.stats.append((ES, EC))
        return [int(a) for a in expl]


def timed_explainability(agent, states, am, profile):
    t0 = time.perf_counter()
    r = eval_explainability(agent, states, am, profile, rng=np.random.default_rng(7))
    sec = time.perf_counter() - t0
    r["ms_per_explanation"] = 1000 * sec / max(r["n"], 1)
    r["ES"] = float(np.mean([s[0] for s in agent.stats])) if agent.stats else 0.0
    r["EC"] = float(np.mean([s[1] for s in agent.stats])) if agent.stats else 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared.pkl")
    ap.add_argument("--model", default="data/cecr_full.pt")
    ap.add_argument("--tag", default="movielens")
    ap.add_argument("--n-train-states", type=int, default=3000)
    ap.add_argument("--n-expl", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.data, "rb") as f:
        d = pickle.load(f)
    am = d["attr_matrix"]
    model = load_model(args.model, d)
    e_u, e_i, e_a = model.e_u_cache, model.e_i_cache, model.e_a_cache

    # training states come from training-split (user, liked item) pairs
    tu, ti, _ = positives(d)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(tu), min(4 * args.n_train_states, len(tu)), replace=False)
    train_states = explanation_states(d, am, [(int(tu[j]), int(ti[j])) for j in sel],
                                      args.n_train_states, seed=args.seed + 100)
    t0 = time.time()
    samples = build_samples(model, train_states, am, e_u, e_i, e_a)
    print(f"dataset={args.tag}  {len(train_states)} train states -> {len(samples[0])} samples "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    net = train_deltanet(model, e_a, samples, args.lam, epochs=args.epochs, seed=args.seed)
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    # Table IV states: identical to eval_cecr.py / run_baselines.py
    pairs = test_pairs(d, args.n_test)
    states = explanation_states(d, am, pairs, args.n_expl)
    profile = liked_attr_profile(d, am)

    results = {}
    runs = [("paper (60 Adam steps)", TimedPaperAgent(model, e_u, e_i, e_a, am, lam=args.lam)),
            ("DeltaNet (1 pass)", AmortizedAgent(model, e_u, e_i, e_a, am, net, args.lam)),
            ("DeltaNet + 5 steps", AmortizedAgent(model, e_u, e_i, e_a, am, net, args.lam, 5))]
    for name, ag in runs:
        r = timed_explainability(ag, states, am, profile)
        results[name] = r
        print(f"  {name:<22} Pr={r['Precision']:.4f} Re={r['Recall']:.4f} PN={r['PN']:.4f} "
              f"PS={r['PS']:.4f}  ES={r['ES']:.3f} EC={r['EC']:.2f}  "
              f"{r['ms_per_explanation']:.2f} ms/expl  (n={r['n']})", flush=True)

    out = f"results/{args.tag}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/cf_amortized_lam{args.lam:g}.json", "w") as f:
        json.dump(results, f, indent=1)
    torch.save(net.state_dict(), f"{out}/deltanet_lam{args.lam:g}.pt")
    print(f"written to {out}/cf_amortized_lam{args.lam:g}.json")


if __name__ == "__main__":
    main()
