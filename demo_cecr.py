"""
Live turn-by-turn demo of the TRAINED CECR replica (the PyTorch one).

The older demo_conversation.py / interactive_demo.py scripts drive the original
NumPy pipeline (cecr.py + data/base_emb.npz), which is the weaker model. This
one drives the actual trained replica and prints every stage of Algorithm 1 as
it happens, so an audience can watch the four components work together:

    GNN offline representations (Eqs. 1-8)
      -> preference refinement through the trained gates (Eqs. 9-13)
      -> ask-vs-recommend decision (Eq. 20)
      -> counterfactual explanation on rejection (Eqs. 15-18)

Usage:
    python demo_cecr.py                      # random conversation, MovieLens
    python demo_cecr.py --seed 7             # reproduce a specific one
    python demo_cecr.py --data data/prepared_douban.pkl --model data/douban_full.pt
"""
import argparse
import os
import pickle

import numpy as np
import torch

from cecr_torch import (CECRModel, UserSimulator, candidate_set,
                        counterfactual_explanation, build_graph_tensors)

TOPK = 10


def load(args):
    with open(args.data, "rb") as f:
        d = pickle.load(f)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    m = CECRModel(d["n_users"], d["n_items"], d["n_attrs"], dim=ck["dim"],
                  gate_sign=ck.get("gate_sign", 1),
                  use_social=ck.get("use_social", False))
    # strict=False: checkpoints trained before the social branch (Eqs. 4-5) was
    # added lack gnn.att_s / gnn.W_S. Those parameters are unused whenever
    # use_social is False, so tolerating them keeps older MovieLens
    # checkpoints loadable.
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    unexpected = [k for k in unexpected]
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    if missing and not all(("att_s" in k or "W_S" in k) for k in missing):
        raise RuntimeError(f"checkpoint is missing non-social weights: {missing}")
    m.eval()
    return d, m


def item_names(d):
    if d.get("name") is None and os.path.exists("data/u.item"):
        import pandas as pd
        it = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None)
        return {i: t for i, t in zip(it[0].values - 1, it[1].values)}
    return {i: f"item #{i}" for i in range(d["n_items"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared.pkl")
    ap.add_argument("--model", default="data/cecr_full.pt")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=20)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 10_000))
    rng = np.random.default_rng(seed)
    d, model = load(args)
    attr_matrix = d["attr_matrix"]
    names = item_names(d)
    attrs = d["genre_names"]
    sim = UserSimulator(attr_matrix)
    e_u, e_i, e_a = model.e_u_cache, model.e_i_cache, model.e_a_cache
    n_items = d["n_items"]

    # pick a held-out (user, liked item) pair
    liked = d["test"][d["test"].rating >= 4][["user", "item"]].values
    u, tgt = (int(x) for x in liked[rng.integers(0, len(liked))])
    t_attrs = sim.attrs_of(tgt)
    while not t_attrs:
        u, tgt = (int(x) for x in liked[rng.integers(0, len(liked))])
        t_attrs = sim.attrs_of(tgt)

    print(f"(seed={seed} - rerun with --seed {seed} to reproduce this conversation)\n")
    print(f"=== Simulated user #{u} | target (ground truth): {names[tgt]!r}")
    print(f"    its attributes: {[attrs[a] for a in sorted(t_attrs)]}\n")

    first = int(rng.choice(sorted(t_attrs)))
    pos_attrs, neg_attrs, asked, rejected = {first}, set(), {first}, []
    print(f"[Turn 0] User: I'm looking for a '{attrs[first]}' one.")

    for rnd in range(1, args.max_rounds + 1):
        cands = candidate_set(attr_matrix, pos_attrs, neg_attrs, rejected, n_items)
        if len(cands) == 0:
            print("\n[System] no candidates left.")
            break
        with torch.no_grad():
            eo = model.overall(e_u[u], pos_attrs, neg_attrs, rejected, e_a, e_i)
        present = np.nonzero(attr_matrix[cands].sum(0) > 0)[0]
        cand_attrs = [int(a) for a in present if a not in asked]
        with torch.no_grad():
            action, s_i, s_a = model.decide_action(eo, cands, cand_attrs, e_i, e_a)
        if not cand_attrs:
            action = "recommend"

        print(f"\n--- Round {rnd} ---   ({len(cands)} candidates left)")
        print(f"[Eq.20] best item score = {float(s_i):+.3f} vs "
              f"best attribute score = {float(s_a):+.3f}  ->  {action.upper()}")

        if action == "ask":
            with torch.no_grad():
                a_star = int(cand_attrs[int(torch.argmax(e_a[cand_attrs] @ eo))])
            asked.add(a_star)
            yes = a_star in t_attrs
            print(f"[System] Do you like '{attrs[a_star]}'?")
            print(f"[User]   {'Yes, I do.' if yes else 'No, I do not.'}")
            (pos_attrs if yes else neg_attrs).add(a_star)
            continue

        with torch.no_grad():
            scores = model.rank(eo, e_i[cands])                # Eq. 14
            order = torch.topk(scores, min(TOPK, len(cands))).indices.cpu().numpy()
        top = cands[order]
        print(f"[System] Top-{len(top)}: {[names[int(x)] for x in top[:3]]} ...")

        if tgt in top:
            print(f"[System] How about {names[tgt]!r}?")
            print("[User]   Yes! That's the one, thanks.")
            print(f"\n*** SUCCESS in round {rnd} ***")
            return

        it = int(top[0])
        print(f"[System] How about {names[it]!r}? "
              f"(attrs: {[attrs[a] for a in np.nonzero(attr_matrix[it])[0]][:4]})")
        print("[User]   No, I don't like this one.")

        outside = np.setdiff1d(cands, top)
        if len(outside):
            with torch.no_grad():
                s_ik = model.rank(eo, e_i[outside]).max()
            dlt, ES, EC, expl, _ = counterfactual_explanation(
                model, eo, e_i[it], torch.as_tensor(attr_matrix[it]), e_a,
                s_ik, lam=100.0, iters=80)
            if len(expl):
                mags = np.abs(dlt[torch.as_tensor(expl)].numpy())
                best = int(expl[int(np.argmax(mags))])
                print(f"   [XAI] counterfactual explanation for the rejection:")
                print(f"         most influential attribute: '{attrs[best]}' "
                      f"(delta = {float(dlt[best]):+.3f})")
                print(f"         ES = {ES:.3f} (Eq.15), EC = {EC} attribute(s) (Eq.17)")
                print(f"         -> rejected mainly because of '{attrs[best]}'; "
                      f"a minimal change there would flip the decision.")
        rejected += [int(x) for x in top]

    print(f"\n*** no success within {args.max_rounds} rounds ***")


if __name__ == "__main__":
    main()
