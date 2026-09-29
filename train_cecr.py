"""
Training for the faithful CECR replica.

Follows Algorithm 1's own two-stage structure:
  Stage 1 (Algorithm 1 step 1)  "Initialization: e_u/e_i/e_a (via GNNs)"
        -> train the GNN encoder (Eqs. 1-8) with a BPR ranking loss.
  Stage 2 (Algorithm 1 steps 3-4, 19-24)  train the conversational components:
        the preference-refinement gates (Eqs. 10-11) and the ranking network g
        (Eq. 14), on realistic conversation STATES, with counterfactual
        samples augmenting the training set (Eq. 19).

Paper's settings (Section V-A) are used throughout: embedding size 128,
regularization Omega = 1e-5, Adam, Top-K = 10, lambda = 100.
"""
import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from cecr_torch import (CECRModel, UserSimulator, build_graph_tensors,
                        candidate_set, counterfactual_explanation, run_conversation)

DEVICE = "cpu"


# --------------------------------------------------------------------------
def bpr(pos_scores, neg_scores):
    """BPR: -log sigmoid(s_pos - s_neg)   (Eq. 19's objective)."""
    return F.softplus(neg_scores - pos_scores).mean()


def stage1_train_gnn(model, graph, d, epochs=30, batch=1024, lr=1e-3, reg=1e-5, seed=0):
    """Eqs. 1-8 trained with BPR over the historical interactions O."""
    rng = np.random.default_rng(seed)
    train = d["train"]
    n_items = d["n_items"]
    pos = train[train.rating >= 4]
    users = pos.user.values
    items = pos.item.values
    pos_sets = pos.groupby("user")["item"].apply(set).to_dict()

    opt = torch.optim.Adam(model.gnn.parameters(), lr=lr, weight_decay=reg)
    n = len(users)
    print(f"[Stage 1] GNN offline learning: {n} positive interactions")
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        t0 = time.time()
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            u = torch.as_tensor(users[sel], dtype=torch.long)
            i_pos = torch.as_tensor(items[sel], dtype=torch.long)
            negs = rng.integers(0, n_items, size=len(sel))
            for j, uu in enumerate(users[sel]):          # resample true negatives
                while negs[j] in pos_sets.get(uu, ()):
                    negs[j] = rng.integers(0, n_items)
            i_neg = torch.as_tensor(negs, dtype=torch.long)

            # encode ONLY the nodes in this batch (full-graph forward per step
            # does not scale past MovieLens: Douban-Book has 21k+ items)
            e_u_b = model.gnn.encode_users(graph, u)
            e_ip = model.gnn.encode_items(graph, i_pos)
            e_in_ = model.gnn.encode_items(graph, i_neg)
            s_pos = (e_u_b * e_ip).sum(-1)
            s_neg = (e_u_b * e_in_).sum(-1)
            loss = bpr(s_pos, s_neg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gnn.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"  [Stage 1] epoch {ep+1}/{epochs}  BPR loss = {tot/max(nb,1):.4f}"
              f"  ({time.time()-t0:.1f}s)")
    model.refresh_cache(graph)


# --------------------------------------------------------------------------
def sample_states(d, attr_matrix, rng, batch, n_neg=32, max_pos=3, max_neg_a=2, max_rej=4):
    """
    Build a batch of realistic mid-conversation states from the TRAIN split:
    a user, their liked target item, some confirmed positive attributes of it,
    some rejected attributes, and some already-rejected candidate items.
    This is what the gates (Eqs. 10-11) and g (Eq. 14) actually see at
    conversation time, so it is what they are trained on.
    """
    train = d["train"]
    n_items = d["n_items"]
    pos = train[train.rating >= 4]
    pairs = pos[["user", "item"]].values

    users, targets = [], []
    pos_a, neg_a, rej = [], [], []
    negatives = []

    picks = rng.integers(0, len(pairs), size=batch * 3)
    for p in picks:
        if len(users) >= batch:
            break
        u, tgt = int(pairs[p][0]), int(pairs[p][1])
        t_attrs = np.nonzero(attr_matrix[tgt])[0]
        if len(t_attrs) == 0:
            continue
        k = int(rng.integers(1, min(max_pos, len(t_attrs)) + 1))
        pa = rng.choice(t_attrs, size=k, replace=False).tolist()
        non = np.setdiff1d(np.arange(attr_matrix.shape[1]), t_attrs)
        na = (rng.choice(non, size=int(rng.integers(0, max_neg_a + 1)), replace=False).tolist()
              if len(non) else [])
        cands = candidate_set(attr_matrix, pa, na, [], n_items)
        cands = cands[cands != tgt]
        if len(cands) < 2:
            continue
        nr = int(rng.integers(0, max_rej + 1))
        rj = rng.choice(cands, size=min(nr, len(cands)), replace=False).tolist()
        pool = np.setdiff1d(cands, rj)
        if len(pool) == 0:
            continue
        ng = rng.choice(pool, size=n_neg, replace=len(pool) < n_neg)

        users.append(u); targets.append(tgt)
        pos_a.append(pa); neg_a.append(na); rej.append(rj)
        negatives.append(ng)

    return users, targets, pos_a, neg_a, rej, np.array(negatives)


def padded(lists, device):
    """Pad ragged index lists -> (idx [B,K], mask [B,K])."""
    K = max((len(x) for x in lists), default=0)
    K = max(K, 1)
    idx = torch.zeros(len(lists), K, dtype=torch.long, device=device)
    msk = torch.zeros(len(lists), K, dtype=torch.bool, device=device)
    for b, x in enumerate(lists):
        if len(x):
            idx[b, :len(x)] = torch.as_tensor(list(x), dtype=torch.long, device=device)
            msk[b, :len(x)] = True
    return idx, msk


def masked_mean(emb, idx, msk, dim):
    """Mean over the valid entries only; zeros where nothing is present."""
    vals = emb[idx] * msk.unsqueeze(-1)
    cnt = msk.sum(1, keepdim=True).clamp(min=1)
    return vals.sum(1) / cnt


def batched_overall(model, e_u, e_i, e_a, users, pos_a, neg_a, rej, use_pref_refine=True):
    """Batched Eqs. 9-13."""
    dev = e_u.device
    eu = e_u[torch.as_tensor(users, dtype=torch.long, device=dev)]
    if not use_pref_refine:
        return eu
    pi, pm = padded(pos_a, dev)
    ni, nm = padded(neg_a, dev)
    eA_pos = masked_mean(e_a, pi, pm, model.dim)            # Eq. 9
    eA_neg = masked_mean(e_a, ni, nm, model.dim)
    e_u_prime = model.pref.user(eu, eA_pos, eA_neg)          # Eq. 10

    ri, rm = padded(rej, dev)
    if rm.any():
        e_rej = e_i[ri]                                      # [B,R,d]
        gated = model.pref.item(e_rej, eA_pos.unsqueeze(1).expand_as(e_rej))  # Eq. 11
        gated = gated * rm.unsqueeze(-1)
        cnt = rm.sum(1, keepdim=True).clamp(min=1)
        e_neg_I = gated.sum(1) / cnt                         # Eq. 12
        has = rm.any(1, keepdim=True).float()
        return e_u_prime - e_neg_I * has                     # Eq. 13
    return e_u_prime


def stage2_train_conversational(model, d, attr_matrix, epochs=30, steps=200, batch=128,
                                lr=1e-3, reg=1e-5, seed=0, use_counterfactual=True,
                                lam=100.0, cf_every=25, cf_iters=40, val_users=None):
    """
    Trains the gates (Eqs. 10-11) + ranking net g (Eq. 14) jointly, with the
    materialised embeddings fine-tuned (Algorithm 1 treats the GNN as the
    initialisation for these, steps 1 & 4).
    """
    rng = np.random.default_rng(seed)
    dev = DEVICE

    e_u = torch.nn.Parameter(model.e_u_cache.clone())
    e_i = torch.nn.Parameter(model.e_i_cache.clone())
    e_a = torch.nn.Parameter(model.e_a_cache.clone())

    params = (list(model.pref.parameters()) + list(model.rank.parameters()) + [e_u, e_i, e_a])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=reg)

    print(f"[Stage 2] conversational training: gates + ranking net g "
          f"(counterfactual augmentation {'ON' if use_counterfactual else 'OFF'})")
    for ep in range(epochs):
        tot, nb, cf_used = 0.0, 0, 0
        t0 = time.time()
        for st in range(steps):
            users, targets, pos_a, neg_a, rej, negs = sample_states(
                d, attr_matrix, rng, batch)
            if not users:
                continue
            e_overall = batched_overall(model, e_u, e_i, e_a, users, pos_a, neg_a, rej)
            tgt = torch.as_tensor(targets, dtype=torch.long, device=dev)
            s_pos = model.rank(e_overall, e_i[tgt])
            ng = torch.as_tensor(negs, dtype=torch.long, device=dev)
            s_neg = model.rank(e_overall.unsqueeze(1).expand(-1, ng.shape[1], -1), e_i[ng])
            loss = F.softplus(s_neg - s_pos.unsqueeze(1)).mean()

            # ---- Eq. 19: augment O with counterfactual samples (u, i, i_k) ----
            if use_counterfactual and st % cf_every == 0:
                b = int(rng.integers(0, len(users)))
                if rej[b]:
                    it = int(rej[b][0])
                    with torch.no_grad():
                        eo_b = e_overall[b].detach()
                        pool = negs[b]
                        sc = model.rank(eo_b, e_i[torch.as_tensor(pool, dtype=torch.long)])
                        k_item = int(pool[int(torch.argmax(sc))])
                        s_ik = model.rank(eo_b, e_i[k_item]).squeeze()
                    dlt, ES, EC, expl, _ = counterfactual_explanation(
                        model, eo_b, e_i[it].detach(),
                        torch.as_tensor(attr_matrix[it], device=dev),
                        e_a.detach(), s_ik, lam=lam, iters=cf_iters)
                    if EC > 0:
                        # the rejected item i must rank BELOW the anchor item k
                        s_i = model.rank(e_overall[b], e_i[it])
                        s_k = model.rank(e_overall[b], e_i[k_item])
                        loss = loss + F.softplus(s_i - s_k).mean()
                        cf_used += 1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item()
            nb += 1

        msg = (f"  [Stage 2] epoch {ep+1}/{epochs}  loss = {tot/max(nb,1):.4f}"
               f"  cf_samples = {cf_used}  ({time.time()-t0:.1f}s)")
        if val_users is not None and (ep + 1) % 5 == 0:
            sr = quick_val(model, d, attr_matrix, e_u.detach(), e_i.detach(), e_a.detach(),
                           val_users, seed=seed)
            msg += f"  val SR@10 = {sr:.3f}"
        print(msg)

    with torch.no_grad():
        model.e_u_cache.copy_(e_u.detach())
        model.e_i_cache.copy_(e_i.detach())
        model.e_a_cache.copy_(e_a.detach())


@torch.no_grad()
def quick_val(model, d, attr_matrix, e_u, e_i, e_a, val_pairs, seed=0, max_rounds=10):
    """Small validation probe: SR@10 over a handful of held-out conversations."""
    sim = UserSimulator(attr_matrix)
    rng = np.random.default_rng(seed)
    ok = 0
    for u, tgt in val_pairs:
        r, _ = run_conversation(model, u, tgt, sim, attr_matrix, e_u, e_i, e_a,
                                max_rounds=max_rounds, rng=rng)
        ok += int(r is not None)
    return ok / max(len(val_pairs), 1)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)         # paper: 128
    ap.add_argument("--stage1-epochs", type=int, default=30)
    ap.add_argument("--stage2-epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)       # paper tunes in [1e-4,1e-1]
    ap.add_argument("--reg", type=float, default=1e-5)      # paper: Omega = 1e-5
    ap.add_argument("--lam", type=float, default=100.0)     # paper: lambda = 100
    ap.add_argument("--gate-sign", type=int, default=1)
    ap.add_argument("--no-counterfactual", action="store_true")
    ap.add_argument("--out", default="data/cecr_full.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="reuse a checkpoint's Stage-1 GNN and only rerun Stage 2 "
                         "(used for the CouRe ablation, so it differs from the full "
                         "model ONLY in the counterfactual augmentation)")
    ap.add_argument("--data", default="data/prepared.pkl",
                    help="prepared dataset pickle (MovieLens or Douban-Book)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with open(args.data, "rb") as f:
        d = pickle.load(f)
    attr_matrix = d["attr_matrix"]

    use_social = d.get("social") is not None and len(d.get("social", []))
    print(f"dataset: {d.get('name','movielens')}  users={d['n_users']} "
          f"items={d['n_items']} attrs={d['n_attrs']}  social={'yes' if use_social else 'no'}")
    graph = build_graph_tensors(d, device=DEVICE)
    model = CECRModel(d["n_users"], d["n_items"], d["n_attrs"],
                      dim=args.dim, gate_sign=args.gate_sign,
                      use_social=bool(use_social)).to(DEVICE)

    val = d["val"]
    vl = val[val.rating >= 4][["user", "item"]].values
    rng = np.random.default_rng(1)
    val_pairs = [(int(u), int(i)) for u, i in vl[rng.choice(len(vl), 60, replace=False)]]

    t0 = time.time()
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # ONLY the Stage-1 GNN is reused; the gates and ranking net stay freshly
        # initialised so this run differs from the full model purely in the
        # counterfactual augmentation (a clean controlled ablation).
        gnn_sd = {k[len("gnn."):]: v for k, v in ck["model"].items() if k.startswith("gnn.")}
        model.gnn.load_state_dict(gnn_sd)
        model.refresh_cache(graph)
        print(f"[Stage 1] reused GNN weights from {args.init_from}; "
              f"gates + ranking net reinitialised")
    else:
        stage1_train_gnn(model, graph, d, epochs=args.stage1_epochs,
                         lr=args.lr, reg=args.reg, seed=args.seed)
    stage2_train_conversational(model, d, attr_matrix,
                                epochs=args.stage2_epochs, steps=args.steps,
                                batch=args.batch, lr=args.lr, reg=args.reg,
                                seed=args.seed,
                                use_counterfactual=not args.no_counterfactual,
                                lam=args.lam, val_users=val_pairs)
    torch.save({"model": model.state_dict(), "dim": args.dim,
                "gate_sign": args.gate_sign, "use_social": bool(use_social),
                "data": args.data}, args.out)
    print(f"saved {args.out}   total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
