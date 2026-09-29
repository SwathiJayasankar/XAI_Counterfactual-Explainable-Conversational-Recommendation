"""
FAITHFUL PyTorch implementation of CECR
"Counterfactual Explainable Conversational Recommendation"
Yu, Li, Wang, Li, Xu -- IEEE TKDE Vol. 36 No. 6, June 2024.

This replaces the earlier hand-written NumPy reproduction. Every trainable
component the paper describes is an actual trained network here, optimized
jointly with Adam (the paper's own optimizer), at the paper's own embedding
size of 128.

Equation map (paper -> code):
  Eq. 1      z_{u,i} = MLP([e_i^(in) (+) e_{y_u,i}])      GNNEncoder.rating_aware
  Eq. 2      e_u^U   = sigma(W . sum_i alpha_ui z_ui + b) GNNEncoder.encode_users
  Eq. 3      alpha_ui  2-layer attention network          AttentionNet  (TRAINED)
  Eq. 4-5    e_u^S social branch                          n/a on MovieLens (see note)
  Eq. 6      stacked MLP layers -> e_u                    GNNEncoder.stack
  Eq. 7      e_i = sigma(W^i . sum_u mu_iu z_iu + b)      GNNEncoder.encode_items
  Eq. 8      mu_iu     2-layer attention network          AttentionNet  (TRAINED)
  (psi_iu)   attribute aggregation                        GNNEncoder.encode_attrs
  Eq. 9      e^-_A, e^+_A   feedback aggregation          PreferenceRefinement
  Eq. 10     C^-_u, C^+_u gates -> e'_u                   PreferenceRefinement.user   (TRAINED)
  Eq. 11     C^-_i gate -> e'_{i-}                        PreferenceRefinement.item   (TRAINED)
  Eq. 12     e^-_I = mean of e'_{i-}                      PreferenceRefinement.rejected_items
  Eq. 13     e_overall = e'_u - e^-_I                     CECRModel.overall
  Eq. 14     s_ui = multi-layer ranking net g             RankingNet                  (TRAINED)
  Eq. 15-16  ES = s_ui - s_{u,i_Delta};  threshold eps    counterfactual_explanation
  Eq. 17-18  Delta search objective + updates             counterfactual_explanation
  Eq. 19     BPR loss + counterfactual sample (u,i,i_k)   train_cecr.py
  Eq. 20     dot-product ask-vs-recommend decision        CECRModel.decide_action
  Algorithm 1  full multi-round conversation loop         run_conversation

NOTE on Eqs. 4-5 (social branch): MovieLens-100k has no user-user social
graph (the paper only describes social relations for Douban-Book). We keep
Eq. 6's concatenation shape intact by feeding a zero social vector, so the
architecture is structurally identical rather than surgically removed.

NOTE on Eq. 10's sign: the paper literally prints e'_u = e_u (*) (C^-_u - C^+_u),
which would SUPPRESS the user representation along dimensions supported by
POSITIVE feedback -- semantically backwards, and this paper is already known
to contain at least one direction typo (its Eq. 22 PS definition repeats PN's
"item removed from Top-K" condition, contradicting the standard sufficiency
definition from its own cited reference, Tan et al. CIKM'21). We therefore
implement (C^+_u - C^-_u) by default but keep the literal form available via
`gate_sign=-1`, and we report both empirically rather than assuming.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class AttentionNet(nn.Module):
    """Eqs. 3 / 5 / 8: a genuine two-layer attention network.

        alpha*_ui = W_2 . sigma(W_1 . [z_{u,i} (+) e_u^(in)] + b_1) + b_2
        alpha_ui  = softmax over the neighbourhood

    (The earlier NumPy version replaced this with a fixed similarity-softmax
    that had no trainable parameters at all -- this is the real thing.)
    """

    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(2 * dim, dim)
        self.l2 = nn.Linear(dim, 1)

    def forward(self, z, e_target, mask):
        # z: [N_nodes, K, dim]   e_target: [N_nodes, dim]   mask: [N_nodes, K]
        tgt = e_target.unsqueeze(1).expand_as(z)
        logits = self.l2(F.relu(self.l1(torch.cat([z, tgt], dim=-1)))).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=1)
        return torch.nan_to_num(w, nan=0.0)          # rows with no neighbours


class GNNEncoder(nn.Module):
    """Eqs. 1-8: GNN-based offline learning over the U-I-A tripartite graph."""

    def __init__(self, n_users, n_items, n_attrs, dim=128, n_layers=2, use_social=False):
        super().__init__()
        self.dim = dim
        self.use_social = use_social
        self.e_u_in = nn.Embedding(n_users, dim)
        self.e_i_in = nn.Embedding(n_items, dim)
        self.e_a_in = nn.Embedding(n_attrs, dim)
        # Eq. 1's e_{y_u,i}: a dense vector for the rating (paper cites GraphRec [39],
        # which uses a learned embedding table per rating level). 0 = padding.
        self.rating_emb = nn.Embedding(6, dim, padding_idx=0)

        self.mlp_z = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())   # Eq. 1

        self.att_u = AttentionNet(dim)      # Eq. 3
        self.att_i = AttentionNet(dim)      # Eq. 8
        self.att_a = AttentionNet(dim)      # psi_iu
        self.att_s = AttentionNet(dim)      # Eq. 5  (social attention beta_ko)

        self.W_U = nn.Linear(dim, dim)      # Eq. 2
        self.W_I = nn.Linear(dim, dim)      # Eq. 7
        self.W_A = nn.Linear(dim, dim)
        self.W_S = nn.Linear(dim, dim)      # Eq. 4  (social aggregation)

        # Eq. 6: e_u^(0) = MLP([e_u^U (+) e_u^S]) then l stacked layers
        self.mlp_us = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())
        self.stack = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for emb in [self.e_u_in, self.e_i_in, self.e_a_in, self.rating_emb]:
            nn.init.normal_(emb.weight, std=0.1)
        with torch.no_grad():
            self.rating_emb.weight[0].zero_()

    def rating_aware(self, neigh_emb, ratings):
        """Eq. 1: z = MLP([neighbour embedding (+) rating embedding])."""
        return self.mlp_z(torch.cat([neigh_emb, self.rating_emb(ratings)], dim=-1))

    def item_space_user(self, graph, uidx, k=None):
        """Eqs. 2-3: the item-space user latent factor e_u^U for the given users.

        `k` optionally truncates the sampled neighbourhood (used for the second
        hop of the social branch, so a 2-hop pass stays tractable on the larger
        datasets). Neighbour lists were randomly sampled when the graph tensors
        were built, so taking the first k columns is itself a random subsample.
        """
        nb = graph["u_nb_items"][uidx]
        rt = graph["u_nb_ratings"][uidx]
        mk = graph["u_nb_mask"][uidx]
        if k is not None:
            nb, rt, mk = nb[:, :k], rt[:, :k], mk[:, :k]
        z = self.rating_aware(self.e_i_in(nb), rt)                    # Eq. 1
        alpha = self.att_u(z, self.e_u_in(uidx), mk)                  # Eq. 3
        agg = (alpha.unsqueeze(-1) * z).sum(dim=1)
        return F.relu(self.W_U(agg))                                  # Eq. 2

    def social_user(self, graph, uidx, k2=10):
        """Eqs. 4-5: the social-space user latent factor e_u^S.

        Aggregates the item-space factors u_o^U of u's friends with a trained
        attention (Eq. 5). Only available when the dataset ships a user-user
        graph -- Douban-Book does, MovieLens-100k does not.
        """
        fr = graph["u_friends"][uidx]                                 # [B,F]
        fm = graph["u_friends_mask"][uidx]
        B, Fr = fr.shape
        flat = fr.reshape(-1)
        u_o_U = self.item_space_user(graph, flat, k=k2).reshape(B, Fr, self.dim)
        beta = self.att_s(u_o_U, self.e_u_in(uidx), fm)               # Eq. 5
        agg = (beta.unsqueeze(-1) * u_o_U).sum(dim=1)
        return F.relu(self.W_S(agg))                                  # Eq. 4

    def encode_users(self, graph, uidx=None):
        """Eq. 6: combine the item-space and social-space factors, then stack."""
        if uidx is None:
            uidx = torch.arange(self.e_u_in.num_embeddings, device=self.e_u_in.weight.device)
        e_u_U = self.item_space_user(graph, uidx)
        if self.use_social and "u_friends" in graph:
            e_u_S = self.social_user(graph, uidx)
        else:
            e_u_S = torch.zeros_like(e_u_U)       # MovieLens: no social graph
        h = self.mlp_us(torch.cat([e_u_U, e_u_S], dim=-1))            # Eq. 6
        for layer in self.stack:
            h = F.relu(layer(h))
        return h

    def encode_items(self, graph, iidx=None):
        """Eqs. 7-8."""
        if iidx is None:
            iidx = torch.arange(self.e_i_in.num_embeddings, device=self.e_i_in.weight.device)
        z = self.rating_aware(self.e_u_in(graph["i_nb_users"][iidx]),
                              graph["i_nb_ratings"][iidx])
        alpha = self.att_i(z, self.e_i_in(iidx), graph["i_nb_mask"][iidx])   # Eq. 8
        agg = (alpha.unsqueeze(-1) * z).sum(dim=1)
        return F.relu(self.W_I(agg))                                  # Eq. 7

    def encode_attrs(self, e_i, memb_items, memb_mask):
        """Attribute aggregation over the items carrying that attribute."""
        z = e_i[memb_items]                                           # [A,K,d]
        alpha = self.att_a(z, self.e_a_in.weight, memb_mask)
        agg = (alpha.unsqueeze(-1) * z).sum(dim=1)
        return F.relu(self.W_A(agg))

    def forward(self, graph, chunk=4096):
        """Encode every node (used to materialise the caches); chunked so the
        larger datasets do not need the whole graph resident at once."""
        dev = self.e_u_in.weight.device
        e_i = torch.cat([self.encode_items(graph, torch.arange(s, min(s + chunk,
                        self.e_i_in.num_embeddings), device=dev))
                         for s in range(0, self.e_i_in.num_embeddings, chunk)])
        e_u = torch.cat([self.encode_users(graph, torch.arange(s, min(s + chunk,
                        self.e_u_in.num_embeddings), device=dev))
                         for s in range(0, self.e_u_in.num_embeddings, chunk)])
        e_a = self.encode_attrs(e_i, graph["a_items"], graph["a_mask"])
        return e_u, e_i, e_a


class PreferenceRefinement(nn.Module):
    """Eqs. 9-13: the trained gating functions C^-_u, C^+_u, C^-_i."""

    def __init__(self, dim, gate_sign=1):
        super().__init__()
        self.W3 = nn.Linear(3 * dim, dim)    # C^-_u  (Eq. 10)
        self.W4 = nn.Linear(3 * dim, dim)    # C^+_u  (Eq. 10)
        self.W5 = nn.Linear(3 * dim, dim)    # C^-_i  (Eq. 11)
        self.gate_sign = gate_sign
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def user(self, e_u, eA_pos, eA_neg):
        """Eq. 10: gated update of the long-term user representation."""
        C_neg = torch.sigmoid(self.W3(torch.cat([eA_neg, e_u, eA_neg * e_u], dim=-1)))
        C_pos = torch.sigmoid(self.W4(torch.cat([eA_pos, e_u, eA_pos * e_u], dim=-1)))
        if self.gate_sign >= 0:
            return e_u * (C_pos - C_neg)     # semantically-correct orientation
        return e_u * (C_neg - C_pos)         # the paper's literal (likely typo'd) form

    def item(self, e_i_rej, eA_pos):
        """Eq. 11: distil the positive information out of a rejected item."""
        C = torch.sigmoid(self.W5(torch.cat([eA_pos, e_i_rej, eA_pos * e_i_rej], dim=-1)))
        return e_i_rej * C

    def rejected_items(self, e_i_rej, eA_pos):
        """Eq. 12: e^-_I = mean over rejected items of e'_{i-}."""
        if e_i_rej.numel() == 0:
            return None
        return self.item(e_i_rej, eA_pos.expand_as(e_i_rej)).mean(dim=0)


class RankingNet(nn.Module):
    """Eq. 14: the multi-layer ranking model g that produces s_{u,i}.

    Algorithm 1 step 10 ranks the Top-K list with THIS network (the plain dot
    product of Eq. 20 is only for the ask-vs-recommend decision).
    """

    def __init__(self, dim, M=3):
        super().__init__()
        self.g = nn.Linear(2 * dim, dim)
        self.hidden = nn.ModuleList([nn.Linear(dim, dim) for _ in range(M - 2)])
        self.out = nn.Linear(dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, e_u, e_i):
        """e_u: [d] or [B,d];  e_i: [d], [N,d] or [B,N,d] -> scores."""
        if e_i.dim() == 1:
            e_i = e_i.unsqueeze(0)
        if e_u.dim() == 1:
            e_u = e_u.unsqueeze(0).expand(e_i.shape[0], -1)
        h = F.relu(self.g(torch.cat([e_u, e_i], dim=-1)))
        for layer in self.hidden:
            h = F.relu(layer(h))
        return self.out(h).squeeze(-1)


class CECRModel(nn.Module):
    """Everything together: the four components of Fig. 2."""

    def __init__(self, n_users, n_items, n_attrs, dim=128, gate_sign=1, M=3, n_layers=2,
                 use_social=False):
        super().__init__()
        self.dim, self.n_items, self.n_attrs = dim, n_items, n_attrs
        self.gnn = GNNEncoder(n_users, n_items, n_attrs, dim, n_layers, use_social=use_social)
        self.pref = PreferenceRefinement(dim, gate_sign)
        self.rank = RankingNet(dim, M)
        # materialised offline representations (Algorithm 1 step 1); refreshed
        # from the GNN and fine-tuned during conversational training
        self.register_buffer("e_u_cache", torch.zeros(n_users, dim))
        self.register_buffer("e_i_cache", torch.zeros(n_items, dim))
        self.register_buffer("e_a_cache", torch.zeros(n_attrs, dim))

    @torch.no_grad()
    def refresh_cache(self, graph):
        e_u, e_i, e_a = self.gnn(graph)
        self.e_u_cache.copy_(e_u)
        self.e_i_cache.copy_(e_i)
        self.e_a_cache.copy_(e_a)

    # ---- Eqs. 9-13 ----
    def attr_feedback(self, pos_attrs, neg_attrs, e_a):
        """Eq. 9: mean-pool the positive and negative attribute feedback."""
        eA_pos = e_a[list(pos_attrs)].mean(0) if pos_attrs else torch.zeros(self.dim, device=e_a.device)
        eA_neg = e_a[list(neg_attrs)].mean(0) if neg_attrs else torch.zeros(self.dim, device=e_a.device)
        return eA_pos, eA_neg

    def overall(self, e_u, pos_attrs, neg_attrs, rejected, e_a, e_i, use_pref_refine=True):
        """Eq. 13: e_overall = e'_u - e^-_I."""
        if not use_pref_refine:
            return e_u
        eA_pos, eA_neg = self.attr_feedback(pos_attrs, neg_attrs, e_a)
        e_u_prime = self.pref.user(e_u, eA_pos, eA_neg)            # Eq. 10
        if rejected:
            e_neg_I = self.pref.rejected_items(e_i[list(rejected)], eA_pos)   # Eqs. 11-12
            return e_u_prime - e_neg_I                                        # Eq. 13
        return e_u_prime

    # ---- Eq. 20 ----
    def decide_action(self, e_overall, cand_items, cand_attrs, e_i, e_a):
        """Compare the best item score with the best attribute score (dot products)."""
        best_i = (e_i[cand_items] @ e_overall).max() if len(cand_items) else torch.tensor(-1e9)
        best_a = (e_a[cand_attrs] @ e_overall).max() if len(cand_attrs) else torch.tensor(-1e9)
        return ("recommend" if best_i >= best_a else "ask"), best_i, best_a


# --------------------------------------------------------------------------
# Eqs. 15-18 : counterfactual explanation  (the XAI core)
# --------------------------------------------------------------------------
def counterfactual_explanation(model, e_overall, e_item, item_attr_mask, e_a,
                               s_ik, lam=100.0, l1_w=1.0, iters=100, lr=0.05):
    """
    Search for the minimal attribute change vector Delta that pushes item i
    below the k-th ranked item k  (Eq. 16's threshold condition), minimising

        ||Delta||_2^2 + ||Delta||_1 + lambda * (-log sigmoid(s_{u,i_k} - s_{u,i_Delta}))

    (Eq. 17, with ||Delta||_0 relaxed to the l1 norm exactly as the paper
    prescribes), by gradient descent (Eq. 18) -- here with autograd + Adam.

    Returns Delta, ES (Eq. 15), EC, and the explanation's attribute indices.
    """
    mask = item_attr_mask.float()
    delta = torch.zeros(model.n_attrs, device=e_item.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    with torch.no_grad():
        s_orig = model.rank(e_overall, e_item).squeeze()

    for _ in range(iters):
        d = delta * mask
        e_i_delta = e_item - d @ e_a                      # apply Delta in embedding space
        s_delta = model.rank(e_overall, e_i_delta).squeeze()
        rank_term = F.softplus(s_delta - s_ik)            # = -log sigmoid(s_ik - s_delta)
        loss = (d ** 2).sum() + l1_w * d.abs().sum() + lam * rank_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            # the paper gives Delta's range as [-1, 1] for binary attributes,
            # so a change may either remove or add attribute presence; clamping
            # at 0 from below would pin delta at its initial value whenever the
            # score-reducing direction happens to be negative
            delta.clamp_(-1.0, 1.0)

    with torch.no_grad():
        d = (delta * mask).detach()
        e_i_delta = e_item - d @ e_a
        s_delta = model.rank(e_overall, e_i_delta).squeeze()
        ES = (s_orig - s_delta).item()                     # Eq. 15
        expl = torch.nonzero(d.abs() > 0.05).flatten()
        EC = int(expl.numel())                             # Eq. 17's ||Delta||_0
    return d, ES, EC, expl.cpu().numpy(), e_i_delta.detach()


# --------------------------------------------------------------------------
# User simulator + Algorithm 1
# --------------------------------------------------------------------------
class UserSimulator:
    """Section IV-D-1: binary feedback against the target item's real attributes."""

    def __init__(self, attr_matrix):
        self.attr_matrix = attr_matrix

    def attrs_of(self, item):
        return set(np.nonzero(self.attr_matrix[item])[0].tolist())


def candidate_set(attr_matrix, pos_attrs, neg_attrs, rejected, n_items):
    """
    Standard multi-round CRS candidate handling (Fig. 1: "the CRS will exclude
    these candidates and search for others"), as used by EAR/FPAN/CPR:
    keep items carrying ALL confirmed positive attributes, drop items carrying
    any rejected attribute, drop already-rejected items.
    """
    keep = np.ones(n_items, dtype=bool)
    for a in pos_attrs:
        keep &= attr_matrix[:, a] > 0
    for a in neg_attrs:
        keep &= attr_matrix[:, a] == 0
    if rejected:
        keep[list(rejected)] = False
    return np.nonzero(keep)[0]


def run_conversation(model, u, target_item, sim, attr_matrix, e_u, e_i, e_a,
                     max_rounds=20, topk=10, rng=None,
                     use_pref_refine=True, use_score_decision=True,
                     collect_cf=False, lam=100.0, cf_iters=60):
    """
    Algorithm 1. Returns (success_round or None, list of counterfactual records).

    Each round: decide via Eq. 20, then either ask an attribute or recommend
    the Top-K ranked by the trained model g (Eq. 14 / Algorithm 1 step 10).
    """
    rng = rng or np.random.default_rng(0)
    n_items = attr_matrix.shape[0]
    target_attrs = sim.attrs_of(target_item)
    if not target_attrs:
        return None, []

    first = int(rng.choice(sorted(target_attrs)))
    pos_attrs, neg_attrs, asked = {first}, set(), {first}
    rejected = []
    cf_records = []

    for rnd in range(1, max_rounds + 1):
        cands = candidate_set(attr_matrix, pos_attrs, neg_attrs, rejected, n_items)
        if len(cands) == 0:
            break
        with torch.no_grad():
            e_overall = model.overall(e_u[u], pos_attrs, neg_attrs, rejected,
                                      e_a, e_i, use_pref_refine)
        # attributes still worth asking: unasked ones present in the candidate pool
        present = np.nonzero(attr_matrix[cands].sum(0) > 0)[0]
        cand_attrs = [int(a) for a in present if a not in asked]

        if use_score_decision:
            with torch.no_grad():
                action, _, _ = model.decide_action(e_overall, cands, cand_attrs, e_i, e_a)
        else:
            action = rng.choice(["ask", "recommend"])      # RecPe ablation
        if not cand_attrs:
            action = "recommend"

        if action == "recommend":
            with torch.no_grad():
                scores = model.rank(e_overall, e_i[cands])       # Eq. 14 ranking
            k = min(topk, len(cands))
            top = cands[torch.topk(scores, k).indices.cpu().numpy()]
            if target_item in top:
                return rnd, cf_records
            for it in top:
                rejected.append(int(it))
            if collect_cf and len(cands) > len(top):
                # anchor k = best item just outside the Top-K (Eq. 16's item k)
                outside = np.setdiff1d(cands, top, assume_unique=False)
                if len(outside):
                    with torch.no_grad():
                        os_scores = model.rank(e_overall, e_i[outside])
                        k_item = int(outside[int(torch.argmax(os_scores))])
                        s_ik = model.rank(e_overall, e_i[k_item]).squeeze()
                    it = int(top[0])
                    d, ES, EC, expl, _ = counterfactual_explanation(
                        model, e_overall, e_i[it],
                        torch.as_tensor(attr_matrix[it], device=e_i.device),
                        e_a, s_ik, lam=lam, iters=cf_iters)
                    cf_records.append(dict(user=int(u), item=it, k_item=k_item,
                                           ES=ES, EC=EC, expl=expl.tolist()))
        else:
            with torch.no_grad():
                a_scores = e_a[cand_attrs] @ e_overall            # Eq. 20
            a_star = int(cand_attrs[int(torch.argmax(a_scores))])
            asked.add(a_star)
            (pos_attrs if a_star in target_attrs else neg_attrs).add(a_star)

    return None, cf_records


# --------------------------------------------------------------------------
# graph tensors
# --------------------------------------------------------------------------
def build_graph_tensors(d, max_nb=40, max_friends=10, device="cpu"):
    """Padded neighbour tensors for the tripartite graph (train split only)."""
    train = d["train"]
    n_users, n_items, n_attrs = d["n_users"], d["n_items"], d["n_attrs"]
    rng = np.random.default_rng(0)

    def pad(groups, n_nodes):
        idx = np.zeros((n_nodes, max_nb), dtype=np.int64)
        rat = np.zeros((n_nodes, max_nb), dtype=np.int64)
        msk = np.zeros((n_nodes, max_nb), dtype=bool)
        for node, (nbs, rs) in groups.items():
            if len(nbs) > max_nb:
                sel = rng.choice(len(nbs), max_nb, replace=False)
                nbs, rs = nbs[sel], rs[sel]
            k = len(nbs)
            idx[node, :k] = nbs
            rat[node, :k] = rs
            msk[node, :k] = True
        return (torch.as_tensor(idx, device=device),
                torch.as_tensor(rat, device=device),
                torch.as_tensor(msk, device=device))

    ug = {u: (g.item.values, g.rating.values) for u, g in train.groupby("user")}
    ig = {i: (g.user.values, g.rating.values) for i, g in train.groupby("item")}
    u_nb_items, u_nb_ratings, u_nb_mask = pad(ug, n_users)
    i_nb_users, i_nb_ratings, i_nb_mask = pad(ig, n_items)

    attr_matrix = d["attr_matrix"]
    a_items = np.zeros((n_attrs, max_nb), dtype=np.int64)
    a_mask = np.zeros((n_attrs, max_nb), dtype=bool)
    for a in range(n_attrs):
        members = np.nonzero(attr_matrix[:, a])[0]
        if len(members) > max_nb:
            members = rng.choice(members, max_nb, replace=False)
        a_items[a, :len(members)] = members
        a_mask[a, :len(members)] = True

    graph = dict(u_nb_items=u_nb_items, u_nb_ratings=u_nb_ratings, u_nb_mask=u_nb_mask,
                 i_nb_users=i_nb_users, i_nb_ratings=i_nb_ratings, i_nb_mask=i_nb_mask,
                 a_items=torch.as_tensor(a_items, device=device),
                 a_mask=torch.as_tensor(a_mask, device=device))

    # --- user-user social graph (Eqs. 4-5), when the dataset provides one ---
    if d.get("social") is not None and len(d["social"]):
        friends = np.zeros((n_users, max_friends), dtype=np.int64)
        fmask = np.zeros((n_users, max_friends), dtype=bool)
        adj = {}
        for u, v in d["social"]:
            adj.setdefault(int(u), []).append(int(v))
        for u, vs in adj.items():
            if len(vs) > max_friends:
                vs = rng.choice(vs, max_friends, replace=False)
            friends[u, :len(vs)] = vs
            fmask[u, :len(vs)] = True
        graph["u_friends"] = torch.as_tensor(friends, device=device)
        graph["u_friends_mask"] = torch.as_tensor(fmask, device=device)
    return graph
