"""
Re-implementations of every model the CECR paper compares against, so they can
be trained and evaluated on OUR machine, on OUR splits, through the SAME
conversation harness (crs_bench.py) as CECR.

Table II / Fig. 3 / Fig. 6(d)  (category I, non-explainable CRSs)
  EAR      Lei et al., WSDM'20   FM estimation (attribute-aware BPR) + policy
                                 network trained with REINFORCE + reflection
  FPAN     Xu et al., WSDM'21    EAR + gating modules that adapt the user
                                 embedding with attribute- and item-level feedback
  UNICORN  Deng et al., SIGIR'21 graph-based state (GCN + Transformer) and a
                                 unified dueling double-DQN over attribute AND
                                 item actions, TransE-pretrained graph embeddings
  KBRD     Chen et al., EMNLP'19 R-GCN over the item-attribute knowledge graph,
                                 self-attentive user representation over the
                                 mentioned entities
Table IV  (category II, explainable CRSs)
  CPR      Lei et al., KDD'20    path reasoning user -> attribute -> item; the
                                 explanation is the attribute path
  CAEC     Kondylidis et al. '21 category-aware preference prediction; the
                                 explanation is the item's categories the user is
                                 predicted to prefer
  SAUR     Zhang et al., CIKM'18 multi-memory network over the user's history
                                 ("search records") and stated aspects; the
                                 explanation is the attention-weighted aspects

Declared simplifications (forced by the data/compute, applied identically to
every dataset):
  * none of the three datasets carries dialogue text or search logs, so CAEC's
    BERT utterance encoder and SAUR's query memory are fed the conversation
    STATE (accepted / rejected attributes) instead of natural language;
  * EAR/FPAN's policy chooses ask-vs-recommend and the attribute is the
    preference-weighted max-entropy one (EAR's own pre-training strategy),
    because one softmax output per attribute is not trainable with 245-333
    attributes on this budget; UNICORN keeps its full unified action space
    (top-10 attributes + top-10 items);
  * KBRD has no question policy of its own: it asks the max-entropy attribute
    and recommends when its confidence passes a threshold tuned on validation.
All models share the paper's settings: embedding 128, Adam, Omega = 1e-5,
Top-K = 10.
"""
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crs_bench import MAX_ROUNDS, TOPK, attr_entropy, run_conversation
from train_cecr import sample_states, padded, masked_mean

DIM = 128
REG = 1e-5


def bpr(pos, neg):
    return F.softplus(neg - pos).mean()


def positives(d):
    tr = d["train"]
    pos = tr[tr.rating >= 4]
    # items without any attribute can never be reached in a conversation
    pos = pos[d["attr_matrix"][pos.item.values].sum(1) > 0]
    return (pos.user.values.astype(np.int64), pos.item.values.astype(np.int64),
            pos.groupby("user")["item"].apply(list).to_dict())


def attr_members(attr_matrix):
    return [np.nonzero(attr_matrix[:, a])[0] for a in range(attr_matrix.shape[1])]


def len_bin(n):
    return int(np.digitize(n, [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]))


# ==========================================================================
# Shared estimation backbone: attribute-aware factorisation machine (EAR Sec. 4.1)
# ==========================================================================
class AttrFM(nn.Module):
    """s(u, i | P_u) = u.x_i + sum_{p in P_u} p.x_i  =  (u + sum p) . x_i
    with the item's own attributes as FM side features:
        x_i = v_i + sum_{a in A_i} e_a / sqrt|A_i|
    (the side features are what lets PN/PS remove attributes from an item)."""

    def __init__(self, n_users, n_items, attr_matrix, dim=DIM):
        super().__init__()
        n_attrs = attr_matrix.shape[1]
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        self.A = nn.Embedding(n_attrs, dim)
        for e in (self.U, self.V, self.A):
            nn.init.normal_(e.weight, std=0.1)
        am = torch.as_tensor(attr_matrix, dtype=torch.float32)
        self.register_buffer("am", am / am.sum(1, keepdim=True).clamp(min=1).sqrt())

    def item_vec(self, items):
        return self.V(items) + self.am[items] @ self.A.weight

    def item_vec_with(self, item, keep):
        """x_i rebuilt with only the attributes in `keep`."""
        x = self.V.weight[item]
        if keep:
            x = x + self.A.weight[list(keep)].sum(0) / np.sqrt(len(keep))
        return x

    def query(self, u_vec, pos):
        return u_vec + (self.A.weight[list(pos)].sum(0) if pos else 0.0)


def train_fm(fm, d, attr_matrix, epochs=10, batch=1024, lr=1e-3, seed=0, log=print):
    """EAR's attribute-aware BPR: item-level (random + attribute-consistent hard
    negative) and attribute-level ranking, jointly (EAR Eq. 2-4)."""
    rng = np.random.default_rng(seed)
    users, items, _ = positives(d)
    n_items, n_attrs = attr_matrix.shape
    members = attr_members(attr_matrix)
    item_attrs = [np.nonzero(attr_matrix[i])[0] for i in range(n_items)]
    opt = torch.optim.Adam(fm.parameters(), lr=lr, weight_decay=REG)
    n = len(users)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            B = len(sel)
            P, hard, a_pos, a_neg = [], [], [], []
            for i in items[sel]:
                ia = item_attrs[i]
                k = int(rng.integers(0, min(3, len(ia)) + 1))
                p = rng.choice(ia, size=k, replace=False).tolist() if k else []
                P.append(p)
                hard.append(int(rng.choice(members[p[0]])) if p else int(rng.integers(n_items)))
                rest = [a for a in ia if a not in p] or list(ia)
                a_pos.append(int(rng.choice(rest)))
                while True:
                    an = int(rng.integers(n_attrs))
                    if attr_matrix[i, an] == 0:
                        break
                a_neg.append(an)
            pi, pm = padded(P, "cpu")
            u = fm.U(torch.as_tensor(users[sel]))
            q = u + (fm.A(pi) * pm.unsqueeze(-1)).sum(1)
            xp = fm.item_vec(torch.as_tensor(items[sel]))
            xn = fm.item_vec(torch.as_tensor(rng.integers(0, n_items, size=B)))
            xh = fm.item_vec(torch.as_tensor(hard))
            sp = (q * xp).sum(-1)
            loss = bpr(sp, (q * xn).sum(-1)) + bpr(sp, (q * xh).sum(-1))
            loss = loss + bpr((q * fm.A(torch.as_tensor(a_pos))).sum(-1),
                              (q * fm.A(torch.as_tensor(a_neg))).sum(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        log(f"  [FM] epoch {ep+1}/{epochs} loss={tot/nb:.4f} ({time.time()-t0:.1f}s)")
    return fm


# ==========================================================================
# EAR: estimation (FM) + action (policy network, REINFORCE) + reflection
# ==========================================================================
class PolicyNet(nn.Module):
    def __init__(self, in_dim, hidden=64, n_out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


ACTIONS = ("ask", "recommend")


class EARAgent:
    """State = [entropy of each attribute over candidates, preference of each
    attribute, conversation history, candidate-list length] (EAR Sec. 4.2)."""

    def __init__(self, fm, attr_matrix, user_hist, reflection=True):
        self.fm, self.am = fm, attr_matrix
        self.n_attrs = attr_matrix.shape[1]
        self.user_hist = user_hist
        self.reflection = reflection
        with torch.no_grad():
            self.X = fm.item_vec(torch.arange(attr_matrix.shape[0]))
        self.policy = PolicyNet(2 * self.n_attrs + 3 * MAX_ROUNDS + 10)
        self.training = False
        self.mode = "policy"            # "rule" = EAR's max-entropy pre-training strategy

    # ---- estimation ----
    def user_vec(self, st):
        return self.uvec

    def begin(self, st):
        self.uvec = self.fm.U.weight[st.u].detach().clone()
        self.logps, self.rewards, self._decided = [], [], False

    @torch.no_grad()
    def prepare(self, st, cands):
        q = self.fm.query(self.user_vec(st), st.pos)
        self.q = q
        self.scores = (self.X[cands] @ q).numpy()
        self.pref = torch.sigmoid(self.fm.A.weight @ q).numpy()
        self.ent = attr_entropy(self.am, cands)

    def features(self, st, cands):
        ent = self.ent.copy()
        ent[list(st.asked)] = 0.0
        his = np.zeros((MAX_ROUNDS, 3), dtype=np.float32)
        for t, h in enumerate(st.history[:MAX_ROUNDS]):
            his[t, ("ask_suc", "ask_fail", "rec_fail").index(h)] = 1.0
        ln = np.zeros(10, dtype=np.float32)
        ln[len_bin(len(cands))] = 1.0
        return torch.as_tensor(np.concatenate([ent, self.pref, his.ravel(), ln]),
                               dtype=torch.float32)

    # ---- action ----
    def decide(self, st, cands, cand_attrs, rng):
        if self.mode == "rule":
            return "recommend" if len(cands) <= TOPK else "ask"
        logits = self.policy(self.features(st, cands))
        if self.training:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            self.logps.append(dist.log_prob(a))
            self._decided = True
            return ACTIONS[int(a)]
        return ACTIONS[int(torch.argmax(logits))]

    def pick_attr(self, st, cands, cand_attrs):
        w = self.ent[cand_attrs] * self.pref[cand_attrs]
        return cand_attrs[int(np.argmax(w))]

    def rank(self, st, cands):
        return self.scores

    # ---- reflection (EAR Sec. 4.3): rejected items become BPR negatives ----
    def reflect(self, st, rejected):
        if not self.reflection:
            return
        hist = self.user_hist.get(st.u, [])
        if not hist:
            return
        u = self.uvec.clone().requires_grad_(True)
        opt = torch.optim.SGD([u], lr=0.05)
        xr = self.X[torch.as_tensor(rejected[:TOPK])]
        xh = self.X[torch.as_tensor(hist[:50])]
        for _ in range(3):
            q = self.fm.query(u, st.pos)
            loss = F.softplus((xr @ q).unsqueeze(1) - (xh @ q).unsqueeze(0)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.uvec = u.detach()

    def feedback(self, st, kind, reward, done):
        if not self.training:
            return
        if self._decided:
            self.rewards.append(reward)
            self._decided = False
        elif self.rewards:
            self.rewards[-1] += reward


def train_policy_reinforce(agent, pairs, attr_matrix, episodes=1000, gamma=0.7,
                           lr=1e-3, pretrain=200, seed=0, log=print):
    """EAR Sec. 4.2: pre-train on the max-entropy strategy, then REINFORCE."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(agent.policy.parameters(), lr=lr, weight_decay=REG)
    # --- supervised pre-training: imitate the rule "recommend once <= K left" ---
    agent.training, agent.mode = False, "rule"
    X, Y = [], []
    orig_decide = agent.decide

    def recording_decide(st, cands, cand_attrs, r):
        a = orig_decide(st, cands, cand_attrs, r)
        X.append(agent.features(st, cands))
        Y.append(ACTIONS.index(a))
        return a

    agent.decide = recording_decide
    for e in range(pretrain):
        u, t = pairs[int(rng.integers(len(pairs)))]
        run_conversation(agent, u, t, attr_matrix, rng)
    del agent.decide
    if X:
        Xt, Yt = torch.stack(X), torch.as_tensor(Y)
        for _ in range(200):
            loss = F.cross_entropy(agent.policy(Xt), Yt)
            opt.zero_grad(); loss.backward(); opt.step()
    # --- REINFORCE ---
    agent.mode, agent.training = "policy", True
    succ, t0 = [], time.time()
    for e in range(episodes):
        u, t = pairs[int(rng.integers(len(pairs)))]
        r = run_conversation(agent, u, t, attr_matrix, rng)
        succ.append(r is not None)
        if agent.logps:
            G, ret = 0.0, []
            for rw in reversed(agent.rewards[:len(agent.logps)]):
                G = rw + gamma * G
                ret.insert(0, G)
            ret = torch.as_tensor(ret, dtype=torch.float32)
            if len(ret) > 1:
                ret = (ret - ret.mean()) / (ret.std() + 1e-6)
            loss = -(torch.stack(agent.logps[:len(ret)]) * ret).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if (e + 1) % 250 == 0:
            log(f"  [REINFORCE] ep {e+1}/{episodes} train SR={np.mean(succ[-250:]):.3f}"
                f" ({time.time()-t0:.0f}s)")
    agent.training = False
    return agent


# ==========================================================================
# FPAN: EAR + gated adaptation of the user embedding from online feedback
# ==========================================================================
class FPANGates(nn.Module):
    """Attribute-level gates (accepted / rejected attributes) and item-level
    gates (rejected items, distilled by the accepted attributes)."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.Wp = nn.Linear(3 * dim, dim)
        self.Wn = nn.Linear(3 * dim, dim)
        self.Wr = nn.Linear(3 * dim, dim)

    def forward(self, eu, p, n, rej, rej_mask):
        gp = torch.sigmoid(self.Wp(torch.cat([eu, p, eu * p], -1)))
        gn = torch.sigmoid(self.Wn(torch.cat([eu, n, eu * n], -1)))
        out = eu + gp * p - gn * n
        if rej is not None:
            pe = p.unsqueeze(1).expand_as(rej)
            gr = torch.sigmoid(self.Wr(torch.cat([rej, pe, rej * pe], -1)))
            r = (rej * gr * rej_mask.unsqueeze(-1)).sum(1) / rej_mask.sum(1, keepdim=True).clamp(min=1)
            out = out - r
        return out


def train_fpan(fm, gates, d, attr_matrix, epochs=10, steps=150, batch=128, lr=1e-3,
               seed=0, log=print):
    rng = np.random.default_rng(seed)
    params = list(gates.parameters()) + list(fm.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=REG)
    for ep in range(epochs):
        tot, t0 = 0.0, time.time()
        for _ in range(steps):
            users, tg, pa, na, rj, negs = sample_states(d, attr_matrix, rng, batch)
            eu = fm.U(torch.as_tensor(users))
            pi, pm = padded(pa, "cpu"); ni, nm = padded(na, "cpu"); ri, rm = padded(rj, "cpu")
            p = masked_mean(fm.A.weight, pi, pm, DIM)
            n = masked_mean(fm.A.weight, ni, nm, DIM)
            rej = fm.item_vec(ri.reshape(-1)).reshape(*ri.shape, -1)
            q = gates(eu, p, n, rej, rm)
            sp = (q * fm.item_vec(torch.as_tensor(tg))).sum(-1)
            xn = fm.item_vec(torch.as_tensor(negs).reshape(-1)).reshape(*negs.shape, -1)
            sn = (q.unsqueeze(1) * xn).sum(-1)
            loss = F.softplus(sn - sp.unsqueeze(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"  [FPAN gates] epoch {ep+1}/{epochs} loss={tot/steps:.4f} ({time.time()-t0:.1f}s)")


class FPANAgent(EARAgent):
    def __init__(self, fm, gates, attr_matrix, user_hist):
        super().__init__(fm, attr_matrix, user_hist, reflection=False)
        self.gates = gates

    @torch.no_grad()
    def prepare(self, st, cands):
        A = self.fm.A.weight
        p = A[list(st.pos)].mean(0) if st.pos else torch.zeros(DIM)
        n = A[list(st.neg)].mean(0) if st.neg else torch.zeros(DIM)
        rej, rm = None, None
        if st.rejected:
            rej = self.X[torch.as_tensor(st.rejected)].unsqueeze(0)
            rm = torch.ones(1, len(st.rejected), dtype=torch.bool)
        q = self.gates(self.uvec.unsqueeze(0), p.unsqueeze(0), n.unsqueeze(0), rej, rm)[0]
        self.q = q
        self.scores = (self.X[cands] @ q).numpy()
        self.pref = torch.sigmoid(A @ q).numpy()
        self.ent = attr_entropy(self.am, cands)


# ==========================================================================
# UNICORN: graph-based unified policy learning (dueling double DQN)
# ==========================================================================
class KGEmb(nn.Module):
    """Graph embeddings pre-trained with TransE on (user-interact-item,
    item-belong-attribute) triples, as UNICORN does, jointly with the
    dot-product preference objective its action selection uses."""

    def __init__(self, n_users, n_items, n_attrs, dim=DIM):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.I = nn.Embedding(n_items, dim)
        self.A = nn.Embedding(n_attrs, dim)
        self.R = nn.Embedding(2, dim)
        for e in (self.U, self.I, self.A, self.R):
            nn.init.normal_(e.weight, std=0.1)


def train_kg(kg, d, attr_matrix, epochs=10, batch=1024, lr=1e-3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    users, items, _ = positives(d)
    ia, aa = np.nonzero(attr_matrix)
    n_items, n_attrs = attr_matrix.shape
    opt = torch.optim.Adam(kg.parameters(), lr=lr, weight_decay=REG)
    n = len(users)
    item_attrs = [np.nonzero(attr_matrix[i])[0] for i in range(n_items)]
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            B = len(sel)
            u = torch.as_tensor(users[sel]); i = torch.as_tensor(items[sel])
            ineg = torch.as_tensor(rng.integers(0, n_items, size=B))
            # TransE: user + r_interact ~ item ; item + r_belong ~ attribute
            h, t, tn = kg.U(u) + kg.R.weight[0], kg.I(i), kg.I(ineg)
            l_te = F.relu(1.0 + (h - t).norm(dim=-1) - (h - tn).norm(dim=-1)).mean()
            k = rng.integers(0, len(ia), size=B)
            hi = kg.I(torch.as_tensor(ia[k])) + kg.R.weight[1]
            ta = kg.A(torch.as_tensor(aa[k]))
            tna = kg.A(torch.as_tensor(rng.integers(0, n_attrs, size=B)))
            l_te = l_te + F.relu(1.0 + (hi - ta).norm(dim=-1) - (hi - tna).norm(dim=-1)).mean()
            # preference w_v = sigma(e_u.e_v + sum_p e_v.e_p)  (UNICORN Eq. 9)
            P = [rng.choice(item_attrs[x], size=min(len(item_attrs[x]), int(rng.integers(0, 3))),
                            replace=False).tolist() for x in items[sel]]
            pi, pm = padded(P, "cpu")
            q = kg.U(u) + (kg.A(pi) * pm.unsqueeze(-1)).sum(1)
            l_pr = bpr((q * kg.I(i)).sum(-1), (q * kg.I(ineg)).sum(-1))
            loss = 0.5 * l_te + l_pr
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        log(f"  [TransE+pref] epoch {ep+1}/{epochs} loss={tot/nb:.4f} ({time.time()-t0:.1f}s)")


class UnicornQ(nn.Module):
    """Graph-based state: one GCN propagation over the dynamic graph
    {user, accepted attributes, rejected attributes}, then a Transformer over
    the conversation's node sequence (UNICORN Sec. 4.2); dueling Q head."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.type_emb = nn.Embedding(3, dim)
        self.gcn = nn.Linear(dim, dim)
        self.tf = nn.TransformerEncoderLayer(dim, nhead=2, dim_feedforward=dim,
                                             dropout=0.0, batch_first=True)
        self.V = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.Adv = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, tok, typ, mask):
        # tok [B,L,d] typ [B,L] mask [B,L] (True = real node). node 0 = user.
        u = tok[:, :1]
        nbr = ((tok * mask.unsqueeze(-1)).sum(1, keepdim=True)
               / mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1))
        h = torch.where(typ.unsqueeze(-1) == 0, nbr, tok + u)       # GCN message passing
        h = F.relu(self.gcn(h)) + self.type_emb(typ)
        h = self.tf(h, src_key_padding_mask=~mask)
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def q(self, s, act, amask):
        # s [B,d] act [B,N,d] amask [B,N]
        sv = s.unsqueeze(1).expand(-1, act.shape[1], -1)
        adv = self.Adv(torch.cat([sv, act], -1)).squeeze(-1)
        adv = adv.masked_fill(~amask, 0.0)
        mean_adv = adv.sum(1, keepdim=True) / amask.sum(1, keepdim=True).clamp(min=1)
        return (self.V(s) + adv - mean_adv).masked_fill(~amask, -1e9)


class UnicornAgent:
    K_P, K_V = 10, 10
    L = MAX_ROUNDS + 2

    def __init__(self, kg, attr_matrix):
        self.kg, self.am = kg, attr_matrix
        self.net, self.tgt = UnicornQ(), UnicornQ()
        self.tgt.load_state_dict(self.net.state_dict())
        self.training, self.eps = False, 0.0
        self.mem = deque(maxlen=5000)
        self.gamma = 0.999
        self.rng = np.random.default_rng(0)

    def begin(self, st):
        self.pending = None
        self._decided = False

    def tokens(self, st):
        E = self.kg
        toks = [E.U.weight[st.u]] + [E.A.weight[a] for a in st.pos] + [E.A.weight[a] for a in st.neg]
        typ = [0] + [1] * len(st.pos) + [2] * len(st.neg)
        toks, typ = toks[:self.L], typ[:self.L]
        T = torch.zeros(self.L, DIM); T[:len(toks)] = torch.stack(toks).detach()
        ty = torch.zeros(self.L, dtype=torch.long); ty[:len(typ)] = torch.as_tensor(typ)
        m = torch.zeros(self.L, dtype=torch.bool); m[:len(toks)] = True
        return T, ty, m

    @torch.no_grad()
    def prepare(self, st, cands):
        E = self.kg
        q = E.U.weight[st.u] + (E.A.weight[list(st.pos)].sum(0) if st.pos else 0.0) \
            - (E.A.weight[list(st.neg)].sum(0) if st.neg else 0.0)
        self.scores = (E.I.weight[cands] @ q).numpy()                  # w_v
        w = 1 / (1 + np.exp(-np.clip(self.scores, -30, 30)))
        self.went = attr_entropy(self.am, cands, w)                    # weighted entropy

    def actions(self, cands, cand_attrs):
        """Action selection strategy: top-K_p attributes by weighted entropy,
        top-K_v items by preference."""
        top_a = [cand_attrs[i] for i in np.argsort(-self.went[cand_attrs])[:self.K_P]]
        top_v = cands[np.argsort(-self.scores)[:self.K_V]].tolist()
        emb = torch.cat([self.kg.A.weight[top_a], self.kg.I.weight[top_v]]).detach()
        return [("a", a) for a in top_a] + [("v", v) for v in top_v], emb

    def decide(self, st, cands, cand_attrs, rng):
        acts, emb = self.actions(cands, cand_attrs)
        T, ty, m = self.tokens(st)
        if self.training:
            self._close(T, ty, m, emb, done=False)
        with torch.no_grad():
            s = self.net.encode(T[None], ty[None], m[None])
            qv = self.net.q(s, emb[None], torch.ones(1, len(acts), dtype=torch.bool))[0]
        if self.training and rng.random() < self.eps:
            k = int(rng.integers(len(acts)))
        else:
            k = int(torch.argmax(qv))
        self.choice = acts[k]
        if self.training:
            self.pending = dict(s=(T, ty, m), a=emb[k], r=0.0)
            self._decided = True
        return "ask" if acts[k][0] == "a" else "recommend"

    def pick_attr(self, st, cands, cand_attrs):
        return self.choice[1]

    def rank(self, st, cands):
        return self.scores

    def _close(self, T, ty, m, emb, done):
        if self.pending is None:
            return
        p = self.pending
        self.mem.append((p["s"], p["a"], p["r"], (T, ty, m), emb, done))
        self.pending = None
        self.learn()

    def feedback(self, st, kind, reward, done):
        if not self.training or self.pending is None:
            return
        self.pending["r"] += reward
        if done:
            z = torch.zeros(self.L, DIM)
            self._close(z, torch.zeros(self.L, dtype=torch.long),
                        torch.zeros(self.L, dtype=torch.bool).index_fill(0, torch.tensor([0]), True),
                        torch.zeros(1, DIM), done=True)

    def learn(self, batch=64):
        if len(self.mem) < batch:
            return
        idx = self.rng.choice(len(self.mem), batch, replace=False)
        S = [self.mem[i] for i in idx]
        T = torch.stack([s[0][0] for s in S]); ty = torch.stack([s[0][1] for s in S])
        m = torch.stack([s[0][2] for s in S])
        a = torch.stack([s[1] for s in S]).unsqueeze(1)
        r = torch.as_tensor([s[2] for s in S], dtype=torch.float32)
        done = torch.as_tensor([s[5] for s in S], dtype=torch.float32)
        T2 = torch.stack([s[3][0] for s in S]); ty2 = torch.stack([s[3][1] for s in S])
        m2 = torch.stack([s[3][2] for s in S])
        N = max(s[4].shape[0] for s in S)
        A2 = torch.zeros(batch, N, DIM); am2 = torch.zeros(batch, N, dtype=torch.bool)
        for b, s in enumerate(S):
            A2[b, :s[4].shape[0]] = s[4]; am2[b, :s[4].shape[0]] = True
        q = self.net.q(self.net.encode(T, ty, m), a, torch.ones(batch, 1, dtype=torch.bool))[:, 0]
        with torch.no_grad():
            on = self.net.q(self.net.encode(T2, ty2, m2), A2, am2)
            best = on.argmax(1, keepdim=True)                           # double DQN
            tq = self.tgt.q(self.tgt.encode(T2, ty2, m2), A2, am2).gather(1, best)[:, 0]
            y = r + self.gamma * (1 - done) * tq
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()


def train_unicorn(agent, pairs, attr_matrix, episodes=1500, lr=1e-4, seed=0, log=print):
    rng = np.random.default_rng(seed)
    agent.opt = torch.optim.Adam(agent.net.parameters(), lr=lr, weight_decay=REG)
    agent.training = True
    succ, t0 = [], time.time()
    for e in range(episodes):
        agent.eps = max(0.05, 0.5 * (1 - e / (0.7 * episodes)))
        u, t = pairs[int(rng.integers(len(pairs)))]
        succ.append(run_conversation(agent, u, t, attr_matrix, rng) is not None)
        if (e + 1) % 20 == 0:
            agent.tgt.load_state_dict(agent.net.state_dict())
        if (e + 1) % 250 == 0:
            log(f"  [DQN] ep {e+1}/{episodes} train SR={np.mean(succ[-250:]):.3f}"
                f" eps={agent.eps:.2f} ({time.time()-t0:.0f}s)")
    agent.training = False
    return agent


# ==========================================================================
# KBRD: R-GCN knowledge-graph recommender with self-attentive user modelling
# ==========================================================================
class KBRD(nn.Module):
    def __init__(self, n_items, attr_matrix, attr_names, dim=DIM, n_bases=4):
        super().__init__()
        n_attrs = attr_matrix.shape[1]
        self.n_items, self.n_ent = n_items, n_items + n_attrs
        self.E = nn.Embedding(self.n_ent, dim)
        nn.init.normal_(self.E.weight, std=0.1)
        kinds = sorted({(nm.split(":")[0] if ":" in nm else "genre") for nm in attr_names})
        kind_of = [kinds.index(nm.split(":")[0] if ":" in nm else "genre") for nm in attr_names]
        ii, aa = np.nonzero(attr_matrix)
        self.adj = []
        for k in range(len(kinds)):
            sel = np.array([kind_of[a] == k for a in aa])
            src_i, dst_a = ii[sel], aa[sel] + n_items
            for rows, cols in ((src_i, dst_a), (dst_a, src_i)):      # both directions
                deg = np.bincount(rows, minlength=self.n_ent).astype(np.float32)
                val = 1.0 / np.maximum(deg[rows], 1)
                self.adj.append(torch.sparse_coo_tensor(
                    torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long),
                    torch.as_tensor(val), (self.n_ent, self.n_ent)).coalesce())
        R = len(self.adj)
        self.bases = nn.Parameter(torch.randn(n_bases, dim, dim) * (1 / np.sqrt(dim)))
        self.coef = nn.Parameter(torch.randn(R, n_bases) * 0.1)          # basis decomposition
        self.W0 = nn.Linear(dim, dim)
        self.att1 = nn.Linear(dim, dim)
        self.att2 = nn.Linear(dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_items))

    def entities(self):
        x = self.E.weight
        h = self.W0(x)
        W = torch.einsum("rb,bij->rij", self.coef, self.bases)
        for r, A in enumerate(self.adj):
            h = h + torch.sparse.mm(A, x @ W[r])
        return F.relu(h)

    def user(self, H, ment, mmask):
        hm = H[ment]                                                   # [B,M,d]
        a = self.att2(torch.tanh(self.att1(hm))).squeeze(-1).masked_fill(~mmask, -1e9)
        return (torch.softmax(a, -1).unsqueeze(-1) * hm).sum(1)

    def scores(self, H, u):
        return u @ H[:self.n_items].T + self.bias


def kbrd_mentions(hist, attrs, n_items, rng, n_hist=10):
    h = list(hist)
    if len(h) > n_hist:
        h = rng.choice(h, n_hist, replace=False).tolist()
    return h + [n_items + a for a in attrs]


def train_kbrd(model, d, attr_matrix, steps=2000, batch=256, lr=1e-3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    users, items, hist = positives(d)
    item_attrs = [np.nonzero(attr_matrix[i])[0] for i in range(attr_matrix.shape[0])]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=REG)
    t0, tot = time.time(), 0.0
    for s in range(steps):
        sel = rng.integers(0, len(users), size=batch)
        M = []
        for u, i in zip(users[sel], items[sel]):
            ia = item_attrs[i]
            k = int(rng.integers(0, min(3, len(ia)) + 1))
            others = [x for x in hist[u] if x != i]
            M.append(kbrd_mentions(others, rng.choice(ia, k, replace=False).tolist() if k else [],
                                   model.n_items, rng) or [model.n_items + int(ia[0])])
        mi, mm = padded(M, "cpu")
        H = model.entities()
        loss = F.cross_entropy(model.scores(H, model.user(H, mi, mm)), torch.as_tensor(items[sel]))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
        if (s + 1) % 250 == 0:
            log(f"  [KBRD] step {s+1}/{steps} CE={tot/250:.4f} ({time.time()-t0:.0f}s)")
            tot = 0.0


class KBRDAgent:
    def __init__(self, model, attr_matrix, user_hist, tau=0.1, seed=0):
        self.m, self.am, self.tau = model, attr_matrix, tau
        with torch.no_grad():
            self.H = model.entities()
        rng = np.random.default_rng(seed)
        self.ment = {u: kbrd_mentions(h, [], model.n_items, rng) for u, h in user_hist.items()}

    def begin(self, st):
        pass

    @torch.no_grad()
    def prepare(self, st, cands):
        M = self.ment.get(st.u, []) + [self.m.n_items + a for a in st.pos]
        mi = torch.as_tensor(M)[None]
        u = self.m.user(self.H, mi, torch.ones_like(mi, dtype=torch.bool))
        self.scores = self.m.scores(self.H, u)[0][cands].numpy()
        self.ent = attr_entropy(self.am, cands)

    def decide(self, st, cands, cand_attrs, rng):
        p = np.exp(self.scores - self.scores.max())
        p /= p.sum()
        top = np.sort(p)[-TOPK:].sum()
        return "recommend" if (top >= self.tau or len(cands) <= TOPK) else "ask"

    def pick_attr(self, st, cands, cand_attrs):
        return cand_attrs[int(np.argmax(self.ent[cand_attrs]))]

    def rank(self, st, cands):
        return self.scores


# ==========================================================================
# Explainable CRSs for Table IV
# ==========================================================================
class FMExplainerBase:
    """Shared scoring + counterfactual hooks for FM-backed explainers."""

    def __init__(self, fm, attr_matrix):
        self.fm, self.am = fm, attr_matrix
        with torch.no_grad():
            self.X = fm.item_vec(torch.arange(attr_matrix.shape[0]))

    def begin(self, st):
        pass

    @torch.no_grad()
    def prepare(self, st, cands):
        self.q = self.fm.query(self.fm.U.weight[st.u], st.pos)

    @torch.no_grad()
    def rank(self, st, cands):
        return (self.X[cands] @ self.q).numpy()

    @torch.no_grad()
    def cf_score(self, st, item, keep):
        return float(self.fm.item_vec_with(item, keep) @ self.q)


class CPRAgent(FMExplainerBase):
    """CPR walks user -> accepted attribute -> item on the graph; the path's
    attribute nodes ARE the explanation of the recommended item."""

    def explain(self, st, item, kth):
        own = set(np.nonzero(self.am[item])[0].tolist())
        return sorted(st.pos & own)


class CAECNet(nn.Module):
    """Category-aware preference predictor (CAEC): encodes the conversation
    state and predicts the user's preference for every category; items are
    scored by their categories' predicted preference plus a CF term."""

    def __init__(self, fm, n_attrs, dim=DIM):
        super().__init__()
        self.fm = fm
        self.enc = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.C = nn.Embedding(n_attrs, dim)
        nn.init.normal_(self.C.weight, std=0.1)
        self.cb = nn.Parameter(torch.zeros(n_attrs))

    def state(self, eu, p, n):
        return self.enc(torch.cat([eu, p, n], -1))

    def cat_logits(self, h):
        return h @ self.C.weight.T + self.cb

    def item_score(self, h, logits, items_attr_norm, x):
        return (items_attr_norm * logits.unsqueeze(-2)).sum(-1) + (h.unsqueeze(-2) * x).sum(-1)


def _state_batch(fm, users, pa, na):
    pi, pm = padded(pa, "cpu"); ni, nm = padded(na, "cpu")
    return (fm.U(torch.as_tensor(users)), masked_mean(fm.A.weight, pi, pm, DIM),
            masked_mean(fm.A.weight, ni, nm, DIM))


def train_caec(net, d, attr_matrix, epochs=8, steps=150, batch=128, lr=1e-3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    am_t = torch.as_tensor(attr_matrix)
    amn = net.fm.am
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=REG)
    for ep in range(epochs):
        tot, t0 = 0.0, time.time()
        for _ in range(steps):
            users, tg, pa, na, rj, negs = sample_states(d, attr_matrix, rng, batch)
            eu, p, n = _state_batch(net.fm, users, pa, na)
            h = net.state(eu, p, n)
            lg = net.cat_logits(h)
            tg_t = torch.as_tensor(tg)
            l_cat = F.binary_cross_entropy_with_logits(lg, am_t[tg_t])
            ng = torch.as_tensor(negs)
            sp = net.item_score(h, lg, amn[tg_t].unsqueeze(1), net.fm.V(tg_t).unsqueeze(1))[:, 0]
            sn = net.item_score(h, lg, amn[ng], net.fm.V(ng))
            loss = l_cat + F.softplus(sn - sp.unsqueeze(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"  [CAEC] epoch {ep+1}/{epochs} loss={tot/steps:.4f} ({time.time()-t0:.1f}s)")


class CAECAgent:
    def __init__(self, net, attr_matrix):
        self.net, self.am = net, attr_matrix

    def begin(self, st):
        pass

    @torch.no_grad()
    def prepare(self, st, cands):
        eu, p, n = _state_batch(self.net.fm, [st.u], [sorted(st.pos)], [sorted(st.neg)])
        self.h = self.net.state(eu, p, n)[0]
        self.lg = self.net.cat_logits(self.h)

    @torch.no_grad()
    def rank(self, st, cands):
        c = torch.as_tensor(cands)
        return self.net.item_score(self.h, self.lg, self.net.fm.am[c], self.net.fm.V(c)).numpy()

    @torch.no_grad()
    def explain(self, st, item, kth):
        """The item's categories the user is predicted to prefer."""
        own = np.nonzero(self.am[item])[0]
        pr = torch.sigmoid(self.lg[own]).numpy()
        sel = own[pr >= 0.5]
        return sel.tolist() if len(sel) else [int(own[int(np.argmax(pr))])]

    @torch.no_grad()
    def cf_score(self, st, item, keep):
        keep = sorted(keep)
        cat = self.lg[keep].sum() / np.sqrt(len(keep)) if keep else 0.0
        return float(cat + self.h @ self.net.fm.V.weight[item])


class SAURNet(nn.Module):
    """Multi-memory network: hop 1 attends over the user's historical records
    (item memory), hop 2 over the aspects stated in the conversation."""

    def __init__(self, fm, dim=DIM):
        super().__init__()
        self.fm = fm
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, eu, mem, mmask, asp, amask):
        a1 = ((mem @ eu.unsqueeze(-1)).squeeze(-1)).masked_fill(~mmask, -1e9)
        a1 = torch.softmax(a1, -1)
        q1 = eu + self.W1((a1.unsqueeze(-1) * mem).sum(1))
        a2 = ((asp @ q1.unsqueeze(-1)).squeeze(-1)).masked_fill(~amask, -1e9)
        a2 = torch.softmax(a2, -1) * amask.any(1, keepdim=True)
        q = q1 + self.W2((a2.unsqueeze(-1) * asp).sum(1))
        return q, a1, a2


def saur_memory(fm, hist_lists):
    mi, mm = padded(hist_lists, "cpu")
    mem = fm.item_vec(mi.reshape(-1)).reshape(*mi.shape, -1)
    return mi, mem, mm


def train_saur(net, d, attr_matrix, user_mem, epochs=8, steps=150, batch=128, lr=1e-3,
               seed=0, log=print):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=REG)
    fm = net.fm
    for ep in range(epochs):
        tot, t0 = 0.0, time.time()
        for _ in range(steps):
            users, tg, pa, na, rj, negs = sample_states(d, attr_matrix, rng, batch)
            _, mem, mm = saur_memory(fm, [[x for x in user_mem.get(u, []) if x != t] or [0]
                                          for u, t in zip(users, tg)])
            ai, am_ = padded(pa, "cpu")
            q, _, _ = net(fm.U(torch.as_tensor(users)), mem, mm, fm.A(ai), am_)
            sp = (q * fm.item_vec(torch.as_tensor(tg))).sum(-1)
            xn = fm.item_vec(torch.as_tensor(negs).reshape(-1)).reshape(*negs.shape, -1)
            loss = F.softplus((q.unsqueeze(1) * xn).sum(-1) - sp.unsqueeze(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log(f"  [SAUR] epoch {ep+1}/{epochs} loss={tot/steps:.4f} ({time.time()-t0:.1f}s)")


class SAURAgent(FMExplainerBase):
    def __init__(self, net, attr_matrix, user_mem):
        super().__init__(net.fm, attr_matrix)
        self.net, self.user_mem = net, user_mem

    @torch.no_grad()
    def prepare(self, st, cands):
        hist = self.user_mem.get(st.u, []) or [int(cands[0])]
        mi, mem, mm = saur_memory(self.fm, [hist])
        pos = sorted(st.pos)
        ai, am_ = padded([pos], "cpu")
        self.q, a1, a2 = self.net(self.fm.U.weight[st.u][None], mem, mm, self.fm.A(ai), am_)
        self.q = self.q[0]
        # aspect relevance: attention mass of the records (and stated aspects)
        # that carry each aspect -- "search records as explanations"
        rel = a1[0].numpy() @ self.am[mi[0].numpy()]
        for k, a in enumerate(pos):
            rel[a] += float(a2[0, k])
        self.rel = rel

    @torch.no_grad()
    def explain(self, st, item, kth):
        own = np.nonzero(self.am[item])[0]
        r = self.rel[own]
        if r.max() <= 0:
            return []
        return own[r >= 0.5 * r.max()].tolist()
