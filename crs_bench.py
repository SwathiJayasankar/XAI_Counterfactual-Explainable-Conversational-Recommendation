"""
Shared multi-round CRS benchmark harness -- CECR and every baseline run through
THIS loop, so Table II / Fig. 3 / Fig. 4 / Fig. 6 / Table IV compare models on
identical conversations.

Protocol (paper Section IV-D-1 / V-A, the EAR/FPAN/UNICORN/CPR convention):
  * test conversations = held-out (user, liked item) pairs, rating >= 4
  * the simulated user opens with one attribute of the target item; the first
    attribute is drawn from a PER-CONVERSATION rng, so it is the same for every
    model no matter how much randomness a model consumes internally
  * each round the agent either asks one attribute (yes iff the target has it)
    or recommends its Top-K; a recommendation succeeds iff the target is in it
  * candidates = items carrying all accepted attributes, none of the rejected
    attributes, and not already rejected
  * at most MAX_ROUNDS = 20 rounds; SR@r and AT (failures count as 20)

Agent interface (duck-typed):
  begin(state)                         new conversation
  prepare(state, cands)                once per round, before any query
  decide(state, cands, cand_attrs, rng) -> "ask" | "recommend"
  pick_attr(state, cands, cand_attrs) -> attribute id
  rank(state, cands) -> np.ndarray of scores (higher = better)
  feedback(state, kind, reward, done)  optional, used by the RL agents
  reflect(state, rejected_items)       optional (EAR's reflection stage)
"""
import time

import numpy as np
import pandas as pd
import torch

from cecr_torch import candidate_set, counterfactual_explanation

MAX_ROUNDS = 20
TOPK = 10

# reward shaping shared by the RL baselines (EAR / CPR / UNICORN papers)
REWARDS = dict(rec_suc=1.0, rec_fail=-0.1, ask_suc=0.01, ask_fail=-0.1, quit=-0.3)


class State:
    def __init__(self, u, first):
        self.u = u
        self.pos = {first}
        self.neg = set()
        self.asked = {first}
        self.rejected = []
        self.history = []          # per-round outcome: ask_suc / ask_fail / rec_fail
        self.round = 0


def test_pairs(d, n, seed=7, split="test"):
    """The same (user, liked item) pairs eval_cecr.py has always used."""
    df = d[split]
    liked = df[df.rating >= 4]
    rng = np.random.default_rng(seed)
    arr = liked[["user", "item"]].values
    sel = rng.choice(len(arr), min(n, len(arr)), replace=False)
    return [(int(u), int(i)) for u, i in arr[sel]]


def attr_entropy(attr_matrix, cands, weights=None):
    """Binary entropy of each attribute over the candidate set (optionally
    preference-weighted, as in CPR/UNICORN's weighted entropy)."""
    sub = attr_matrix[cands]
    if weights is None:
        p = sub.mean(0)
    else:
        w = weights / max(weights.sum(), 1e-12)
        p = w @ sub
    p = np.clip(p.astype(np.float64), 1e-9, 1 - 1e-9)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype(np.float32)


def run_conversation(agent, u, tgt, attr_matrix, rng, max_rounds=MAX_ROUNDS, topk=TOPK):
    """One simulated conversation; returns the success round or None."""
    n_items = attr_matrix.shape[0]
    t_attrs = set(np.nonzero(attr_matrix[tgt])[0].tolist())
    if not t_attrs:
        return None
    st = State(u, int(rng.choice(sorted(t_attrs))))
    agent.begin(st)
    fb = getattr(agent, "feedback", None)
    for rnd in range(1, max_rounds + 1):
        st.round = rnd
        cands = candidate_set(attr_matrix, st.pos, st.neg, st.rejected, n_items)
        if len(cands) == 0:
            break
        present = np.nonzero(attr_matrix[cands].sum(0) > 0)[0]
        cand_attrs = [int(a) for a in present if a not in st.asked]
        agent.prepare(st, cands)
        action = agent.decide(st, cands, cand_attrs, rng) if cand_attrs else "recommend"
        last = rnd == max_rounds
        if action == "recommend":
            scores = np.asarray(agent.rank(st, cands))
            k = min(topk, len(cands))
            top = cands[np.argsort(-scores, kind="stable")[:k]]
            if tgt in top:
                if fb:
                    fb(st, "rec_suc", REWARDS["rec_suc"], True)
                return rnd
            st.rejected.extend(int(x) for x in top)
            st.history.append("rec_fail")
            if hasattr(agent, "reflect"):
                agent.reflect(st, top)
            if fb:
                fb(st, "rec_fail", REWARDS["quit"] if last else REWARDS["rec_fail"], last)
        else:
            a = int(agent.pick_attr(st, cands, cand_attrs))
            st.asked.add(a)
            ok = a in t_attrs
            (st.pos if ok else st.neg).add(a)
            st.history.append("ask_suc" if ok else "ask_fail")
            if fb:
                r = REWARDS["quit"] if last else REWARDS["ask_suc" if ok else "ask_fail"]
                fb(st, st.history[-1], r, last)
    if fb:
        fb(st, "quit", 0.0, True)       # closes any open transition
    return None


def summarize(rounds, max_rounds=MAX_ROUNDS):
    out = {f"SR@{r}": float(np.mean([x is not None and x <= r for x in rounds]))
           for r in (5, 10, 20)}
    out["AT"] = float(np.mean([x if x is not None else max_rounds for x in rounds]))
    out["curve"] = [float(np.mean([x is not None and x <= r for x in rounds]))
                    for r in range(1, max_rounds + 1)]
    out["n"] = len(rounds)
    return out


def evaluate(agent, pairs, attr_matrix, seed=7, max_rounds=MAX_ROUNDS, topk=TOPK):
    """SR@5/10/20, AT, the per-round SR curve, and mean response time."""
    rounds, turns, secs = [], 0, 0.0
    for idx, (u, tgt) in enumerate(pairs):
        rng = np.random.default_rng([seed, idx])
        t0 = time.perf_counter()
        r = run_conversation(agent, u, tgt, attr_matrix, rng, max_rounds, topk)
        secs += time.perf_counter() - t0
        turns += r if r is not None else max_rounds
        rounds.append(r)
    out = summarize(rounds, max_rounds)
    out["sec_per_conv"] = secs / max(len(pairs), 1)
    out["ms_per_turn"] = 1000 * secs / max(turns, 1)
    return out


# --------------------------------------------------------------------------
# CECR wrapped as an agent (Algorithm 1 via Eq. 14 ranking + Eq. 20 decision)
# --------------------------------------------------------------------------
class CECRAgent:
    def __init__(self, model, e_u, e_i, e_a, attr_matrix,
                 use_pref_refine=True, use_score_decision=True):
        self.m, self.e_u, self.e_i, self.e_a = model, e_u, e_i, e_a
        self.attr_matrix = attr_matrix
        self.use_pref_refine = use_pref_refine
        self.use_score_decision = use_score_decision

    def begin(self, st):
        pass

    @torch.no_grad()
    def prepare(self, st, cands):
        self.eo = self.m.overall(self.e_u[st.u], st.pos, st.neg, st.rejected,
                                 self.e_a, self.e_i, self.use_pref_refine)

    @torch.no_grad()
    def decide(self, st, cands, cand_attrs, rng):
        if not self.use_score_decision:
            return rng.choice(["ask", "recommend"])            # RecPe ablation
        return self.m.decide_action(self.eo, cands, cand_attrs, self.e_i, self.e_a)[0]

    @torch.no_grad()
    def pick_attr(self, st, cands, cand_attrs):
        return cand_attrs[int(torch.argmax(self.e_a[cand_attrs] @ self.eo))]

    @torch.no_grad()
    def rank(self, st, cands):
        return self.m.rank(self.eo, self.e_i[cands]).cpu().numpy()

    # ---- explainability hooks (Table IV) ----
    def explain(self, st, item, kth_score, lam=100.0, cf_iters=60):
        """Eqs. 15-18: the counterfactual attribute change Delta."""
        _, _, _, expl, _ = counterfactual_explanation(
            self.m, self.eo, self.e_i[item],
            torch.as_tensor(self.attr_matrix[item]), self.e_a,
            torch.as_tensor(kth_score), lam=lam, iters=cf_iters)
        return [int(a) for a in expl]

    @torch.no_grad()
    def cf_score(self, st, item, keep_attrs):
        """Score of `item` when only `keep_attrs` of its attributes remain:
        removed attributes are subtracted in embedding space, exactly as the
        counterfactual item i_Delta of Eq. 18 is built."""
        own = np.nonzero(self.attr_matrix[item])[0]
        drop = [int(a) for a in own if int(a) not in keep_attrs]
        e = self.e_i[item] - (self.e_a[drop].sum(0) if drop else 0.0)
        return float(self.m.rank(self.eo, e))


# --------------------------------------------------------------------------
# Table IV: explainability, model-agnostic
# --------------------------------------------------------------------------
def liked_attr_profile(d, attr_matrix):
    """Ground truth p_{u,i}^(a) of Eq. 21 = "user u has positive feedback on
    attribute a". Built from the user's HELD-OUT (val+test) liked items so it
    is independent of both the training data and the conversation itself:
    a is liked by u if it occurs among u's held-out liked items more often than
    its base rate across all training interactions (lift > 1)."""
    held = pd.concat([d["val"], d["test"]])
    held = held[held.rating >= 4]
    base = attr_matrix[d["train"].item.values].mean(0)
    items_by_user = held.groupby("user")["item"].apply(list).to_dict()
    return items_by_user, base


def liked_attrs(profile, attr_matrix, u, exclude_item):
    items_by_user, base = profile
    its = [i for i in items_by_user.get(u, []) if i != exclude_item]
    if not its:
        return set()
    share = attr_matrix[its].mean(0)
    return set(np.nonzero((share > base) & (share > 0))[0].tolist())


def explanation_states(d, attr_matrix, pairs, n_states, seed=7, n_asks=2):
    """Model-independent conversation states for Table IV: the opening
    attribute plus `n_asks` max-entropy questions, identical for every model."""
    n_items = attr_matrix.shape[0]
    out = []
    for idx, (u, tgt) in enumerate(pairs):
        if len(out) >= n_states:
            break
        rng = np.random.default_rng([seed, idx])
        t_attrs = set(np.nonzero(attr_matrix[tgt])[0].tolist())
        if not t_attrs:
            continue
        st = State(u, int(rng.choice(sorted(t_attrs))))
        for _ in range(n_asks):
            cands = candidate_set(attr_matrix, st.pos, st.neg, [], n_items)
            ca = [a for a in np.nonzero(attr_matrix[cands].sum(0) > 0)[0] if a not in st.asked]
            if not ca:
                break
            ent = attr_entropy(attr_matrix, cands)
            a = int(ca[int(np.argmax(ent[ca]))])
            st.asked.add(a)
            (st.pos if a in t_attrs else st.neg).add(a)
        cands = candidate_set(attr_matrix, st.pos, st.neg, [], n_items)
        if len(cands) > TOPK:
            out.append((st, tgt, cands))
    return out


def eval_explainability(agent, states, attr_matrix, profile, topk=TOPK, rng=None,
                        random_expl=False):
    """Eq. 21 (Precision / Recall) and Eq. 22 (PN / PS) over each state's
    Top-K list R_{u,K}. PN: drop the explanation attributes -> item must leave
    the Top-K. PS: keep only the explanation attributes -> item must stay
    (standard sufficiency of the paper's ref. [47]; the printed Eq. 22 repeats
    PN's condition, see explain.txt)."""
    rng = rng or np.random.default_rng(0)
    n_attrs = attr_matrix.shape[1]
    P, R, PN, PS = [], [], [], []
    for st, tgt, cands in states:
        agent.begin(st)
        agent.prepare(st, cands)
        scores = np.asarray(agent.rank(st, cands))
        order = np.argsort(-scores, kind="stable")[:topk]
        kth = float(scores[order[-1]])
        liked = liked_attrs(profile, attr_matrix, st.u, tgt)
        for it in cands[order]:
            it = int(it)
            own = set(np.nonzero(attr_matrix[it])[0].tolist())
            if not own:
                continue
            if random_expl:
                expl = rng.choice(n_attrs, size=int(rng.integers(1, 4)), replace=False).tolist()
            else:
                expl = agent.explain(st, it, kth)
            if not expl:
                continue
            expl = set(int(a) for a in expl)
            p = liked & own                         # p_{u,i}^(a) = 1
            hit = len(p & expl)
            P.append(hit / len(expl))
            R.append(hit / len(p) if p else 0.0)
            expl_owned = expl & own
            PN.append(1.0 if agent.cf_score(st, it, own - expl_owned) < kth else 0.0)
            PS.append(1.0 if agent.cf_score(st, it, expl_owned) >= kth else 0.0)
    m = lambda x: float(np.mean(x)) if x else 0.0
    return dict(Precision=m(P), Recall=m(R), PN=m(PN), PS=m(PS), n=len(P))
