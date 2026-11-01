"""
Improvement #3: CECR's ask-vs-recommend decision.

The paper's Eq. 20 compares max_i e_i . e_overall against max_a e_a . e_overall.
Item and attribute embeddings come out of different GNN branches with different
norms, so the comparison is dominated by embedding scale rather than by whether
asking or recommending is the better move. Two replacements, both keeping
CECR's ranking, attribute choice and explanations untouched:

  * "cosine" : Eq. 20 on cosine similarities (scale-free, no training)
  * "policy" : a small policy network over the conversation state, pre-trained
               to imitate Eq. 20 and then optimised with REINFORCE on training
               conversations, as EAR does (baselines.train_policy_reinforce)
  * "threshold": ask while more than `threshold` candidates remain, else
               recommend; the threshold is chosen on validation conversations
"""
import numpy as np
import torch
import torch.nn.functional as F

from baselines import ACTIONS, PolicyNet, len_bin
from crs_bench import MAX_ROUNDS, TOPK, CECRAgent, attr_entropy

N_TOP = TOPK + 1                 # Top-K scores plus the first item outside it


class CECRDecisionAgent(CECRAgent):
    """CECR with a replaceable decision rule. decision in {"eq20", "cosine", "policy"}."""

    def __init__(self, model, e_u, e_i, e_a, attr_matrix, decision="policy", pick="pref",
                 threshold=200):
        super().__init__(model, e_u, e_i, e_a, attr_matrix)
        self.decision = decision
        self.threshold = threshold
        self.pick = pick             # "pref" = CECR's argmax e_a . e_overall; "entropy" = EAR-style
        self.policy = PolicyNet(10 + 3 * MAX_ROUNDS + N_TOP + 8)
        self.training = False
        self.mode = "policy"         # "rule" = Eq. 20, used for imitation pre-training
        self.e_i_n = F.normalize(e_i, dim=-1)
        self.e_a_n = F.normalize(e_a, dim=-1)

    def begin(self, st):
        self.logps, self.ents, self.rewards, self._decided = [], [], [], False

    @torch.no_grad()
    def prepare(self, st, cands):
        super().prepare(st, cands)
        self.scores = self.m.rank(self.eo, self.e_i[cands])

    @torch.no_grad()
    def features(self, st, cands, cand_attrs):
        ln = np.zeros(10, dtype=np.float32)
        ln[len_bin(len(cands))] = 1.0
        his = np.zeros((MAX_ROUNDS, 3), dtype=np.float32)
        for t, h in enumerate(st.history[:MAX_ROUNDS]):
            his[t, ("ask_suc", "ask_fail", "rec_fail").index(h)] = 1.0
        # Top-K ranker scores, standardised over the candidate pool, so the
        # policy sees how far the leaders stand out from the rest
        s = self.scores
        z = (s - s.mean()) / (s.std() + 1e-6) if len(s) > 1 else s * 0
        top = torch.zeros(N_TOP)
        k = min(N_TOP, len(z))
        top[:k] = torch.topk(z, k).values
        eo_n = F.normalize(self.eo, dim=-1)
        best_i = (self.e_i[cands] @ self.eo).max()
        best_a = (self.e_a[cand_attrs] @ self.eo).max()
        cos_i = (self.e_i_n[cands] @ eo_n).max()
        cos_a = (self.e_a_n[cand_attrs] @ eo_n).max()
        ent = attr_entropy(self.attr_matrix, cands)[cand_attrs].max()
        extra = torch.tensor([
            st.round / MAX_ROUNDS,
            len(cands) <= TOPK,
            np.log1p(len(cands)) / 10,
            float(ent),
            torch.tanh(best_i - best_a),           # Eq. 20's margin
            cos_i - cos_a,                         # scale-free margin
            len(st.rejected) / (TOPK * MAX_ROUNDS),
            len(cand_attrs) / self.attr_matrix.shape[1],
        ], dtype=torch.float32)
        return torch.cat([torch.as_tensor(ln), torch.as_tensor(his.ravel()), top, extra])

    def decide(self, st, cands, cand_attrs, rng):
        if self.decision == "eq20" or self.mode == "rule":
            return super().decide(st, cands, cand_attrs, rng)
        if len(cands) <= TOPK:
            return "recommend"                     # the whole pool fits in one list
        if self.decision == "threshold":
            return "ask" if len(cands) > self.threshold else "recommend"
        if self.decision == "cosine":
            eo_n = F.normalize(self.eo, dim=-1)
            ci = (self.e_i_n[cands] @ eo_n).max()
            ca = (self.e_a_n[cand_attrs] @ eo_n).max()
            return "recommend" if ci >= ca else "ask"
        logits = self.policy(self.features(st, cands, cand_attrs))
        if self.training:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            self.logps.append(dist.log_prob(a))
            self.ents.append(dist.entropy())
            self._decided = True
            return ACTIONS[int(a)]
        return ACTIONS[int(torch.argmax(logits))]

    @torch.no_grad()
    def pick_attr(self, st, cands, cand_attrs):
        if self.pick == "pref":
            return super().pick_attr(st, cands, cand_attrs)
        # preference-weighted entropy: questions that split the pool AND that
        # the user plausibly cares about (EAR / CPR's weighted entropy)
        ent = attr_entropy(self.attr_matrix, cands)[cand_attrs]
        pref = torch.sigmoid(self.e_a[cand_attrs] @ self.eo).numpy()
        return cand_attrs[int(np.argmax(ent * pref))]

    def feedback(self, st, kind, reward, done):
        if not self.training:
            return
        if self._decided:
            self.rewards.append(reward)
            self._decided = False
        elif self.rewards:
            self.rewards[-1] += reward
