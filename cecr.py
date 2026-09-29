"""
Core reproduction of CECR (Section IV of the paper).

Implements, following the paper's own equation numbers in comments:
  - Preference refinement          (Eqs. 9-13)
  - Counterfactual explanation      (Eqs. 14-18)   <-- the XAI core, implemented in full
  - Counterfactual sample generation / retraining  (Eq. 19)
  - Recommend-or-ask decision       (Eq. 20)
  - The full conversation loop      (Algorithm 1)

Two documented simplifications vs. the paper (everything else follows it directly):
  1. e_u/e_i/e_a offline embeddings come from a lightweight BPR-MF instead of a
     multi-layer GNN (see base_embeddings.py).
  2. The neural gating functions C-_u, C+_u, C-_i (Eqs. 9-11) are replaced with an
     equivalent *linear* preference-shift: since those gates would otherwise need
     their own (W3, W4, W5) weights trained jointly with the recommender -- out of
     scope for a lightweight reproduction -- we keep the same mechanism (long-term
     preference pulled towards liked attributes / pushed away from disliked ones)
     without the extra untrained neural layers. This is the "preference refinement"
     component tested in the ablation (PreRi).
The scoring function s(u,i) = e_overall . e_i is the same dot-product form the
paper itself uses at the final recommendation stage (Eq. 20); we use it
consistently everywhere instead of the deeper MLP of Eq. 14, for transparency.
"""
import numpy as np

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class CECR:
    def __init__(self, eu, ei, ea, attr_matrix, dim,
                 use_pref_refine=True, use_counterfactual=True, use_score_decision=True,
                 eta=1.0, lr_cf=0.15, cf_iters=60, l2=0.05, l1=0.08, lam=6.0, seed=0,
                 ranker=None):
        # base (offline) representations - copies are refined per-conversation
        self.eu0, self.ei0, self.ea0 = eu, ei, ea
        self.attr_matrix = attr_matrix           # [n_items, n_attrs] binary
        self.dim = dim
        self.n_items, self.n_attrs = attr_matrix.shape

        # Eq. 14 / Algorithm 1's trained ranking network g. When present, this
        # (not the raw dot product) drives Top-K ranking and the counterfactual
        # score s_u,i -- the dot product remains ONLY for the ask-vs-recommend
        # decision (Eq. 20, decide_action), which the paper genuinely defines
        # as a plain dot product.
        self.ranker = ranker

        # ablation switches
        self.use_pref_refine = use_pref_refine
        self.use_counterfactual = use_counterfactual
        self.use_score_decision = use_score_decision

        # hyperparameters
        self.eta = eta            # strength of attribute-level preference shift (Eqs 9-13 analogue)
        self.lr_cf = lr_cf        # learning rate for counterfactual Delta search (Eq 18)
        self.cf_iters = cf_iters
        self.l2, self.l1, self.lam = l2, l1, lam   # Eq (17) coefficients
        self.rng = np.random.default_rng(seed)

    # ---------- Preference refinement (Eqs. 9-13) ----------
    def refine_user(self, eu, pos_attrs, neg_attrs):
        """Update long-term user rep with attribute-level online feedback."""
        if not self.use_pref_refine or (not pos_attrs and not neg_attrs):
            return eu
        eA_pos = self.ea0[list(pos_attrs)].mean(axis=0) if pos_attrs else np.zeros(self.dim)
        eA_neg = self.ea0[list(neg_attrs)].mean(axis=0) if neg_attrs else np.zeros(self.dim)
        # linear analogue of the C+_u / C-_u gating (Eq. 9-10): pull towards liked
        # attributes, push away from disliked ones
        e_prime_u = eu + self.eta * (eA_pos - eA_neg)
        return e_prime_u

    def refine_rejected_item(self, ei_vec, pos_attrs):
        """Distill positive info out of a rejected item (Eq. 11)."""
        if not self.use_pref_refine or not pos_attrs:
            return ei_vec
        eA_pos = self.ea0[list(pos_attrs)].mean(axis=0)
        gate = SIGMOID(eA_pos @ ei_vec)          # scalar gate, analogue of C-_i
        return ei_vec * gate + eA_pos * (1 - gate)

    def overall_preference(self, e_prime_u, rejected_item_reps):
        """Eq. 12-13: eoverall = e'_u - e-_I """
        if not rejected_item_reps:
            return e_prime_u
        e_neg_I = np.mean(rejected_item_reps, axis=0)
        return e_prime_u - e_neg_I

    def score_item(self, eoverall, item_vec):
        """s_u,i (Eq. 14): via the trained ranker g if present, else raw dot product."""
        if self.ranker is not None:
            return self.ranker.score(eoverall, item_vec)
        return eoverall @ item_vec

    # ---------- Counterfactual reasoning (Eqs. 14-18): the XAI core ----------
    def counterfactual_explanation(self, eoverall, item_vec, item_attrs, s_ik):
        """
        Search for the smallest attribute-level change vector Delta (one entry
        per attribute) that removes `item_vec` from being preferred over the
        k-th ranked item (score s_ik), following Eqs. 15-18.
        Returns: Delta, explanation-strength (ES), explanation-complexity (EC),
                 and the resulting counterfactual item embedding.
        """
        d = self.n_attrs
        delta = np.zeros(d)
        # Only attributes the item actually possesses are eligible for change:
        # Delta represents "how much of attribute a's presence to remove from item i",
        # which is only meaningful for attributes a in item_attrs (Fig. 5's case study
        # explains rejection via one of the item's own genres, e.g. "Mystery").
        mask = (item_attrs > 0).astype(float)

        def score(vec):
            return self.score_item(eoverall, vec)

        def score_and_grad(vec):
            if self.ranker is not None:
                return self.ranker.score_and_grad_wrt_item(eoverall, vec)
            return eoverall @ vec, eoverall   # d(dot product)/d(vec) = eoverall

        s_orig = score(item_vec)

        for t in range(self.cf_iters):
            item_delta_vec = item_vec - (delta * mask) @ self.ea0   # Eq: e_i + Delta (attribute-level)
            s_delta, d_s_d_vec = score_and_grad(item_delta_vec)
            # ranking (BPR-style) term of Eq. 17, gradient derived in closed form:
            # d/d(s_delta) [-log sigmoid(s_ik - s_delta)] = sigmoid(s_delta - s_ik)
            rank_grad_scalar = SIGMOID(s_delta - s_ik)
            grad_score = -(self.ea0 @ d_s_d_vec) * mask               # d(s_delta)/d(delta)  -> [n_attrs]

            grad = 2 * self.l2 * delta + self.l1 * np.sign(delta) + self.lam * rank_grad_scalar * grad_score
            delta = delta - self.lr_cf * grad
            delta = np.clip(delta, 0.0, 1.0) * mask

        item_delta_vec = item_vec - (delta * mask) @ self.ea0
        s_delta_final = score(item_delta_vec)
        ES = s_orig - s_delta_final                              # Eq. 15
        EC = np.sum(delta > 0.05)                                 # Eq. 16-17 complexity (support size)
        explanation_attrs = np.where(delta > 0.05)[0]
        return delta, ES, EC, explanation_attrs, item_delta_vec

    # ---------- Attribute-based candidate filtering ----------
    def filter_candidates(self, candidate_items, pos_attrs, neg_attrs):
        """
        Hard-filter candidates by attribute match before scoring: keep only
        items carrying at least one liked attribute and none of the disliked
        ones. This is needed because the soft preference-shift in
        refine_user() alone is too weak to reliably steer the ranking
        (attribute embeddings are much smaller in magnitude than user
        embeddings), so without this filter genre choices barely affect which
        item comes out on top.
        """
        if not candidate_items:
            return candidate_items
        cand = np.array(candidate_items)
        sub = self.attr_matrix[cand]
        if pos_attrs:
            mask_pos = sub[:, list(pos_attrs)].sum(axis=1) > 0
        else:
            mask_pos = np.ones(len(cand), dtype=bool)
        if neg_attrs:
            mask_neg = sub[:, list(neg_attrs)].sum(axis=1) == 0
        else:
            mask_neg = np.ones(len(cand), dtype=bool)
        both = cand[mask_pos & mask_neg]
        if len(both) > 0:
            return both.tolist()
        pos_only = cand[mask_pos]
        if len(pos_only) > 0:
            return pos_only.tolist()
        return cand.tolist()   # fallback: no item matches at all, don't filter

    # ---------- Recommendation decision (Eq. 20) ----------
    def decide_action(self, eoverall, candidate_items, asked_attrs):
        item_scores = eoverall @ self.ei0[candidate_items].T if len(candidate_items) else np.array([])
        remaining_attrs = [a for a in range(self.n_attrs) if a not in asked_attrs]
        attr_scores = eoverall @ self.ea0[remaining_attrs].T if remaining_attrs else np.array([])

        if not self.use_score_decision:
            # RecPe ablation: choose an action at random instead of comparing scores
            return self.rng.choice(["recommend", "ask"]), remaining_attrs

        best_item_score = item_scores.max() if len(item_scores) else -1e9
        best_attr_score = attr_scores.max() if len(attr_scores) else -1e9
        if best_item_score >= best_attr_score or not remaining_attrs:
            return "recommend", remaining_attrs
        else:
            return "ask", remaining_attrs

    def top_k(self, eoverall, candidate_items, k=10):
        """Algorithm 1, step 10: rank by the trained model g (Eq. 14) when
        available, not the raw dot product (which Eq. 20 reserves for the
        ask-vs-recommend decision only)."""
        if self.ranker is not None:
            scores = self.ranker.score_batch(eoverall, self.ei0[candidate_items])
        else:
            scores = eoverall @ self.ei0[candidate_items].T
        order = np.argsort(-scores)[:k]
        return [candidate_items[i] for i in order], scores[order]
