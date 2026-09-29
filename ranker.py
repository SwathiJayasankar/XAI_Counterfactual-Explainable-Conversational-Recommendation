"""
The "target model g" of Eq. 14 / Algorithm 1 -- a trained ranking network,
previously MISSING from this reproduction (top_k and the counterfactual score
s_u,i were both a raw dot product e_overall . e_i everywhere).

Per the paper:
  - Algorithm 1, step 4:  "Train the model g of (14) by the original dataset O"
  - Algorithm 1, step 10: "Recommend Top-K item i to user u based on model g of (14)"
  - Algorithm 1, step 19: "Re-train the target model g of (14) by O U (u,i,ik)"
  - Eq. 20's plain dot product is used ONLY for the ask-vs-recommend DECISION,
    not for ranking the actual Top-K list -- that distinction is kept here:
    cecr.py's decide_action() still uses the raw dot product (unchanged),
    while top_k() and the counterfactual score s_u,i now go through this g.

Architecture (2-layer MLP, matching Eq. 14 with M=2):
    h0 = concat(e_u, e_i)                      [2*dim]
    h1 = ReLU(W1 . h0 + b1)                     [dim]      <- "g(e_u,e_i)" + 1st ReLU
    s  = W2 . h1 + b2                           scalar     <- final ranking score

Trained end-to-end with plain NumPy (manual forward + backward), BPR loss,
exactly like gnn_embeddings.py -- no framework needed.
"""
import numpy as np
import pickle

relu = lambda x: np.maximum(x, 0)
relu_grad = lambda pre: (pre > 0).astype(np.float32)


class Ranker:
    def __init__(self, dim, lr=0.02, reg=1e-3, seed=0):
        self.dim = dim
        self.lr, self.reg = lr, reg
        rng = np.random.default_rng(seed)
        init = lambda *shape: rng.normal(0, 0.05, shape).astype(np.float32)
        self.W1, self.b1 = init(dim, 2 * dim), np.zeros(dim, dtype=np.float32)
        self.W2, self.b2 = init(dim), 0.0

    def copy(self):
        r = Ranker(self.dim, self.lr, self.reg)
        r.W1, r.b1 = self.W1.copy(), self.b1.copy()
        r.W2, r.b2 = self.W2.copy(), self.b2.copy()
        return r

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, dim=self.dim)

    @classmethod
    def load(cls, path, lr=0.02, reg=1e-3):
        d = np.load(path)
        r = cls(int(d["dim"]), lr=lr, reg=reg)
        r.W1, r.b1, r.W2, r.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return r

    # ---------- forward ----------
    def _forward(self, eu, ei):
        h0 = np.concatenate([eu, ei])
        pre1 = self.W1 @ h0 + self.b1
        h1 = relu(pre1)
        s = self.W2 @ h1 + self.b2
        return s, (h0, pre1, h1)

    def score(self, eu, ei):
        """Single (u, i) score s_u,i."""
        s, _ = self._forward(eu, ei)
        return s

    def score_batch(self, eu, ei_batch):
        """Score one user against many candidate items at once."""
        h0 = np.concatenate([np.broadcast_to(eu, (len(ei_batch), self.dim)), ei_batch], axis=1)
        pre1 = h0 @ self.W1.T + self.b1
        h1 = relu(pre1)
        return h1 @ self.W2 + self.b2

    def score_and_grad_wrt_item(self, eu, ei):
        """
        Returns (s_u,i, d s_u,i / d e_i) -- needed by the counterfactual Delta
        search, which perturbs the ITEM embedding and needs the score's
        sensitivity to that perturbation.
        """
        s, (h0, pre1, h1) = self._forward(eu, ei)
        d_h1 = self.W2                          # ds/dh1
        d_pre1 = d_h1 * relu_grad(pre1)         # ds/dpre1
        d_h0 = self.W1.T @ d_pre1               # ds/dh0  [2*dim]
        d_ei = d_h0[self.dim:]                  # second half of concat = e_i
        return s, d_ei

    @staticmethod
    def _clip(g, max_norm=5.0):
        n = np.linalg.norm(g)
        if n > max_norm:
            g = g * (max_norm / (n + 1e-9))
        return g

    # ---------- BPR training ----------
    def train_step(self, eu, ei_pos, ei_neg):
        s_pos, (h0p, pre1p, h1p) = self._forward(eu, ei_pos)
        s_neg, (h0n, pre1n, h1n) = self._forward(eu, ei_neg)
        x = np.clip(s_pos - s_neg, -30, 30)
        sig = 1.0 / (1.0 + np.exp(-x))
        grad = (1 - sig)   # d(-log sigmoid(x))/dx = -(1-sig); ascend => +grad

        # ----- backprop for the positive branch (+grad), negative branch (-grad) -----
        for h0, pre1, h1, sign in [(h0p, pre1p, h1p, 1.0), (h0n, pre1n, h1n, -1.0)]:
            d_s = sign * grad
            self.W2 += self.lr * (self._clip(d_s * h1) - self.reg * self.W2)
            self.b2 += self.lr * np.clip(d_s, -1.0, 1.0)
            d_h1 = d_s * self.W2
            d_pre1 = self._clip(d_h1 * relu_grad(pre1))
            self.W1 += self.lr * (self._clip(np.outer(d_pre1, h0)) - self.reg * self.W1)
            self.b1 += self.lr * d_pre1
        return -np.log(sig + 1e-9)


def pretrain_ranker(d, eu0, ei0, ea0, dim, epochs=15, n_triples_per_epoch=8000, seed=0, eta=1.0):
    """
    Algorithm 1, step 4: 'Train the model g of (14) by the original dataset O.'

    IMPORTANT: at conversation time g is queried with e_overall (the
    preference-REFINED vector, Eqs. 9-13), not the raw offline e_u. If g is
    only ever trained on raw e_u it never sees inputs that look like
    e_overall and its ranking quality degrades on exactly the vectors it
    will actually be asked to score. So for a fraction of training triples
    we synthesize a plausible e_overall-like vector (a random attribute-level
    preference shift, mirroring cecr.py's refine_user) and train on THAT
    instead -- this mirrors Algorithm 1's real loop, which keeps re-training
    g on the round-by-round updated e'_u (step 19), not just the initial e_u.
    """
    rng = np.random.default_rng(seed)
    train_df = d["train"]
    n_items, n_attrs = d["n_items"], d["n_attrs"]
    pos_by_user = train_df[train_df.rating >= 4].groupby("user")["item"].apply(list).to_dict()
    users = np.array(list(pos_by_user.keys()))

    ranker = Ranker(dim, lr=0.01, reg=5e-3, seed=seed)
    for ep in range(epochs):
        idx = rng.choice(len(users), size=n_triples_per_epoch)
        total = 0.0
        for k in idx:
            u = int(users[k])
            i_pos = int(rng.choice(pos_by_user[u]))
            i_neg = int(rng.integers(0, n_items))
            while i_neg in pos_by_user[u]:
                i_neg = int(rng.integers(0, n_items))

            eu_vec = eu0[u]
            if rng.random() < 0.5:
                n_pos = int(rng.integers(1, 3))
                pos_a = rng.choice(n_attrs, size=n_pos, replace=False)
                eA_pos = ea0[pos_a].mean(axis=0)
                eA_neg = np.zeros(dim)
                if rng.random() < 0.5:
                    n_neg = int(rng.integers(1, 3))
                    neg_a = rng.choice(n_attrs, size=n_neg, replace=False)
                    eA_neg = ea0[neg_a].mean(axis=0)
                eu_vec = eu0[u] + eta * (eA_pos - eA_neg)

            total += ranker.train_step(eu_vec, ei0[i_pos], ei0[i_neg])
        print(f"  [Ranker] epoch {ep+1}/{epochs}  avg BPR loss = {total/n_triples_per_epoch:.4f}")
    return ranker


if __name__ == "__main__":
    with open("data/prepared.pkl", "rb") as f:
        d = pickle.load(f)
    emb = np.load("data/base_emb.npz")
    eu0, ei0, ea0 = emb["eu"], emb["ei"], emb["ea"]
    dim = eu0.shape[1]
    print("Pretraining ranking network g (Eq. 14) via BPR (Algorithm 1, step 4)...")
    ranker = pretrain_ranker(d, eu0, ei0, ea0, dim, epochs=20, n_triples_per_epoch=8000)
    ranker.save("data/ranker.npz")
    print("saved data/ranker.npz")
