"""
STAGE 1 of the pipeline: "GNN-based offline learning" (paper Section IV-A, Eqs. 1-8).

This replaces the earlier plain matrix-factorization stand-in with an actual
Graph Neural Network that does neighbor aggregation over the tripartite
user-item-attribute graph:

  Eq. 1  z_{u,i} = MLP(e_i^(in) (+) e_{y_u,i})           rating-aware representation
  Eq. 2  e_u^U   = ReLU(W . sum_i alpha_ui * z_{u,i} + b)  item-space user aggregation
  Eq. 3  alpha_ui                                          attention over a user's neighbors
  Eq. 7  e_i     = ReLU(Wi . sum_u mu_iu * z_{i,u} + b)    item aggregation over its raters
  Eq. 8  mu_iu                                             attention over an item's neighbors
  (analogous)  e_a = attribute embedding, aggregated over items sharing attribute a
  Eq. 6  one further MLP layer on top of the aggregated representation (stacking)

DOCUMENTED SIMPLIFICATIONS (only these two, everything else is as in the paper):
  1. No social graph exists in MovieLens-100k (no user-user "friend" links), so the
     social-space branch e_u^S (Eq. 4-5) is dropped; e_u is built from the item-space
     branch e_u^U only. This is a dataset limitation, not a modelling choice.
  2. The attention weights alpha_ui / mu_iu are computed as a similarity-based
     softmax (dot product between the neighbor's representation and the target
     node's own embedding) and treated as fixed during backprop for that
     minibatch, instead of learning their own 2-layer attention network's weights
     (Eq. 3/eq.8's W_alpha1, W_alpha2). This keeps the backward pass simple
     while keeping the *aggregation itself* (the actual "graph" part: gathering
     and weighting neighbor messages) fully real and learned.

Everything is trained end-to-end with plain NumPy (manual forward + backward
pass, gradient descent) using a BPR ranking loss, so no deep-learning framework
is required.
"""
import numpy as np
import pickle

SEED = 0
rng = np.random.default_rng(SEED)
relu = lambda x: np.maximum(x, 0)
relu_grad = lambda pre: (pre > 0).astype(np.float32)


class GNNOfflineLearner:
    """A 2-layer (message-passing + stack) GNN over the U-I-A tripartite graph."""

    def __init__(self, n_users, n_items, n_attrs, dim=32, lr=0.01, reg=5e-3, lr_w_scale=0.15):
        self.n_users, self.n_items, self.n_attrs, self.dim = n_users, n_items, n_attrs, dim
        self.lr, self.reg = lr, reg
        self.lr_w = lr * lr_w_scale     # weight matrices are updated far more often than
                                        # any single embedding row, so they need a smaller step
        d = dim
        init = lambda *shape: rng.normal(0, 0.05, shape).astype(np.float32)

        # base (input) node embeddings e^(in)
        self.e_u_in = init(n_users, d)
        self.e_i_in = init(n_items, d)

        # Eq.1: rating-aware MLP,  input = [neighbor_emb(d) ; rating(1)] -> d
        self.Wz, self.bz = init(d, d + 1), np.zeros(d, dtype=np.float32)
        # Eq.2/Eq.7: aggregation output layers (one per branch, both project agg(d) -> d)
        self.WU, self.bU = init(d, d), np.zeros(d, dtype=np.float32)
        self.WI, self.bI = init(d, d), np.zeros(d, dtype=np.float32)
        # Eq.6: one extra stacked layer on top of the user aggregation (depth=2 GNN)
        self.W2, self.b2 = init(d, d), np.zeros(d, dtype=np.float32)
        # attribute aggregation output layer
        self.WA, self.bA = init(d, d), np.zeros(d, dtype=np.float32)

    # ---------- forward pieces ----------
    def _z(self, neighbor_embs, ratings):
        """Eq. 1: rating-aware representation for a batch of neighbors."""
        x = np.concatenate([neighbor_embs, ratings[:, None]], axis=1)   # [n, d+1]
        pre = x @ self.Wz.T + self.bz
        return relu(pre), pre, x

    def _attention(self, z, query_vec):
        """Similarity-softmax attention over neighbors (fixed during backward)."""
        scores = z @ query_vec
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / (w.sum() + 1e-9)
        return w

    def encode_user(self, u, items, ratings):
        """Item-space user aggregation (Eq. 2-3) + one stacked layer (Eq. 6)."""
        neigh_i = self.e_i_in[items]
        z, z_pre, z_in = self._z(neigh_i, ratings)
        alpha = self._attention(z, self.e_u_in[u])
        agg = (alpha[:, None] * z).sum(axis=0)
        pre_U = self.WU @ agg + self.bU
        eU = relu(pre_U)
        pre_2 = self.W2 @ eU + self.b2
        e_final = relu(pre_2)
        cache = dict(z=z, z_pre=z_pre, z_in=z_in, alpha=alpha, agg=agg,
                     pre_U=pre_U, eU=eU, pre_2=pre_2, items=items)
        return e_final, cache

    def encode_item(self, i, users, ratings):
        """Item aggregation over its raters (Eq. 7-8)."""
        neigh_u = self.e_u_in[users]
        z, z_pre, z_in = self._z(neigh_u, ratings)
        mu = self._attention(z, self.e_i_in[i])
        agg = (mu[:, None] * z).sum(axis=0)
        pre_I = self.WI @ agg + self.bI
        e_final = relu(pre_I)
        cache = dict(z=z, z_pre=z_pre, z_in=z_in, mu=mu, agg=agg, pre_I=pre_I, users=users)
        return e_final, cache

    def encode_attribute(self, item_embs):
        """Attribute embedding = attention-aggregated mean of the items sharing it."""
        if len(item_embs) == 0:
            return np.zeros(self.dim, dtype=np.float32), None
        query = item_embs.mean(axis=0)
        w = self._attention(item_embs, query)
        agg = (w[:, None] * item_embs).sum(axis=0)
        pre_A = self.WA @ agg + self.bA
        return relu(pre_A), dict(item_embs=item_embs, w=w, agg=agg, pre_A=pre_A)

    # ---------- one BPR training step over a (u, i_pos, i_neg) triple ----------
    def train_step(self, u, i_pos, i_neg, user_items, user_ratings,
                    item_users_pos, item_ratings_pos, item_users_neg, item_ratings_neg):
        eu, cache_u = self.encode_user(u, user_items, user_ratings)
        eip, cache_ip = self.encode_item(i_pos, item_users_pos, item_ratings_pos)
        ein, cache_in = self.encode_item(i_neg, item_users_neg, item_ratings_neg)

        x = eu @ (eip - ein)
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        grad_common = (1 - sig)     # d(-log sigmoid(x))/dx = -(1-sig);  we ASCEND so use +grad_common

        # ----- gradients into eu, eip, ein -----
        d_eu = self._clip(grad_common * (eip - ein) - self.reg * eu, 2.0)
        d_eip = self._clip(grad_common * eu - self.reg * eip, 2.0)
        d_ein = self._clip(-grad_common * eu - self.reg * ein, 2.0)

        self._backward_user(cache_u, d_eu, user_items, user_ratings)
        self._backward_item(cache_ip, d_eip, item_users_pos, item_ratings_pos)
        self._backward_item(cache_in, d_ein, item_users_neg, item_ratings_neg)
        return -np.log(sig + 1e-9)

    @staticmethod
    def _clip(g, max_norm=5.0):
        n = np.linalg.norm(g)
        if n > max_norm:
            g = g * (max_norm / (n + 1e-9))
        return g

    def _backward_user(self, cache, d_out, items, ratings):
        # d_out flows through relu(W2 @ eU + b2)
        d_pre2 = self._clip(d_out * relu_grad(cache["pre_2"]))
        self.W2 += self.lr_w * (np.outer(d_pre2, cache["eU"]) - self.reg * self.W2)
        self.b2 += self.lr * d_pre2
        d_eU = self.W2.T @ d_pre2
        d_preU = self._clip(d_eU * relu_grad(cache["pre_U"]))
        self.WU += self.lr_w * (np.outer(d_preU, cache["agg"]) - self.reg * self.WU)
        self.bU += self.lr * d_preU
        d_agg = self.WU.T @ d_preU
        d_z = cache["alpha"][:, None] * d_agg[None, :]           # broadcast to [n_neighbors, d]
        d_zpre = self._clip(d_z * relu_grad(cache["z_pre"]))
        self.Wz += self.lr_w * (d_zpre.T @ cache["z_in"] - self.reg * self.Wz)
        self.bz += self.lr * d_zpre.sum(axis=0)
        d_ein = d_zpre @ self.Wz[:, :self.dim]
        for k, it in enumerate(items):
            self.e_i_in[it] += self.lr * (d_ein[k] - self.reg * self.e_i_in[it])

    def _backward_item(self, cache, d_out, users, ratings):
        d_preI = self._clip(d_out * relu_grad(cache["pre_I"]))
        self.WI += self.lr_w * (np.outer(d_preI, cache["agg"]) - self.reg * self.WI)
        self.bI += self.lr * d_preI
        d_agg = self.WI.T @ d_preI
        d_z = cache["mu"][:, None] * d_agg[None, :]
        d_zpre = self._clip(d_z * relu_grad(cache["z_pre"]))
        self.Wz += self.lr_w * (d_zpre.T @ cache["z_in"] - self.reg * self.Wz)
        self.bz += self.lr * d_zpre.sum(axis=0)
        d_eun = d_zpre @ self.Wz[:, :self.dim]
        for k, u in enumerate(users):
            self.e_u_in[u] += self.lr * (d_eun[k] - self.reg * self.e_u_in[u])


def build_neighbor_lists(train_df, n_users, n_items, max_neighbors=30):
    pos = train_df[train_df.rating >= 1]
    user_items, user_ratings = {}, {}
    item_users, item_ratings = {}, {}
    for u, grp in pos.groupby("user"):
        items = grp.item.values
        rts = (grp.rating.values / 5.0).astype(np.float32)
        if len(items) > max_neighbors:
            idx = rng.choice(len(items), max_neighbors, replace=False)
            items, rts = items[idx], rts[idx]
        user_items[u] = items
        user_ratings[u] = rts
    for i, grp in pos.groupby("item"):
        users = grp.user.values
        rts = (grp.rating.values / 5.0).astype(np.float32)
        if len(users) > max_neighbors:
            idx = rng.choice(len(users), max_neighbors, replace=False)
            users, rts = users[idx], rts[idx]
        item_users[i] = users
        item_ratings[i] = rts
    return user_items, user_ratings, item_users, item_ratings


def train_gnn(d, dim=32, epochs=12, n_triples_per_epoch=8000):
    n_users, n_items, n_attrs = d["n_users"], d["n_items"], d["n_attrs"]
    train_df = d["train"]
    user_items, user_ratings, item_users, item_ratings = build_neighbor_lists(train_df, n_users, n_items)
    pos_by_user = train_df[train_df.rating >= 4].groupby("user")["item"].apply(list).to_dict()
    users_with_pos = np.array([u for u in pos_by_user if u in user_items and len(pos_by_user[u]) > 0])

    gnn = GNNOfflineLearner(n_users, n_items, n_attrs, dim=dim, lr=0.02, reg=1e-3)

    for ep in range(epochs):
        idx = rng.choice(len(users_with_pos), size=n_triples_per_epoch)
        total_loss = 0.0
        for k in idx:
            u = int(users_with_pos[k])
            i_pos = int(rng.choice(pos_by_user[u]))
            if i_pos not in item_users:
                continue
            i_neg = int(rng.integers(0, n_items))
            while i_neg in pos_by_user[u] or i_neg not in item_users:
                i_neg = int(rng.integers(0, n_items))
            loss = gnn.train_step(u, i_pos, i_neg,
                                   user_items[u], user_ratings[u],
                                   item_users[i_pos], item_ratings[i_pos],
                                   item_users[i_neg], item_ratings[i_neg])
            total_loss += loss
        print(f"  [GNN] epoch {ep+1}/{epochs}  avg BPR loss = {total_loss/n_triples_per_epoch:.4f}")

    # ---- materialize final e_u, e_i for every node (used by downstream stages) ----
    eu_final = np.zeros((n_users, dim), dtype=np.float32)
    for u in range(n_users):
        if u in user_items and len(user_items[u]) > 0:
            eu_final[u], _ = gnn.encode_user(u, user_items[u], user_ratings[u])
        else:
            eu_final[u] = gnn.e_u_in[u]

    ei_final = np.zeros((n_items, dim), dtype=np.float32)
    for i in range(n_items):
        if i in item_users and len(item_users[i]) > 0:
            ei_final[i], _ = gnn.encode_item(i, item_users[i], item_ratings[i])
        else:
            ei_final[i] = gnn.e_i_in[i]

    ea_final = np.zeros((n_attrs, dim), dtype=np.float32)
    attr_matrix = d["attr_matrix"]
    for a in range(n_attrs):
        items_with_a = np.where(attr_matrix[:, a] > 0)[0]
        if len(items_with_a) > 0:
            ea_final[a], _ = gnn.encode_attribute(ei_final[items_with_a])

    return eu_final, ei_final, ea_final, gnn


if __name__ == "__main__":
    with open("data/prepared.pkl", "rb") as f:
        d = pickle.load(f)
    print("Training GNN-based offline learning module (Eqs. 1-8)...")
    eu, ei, ea, gnn = train_gnn(d, dim=32, epochs=9)
    np.savez("data/base_emb.npz", eu=eu, ei=ei, ea=ea)
    print("saved data/base_emb.npz  shapes:", eu.shape, ei.shape, ea.shape)
