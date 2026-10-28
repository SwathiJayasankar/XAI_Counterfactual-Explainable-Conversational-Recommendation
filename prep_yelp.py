"""
Data preparation for Yelp -- the paper's first benchmark.

Source: https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding
(the same repo the paper's footnotes cite for Douban-Book and MovieLens).

DECLARED DEVIATION: the paper's Table III lists Yelp at 27,675 users / 70,311
items / 1,368,606 interactions, which is the EAR/CPR release of Yelp (built
from yelp.com/dataset, footnote 1). That release is not available locally and
would not fit this CPU-only budget together with seven baselines, so we use the
HIN-repo Yelp (16,239 users / 14,284 businesses / 198,397 interactions) with
exactly the paper's preprocessing applied.

Per the paper (Section V-A):
  - items are businesses with attributes "including categories ... et al."
    -> business category + city
  - "we remove the users with less than 10 interactions and low-frequency
    attributes"
  - split 70% / 20% / 10% train/val/test

Yelp also ships a user-user friendship graph (user_user.dat), so the social
branch (Eqs. 4-5) is enabled here as on Douban-Book.
"""
import pickle

import numpy as np
import pandas as pd

DIR = "data/yelp"
SEED = 42
MIN_USER_INTER = 10     # paper: drop users with < 10 interactions
MIN_ATTR_ITEMS = 20     # paper: drop low-frequency attributes


def load(min_attr_items=MIN_ATTR_ITEMS):
    rng = np.random.default_rng(SEED)

    ub = pd.read_csv(f"{DIR}/user_business.dat", sep="\t", names=["user", "item", "rating"])
    print(f"raw: {ub.user.nunique()} users, {ub.item.nunique()} items, {len(ub)} interactions")

    # --- drop users with fewer than 10 interactions ---
    keep_u = ub.user.value_counts()
    keep_u = set(keep_u[keep_u >= MIN_USER_INTER].index)
    ub = ub[ub.user.isin(keep_u)]

    # --- attributes: category + city, low-frequency ones removed ---
    attr_frames = []
    for fname, kind in [("business_category", "category"), ("business_city", "city")]:
        df = pd.read_csv(f"{DIR}/{fname}.dat", sep="\t", names=["item", "val", "w"])
        df["attr"] = kind + ":" + df.val.astype(str)
        attr_frames.append(df[["item", "attr"]])
    attrs = pd.concat(attr_frames, ignore_index=True)
    attrs = attrs[attrs.item.isin(set(ub.item.unique()))]

    counts = attrs.attr.value_counts()
    keep_a = set(counts[counts >= min_attr_items].index)
    attrs = attrs[attrs.attr.isin(keep_a)]
    print(f"attributes kept (>= {min_attr_items} items): {len(keep_a)}")

    # keep only items that still carry at least one attribute (a conversational
    # recommender can never reach an item with no askable attribute)
    items_with_attr = set(attrs.item.unique())
    ub = ub[ub.item.isin(items_with_attr)]

    # --- reindex users/items/attributes to contiguous ids ---
    uids = {u: k for k, u in enumerate(sorted(ub.user.unique()))}
    iids = {i: k for k, i in enumerate(sorted(ub.item.unique()))}
    aids = {a: k for k, a in enumerate(sorted(keep_a))}
    ub = ub.assign(user=ub.user.map(uids), item=ub.item.map(iids))

    n_users, n_items, n_attrs = len(uids), len(iids), len(aids)
    attr_matrix = np.zeros((n_items, n_attrs), dtype=np.float32)
    for it, a in zip(attrs.item.values, attrs.attr.values):
        if it in iids:
            attr_matrix[iids[it], aids[a]] = 1.0

    # --- social graph (Eqs. 4-5) ---
    social = pd.read_csv(f"{DIR}/user_user.dat", sep="\t", names=["u", "v", "w"])
    social = social[social.u.isin(uids) & social.v.isin(uids)]
    social = social.assign(u=social.u.map(uids), v=social.v.map(uids))
    print(f"social edges kept: {len(social)}")

    # --- 70/20/10 split, per user so every user is represented ---
    ub = ub.reset_index(drop=True)
    train_idx, val_idx, test_idx = [], [], []
    for _, grp in ub.groupby("user"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(0.7 * n))
        n_va = max(0, int(0.2 * n))
        train_idx.extend(idx[:n_tr])
        val_idx.extend(idx[n_tr:n_tr + n_va])
        test_idx.extend(idx[n_tr + n_va:])

    d = dict(
        name="yelp",
        n_users=n_users, n_items=n_items, n_attrs=n_attrs,
        attr_matrix=attr_matrix,
        genre_names=[a for a, _ in sorted(aids.items(), key=lambda kv: kv[1])],
        train=ub.loc[train_idx].reset_index(drop=True),
        val=ub.loc[val_idx].reset_index(drop=True),
        test=ub.loc[test_idx].reset_index(drop=True),
        ratings=ub,
        social=social[["u", "v"]].values.astype(np.int64),
    )
    print(f"final: {n_users} users, {n_items} items, {n_attrs} attributes, "
          f"{len(ub)} interactions")
    print(f"split: train {len(d['train'])} / val {len(d['val'])} / test {len(d['test'])}")
    print(f"avg attributes per item: {attr_matrix.sum(1).mean():.2f}")
    return d


if __name__ == "__main__":
    d = load()
    with open("data/prepared_yelp.pkl", "wb") as f:
        pickle.dump(d, f)
    print("saved data/prepared_yelp.pkl")
