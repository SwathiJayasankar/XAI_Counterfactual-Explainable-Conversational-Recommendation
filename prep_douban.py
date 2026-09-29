"""
Data preparation for Douban-Book -- the paper's second benchmark.

Source: https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding
(exactly the repo the paper's footnote 2 cites). The raw counts match the
paper's Table III exactly: 13,024 users / 22,347 items / 792,062 interactions.

Per the paper (Section V-A):
  - items are books with "attributes including author, publisher, and year"
  - "we remove the users with less than 10 interactions and low-frequency
    attributes"
  - split 70% / 20% / 10% train/val/test

Unlike MovieLens-100k, Douban-Book ALSO ships a user-user social graph
(user_user.dat), so the paper's social branch (Eqs. 4-5) is actually
implementable on this dataset -- it is loaded here and used by the model.
"""
import pickle

import numpy as np
import pandas as pd

DIR = "data/douban_book"
SEED = 42
MIN_USER_INTER = 10     # paper: drop users with < 10 interactions
MIN_ATTR_ITEMS = 20     # paper: drop low-frequency attributes


def load(min_attr_items=MIN_ATTR_ITEMS):
    rng = np.random.default_rng(SEED)

    ub = pd.read_csv(f"{DIR}/user_book.dat", sep="\t", names=["user", "item", "rating"])
    print(f"raw: {ub.user.nunique()} users, {ub.item.nunique()} items, {len(ub)} interactions")

    # --- drop users with fewer than 10 interactions ---
    keep_u = ub.user.value_counts()
    keep_u = set(keep_u[keep_u >= MIN_USER_INTER].index)
    ub = ub[ub.user.isin(keep_u)]

    # --- attributes: author + publisher + year, low-frequency ones removed ---
    attr_frames = []
    for fname, kind in [("book_author", "author"), ("book_publisher", "publisher"),
                        ("book_year", "year")]:
        df = pd.read_csv(f"{DIR}/{fname}.dat", sep="\t", names=["item", "val"])
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
        name="douban_book",
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
    with open("data/prepared_douban.pkl", "wb") as f:
        pickle.dump(d, f)
    print("saved data/prepared_douban.pkl")
