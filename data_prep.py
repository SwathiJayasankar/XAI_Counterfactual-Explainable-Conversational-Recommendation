"""
Data preparation for the CECR reproduction.

Dataset: MovieLens-100k (same family of dataset used in the paper, which used
MovieLens among its 3 benchmarks). We treat:
  - users  U : 943 users
  - items  I : 1682 movies
  - attributes A : 18 genres (the 19th, 'unknown', is dropped)

This builds the tripartite user-item-attribute structure U, I, A described in
Section III (Preliminary) of the paper, and a 70/20/10 train/val/test split
over (user, item) interactions, exactly as stated in the paper's Experimental
Settings ("Each dataset is split as training/validation/test with
70%/20%/10%").
"""
import numpy as np
import pandas as pd
import pickle

DATA_DIR = "data"
SEED = 42
rng = np.random.default_rng(SEED)

GENRE_NAMES = ["unknown","Action","Adventure","Animation","Children's","Comedy",
               "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
               "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

def load():
    ratings = pd.read_csv(f"{DATA_DIR}/u.data", sep="\t",
                           names=["user", "item", "rating", "ts"])
    items = pd.read_csv(f"{DATA_DIR}/u.item", sep="|", encoding="latin-1",
                         names=["item", "title", "release", "vrelease", "url"] + GENRE_NAMES)

    # attribute (genre) matrix, drop 'unknown' -> 18 attributes
    genre_cols = GENRE_NAMES[1:]
    item_attr = items.set_index("item")[genre_cols]
    item_attr.index = item_attr.index - 1          # 0-indexed
    n_items = ratings.item.max()
    n_users = ratings.user.max()
    n_attrs = len(genre_cols)

    attr_matrix = np.zeros((n_items, n_attrs), dtype=np.float32)
    for it, row in item_attr.iterrows():
        if 0 <= it < n_items:
            attr_matrix[it] = row.values

    ratings["user"] -= 1
    ratings["item"] -= 1

    # 70/20/10 split per paper's protocol, done per-user to keep every user represented
    train_rows, val_rows, test_rows = [], [], []
    for u, grp in ratings.groupby("user"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(0.7 * n))
        n_val = max(0, int(0.2 * n))
        train_rows.extend(idx[:n_tr])
        val_rows.extend(idx[n_tr:n_tr + n_val])
        test_rows.extend(idx[n_tr + n_val:])

    train = ratings.loc[train_rows].reset_index(drop=True)
    val = ratings.loc[val_rows].reset_index(drop=True)
    test = ratings.loc[test_rows].reset_index(drop=True)

    return dict(n_users=n_users, n_items=n_items, n_attrs=n_attrs,
                attr_matrix=attr_matrix, genre_names=genre_cols,
                train=train, val=val, test=test, ratings=ratings)


if __name__ == "__main__":
    d = load()
    print("users:", d["n_users"], "items:", d["n_items"], "attrs:", d["n_attrs"])
    print("train/val/test sizes:", len(d["train"]), len(d["val"]), len(d["test"]))
    with open("data/prepared.pkl", "wb") as f:
        pickle.dump(d, f)
    print("saved data/prepared.pkl")
