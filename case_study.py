import numpy as np, pickle
from cecr import CECR
from data_prep import GENRE_NAMES

with open("data/prepared.pkl", "rb") as f:
    d = pickle.load(f)
emb = np.load("data/base_emb.npz")
eu0, ei0, ea0 = emb["eu"], emb["ei"], emb["ea"]
attr_matrix = d["attr_matrix"]
genre_names = d["genre_names"]
items = None
import pandas as pd
items_df = pd.read_csv("data/u.item", sep="|", encoding="latin-1",
                        names=["item", "title", "release", "vrelease", "url"] + GENRE_NAMES)
titles = items_df.set_index(items_df.item - 1)["title"].to_dict()

model = CECR(eu0, ei0, ea0, attr_matrix, eu0.shape[1])
rng = np.random.default_rng(9)

test = d["test"]
liked = test[test.rating >= 4]

# find a user/target with a clean, illustrative genre profile
found = False
for u in rng.permutation(liked.user.unique())[:400]:
    cand = liked[liked.user == u].item.values
    for target in cand:
        attrs = np.where(attr_matrix[target] > 0)[0]
        if 2 <= len(attrs) <= 3:
            found = True
            break
    if found:
        break

print(f"User {u}, target movie: '{titles[target]}'  genres={[genre_names[a] for a in attrs]}\n")

pos_attrs = {int(attrs[0])}
neg_attrs = set()
eu = eu0[u]

# round 1: system asks about a genre the item does NOT have -> user says no (reject)
other_genre_idx = [g for g in range(len(genre_names)) if g not in attrs][0]
neg_attrs.add(other_genre_idx)
print(f"Turn 1  System: Do you like '{genre_names[other_genre_idx]}' movies?")
print(f"Turn 2  User:   No, I don't like that.\n")

e_prime_u = model.refine_user(eu, pos_attrs, neg_attrs)
eoverall = model.overall_preference(e_prime_u, [])

pool = rng.choice([x for x in range(d["n_items"]) if x != target], size=300, replace=False).tolist()
pool = model.filter_candidates(pool, pos_attrs, neg_attrs)
scores = eoverall @ ei0[pool].T
order = np.argsort(-scores)
rejected_item = int(pool[order[0]])   # top-1 distractor -> proposed & rejected
k_item = int(pool[order[min(9, len(order) - 1)]])
s_ik = eoverall @ ei0[k_item]

print(f"Turn 3  System: What about the movie '{titles[rejected_item]}'?"
      f"  (genres: {[genre_names[a] for a in np.where(attr_matrix[rejected_item]>0)[0]]})")
print(f"Turn 4  User:   No, I don't like this one.\n")

delta, ES, EC, expl_attrs, cf_vec = model.counterfactual_explanation(
    eoverall, ei0[rejected_item], attr_matrix[rejected_item], s_ik)

print("Counterfactual explanation for the rejection:")
if len(expl_attrs):
    top_attr = expl_attrs[np.argmax(np.abs(delta[expl_attrs]))]
else:
    top_attr = int(np.argmax(np.abs(delta)))
print(f"  Most influential attribute: '{genre_names[top_attr]}'  (delta={delta[top_attr]:.3f})")
print(f"  Explanation strength (ES) = {ES:.3f},  complexity (EC) = {EC} attribute(s) changed")
print(f"  -> 'This movie was rejected mainly because of its \"{genre_names[top_attr]}\" attribute; "
      f"minimal change to that attribute would flip the recommendation.'\n")

# turn 5-6: propose the actual target and accept
print(f"Turn 5  System: What about the movie '{titles[target]}'?")
print(f"Turn 6  User:   Yes, I like this movie, thanks!")

with open("out/case_study.txt", "w") as f:
    f.write(f"User {u}, target movie: '{titles[target]}'  genres={[genre_names[a] for a in attrs]}\n\n")
    f.write(f"Turn 1  System: Do you like '{genre_names[other_genre_idx]}' movies?\n")
    f.write("Turn 2  User:   No, I don't like that.\n\n")
    f.write(f"Turn 3  System: What about the movie '{titles[rejected_item]}'? "
            f"(genres: {[genre_names[a] for a in np.where(attr_matrix[rejected_item]>0)[0]]})\n")
    f.write("Turn 4  User:   No, I don't like this one.\n\n")
    f.write("Counterfactual explanation for the rejection:\n")
    f.write(f"  Most influential attribute: '{genre_names[top_attr]}'  (delta={delta[top_attr]:.3f})\n")
    f.write(f"  ES = {ES:.3f}, EC = {EC}\n\n")
    f.write(f"Turn 5  System: What about the movie '{titles[target]}'?\n")
    f.write("Turn 6  User:   Yes, I like this movie, thanks!\n")
