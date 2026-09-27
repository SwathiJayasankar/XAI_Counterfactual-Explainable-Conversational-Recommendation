"""
DEMO SCRIPT for presenting in class.

Unlike simulate.py (which runs silently over 100s of users to compute
statistics), this script runs ONE conversation and prints every single turn
live, so an audience can watch the four stages work together:

    GNN embeddings (already trained) -> preference refinement -> ask/recommend
    decision -> (if rejected) counterfactual explanation -> next round

Usage:
    python3 demo_conversation.py            # random user/target movie
    python3 demo_conversation.py 42         # a specific test-user index
"""
import sys
import numpy as np
import pickle
import pandas as pd
from cecr import CECR
from data_prep import GENRE_NAMES

with open("data/prepared.pkl", "rb") as f:
    d = pickle.load(f)
emb = np.load("data/base_emb.npz")
eu0, ei0, ea0 = emb["eu"], emb["ei"], emb["ea"]
attr_matrix = d["attr_matrix"]
genre_names = d["genre_names"]
n_items = d["n_items"]

items_df = pd.read_csv("data/u.item", sep="|", encoding="latin-1",
                        names=["item", "title", "release", "vrelease", "url"] + GENRE_NAMES)
titles = items_df.set_index(items_df.item - 1)["title"].to_dict()

model = CECR(eu0, ei0, ea0, attr_matrix, eu0.shape[1])

seed = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(0, 10000)
rng = np.random.default_rng(seed)
print(f"(seed={seed} — rerun with this number to reproduce this exact conversation)\n")

test = d["test"]
liked = test[test.rating >= 4]
u = int(rng.choice(liked.user.unique()))
target = int(rng.choice(liked[liked.user == u].item.values))
target_attrs = set(np.where(attr_matrix[target] > 0)[0].tolist())

print(f"=== Simulated user #{u}  |  target movie (ground truth): '{titles[target]}' "
      f"(genres: {[genre_names[a] for a in target_attrs]}) ===\n")

pos_attrs, neg_attrs, asked_attrs = set(), set(), set()
first_attr = int(rng.choice(list(target_attrs)))
pos_attrs.add(first_attr)
asked_attrs.add(first_attr)
print(f"[Turn 0] User: I'm looking for a '{genre_names[first_attr]}' movie.\n")

candidate_items = [i for i in range(n_items) if i != target]
rng.shuffle(candidate_items)
candidate_items = candidate_items[:400] + [target]
rejected_items, rejected_item_reps = [], []
eu = eu0[u].copy()
MAX_ROUNDS = 10

for rnd in range(1, MAX_ROUNDS + 1):
    print(f"--- Round {rnd} ---")
    e_prime_u = model.refine_user(eu, pos_attrs, neg_attrs)
    eoverall = model.overall_preference(e_prime_u, rejected_item_reps)

    filtered_candidates = model.filter_candidates(candidate_items, pos_attrs, neg_attrs)
    action, remaining_attrs = model.decide_action(eoverall, filtered_candidates, asked_attrs)

    if action == "recommend" or not remaining_attrs:
        topk_items, scores = model.top_k(eoverall, filtered_candidates, k=10)
        print(f"[System] (recommend action) Top-10 candidates: "
              f"{[titles[i][:28] for i in topk_items[:3]]} ...")
        if target in topk_items:
            print(f"[System] Here's '{titles[target]}' — I think you'll like it!")
            print(f"[User]   Yes! Exactly what I wanted.\n")
            print(f"*** SUCCESS in round {rnd} ***")
            break
        else:
            rejected_here = topk_items[0]
            print(f"[System] What about '{titles[rejected_here]}'?")
            print(f"[User]   No, I don't like that one.\n")
            item_vec = model.refine_rejected_item(ei0[rejected_here], pos_attrs)
            rejected_item_reps.append(item_vec)
            rejected_items.append(rejected_here)

            outside = [c for c in candidate_items if c not in topk_items]
            if outside:
                k_item = int(rng.choice(outside[:20]))
                s_ik = eoverall @ ei0[k_item]
                delta, ES, EC, expl_attrs, _ = model.counterfactual_explanation(
                    eoverall, ei0[rejected_here], attr_matrix[rejected_here], s_ik)
                if len(expl_attrs):
                    top_attr = expl_attrs[np.argmax(delta[expl_attrs])]
                    print(f"[Explain] Counterfactual reasoning: rejected mainly because of "
                          f"'{genre_names[top_attr]}' (strength={ES:.2f}, "
                          f"{len(expl_attrs)} attribute(s) changed)\n")
            candidate_items = [c for c in candidate_items if c not in rejected_items]
    else:
        attr_scores = eoverall @ ea0[remaining_attrs].T
        a_star = remaining_attrs[int(np.argmax(attr_scores))]
        asked_attrs.add(a_star)
        print(f"[System] Do you like '{genre_names[a_star]}' movies?")
        if a_star in target_attrs:
            pos_attrs.add(a_star)
            print(f"[User]   Yes, I like that.\n")
        else:
            neg_attrs.add(a_star)
            print(f"[User]   No, I don't.\n")
else:
    print("*** Conversation ended without success (max rounds reached) ***")
