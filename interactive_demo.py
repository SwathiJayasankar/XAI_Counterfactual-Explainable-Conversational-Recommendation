"""
INTERACTIVE DEMO: a real conversation between YOU and the model in the terminal.

Unlike demo_conversation.py (which simulates a fake user with a hidden target
movie), here there is no ground truth - you just answer honestly and see
where the conversation goes, exactly like a real user would.

Usage:
    python3 interactive_demo.py
"""
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


def ask_yes_no(prompt):
    while True:
        ans = input(prompt + " (y/n, or 'quit'): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        if ans in ("quit", "q", "exit"):
            print("\nEnding conversation. Thanks for trying CECR!")
            raise SystemExit
        print("  please type y, n, or quit")


def pick_starting_genre():
    print("What kind of movie are you looking for today? Pick a genre by number:\n")
    for i, g in enumerate(genre_names):
        print(f"  {i:>2}) {g}")
    while True:
        raw = input("\nYour choice (number): ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(genre_names):
            return int(raw)
        print("  please enter a valid number from the list above")


def main():
    print("=" * 70)
    print("  CECR interactive conversational recommender  (type 'quit' anytime)")
    print("=" * 70 + "\n")

    # pick a fresh "cold-start" user (average of all users) as the base
    # long-term embedding, since you are not one of the 943 dataset users
    eu = eu0.mean(axis=0)

    first_attr = pick_starting_genre()
    pos_attrs = {first_attr}
    neg_attrs = set()
    asked_attrs = {first_attr}
    print(f"\n[You] I'm looking for a '{genre_names[first_attr]}' movie.\n")

    candidate_items = list(range(n_items))
    rejected_items, rejected_item_reps = [], []
    MAX_ROUNDS = 15

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"--- Round {rnd} " + "-" * 40)
        e_prime_u = model.refine_user(eu, pos_attrs, neg_attrs)
        eoverall = model.overall_preference(e_prime_u, rejected_item_reps)

        filtered_candidates = model.filter_candidates(candidate_items, pos_attrs, neg_attrs)
        action, remaining_attrs = model.decide_action(eoverall, filtered_candidates, asked_attrs)

        if action == "recommend" or not remaining_attrs:
            topk_items, scores = model.top_k(eoverall, filtered_candidates, k=1)
            rec_item = topk_items[0]
            genres_of_rec = [genre_names[a] for a in np.where(attr_matrix[rec_item] > 0)[0]]
            print(f"[System] How about '{titles[rec_item]}'?  (genres: {genres_of_rec})")
            liked = ask_yes_no("[You]  ")

            if liked:
                print(f"\n*** Recommendation accepted after {rnd} round(s): '{titles[rec_item]}' ***")
                return
            else:
                print()
                item_vec = model.refine_rejected_item(ei0[rec_item], pos_attrs)
                rejected_item_reps.append(item_vec)
                rejected_items.append(rec_item)
                candidate_items = [c for c in candidate_items if c not in rejected_items]

                outside = [c for c in candidate_items][:20]
                if outside:
                    k_item = int(np.random.choice(outside))
                    s_ik = eoverall @ ei0[k_item]
                    delta, ES, EC, expl_attrs, _ = model.counterfactual_explanation(
                        eoverall, ei0[rec_item], attr_matrix[rec_item], s_ik)
                    if len(expl_attrs):
                        top_attr = expl_attrs[np.argmax(delta[expl_attrs])]
                        print(f"[Explain] Counterfactual reasoning: you probably rejected this "
                              f"mainly because of its '{genre_names[top_attr]}' attribute "
                              f"(explanation strength={ES:.2f}, {len(expl_attrs)} attribute(s) "
                              f"would need to change)\n")
        else:
            attr_scores = eoverall @ ea0[remaining_attrs].T
            a_star = remaining_attrs[int(np.argmax(attr_scores))]
            asked_attrs.add(a_star)
            liked = ask_yes_no(f"[System] Do you like '{genre_names[a_star]}' movies?")
            print()
            if liked:
                pos_attrs.add(a_star)
            else:
                neg_attrs.add(a_star)

    print(f"*** No accepted recommendation after {MAX_ROUNDS} rounds. Conversation ended. ***")


if __name__ == "__main__":
    main()
