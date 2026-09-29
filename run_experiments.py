import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import run_simulation
from cecr import CECR

N_TEST_USERS = 250
MAX_ROUNDS = 10

configs = [
    # name,                pref_refine, counterfactual, score_decision, hard_filter
    ("CECR (full)",         True,  True,  True,  True),
    ("- PreRi (no pref. refine)",  False, True,  True,  True),
    ("- CouRe (no counterfactual)", True,  False, True,  True),
    ("- RecPe (random decision)",   True,  True,  False, True),
    ("CECR (no hard filter, paper-literal)", True, True, True, False),
]

results = {}
for name, pr, cf, sd, hf in configs:
    print(f"Running: {name} ...")
    res = run_simulation(name, pr, cf, sd, max_rounds=MAX_ROUNDS,
                          n_test_users=N_TEST_USERS, seed=7,
                          collect_explanations=(name == "CECR (full)"),
                          use_hard_filter=hf)
    results[name] = res
    print("  ", res["sr_at_r"], "AT=", round(res["AT"], 2))

with open("out/results.pkl", "wb") as f:
    pickle.dump(results, f)

# ---------------------------------------------------------------
# RESULT 1  (reproduces paper's Fig. 3 - "SR at different rounds")
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4.2))
for name, _, _, _, _ in configs:
    plt.plot(range(1, MAX_ROUNDS + 1), results[name]["sr_curve"], marker="o", label=name)
plt.xlabel("Conversation round")
plt.ylabel("Success Rate")
plt.title("Fig.3-style: Success Rate vs. conversation round (MovieLens-100k)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("out/fig3_sr_vs_round.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# RESULT 2  (reproduces paper's Table II - "Recommendation performance")
# ---------------------------------------------------------------
table2_rows = []
for name, _, _, _, _ in configs:
    r = results[name]
    table2_rows.append([name, r["sr_at_r"]["SR@5"], r["sr_at_r"]["SR@10"],
                         r["sr_at_r"]["SR@20"], round(r["AT"], 2)])
with open("out/table2_performance.csv", "w") as f:
    f.write("Model,SR@5,SR@10,SR@20,AT\n")
    for row in table2_rows:
        f.write(",".join(str(x) for x in row) + "\n")

# ---------------------------------------------------------------
# RESULT 3  (reproduces paper's Fig. 4 - Ablation study, SR@20 + AT)
# ---------------------------------------------------------------
names = [c[0] for c in configs[:4]]   # ablation chart excludes the separate filter-comparison config
sr20 = [results[n]["sr_at_r"]["SR@20"] for n in names]
ats = [results[n]["AT"] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(len(names)), sr20, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
axes[0].set_ylabel("SR@20")
axes[0].set_title("Ablation: SR@20")
axes[1].plot(range(len(names)), ats, marker="o", color="crimson")
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
axes[1].set_ylabel("Average Turn (AT)")
axes[1].set_title("Ablation: Average Turns")
plt.tight_layout()
plt.savefig("out/fig4_ablation.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# RESULT 4  (reproduces paper's Table IV - Explainability: Precision/Recall/PN/PS)
#   -- this is the core XAI evaluation --
# ---------------------------------------------------------------
with open("data/prepared.pkl", "rb") as f:
    d = pickle.load(f)
emb = np.load("data/base_emb.npz")
eu0, ei0, ea0 = emb["eu"], emb["ei"], emb["ea"]
attr_matrix = d["attr_matrix"]
dim = eu0.shape[1]
model = CECR(eu0, ei0, ea0, attr_matrix, dim)

rng = np.random.default_rng(123)


def explain_metrics(use_random_explanation, n_eval=150, K=10):
    precisions, recalls, pns, pss = [], [], [], []
    test = d["test"]
    liked = test[test.rating >= 4]
    users_avail = liked.user.unique()
    rng.shuffle(users_avail)
    users_avail = users_avail[:n_eval]

    for u in users_avail:
        cand = liked[liked.user == u].item.values
        i = int(rng.choice(cand))
        user_pos_attrs = set(np.where(attr_matrix[i] > 0)[0].tolist())
        if len(user_pos_attrs) < 2:
            continue
        eu = eu0[u]
        eoverall = model.refine_user(eu, user_pos_attrs, set())

        pool = [x for x in range(d["n_items"]) if x != i]   # full catalog, not a 300-item subsample
        candidate_items = pool + [i]
        s_ik = np.sort(eoverall @ ei0[pool].T)[::-1][K - 1]   # score of the K-th ranked distractor

        if use_random_explanation:
            n_expl = rng.integers(1, 4)
            expl_attrs = rng.choice(np.arange(attr_matrix.shape[1]), size=n_expl, replace=False)
        else:
            delta, ES, EC, expl_attrs, cf_vec = model.counterfactual_explanation(
                eoverall, ei0[i], attr_matrix[i], s_ik)
            if len(expl_attrs) == 0:
                expl_attrs = np.array([int(np.argmax(np.abs(delta)))])

        p_ui = np.array([1.0 if a in user_pos_attrs else 0.0 for a in range(attr_matrix.shape[1])])
        indicator = np.zeros(attr_matrix.shape[1]); indicator[expl_attrs] = 1
        num = np.sum(p_ui * indicator)
        precision = num / max(1, indicator.sum())
        recall = num / max(1e-9, p_ui.sum())
        precisions.append(precision); recalls.append(recall)

        # PN: zero-out explanation attributes -> item should DROP from top-K
        i_star_vec = ei0[i] - (attr_matrix[i][expl_attrs, None] * ea0[expl_attrs]).sum(axis=0)
        scores_pool = eoverall @ ei0[pool].T
        topk_scores = np.sort(scores_pool)[::-1][:K]
        s_i_star = eoverall @ i_star_vec
        pn = 1.0 if s_i_star < topk_scores.min() else 0.0
        pns.append(pn)

        # PS: keep ONLY explanation attributes -> item should STAY in top-K
        keep_mask = np.zeros(attr_matrix.shape[1]); keep_mask[expl_attrs] = 1
        i_circ_vec = (attr_matrix[i] * keep_mask)[:, None] * ea0
        i_circ_vec = i_circ_vec.sum(axis=0)
        s_i_circ = eoverall @ i_circ_vec
        ps = 1.0 if s_i_circ >= topk_scores.min() else 0.0
        pss.append(ps)

    return dict(Precision=np.mean(precisions), Recall=np.mean(recalls),
                PN=np.mean(pns), PS=np.mean(pss), n=len(precisions))


expl_cecr = explain_metrics(use_random_explanation=False)
expl_rand = explain_metrics(use_random_explanation=True)

with open("out/table4_explainability.csv", "w") as f:
    f.write("Model,Precision,Recall,PN,PS\n")
    f.write(f"Random,{expl_rand['Precision']:.4f},{expl_rand['Recall']:.4f},{expl_rand['PN']:.4f},{expl_rand['PS']:.4f}\n")
    f.write(f"CECR,{expl_cecr['Precision']:.4f},{expl_cecr['Recall']:.4f},{expl_cecr['PN']:.4f},{expl_cecr['PS']:.4f}\n")

print("Table IV (explainability):")
print(" Random:", expl_rand)
print(" CECR  :", expl_cecr)

# ---------------------------------------------------------------
# RESULT 5  (reproduces paper's Fig. 6(a) - lambda sensitivity of the
#            counterfactual explanation: strength (ES) vs complexity (EC))
#   -- the deepest XAI-specific analysis --
# ---------------------------------------------------------------
lambdas = [1, 5, 10, 25, 50, 75, 100]
avg_ES, avg_EC = [], []
test = d["test"]
liked = test[test.rating >= 4]
users_avail = liked.user.unique()
rng2 = np.random.default_rng(555)
rng2.shuffle(users_avail)
sample_users = users_avail[:60]

for lam in lambdas:
    model_lam = CECR(eu0, ei0, ea0, attr_matrix, dim, lam=lam)
    ES_list, EC_list = [], []
    for u in sample_users:
        cand = liked[liked.user == u].item.values
        i = int(rng2.choice(cand))
        user_pos_attrs = set(np.where(attr_matrix[i] > 0)[0].tolist())
        eoverall = model_lam.refine_user(eu0[u], user_pos_attrs, set())
        pool = np.array([x for x in range(d["n_items"]) if x != i])   # full catalog
        s_ik = np.sort(eoverall @ ei0[pool].T)[::-1][9]
        delta, ES, EC, expl_attrs, _ = model_lam.counterfactual_explanation(
            eoverall, ei0[i], attr_matrix[i], s_ik)
        ES_list.append(ES); EC_list.append(EC)
    avg_ES.append(np.mean(ES_list))
    avg_EC.append(np.mean(EC_list))
    print(f"lambda={lam:>4}  ES={np.mean(ES_list):.4f}  EC={np.mean(EC_list):.2f}")

fig, ax1 = plt.subplots(figsize=(6, 4.2))
ax1.plot(lambdas, avg_EC, marker="o", color="crimson", label="complexity (EC)")
ax1.set_xlabel("lambda")
ax1.set_ylabel("Explanation Complexity (EC)", color="crimson")
ax2 = ax1.twinx()
ax2.plot(lambdas, avg_ES, marker="s", color="royalblue", label="strength (ES)")
ax2.set_ylabel("Explanation Strength (ES)", color="royalblue")
plt.title("Fig.6(a)-style: impact of lambda on counterfactual explanation")
fig.tight_layout()
plt.savefig("out/fig6a_lambda_sensitivity.png", dpi=140)
plt.close()

with open("out/fig6a_data.csv", "w") as f:
    f.write("lambda,ES,EC\n")
    for l, es, ec in zip(lambdas, avg_ES, avg_EC):
        f.write(f"{l},{es:.4f},{ec:.4f}\n")

print("\nAll done. Outputs in out/")
