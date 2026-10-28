#!/usr/bin/env bash
# Full reproduction of the CECR paper (all tables + figures, all baselines),
# CPU-only, reduced-epoch budget. Logs go to logs/, results to results/<dataset>/.
set -e
mkdir -p logs

# ---- data (Table III) ----
python data_prep.py
python prep_douban.py
python prep_yelp.py

# ---- CECR training (MovieLens: 25/25 epochs, Yelp: 10/20, Douban: 5/15) ----
python train_cecr.py --data data/prepared.pkl --stage1-epochs 25 --stage2-epochs 25 --out data/cecr_full.pt
python train_cecr.py --data data/prepared.pkl --stage2-epochs 25 --no-counterfactual \
       --init-from data/cecr_full.pt --out data/cecr_nocf.pt                      # CouRe ablation
python train_cecr.py --data data/prepared_yelp.pkl --stage1-epochs 10 --stage2-epochs 20 --out data/cecr_yelp_full.pt
python train_cecr.py --data data/prepared_yelp.pkl --stage2-epochs 20 --no-counterfactual \
       --init-from data/cecr_yelp_full.pt --out data/cecr_yelp_nocf.pt
python train_cecr.py --data data/prepared_douban.pkl --stage1-epochs 5 --stage2-epochs 15 --out data/cecr_douban_full.pt

# ---- CECR evaluation (same harness as the baselines) ----
python eval_cecr.py --data data/prepared.pkl --model data/cecr_full.pt \
       --model-noucf data/cecr_nocf.pt --out results/movielens --skip-lambda          # Table II/IV, Fig 3/4/5
python eval_cecr.py --data data/prepared_yelp.pkl --model data/cecr_yelp_full.pt \
       --model-noucf data/cecr_yelp_nocf.pt --out results/yelp --topk-sweep           # + Fig 6
python eval_cecr.py --data data/prepared_douban.pkl --model data/cecr_douban_full.pt \
       --out results/douban --skip-ablations --skip-expl --skip-lambda                # Table II, Fig 3

# ---- baselines: EAR, FPAN, UNICORN, KBRD (+ CPR, CAEC, SAUR for Table IV) ----
python run_baselines.py --data data/prepared.pkl        --tag movielens --expl
python run_baselines.py --data data/prepared_yelp.pkl   --tag yelp      --expl
python run_baselines.py --data data/prepared_douban.pkl --tag douban

# ---- tables, figures, report ----
python make_paper_figs.py
