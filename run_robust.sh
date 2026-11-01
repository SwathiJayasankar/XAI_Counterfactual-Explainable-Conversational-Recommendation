#!/usr/bin/env bash
# Improvement #5: robust MovieLens evaluation over 5 seeds x answer noise {0, 0.1, 0.2}.
# Every seed retrains CECR and every baseline; logs go to logs/robust_*.txt.
set -e
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SEEDS="0 1 2 3 4"

# baselines: one process per seed, in parallel
for s in $SEEDS; do
  python run_baselines.py --data data/prepared.pkl --tag movielens --robust --seed $s \
         > logs/robust_baselines_s$s.txt 2>&1 &
done

# CECR: seed 0 is the existing checkpoint; seeds 1-4 are retrained (25/25 epochs)
for s in $SEEDS; do
  ckpt=data/cecr_full.pt
  if [ "$s" != "0" ]; then
    ckpt=data/cecr_full_s$s.pt
    [ -f "$ckpt" ] || python train_cecr.py --data data/prepared.pkl --stage1-epochs 25 \
        --stage2-epochs 25 --seed $s --out $ckpt > logs/robust_train_cecr_s$s.txt 2>&1
  fi
  python robust_cecr.py --data data/prepared.pkl --model $ckpt --tag movielens --seed $s \
         > logs/robust_cecr_s$s.txt 2>&1
done

wait
python summarize_robust.py --tag movielens > logs/robust_summary.txt 2>&1
echo "robust study done -> results/movielens/robust/summary.md"
