# Robust evaluation: movielens

Seeds: [0, 1, 2, 3, 4] (5).  Mean +- 95% CI (t over seeds).  Answer noise = probability each attribute answer is flipped.

## Answer noise 0

| Model | SR@5 | SR@10 | SR@20 | AT |
|---|---|---|---|---|
| CECR | 0.688 +- 0.021 | 0.847 +- 0.024 | 0.948 +- 0.005 | 5.20 +- 0.23 |
| CECR+ask | 0.717 +- 0.021 | 0.895 +- 0.015 | 0.976 +- 0.005 | 4.79 +- 0.24 |
| CECR+policy | 0.693 +- 0.011 | 0.848 +- 0.024 | 0.948 +- 0.005 | 5.17 +- 0.21 |
| EAR | 0.643 +- 0.027 | 0.789 +- 0.017 | 0.920 +- 0.015 | 6.13 +- 0.30 |
| FPAN | 0.689 +- 0.011 | 0.850 +- 0.014 | 0.947 +- 0.011 | 5.25 +- 0.24 |
| KBRD | 0.673 +- 0.012 | 0.866 +- 0.010 | 0.953 +- 0.008 | 5.27 +- 0.21 |
| UNICORN | 0.087 +- 0.021 | 0.510 +- 0.040 | 0.755 +- 0.039 | 11.79 +- 0.51 |

Paired tests vs **CECR+ask** (pooled over seeds):

| Model | mean turns diff | Wilcoxon p | SR@10 diff | McNemar p |
|---|---|---|---|---|
| CECR | +0.41 | 0.85 | -0.048 | 2.4e-24 |
| CECR+policy | +0.38 | 0.37 | -0.048 | 7.5e-24 |
| EAR | +1.34 | 3.1e-28 | -0.106 | 6.2e-41 |
| FPAN | +0.46 | 0.08 | -0.045 | 2.5e-11 |
| KBRD | +0.48 | 1.5e-05 | -0.030 | 1.6e-05 |
| UNICORN | +7.00 | 0 | -0.385 | 8.5e-204 |

(positive turns diff = the model needs more turns than the reference)

## Answer noise 0.1

| Model | SR@5 | SR@10 | SR@20 | AT |
|---|---|---|---|---|
| CECR | 0.687 +- 0.024 | 0.840 +- 0.027 | 0.940 +- 0.019 | 5.29 +- 0.36 |
| CECR+ask | 0.678 +- 0.024 | 0.830 +- 0.022 | 0.900 +- 0.017 | 5.80 +- 0.40 |
| CECR+policy | 0.693 +- 0.011 | 0.846 +- 0.021 | 0.946 +- 0.005 | 5.20 +- 0.18 |
| EAR | 0.643 +- 0.027 | 0.789 +- 0.017 | 0.920 +- 0.015 | 6.13 +- 0.30 |
| FPAN | 0.689 +- 0.011 | 0.850 +- 0.014 | 0.947 +- 0.011 | 5.25 +- 0.24 |
| KBRD | 0.660 +- 0.011 | 0.816 +- 0.033 | 0.866 +- 0.059 | 6.07 +- 0.64 |
| UNICORN | 0.060 +- 0.011 | 0.278 +- 0.038 | 0.380 +- 0.036 | 15.68 +- 0.47 |

Paired tests vs **CECR+ask** (pooled over seeds):

| Model | mean turns diff | Wilcoxon p | SR@10 diff | McNemar p |
|---|---|---|---|---|
| CECR | -0.51 | 1.4e-09 | +0.011 | 0.13 |
| CECR+policy | -0.60 | 7.8e-13 | +0.016 | 0.021 |
| EAR | +0.33 | 1.9e-05 | -0.040 | 5.2e-06 |
| FPAN | -0.55 | 2.3e-05 | +0.020 | 0.012 |
| KBRD | +0.27 | 0.001 | -0.014 | 0.061 |
| UNICORN | +9.87 | 0 | -0.552 | 0 |

(positive turns diff = the model needs more turns than the reference)

## Answer noise 0.2

| Model | SR@5 | SR@10 | SR@20 | AT |
|---|---|---|---|---|
| CECR | 0.685 +- 0.029 | 0.837 +- 0.032 | 0.936 +- 0.027 | 5.35 +- 0.48 |
| CECR+ask | 0.639 +- 0.020 | 0.771 +- 0.024 | 0.832 +- 0.016 | 6.74 +- 0.32 |
| CECR+policy | 0.692 +- 0.012 | 0.844 +- 0.022 | 0.944 +- 0.007 | 5.22 +- 0.20 |
| EAR | 0.643 +- 0.027 | 0.789 +- 0.017 | 0.920 +- 0.015 | 6.13 +- 0.30 |
| FPAN | 0.689 +- 0.011 | 0.850 +- 0.014 | 0.947 +- 0.011 | 5.25 +- 0.24 |
| KBRD | 0.641 +- 0.021 | 0.772 +- 0.048 | 0.806 +- 0.075 | 6.75 +- 0.86 |
| UNICORN | 0.041 +- 0.004 | 0.138 +- 0.024 | 0.174 +- 0.020 | 17.90 +- 0.29 |

Paired tests vs **CECR+ask** (pooled over seeds):

| Model | mean turns diff | Wilcoxon p | SR@10 diff | McNemar p |
|---|---|---|---|---|
| CECR | -1.39 | 2.8e-33 | +0.066 | 2e-16 |
| CECR+policy | -1.53 | 5.6e-41 | +0.073 | 6.4e-21 |
| EAR | -0.62 | 0.035 | +0.018 | 0.057 |
| FPAN | -1.49 | 2.4e-23 | +0.079 | 2.1e-19 |
| KBRD | +0.01 | 0.22 | +0.000 | 1 |
| UNICORN | +11.16 | 0 | -0.634 | 0 |

(positive turns diff = the model needs more turns than the reference)
