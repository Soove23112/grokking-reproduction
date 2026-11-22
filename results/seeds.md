# Key frequencies across seeds

Key frequencies at epoch 19,500 (the last checkpoint; the plan said 20 000, which is never saved); neurons: how many neurons have that frequency as their best one with FVE > 0.85; ranks of each final key frequency among the 56 W_E frequency norms.

## First grid

| run | init | data | t_test | key frequencies | neurons | most neurons on another frequency | ranks at epoch 0 | 500 | 1 000 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | 0 | 0 | 6800 | 4, 13, 25, 35 | 156, 155, 68, 60 | 0 | 9, 4, 17, 15 | 4, 3, 2, 1 | 5, 3, 2, 1 |
| seeds_i1_d0 | 1 | 0 | 5200 | 1, 7, 14, 17, 43 | 114, 105, 65, 71, 103 | 0 | 7, 46, 25, 37, 22 | 3, 8, 2, 1, 4 | 3, 7, 2, 1, 5 |
| seeds_i2_d0 | 2 | 0 | 8200 | 7, 16, 28, 29 | 84, 153, 102, 69 | 0 | 18, 39, 26, 35 | 2, 5, 3, 1 | 2, 4, 3, 1 |
| seeds_i3_d0 | 3 | 0 | 9000 | 14, 31, 49, 56 | 143, 78, 69, 125 | 0 | 29, 20, 16, 14 | 3, 2, 1, 7 | 3, 2, 1, 7 |
| seeds_i4_d0 | 4 | 0 | 7700 | 7, 8, 14, 30 | 69, 27, 116, 130 | 0 | 33, 1, 7, 40 | 3, 2, 1, 4 | 3, 2, 1, 4 |
| seeds_i5_d0 | 5 | 0 | 5800 | 7, 25, 34, 42 | 177, 111, 73, 83 | 0 | 4, 34, 1, 7 | 3, 5, 1, 2 | 3, 5, 1, 2 |
| seeds_i6_d0 | 6 | 0 | 3900 | 14, 29, 42, 48 | 85, 93, 123, 177 | 0 | 18, 2, 22, 9 | 3, 2, 6, 4 | 3, 2, 5, 4 |
| seeds_i7_d0 | 7 | 0 | 6800 | 7, 23, 34, 55 | 115, 103, 95, 98 | 0 | 33, 28, 5, 2 | 2, 4, 1, 3 | 2, 4, 1, 3 |
| seeds_i0_d0_rep | 0 | 0 | 7500 | 4, 13, 25, 35 | 175, 149, 61, 59 | 0 | 9, 4, 17, 15 | 4, 3, 2, 1 | 5, 3, 2, 1 |
| seeds_i0_d1 | 0 | 1 | 8800 | 3, 7, 8, 21, 35 | 90, 30, 83, 76, 104 | 0 | 11, 23, 32, 5, 15 | 2, 1, 4, 3, 5 | 3, 1, 4, 2, 5 |
| seeds_i0_d2 | 0 | 2 | 11100 | 4, 13, 15, 31 | 112, 105, 125, 56 | 0 | 9, 4, 35, 7 | 2, 5, 3, 1 | 2, 4, 3, 1 |
| seeds_i1_d1 | 1 | 1 | 4800 | 3, 7, 11, 45 | 88, 54, 83, 77 | 0 | 1, 46, 4, 34 | 1, 9, 3, 2 | 1, 9, 3, 2 |
| seeds_i1_d2 | 1 | 2 | 5100 | 3, 26, 40 | 102, 176, 114 | 0 | 1, 2, 48 | 1, 3, 2 | 1, 3, 2 |

## Hypotheses

1. Counts on data seed 0 (grokked runs): [4, 4, 4, 4, 4, 4, 4, 5].
2. Mean Jaccard, same init / different split: 0.12 (6 pairs); different init / same split: 0.10 (30 pairs).
3. Repeat of the main model: [4, 13, 25, 35] (main model at epoch 19,500: [4, 13, 25, 35]).
4. Runs with at least 3 final key frequencies in the W_E top 5 at epoch 1 000: 8 of 8.
5. Final key frequencies in the W_E top 10 at epoch 0: 12 of 33 (36%; chance 18%).

## Exploratory (found after looking at the first grid, not pre-registered)

- Data seed 0, 9 models incl. the authors: most frequent key frequencies 7 (5), 14 (5), 42 (3), 25 (2); multiples of 7: 18 of 38 (chance 8/56 = 14%).
- Data seed 1, 2 models: most frequent key frequencies 3 (2), 7 (2), 8 (1), 21 (1); multiples of 7: 4 of 9 (chance 8/56 = 14%).
- Data seed 2, 2 models: most frequent key frequencies 4 (1), 13 (1), 15 (1), 31 (1); multiples of 7: 0 of 7 (chance 8/56 = 14%).

## Follow-up: symmetry check and seeds 8–15

### Symmetry check

Run sym_i0_d0_u7: key frequencies [5, 18, 49, 52], expected [5, 18, 49, 52]; ranks at epochs 0 / 500 / 1 000 equal to those of the matching frequencies of the main model: True; t_test 6300 (main model 6800, its repeat 7500).

### Seeds 8–15 on data seed 0

| run | init | data | t_test | key frequencies | neurons | most neurons on another frequency | ranks at epoch 0 | 500 | 1 000 |
|---|---|---|---|---|---|---|---|---|---|
| seeds_i8_d0 | 8 | 0 | 3400 | 14, 17, 42, 49 | 77, 153, 83, 134 | 0 | 33, 21, 29, 18 | 2, 3, 1, 4 | 2, 4, 1, 3 |
| seeds_i9_d0 | 9 | 0 | 6200 | 23, 29, 34 | 194, 111, 48 | 0 | 29, 2, 11 | 4, 1, 2 | 5, 1, 2 |
| seeds_i10_d0 | 10 | 0 | 10400 | 5, 7, 20, 43 | 128, 109, 96, 40 | 0 | 29, 27, 18, 34 | 6, 1, 2, 3 | 5, 1, 2, 4 |
| seeds_i11_d0 | 11 | 0 | 5500 | 1, 34, 42 | 101, 112, 152 | 0 | 11, 27, 28 | 1, 3, 2 | 1, 3, 2 |
| seeds_i12_d0 | 12 | 0 | 8400 | 25, 30, 31, 42 | 97, 47, 90, 123 | 0 | 31, 1, 3, 19 | 2, 1, 5, 7 | 2, 1, 6, 5 |
| seeds_i13_d0 | 13 | 0 | 6300 | 2, 7, 17, 42 | 124, 76, 120, 133 | 0 | 28, 20, 9, 15 | 2, 1, 5, 4 | 2, 1, 5, 4 |
| seeds_i14_d0 | 14 | 0 | 10300 | 7, 14, 34, 37, 39 | 130, 59, 102, 79, 103 | 0 | 6, 3, 43, 2, 1 | 5, 2, 1, 4, 3 | 5, 2, 1, 4, 3 |
| seeds_i15_d0 | 15 | 0 | 9500 | 25, 40, 52 | 116, 121, 131 | 0 | 29, 4, 36 | 2, 1, 5 | 2, 1, 5 |
| sym_i0_d0_u7 | 0 | 0 | 6300 | 5, 18, 49, 52 | 56, 171, 150, 72 | 0 | 15, 4, 9, 17 | 1, 3, 4, 2 | 1, 3, 5, 2 |

6. Frequency 7 among the key frequencies: 3 of 8 new runs (prediction: at least 3). Frequency 14: 2 of 8.
7. Runs with exactly 4 key frequencies: 4 of 8 grokked new runs (prediction: at least 5); counts [3, 3, 3, 4, 4, 4, 4, 5].

All 16 of our models on data seed 0: counts {3: 3, 4: 11, 5: 2}; most frequent key frequencies 7 (8), 14 (6), 42 (6), 34 (5), 25 (4).

Rank of frequency 7 among the 56 W_E frequencies at epoch 0, all models on data seed 0 (* = 7 becomes a key frequency): seed0 23, i1_d0 46*, i2_d0 18*, i3_d0 50, i4_d0 33*, i5_d0 4*, i6_d0 32, i7_d0 33*, i8_d0 26, i9_d0 36, i10_d0 27*, i11_d0 4, i12_d0 43, i13_d0 20*, i14_d0 6*, i15_d0 50.

### Key-frequency count under other thresholds of the rule

| run | FVE > 0.8, ≥ 5 | FVE > 0.8, ≥ 10 | FVE > 0.8, ≥ 20 | FVE > 0.85, ≥ 5 | FVE > 0.85, ≥ 10 | FVE > 0.85, ≥ 20 | FVE > 0.9, ≥ 5 | FVE > 0.9, ≥ 10 | FVE > 0.9, ≥ 20 |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i1_d0 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| seeds_i2_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i3_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i4_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 3 |
| seeds_i5_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i6_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i7_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i0_d0_rep | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i0_d1 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 4 |
| seeds_i0_d2 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i1_d1 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i1_d2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| seeds_i8_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i9_d0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| seeds_i10_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i11_d0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| seeds_i12_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i13_d0 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| seeds_i14_d0 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| seeds_i15_d0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| sym_i0_d0_u7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
