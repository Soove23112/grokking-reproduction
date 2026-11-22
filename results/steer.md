# Keeping a few frequencies in W_E

t_test: first epoch with test accuracy above 99% (checked every 10 epochs); speed-up: t_test(none) / t_test.

| init | none | scale | top4_once | rand4_once | top4_persist | rand4_persist | top 4 at epoch 100 | final keys of none |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 660 | 5 330 | 370 | 860 | 230 | 430 | [35, 25, 13, 4] | [4, 13, 25, 35] |
| 1 | 5 100 | 3 790 | 440 | 870 | 620 | 250 | [14, 17, 1, 38] | [1, 14, 17, 43] |
| 2 | 8 100 | 7 150 | 390 | 540 | 530 | 220 | [7, 28, 29, 16] | [7, 16, 28, 29] |
| 3 | 8 590 | 6 530 | 460 | 1 140 | 320 | 280 | [49, 14, 2, 50] | [14, 31, 49, 56] |
| 4 | 8 150 | 6 810 | 490 | 610 | 260 | 330 | [14, 8, 7, 38] | [7, 8, 14, 30, 40] |

Median speed-up: scale 1.2, top4_once 18.0, rand4_once 7.7, top4_persist 26.8, rand4_persist 24.7.

| init | cond | forced / kept frequencies | final key frequencies | final test loss | W_E norm kept |
|---|---|---|---|---|---|
| 0 | none |  | [4, 13, 25, 35] | 1.16e-05 |  |
| 0 | scale |  | [13, 25, 35] | 9.17e-07 | 0.3157000243663788 |
| 0 | top4_once | [35, 25, 13, 4] | [4, 13, 25, 35] | 4.87e-07 | 0.3157000243663788 |
| 0 | rand4_once | [11, 30, 36, 48] | [11, 30, 48] | 2.96e-07 | 0.3157000243663788 |
| 0 | top4_persist | [35, 25, 13, 4] | [4, 13, 25, 35] | 2.51e-05 | 0.3157000243663788 |
| 0 | rand4_persist | [11, 30, 36, 48] | [11, 30, 36, 48] | 1.46e-03 | 0.3157000243663788 |
| 1 | none |  | [1, 14, 17, 43] | 5.83e-06 |  |
| 1 | scale |  | [1, 14, 17, 43] | 7.99e-06 | 0.31705325841903687 |
| 1 | top4_once | [14, 17, 1, 38] | [1, 14, 17] | 2.84e-07 | 0.31705325841903687 |
| 1 | rand4_once | [2, 34, 46, 49] | [2, 46, 49] | 3.77e-07 | 0.31705325841903687 |
| 1 | top4_persist | [14, 17, 1, 38] | [1, 14, 17] | 2.55e-03 | 0.31705325841903687 |
| 1 | rand4_persist | [2, 34, 46, 49] | [2, 46, 49] | 8.60e-07 | 0.31705325841903687 |
| 2 | none |  | [7, 16, 28, 29] | 2.82e-06 |  |
| 2 | scale |  | [7, 16, 28, 29] | 3.14e-04 | 0.31445398926734924 |
| 2 | top4_once | [7, 28, 29, 16] | [7, 16, 28, 29] | 3.66e-07 | 0.31445398926734924 |
| 2 | rand4_once | [21, 22, 34, 44] | [34, 44] | 2.52e-07 | 0.31445398926734924 |
| 2 | top4_persist | [7, 28, 29, 16] | [7, 16, 28, 29] | 1.83e-03 | 0.31445398926734924 |
| 2 | rand4_persist | [21, 22, 34, 44] | [34, 44] | 7.86e-07 | 0.31445398926734924 |
| 3 | none |  | [14, 31, 49, 56] | 8.61e-07 |  |
| 3 | scale |  | [1, 14, 31, 49, 56] | 8.74e-07 | 0.30525410175323486 |
| 3 | top4_once | [49, 14, 2, 50] | [2, 14, 49] | 2.75e-07 | 0.30525410175323486 |
| 3 | rand4_once | [11, 16, 17, 32] | [16, 17] | 3.42e-07 | 0.30525410175323486 |
| 3 | top4_persist | [49, 14, 2, 50] | [2, 14, 49] | 4.05e-07 | 0.30525410175323486 |
| 3 | rand4_persist | [11, 16, 17, 32] | [16, 17] | 7.00e-07 | 0.30525410175323486 |
| 4 | none |  | [7, 8, 14, 30, 40] | 1.05e-04 |  |
| 4 | scale |  | [7, 8, 14, 30] | 6.17e-07 | 0.31785503029823303 |
| 4 | top4_once | [14, 8, 7, 38] | [7, 8, 14, 38] | 3.13e-07 | 0.31785503029823303 |
| 4 | rand4_once | [1, 9, 40, 56] | [1, 40, 56] | 2.76e-07 | 0.31785503029823303 |
| 4 | top4_persist | [14, 8, 7, 38] | [7, 8, 14] | 4.69e-07 | 0.31785503029823303 |
| 4 | rand4_persist | [1, 9, 40, 56] | [1, 40, 56] | 4.02e-07 | 0.31785503029823303 |

## Hypotheses

1. top4_once earlier than none: 5 of 5; median speed-up 18.0 (prediction: 5 of 5, at least 5).
2. Median speed-up of scale: 1.2 (prediction: at most 1.5).
3. top4_once earlier than rand4_once: 5 of 5 (prediction: at least 4).
4. rand4_persist reaches 99%: 5 of 5; earlier than none: 5 of 5; rand4_persist / top4_persist within 0.5-2: 3 of 5 (prediction: 5, 5, at least 4).
5. rand4_persist runs the forced frequencies (>= 3 of 4 key, no other key, test loss within 10x of none): 2 of 5 (prediction: 5 of 5).
6. Top 4 at epoch 100 hold >= 3 final keys of none: 4 of 5 (prediction: at least 4).
