# Same starting state: top-4 against random 4 frequencies of equal norm

t_test: first epoch with test accuracy above 99% (checked every 10 epochs). All continuations of one initialisation start from the same saved state at epoch 100.

| init | none | top4 | rand4n | top4 frequencies | rand4n frequencies | rand4n factor | final key frequencies none / top4 / rand4n | final test loss none / top4 / rand4n |
|---|---|---|---|---|---|---|---|---|
| 0 | 6660 | 370 | 610 | [35, 25, 13, 4] | [11, 30, 36, 48] | 1.176 | [4, 13, 25, 35] / [4, 13, 25, 35] / [11, 30, 48] | 1.2e-05 / 4.9e-07 / 2.6e-07 |
| 1 | 5100 | 440 | 700 | [14, 17, 1, 38] | [2, 34, 46, 49] | 1.144 | [1, 14, 17, 43] / [1, 14, 17] / [2, 46, 49] | 5.8e-06 / 2.8e-07 / 3.6e-07 |
| 2 | 8100 | 390 | 500 | [7, 28, 29, 16] | [21, 22, 34, 44] | 1.140 | [7, 16, 28, 29] / [7, 16, 28, 29] / [34, 44] | 2.8e-06 / 3.7e-07 / 2.7e-07 |
| 3 | 8590 | 460 | 830 | [49, 14, 2, 50] | [11, 16, 17, 32] | 1.135 | [14, 31, 49, 56] / [2, 14, 49] / [16, 17] | 8.6e-07 / 2.8e-07 / 3.7e-07 |
| 4 | 8150 | 490 | 540 | [14, 8, 7, 38] | [1, 9, 40, 56] | 1.167 | [7, 8, 14, 30, 40] / [7, 8, 14, 38] / [1, 40, 56] | 1.0e-04 / 3.1e-07 / 3.1e-07 |

## Hypotheses

1. top4 earlier than rand4n: 5 of 5; t_test(rand4n) / t_test(top4): median 1.59, range 1.10–1.80 (prediction: at least 4 of 5, median at least 1.5).
2. rand4n earlier than none: 5 of 5 (prediction: 5 of 5).

## Speed-ups

- none / top4: median 18.00, range 11.59–20.77.
- none / rand4n: median 10.92, range 7.29–16.20.
- epochs after the change, rand4n / top4: median 1.76, range 1.13–2.03.

## At the projection (epoch 100, just before and just after)

Chance level of the loss: ln 113 = 4.73.

| init | cond | train acc before → after | train loss before → after | test acc before → after | test loss before → after | W_E norm before → after |
|---|---|---|---|---|---|---|
| 0 | top4 | 43.1% → 3.0% | 2.45 → 4.87 | 1.0% → 1.5% | 7.44 → 5.31 | 14.37 → 4.54 |
| 0 | rand4n | 43.1% → 1.9% | 2.45 → 5.12 | 1.0% → 1.0% | 7.44 → 5.49 | 14.37 → 4.54 |
| 1 | top4 | 45.1% → 3.7% | 2.34 → 4.80 | 1.3% → 1.9% | 7.52 → 5.31 | 14.48 → 4.59 |
| 1 | rand4n | 45.1% → 2.1% | 2.34 → 5.06 | 1.3% → 1.2% | 7.52 → 5.43 | 14.48 → 4.59 |
| 2 | top4 | 45.5% → 2.6% | 2.35 → 4.93 | 1.8% → 1.4% | 7.48 → 5.40 | 14.32 → 4.50 |
| 2 | rand4n | 45.5% → 2.2% | 2.35 → 5.08 | 1.8% → 1.6% | 7.48 → 5.44 | 14.32 → 4.50 |
| 3 | top4 | 44.7% → 2.2% | 2.38 → 4.92 | 1.9% → 1.3% | 7.41 → 5.31 | 14.45 → 4.41 |
| 3 | rand4n | 44.7% → 1.9% | 2.38 → 5.04 | 1.9% → 1.0% | 7.41 → 5.37 | 14.45 → 4.41 |
| 4 | top4 | 45.6% → 3.6% | 2.35 → 4.89 | 0.4% → 1.4% | 7.78 → 5.38 | 14.36 → 4.56 |
| 4 | rand4n | 45.6% → 2.7% | 2.35 → 5.01 | 0.4% → 1.3% | 7.78 → 5.41 | 14.36 → 4.56 |
