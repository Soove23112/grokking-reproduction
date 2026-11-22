# Does the initial state predict the winning frequencies? (exploratory)

21 runs, one per distinct (initialisation, data) pair, the authors' included. Neurons counted: FVE > 0.85 in the final model. A_a, A_b, A_x: magnitudes of the Fourier coefficients Fa, Fb, Fx of a neuron at epoch 0; D: its phase misalignment (see the docstring of scripts/lottery.py).

## Per neuron: lottery winner at epoch 0 = the neuron's final frequency

| run | clean neurons | exact match | with neurons shuffled |
|---|---|---|---|
| seed0 | 439 | 5.5% | 3.0% |
| seeds_i1_d0 | 458 | 5.7% | 3.1% |
| seeds_i2_d0 | 408 | 3.9% | 1.5% |
| seeds_i3_d0 | 415 | 4.6% | 2.4% |
| seeds_i4_d0 | 342 | 4.1% | 2.4% |
| seeds_i5_d0 | 444 | 5.2% | 3.2% |
| seeds_i6_d0 | 478 | 3.3% | 2.0% |
| seeds_i7_d0 | 411 | 4.4% | 2.4% |
| seeds_i0_d1 | 383 | 4.4% | 2.9% |
| seeds_i0_d2 | 398 | 3.8% | 2.5% |
| seeds_i1_d1 | 302 | 5.6% | 2.3% |
| seeds_i1_d2 | 392 | 4.1% | 2.4% |
| seeds_i8_d0 | 447 | 3.1% | 2.1% |
| seeds_i9_d0 | 353 | 5.1% | 2.3% |
| seeds_i10_d0 | 373 | 3.2% | 2.0% |
| seeds_i11_d0 | 365 | 4.7% | 2.3% |
| seeds_i12_d0 | 357 | 5.6% | 3.0% |
| seeds_i13_d0 | 453 | 3.1% | 2.5% |
| seeds_i14_d0 | 473 | 4.7% | 3.4% |
| seeds_i15_d0 | 368 | 3.5% | 1.9% |
| authors | 433 | 4.2% | 2.7% |

## Over the whole model: rank of each final key frequency by a per-frequency score at epoch 0

Chance: top-n about 7%, top 10 18%, median rank 28.5. 95% intervals: bootstrap over 17 initialisation groups (runs sharing an initialisation resampled together).

| score | keys in top-n | keys in top 10 | 95% interval | median rank |
|---|---|---|---|---|
| full (He et al.): sum over neurons of max(0, A_a A_b A_x cos D) | 24/84 (29%) | 49% | 39%–58% | 11 |
| signed: sum of A_a A_b A_x cos D | 15/84 (18%) | 29% | 20%–36% | 20 |
| magnitude only: sum of A_a A_b A_x | 26/84 (31%) | 54% | 45%–62% | 10 |
| phase only: sum of cos D | 5/84 (6%) | 27% | 20%–35% | 22 |
| input side only: sum of A_a A_b | 27/84 (32%) | 51% | 41%–61% | 10 |
| output side only: sum of A_x | 5/84 (6%) | 33% | 26%–40% | 22 |
| W_E frequency norm | 19/84 (23%) | 37% | 25%–47% | 18 |
