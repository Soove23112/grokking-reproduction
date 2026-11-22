# Pre-registration

This document records what we fixed before running our own model, and how. The results are in the [README](../README.md).

The success criteria, the key-frequency rule, the phase-boundary rules and the weight decay hypotheses were written down on 18 November 2026, after a 2 000-epoch smoke test of the training loop and before the main run (`runs/seed0`) started on 19 November; the criterion "100% training accuracy by epoch 2 000" was therefore set with the smoke test in view. The training-fraction hypotheses were written down on 19 November, eight seconds before the first training-fraction run started. These records live in the project's working history, which is not part of this repository. Where we later departed from these rules, the README says so and reports the pre-registered version alongside.

## Verification strategy

The authors published their full training run, `grokking_addition_full_run.pth`: weights every 100 epochs, the recorded losses and the configuration. The model analysed in the paper is `state_dicts[400]`, the checkpoint at epoch 40 000. We therefore check in three directions.

1. Our model code on the authors' weights reproduces the losses they recorded. This tests the model and the data split.
2. Our analysis code on the authors' trajectory reproduces their numbers and their phase boundaries. This tests the analysis and calibrates the phase thresholds.
3. The same code on our own model shows the same phenomenon. This is the reproduction.

The key frequencies of a new seed will differ from the authors' (the paper says so), so on our model we compare their number and the structure they produce, not their values.

We committed to analysing the main run (`init_seed = 0`, the paper's configuration) whatever its outcome, without trying further seeds until one worked. If the model had not grokked by epoch 40 000, we would first have checked the weight decay, the training fraction and the attention mask rather than change hyperparameters.

## Success criteria

| what we check | paper | on the authors' weights | on our model |
|---|---|---|---|
| number of parameters | 226 816 (from the code) | — | exactly 226 816 |
| losses at the authors' checkpoints | recorded in their file | match to 1e-6 relative | — |
| 100% training accuracy | by ~1 400 epochs | — | by ≤ 2 000 epochs |
| > 99% test accuracy | ~10 000–14 000 epochs | — | epoch recorded; > 40 000 means not reproduced |
| key frequencies (rule below) | 14, 35, 41, 42, 52 | exactly these five | 3 to 6 frequencies |
| W_L: residual outside the 10 key directions | < 0.55% | < 0.6% | < 2% |
| fraction of logit variance explained by the cosine fit | 95% | 94–96% | ≥ 90% |
| neurons explained by one frequency (> 85%) | 433 / 512 | 433 ± 5 | ≥ 70% |
| removing one key frequency from the logits | the loss rises | yes, for each | yes, for each |
| MLP output projected onto the null space of W_L | 5.27 | 5.27 ± 0.1 | worse than ln 113 |
| order of phases (thresholds below) | 1 400 / 9 400 / 14 000 | ± 1 000 after calibration | t_excl < t_restr < t_test |

We do not compare relative changes of the loss (−70%, +3%) on our model against the paper. They are meaningful only in float64: the float32 resolution is about 10⁻⁷, the same order as the losses themselves.

## Key-frequency rule

A frequency is key if at least N = 10 neurons have more than 85% of their variance explained by it. We consider frequencies 1 to 56; the authors' `calculate_key_freqs` iterates over `range(1, p//2)` and so skips 56.

Before applying the rule to our model we check that on the authors' weights it gives exactly their five frequencies. If it did not, we would tune N on the authors' weights and fix it. On the authors' weights the rule gives 14, 35, 41, 42 and 52 with N = 10, so N was not changed.

The set of key frequencies is determined on the final checkpoint and then applied to every checkpoint of the trajectory.

## Phase-boundary rules

The paper places the phase boundaries by eye on figure 7, at 1 400 / 9 400 / 14 000 epochs. We pre-registered three rules:

- t_excl: the first epoch at which the excluded loss on the training set exceeds its minimum by a factor k;
- t_restr: the first epoch at which the restricted loss falls below a fraction q of its initial value;
- t_test: the first epoch at which the test loss falls below 10⁻³.

The parameters k and q were to be calibrated on the authors' trajectory so as to give approximately 1 400 / 9 400 / 14 000, recorded, and only then applied to our model. The restricted loss is computed over all 12 769 pairs, as in the authors' code (`mode='all'`, `bias_correction=True`). The excluded loss is computed on the training set with all key frequencies removed at once, as in the authors' notebook (`excluded_loss_2D_full`). The plan also called for showing the per-frequency variant from `transformers.py` alongside; the reported results use the all-at-once version only.

### Calibration and the change of rules

On the authors' trajectory the pre-registered rules cannot give the paper's boundaries. t_test is 11 200, not 14 000, and for any k > 1 t_excl lies after the minimum of the excluded loss, so it cannot coincide with the end of memorisation at 1 400. With k = 1.1 and q = 1.6·10⁻³, calibrated for 1 400 and 9 400, the rules give 1 600 / 9 400 / 11 200 on the authors' trajectory and 1 000 / 6 100 / 7 100 on ours.

We replaced them with the rules now in `grokking/phases.py`:

- t1, end of memorisation: the epoch of the minimum of the excluded loss;
- t2, start of cleanup: the first epoch with test loss below ln 113;
- t3, end of cleanup: the first epoch at which the sum of squared weights is at most 1.05 times its final value.

The only parameter is the 5% in t3. On the authors' trajectory the rules give 1 400 / 9 400 / 13 800 (the test loss is 4.85 at epoch 9 300 and 4.15 at 9 400). The rules were calibrated on the authors' trajectory only, but they were written after our model's trajectory had been computed; of that trajectory we had seen only the log lines for epochs 0 and 5 000. The t3 threshold is sensitive: 1.02 gives 16 100 on the authors' trajectory and 1.1 gives 13 000.

## Weight decay sweep

- Grid: λ ∈ {0, 0.3, 1, 3}, initialisation seeds 0, 1 and 2, data split seed 0; 12 runs, metrics only. The run with λ = 1 and seed 0 is the main run.
- Hypotheses, on the direction only: at λ = 0 there is no jump within 40 000 epochs; t_test(0.3) > t_test(1) > t_test(3).
- Metric: t_test, the first epoch with test accuracy above 99%. A run without a jump is recorded as "> 40 000".
- Over the three seeds we report the median and the range; the median is computed with the censored values included. We report no p-values, which are not meaningful on three points.
- Curves per seed are drawn as in figure 2 of the paper.
- The contradiction in the text of appendix D.1 is discussed in the report.

## Training-fraction experiment

Appendix C.2.2 of the paper (figures 20 and 21) states that with 10 to 20% of the pairs in training the model does not generalise within 40 000 epochs, that at 30 to 50% it groks, later the less data there is, and that from 60% up it generalises immediately. The explanation proposed by Varma et al. (2023) is that memorisation needs more weight the more pairs there are, while the algorithm needs the same weight regardless, so at small fractions memorisation is cheaper than the algorithm and at large fractions it is more expensive.

Runs:

- fractions 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 and 0.9;
- initialisation seeds 0, 1 and 2, a single split (data seed 0): the training pairs are the first frac · 12 769 pairs of the same shuffled list;
- everything else as for the main model: λ = 1, learning rate 10⁻³, up to 40 000 epochs;
- a run stops early once the test loss falls below 10⁻⁶, after which nothing changes; for fraction 0.3 and seed 0 we use the main run.

Metrics, logged every 100 epochs:

- t_train, the first epoch with 100% training accuracy;
- t_test, the first epoch with test accuracy above 99%, or "> 40 000" if there is none;
- delay = t_test − t_train;
- additionally, as in figure 21, the first epochs at which the training and the test loss fall below 10⁻⁶.

Hypotheses:

1. Fractions 0.1 and 0.2: t_test > 40 000 for all three seeds.
2. Fractions 0.3, 0.4 and 0.5: every seed groks, with t_train ≤ 1 000 and a delay of at least 1 000 epochs.
3. The median t_test decreases: 0.3 > 0.4 > 0.5.
4. Fractions 0.6 to 0.9: the model generalises immediately, with a delay below 1 000 epochs for every seed.

The 1 000-epoch threshold in hypotheses 2 and 4 was chosen before the runs. It is a convention: the paper gives no number for "immediately".

The run with fraction 0.3 and seed 0 is the main model and was known when these hypotheses were written; at 30% only seeds 1 and 2 were run blind.

## Key frequencies across seeds

Written on 20 November 2026 before any of the runs below started. Known at that time: the main model (`init_seed = 0`, data seed 0) has key frequencies 4, 13, 25, 35; the authors' model (unseeded initialisation, data seed 0) has 14, 35, 41, 42, 52; both sets are unchanged from epoch 15 000 to the end of training; in the authors' trajectory three of the five key frequencies (14, 42, 52) are the top three W_E frequencies by epoch 100, while at initialisation they rank 18th, 8th and 2nd.

Question: what fixes which frequencies a model uses and how many, the random initialisation or the training split?

Runs, all with λ = 1, fraction 0.3, learning rate 10⁻³, 20 000 epochs on MPS, checkpoints every 500 epochs:

- initialisation seeds 1 to 7 with data seed 0 (with the main model, 8 initialisations on one split);
- a repeat of the main model, initialisation seed 0 and data seed 0, to measure run-to-run differences on MPS, which is not bit-reproducible;
- initialisation seeds 0 and 1, each with data seeds 1 and 2 (4 runs).

Metrics, at epoch 20 000 with the existing rule (a frequency is key if it is the best frequency, with FVE above 0.85, of at least 10 neurons):

- the set of key frequencies of each run and its size;
- overlap between two runs: the Jaccard index of their key sets (size of the intersection over size of the union); for random 4- or 5-element subsets of 56 frequencies it is about 0.04;
- for each run, the rank of every final key frequency among the 56 W_E frequency norms at epochs 0, 500 and 1 000. (Corrected a few minutes after the runs started and before any result was seen: the first draft also listed epoch 100, for which checkpoints every 500 epochs give no weights.)

Hypotheses:

1. Every run that groks has 3 to 6 key frequencies, and the 8 initialisations on data seed 0 do not all have the same number.
2. The initialisation matters more than the split: the mean Jaccard index over pairs with the same initialisation and different splits is higher than over pairs with different initialisations and the same split.
3. The repeat of the main model has the key frequencies 4, 13, 25, 35.
4. Frequencies are chosen early: in at least 6 of the 8 initialisations on data seed 0, the top five W_E frequencies at epoch 1 000 contain at least 3 of the final key frequencies.
5. The initial weights do not pick the frequencies through their norm: pooled over the 8 initialisations, at most 30% of the final key frequencies are among the top 10 W_E frequencies at epoch 0 (chance is 10/56, about 18%).

A run that does not reach 99% test accuracy by epoch 20 000 is reported as not grokked and left out of hypotheses 1, 2 and 4.

Deviation, noted after the runs: a 20 000-epoch run trains epochs 0 to 19 999, so with checkpoints every 500 epochs the last saved weights are at epoch 19 500, not 20 000. Key frequencies are read at epoch 19 500 for every run, the main model included; they are stable from epoch 15 000 on.

## Follow-up: symmetry check and seeds 8–15

Written on 21 November 2026 before these runs started, after the results of the section above were seen. Those results showed frequency 7 in 5 of 9 models on data seed 0 (authors included), frequency 14 in 5 of 9, and 4 key frequencies in 7 of 8 of our initialisations.

Symmetry check, one run (`sym_i0_d0_u7`): initialisation seed 0, data seed 0, every number renamed x → 7x mod 113 in both the training data and the initial weights (`--relabel 7`). If the code treats all numbers alike, this is the main model under new names, so its key frequencies must be those of the main model multiplied by 7⁻¹ = 97 mod 113 and folded into 1..56: 5, 18, 49, 52. The ranks of these frequencies in W_E at epochs 0, 500 and 1 000 must be those of the main model (9, 4, 17, 15; 4, 3, 2, 1; 5, 3, 2, 1). Any other set means the code breaks the symmetry, and the frequency counts across seeds have to be re-examined.

Seeds 8–15, data seed 0, the same settings as above (8 runs):

6. If data seed 0 itself favours frequency 7, it appears among the key frequencies of at least 3 of the 8 new runs. If every frequency were equally likely, a given frequency would do so with probability about 2%. The same is reported for frequency 14, without a separate prediction.
7. Four key frequencies is the usual count: at least 5 of the 8 new runs have exactly 4.

The count is also reported under other thresholds of the key-frequency rule (at least 5, 10 or 20 neurons; FVE above 0.8, 0.85 or 0.9), and with the number of neurons on each key frequency, to see whether a fifth frequency is a borderline case of the rule.

## Keeping a few frequencies in W_E

Written on 21 November 2026 before any run of `scripts/steer.py`. Known at that time, from a pilot in a scratch script (CPU, one run per condition, test accuracy checked every 50 epochs): on the main model (`init_seed = 0`) the baseline grokked at epoch 6 350; projecting the number embeddings W_E[:113] once at epoch 100 onto the constant and the cos/sin of the key frequencies 4, 13, 25, 35 gave 400; the same with a random set of 4 frequencies gave 2 150 (2 650 on `init_seed = 2`, where the key frequencies gave 400); re-applying the projection after every step gave 250 for the key frequencies and 300 to 450 for four random sets; multiplying W_E by 0.32 at epoch 100 without changing its directions gave 5 450; keeping the top 9 or 20 SVD directions of W_E, or a random 9-dimensional subspace, after every step never reached 99% test accuracy in 15 000 epochs. On `init_seed = 2` the 4 largest W_E frequencies at epoch 100 were 7, 29, 28, 16, its final key frequencies.

Question: does keeping only a few frequencies in W_E speed up grokking, is the speed-up more than the shrinking of W_E, and does it matter which frequencies are kept?

Runs: initialisation seeds 0 to 4, data seed 0, λ = 1, fraction 0.3, CPU, test accuracy checked every 10 epochs, at most 15 000 epochs, 1 000 more epochs after the test accuracy first exceeds 99%. Six conditions per initialisation, all applied at epoch 100 (definitions in `scripts/steer.py`): `none`, `scale`, `top4_once`, `rand4_once`, `top4_persist`, `rand4_persist`. "Top 4" is read from W_E at epoch 100 with no knowledge of the final model; the random 4 are drawn outside the top 4 with a seed fixed by the initialisation; `scale` multiplies W_E by the factor that `top4_once` would shrink its norm by.

Metric: t_test, the first epoch with test accuracy above 99%; the speed-up of a condition is t_test(none) / t_test(condition) on the same initialisation.

Hypotheses:

1. The detected-frequency projection works: `top4_once` groks earlier than `none` in 5 of 5 initialisations, with a median speed-up of at least 5.
2. Shrinking alone does little: the median speed-up of `scale` is at most 1.5.
3. In the one-off form it matters which frequencies are kept: `top4_once` groks earlier than `rand4_once` in at least 4 of 5 initialisations.
4. Under the constraint after every step any frequencies work: `rand4_persist` reaches 99% in 5 of 5 initialisations, earlier than `none` in 5 of 5, and t_test(rand4_persist) / t_test(top4_persist) lies between 0.5 and 2 in at least 4 of 5.
5. The constrained models with random frequencies run the same algorithm on those frequencies: in each `rand4_persist` final model at least 3 of the 4 forced frequencies are key frequencies by the existing rule, no other frequency is, and its test loss is within a factor 10 of the final test loss of `none` on the same initialisation.
6. The top 4 W_E frequencies at epoch 100 contain at least 3 of the final key frequencies of the `none` run in at least 4 of 5 initialisations.

A run that does not reach 99% by epoch 15 000 counts as not faster than any run that does.

## Same starting state: top-4 against random 4 frequencies of equal norm

Written on 21 November 2026 before any run of `scripts/steer_shared.py`, after the results of the section above (in `results/steer.md`): projecting W_E once at epoch 100 onto its 4 largest frequencies gave a median speed-up of 18.0 over 5 initialisations; onto 4 random frequencies, 7.7; `top4_once` was earlier than `rand4_once` in 5 of 5. In those runs the random projection kept a smaller part of W_E than the top-4 one, and each condition was trained from scratch, so the comparison mixed the choice of frequencies with the size of W_E.

Design. For each initialisation seed 0 to 4 (data seed 0, λ = 1, fraction 0.3, CPU) the model is trained to the start of epoch 100 once, and that state — weights, AdamW moments and the learning-rate schedule — is saved. Three continuations start from it:

- `none`: no change;
- `top4`: W_E[:113] projected onto the constant and the cos/sin of its 4 largest frequencies at that moment;
- `rand4n`: projected onto the constant and the cos/sin of 4 random frequencies outside those 4 (seed fixed by the initialisation, the same draw as in `scripts/steer.py`), then multiplied by the factor that makes the norm of W_E[:113] equal to its norm after the `top4` projection. The factor is recorded.

Train and test loss and accuracy are recorded at epoch 100 just before and just after the change, then every 10 epochs; weights are saved at those points up to epoch 400 and every 100 epochs after. Each run stops 1 000 epochs after test accuracy first exceeds 99%, or at epoch 15 000. Restricted and excluded loss are computed afterwards from the saved weights, for the projected frequencies and for the final key frequencies of each run; they are reported only after checking that they remain meaningful when the final key frequencies differ from the projected ones.

Metrics: t_test (first epoch with test accuracy above 99%), the time after the change (t_test − 100), and wall-clock minutes; ratios are computed per initialisation and reported as median and range.

Hypotheses:

1. `top4` reaches 99% earlier than `rand4n` in at least 4 of 5 initialisations, and the median of the per-initialisation ratios t_test(rand4n) / t_test(top4) is at least 1.5.
2. `rand4n` still reaches 99% earlier than `none` in 5 of 5 initialisations.

The immediate change in train and test loss and accuracy at the projection, and the course of the recovery, are reported descriptively, with no prediction and no explanation assigned in advance.
