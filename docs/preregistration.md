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
