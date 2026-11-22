# Grokking: why it happens and what controls its timing (map for positioning F2 and F5)

Verification tags used below:
- [FULL] the text was read in this session (PDF or page).
- [ABS] only the arXiv abstract page or repo README was read in this session. Treat details as abstract-level claims.
- [SNIPPET] only a search-engine snippet was seen. Weakest evidence.
- [KNOWN] a well-known paper cited from prior knowledge (arXiv ID and main claim). It was not re-fetched in this session; the report writer should treat the specific numbers as unverified.

Research date: 2026-11-21. About 17 tool calls. OpenReview reviews of Nanda et al. 2023 could not be retrieved (see Gaps).

## Q1. Theories of why delayed generalization (grokking) happens

### Takeaway
By 2026 the field has converged on a "two solutions, different speeds and costs" picture. A fast memorizing solution is found first. A slower generalizing solution then wins because regularization, mostly weight decay, or an implicit bias favours lower norm or higher efficiency. The time between the two is set by how slowly the norm or feature-learning dynamics move. The main variants are circuit efficiency (Varma), lazy-to-rich or kernel-to-feature transitions (Kumar; Lyu), weight-norm "LU" landscapes (Omnigrok), representation-learning effective theory (Liu 2022), and numerical artefacts (slingshot; softmax collapse). Several 2025–2026 papers make this into a "weight-decay clock": delay ∝ 1/(ηλ).

### Cited Findings

Origin and the mechanistic baseline
- Power et al. 2022 coined "grokking" on algorithmic datasets. They found that smaller training fractions need much more optimization to generalize, and that weight decay was the most effective intervention for data efficiency. [KNOWN] — [arXiv 2201.02177](https://arxiv.org/abs/2201.02177)
- Nanda et al. 2023 (ICLR 2023, oral, top 25%) reverse-engineered the Fourier-multiplication algorithm in a 1-layer transformer for a+b mod 113. They split training into memorization, circuit formation, and cleanup. Cleanup is driven by weight decay, and the circuit forms "well before grokking occurs". [FULL] — [arXiv 2301.05217](https://arxiv.org/abs/2301.05217); [ICLR listing](https://iclr.cc/virtual/2023/poster/11385)
- In Nanda et al., memorization finishes around 1,400 epochs in all 5 seeds, but "circuit formation and cleanup occur at different times" across seeds. So the grokking time varies by seed even with fixed hyperparameters. [FULL] — [arXiv 2301.05217, App. C.2.1, Fig. 18](https://arxiv.org/pdf/2301.05217)

Circuit efficiency (memorization vs generalization)
- Varma, Shah, Kenton, Kramár, Kumar 2023 compare two circuits. The memorizing circuit's parameter-norm cost grows with dataset size, while the generalizing circuit's cost is roughly constant. This defines a critical dataset size D_crit. Above D_crit generalization is more "efficient" and wins under weight decay, and grokking delay comes from the generalizing circuit being learned more slowly. They predicted and confirmed "ungrokking" (going back to memorization when retrained on less than D_crit data) and "semi-grokking". [KNOWN] — [arXiv 2309.02390](https://arxiv.org/abs/2309.02390)

Lazy-to-rich / kernel-to-feature-learning
- Kumar, Bordelon, Gershman, Pehlevan (ICLR 2024) frame grokking as a transition from lazy (NTK, kernel) training to rich feature learning. The delay is controlled by the rate of feature learning, set by the output or initial scale, and by how well the initial NTK aligns with the target. They show grokking without weight decay in polynomial regression. [KNOWN] — [arXiv 2310.06110](https://arxiv.org/abs/2310.06110)
- Lyu, Jin, Li, Du, Lee, Hu (ICLR 2024), "Dichotomy of early and late phase implicit biases". With large initialization and small weight decay, homogeneous nets first follow a kernel predictor and later switch abruptly to a min-norm or max-margin predictor. The transition time scales like log(init scale)/λ. [KNOWN; the exact scaling form is from memory and should be checked] — [arXiv 2311.18817](https://arxiv.org/abs/2311.18817)
- Mohamadi et al. 2024, "Why do you grok?". In the kernel regime, modular addition needs a constant fraction of all p² samples, while ℓ∞-regularized two-layer nets generalize with far fewer. This is a theoretical lazy-vs-rich separation. [KNOWN] — [arXiv 2407.12332](https://arxiv.org/abs/2407.12332)

Weight norm (Omnigrok) and effective theory
- Liu, Kitouni, Nolte, Michaud, Tegmark, Williams (NeurIPS 2022), effective theory of representation learning. Generalization happens when structured embeddings (e.g. a ring for modular addition) form. They give phase diagrams over learning rate and weight decay with four phases: comprehension, grokking, memorization, confusion. A "Goldilocks zone" sits between memorization and confusion, and they define a critical training-set size. [KNOWN] — [arXiv 2205.10343](https://arxiv.org/abs/2205.10343)
- Liu, Michaud, Tegmark (ICLR 2023), Omnigrok, the "LU mechanism". Training loss vs weight norm is L-shaped and test loss is U-shaped. A large initial norm puts the model on the memorizing side, and weight decay slowly moves it to the "Goldilocks" norm. Constraining the norm removes most grokking on MNIST, IMDb, and QM9. On algorithmic datasets they argue that representation learning, not just norm, matters. [KNOWN] — [arXiv 2210.01117](https://arxiv.org/abs/2210.01117)
- Ootani 2026 reproduces Omnigrok's inverted-U dependence on weight decay at about 12K parameters. Grokking is "a conditional, fragile phase transition", gated by training-set coverage, and the threshold tracks the modulus. [ABS] — [arXiv 2607.05104](https://arxiv.org/abs/2607.05104)

Numerical and optimizer artefacts
- Thilak et al. 2022, the Slingshot mechanism. With adaptive optimizers (Adam) and no explicit regularization, the last-layer weight norm grows in cycles, and generalization tends to coincide with these "slingshot" spikes. [KNOWN] — [arXiv 2206.04817](https://arxiv.org/abs/2206.04817)
- Prieto, Barsbey, Mediano, Birdal (ICLR 2025), "Grokking at the edge of numerical stability":
  - Without regularization, models drift along a "naive loss minimization" direction that scales up the logits.
  - This leads to Softmax Collapse: floating-point errors make gradients from correctly classified samples zero, which stalls learning.
  - Weight decay delays or prevents this collapse.
  - StableMax and the ⊥Grad optimizer give grokking without regularization.
  [KNOWN plus a search snippet of the ICLR 2025 proceedings] — [arXiv 2501.04697](https://arxiv.org/abs/2501.04697); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/c9e6ac15e689e06139d7b39e1667b165-Paper-Conference.pdf)
- Relevant to reproductions: open issue #2 in the official Nanda et al. repo reports a bug. `cross_entropy_high_precision` casts logits to `torch.float32`, although its own comment says it must use float64 "because log_softmax has a float32 underflow on overly confident data". The issue says this is what causes loss spikes and slingshots in the original code. [FULL] — [GitHub issue #2](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/issues/2)

Pattern-learning speed, double descent, circuit competition
- Davies, Langosco, Krueger 2022/2023 unify grokking and epoch-wise double descent. Patterns are learned at different speeds, and a slow but well-generalizing pattern eventually overtakes a fast memorizing one. [KNOWN] — [arXiv 2303.06173](https://arxiv.org/abs/2303.06173)
- Huang et al. (COLM 2024) give a circuit-competition view that unifies grokking, double descent and emergent abilities through memorization vs generalization circuits and critical dataset size. [KNOWN] — [arXiv 2402.15175](https://arxiv.org/abs/2402.15175)
- Merrill, Tsilivis, Shukla 2023 find grokking on sparse parity to be competition between a dense memorizing subnetwork and a sparse generalizing one. [KNOWN] — [arXiv 2303.11873](https://arxiv.org/abs/2303.11873)

Statistical-physics and solvable models
- Rubin, Seroussi, Ringel (ICLR 2024) treat grokking as a first-order phase transition in two-layer nets. There is a mixed phase ("droplet") in which the generalizing feature coexists with the kernel-like solution. [KNOWN] — [arXiv 2310.03789](https://arxiv.org/abs/2310.03789)
- Žunkovič and Ilievski (JMLR 2024) give exactly solvable grokking phase transitions in models that learn local rules. The grokking time depends on the training-set size and exhibits critical behaviour. [KNOWN] — [arXiv 2210.15435](https://arxiv.org/abs/2210.15435)
- Levi, Beck, Bar-Sinai (ICLR 2024) show that linear teacher–student estimators "grok without understanding". The delay is an accuracy-threshold artefact, and its timing is set analytically by input/output dimension, train size, regularization and init. [KNOWN] — [arXiv 2310.16441](https://arxiv.org/abs/2310.16441)

Grokking without neural nets, and "always grok"
- Mallinar, Beaglehole, Zhu, Radhakrishnan, Pandit, Belkin. Recursive Feature Machines, which are kernel machines with AGOP feature learning, grok modular arithmetic and learn Fourier-like / block-circulant features. So grokking is not specific to neural nets or SGD. [KNOWN] — [arXiv 2407.20199](https://arxiv.org/abs/2407.20199)
- Humayun, Balestriero, Baraniuk (ICML 2024), "Deep networks always grok". Adversarial robustness and the local complexity of the spline partition change late in training ("delayed robustness"), even on standard datasets. [KNOWN] — [arXiv 2402.15555](https://arxiv.org/abs/2402.15555)

2025–2026 theory (weight-decay clock, norm minimization on the zero-loss manifold)
- "A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation" (2025). After the data is fit, gradient flow with small weight decay minimizes the norm along the interpolation manifold, which is a slow second phase. [SNIPPET] — [arXiv 2505.20172](https://arxiv.org/pdf/2505.20172)
- "The Geometry of Grokking: Norm Minimization on the Zero-Loss Manifold" (2025). After memorization, GD effectively minimizes the weight norm on the zero-loss manifold. This is proved only in the limit of infinitesimal learning rate and weight decay. [SNIPPET] — [arXiv 2511.01938](https://arxiv.org/abs/2511.01938)
- "Grokking on the Weight-Decay Clock: A Rate Hierarchy from Softly Broken Symmetries" (2026). It gives an exactly solvable late-time relaxation. Grokking time ∝ (1−β)/(ηλ) in the weak-regularization regime, with β the heavy-ball momentum. On modular addition, the measured delay follows this scaling, and the theory separates coupled L2 from decoupled weight decay. [ABS] — [arXiv 2607.23967](https://arxiv.org/abs/2607.23967)
- "A Spectral Theory of Grokking: Weight Decay induces Feature Learning" (2026). "The grokking timescale is controlled by the product of learning rate and weight decay". They report inverse-product scaling on an 84×90 (lr, wd) grid for MLPs and a 42×45 grid for a one-block transformer on modular addition. Feature learning slows logarithmically near a critical decay, and too much decay prevents fitting. [ABS] — [arXiv 2609.26679](https://arxiv.org/abs/2609.26679)
- He, Wang, Chen, Yang (Feb 2026), "Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking". They describe three stages: memorization, then two generalization phases driven by competition between loss and weight decay. The frequency inside each neuron is chosen by a lottery ticket at initialization. [ABS] — [arXiv 2602.16849](https://arxiv.org/abs/2602.16849)
- "Statistical Vignettes on the Grokking Phenomenon" (2026) argues that delayed generalization can persist without weight decay. Regularization decides whether the target test performance is reachable at all. [SNIPPET] — [arXiv 2610.18811](https://arxiv.org/pdf/2610.18811)
- Kumar and Sankaranarayanan (Oct 2026), "Flat Loss, Evolving Features". In modular addition, the loss is flat while features keep evolving. A theorem gives "a constant gain in the Fourier fraction of the hidden features". [ABS] — [arXiv 2610.22593](https://arxiv.org/abs/2610.22593)
- Prakash and Martin 2025 (HTSR, heavy-tailed self-regularization) report a late "generalization collapse" (anti-grokking) after grokking, even with weight decay. [SNIPPET] — [arXiv 2506.04434](https://arxiv.org/pdf/2506.04434)
- Clauw et al. 2024 study information-theoretic progress measures (synergy). Grokking is an emergent phase transition, and low weight decay gives delayed generalization. [SNIPPET] — [arXiv 2408.08944](https://arxiv.org/pdf/2408.08944)
- Sager, Grewe, Aceituno, Stadelmann (Nov 2026), "Grokking Can Be Steered". Across Nero, Adam, RMSProp, Muon, Shampoo and seven other learning rules, every rule that groks modular addition recovers the same Fourier algorithm. Circuit size, however, depends on the rule: Nero gives distributed circuits, Adam concentrated ones, and credit-scaled weight decay expands circuits from 60% to 99% of neurons. [ABS] — [arXiv 2611.19477](https://arxiv.org/abs/2611.19477)

### Inferences
- Most accounts share the same timing logic. The generalizing solution is preferred at convergence (lower norm, higher margin, more efficient), and how long it takes to get there is set by a slow variable: weight-norm decay at rate ηλ, the rate of feature learning, or the escape from the kernel regime. The circuit-efficiency, Omnigrok, Lyu-dichotomy and 2025–26 "norm minimization on the zero-loss manifold / weight-decay clock" papers are largely the same story at different levels of formality.
- The numerical-stability line (Prieto 2025; float32 issue #2 in the Nanda repo) is separate. It explains failures to grok without weight decay and slingshot spikes as floating-point effects. This matters for reproducing Nanda et al., because their "high-precision" loss is not actually float64.

### Gaps
- The exact constants and functional forms in Lyu et al. (log α/λ) and Varma et al. were not re-verified in this session (marked [KNOWN]).
- The 2607.23967 and 2609.26679 abstracts give no numeric fits or architectures beyond grid sizes. Whether either fits the Nanda 1-layer transformer setup specifically is unknown.

## Q2. Timing knobs: weight decay, data fraction, init scale, learning rate, optimizer, regularizer type

### Takeaway
Weight decay and the learning rate act through their product ηλ, and grokking time scales roughly as 1/(ηλ). Several 2026 papers state this explicitly, and Lyu et al. state it with a log(init-scale) factor. Data fraction moves the run between three regimes (no generalization, grokking, immediate generalization) around a critical data size. Larger init norm or output scale lengthens the delay. Dropout can replace weight decay but makes grokking slower. ℓ1 did not induce grokking in Nanda et al. Gradient-filtering optimizers (Grokfast) and StableMax/⊥Grad shorten or remove the delay.

### Cited Findings
Weight decay and learning rate
- Grokking time ∝ (1−β)/(ηλ) (weight-decay clock), verified on modular addition. [ABS] — [arXiv 2607.23967](https://arxiv.org/abs/2607.23967)
- Grokking timescale ∝ 1/(η·λ) on large (lr, wd) grids; too much λ prevents fitting. [ABS] — [arXiv 2609.26679](https://arxiv.org/abs/2609.26679)
- Omnigrok-style inverted U in weight decay: too little gives no or very late grokking, too much underfits. Reproduced at 12K params. [ABS] — [arXiv 2607.05104](https://arxiv.org/abs/2607.05104); [Omnigrok, arXiv 2210.01117](https://arxiv.org/abs/2210.01117) [KNOWN]
- Phase diagrams over decoder learning rate and weight decay give a bounded "grokking" region next to comprehension, memorization and confusion. [KNOWN] — [arXiv 2205.10343](https://arxiv.org/abs/2205.10343)
- Nanda et al.: with λ=1, P=53 never generalized, while λ=5 grokked. For P=401, λ ∈ {0.3, 0.5, 1, 3, 5, 8} all generalized immediately (no grokking). So the λ needed depends on the modulus and the absolute number of training pairs. [FULL] — [arXiv 2301.05217, App. C.2.2](https://arxiv.org/pdf/2301.05217)
- Nanda et al.: λ=0 gives no grokking, and the excluded loss does not rise. [FULL] — [arXiv 2301.05217, App. D.1, Fig. 26](https://arxiv.org/pdf/2301.05217)
- Low weight decay delays generalization; higher decay reduces the generalization gap and sharpens it. [SNIPPET] — [arXiv 2408.08944](https://arxiv.org/pdf/2408.08944)

Data fraction / critical data size
- Zhu et al. 2024, "Critical data size of language models from a grokking perspective". There is a data-efficiency phase transition at a critical size. By rescaling init, learning rate and weight decay ("grokking configuration") they induce grokking on Yelp and IMDb and show memorization → generalization regimes as data grows. [KNOWN] — [arXiv 2401.10463](https://arxiv.org/abs/2401.10463)
- Varma et al. give a critical dataset size D_crit from circuit efficiency. Below it the run memorizes or ungrokks; near it the run semi-groks. [KNOWN] — [arXiv 2309.02390](https://arxiv.org/abs/2309.02390)
- Nanda et al.: "models trained on smaller fractions of data take longer to grok" at λ=1 (Fig. 20), while train loss falls at the same rate. For P=401, 5% and 10% of pairs never generalized and 20% generalized immediately. [FULL] — [arXiv 2301.05217, App. C.2.2 and D.1](https://arxiv.org/pdf/2301.05217)
- Training-set coverage gates grokking; the threshold tracks output cardinality (the modulus). [ABS] — [arXiv 2607.05104](https://arxiv.org/abs/2607.05104)
- Power et al. 2022: median optimization steps to generalize grow steeply as the training fraction shrinks. [KNOWN] — [arXiv 2201.02177](https://arxiv.org/abs/2201.02177)

Initialization scale
- A large initial weight norm causes or lengthens grokking (Omnigrok). Scaling the init up induces grokking on MNIST. [KNOWN] — [arXiv 2210.01117](https://arxiv.org/abs/2210.01117)
- Large init plus small weight decay gives a sharp kernel → min-norm transition at a time growing with log(init scale)/λ. [KNOWN] — [arXiv 2311.18817](https://arxiv.org/abs/2311.18817)
- Output scale / laziness controls the delay; smaller init gives richer, faster feature learning. [KNOWN] — [arXiv 2310.06110](https://arxiv.org/abs/2310.06110)
- A search summary attributes to Liu et al. and Fan et al. the result that multiplying initial weights by a scale factor can cause or amplify grokking. [SNIPPET, secondary] — [search result pointing to arXiv 2407.12332](https://arxiv.org/pdf/2407.12332)

Optimizer and gradient filtering
- Slingshot: Adam-type optimizers produce norm-growth cycles tied to generalization without explicit regularization. [KNOWN] — [arXiv 2206.04817](https://arxiv.org/abs/2206.04817)
- StableMax / ⊥Grad remove the need for weight decay by preventing Softmax Collapse. [KNOWN] — [arXiv 2501.04697](https://arxiv.org/abs/2501.04697)
- Grokfast (Lee et al. 2024) amplifies slow (low-frequency) gradient components and reports up to about 50× faster grokking on algorithmic tasks. [KNOWN] — [arXiv 2405.20233](https://arxiv.org/abs/2405.20233)
- Decoupled weight decay (AdamW) and coupled L2 are predicted to differ. [ABS] — [arXiv 2607.23967](https://arxiv.org/abs/2607.23967)
- The learning rule changes circuit size, not the algorithm. [ABS] — [arXiv 2611.19477](https://arxiv.org/abs/2611.19477)

Regularizer type
- Nanda et al.: replacing weight decay with dropout (p = 0.2 or 0.5) gave grokking in all three seeds, while ℓ1 regularization gave no grokking. Dropout models "seem to take much longer to grok", their W_E/W_L are not Fourier-sparse (Gini ~0.19–0.26 vs ~0.75 with weight decay), and p = 0.2 models reach only ~99.6% test accuracy. [FULL] — [arXiv 2301.05217, App. C.2.3, D.1, Table 5](https://arxiv.org/pdf/2301.05217)
- Doshi et al. (ICLR 2024): weight decay, dropout and BatchNorm all push networks to drop memorized corrupted labels and keep the generalizing Fourier circuit. [KNOWN] — [arXiv 2310.13061](https://arxiv.org/abs/2310.13061)

Seed and numerical environment
- Seed alone moves grokking time in Nanda et al.: memorization ends at ~1,400 epochs for all 5 seeds, but grokking epochs differ. [FULL] — [arXiv 2301.05217, Fig. 18](https://arxiv.org/pdf/2301.05217)
- CPU thread count and CPU vs GPU flip a minority of same-seed outcomes, and single-seed narratives were overturned by multi-seed control. [ABS] — [arXiv 2607.05104](https://arxiv.org/abs/2607.05104)

### Inferences
- F5 numbers fit 1/λ scaling almost exactly. Our medians give t_grok·λ = 21,500·0.3 = 6,450; 6,800·1 = 6,800; 2,500·3 = 7,500. That is roughly constant, ~6.5–7.5k epochs·λ, i.e. 6.5–7.5 "decay time constants" 1/(ηλ) at η = 1e-3. This is what the 2026 weight-decay-clock and spectral theories predict (2607.23967, 2609.26679), and it matches the direction stated in Nanda's own text and Fig. 27 caption. The printed numbers (3k at λ=0.3, 20k at λ=3) would give t·λ = 900 and 60,000, a 67× spread that no theory predicts. Varying η at fixed λ would test whether the clock is ηλ and not λ alone.
- The 1.2× speed-up from shrinking only W_E is roughly what a pure weight-decay clock would predict. Under AdamW's decoupled decay, w ← (1−ηλ)w per step, so one e-fold of shrinkage takes 1/(ηλ) = 1,000 epochs at η=1e-3, λ=1. Scaling W_E by 0.32 skips about ln(1/0.32) ≈ 1.14 e-folds ≈ 1,140 epochs, and 6,800/(6,800−1,140) ≈ 1.2×. This is a back-of-envelope consistency check (only W_E was scaled; other matrices were not), not a proof. It fits Omnigrok's point that on algorithmic tasks norm alone is not the bottleneck, and Lyu's log(α) dependence, where init scale enters only logarithmically.
- The F5 data-fraction sweep (10–20% none, 30–40% grokking, ≥60% immediate) matches the three-regime picture of Varma (D_crit), Zhu (critical data size), Liu 2022 and Power 2022 qualitatively.

### Gaps
- No paper found that reports a Nanda-setup (1L transformer, P=113, 30%, AdamW) sweep of λ ∈ {0.3, 1, 3} with grokking epochs. Our sweep appears to be the first public numeric check, as far as this search reaches.
- The Grokfast speed-up factor and the Zhu et al. details are [KNOWN] and were not re-fetched.

## Q3. Does the specific train/test split (not only its size) determine which features or frequencies are learned?

### Takeaway
Peer-reviewed work documents that key frequencies differ across runs, and it attributes this to the "seed" or to initialization (lottery ticket). I found no peer-reviewed paper that isolates the split's identity as a cause. The only direct precedent is an unreviewed 2026 GitHub project, co-designed by a Claude model. It reports that init and split each independently predict the early leading frequencies, that the split systematically biases the choice, and that editing the split causally steers which frequency wins. F2 (frequency 7 in 8/16 models on the paper's split, plus the x→7x relabelling check) is therefore new evidence for split-driven bias in the exact Nanda setup.

### Cited Findings
- Nanda et al.: in other seeds, "the specific frequencies differ by seed". The main text and appendix do not say whether the train/test split was re-drawn per seed, so seed and split are confounded. [FULL] — [arXiv 2301.05217, App. C.2.1](https://arxiv.org/pdf/2301.05217)
- Nanda et al. Table 5 lists different key frequencies per training fraction: 40% gives [17, 43, 49, 55], 50% [2, 17, 31, 41, 44], 60% [2, 23, 34, 51], 70% [14, 15, 26], 80% [38, 41], 90% [3, 26, 34, 43]. Frequency 7 appears in only one 2-layer run ([7, 12, 35, 49]), one P=109 run and two dropout runs. The main model's key frequencies (14, 35, 41, 42, 52) differ from all of these. Seed and split are again confounded. [FULL] — [arXiv 2301.05217, Table 5](https://arxiv.org/pdf/2301.05217)
- He et al. 2026: frequency selection is a lottery ticket at initialization. Inside each neuron, frequencies compete and the winner is set by the initial spectral magnitude and phase alignment. This is an init-centred account and does not discuss the split. [ABS] — [arXiv 2602.16849](https://arxiv.org/abs/2602.16849)
- Chughtai, Chan, Nanda (ICML 2023), "A toy model of universality". On group-composition tasks, which irreducible representations are learned varies across runs. "Universality" holds for the algorithm family but not for the specific representations. [KNOWN] — [arXiv 2302.03025](https://arxiv.org/abs/2302.03025)
- Zhong, Liu, Tegmark, Andreas (NeurIPS 2023), "The Clock and the Pizza". Hyperparameters such as attention strength and width select qualitatively different algorithms for modular addition. [KNOWN] — [arXiv 2306.17844](https://arxiv.org/abs/2306.17844)
- Morwani et al. (ICLR 2024), margin maximization. Max-margin solutions for modular addition use Fourier features and are symmetric across frequencies on the full dataset, so the margin objective alone does not pick specific frequencies. [KNOWN] — [arXiv 2311.07568](https://arxiv.org/abs/2311.07568)
- McCracken et al.: 1-layer MLPs find about ⌊n/2⌋ frequencies while 1-layer transformers learn far fewer, and changing hyperparameters gives different circuits. [SNIPPET] — [arXiv 2505.18266](https://arxiv.org/pdf/2505.18266)
- Unreviewed direct precedent, GitHub "grokking-frequency-selection" (Seungho Yu with Claude, © 2026). Setup: 1L transformer, 30% split, 32×32 init×split factorial. Results:
  - Step-500 top-3 frequency hits: init only 655/3072, split only 959/3072, combined 1,265/3072, chance 165.
  - A per-frequency split statistic D(k) has within-split Spearman ρ = 0.709.
  - Removing the top 10% of pairs for a split's favourite frequency shifts that frequency's rank (z = −0.7 to −1.0 vs +1.2 to +1.3 for random removal; Holm p = 0.0002).
  - Steered splits make the target frequency lead in 96/96 runs (4% on original splits), with final circuits containing it in 16/16 vs 1/16 (Fisher p = 3·10⁻⁸).
  - Additive relabelling a → a+t check: agreement 0.955.
  - Negative results: per-pair datamodels have R² ≈ 0.03, and the step-500 proxy recovers only 56.5% of final key frequencies.
  - The README says prior work treated the split effect as zero-mean noise, without naming papers, and the work "has not been peer reviewed".
  [ABS/README] — [GitHub yuseunghogi-commits/grokking-frequency-selection](https://github.com/yuseunghogi-commits/grokking-frequency-selection)
- The nnsight reproduction (λ=1, 30%, init seed 999, split seed 598) finds key frequencies 19, 23, 36, 45, again different from the paper's main model. [FULL] — [nnsight tutorial](https://nnsight.net/tutorials/mini-papers/grokking-progress-measures/)
- The learning rule changes circuit size but always gives the Fourier algorithm. The abstract does not say whether specific frequencies are preserved. [ABS] — [arXiv 2611.19477](https://arxiv.org/abs/2611.19477)

### Inferences
- Theory explains why only the split or the init can pick frequencies. On the full dataset, the task is invariant under multiplicative relabelling a → ca, b → cb (c ≠ 0 mod p), which permutes the non-zero frequencies. Max-margin solutions are also frequency-symmetric (Morwani). So any consistent preference for one frequency across inits must come from symmetry breaking by the finite training set. F2's x → 7x relabelling check (data and init transformed together reproduce the main model under new names) is the right control for this. It rules out code-level or architectural asymmetries and points to the split.
- The GitHub precedent uses an additive relabelling a → a+t. That shift preserves each frequency's identity and changes only phases, so it tests something different from F2's multiplicative check, which permutes frequencies. F2's multiplicative check is the stronger test of "the split picks the frequency label".
- The GitHub project reports that "simple split statistics" (per-pair datamodels) fail (R² ≈ 0.03), while a per-frequency statistic D(k) works (ρ ≈ 0.71). This matches F2's finding that simple split statistics did not explain frequency 7, and suggests trying a per-frequency, Fourier-domain statistic of the split's coverage.
- Independence caveat: the precedent was designed and run with a Claude model (the same family as this research agent) and is not peer-reviewed. Its claims should be cited as preliminary and should not count as independent confirmation.

### Gaps
- No peer-reviewed paper found that holds init fixed and varies only the split (or the reverse) in the Nanda transformer setup.
- Whether Nanda et al.'s 5 seeds shared one split was not determined from the paper text. The repo code was not inspected in this session.
- The README does not show D(k)'s exact definition; it would need a read of the repo code.

## Q4. Has anyone flagged errors in Nanda et al. 2023 App. D.1, or reproduced its weight-decay sweep with numbers?

### Takeaway
App. D.1 is internally inconsistent. Its sentence says that smaller weight decay gives slower grokking, but the numbers it prints (≈3k epochs at λ=0.3, 5–10k at λ=1, 20k at λ=3) say the opposite. The main text and the Fig. 27 caption agree with the sentence, not the numbers. I found no public erratum, GitHub issue, LessWrong post or follow-up paper that flags this, and no public reproduction of the λ ∈ {0.3, 1, 3} sweep with grokking epochs. Our sweep (λ=0: none in 40k; medians 21,500 / 6,800 / 2,500 at λ = 0.3 / 1 / 3) matches the prose and the theory, which is consistent with the two outer numbers being swapped.

### Cited Findings
- Verbatim from App. D.1 (arXiv v3): "Smaller amounts of weight decay lead to slower grokking, while larger amounts of weight decay lead to faster grokking—on average, it takes around 3k epochs for models to grok with weight decay λ = 0.3, 5-10k epochs for the models to grok with weight decay λ = 1.0, and 20k epochs for the models to grok with weight decay λ = 3.0." [FULL] — [arXiv 2301.05217v3, App. D.1](https://arxiv.org/pdf/2301.05217v3)
- The Fig. 27 caption says: "Less aggressive weight decay leads to slower grokking." [FULL] — [arXiv 2301.05217, Fig. 27](https://arxiv.org/pdf/2301.05217)
- The main text (Sec. 3, λ=1) says the network generalizes "after around 10,000 epochs", while D.1 says "5-10k" for λ=1. These are consistent with each other and with our λ=1 median of 6,800. [FULL] — [arXiv 2301.05217, Sec. 3](https://arxiv.org/pdf/2301.05217)
- The main text says D.1 provides "additional evidence that weight decay is necessary for grokking: smaller amounts of weight decay causes the network" to grok more slowly or not at all (line truncated in extraction). [FULL] — [arXiv 2301.05217, Sec. 5.3](https://arxiv.org/pdf/2301.05217)
- The official repo has 3 issues in total: #1 file paths, #2 the float32 high-precision loss bug, #3 a missing file. None concerns D.1 or weight-decay timing. [FULL via gh CLI] — [GitHub issues](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/issues)
- The nnsight reproduction at λ=1 expects that "grokking lands around epoch 20–25k". This is stated as an expectation, not a measured number. It sits above both the paper's ~10k and our 6,800 median, so its setup differences (35k epochs, specific seeds) or the seed variance matter. [FULL] — [nnsight tutorial](https://nnsight.net/tutorials/mini-papers/grokking-progress-measures/)
- One search engine's summary misquoted D.1 as "3k epochs … λ = 0". This shows the passage is easy to misread, but the real v3 text says λ = 0.3. [SNIPPET, flagged as erroneous] — [search result for arXiv 2301.05217](https://arxiv.org/pdf/2301.05217)
- Independent theory predicts the direction of the prose and of our sweep: time ∝ 1/(ηλ). [ABS] — [arXiv 2607.23967](https://arxiv.org/abs/2607.23967); [arXiv 2609.26679](https://arxiv.org/abs/2609.26679)

### Inferences
- The likeliest explanation is a transcription swap of the λ=0.3 and λ=3 numbers. With the swap, t·λ ≈ 20k·0.3 = 6k and 3k·3 = 9k, which brackets our 6.5–7.5k and matches the 1/λ clock. Without the swap, the numbers contradict the sentence they sit in, the figure caption, and all ηλ-clock theories.
- The 2-layer, P=53 and P=401 statements in C.2.2 are internally consistent and match the "critical data size" picture, so the D.1 inconsistency looks like a local error, not a systematic one.

### Gaps
- The OpenReview forum and reviews for Nanda et al. (ICLR 2023) were not reached. Search returned only the ICLR virtual page, so whether any reviewer raised D.1 is unknown.
- LessWrong / Alignment Forum posts (including the original 2022 Nanda and Lieberum post) were not searched in this session's budget.
- It was not checked whether arXiv v1 or v2 had different D.1 numbers (only v3 text was extracted).
