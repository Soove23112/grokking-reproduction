| what | paper | authors' weights | our model | authors match | ours match |
|---|---|---|---|---|---|
| loss over all pairs, final checkpoint | 2.41e-7 | 2.41e-07 | 1.48e-07 | yes |  |
| 100% train accuracy, epoch | memorised by ~1 400 | — | 200 |  | yes |
| >99% test accuracy, epoch | ~10 000 | — | 6800 |  | yes |
| key frequencies (rule: >=10 neurons with FVE > 85%) | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 4, 13, 25, 35 | yes | yes |
| frequencies with a large norm in W_E (>10 medians) | 6 | 6 | 5 | yes | yes |
| neuron 0: frequency | 42 | 42 | 13 | yes |  |
| head 0: attention frequency | 35 | 35 | 35 | yes |  |
| neurons with single-frequency FVE > 85% | 433 / 512 | 433 / 512 | 439 / 512 | yes | yes |
| neurons of frequency 14 | 44 | 44 | — | yes |  |
| W_L outside the cos/sin directions of key frequencies (share of squared norm) | < 0.55% | 0.55% | 0.21% | yes | yes |
| large 2D Fourier components of the logits (>30% of max) | 20 (4 per frequency) | 20 | 16 | yes | yes |
| table 1: FVE of u_k, v_k projections by one cos/sin(w(a+b)) term, minimum | 93.2% | 93.2% | 96.8% | yes | yes |
| logits ~ sum alpha_k cos(w_k(a+b-c)): fraction of variance | 95% | 96.1% | 97.9% | NO | yes |
|   ... loss of this approximation | 4.7e-8 | 4.4e-08 | 4.6e-08 |  |  |
| ablation: removing any key frequency from the logits raises the loss | yes | min 1.1e-03 | min 8.8e-03 | yes | yes |
| ablation: removing any non-key frequency barely changes the loss | yes | max 2.5e-07 | max 1.5e-07 | yes | yes |
| logits: keep only the 40 key-frequency components (4.4) | −70% (7.24e-8) | -70% (7.24e-08) | -42% (8.62e-08) | yes | yes |
| restricted loss (5.1): constant + cos/sin(w_k(a+b)) | — (fig. 7 only) | -77% (5.55e-08) | -63% (5.49e-08) |  |  |
| neurons with FVE > 85% replaced by their frequency polynomial | +3% (2.48e-7) | -37% | -2% |  |  |
| all neurons: keep only cos/sin(w_k(a+b)) of key frequencies | −77% (5.54e-8) | -67% | -40% |  |  |
| MLP activations projected onto the 10 W_L directions | −50% (1.19e-7) | -0% | +5% |  |  |
| ... onto their null space: loss | 5.27 (worse than chance, ln 113 = 4.73) | 3.98 | 4.17 | NO | NO |
| Gini W_E / W_L at the end | ~0.8 on fig. 7; 0.62–0.80 / 0.73–0.91 in table 5 | 0.74 / 0.80 | 0.78 / 0.85 |  |  |
| phases t1 / t2 / t3, epochs | 1 400 / 9 400 / 14 000 | 1400 / 9400 / 13800 | 800 / 5600 / 9400 | yes | yes |
