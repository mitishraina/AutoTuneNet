## Support matrix

| Method | MNIST · stationary | MNIST · label noise | MNIST · gradient noise | MNIST · bad initial LR | CIFAR-10 · stationary | CIFAR-10 · label noise | CIFAR-10 · bad initial LR | Training cost per cell |
|---|---|---|---|---|---|---|---|---|
| Fixed LR *(reference)* | 96.52 ± 0.83 #7 | 94.98 ± 0.68 #7 | 95.82 ± 0.72 #7 | 10.00 ± 0.38 #4 | 74.24 ± 0.56 #5 | 70.33 ± 0.94 #6 | 10.00 ± 0.00 #2 | 10 epochs |
| AutoTuneNet | 98.05 ± 0.49 (+1.53) #5 | 96.11 ± 0.36 (+1.13) #6 | 97.20 ± 0.75 (+1.38) #5 | **11.35 ± 0.00 (+1.35) #1** | 72.45 ± 2.20 (-1.79) #7 | 69.95 ± 3.74 (-0.37) #7 | 10.00 ± 0.01 (+0.00) #2 | 10 epochs |
| Cosine schedule | 98.07 ± 0.48 (+1.55) #4 | 96.92 ± 0.75 (+1.93) #4 | 97.71 ± 0.55 (+1.89) #4 | **11.35 ± 0.00 (+1.35) #1** | 73.27 ± 0.64 (-0.97) #6 | 71.70 ± 0.94 (+1.38) #3 | 9.99 ± 0.01 (-0.01) #4 | 10 epochs |
| ReduceLROnPlateau | 97.61 ± 0.43 (+1.09) #6 | 96.44 ± 1.04 (+1.46) #5 | 97.05 ± 0.51 (+1.23) #6 | 10.76 ± 1.02 (+0.76) #3 | 74.60 ± 0.79 (+0.36) #4 | **72.23 ± 1.66 (+1.91) #1** | **13.25 ± 5.63 (+3.25) #1** | 10 epochs |
| Optuna TPE | 98.94 ± 0.02 (+2.42) #2 | 98.29 ± 0.10 (+3.31) #3 | 98.94 ± 0.05 (+3.12) #3 | n/a | 75.31 ± 0.79 (+1.07) #3 | 70.35 ± 0.81 (+0.03) #5 | n/a | 80 epochs · 7 restarts |
| ASHA | **99.03 ± 0.12 (+2.51) #1** | 98.31 ± 0.41 (+3.32) #2 | 98.96 ± 0.04 (+3.14) #2 | n/a | 75.45 ± 0.41 (+1.21) #2 | 71.22 ± 1.02 (+0.89) #4 | n/a | 67 epochs · 15 restarts |
| Hyperband | 98.92 ± 0.01 (+2.40) #3 | **98.58 ± 0.15 (+3.60) #1** | **99.08 ± 0.12 (+3.26) #1** | n/a | **75.54 ± 0.58 (+1.30) #1** | 71.86 ± 1.06 (+1.53) #2 | n/a | 106 epochs · 15 restarts |
| PBT | — | — | — | — | — | — | — | — |

Test accuracy (%) on the held-out split, mean ± std over seeds; (±x) = difference from Fixed LR in percentage points; #rank within the column; **bold** = best; — = not run yet; n/a = the scenario does not apply.
