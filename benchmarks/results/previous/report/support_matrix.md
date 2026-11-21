## Support matrix

| Method | MNIST · stationary | MNIST · label noise | MNIST · gradient noise | MNIST · bad initial LR | CIFAR-10 · stationary | CIFAR-10 · label noise | CIFAR-10 · bad initial LR | Training cost per cell |
|---|---|---|---|---|---|---|---|---|
| Fixed LR *(reference)* | 96.33 ± 0.56 #3 | 94.67 ± 1.21 #3 | 95.57 ± 1.00 #3 | 10.00 ± 0.38 #3 | **75.38 ± 0.38 #1** | 70.75 ± 1.21 #3 | **10.00 ± 0.00 #1** | 10 epochs |
| AutoTuneNet | 98.05 ± 0.61 (+1.72) #2 | 96.24 ± 1.00 (+1.58) #2 | 97.26 ± 0.71 (+1.69) #2 | **11.35 ± 0.00 (+1.35) #1** | 72.86 ± 2.04 (-2.52) #3 | 70.78 ± 1.87 (+0.04) #2 | **10.00 ± 0.00 (+0.00) #1** | 10 epochs |
| Cosine schedule | **98.15 ± 0.53 (+1.82) #1** | **97.11 ± 0.70 (+2.45) #1** | **97.58 ± 0.53 (+2.01) #1** | **11.35 ± 0.00 (+1.35) #1** | 73.72 ± 0.85 (-1.66) #2 | **72.08 ± 1.02 (+1.34) #1** | **10.00 ± 0.00 (+0.00) #1** | 10 epochs |
| ReduceLROnPlateau | pending | pending | pending | pending | pending | pending | pending | — |
| Optuna TPE | pending | pending | pending | n/a | pending | pending | n/a | — |
| ASHA | pending | pending | pending | n/a | pending | pending | n/a | — |
| Hyperband | pending | pending | pending | n/a | pending | pending | n/a | — |
| PBT | pending | pending | pending | pending | pending | pending | pending | — |

Test accuracy (%) on the held-out split, mean ± std over seeds; (±x) = difference from Fixed LR in percentage points; #rank within the column; **bold** = best; pending = not run yet; n/a = the scenario does not apply.
