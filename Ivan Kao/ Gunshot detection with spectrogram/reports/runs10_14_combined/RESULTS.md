# Runs 10–14 combined held-out test

All five test sets evaluated using the final Run14 model. No additional training or threshold changes. The separate 800-recording external benchmark is excluded.

| Metric | Result |
|---|---:|
| Test Recordings | 806 |
| Accuracy | 97.77% |
| Gunshot recall | 94.44% |
| Gunshot precision | 89.47% |
| Gunshot F1 | 91.89% |
| ROC AUC | 0.9960 |
| Decision threshold | 0.21 |
| Gunshots detected | 102 |
| Gunshots missed | 6 |
| False alarms | 12 |
| Correct non-gunshots | 686 |

Counts are audio test clips/excerpts, including 500 AudioMNIST, 54 office, 60 DEMAND, 120 forest, and 72 M16/firecracker clips. All 806 are excluded from final checkpoint development hashes.
