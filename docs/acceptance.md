# Delivery acceptance

The requested engineering outcome is implemented and verified. This is a
CPU-scale research project with an honest negative comparison result, not an
unearned claim of a groundbreaking decoder.

| Requested outcome | Evidence and practical limit |
| --- | --- |
| Real circuit-level surface-code simulation at d3/d5/d7 | `src/qubitguard/simulation.py`; retained `.stim`/`.dem` files; independent lossless-grid, replay and graph tests in both X/Z memory bases. Neural training is X memory, rounds=d. |
| Genuinely trained original PyTorch model | 13 primary checkpoints plus one full-map revision; fit histories/losses, parameter counts and checksums. New random initialization; no external pretrained weights. |
| End-to-end corrections and logical predictions | Inference consumes detector history, declared noise and matching context. API correction parity/logical bits are reconstructed independently in tests. An odd logical graph cycle implements the learned frame toggle. This is a graphlike recovery, not a uniquely identified physical fault chain. |
| Fair strong classical comparisons | 266 primary rows from identical 840,000 test shots; both oracle-noise standard and correlated PyMatching, correlated flags enabled at construction and inference. Independent uncorrelated graph test checks shared-state behavior. |
| Independent samples and unseen noise | Disjoint train/validation/test seed namespaces; unseen p interpolation/extrapolation and measurement-heavy profile; follow-up uses a fourth namespace for 280,000 new d3 test shots. No hardware/leakage transfer claim. |
| Statistical rigor | Exact failure counts, Wilson intervals, paired win/loss counts, conservative paired difference intervals, exact discordance tests and family-wide Bonferroni p values. All 322 rows and 406 comparisons reconcile with raw NPZ. Zero counts retain upper-bound uncertainty. |
| Accuracy, latency, size and efficiency | Benchmark JSON/CSV, all model summaries, learning curves, full CPU pipeline latency including matching/features, batch256 throughput and observed batch1 median/p95. OS workload is uncontrolled; no hard-real-time or causal training-speed claim. |
| Ablations and iteration | d3 direct, neural-time, correlated-context and equal-budget single-p controls; three primary initialization seeds at each distance. Full-map readout revision used the same data/steps and a new test stream, failed to improve, and is retained transparently. |
| Reusable data, checkpoints and configurations | 75 NPZ datasets with exact source circuits/DEMs, metadata and hashes; versioned configs; 14 checkpoints with licenses and loaders; raw predictions; exact dependency lock. Cache/partial-artifact guards reject incompatible reuse. |
| Reproduction | Recorded full primary dataset replay and a complete d3 retraining with identical tensors/threshold/epoch/checkpoint bytes during development. Source-cache/storage safeguards were added transparently. Independent review sampled inference from all14 models; it did not independently rerun every training fit. |
| Automated tests | `artifacts/test-results.txt`: 35 passed, one dependency deprecation warning. `artifacts/verification.json` reconciles current source/configuration, data, checkpoints and numerical evidence. This records the original scientific verification; public commit CI is linked from README. |
| Real interactive playground | Loopback server, dependency-free client, real Stim DEM shots, representative faults, lattice/space-time modes, time stepping/playback, distance/noise/profile/basis/seed controls, model/classical recoveries and separate syndrome/logical outcomes. `ui-verification.json` records desktop/mobile and deliberate failure checks; two current-source screenshots are retained. |
| Report, plots and professional documentation | Eight-page rendered and visually inspected technical PDF, editable report text, standalone SVG/PNG/PDF scientific figures, README, model/data cards, BibTeX, contribution guide, citation file, source/weight notices and GitHub CI workflow. |
| Foundations and licensing | Primary-source research notes and citations cover Stim, PyMatching, NVIDIA/AlphaQubit and close precedents. NVIDIA source/weights terms are distinct; gated weights not downloaded. Published NVIDIA GPU performance is contextual, not a falsely reproduced CPU benchmark. |
| Scientific honesty | No primary neural gain over correlated matching survives the declared adjustment; two d5 high-noise settings significantly favor correlated matching. No threshold, hardware, real-time or decoder-family novelty claim. Unresolved directions and tested failures are documented. |

The public repository and versioned release preserve this scientific artifact.
The interactive application runs locally using `scripts/serve.sh` or the
documented console command. Publication changes are listed in CHANGELOG.md.
