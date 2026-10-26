# Independent scientific correctness review

Review date: 25 October 2026. This review examined the source and scientific contracts independently of model-performance claims. It is an engineering review, not peer review or a reproduction of AlphaQubit or NVIDIA results.

## Contracts examined

- Stim-generated rotated surface-code memories, with rounds equal to distance and an explicit X-memory training scope.
- Independent train, validation and test stream namespaces. Three initialization seeds share their training data deliberately, isolating initialization variation.
- Standard and two-pass correlated PyMatching on the same test syndromes, using the test circuit's true noise model.
- Truth-free neural inference. Observable truth supplies training targets and scoring only. Context uses syndrome density, declared noise and matching predictions.
- Graph recovery parity and logical-frame toggles. A closed odd-logical graph cycle must change the logical prediction without changing any detector parity.
- DEM-sampled playground mechanisms and explanatory circuit locations. The physical locations are representatives, not uniquely observed microscopic faults.
- Validation-only checkpoint/threshold selection, exact paired tests, confidence intervals, ablation interpretation and complete inference costs.

## Findings and correction status

The earlier review identified a sparse-count problem in the paired normal confidence interval: zero discordances yielded a zero-width interval. The current source uses a conservative exact interval formed from simultaneous Clopper-Pearson bounds on paired win/loss probabilities. This retains uncertainty when no discordant failures occur. The exact paired binomial test is also reported; sweep-wide multiplicity adjustment must remain separate from marginal confidence bars.

The reviewed follow-up source resolves the reported reuse gaps. Experiment signatures include effective configuration, package versions and hashes of simulation, data, model, training and provenance source. Cached checkpoint reuse checks the checkpoint and every referenced training/validation dataset hash. Dataset reuse checks generator source, setting, split, seed, shot count, package versions, file hash and circuit hash. Evaluation checks the expected model roster, configuration/source signature, checkpoint checksum and bundle distance/variant. An incomplete checkpoint/summary pair now fails explicitly. Mutation tests exercise these guards in isolated temporary fixtures. The original long-running job loaded the earlier source before these guards landed; release provenance augmentation therefore requires the builder's separate exact dataset replay and fresh-run tensor comparison, rather than simply stamping current hashes onto old results. That migration evidence is outside this source review and must be retained with the release.

The no-time ablation removes temporal order from the neural branch. It retains classical matching context, which already reflects the full syndrome history. It must therefore be described as a neural temporal-feature ablation, not as a wholly time-blind decoder.

The measurement-heavy profile changes measurement/reset and idle-data channel rates. It is a channel-imbalance test, not a specifically Pauli-Z-biased, leakage or burst-error experiment. Both matching baselines receive the true shifted model, so this is a strong matched-noise baseline comparison.

Batch throughput and sequential single-shot latency describe different quantities. Neural inference includes matching dependencies, features and grid conversion. The follow-up evaluation source now collects sequential batch-one median and p95 timings for both matching baselines as well as the neural pipelines, resolving the comparison gap. The playground's individual request time is a demonstration diagnostic, not a benchmark.

## Automated independent checks

`tests/test_review.py` covers both memory bases and distances 3, 5 and 7, graph parity, logical-cycle toggles, lossless detector grids, seeded replay, split namespaces, tamper rejection, ablation isolation and statistical extremes.

`tests/test_scientific_contracts.py` adds:

- Reconstruction of each API shot's detector syndrome and true logical flip from its recorded sampled DEM mechanisms.
- Reconstruction of displayed correction-edge parities and predicted logical flips without trusting the returned certificate flag.
- Repeat-seed equality of scientific API outputs and rejection of invalid controls.
- Explicitly unavailable neural predictions outside the X-memory training scope.
- Invariance of trained inference to altered hidden observable truth.
- Nonzero paired uncertainty when no decoder disagreements occur.
- Rejection of unsupported experiment knobs.
- Comparison of the standard baseline against a separately constructed uncorrelated matching graph after correlated decoding has exercised shared state.
- Preservation and rejection of incomplete checkpoint pairs, changed cached training data, mutated dataset metadata, missing model rosters and edits to every relevant source component.

The release's recorded test output is the authority for execution status. These tests cannot establish an accuracy advantage, quantum-hardware validity, threshold scaling, calibrated probabilities, unique physical fault identification or hard-real-time timing. Those claims require separate evidence and are outside this CPU-scale study.

## Execution record

The initial added API/science subset passed 10 tests in 75.74 seconds. The expanded complete suite then passed 22 tests and failed one endpoint assertion in 36.70 seconds. The failure was a numerical Wilson endpoint (`wilson(0, 100)` returned approximately `3.47e-18` for the lower bound, rather than the exact analytic zero); it is not evidence of failed graph recovery or truth leakage. The builder then clamped the known binomial endpoints explicitly; the final rerun below covers that correction. The API/science tests also emitted one dependency deprecation warning about Starlette's current `httpx` TestClient integration; requests completed successfully. Final release verification must record the rerun after corrections.

Final follow-up rerun: **32 passed**, one dependency deprecation warning, in **14.29 seconds**. Output: `work/science-review-final-tests.log` in the task workspace. Relevant simulation/data/model/training/provenance source fingerprint at review: `560fff882b7c72d12ff71b6d42c52f48e98be8a3e90f6985bf124401bafd71dc`. Source inspection confirmed the cache, evaluation-roster, endpoint and baseline batch-one fixes described above. No model ranking or performance claim follows from this test result.
