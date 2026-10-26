# Independent release evidence audit

25 October 2026. Scope: current saved results, report, README, model/data cards,
configuration, numerical source and selected runtime inference. This review did
not rerun the entire training experiment or independently reproduce external
papers. It found no numerical result that supports neural superiority over
correlated matching.

## Outcome

The saved artifacts support a legitimate trained CPU-scale decoding study with
an honest negative result. The core numerical claims pass independent
reconciliation. Both requested release corrections are closed: the final
execution record is retained and consistent with the audited artifacts, and
the report uses a supported declaration of the primary seed. Accepted within
the explicit scope and evidence limits below.

## Independently checked evidence

- All **266 primary benchmark rows** reconcile exactly with retained raw
  predictions and truth. All **364 paired comparisons** have the correct
  discordant wins/losses, exact two-sided binomial p value and global Bonferroni
  adjustment. The primary seed17 comparison against correlated matching has
  zero significant wins and two significant losses: d5 uniform p=0.015
  (43 wins, 130 losses; adjusted p=8.4962e-9), and d5 measurement-heavy p=0.015
  (173 wins, 373 losses; adjusted p=2.5645e-15).
- All **56 follow-up result rows** and **42 follow-up paired comparisons**
  reconcile with their separate raw predictions. Full-map mixture failures are
  6,525 uniform and 7,431 shifted, versus compact 6,503 and 7,379, and correlated
  matching 6,495 and 7,412. The report describes the unsuccessful revision
  accurately.
- The report's p=0.007 table, six primary profile totals, d3 ablation totals,
  selected epochs, parameter counts, fit-time ranges, latency values and d7
  seed29 failure example agree with their saved records. Primary storage has
  **61 NPZ files and 1,152,000 shots**. Follow-up storage has 14 test files and
  280,000 shots. All **75 dataset checksums** match. The primary raw-prediction
  checksum and follow-up checkpoint checksum also match their manifests.
- Every primary checkpoint was loaded independently. For each of the 13
  checkpoints, scores and predictions on 256 retained high-noise shifted shots
  match the saved outputs exactly. Parameter counts match the summaries.
  Flipping hidden truth leaves every checked prediction unchanged. A separate
  256-shot check of the full-map checkpoint likewise reproduces saved scores
  and predictions exactly. These are sampled runtime checks, not full-sweep
  recomputation.
- Source inspection confirms validation-only threshold/epoch selection,
  independent train/validation/test seed namespaces, identical test syndromes
  across methods, exact test-noise DEMs for both baselines, enabled correlated
  construction/inference, and matching/feature conversion included in measured
  neural inference costs. The single-p training seed reuse is disclosed; no
  train/test seed collision was found. A seed namespace is reproducibility
  evidence, not a mathematical proof of statistical independence.
- The paired interval implementation uses simultaneous Clopper-Pearson bounds
  for win/loss probabilities, giving a conservative marginal 95% difference
  interval with uncertainty at zero discordances. Documentation correctly
  separates these marginal bars from globally adjusted p values.
- Visual inspection of the uniform-rate and paired-follow-up PNG figures found
  readable labels and honest uncertainty/scope descriptions. The zero-failure
  marker is identified as an upper bound, and the follow-up difference plot
  identifies which sign favors the new model. Plot source reads saved results.

## Scope and claim boundaries

The neural branch predicts a logical-frame residual after matching. It does not
remove local physical errors before matching. A detector-parity certificate
checks the graph recovery class and cannot by itself establish logical success
or identify a unique physical fault trajectory. The docs preserve these
distinctions. X-memory neural support, rounds=distance, channel-imbalance noise,
single-initialization ablations and the adaptive nature of the full-map trial
are disclosed. Neither threshold scaling nor real-time/hardware performance is
claimed. Fit times and 32-shot latency tails are correctly limited.

The [official NVIDIA model card](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast)
currently identifies NVIDIA Open Model License terms and gated model-file
access. The [source license](https://raw.githubusercontent.com/NVIDIA/Ising-Decoding/main/LICENSE)
is Apache-2.0. These support the documentation's separation of source and model
terms. The [versioned pre-decoder paper](https://arxiv.org/abs/2604.12841v1)
supports the stated GPU/context distinction. No external weights appear in the
reviewed checkpoint roster. This is an artifact inspection, not an exhaustive
forensic proof that no external artifact was ever accessed or that no source
was copied. A direct Nature fetch was unavailable, so detailed AlphaQubit
methodology was not independently rechecked in this audit.

## Release corrections and limits

1. **Closed:** `artifacts/verification.json` and `artifacts/test-results.txt`
   now exist. The execution log records 35 passed tests, one dependency
   deprecation warning and no failures. The verification counts agree with
   this audit's 322 result rows, 406 paired comparisons, 75 datasets and 14
   checkpoints. Its numerical artifact hashes and current source fingerprint
   independently match. This reviewer inspected the retained test output; it
   is distinct from independently rerunning the suite.
2. **Closed:** the technical report Markdown and its generator now say,
   "Seed17 is the declared primary model; all three initialization seeds are
   retained." This removes the unsupported pre-test chronology assertion and
   does not change any error count. PDF regeneration/visual acceptance belongs
   to the final report delivery check.
3. `deterministic-reproduction.json` reports full d3 tensor/threshold/epoch and
   checkpoint-byte equality. This review checked its consistency with the
   current original checkpoint, but did not rerun training or inspect a retained
   second complete run. Treat the retraining as recorded builder evidence,
   distinct from this independent numerical reconciliation.

Initial sampled runtime checks used `PYTHONPATH=src`. On final follow-up the
package is installed as an editable distribution in the supplied task runtime.
This reviewer independently imported it without `PYTHONPATH` and ran its
installed `qubitguard verify` console command successfully, with zero checksum
failures. The installation and checksum-check follow-up closes this environment
note; it does not extend the sampled inference or retraining scope.
No published deployment, remote CI run, cross-platform bitwise replay or
scientific novelty beyond the documented engineering contribution was verified.

## Audited numerical artifact identities

SHA256 at review:

```text
artifacts/benchmark.json
f1fda24f16200ca8c595800747c8f4cc39558d815d09e847861062b0e00717eb
artifacts/heldout-predictions.npz
dfd014f5cc5590ade28e82358462a507c6b62775f0d9c83abca718680cc9607f
experiments/fullmap-v2/artifacts/followup.json
2661822a6ff63681323beffb0d8bce15b60c523e7863061f725e6be96e4db93f
experiments/fullmap-v2/artifacts/heldout-predictions.npz
3f6e3373ffdb2d65817498d97a5449a14f06e5ad8245e51bdf066bdb32bf47a2
```
