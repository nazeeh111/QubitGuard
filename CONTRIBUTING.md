# Contributing

Keep simulator truth out of inference inputs. A new decoder must consume the
same held-out detector shots as both matching baselines. Keep exact circuits,
model/source versions, seeds and failure counts with results. Thresholds,
architecture and stopping choices belong to validation, never test selection.

Run `pytest -q`, `qubitguard verify`, and targeted physical/statistical
invariants for behavior you change. A changed algorithm or training source
invalidates old checkpoint compatibility by design. Use a fresh experiment
root, preserve the previous run, and report new data budgets and full inference
costs. Do not silently update a published benchmark with an incomparable run.

Use new independent test seed policy/version for a new confirmatory study.
Repeated analysis of the current saved test sweep is exploratory. Add ablations
that can distinguish competing mechanisms, not only favorable examples.
Confidence intervals and negative results are part of the artifact.

Corrections in this project are graphlike DEM representatives. Do not relabel
them as unique physical fault trajectories. Preserve the model card's scope,
source/weight license separation and honest unsupported states in the demo.

The project uses no paid API or external runtime service. Keep the playground
loopback-only unless a separately reviewed hosting/security design is supplied.
