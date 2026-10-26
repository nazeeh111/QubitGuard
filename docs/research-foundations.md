# Research foundations and comparison contract

Evidence checked 25 October 2026. This document separates source-reported results from proposed QubitGuard experiments. Reading a paper or repository is not a local reproduction. Upstream `main` links below are mutable research pointers; QubitGuard's environment lock, circuit bytes, dataset hashes, seed manifests and checkpoints are the authoritative versions for its own experiments.

## Established methods and what transfers

**AlphaQubit (Nature, 2024).** Bausch et al. use recurrent transformer processing of surface-code syndrome histories, including readout and leakage information where available. Their evidence includes Sycamore experiments at distances 3 and 5 and simulated scaling to distance 11. The methods describe 5.4 million model parameters and training budgets reaching billions of examples. The direct transfer is the importance of temporal syndrome structure and logical prediction, not a claim that a small CPU model reproduces AlphaQubit. A QubitGuard experiment with Pauli noise has no leakage, soft-readout or real-device validation. [Paper and methods](https://www.nature.com/articles/s41586-024-08148-8), DOI 10.1038/s41586-024-08148-8.

**NVIDIA pre-decoding (2026).** Chamberland et al., arXiv:2604.12841v1, describe local spatiotemporal convolutional corrections followed by global decoding. Labels are simplified within syndrome/logical equivalence classes. The paper reports gains over both uncorrelated and correlated matching in specified regimes; its fast small model can underperform matching at lower noise. Reported speedups use NVIDIA GB300 and large space-time volumes, with transfer overhead treated under their NVQLink assumptions. These are conditional source results, not CPU expectations or QubitGuard measurements. [Versioned paper](https://arxiv.org/html/2604.12841v1).

**NVIDIA source and default noise.** The repository separates pre-decoding from downstream PyMatching and exposes a 25-parameter Pauli-noise family, including preparation, measurement, idle and two-qubit gate faults. Its public default is distance 7 and seven rounds, with noise parameters corresponding to nominal p=0.003. This differs from merely enabling one Stim depolarization argument. A comparison must align exact circuits and every noise parameter before treating rates as comparable. The source is Apache-2.0; preserve relevant notices if code is incorporated. QubitGuard should independently implement its own model rather than relabel upstream code. [Repository](https://github.com/NVIDIA/Ising-Decoding), [configuration](https://raw.githubusercontent.com/NVIDIA/Ising-Decoding/main/conf/config_public.yaml).

**Weights are separately licensed.** The NVIDIA Fast model card identifies the NVIDIA Open Model License, approximately 910,000 parameters, an Ampere minimum GPU requirement, and access conditions requiring agreement and information sharing. Its stated scope is one synthetic noise distribution, with retraining advised for changed noise or layouts. Source-code licensing is not permission to redistribute these weights. QubitGuard uses newly initialized and locally trained weights; it neither accepts these conditions on the user's behalf nor downloads or redistributes NVIDIA weights. [Official model card](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast).

**Compute mismatch matters.** NVIDIA's training guide recommends at least 100 epochs for several models and, for an eight-GPU setup, 67 million shots per epoch. The public workflow offers a Stim/PyTorch path, but published performance still belongs to its stated hardware and experiment. There is no basis to extrapolate its convergence or timing to a laptop-sized experiment. [Official training guide](https://raw.githubusercontent.com/NVIDIA/Ising-Decoding/main/TRAINING.md).

## Strong baseline details

PyMatching's current implementation supports two-pass correlated matching for decomposable hyperedges. Both construction and inference must enable correlations:

```python
dem = circuit.detector_error_model(decompose_errors=True)
standard = pymatching.Matching.from_detector_error_model(dem)
correlated = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
p0 = standard.decode_batch(syndromes)
p1 = correlated.decode_batch(syndromes, enable_correlations=True)
```

The `^` separators in the decomposed DEM carry linked components; flattening those into independent errors would damage the correlated baseline. `decode_to_edges_array(..., enable_correlations=True)` can expose actual selected graph edges. Such edges describe a recovery in the detector graph, not a unique microscopic error history. Use the installed version's signatures, since older stable documentation omits newer features. [Official implementation](https://raw.githubusercontent.com/oscarhiggott/PyMatching/master/src/pymatching/matching.py), [baseline tutorial](https://pymatching.readthedocs.io/en/stable/).

Stim's DEM decomposition represents correlated components, while `approximate_disjoint_errors` explicitly introduces an approximation for mutually exclusive Pauli-channel cases. Record any such approximation; do not silently ignore decomposition failures. Seed reproducibility is conditional on Stim version, machine instruction support and the sequence of sample calls, so retain generated data and circuit hashes for durable replay. Deterministic injected-fault examples should be simulated with the same circuit and measurement-to-detector conversion as training data. [Official API reference](https://raw.githubusercontent.com/quantumlib/Stim/main/doc/python_api_reference_vDev.md).

## Close precedents and originality boundary

Local neural pre-processing is established, including Meinerz, Park and Trebst's scalable neural decoder. A compact network plus matching is not by itself a new scientific discovery. [2021 paper, v2](https://arxiv.org/abs/2101.07285v2).

Gicev, Hollenberg and Usman's fully convolutional 3D work explicitly studies spatiotemporal surface-code circuit noise and local/global decoding. Convolution over syndrome volumes is therefore a prior mechanism, not an original QubitGuard invention. [2025 paper](https://arxiv.org/abs/2506.16113).

An October 2026 preprint by Lange et al. describes modular graph-recurrent decoding and transfer across noise conditions and logical operations. It is relevant context for any broad portability claim, but is a preprint rather than an independently reproduced QubitGuard result. [2026 preprint](https://arxiv.org/abs/2610.03869).

QubitGuard's defensible contribution is an independently implemented, reproducible small-compute engineering study with an explicit model/hybrid hypothesis, held-out noise tests, failure cases and measured cost. Any claim of a novel algorithm needs a separate, broader prior-art assessment.

## Proposed falsifiable CPU experiment

This section is QubitGuard design reasoning, not a claim from the cited studies.

A direct logical classifier can be compact, but at low noise it can minimize loss by predicting no logical flip. Raw accuracy hides this failure because many shots are easy. A matching residual target offers a different experiment:

`residual_target = true_observable XOR standard_matching_prediction`.

A small spatiotemporal model can learn when a syndrome-consistent classical recovery belongs to the wrong logical class. The predicted final logical frame is the XOR of matching and the network's residual bit. An optional fixed logical representative can realize that toggle physically without changing the syndrome. This is neural post-decoding, not a density-reducing pre-decoder. Never advertise its neural runtime alone as end-to-end latency. Strong residual class imbalance and matching mistakes' rarity may make this harder than direct prediction; count available positive examples before spending compute.

The decisive question is whether temporal features and mixed-noise training reduce independent test failures enough to justify their incremental cost. Compare a compact direct model, the proposed hybrid, a temporal-feature ablation and a single-noise training control. Keep the training sample budget and optimization steps comparable. If learning fails, report the measured failure and preserve the trained model; do not search repeatedly on the test set until a favorable slice appears.

## Measurement requirements

- Use rotated surface-code memory distances 3, 5 and 7, with an explicit number of rounds and memory basis. A single memory basis evaluates one logical observable, not a complete universal quantum computer.
- Serialize exact circuits, detector coordinates, decomposed DEM and all enabled noise channels. Uniform depolarizing circuit noise already contains two-qubit fault propagation and X/Z correlations. Distinguish it from separately injected spatial bursts, leakage or drift.
- Draw independent train, validation and test seed namespaces. Reserve test syndromes for final comparisons. Select checkpoints, thresholds and hyperparameters on validation only. Noise values unseen during training test interpolation or extrapolation; they do not establish transfer to arbitrary hardware noise.
- Decode identical test shots with all methods. Give both matching baselines the true test noise model when claiming a strong oracle baseline. A mismatched calibration baseline can be a separate robustness experiment with an explicit label.
- Report failures/shots and logical failure probability per complete experiment. Use a binomial interval such as Wilson even when zero failures occur. Dividing shot-level failure by rounds is not an exact per-round estimator; fit across round counts if claiming a stationary per-round rate.
- Use paired comparisons on the same shots. Report neural-only wins and baseline-only wins, an exact paired test or paired bootstrap interval, and multiplicity handling for a sweep. Overlapping or non-overlapping separate confidence bars alone do not establish paired significance.
- Measure warmed CPU batch-one latency and batch throughput separately, including preprocessing, tensor conversion, matching and neural inference for the hybrid. State device, threads, batch, precision, warmup, repetitions and timing quantiles. Sampling/training are separate costs.
- Record parameters, checkpoint bytes, training wall time, samples seen and effective shots/second. Repeat initialization seeds when making model-comparison claims. A single trained seed establishes feasibility, not stable ranking.
- Preserve a deliberately selected failure example and a random example. Training truth may include observable labels and fault histories; production inference receives detector data and declared configuration only. The UI must never feed hidden truth into its predictions.

## Licensing and attribution ledger

| Component | Observed source terms | Project treatment |
| --- | --- | --- |
| Stim | [Apache-2.0](https://raw.githubusercontent.com/quantumlib/Stim/main/LICENSE) | Dependency, cite simulator; retain version |
| PyMatching | [Apache-2.0](https://raw.githubusercontent.com/oscarhiggott/PyMatching/master/LICENSE) | Dependency, cite Sparse Blossom/PyMatching; retain version |
| Ising-Decoding source | [Apache-2.0](https://raw.githubusercontent.com/NVIDIA/Ising-Decoding/main/LICENSE) | Research reference; no source copying required |
| NVIDIA pretrained weights | NVIDIA Open Model License, gated | Not downloaded or redistributed |
| AlphaQubit article | Open access research article | Cite, do not copy figures or model artifacts |
| QubitGuard trained weights | Locally generated from synthetic data | Ship with project license and provenance/model card |

This ledger is an engineering record, not legal advice. Check installed dependency metadata and retained files before redistribution; source pages can change.
