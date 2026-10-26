# FrameNet model card

**Task:** predict one final logical observable in rotated surface-code X memories,
with d rounds at distance d. This is classical inference on simulated syndrome
bits, not a quantum neural network or inference on a quantum processor.

**Model family:** independently implemented compact axial spatial/temporal
convolutions, validity-mask input, boundary-aware adaptive pooling and small
logical readout. Full residual model has 4,209 trainable parameters (width 8).
Every full model is trained from random initialization on synthetic Stim data.
There is no external pretrained checkpoint.

**Input:** detector-event history and detector-validity mask, standard matching
logical prediction, correlated matching logical prediction, their disagreement,
syndrome density, and the declared nominal noise strength. Ground-truth
observables and sampled error histories are never inference inputs.

**Target:** true observable XOR standard matching prediction. The final
prediction XORs the learned residual bit with standard matching. A graphlike
recovery certificate can be constructed by XORing the matching correction with
a closed odd-logical graph cycle. This demonstrates consistency in the detector
error model; it does not uniquely identify microscopic faults.

**Training scope:** separate models for d3, d5 and d7; X memory only, rounds=d;
three initialization seeds on the same retained training data. Mixed nominal
p in {0.003, 0.007, 0.012}; all four circuit channels enabled. 60,000 training
shots and 24,000 validation shots per distance; 10 epochs, AdamW, float32 CPU,
class-weighted binary cross entropy. Checkpoint and threshold are selected
on validation data. A sigmoid output is an uncalibrated class-weighted score.

**Controls:** d3 direct logical prediction, neural time-order ablation,
correlated-context ablation, same-budget single-p training. These have one
initialization each. The time ablation still receives full-history classical
matching context. Separate model files and exact settings are recorded in the
checkpoint summaries and benchmark artifacts.

**Evaluation:** same independent held-out shots for all decoders; matched-noise
standard and correlated PyMatching; unseen p values and a measurement-heavy
channel-imbalance profile. Exact counts, Wilson intervals, paired comparisons,
seed variability, latency and training efficiency accompany the technical
report. No advantage should be inferred from a single successful demo shot.

**Known limits:** no leakage, coherent noise, soft analog measurements,
hardware calibration, streaming operation, distance transfer, universal
logical-operation coverage or real-time guarantees. Neural Z-memory output is
explicitly unavailable. Low-noise zero-failure counts remain uncertain.
A finite sweep at d3/d5/d7 does not estimate an asymptotic threshold. The
hybrid calls matching before the neural model; include that cost in latency.

**License:** original QubitGuard weights and synthetic data use the project MIT
license. NVIDIA weights were neither downloaded nor used. External dependency
terms remain separate. See `THIRD_PARTY_NOTICES.md` and primary citations.
