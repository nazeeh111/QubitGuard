# Results at a glance

QubitGuard is a measured engineering study. Its compact neural hybrid does not
establish a statistically meaningful advantage over correlated PyMatching in
the tested conditions. The negative result is the result, not a missing feature.

## Reference operating point

Uniform circuit noise, p=0.007, rounds=distance, X memory, 20,000 independent
shots per row. Counts are full-shot logical failures, not per-round rates.
The neural column is the declared primary initialization, seed 17.

| Distance | Standard matching | Correlated matching | FrameNet hybrid |
| --- | ---: | ---: | ---: |
| 3 | 763 / 20,000 | 705 / 20,000 | 706 / 20,000 |
| 5 | 789 / 20,000 | 626 / 20,000 | 626 / 20,000 |
| 7 | 869 / 20,000 | 603 / 20,000 | 603 / 20,000 |

![Uniform logical error rates with Wilson intervals](../figures/logical-error-uniform.png)

There are no primary neural wins over correlated matching after the declared
364-comparison Bonferroni adjustment. Two d5 high-noise settings significantly
favor correlated matching. Neural inference also includes the cost of both
matching passes, features and the network. It is slower in this CPU study.

## What the experiments teach

- Much of the improvement over standard matching comes from access to the
  correlated baseline. The context ablation exposes this dependence.
- Seed variation and shifted high-noise settings matter. A favorable isolated
  example cannot establish decoder quality.
- Removing pooled geometry with a larger 25,713-parameter readout did not
  improve its separate, fresh 280,000-shot d3 test mixture. The checkpoint and
  unsuccessful revision are retained.
- A correction can satisfy every detector parity and still have the wrong
  logical class. The simulator displays both checks separately.

![Ablation results](../figures/ablations.png)
![Complete CPU inference cost](../figures/cpu-latency.png)

## Read the evidence

[Technical report](technical-report.pdf) · [Primary numerical table](../artifacts/benchmark.csv) ·
[Paired comparisons](../artifacts/paired-comparisons.csv) · [Model card](model-card.md) ·
[Data card](data-card.md) · [Independent scientific audit](release-audit.md)

Primary test shots: 840,000 across 42 settings. Follow-up test shots: 280,000
across 14 settings. Model comparisons share each setting's held-out shots;
these counts do not multiply with the number of decoders. All 322 result rows
and 406 paired comparisons reconcile with retained raw predictions.

## Limitations and next experiments

This is simulated X-memory training at d3/d5/d7 with rounds=distance, limited
training compute and simple circuit-level noise. It does not establish a
threshold, hardware transfer, local pre-decoding, quantum advantage or real-time
decoding. Measurement-heavy noise is a channel imbalance, not a Pauli-Z bias.
The baseline knows the true test noise model. The neural score is uncalibrated.

Useful next experiments are mismatch between assumed and actual noise,
syndrome-only learned models with stronger rare-event objectives, joint X/Z
memory, longer memories, physically motivated leakage/crosstalk, and larger
validation budgets. Each needs fresh test streams, fair baseline retuning and
complete cost accounting. No predicted improvement is presented as a result.
