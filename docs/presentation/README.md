# Same syndrome. Different logical class.

This is retained-checkpoint inference on one **deliberately selected** Stim detector-error-model shot. It illustrates the difference between detector parity and logical success, not aggregate accuracy. Source commit `9b9c7885584bedb1ef0965ed9d96b1ea83b01dc7`, actual tree `396707edf6833e2fcd994683779ce3e840febe74`, QubitGuard `0.1.1.dev0`. The recorded runtime is CPython 3.13.2, Stim 1.16.0, PyMatching 2.4.0 and Torch 2.14.1 on macOS arm64. The full source and retained checkpoint are required. No retraining was performed.

## Reproduce one saved record

After the [repository setup](../../README.md#run-the-playground), from its root:

```sh
.venv/bin/qubitguard verify --root "$PWD"
.venv/bin/qubitguard simulate --root "$PWD" --distance 3 \
  --noise 0.007 --seed 75 --profile uniform --basis x --out shot75.json
```

The output is a data-only `qubitguard.shot/v1` JSON file. Read it with any JSON reader, for example:

```sh
.venv/bin/python -c 'import json; s=json.load(open("shot75.json"))["shot"]; print(s["predictions"]); print(s["certificate"]); print(s["success"])'
```

An existing destination is preserved and refused, so choose a new path to repeat the command. The two actual exports used for this figure are [selected-toggle.json](selected-toggle.json) and [selected-failure.json](selected-failure.json). Per-request timings are retained as recorded metadata but are not plotted or treated as a latency benchmark. Scientific fields repeated exactly in the same pinned runtime after excluding timings; seeds are not cross-version or cross-machine guarantees. The file has no executable checkpoint state and the playground has no JSON-import control. To replay it there, set the saved settings and simulate again.

## Read the detector history

The matrix includes every detector, not only active events. Columns are the exact Stim `(x,y)` coordinates sorted lexicographically; rows are detector time `t`, including the boundary layers. A filled `1` is an event, an open `0` is a detector with no event, and a dash means that coordinate has no detector at that boundary time. There are 24 valid bits in 32 matrix slots, with 8 absent slots. These are detector coordinates, not physical length units. Nominal `p=0.007` is the modeled probability for each uniform circuit-noise channel, not a measured hardware error rate.

All three graphlike recoveries satisfy the detector syndrome:

| Decoder | Graph edges | Predicted logical flip | Parity certificate | Scored outcome |
| --- | ---: | ---: | --- | --- |
| Standard matching | 4 | 0 | valid | wrong |
| Correlated matching | 4 | 1 | valid | matches truth |
| FrameNet hybrid | 5 | 1 | valid | matches truth |

The simulated true observable is `1`, used only to score outcomes. It is not a model input. The hybrid reads detector history, validity mask and matching-derived context, then applies a learned logical toggle to standard matching. Its uncalibrated residual score is not a confidence probability. The standard/hybrid edge-set symmetric difference is exactly `D1—D2`, `D1—B`, `D2—B`. Detector parity cancels at D1 and D2, while the edge logical bits XOR to 1. B denotes the matching graph boundary, whose parity is unconstrained. The triangle is a **topological graph schematic**, not a physical lattice or spatial path. Edge counts are not a quality ranking.

A second retained teaching case, d3 uniform X, `p=0.012`, seed 4, has **all three valid parity certificates and all three wrong logical predictions** (prediction1, truth0). Regenerate it with `--noise 0.012 --seed 4` and a new output path. Both cases were already selected in [demo-examples.json](../../artifacts/demo-examples.json); this task did not search for favorable seeds.

## Keep the illustration separate from the study

At d3 uniform `p=0.007`, the retained primary test has 20,000 independent shared shots: standard 763 failures, correlated 705, hybrid 706 (declared primary model seed 17). These are full-memory-shot counts. The retained study uses 95% Wilson marginal intervals and paired comparisons with a 364-comparison primary adjustment; it does not establish an advantage over correlated matching. No interval is attached to the selected one-shot illustration because it is not an error-rate estimate. See [results](../results.md), [technical report](../technical-report.md), [model card](../model-card.md) and [raw benchmark](../../artifacts/benchmark.csv). The model is trained for X memories only; neural Z-memory output remains unavailable. This is simulation, not hardware or quantum advantage.

## Actual interface and figure provenance

[Current playground capture](playground-current.png) shows this actual saved-settings replay. Switch decoder tabs to inspect each recovery, select Space-time, and step the detector timeline. Changing the displayed time does not resample the shot. Existing expert controls and defaults are preserved.

[provenance.json](provenance.json) pins the checkpoint, producing source, selected cases and exact toggle-cycle derivation. `python docs/presentation/render.py` regenerates both PNG/SVG sizes using Matplotlib, reading the raw JSON and verifying the retained checkpoint hash. All 24 detector bits and all 8 absent slots are encoded without smoothing, interpolation or aggregation. The figure uses the product's existing green/paper identity, explicit text labels and a portable sans-serif fallback. Important values and semantics are also available in this Markdown and the raw JSON. The included PNGs are rendered scientific diagrams, not screenshots. The linked playground image is an actual browser capture.
