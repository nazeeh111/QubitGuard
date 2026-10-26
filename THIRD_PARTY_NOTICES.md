# Attribution and artifact rights

Original QubitGuard implementation, synthetic datasets and locally trained
checkpoints are supplied under the root MIT license. All checkpoints start
from random initialization. No third-party model weights are included.

Stim and PyMatching are dependencies under Apache-2.0. PyTorch is a dependency
under its BSD-style license. NumPy and SciPy have BSD-style licenses;
Matplotlib uses its project license; FastAPI and uvicorn use MIT/BSD terms.
Transitive packages retain their own licenses. The environment is installed
from the package registry; this source archive does not redistribute its
Python environment. `requirements-lock.txt` records exact installed versions.
Consult the installed distributions' LICENSE/NOTICE files when redistributing
an environment or binary bundle.

NVIDIA Ising-Decoding is a research reference with Apache-2.0 source.
Its model weights have separate NVIDIA Open Model License terms and gated
access. QubitGuard did not download, use or redistribute these weights.
No source files, model structures or figures were copied from NVIDIA or
AlphaQubit. Convolutional and hybrid decoding are established prior art.

Primary citations and checked source/model license links are recorded in
[research foundations](docs/research-foundations.md) and [references](docs/references.bib).
