PAR pilot: setup and replay boundaries
======================================

Read RESEARCH_STATUS.rst before running a study. These drivers implement a
coarse-scale transfer and diagnostic pilot, not the complete published PAR
three-scale sampler and not a validated protein ensemble model.

Pinned external code
---------------------

From the repository root::

    git submodule update --init third_party/par-protein third_party/par_preflight_openfold_v1
    git -C third_party/par-protein rev-parse HEAD
    git -C third_party/par_preflight_openfold_v1 rev-parse HEAD

Expected commits:

* PAR: 92d1c3ecc9822f897b66d53b3852059e6750aee2.
* OpenFold: be2ec1841f16c966c65ae0e7599ebbadc725757d.

Set import paths from the root for PAR processes::

    export PYTHONPATH="$PWD/third_party/par-protein:$PWD/third_party/par_preflight_openfold_v1:$PWD/upstream/experiments${PYTHONPATH:+:$PYTHONPATH}"

Use a separate environment for this branch. The original runs used Python 3.12
and PyTorch 2.6.0+cu124, plus the project's research dependencies and dm-tree
0.1.9. An environment listing is diagnostic, not a portable lockfile. This
repository does not claim a newly verified clean-install PAR environment.
The project-local loader checks that pair/triangle operators are disabled
before guarding unused OpenFold imports; an active unsupported branch fails.

The tested PyTorch CUDA build does not support the RTX PRO 4000 / SM120 device
encountered on this cluster. Select supported hardware or validate an explicit
new environment. Historical CPU runs use two Torch threads per training arm.

Public checkpoint
------------------

The acquisition script downloads one public weight file and verifies its exact
size and SHA256. It refuses to overwrite a present checkpoint::

    python upstream/experiments/acquire_par_coarse_backbone.py

* Model repository: ByteDance-Seed/PAR.
* Revision: 7fb609346f5d9a2baa76add1f387aee1bdb8874b.
* Remote file: checkpoints/3scale_by_ratio_60m/200k.ckpt.
* Local file: artifacts/par_coarse_backbone_v1/official_checkpoint/PAR_3scale_by_ratio_60m_200k.ckpt.
* Bytes: 1,405,820,768.
* SHA256: e6fbed2fc5bcd33f502094f499dde4b5252e074a2c664a7853308cf49419ff65.

No weights are committed. The loader uses strict state loading, restores the
verified nonlearned level-index buffer and freezes the official model. Only
explicitly registered AA adapter / optional native-flow LoRA parameters train.

What the source-only checkout can test
--------------------------------------

The small mathematical tests below need no model weights or training labels.
Run with a working core environment from the repository root::

    python -m unittest discover -s upstream/tests -p test_par_coarse_backbone.py -v
    python -m unittest discover -s upstream/tests -p test_par_coarse_capacity.py -v
    python -m unittest discover -s upstream/tests -p test_par_chirality_audit.py -v
    python -m unittest discover -s upstream/tests -p test_par_signed_volume.py -v
    python -m unittest discover -s upstream/tests -p test_par_signed_volume_replication.py -v

These tests do not demonstrate model quality or replace real-checkpoint preflight.
Earlier portable scripts/sample_generator.py demos use the OLD mixed discrete
model; they cannot load the later continuous-fine checkpoints or PAR adapters.

Inputs needed for historical numerical replay
----------------------------------------------

Historical runs need external, locally shared artifacts:

* AA-only ESM3 feature matrices H[N,1536] and their feature index.
* Exact PDB/MD train and development membership, training labels and frame keys.
* Matching own-model codec/fine-flow weights for full-chain evaluation.
* Parent AA adapter checkpoints, Adam moments, RNG state and compact traces for
  continuation, capacity and signed-volume replication studies.
* Each completed run's protocol.json, preflight.json, hashes and parent reviews.

The two-target diagnosis uses only 1b2s_E R1:frame1130 and 1ef1_D R1:1080.
It is deliberately a training-pair check. Do not replace those targets, noise
keys or parent checkpoints and call the result a reproduction of the old run.

Latest archival entry points:

* run_par_two_protein_fit.py and run_par_two_protein_continuation.py.
* run_par_coarse_capacity.py and audit_par_coarse_chirality.py.
* run_par_signed_volume.py and run_par_signed_volume_replication.py.

Their stage ordering, paths and exact freeze checks are defined in code and
corresponding upstream/scripts/*.sbatch files. They are not standalone training
APIs. The signed-volume family is complete and failed its fixed replication
criterion; do not restart it as an automatic extension.

Start a new hypothesis by copying the relevant driver into a NEW versioned
namespace, inspecting data and checkpoint paths, checking real-weight preflight
and freezing a new protocol. Keep old numeric sources and archived protocols
unchanged. Never remove a hash check merely to get a modified study to run.
