# Coarse-to-fine protein conformation research

Project source and progress for teammates: structural encoders/decoders, codebooks, continuous coarse/fine generators, PAR coarse transfer, frozen-decoder feedback search, experiment drivers, tests, and analysis/plotting scripts. The folder layout is preserved so existing imports remain recognizable.

**Status, November 26, 2026:** local geometry refinement on the original model has independent confirmation. The new multiscale generator remains a prototype. The [completed chiral-velocity study](docs/CHIRAL_VELOCITY_RESULTS_2026-11-26.rst) improves coarse error by 4.77% / 9.99% against the native control and beats a parameter-matched polar control in both fixed training-draw seeds. This is a two-TRAIN-structure learnability result; the 2 Å fitting target remains unmet, with no independent generalization or overall superiority claim. The [fixed-weight self-conditioning probe](docs/SELF_CONDITION_PROBE_2026-11-26.rst) stopped its first attempt on a strict replay mismatch; recovery job 58067466 uses the parent hardware and unchanged numeric protocol/tolerance, preserving the original outputs. No new quality result is available. The earlier signed-volume-loss replication remains unsuccessful.

Start with [the research status](docs/RESEARCH_STATUS.rst), [the Chinese progress and handoff](docs/PROGRESS_2026-11-24_ZH.rst), or [the aggregate results and provenance](docs/progress_2026-11-24.json).

This source checkout includes selected aggregate progress, but no model checkpoints, cached sequence features, raw result files, manuscripts, figures, trajectories, or training corpus. The runnable synthetic smoke check needs none of those artifacts. Real reconstruction/generation uses explicitly supplied local inputs and matching checkpoints.

## Quick start

Python **3.12** is the tested interpreter for these research additions. The original author environment remains under `upstream/environment.yml`.

```bash
git clone https://github.com/gwen-zh/coarse-to-fine.git
cd coarse-to-fine
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU setup; use the cu124 PyTorch index for the tested CUDA 12.4 variant.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
python scripts/run_core_tests.py
python scripts/smoke_pipeline.py
```

The smoke pipeline uses small randomly initialized models and synthetic features to check the coarse -> fine -> coordinates interface. It is a software check, not a protein prediction result.

[中文上手说明](docs/TEAM_START_ZH.rst) · [Third-party provenance](THIRD_PARTY_NOTICES.rst)

## Current architecture

There are three separate branches; their checkpoints, metrics and token semantics are not interchangeable.

| Branch | Representation / generation | Current evidence |
| --- | --- | --- |
| Original Hierarchical ConfGen + search | Released coarse-to-fine path; decoder-guided discrete fine-token edits with frozen weights | Two ungated angle endpoints improve on 64 independent confirmation proteins / 512 conformations |
| Own continuous multiscale prototype | Physical coarse centers `M x 3`; local-frame fine latents `N x 8`; conditional coarse/fine flow and own rigid-frame decoder | Reconstruction works; reliable sequence-to-layout generation and ensemble coverage remain unresolved |
| PAR coarse-transfer pilot | Pinned pretrained PAR modules + AA adapter / optional flow LoRA; coarse output connects to the own fine/decoder interface | Two-training-structure diagnostics; no full sequence-conditioned three-scale PAR generator or independent superiority claim |

For the current own-model prototype, `M = ceil(N/4)` and each coarse point is the mean Cα coordinate of up to four consecutive residues. Fine latents are continuous and expressed in coarse-derived local frames. The current generator does **not** snap them to a codebook. The optional 1,024-entry codebook and 16D coarse encoder remain available for historical controls.

Representation training takes known N/CA/C coordinates. Generation takes frozen, sequence-only ESM3 features and noise:

```text
AA sequence -> AA-only features H[N,1536]
H + coarse noise -> coarse flow -> C[M,3]
H + C + fine noise -> conditional fine flow -> F[N,8]
C + F -> local frames / own rigid decoder -> N/CA/C coordinates [N,3,3]
```

The fine generator is conditioned on the **same sampled coarse** that the decoder uses. The rigid output fixes residue-internal geometry and enforces zero block Cα centroid offsets; peptide connections and ensemble quality still need evaluation. Multiple noise draws supply stochasticity, not proof of correct thermodynamic populations.

The PAR pilot replaces only the coarse module. Its current interface uses physical block means and `ceil(N/4)`, whereas native PAR uses interpolation and `floor(N/4)`. It is not a reproduction of PAR's published complete sampler. See [PAR setup and replay boundaries](docs/PAR_REPRODUCTION.rst).

The existing `scripts/demo_representation.py` and `scripts/sample_generator.py` commands below are **legacy portable demos** for the older 16D coarse / 1,024-fine-token model. They do not load the current continuous-fine model or PAR adapters.

## Source map

| Path | What to edit |
| --- | --- |
| `upstream/experiments/sequence_multiscale_ensemble.py` | Current physical-center / continuous-fine codec and paired flow interfaces |
| `upstream/experiments/block_stable_frame_tokenizer.py` | Local-frame fine representation and rigid backbone output |
| `upstream/experiments/hierarchical_ensemble_flow.py` | Own continuous coarse / conditional fine velocity models |
| `upstream/experiments/explicit_coarse_fine_flow.py` | Additional coarse geometric context for fine generation |
| `upstream/experiments/par_coarse_backbone.py` | Strict public PAR loading, AA conditioning and coarse sampling |
| `upstream/experiments/par_coarse_capacity.py` | Native-flow LoRA capacity control and geometry diagnostics |
| `upstream/experiments/par_signed_volume_replication.py` | Closed fixed-seed replication of signed-volume supervision |
| `upstream/experiments/par_chiral_velocity.py` | New equal-parameter polar/axial velocity-residual control under original MSE |
| `upstream/experiments/hierarchical_rigid_vq.py` | Structural encoder/decoder, EMA codebooks, rigid N/CA/C output |
| `upstream/experiments/hierarchical_latent_contract.py` | Block pooling and exact center/offset composition |
| `upstream/experiments/mixed_hierarchical_generator.py` | Earlier continuous coarse / masked fine-token generator |
| `upstream/experiments/fixed_pool_geometry_search.py` | Frozen-decoder token search |
| `upstream/experiments/` | Current training, diagnostic, analysis and plotting sources |
| `upstream/tests/` | All existing tests; archival tests may need external run artifacts |
| `scripts/` | Portable loaders, local checkpoint export, demos, core tests, and Slurm launcher |
| `upstream/scripts/` | Historical cluster scripts with their original paths/configurations |
| `upstream/Hierarchical_VQ_VAE/` | Original tokenizer implementation plus our source changes |
| `upstream/hierarchical_ConfGen/` | Original generator implementation plus our source changes |

## Use locally shared legacy demo models

For trusted training checkpoints, create compact **local** snapshots:

```bash
python scripts/export_inference_checkpoint.py tokenizer \
  --warmup /path/to/continuous.pt --final /path/to/final.pt \
  --output artifacts/local_models/tokenizer.pt
python scripts/export_inference_checkpoint.py generator \
  --source /path/to/generator/final.pt \
  --output artifacts/local_models/generator.pt
```

Native structure reconstruction (input NPZ array `backbone[N,3,3]`, atom order N/CA/C, angstroms):

```bash
python scripts/demo_representation.py \
  --input /path/to/reference_backbone.npz \
  --checkpoint artifacts/local_models/tokenizer.pt
```

Sequence-feature-only sampling (input NPZ array `h[N,1536]` from the matching frozen AA-only feature extractor):

```bash
python scripts/sample_generator.py \
  --features /path/to/aa_features.npz \
  --generator artifacts/local_models/generator.pt \
  --tokenizer artifacts/local_models/tokenizer.pt --samples 8 --device cuda
```

Sampling does not read a reference structure or native structural labels. Checkpoint and fine-token semantics must match; equal vocabulary size is not sufficient. The existing new-model generator is experimental and has known quality/generalization limitations. See each driver's protocol and evaluation code before drawing quality conclusions.

## Full experiments

```bash
python -m pip install -r requirements-research.txt
python scripts/attach_shared_artifacts.py --source /path/to/shared/artifacts
```

The linker connects selected project directories only and refuses to overwrite files. It does not link authentication caches. Historical drivers rely on frozen versioned protocols, complete datasets, and sometimes original machine paths. They are archival replay drivers, not generic one-command training APIs. Prepare a **new** versioned run and update paths deliberately for changed experiments; do not delete hash checks to relabel changed code as an old experiment.

Continuous-model drivers include `run_sequence_multiscale_ensemble.py`, `run_generator_curriculum.py` and `run_multiframe_coarse_context.py` under `upstream/experiments/`. The PAR diagnostic sequence includes `run_par_two_protein_fit.py`, `run_par_two_protein_continuation.py`, `run_par_coarse_capacity.py`, `audit_par_coarse_chirality.py` and `run_par_signed_volume_replication.py`. These are archival drivers, not ready-to-run fresh-study APIs. Full data, parent states and frozen protocols remain local/shared; [read the replay requirements](docs/PAR_REPRODUCTION.rst).

Earlier VQ/mixed-model training drivers (`train_hierarchical_tokenizer.py`, `train_mixed_hierarchical_generator.py`, `continue_mixed_hierarchical_generator.py`) remain included for provenance. No completed study is automatically restarted by this update.

## Slurm

Choose your own cluster resources:

```bash
sbatch --partition=YOUR_PARTITION --cpus-per-task=4 \
  scripts/slurm/run_experiment.sbatch scripts/smoke_pipeline.py
```

The launcher uses the submission directory and `.venv/bin/python`; override `CTF_PROJECT_ROOT` or `CTF_PYTHON` as needed. Existing `upstream/scripts/*.sbatch` retain their original cluster settings for provenance; adapt them in a new experiment version.

## Optional upstream projects

For BioEmu evaluation or the frame-flow branch:

```bash
git submodule update --init --recursive
git -C upstream/protein-frame-flow apply ../../docs/patches/protein-frame-flow-lazy-scatter.patch
```

The compatibility patch defers optional `torch_scatter` imports; functions using scatter still require that extension. Optional projects retain their own dependencies and licenses. Original author data/checkpoint download instructions are in [upstream/README.md](upstream/README.md).

For the PAR pilot, initialize only its pinned dependencies:

```bash
git submodule update --init third_party/par-protein third_party/par_preflight_openfold_v1
```

Follow [the PAR-specific setup notes](docs/PAR_REPRODUCTION.rst). Public pretrained weights must be downloaded separately and verified; they are not included here. Neither these submodules nor a passing synthetic test establish generation quality.

## Collaborating

Create one branch per hypothesis, keep model/seed/config/data splits recorded, and write outputs to ignored `artifacts/<new_run>/`. Keep original-model and new-model experiments distinct. Compare reference accuracy, geometry, sample diversity, and compute cost together; algorithmic gates are not independent validation metrics.
