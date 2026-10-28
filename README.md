# Coarse-to-fine protein conformation research

Current project source code for teammates: structural encoders/decoders, codebooks, generators, frozen-decoder feedback search, experiment drivers, tests, and analysis/plotting scripts. The folder layout is preserved so existing imports remain recognizable.

This **code-only** checkout does not publish model checkpoints, cached sequence features, unpublished result files, manuscripts, figures, trajectories, or the training corpus. The runnable synthetic smoke check needs none of those artifacts. Real reconstruction/generation uses explicitly supplied local inputs and matching checkpoints.

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

- Sequence blocks contain up to four residues; `M = ceil(N/4)`.
- Coarse representation: one continuous16D vector per block, decoded into a block-center coordinate.
- Fine representation: one8D vector per residue, quantized into a1,024-entry codebook.
- Structural decoder: coarse + fine + adjacent-block context -> proper residue rigid frames -> N/CA/C coordinates.
- Fine offsets have zero block CA centroid by construction. This separates block-center placement from local offsets; it is not proof of learned topology.
- Generator: sequence-only features condition continuous coarse flow, then a masked fine-token generator, then the structural decoder.

The current hybrid representation keeps coarse continuous and bypasses the coarse codebook. The full VQ implementation and earlier alternatives are also in the experiment sources. Models with changed token semantics require matched generator adaptation.

## Source map

| Path | What to edit |
| --- | --- |
| `upstream/experiments/hierarchical_rigid_vq.py` | Structural encoder/decoder, EMA codebooks, rigid N/CA/C output |
| `upstream/experiments/hierarchical_latent_contract.py` | Block pooling and exact center/offset composition |
| `upstream/experiments/mixed_hierarchical_generator.py` | Continuous coarse flow and conditional fine masked generation |
| `upstream/experiments/fixed_pool_geometry_search.py` | Frozen-decoder token search |
| `upstream/experiments/` | Current training, diagnostic, analysis and plotting sources |
| `upstream/tests/` | All existing tests; archival tests may need external run artifacts |
| `scripts/` | Portable loaders, local checkpoint export, demos, core tests, and Slurm launcher |
| `upstream/scripts/` | Historical cluster scripts with their original paths/configurations |
| `upstream/Hierarchical_VQ_VAE/` | Original tokenizer implementation plus our source changes |
| `upstream/hierarchical_ConfGen/` | Original generator implementation plus our source changes |

## Use locally shared trained models

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

Principal training drivers: `train_hierarchical_tokenizer.py`, `train_mixed_hierarchical_generator.py`, and `continue_mixed_hierarchical_generator.py` under `upstream/experiments/`. Their protocol builders, smoke checks, evaluation and plotting sources are included. Full historical data and optimizer states remain local/shared.

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

## Collaborating

Create one branch per hypothesis, keep model/seed/config/data splits recorded, and write outputs to ignored `artifacts/<new_run>/`. Keep original-model and new-model experiments distinct. Compare reference accuracy, geometry, sample diversity, and compute cost together; algorithmic gates are not independent validation metrics.
