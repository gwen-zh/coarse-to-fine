Sequence-conditioned spatial multiscale ensemble pilot
=====================================================

Goal
----

Given one AA sequence, sample coarse block layouts and then compatible fine
details. Repeated independent noise seeds produce an ensemble, without a native
structure as an inference input. This is a spatial hierarchy, not a model of
time-indexed MD transitions.

Implementation
--------------

The shared Transformer parameterizes two conditional flow velocity fields::

    AA + coarse noise -> coarse flow -> M = ceil(N/4) coarse states
    AA + sampled coarse + fine noise -> fine flow -> N continuous 8D vectors
    coarse + fine -> frozen local-frame decoder -> N x (N, CA, C) coordinates

The first experiment contains three fresh-training arms, each with two seeds:

* ``latent16_teacher``: learned coarse 16D vectors, paired native context.
* ``centers3_teacher``: directly generated block-center coordinates, paired context.
* ``centers3_scheduled``: direct centers with probability-0.5 predicted context.

Every generated latent pair is decoded both continuously and after nearest-code
quantization. These outlets share exactly the same model and noise samples.
The existing 1024-entry book is only used by the quantized control; continuous
generation does not snap to it. Native reconstruction under each coarse
convention is recorded before generator training.

The coarse sampler uses only M positions; the fine sampler includes coarse
prefix context plus N residue states. Both scales condition on ordered AA block
embeddings, and fine explicitly sees the generated coarse. Gaussian initial
noise is a base sampling distribution, not a claim that encoded latents follow
a Gaussian.

Representation weights are frozen at the completed local-frame PDB-pretrained
seed-2026 checkpoint. Coarse and fine generators are fresh, width 128, four
Transformer layers. Continuous targets use flow matching, not cross entropy.

Scheduled context
-----------------

Following the mechanism in PAR, construct a clean-endpoint estimate from this
same training example's noised coarse state and velocity prediction. Detach
the estimate and sometimes use it to condition fine training. Re-encode the
same backbone in frames derived from this estimated coarse; do not reuse fine
labels expressed in different frames. This is robustness training, not proof
that train/inference exposure bias has been eliminated. Full coarse rollouts
remain different from endpoint estimates on native noising paths.

Official PAR is a reference, not this pilot's pretrained backbone. The pinned
revision and hashes are in protocol.json. PAR uses residual multiscale coordinate
interpolation; this pilot retains disjoint stride-four block means and our
rigid N/CA/C decoder. It does not reproduce PAR's results or implement all of
its scales, noisy-context learning, or sampling settings. No official setup
scripts or remote code are executed.

Data and conclusions
--------------------

Only ATLAS 1b2s_E R1 trains the generators: 1001 frames, 90 residues. The existing
64-frame R2 development set is reused. This is a one-sequence density-fit and
pipeline comparison. Sequence conditioning is implemented, but unseen-sequence
generalization cannot be established from this training set.

The earlier R3 test has already been consumed by a different representation;
this experiment does not read it. Future confirmation requires fresh data.
PDB training structures trained the representation, not the MD generators in
this pilot. Ensembles are not claimed to have calibrated Boltzmann weights.

Compare nearest-reference error in both directions, distance/Rg JS divergence,
RMSF, pairwise spread, backbone geometry and measured latency. Record raw samples
without geometry filtering. Empirical R1 and native single-frame controls are
diagnostics. Oracle coarse is explicitly separated from AA-only generation.

Run
---

From the project root, with the official sparse checkout in third_party::

    .venv/bin/python -m unittest discover -s upstream/tests -p test_sequence_multiscale_ensemble.py
    sbatch upstream/scripts/sequence_multiscale_ensemble.sbatch prepare
    # After prepare, submit --array=0-5 training and then a dependent report job.

After final checkpoints exist, AA-only inference needs no trajectory cache::

    .venv/bin/python upstream/experiments/run_sequence_multiscale_ensemble.py \
        --stage generate --arm centers3_teacher --seed 2026 \
        --sequence YOUR_SEQUENCE --samples 8 --output artifacts/example_ensemble

The current model has been trained on one protein. Arbitrary new sequences are
extrapolations until a separate multi-protein training and validation experiment.
