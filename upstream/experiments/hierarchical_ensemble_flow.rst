AA-only hierarchical ensemble flow: current implementation
========================================================

The new model samples a conditional joint distribution:

    p(c, f | AA) = p(c | AA) p(f | c, AA)

The coarse state has ceil(N/4) positions and 16 continuous components per
position. The fine state has N positions and 8 continuous components. A shared
Transformer is called as a coarse velocity field and then as a conditional fine
velocity field. Each stage integrates a sampled noise state toward its learned
conditional distribution. The fine stage receives the exact coarse sample being
decoded, through both broadcast conditioning and attention to coarse tokens.
AA order within a block is retained by ordered flatten pooling.

The generated fine vectors are snapped to the existing 1024-entry fine codebook
only at the decoder interface. The structural encoder, decoder, and codebook
remain frozen. This prototype uses learned AA embeddings, not ESM sequence
features. Generation needs only an AA sequence and random seeds. Real structures
supply paired latent training targets, not inference inputs.

Training and scope
------------------

Both objectives regress flow velocity along a noised interpolation path. This
is not a deterministic AA-to-coarse MSE regressor, and the fine objective is not
cross entropy. Fine training uses the native coarse representation paired with
the same snapshot's fine representation, plus small normalized coarse jitter.
The jitter is a local robustness augmentation; it does not establish robustness
to every sampled-coarse error. Arbitrary sampled coarse states must not be paired
with unrelated fine labels.

Sampling noise supplies variation, not evidence about the unknown physical
state. Data processing inequalities still apply. A deterministic supervised
baseline can quantify one point estimator's performance; it is not a theoretical
predictability upper bound. Latent distance and decoded geometric similarity
remain separate measurements. A previous correlation on another model is not a
measurement for this new codebook.

Probe records
-------------

Frozen protocols, full sampled outputs, metrics, and figure manifests live in
artifacts/hierarchical_ensemble_flow_v1. These are local experiment artifacts,
not bundled model weights or a production validation claim.

Native MD next step
-------------------

A separate ATLAS pilot selects a small public TRAIN protein using only length,
sequence-overlap screening, availability, and archive size. R1 is for training,
R2 is development, and R3 is sealed. First check frozen-tokenizer reconstruction
on predetermined R1/R2 frames, including cross-block links. Train an MD flow only
if its representation ceiling is adequate. A one-sequence replicate split tests
trajectory transfer within that sequence, not generalization to unseen proteins.
