Sequence conditions with a fixed structural representation
=========================================================

The previous small generator fit several seen MD sequences but failed on two
new development sequences. This experiment changes sequence conditioning and
keeps the structural encoders, fine book, continuous decoder, target labels,
coordinate gauge, normalization, data mixture, optimizer and sampling budget
fixed.

AA_lookup and ESM3_context have the same small-generator trainable parameter
count and exactly the same initial trainable weights. Both use a trained
LayerNorm(1536) and Linear(1536,128) projection. The control receives a fixed
random 1536D per-AA lookup; the experimental arm receives frozen contextual
features from the original ESM3 sequence-only forward. Ordered stride-four
block embeddings and the conditional coarse-to-fine flow architecture remain
the same. Our structural encoder and decoder do not become ESM3 components.

Both arms train fresh for 4096 updates with the identical MD_plus_PDB recipe:
three balanced MD examples plus one static PDB training example per update.
All four batch examples and noise/time draws are checked between arms. Two
training seeds are used, with final endpoints only. There is no automatic
extension or checkpoint selection.

Existing AA-only features for 899 PDB training proteins are reused only after
the original corpus identity and per-file hashes are checked. Nine new sequence
requests cover seven MD sequences and the two exact shuffled-AA sensitivity
controls used by the previous evaluator. The extraction process receives no
coordinates or native tokens. Re-extract an old PDB anchor before accepting
the new cache: relative RMSE <=0.01 and mean per-residue cosine >=0.999. Record
GPU, weight SHA256, feature SHA256, extraction time and repeated-forward checks.

Matching the small generator does not equalize total model capacity or compute.
Frozen ESM3 adds a 1.4B pretrained model and feature-extraction cost; report that
cost separately. The two development sequences are new relative to our training
corpus, but their exposure during ESM3 pretraining is unknown. No strict claim
of foundation-pretraining exclusion is justified.

The primary readout remains raw AA-only generation on the two new development
proteins, jointly assessing error, geometry, distribution descriptors, coverage,
spread and time. Oracle coarse and scrambled-AA outputs remain diagnostics.
All outputs are retained, and both reserved test proteins and R3 coordinates
remain unopened. Stronger conditioning is a hypothesis, not a promised gain.
