Fine-only flow and exact geometry pilot
=======================================

Question: does geometric supervision improve fine generation when coarse
sampling and the structural representation are held fixed?

Initialize both arms from generator_curriculum_v1/interleaved, seeds 2026/2027,
fixed 12,288-update endpoints. Duplicate the generator: retain one immutable
copy for coarse, train the other copy only through its fine velocity field.
At epoch zero both sampling paths reproduce the parent exactly. The frozen
ESM3 sequence feature bank is shared; our structural encoder/decoder/book and
all normalization buffers remain frozen. Inference flow NFEs remain 16+16,
though storing separate coarse/fine generator weights increases model storage.

Budget per arm and seed: 2,048 optimizer updates, four examples each (three
uniform PDB training proteins and one uniform MD training protein/frame).
Same initial weights, examples, Gaussian/time draws, AdamW settings and cosine
LR 1e-4 to 1e-5. Fresh optimizer in both arms. No best-checkpoint selection.

L = mean fine velocity MSE + lambda * geometry(full sampled fine rollout)

lambda = 0 or 0.01, fixed before training. Geometry is C-N squared hinge
outside [1.1,1.7] Angstrom, plus the same hinge averaged over block boundaries,
plus 0.1 CA-clash squared hinge below 3 Angstrom. One batch example (slot update modulo 4) gets an independent full 16-step
Euler rollout. Both arms execute its complete forward/backward graph, including
weight zero in the control. Activation checkpointing saves memory and recomputes
velocity forwards in backward; no prefix is detached. Actual
runtime is still reported. These are geometric proxies, not physical energies.

Teacher flow labels come from the same native fine/coarse training pair.
Geometry is measured on a complete sampled fine trajectory, with native coarse
as condition. This does not remove exposure bias from generated coarse.
The decoder is called directly with autograd enabled; no quantization and no
straight-through gradient estimator. It uses no reference geometry in sampling.

Final evaluation: unchanged two reused development proteins, all 64 raw samples
per protein. Report paired accuracy, geometry, spread/coverage, latency and both
seeds. Parent outputs are the fixed before-training reference. True-coarse and
eight fixed PDB training probes are diagnostics. No test or R3 coordinates are
opened, no samples rejected, no reference-based acceptance or automatic model
replacement. Failure or diversity loss is reported along with any CN gain.

The first endpoint-only integration is retained as endpoint_integration_check.json.
Its teacher endpoints had negligible geometry penalties despite poor rollout
geometry; this motivated full-rollout supervision before protocol freezing.
