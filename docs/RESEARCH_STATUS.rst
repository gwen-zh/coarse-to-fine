Research status — 2026-11-24
============================

The confirmed result is a weight-free, decoder-guided refinement method on the
original Hierarchical ConfGen path. The independently trained multiscale model
and sequence-conditioned PAR transfer remain research prototypes. There is no
confirmed end-to-end improvement over the published model, and no demonstrated
improvement in thermodynamic ensemble fidelity.

This update summarizes completed experiments; it does not announce a new run.
The latest reviewed numerical experiment is the signed-volume replication
reviewed on November 10 Eastern time / November 11 UTC. A live queue check on
November 24 found no active project training. Old PAR-transfer evaluation/report
jobs remain pending on failed or unresolved dependencies; they are not running
experiments. No chirality-sensitive architecture experiment has been submitted.

Confirmed original-model result
-------------------------------

Frozen-weight fine-token search was evaluated on a fixed, outcome-blind,
sequence-screened cohort of 64 proteins with 8 conformations each (512 total).
The cohort is independent of method development, not proven absent from ESM3
pretraining. It has now been consumed and must not be used for further tuning.

* Gated C–N bond outlier rate: 9.50015% -> 7.53809%, a 20.65% relative reduction.
* Ungated CA–C–N angle-error change: -0.14889 degrees, nominal 97.5% bootstrap
  interval [-0.18642, -0.11437].
* Ungated C–N–CA angle-error change: -0.15304 degrees, nominal 97.5% bootstrap
  interval [-0.19491, -0.11557]. Both prespecified angle endpoints passed the
  declared two-comparison correction. Neither angle was used to select outputs.
* Reference CA RMSD: 3.24891 -> 3.25096 Å; sample spread: 3.19169 -> 3.19545 Å.
  Difference intervals cross zero. This is not equivalence, noninferiority, or
  evidence of unchanged physical ensemble coverage.
* Mean edited fine-token fraction: 1.49237%. Recorded mean added search time:
  20.76 seconds per conformation; recorded base generation time: 6.03 seconds.
  Timing and hardware scope must accompany practical efficiency comparisons.

The C–N improvement is a direct target of acceptance and is not independent
evidence. The two ungated angle endpoints provide that evidence. Random-site
search has similar angle gains; targeted localization superiority is unconfirmed.
The method changes tokens, not pretrained weights. Its original 4096-token
semantics are separate from the new model's representations.

Current model and training/generation interfaces
------------------------------------------------

The older portable demo in scripts/sample_generator.py remains an archival
16-dimensional-coarse / 1024-fine-token model. It is not the latest prototype.

The later continuous model uses, for N residues and M = ceil(N/4):

* Coarse: M x 3 physical coordinates, each the mean Cα position of up to four
  consecutive residues. The older learned 16D coarse encoder is bypassed.
* Fine: N x 8 continuous, pre-quantization features encoded in coarse-derived
  proper local frames. The current generation path does not snap to the fine
  codebook. A 1024-entry EMA codebook remains an optional historical control.
* Decoder: coarse centers, local fine features and neighboring context produce
  proper per-residue rigid frames and fixed N/Cα/C templates. A zero-centroid
  constraint preserves each block's Cα mean. This does not enforce inter-residue
  peptide connections or guarantee valid generated geometry.
* Sequence condition: frozen ESM3 AA-only features H[N,1536]. The structure
  encoder/decoder are our trained models, not the ESM3 structure tokenizer.

Representation training reads known backbone coordinates. Generator training
reads sequences/features and known structure labels. Generation reads sequence
features and random noise only; native target coordinates are evaluation-only.

The own-model factorization is p(C,F|AA) = p(C|AA) p(F|C,AA): sample a coarse
field first, then sample compatible fine features conditioned on that SAME
coarse field, then decode. Multiple noise draws can produce an ensemble; this
alone does not establish ensemble accuracy or calibrated populations.

PAR branch
-----------

The current PAR pilot adds a zero-initialized, ordered-four-residue AA adapter
to a fixed public PAR model, retaining its native flow module. It generates
coarse coordinates for the own fine-flow/decoder interface. It has NOT yet
implemented or validated a complete sequence-conditioned PAR L/4 -> L/2 -> L
next-scale generator. In particular, do not describe this pilot as a full
reproduction of PAR's published scores.

The public checkpoint contains 117,107,160 frozen parameters. The AA adapter
has 464,512 trainable parameters; adding rank-8 LoRA in the final two native
flow blocks gives 513,152 trainable parameters in total. The official coarse
path uses interpolation and floor(N/4); our physical-block interface uses
means and ceil(N/4). Native PAR quantities are in nm; interface coordinates and
reported RMSD are in Å. These distinctions are tested and documented.

What the completed experiments establish
---------------------------------------

All RMSDs in the following PAR studies measure block-center coordinates on
TWO repeatedly used TRAIN structures (1b2s_E and 1ef1_D, 90 residues each,
one fixed MD frame each). They are not full-chain test RMSDs or ensemble scores.

1. Sequence conditioning is useful on the training pair. After 1024 updates,
   the uniform-time AA-conditioned arm has mean proper coarse RMSD 6.047 Å;
   the same-length, AA-blind arm has 14.195 Å. This is a learnability control,
   not new-sequence generalization.
2. A fixed continuation to 4096 updates reduces the AA-conditioned uniform
   arm from 6.047 to 3.746 Å on the SAME 32 sampling keys. Both proteins still
   miss the predeclared mean <= 2 Å fitting target. Increasing integration
   steps from 16 to 64 does not rescue the fixed diagnostic subset.
3. With 64 common sampling keys, adapter-only versus native-flow LoRA gives
   3.34128 versus 3.31968 Å after matched 1024-update continuations. The added
   benefit is only 0.02160 Å and changes sign across noise subsets. Both arms
   have 41/128 negative-determinant unconstrained orthogonal fits. This does
   not establish a useful LoRA upgrade or resolve mirror-like generation.
4. Auditing 4096 saved training draws finds no reflected target in the checked
   construction/augmentation path. Rotation/reflection probes do not establish
   exact O(3) equivariance or a causal explanation of the mirror branch.
5. Signed-volume auxiliary supervision gives an exploratory 3.64769 -> 2.97259 Å
   result in the original matched run (18.5% reduction), but the fixed new-seed
   replication FAILS its directional-consistency criterion:

   * Training-draw seed 2027: control 3.72671 -> candidate 3.19819 Å.
   * Training-draw seed 2028: control 3.70426 -> candidate 3.96524 Å.

   Seeds vary training noise/time/rotation/self-conditioning, with a COMMON
   parameter and Adam warm start. They are not independent pretrained-model
   initializations. The fresh-noise 2026 checkpoint anchor is exploratory and
   cannot be counted as another successful training replication.

For seed 2028, the first protein worsens by 0.62614 Å. Of this change, 0.59981 Å
is contributed by paired outputs whose preferred orthogonal-fit parity changes;
this is a descriptive decomposition, not proof of cause. Lower supervised
signed-volume loss does not reliably improve free-running geometry. All 1280
requested replication outputs were retained and finite. Neither new seed meets
the two-protein fitting target. Defaults remain unchanged; this objective family
is closed without adaptive extra seeds, weight scans, or full-MD expansion.

Earlier fine-conditioning interventions show that providing a better coarse
layout improves own-model output, but local geometry remains poor. The dominant
open issue is reliable sequence-to-coarse generation, including handedness and
coverage; reconstruction quality alone does not resolve it.

Data and interpretation boundaries
----------------------------------

The own-model corpus combines 899 static PDB training proteins with 2264 MD
frames from only five training sequences. Reused development and independent
confirmation sets have different roles. Frames, seeds, and conformations must
not be treated as independent proteins. Static PDB data are not equilibrium
samples. More stochastic samples do not by themselves establish thermal fidelity.

The new model's coarse training RMSDs cannot be compared directly with the
original-model search cohort's full-chain RMSDs. No overall superiority to
Hierarchical ConfGen has been demonstrated.

Next decision, not an active job
--------------------------------

Specify a separate, bounded chirality-sensitive architecture test under the
original velocity-MSE flow objective, with a parameter-matched control. Verify
zero-initialization replay, proper-rotation behavior, parameter counts and real
gradient paths before freezing a protocol. Report both fixed training targets
and every prespecified training seed. A cross-product residual is a hypothesis,
not a guaranteed fix or a guarantee of ensemble-distribution preservation.

Only after reliable training-pair fitting should a separately specified MD
generalization/coverage study be expanded. Preserve paired seeds, fair compute,
reference accuracy, local geometry, coverage/diversity and runtime checks. In
parallel, write up the independently confirmed original-path search result.

Evidence and handoff
--------------------

* docs/PROGRESS_2026-11-24_ZH.rst: Chinese chronology and concrete next steps.
* docs/progress_2026-11-24.json: selected aggregate numbers and source hashes.
* docs/PAR_REPRODUCTION.rst: pinned external code, dependencies and replay scope.
* docs/source_sync_2026-11-24.json: exact newly synchronized source-file hashes.

Only source, documentation and selected aggregate progress are shared by this
update. Full protocols, raw outputs, weights, cached features, trajectories,
manuscripts and figures remain in the local/shared artifact tree. Completed
protocol inputs are immutable; create a new namespace for any new experiment.
