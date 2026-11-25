Follow-up: matched chiral velocity residuals
===========================================

2026-11-24: Slurm job 58007793 was submitted and started on n041. Five targeted
module tests and actual-PAR-weight preflight passed. The protocol is frozen
with 88 hashed inputs. This is a setup/progress update; no final quality result
is available. The earlier signed-volume-loss family remains closed after its
failed fixed-seed replication.

Question
--------

Can a proper-rotation-compatible, reflection-sensitive residual improve
free-running coarse generation under the ORIGINAL velocity-MSE objective,
relative to both the native model and a parameter-matched polar residual?

The intervention changes the velocity architecture, not the training loss.
No reference-based reflection correction or sample selection is performed.

Fixed design
------------

* Native control: common warm-started AA adapter + last-two-block flow LoRA,
  513,152 trainable parameters.
* Polar control: same base plus a 35,206-parameter zero-output residual,
  548,358 trainable parameters total.
* Axial candidate: the exact same initialized scalar network and parameter
  count as polar, but using pairwise cross-product vector bases.
* Two fixed training-draw seeds, 2029 and 2030; one common parameter and Adam
  warm start. Six runs, exactly 1024 additional updates each.
* Same two repeatedly used TRAIN structures, one fixed MD frame each. This
  checks learnability, not unseen-protein or ensemble performance.
* Same velocity MSE, uniform time, paired random events and fixed local cosine
  LR in all arms. No auxiliary signed-volume/geometry objective.
* A common zero-init baseline plus 64 fresh fixed noise keys per protein at
  each final endpoint, 16 Euler steps. 896 requested outputs in total.

The scalar network sees AA-only conditioning, time, self-conditioning presence,
and reflection-even length/dot geometry. Polar bases are normalized left/right
chain edges and centered radial vectors, for both current and self-conditioned
states. Axial bases are their pairwise cross products. Both residual outputs
are centered. Smooth norm regularization handles collapsed/collinear inputs.

For the residual alone, proper rotations transform both bases consistently;
under reflection, the axial basis acquires an additional determinant sign.
This does NOT establish whole-model equivariance, guarantee correct handedness,
or preserve target ensemble probabilities. Basis magnitudes differ and are
logged; equal parameter count is not isolation of a unique causal mechanism.

Preflight evidence
------------------

* All three initial sampled outputs match exactly on both training targets.
* First-update native MSE and all random training events match exactly.
* Common parent Adam state, identical initial residual parameters and equal
  polar/axial counts verified.
* Actual residual output-head gradients are nonzero on the first update;
  hidden-layer gradients are nonzero on the second update.
* All six preflight optimizer updates are discarded. Official pretrained
  parameters remain unchanged.

Decision rule
-------------

After all six runs finish, compare proper-rotation-aligned coarse macro RMSD
across BOTH training proteins and ALL outputs. Directional consistency requires
axial to beat BOTH polar and native controls for BOTH fixed training-draw seeds.
Report each protein, seed, reflection-permitted fit and runtime regardless of
outcome. Also report the existing two-protein mean <= 2 Å fitting criterion.

No extra outcome-triggered seeds, weight/basis scans, training extension,
intermediate checkpoint ranking, full-MD expansion or automatic promotion.
This directional rule is not a statistical significance or generalization test.

Code and local outputs
----------------------

* upstream/experiments/par_chiral_velocity.py: residuals and original-MSE updater.
* upstream/experiments/run_par_chiral_velocity.py: preflight, freeze, baseline,
  six training runs, all-output report and figures.
* upstream/tests/test_par_chiral_velocity.py: five targeted tests.
* upstream/scripts/par_chiral_velocity.sbatch: bounded CPU allocation and stages.
* docs/chiral_velocity_setup_2026-11-24.json: selected setup/preflight summary
  and exact source hashes.

Local outputs are under artifacts/par_chiral_velocity_v1, including protocol,
preflight, common baseline, per-arm/seed traces/checkpoints and final report.
These raw artifacts and model states are not committed. Completed sources are
now frozen; use a new namespace for changes. The general PAR setup/replay
requirements in PAR_REPRODUCTION.rst also apply.
