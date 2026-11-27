Next bounded study: self-conditioning and mirror branches
==========================================================

Initially submitted on 2026-11-26 as Slurm job 58067270 on n002. This is a
fixed-weight sampling/trajectory experiment, with ZERO optimizer updates.
Four targeted tests and real-checkpoint preflight passed. The normal policy
replays saved outputs with maximum absolute difference 4.10e-5 Å (tolerance
1e-3 Å / 1e-5 relative). All parameters/buffers remain unchanged. The protocol
is frozen with 121 hashed inputs and all six full-sampling workers have started.
No quality result is claimed.

Strict replay stop and recovery
-------------------------------

The first attempt stopped when the axial-2029 / 1ef1_D always-policy replay
failed its original tolerance for 1 of 4416 coordinates; maximum absolute
difference was 0.00118351 Å. All 2176 generated outputs and the failed log are
preserved. No incomplete-attempt quality score was used to redesign the probe.

Recovery job 58067466 is pinned to n041, the completed parent study's hardware.
Before the full probe, it checks the previously failing field/protein against
the frozen native sampler EXACTLY, and against saved parent outputs using the
UNCHANGED 1e-3 Å / 1e-5 relative tolerance. It then reruns all six fields with
the same checkpoints, keys, schedules, batch size, NFE and primary criterion.
The frozen numeric functions are reused without changes. A documented I/O-root
override writes into artifacts/par_self_condition_probe_recovery_v1. The copied
protocol must be byte-identical to the original. Three workers run per wave
with two Torch threads each; two waves fit n041's available memory.

The n041 precheck has now passed: traced and native outputs are exactly equal,
and maximum difference from the saved parent is 0 Å for the previously failing
field/protein. Recovery sampling is running. Read-only independent audit job
58067508 is submitted with dependency afterok:58067466; it checks all recovery
outputs, the unchanged protocol and preservation of the original partial run.

Raw first-attempt and recovery outputs are kept separately, not pooled as extra
independent samples. The recovery manifest records original output hashes,
the original log, wrapper hashes and allocation changes. A replay failure does
not establish a quality effect or prove the hardware is its unique cause.

* Recovery source: upstream/experiments/recover_par_self_condition_probe.py.
* Recovery Slurm: upstream/scripts/par_self_condition_probe_recover.sbatch.
* Recovery setup/provenance: docs/self_condition_probe_recovery_2026-11-26.json.
* Later audit: upstream/experiments/review_par_self_condition_probe.py accepts
  --root artifacts/par_self_condition_probe_recovery_v1 after full completion.

Question
--------

Does recurrent self-conditioning reinforce early mirror-like coarse predictions?
Earlier parent trajectories show reflection-preferred clean predictions becoming
persistent early in integration, but do not establish self-conditioning as the
cause. A fixed intervention tests sensitivity to this recurrence without changing
weights, architecture, loss, noise or model-call budget.

Controlled design
-----------------

Reuse ALL SIX final fields from the completed chiral-velocity study: native,
polar-residual and cross-product-residual models for training-draw checkpoints
2029 and 2030. No best-model selection, retraining or extra seed is introduced.

For every field, sample the SAME two TRAIN proteins with the SAME 64 parent
Gaussian keys per protein. Each policy makes exactly 16 flow calls per sample:

* always: feed the previous raw clean prediction at steps 1..15, matching the
  original sampler. At step 0 there is no previous prediction.
* never: omit the self-conditioning input at all 16 steps.
* delayed_half: omit it at steps 0..7; feed the previous clean prediction at
  steps 8..15. The half-time boundary is fixed; no delay scan.

All policies compute clean predictions and record identical types of trajectory
data. There are 2304 requested outputs: 6 fields x 2 proteins x 3 policies x 64.
All 768 always-policy outputs must replay saved parent outputs within 1e-3 Å /
1e-5 relative tolerance, with the same batch size 32. Failure stops the worker.
Every parameter and buffer, including previously trainable adapters, is frozen.

Predeclared decision
--------------------

Primary: delayed_half minus always proper weighted-aligned coarse macro RMSD
in the AXIAL fields must be negative for BOTH existing training-draw checkpoints.
Also report the mean <= 2 Å fitting criterion separately for each protein.
This is directional consistency, not significance or new-protein validation.

Never-SC and the native/polar fields are secondary controls, not alternative
primary policies to choose after seeing scores. Report every field, protein,
policy and checkpoint. No reference gate, reflection correction, rejection,
sample filtering or adaptive extension. No automatic default promotion or MD
expansion, even if a descriptive result looks favorable.

Save all 17 noisy states and 16 clean predictions in native nm, SC flags and
weighted-centered final coordinates in Å. Reference coordinates are loaded only
in the report after every model worker finishes generation. Time-resolved
reflection-fit counts and paired branch-switch decompositions are descriptive;
changing self-conditioning can alter shape as well as handedness.

Scope and sources
-----------------

These are two repeatedly used training structures, one fixed MD frame each,
and previously used sampling keys. This is neither independent protein testing
nor a thermodynamic ensemble assessment. The prior chiral-velocity 4.77%/9.99%
training-pair gains remain separately documented and do not imply this new
sampling change will improve quality.

Sources under upstream/experiments:

* par_self_condition_probe.py: equal-NFE policies and full trajectory capture.
* run_par_self_condition_probe.py: real-model replay, freeze, all fields, report.

Tests: upstream/tests/test_par_self_condition_probe.py.
Slurm: upstream/scripts/par_self_condition_probe.sbatch.
Selected setup/hashes: docs/self_condition_probe_setup_2026-11-26.json.
Local raw output: artifacts/par_self_condition_probe_v1 (not committed).
