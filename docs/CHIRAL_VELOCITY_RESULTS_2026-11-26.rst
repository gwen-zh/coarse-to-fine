Chiral velocity residual: reviewed results — 2026-11-26
======================================================

本轮六组已完成：交叉积残差在两个预设训练抽样 seed 下都优于原速度场和同参数量
普通残差，达到方向一致性要求。相对原速度场改善 4.77% / 9.99%。这是两个反复
使用的训练结构上的 coarse 诊断；2 Å 拟合门槛未达到，尚无新蛋白/整体模型收益。

Decision
--------

Job 58007793 completed all six fixed runs, 1024 updates each. Independent review
verified every one of 88 frozen inputs, all 6144 training updates, checkpoint and
Adam counters, paired random events, and all 896 requested outputs. Every output
was finite and retained. Explicit coordinate alignment reproduces the reported
means within 5.33e-14 Å. The corrected presentation figure was visually inspected.

The prespecified directional criterion PASSED: axial < native AND axial < polar
proper coarse macro RMSD for BOTH training-draw seeds. This is directional
repeatability, not statistical significance or independent generalization.
The two-protein per-target mean <= 2 Å fitting criterion remains unmet in every
arm. Defaults were not promoted. At review completion no next experiment was
submitted. The subsequent fixed-weight SC probe is documented separately in
SELF_CONDITION_PROBE_2026-11-26.rst (job 58067270, zero optimizer updates).

Primary results
---------------

These are weighted, proper-rotation-aligned errors of 23 coarse block centers,
averaged over 64 samples per protein, then over TWO TRAIN structures. They must
not be compared directly to another cohort's full-chain test RMSD.

.. list-table:: Proper coarse macro RMSD (Å)
   :header-rows: 1

   * - Training-draw seed
     - Native control
     - Polar control
     - Cross-product residual
     - Reduction vs native
   * - 2029
     - 3.35349
     - 3.33431
     - 3.19363
     - 4.77%
   * - 2030
     - 3.66227
     - 3.58104
     - 3.29638
     - 9.99%

Relative reductions against the parameter-matched polar residual are 4.22% and
7.95%. The polar and axial arms each have 548,358 trainable parameters; native
has 513,152. Both residuals add the SAME initialized 35,206-parameter scalar
network and start with exactly zero output. All arms use the original velocity
MSE, the same parent parameter/Adam warm start and paired training random events.

Where the effect occurs
-----------------------

.. list-table:: Per-protein native -> axial proper RMSD (Å)
   :header-rows: 1

   * - Seed
     - 1b2s_E
     - 1ef1_D
   * - 2029
     - 3.90647 -> 3.90384
     - 2.80051 -> 2.48341
   * - 2030
     - 4.06350 -> 3.90980
     - 3.26105 -> 2.68296

Benefits concentrate on 1ef1_D; the first protein's 2029 gain is only 0.00263 Å.
Do not describe a uniform benefit across proteins.

Negative-determinant unconstrained orthogonal fits decrease from native 41 to
axial 38 of 128 outputs for seed 2029, and 47 to 40 for 2030. These are preferred
reflection-permitted alignment branches, not a blanket exact-mirror label.
Mirror-like outputs remain common. The paired posthoc decomposition associates
much of the improvement with branch changes, but does not prove a causal
mechanism. Reflection-permitted shape error does not improve uniformly.

Scope and next decision
-----------------------

Both proteins are repeatedly used, 90-residue TRAIN targets with one fixed MD
frame each. Seeds change training Gaussian/time/rotation/self-conditioning draws,
not the common parameter initialization. No development/test/R3 coordinates or
new proteins were evaluated. This is not thermodynamic ensemble evidence or
overall superiority to Hierarchical ConfGen. It does not retroactively validate
the failed signed-volume auxiliary-loss family.

The scalar inputs and vector bases differ geometrically; equal parameter count
does not isolate a unique chirality mechanism. The residual's transformation law
does not establish whole-model equivariance or preserved ensemble probabilities.

Close this bounded study. Before expanding MD training, separately specify how
to diagnose the remaining mirror modes and achieve reliable training-pair
fitting. Do not extend this frozen study with extra outcome-triggered seeds,
checkpoints, weights or basis scans. Continue reporting the independently
confirmed original-model search result separately.

Reproducibility
---------------

* Frozen numerical driver: upstream/experiments/run_par_chiral_velocity.py.
* Independent audit: upstream/experiments/review_par_chiral_velocity.py.
* Presentation plot: upstream/experiments/plot_par_chiral_velocity_review.py.
* Public selected aggregates/provenance: docs/chiral_velocity_results_2026-11-26.json.
* Original fixed design: docs/CHIRAL_VELOCITY_STUDY_2026-11-24.rst.

Full checkpoints, traces, frozen protocols and per-output alignment audits remain
under the shared/local artifacts/par_chiral_velocity_v1 tree. The original plot
is preserved; the separately named Reviewed plot fixes its display limits and
label placement. Raw artifacts, models and figure binaries are not uploaded.
