Generator PDB pretraining with a matched curriculum control
=========================================================

Question
--------

Frozen ESM3 sequence features helped the small generator modestly, but new
sequence quality remained poor. Test whether an explicit static-structure
pretraining phase helps, while separating it from simply using more updates
or more PDB examples.

Design
------

Both curricula start fresh from the same two seeds, with the same 1,088,779
trainable parameters, ESM3 feature cache, structural encoders/decoder, labels,
normalization and Euler16 coarse/fine samplers. They each use exactly 12,288
optimizer updates, 36,864 PDB examples and 12,288 MD examples:

* PDB_then_MD: 8192 updates of four PDB examples, followed by 4096 updates of
  three MD examples and one PDB example.
* interleaved: 12,288 updates of three PDB examples and one MD example.

PDB samples are uniform draws from the original 899 training proteins. MD
sampling is uniform over the same five training proteins and then their frames.
Pools are fixed per seed before training. Each occurrence is a unique event;
its sample identity and flow-noise/time seed stay fixed regardless of where
that event occurs in a curriculum. Final reports verify equality of every
event, target and draw, plus initialization and LR at every optimizer step.
The intervention includes temporal ordering and minibatch domain composition.

Both use the same two-part LR schedule: cosine 3e-4 to 3e-5 over the first 8192
updates, then cosine 1e-4 to 1e-5 over 4096 updates. Adam state is retained at
the boundary in both arms. This is a new matched-budget trial, not an extension
of previous checkpoints or a direct equal-compute comparison with 4096-step runs.

Readouts
--------

Save fixed checkpoints at 8192 and 12,288 updates; no epoch selection. An
eight-PDB-training-protein probe (four independent samples each) measures
training-set structure learning at the first endpoint. It is diagnostic and
does not select whether to adapt or which checkpoint to report.

Final evaluation uses the same two new-sequence development proteins, raw
samples, reference/coverage errors, geometry, spread, RMSF, fixed-bin JS and
runtime. Oracle coarse and scrambled-AA remain diagnostics. These repeatedly
used development proteins are not independent confirmation; ESM3 pretraining
exposure is unknown. Neither reserved test structures nor R3 coordinates are
opened. No automatic larger run or model replacement follows this trial.
