AA-only shared Transformer: explicit coarse-to-fine prediction
============================================================

Status: implementation with synthetic interface/gradient tests only. Not trained,
not adopted, not a measured quality improvement. Existing flow/masked generator
and frozen structural checkpoint files are unchanged.

Architecture
------------

AA IDs -> learned AA embedding + position encoding -> one shared Transformer
-> residue states h[N,d].

Coarse: mean-pool each group of up to4 residue states -> coarse head -> tanh
continuous z_c[ceil(N/4),16].

Fine: broadcast each predicted coarse vector to its corresponding residues;
concatenate h_i with that predicted vector -> fine head ->1024code logits/residue.
The fine vector dimension remains8 in the current frozen tokenizer. This network
predicts IDs and does not modify the structural codebook.

Fine cross-entropy gradients reach the coarse head without detach. The real
coarse label is used only for the supervised coarse loss, never as the fine
head's forward condition. Coarse targets are normalized using fixed TRAIN-only
per-channel statistics for loss scaling. Input sequence representation is learned
here; no ESM features or reference coordinates are forward inputs.

Supervision
-----------

Keep the existing structural encoders, decoder and fine codebook frozen.
Generate coarse/fine targets with that matched tokenizer. Train coarse normalized
MSE plus fine token cross-entropy. This is a deterministic joint point predictor;
no Gaussian prior, flow solver or sampling mechanism is introduced. A single
shared trunk with two connected heads still requires new training.

Synthetic tests verify tensor shapes, correct tail-block broadcasting, coarse
intervention sensitivity, fine-loss gradients through the coarse head, absence
of true-coarse forward leakage, checkpoint restore, and frozen decoder
compatibility. These tests establish wiring only. After training, verify whether
coarse conditioning is used and useful with same-AA coarse interventions and
matched no-coarse controls; evaluate reference accuracy, geometry and compute.
Multiple-conformation coverage requires an explicit later stochastic design.
