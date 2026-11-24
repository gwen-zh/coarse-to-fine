Spatial multiscale ensembles: next controlled stage
===================================================

The completed one-protein model demonstrated conditional stochastic generation,
not generalization across AA sequences. Its direct-center continuous outlet had
lower nearest-reference error but narrower spread than the 16D coarse control.

The frozen solver probe compares Euler and Heun at 16 and 64 coarse velocity
evaluations. Fine stays Euler16, and old Euler16 outputs are replay-checked.
Higher integration accuracy changes spread substantially but can worsen geometry
and nearest-reference error; no solver is automatically adopted. The next model
trial keeps the previous Euler16/Euler16 solver as a fixed baseline.

New MD corpus
-------------

Four additional training proteins and two development proteins were selected
from the official AlphaFlow ATLAS split metadata, in lexicographic order, with
64-128 standard residues and archive size at most 256 MiB. Selection used only
sequence and availability, never generated quality or MD flexibility.

All selected sequences have screened global aligned identities below 40 percent
to the local PDB representation corpus, existing benchmarks, old MD protein and
previously selected proteins. This is not a CATH/fold split or a guarantee of
absence of local-domain homology. Two official test proteins are fixed by
metadata only, without downloading their structures.

New train proteins use R1 and R2 for training; new development proteins use R1
and R2 only for evaluation. All new R3 members remain compressed and unopened.
The old 1b2s_E pilot keeps R1 as train and R2 as a reused diagnostic. Its prior
R3 confirmation remains consumed and is not reused here.

Coordinates and representation
------------------------------

Read N/CA/C atoms in Angstrom, with trajectory stride 40 fixed in advance. Each
protein gets one reference from its first R1 frame, canonicalized once. Align
all other frames to that fixed reference before encoding. Do not recanonicalize
the moving first residue independently in each frame. This is a label convention;
the generator receives only AA and random noise at inference.

The frozen local-frame structure model supplies 8D continuous fine targets.
Coarse targets are actual block C-alpha means with ceil(N/4) positions. Encoders,
the codebook and decoder are not retrained in this stage. Continuous decoding
does not use the codebook. Every native reconstruction preflight case is retained;
if fixed readiness limits fail, preparation stops instead of removing proteins.

Generator trial
---------------

Both arms have the same small shared Transformer, separate conditional coarse
and fine flow fields, two fresh training seeds, 4096 updates, and four examples
per update:

* MD_only: uniform protein selection followed by a uniform frame, four times.
* MD_plus_PDB: same first three MD examples, with the fourth slot replaced by
  a static PDB training example from the original 899-protein train subset.

Per-slot RNG seeds preserve noise/time draws for the common three examples,
despite variable sequence lengths. Equal-protein normalization is fitted only
on train MD. PDB replay and development cases do not fit those statistics.
Replay changes the number of MD examples and sequence-length compute cost;
report actual elapsed time and keep this limitation explicit.

Primary readout is descriptive macro performance on the two new development
proteins. Retain all raw samples and report error/coverage in both directions,
spread, RMSF, geometry, distance/Rg distributions and wall time. Histogram bins
are fixed in physical units, with overflow; no validation-derived bin fitting.
These JS values are not directly comparable to previous adaptive-bin JS values.

Seen-protein results are diagnostics, not independent validation. Native coarse
is an oracle control. Scrambled-AA conditioning is an OOD sensitivity check and
does not prove sequence-specific correctness. No best-checkpoint selection or
test coordinate access is authorized by this protocol. Two new development
proteins remain too few for a broad generalization or publication claim.
