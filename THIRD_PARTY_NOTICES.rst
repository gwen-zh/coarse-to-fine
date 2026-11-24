Third-party provenance
======================

Original implementation, vendored under upstream/:
https://github.com/vanha9/hierarchical_conformation_generation
Commit:7f0c92eec404f0a85cc03dbae834df8d0811ab9d
Paper:https://doi.org/10.1093/bioinformatics/btag611
Existing source headers/notices are retained. No top-level LICENSE file was
present in that checked-out snapshot. This repository does not assign a new
license to third-party code.

Optional pinned submodules retain their own licenses and data terms:

* microsoft/bioemu-benchmarks:
  63edbfa2ee5f489d66a4bece96f242a1d28e567d (LICENSE and DATASET_LICENSE).
* microsoft/protein-frame-flow:
  f50d8dbbdae827be291e9f73d732b61b195f8816 (LICENSE).
  A compatibility patch is included under docs/patches/.

The optional PAR coarse-transfer pilot pins two additional repositories:

* bytedance-Seed/par-protein:
  92d1c3ecc9822f897b66d53b3852059e6750aee2 (LICENSE).
* aqlaboratory/openfold:
  be2ec1841f16c966c65ae0e7599ebbadc725757d (LICENSE).

Both are recorded as submodules under third_party/. Their upstream source is
unmodified. Project-local, configuration-checked guards for inactive OpenFold
branches live in upstream/experiments/par_coarse_backbone.py; they raise if an
unsupported active branch is requested. This is not a replacement for active
OpenFold operators or permission to ignore the upstream projects' licenses.

PAR weights are obtained separately from the public ByteDance-Seed/PAR model
repository at revision 7fb609346f5d9a2baa76add1f387aee1bdb8874b. Neither model
weights nor training datasets are redistributed with this source update.
