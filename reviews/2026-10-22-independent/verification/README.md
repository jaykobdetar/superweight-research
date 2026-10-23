# Verification materials for the independent repository review

Reviewed commit: e59cfed2df13a1d3a6e6c607648a9d6844790654
Tree: ca708f97aca9f051a2a798b22036acd683c43a7a
Remote main matched at acquisition and on 22 October 2026 at 20:17 UTC.

This package contains review-generated scripts/results, the public Git tree and CPU-audit receipts. It does not contain model weights, private inputs, raw activations or a new model replication. The repository was unchanged by the review. A fresh hash check after the review confirms all 2,364 committed files still match.

Review environment: Python 3.12.14; NumPy 2.3.5; CPU only; one numerical thread. Original project environment differs and is documented in the repository. All eleven supplied public numerical audit entry points tested passed in the review environment. The project auditors often reuse their own frozen statistical helpers; independent_checks.py deliberately does not import them.

## Reproduce the independent checks

Create a directory named snapshot beside this README and checks/, containing the public repository at the exact reviewed commit. For example, in a new workspace:

    git clone https://github.com/jaykobdetar/superweight-research.git snapshot
    git -C snapshot checkout --detach e59cfed2df13a1d3a6e6c607648a9d6844790654
    OPENBLAS_NUM_THREADS=1 python checks/verify_snapshot.py
    OPENBLAS_NUM_THREADS=1 python checks/independent_checks.py
    python checks/check_head_selection.py

These scripts read the snapshot and create only review results under checks/. They do not load models or run model forwards. independent_checks.py replaces its own JSON review result file if rerun; preserve the supplied results elsewhere if needed.

## Key files

- tree.json: the immutable recursive Git tree used for file verification.
- snapshot_verification.json and checks/snapshot_postreview_verification.json: before/after integrity receipts.
- checks/independent_checks.py and .json: independently authored raw-scalar reductions, primary contrast/bootstrap reconstruction, exhaustive 256-resample bootstrap, leave-one-out sign sensitivity and selected OLMo window reaggregations.
- checks/check_head_selection.py and independent_head_selection.json: development-only H14 ranking reaggregation.
- checks/audit_*.json and .log: executions of the project's eleven supplied public CPU-only auditors. These are not independent model acquisitions.
- checks/runner_summary.json: commands' success status and timings.
- checks/report19_metadata_validation.json: standard-library metadata validation, not input or model replication.
- checks/python_syntax.json: all 412 Python sources parse.
- checks/review_link_validation.json: all 44 commit-pinned repository references resolve to local files and valid line ranges.

The sign-flip reference statistic in independent_checks.json is descriptive and would require a symmetry/exchangeability assumption; it is not presented as a randomized-design significance test. Exhaustive empirical-bootstrap intervals coincide with the original 2,000-draw intervals for all 64 primary contrasts. Leave-one-out sign changes describe fragility, not a correction to or rejection of the reported means.
