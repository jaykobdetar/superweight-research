# Working with this repository

A directory that contains Python is not necessarily a software package to upgrade.
Many programs here are **evidence of exactly what was run**: their bytes are
named in a manifest, a review, a protocol, or a published reproduction. Change a
launcher when you want a better interface. Change an intervention or analysis in
a *new, separately identified* study when you want to answer a different question.

## Pick the right starting point

| I want to… | Start here |
| --- | --- |
| Understand a claim or its counterexamples | [Reports by model](README.md#model-specific-reading-routes), [OLMo claim-to-evidence map](olmo/CLAIM_EVIDENCE_MAP.md), [coordinate correction](COORDINATE_ERRATUM.md). |
| Check the bytes I have | Run `olmo-files` below. It reads ten named OLMo package manifests. Other releases retain their own inventories; use the corresponding study guide. A hash match is not a numerical check. |
| Check arithmetic on published observations | Use the catalog below. Each study keeps its own tolerances, seeds, comparison populations and scope. A passing reaggregation is not an independent replication. |
| Replay inference | Start with the relevant [model guide](README.md). The [nine-batch OLMo runner](REPRODUCING.md), [transfer runner](TRANSFER_REPRODUCING.md), and [fixed-H14 local replay](../research/olmo/fixed_h14_susceptibility/REPRODUCE.md) have different inputs, checkpoints and runtime assumptions. None is triggered by the catalog. |
| Inspect how a result was checked at publication | Read [independent snapshot review instructions](../reviews/2026-10-22-independent/verification/README.md). They target a specific commit, not current main. Separate [October publication cutoffs](olmo/PUBLICATION_SCOPE.md) and the [November experiment trail](../research/olmo/fixed_h14_susceptibility/TRAIL.md) preserve negative and incomplete results. |

## Run checks without touching the evidence

Use Python 3.11+ on a POSIX system; the archived numerical scripts use
`resource`. Listing checks and running the three `olmo-files`, `olmo-summary`
and `olmo-h14` checks need only the standard library. For the other eleven,
use the [quick-start environment](../README.md#quick-start) with NumPy 1.26.4.
From the repository root:

```sh
python -B scripts/audits.py list
python -B scripts/audits.py run olmo-files olmo-h14 --out-dir ../audit-first-look
python -B scripts/audits.py run all --out-dir ../audit-all
```

Pick a **new directory outside the checkout** each time. There is no automatic
install, data acquisition or model loading. The launcher runs one auditor at a
time, with assertions enabled and one numerical-library thread. For each check
you get its original JSON report and a log; there is no new common score. If
one fails, the command exits nonzero, retains its diagnostics, and does not run
later checks. A partial file after failure is not a passing result.

The list is intentionally explicit. Some historical scripts write into their
package, some require private inputs, and several require a checkpoint. For
example, the [additive readout audit](GPT2_GATE_ADDITIVE_REPRODUCING.md) needs
both a run and a checkpoint. Follow package instructions in a disposable copy
for those routes. Do not recursively execute every file named `audit` or
`test`.

## Where to put a change

| Surface | Role | Maintenance rule |
| --- | --- | --- |
| [scripts/](../scripts/) | User-facing auditors, replay preparation and report builders. | Put shared **launcher** behavior here. Some older launchers are hash-referenced themselves. Check exact file digests in manifests/reviews before changing one. Keep existing commands working. |
| [experiments/](../experiments/) | Frozen OLMo causal/continuation work and cross-model transfer protocols and code. | Do not retrofit new logic into an old protocol. The source export records transformations as well as digests. |
| [results/](../results/) | Observations, failures, and self-contained reproduction snapshots for later reports. | Code copies inside a reproduction bundle are part of its identity. Similar-looking modules do not imply interchangeable inputs or formulas. |
| [research/olmo/](../research/olmo/) | Packages for developmental, signal-production, native-use, optimizer, and fixed-H14 work. | Start with the package's manifest and reproduction guide, if present. A metadata receipt is not a model result. Keep later studies separate from historical cutoffs. |
| [manifests/](../manifests/), [reviews/](../reviews/) | Input identities, release inventories, independent checks and historical receipts. | Never replace an expected hash or weaken a check just to make a rerun pass; diagnose the mismatch. |
| [docs/](README.md), [reports/](../reports/), [figures/](../figures/), [research log](../RESEARCH_LOG.md) | Interpretation, publication snapshots and navigation. | Edit current navigation; keep earlier reports, figures, limitations and corrections traceable to their publication. |

## Maintaining the catalog

The registry in [audits.py](../scripts/audits.py) records the executable and
its *actual* output convention (`--output`, `--out`, `--report`, or JSON on
stdout). Only add an entry after verifying that it uses public local inputs,
does not update the evidence, produces a JSON object and exits nonzero on
failure. Keep audit-specific numerical code in its original package. A manifest
verifier checks listed files, not the authenticity or completeness of the
manifest itself.

Run the standard-library contract tests before the archived data checks:

```sh
python -B -m unittest discover -s tests -v
(cd research/olmo/fixed_h14_susceptibility && python -B -m unittest -v test_audit)
```

When modifying a launcher, compare its reports on the same checkout and
environment before and after. Wall time and peak memory can vary; a changed
scientific field, file digest, tolerance or denominator requires explanation.
