# Report13 numerical export: prospective static review

This is a pre-result review of the NumPy reader and exporter. It does not interpret the running experiment or certify an actual exported package. No production files were edited by this reviewer.

Reviewed revisions: `audit_additive_results.py` SHA256 `a9d826e7e947ca48ef8c7ebad35b69d65e18379faa1aa7be1729c2fe02906101`; `export_report13.py` SHA256 `d1ff67604eae575117ccc877b94d0eaea4212afeffd49c86b5a0e810c5cc786b`.

The explicit metric universes agree with the protocol: 1,000 primary/anchor records, 248 full and 496 segmented dual-reference summaries, 864 contrasts/216 contrast groups, 96 FP32 nonadditivity records/24 groups, 96 additive family ledgers, 288 additive window views and 1,440 metric groups. Primary exported arrays are scored queries1..510; early array slice0..125 corresponds to queries1..126 and later slice126 onward to queries127..510. Unsliced additive arrays retain query0; their primary slice1..510 and early/later slices match. The paired bootstrap seed, 2,000 resamples and four windows per group match the author analyzer. Starts and score views remain correlated; no cross-start independence or additive-mediation claim is supported.

The FP64 formulas for J, R and joint-minus-summed-singletons, and their shared-bootstrap identity, are coherent. The revision now binds each FP64 P/joint/singleton loss vector to the matching scored FP32 ledger under the declared1e−4 drift guard. Additive index names and hashes are tied to the raw file inventory. It also enforces the exact29-file numerical export universe and checks array names/dtypes/shapes. This closes the four initial issues reported to the author: internal-only additive identities, literal token IDs retained by sanitization, missing frozen-source binding, and stale/unaccounted export members.

Remaining requested refinements before package readiness:

1. The NumPy reader currently validates completed/passed statuses but not the associated1992/1000/992/0 counts and model/calibration closure metadata. Add explicit consistency checks before describing resource/count claims as audited.
2. The exporter currently includes every Tensor found in each additive family dictionary. The author analyzer enforces the expected raw dictionary schema, but an explicit field/shape/dtype allowlist at the export boundary is clearer and protects the stated numerical-only contract before writing files.
3. The review-file copy uses substring exclusions, not a structured privacy schema. Final package review should inspect the actual copied review inventory, confirm no stale unexpected review files remain, and check local links. This is a packaging checkpoint, not a claim that a leak has occurred.

An actual baseline export and adversarial copies remain to be tested after production completes and the author/independent raw audit permits export. No audit success is claimed for absent result files.

## Final pre-export delta

Rechecked `audit_additive_results.py` SHA256 `9f693b763859ea8315d631817b899d10719c10d187488b6c3f17baab14d917fe` and `export_report13.py` SHA256 `0044ac6ccf3c832dad9361dde1e0998044a62250daf0281076fa4511236b2399`. Requested refinements 1 and 2 above are resolved: the reader checks the exact forward/record/calibration counts, completed phase, model and calibration-map identities, unchanged parameters and precision settings; the exporter now permits only the declared 21 additive tensor fields with exact shape, dtype and finite-value checks. No new scientific or privacy blocker was found in these changes.

The remaining checkpoint is an inspection of the actual package, including copied/omitted review evidence and local links. The numerical adversarial fixture is ready at `report13_numeric_adversarial.py` SHA256 `6ea4a6c98a5db105b6838fcf84ef22afb8f1842397b0988255945c6f8f45a67b`. It tests an intact export, ten hash-refreshed semantic corruptions and an unexpected extra file, using only a temporary copy of numerical arrays. It has compiled, but has not been run because the completed export is not yet present. This note is not empirical export approval.
