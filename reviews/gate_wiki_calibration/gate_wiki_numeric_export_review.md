# T18 safe numeric export and standalone audit readiness

Passed for the completed 73-file numeric export described by `manifests/wiki_calibration_numeric_export.json` SHA256 `05ca982a1c57ca41704ce44ad8a2aa6390719952f07c5bb6647fd3d24262ac40`.

The independent export comparison streams private raw shards and exported NPZs. All26,024 arrays /12,100,216 numeric elements agree exactly:25,768 main/suffix/norm arrays and256 calibration/map/direction arrays. The40 JSON files contain prescribed metadata, numeric statistics and runtime fingerprints. Source text, token IDs, predicted IDs, source-reference raw Q/K/V, full hidden/logit arrays, credentials and local private paths are absent. Q/K/V-named entries in runtime ledgers are strictly fingerprint records, not donor tensors. The removed raw source-reference field remains in local raw evidence. Earlier exporter failures do not alter model results.

Deliverable: `superweight-causal-circuits/scripts/audit_wiki_calibration_results.py`, SHA256 `7ef860784f7caa512037eeb9efb6c81f173265738d410c02b2a4214960b91e97`. It uses only NumPy and Python standard-library modules. From the repository:

```sh
python scripts/audit_wiki_calibration_results.py
```

It verifies the exact file/index/array/key universes, hashes, model/revision/parameter/readout/map identities, reviewed layout-adapter metadata, score bounds, shapes/dtypes/finiteness, 72 all-target copies,768 saved suffix comparisons, recorded runtime checks, and nonnegative/equal-matched norms. It recomputes4,032 summary metric groups,912 paired contrasts with their exact seeded bootstrap,162,432 norm reductions and32 fresh calibration covariance/eigenvalue checks. The exported map digests are independently reconstructed from numeric arrays. Maximum NumPy reduction discrepancy is5.3291e-15 within the predeclared2e-12 audit reduction bound; bootstrap results match exactly; fresh-fit maximum error6.0397e-14.

Final actual-export audit took6.28s with119,668KiB RSS. Thirty adversarial fixtures reject missing/extra arrays, wrong shape/dtype, nonfinite values, same-mean copy/suffix permutations, unequal/negative norms, duplicate/missing summaries/contrasts/identities, changed views, omitted runtime guard fields, failed manifests, wrong model/revision/hashes, changed bootstrap/contrast scope and changed layout metadata. They deliberately update export file hashes where appropriate so semantic rejection is exercised. Temporary fixture files are removed. An initial fixture hardlink setup crossed filesystem boundaries and failed before semantic tests; its log is preserved, and the same tests subsequently passed using temporary paths on the original filesystem.

Evidence: `gate_wiki_export_independent.py/.json/.log`, `gate_wiki_portable_audit_guards.py/.json/.log`. No model/GPU or tokenizer work. This is a numeric-export audit, not a full pretrained replay, source-freshness retokenization, independent reconstruction of omitted attention/readout arrays or a signature authenticating the entire research provenance. The source inventory hashes document omitted raw/frozen files; those private bytes are not claimed to be supplied by this export. Standalone preparation/runner/package review is separate.
