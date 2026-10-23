# Native OLMo-1B provenance addendum

This metadata-only follow-up is separate from, and does not modify, the completed twelve-checkpoint evidence.

Read `deliverables/native_provenance_addendum.pdf`. It records a newly located native dense checkpoint metadata release and exact remaining identity, access, optimizer, trainer and data-order prerequisites.

Run `python checks/audit_metadata.py` for an offline standard-library validation of model/Adam-moment shapes, model partition coverage, selected coordinate placement and HTTP receipt hashes. No model tensors are read, no Python checkpoint object is deserialized, and no network is used.

The acquisition scripts are historical records, not instructions to reattempt denied access. The HEAD availability script refuses repeat execution when its previous receipt exists. The sole HEAD probe returned 403; no alternate route, GET or range retry was made. Later planned probes were not executed. The earlier inferred S3 target was not retried.

The source archive includes the three public native metadata objects, a brief maintainer-announcement excerpt, public browser-view records and acquisition receipts. Full third-party article/discussion snapshots retained in the working evidence are not republished in this archive. Receipts may refer to these omitted research-only sources; their URLs and hashes preserve provenance.

No tensor, trainer-state or data-index payload is included. The native checkpoint is maintainer-labelled step954000; numerical equality to the HF conversion is not yet established. Source document transfer was 1,566,181 bytes, excluding browser assets. No further checkpoint acquisition or training is authorized by this addendum.
