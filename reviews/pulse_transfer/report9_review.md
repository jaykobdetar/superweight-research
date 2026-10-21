# Independent scientific and presentation review: Report9

**Verdict: ready as a report of the three completed, audited follow-up batches.** This review found no remaining numerical or scientific-claim blocker. It does not certify the separate sanitized export or a full reproduction run, and no external action was performed.

Reviewed `REPORT9.md`, `report9.html`, `report9_value_findings.md`, `build_report9.py` and all four PNG figures against the independent 776-forward layer-write and 616-forward K/V audits performed by this reviewer and the separate reviewer's 520-forward value-geometry audit. The machine-checkable integration audit is `report9_consistency_audit.py` (`report9_consistency_audit.py`, private raw bundle), with final artifact hashes in [`report9_consistency_audit.json`](report9_consistency_audit.json). It used approximately 25 MiB CPU RSS, with no model or GPU.

All **144 displayed perplexity-ratio cells**, three resource rows, total **1,912 forwards/records** and **120.54 measured worker seconds** match the saved summaries/manifests. The HTML body exactly matches the Markdown rendering; all four embedded PNGs match the source figure files byte-for-byte. Figures were also visually inspected, including the corrected causal diagram and layer-write labels. Axis meanings, log scales, natural/BOS separation and saved uncertainty bars are consistent with the reported quantities. The HTML contains a viewport setting, horizontally scrollable tables, self-contained embedded images and no script. This is source/figure QA, not a browser/device-specific layout test.

The report correctly integrates the layer-3 result as a conditional two-coordinate source-write dependency. It does not imply deletion of the whole MLP, native-trajectory restoration, or whole-network sufficiency. The final-MLP nontransmission result is distinguished explicitly from layer-11 value transmission. Both start conditions and the strong BOS sensitivity appear in the tables and figures.

The K/V section preserves exact architectural closures and treats duplicate recipient/diagonal arms as controls rather than additional samples. Value simple effects, zero-value versus source-block comparisons and zero measured mask-reference loss drift agree with the independent numerical review. It states that keys and interactions depend on corpus/start; no key-independence or additive-mediation claim was found. Fixed source keys are not misrepresented as frozen downstream queries or attention gates.

The new norm/direction claims agree with the separate numerical audit: all four simple effects are positive on all 32 pairs; the signed interaction is positive in **31/32**, with the negative Wiki passage retained; all eight group intervals are positive. The report distinguishes a joint head/layer norm profile from uniform magnitude enlargement, including the layer-11 head-8 exception. It correctly treats the two signed-permuted controls as separately transformed source vectors, rather than a common random direction at two scales, and reports their negative results. No head-specific causal attribution is inferred from the descriptive norm pattern.

Four requested clarity corrections were implemented and rechecked before the final hashes were frozen:

1. The main norm/direction cells explicitly use a **B recipient with E source keys**.
2. The signed interaction is defined as `NLL(DE,NE) − NLL(DE,NB) − NLL(DB,NE) + NLL(DB,NB)`.
3. The B-direction/E-norm intervention's limited native-prediction agreement is stated: **79.71–88.77% natural, 70.00–80.29% BOS**. Approximate loss preservation is not described as native-output restoration.
4. The causal diagram and layer legends specify the selected source-coordinate writes; they no longer suggest that the entire layer-3 MLP is blocked.

Native C zero-value results also retain their approximation caveat and agreement ranges. The source-block comparison is in its own earlier-panel section; the report does not silently subtract results from different panels. The report remains explicit about teacher-forced scoring, n=4 paired descriptive intervals, adaptive sequential research, a single checkpoint/context length, artificial lesions, no multiplicity adjustment, saved-logit/audit limits, and no demonstrated efficiency gain or complete understanding.

Novelty wording is appropriately constrained. The earlier independently checked Report8 attribution review is retained as the source-verification record for the prior-work statements; this integration review does not claim a new exhaustive literature search. Report9 credits exact coordinates, compensation and intervention precedents, identifies the pinned WDD author-code revision, distinguishes code inspection from independent reproduction, and discloses unavailable full-PDF access. It expressly states that a changed estimand does not establish priority. No claim of discovering GPT-2 massive coordinates, self-repair, explicit K/V biases or a unique general sink mechanism was found.

The denied/unexecuted private repository action remains accurately described, with no workaround or credential change. Export hygiene and the forthcoming portable follow-up replay require their own review before expanding this verdict to package completeness or successful reproduction.

Final reviewed report identities:

| File | SHA256 |
|---|---|
| `REPORT9.md` | `6e2698cd964e9682cd89407ab6b05a83d0d16caa683e76eb4a4af34557f2fdfa` |
| `report9.html` | `fd66c7d25301201716664958d61922d30586f43987e0cf0a8cb37fedd54cbcdf` |
| `report9_value_findings.md` | `c4b5b18f32b2d6366948b73a24e067feb382aeaf6cc089c7564e42e1cf41975f` |

If these files change, rerun the consistency audit and review any altered claims before citing this frozen verdict for the changed artifact.
