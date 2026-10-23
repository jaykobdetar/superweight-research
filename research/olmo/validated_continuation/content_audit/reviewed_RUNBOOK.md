# Frozen content-donor assay

Inputs and contrasts were frozen before outputs. The parent granted the baseline-only CPU slot; its results are archived separately. Donor forwards and interventions still require a new CPU-slot grant. Every file written by this assay stays beneath this directory. The shared FP32 tensors and validated core runner are read-only.

## Design and frozen inputs

Eight reciprocal minimal pairs span color, pet, fruit, drink, key material, season, shape and shirt color. Each recipient pair differs in exactly one pre-newline token. All sequences start with token510 (`The`), contain exactly one newline token187 at p=10–12, and have identical query suffixes. The common appended A-label target is used for scoring both recipients, so even the appended suffix token does not differ. Correct and foil logits are selected independently of the observed target. Sequences are17–20tokens, well below the512token runner limit.

Frozen manifest: `frozen_manifest.json`, SHA256 `7c7afdea710322f8bfe287b0146d370637d5dce3baf6fcd8c86ff5a813f95ff1`. The exact tokenizer and core runner hashes are checked on every execution. The model is allenai/OLMo-1B-0724-hf at revision d7cbab742d80589e714b1a2d7f838dcd21cbe143, FP32, all50304LM-head rows.

The baseline stage archives every native, post-boundary mask0, maskp and mask0p result before donor tests. The explicit pre-output amendment is preoutput_amendment_01.json; the original manifest and protocol remain unchanged. Baseline gate: both reciprocal correct-versus-foil margins >0, separately for each world. Failed cases are never deleted or silently omitted. The default experiment runs all eight pairs, regardless of baseline success. A restricted first-pair smoke test is a predeclared engineering validation, not evidence-based selection.

`contrast_protocol.json` specifies the contrasts and interpretation limits. The matched donor says `The record now lists ... as ...` instead of `The note says that ... is ...`. Its opposite-label counterpart differs only in the one fact token. The irrelevant donor discusses rooms/windows, with newline at the same p. Native source0 V transplanted to p is a useful no-later-content reference donor. The opposite whole-input recipient is an oracle that demonstrates the readout can express the other label; it is not another value-only intervention.

## Routing and controls

Primary `frozen_all` scope: replay each baseline-world attention matrix P at all16layers, multiply by the CURRENT recipient V, and apply donor replacement only in layers2–15 and queries>p. Baseline P already includes source0 masking in that world. This holds complete routing fixed, not merely source-p K. V pathways in suffix tokens remain dynamic.

The optional `single_layer_endogenous` scope patches only preselected layer8. It freezes P only there, asserts routing is still baseline at that point, and allows endogenous Q/K/P routing in later layers. It estimates a different effect and must not be pooled with the primary scope.

Eight conditions: identity, matched, mismatched, irrelevant, source0 cross-content negative control, recipient source0→p reference, unscaled fact-token ordinary-source control, and output-norm-matched fact-token message control. The ordinary fact source is p−2, so its distance differs by2tokens: an exact same-query distance match at a distinct source is impossible. Its aggregate multi-head perturbation is norm-matched AFTER W_O to the mismatched newline intervention at each layer/query. A scalar pre-W_O rescaling implements this exactly because W_O is linear; norms are measured and verified after W_O. The normmatched control is a query-dependent message perturbation, not one fixed donor V. The additional unscaled ordinary_raw condition is a literal value replacement at a known carrier. Neither is an exactly distance-matched control.

All per-layer recipient-prefix hashes ≤p must match the recipient baseline exactly. Native versus masked baseline prefix hashes must also match. Identity and cross-content source0 controls must preserve margin within1e−4. A manual target logit from final hidden state and two head rows must agree with the full-vocabulary streamed target logit within1e−3. Final logits are re-normalized over all50304rows, not a truncated vocabulary. Correct-versus-foil log probability is exactly the two-logit difference.

## Commands

From this directory, lightweight checks require no CPU slot:

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 python freeze_inputs.py
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 python test_synthetic.py

After explicit CPU-slot grant, baseline-only smoke (8forwards):

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase baseline --pairs mira_color --cpu-slot-authorized

All frozen baselines (64forwards, or56additional after smoke):

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase baseline --cpu-slot-authorized

Inspect `results/preregistered_v1/baseline_gates.json`. Do not tune or select examples using donor effects. Baseline failures are still retained.

Donor cache stage (24forwards):

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase donors --cpu-slot-authorized

All eight primary conditions (256forwards):

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase interventions --cpu-slot-authorized

Cheaper primary matched/mismatched slice (64forwards, other conditions can be added later with the same caches):

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase interventions --conditions matched,mismatched --cpu-slot-authorized

Optional single-layer paired comparison (64forwards):

    PYTHONDONTWRITEBYTECODE=1 python run_content_donors.py --phase interventions --scope single_layer_endogenous --conditions matched,mismatched --cpu-slot-authorized

Summary, no model execution:

    PYTHONDONTWRITEBYTECODE=1 python summarize_results.py

Each completed run is saved separately, so the commands resume without repeating completed forwards. Fresh experiment versions require a distinct run name. The runner binds input/protocol/amendment/runner/algebra hashes and the passing10-case FP32 validation artifacts. It refuses changed metadata, changed NPZ cache checksums, mismatched saved intervention identities and missing caches. Results contain per-layer prefix hashes, margins, correct-label loss, full-vocabulary probabilities, numerical checks and intervention diagnostics.

## Exact counts and resource estimate

- Baseline stage:8pairs ×2labels ×4source-access controls =64forwards.
- Donor stage:8pairs ×(2paraphrase labels +1irrelevant) =24forwards. Source0→p reuses recipient caches.
- Full primary intervention stage:8pairs ×2labels ×2worlds ×8conditions =256forwards.
- Complete expanded default study:344forwards. At the parent-measured3.4seconds/128-token forward, a planning estimate is1169.6seconds (19.5minutes); short20-token timing is not measured yet. This is not a promised runtime. A first-pair complete smoke is43forwards (~146.2seconds at that reference rate).
- Additional optional single-layer paired comparison:64forwards (~218seconds at the same reference rate).
- Per-sequence cache maximum: P=16×16×20×20×4=409600bytes; three source-V arrays=3×16×16×128×4=393216bytes; total802816bytes before NPZ metadata.88baseline/donor caches upper bound70,647,808bytes (~67.38MiB), plus small JSON results.
- Only one recipient baseline and one donor cache are loaded during an intervention. Core weight maps are closed after each operation. The measured shared runner peak is~430MiB on128tokens; this compact20-token assay targets<1GB RSS. RLIMIT_AS and every model RSS checkpoint enforce cap4,000,000,000bytes, and BLAS/OpenMP/MKL thread counts are forced to2 before NumPy import. No downloads, GPU, training or concurrent full-model jobs are used.

## Evidence interpretation

Primary effect is same-label minus opposite-label donor margin, averaged first within reciprocal pair. Stronger evidence is bidirectional sign transfer: both native baselines and matched-donor predictions favor the recipient label, while both mismatched donors flip to their donor label. A generic loss increase from arbitrary donors is not semantic evidence. Retain all-pair results alongside separately labeled baseline-qualified subsets and keep native/masked worlds separate.

A preserved source0→p condition would show that this task's p value can be supplied by a source unable to encode the later fact, under this particular routing intervention. It would not prove that p never carries content. Failure to transfer likewise does not establish absence of payload: the model can read the original fact directly, donor payload can be weak or distributed, and freezing routing can remove necessary downstream decoding paths. These are short synthetic fact prompts, not proof of the long-corpus newline-sink phenomenon or abstract semantics. The first-newline token has not been assumed to attract sink-like attention. No claim of broad population generalization or formal statistical significance is warranted from eight hand-built pairs.

## Separate p-usage qualification

Pre-output amendment01 adds native→maskp and mask0→mask0p correct-label NLL costs. Both reciprocal costs and the pair mean are archived. The descriptive usage gate is pair-mean cost>0.01nats, distinct from the two-label discrimination gate; neither authorizes dropping examples. A negative donor effect on a case that does not use p is especially uninformative. Observed-target loss remains archived but is not the usage gate because the common appended A target is deliberately a foil for the B recipient.
