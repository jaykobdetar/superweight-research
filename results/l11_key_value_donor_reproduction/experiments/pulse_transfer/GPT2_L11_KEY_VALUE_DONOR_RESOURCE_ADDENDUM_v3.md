# T22 prospective resource addendum v3 — returned outputs and route-specific failure preservation

**Source/resource review only. No CPU fixture, tokenizer, checkpoint, model/GPU or publication authorization.** This preserves the v1 inventory/plan and v2 additions/addendum unchanged. The new exact route table is [t22_symbolic_resource_inventory_v3_routes.json](review/t22_symbolic_resource_inventory_v3_routes.json), SHA256 `b5f616fcdd0e1113da612fa46f8a6c57e6fe6b439937153c42caa7debb2ae8e8`. It binds runner `dbbf251c78d76aff34658ca24440e24552f64d3fa274f309bf3dcfa6628aefa0`, storage `91a985a5bfe5df6233528802fd6b3d4fad05e3b436ef4249d2a2f79f7a863728`, unchanged hook `1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28` and v2 hashes `95c8e14e…`/`73bb998a…`.

## Source delta and branch lifetimes

R1 now makes the return callback counter-only and binds actual output/logit/hidden owners before post-return I/O or profiler exit. Generic post-return failure retains one complete actual512×50257 FP32 readout. An unhooked generic failure additionally retains its actual512×768 hidden once. Explicit identity comparison already owns both full readouts, so it gets no third current copy. Failed suffix comparison retains expected128rows and actual**512**rows: this replaces its old128row actual member by an extra77,194,752bytes, rather than adding another complete actual member. Scores and prefix comparisons remain unchanged.

**Donor suffix checks occur before their KV closures.** On an orthogonal-donor SUFFIX failure, both variants' actual donor later-reference sets remain live, seven MAIN snapshots are complete and current write norms use the matched schema. The old post-closure suffix branch with twelve snapshots and no references does not cover this state automatically. The new branch replaces the12snapshot prefix with7, adds164,253,320bytes of live reference sets and33,280bytes of matched-current norm/mask delta. The analogous PC branch uses five snapshots and only its direction's donor QKV cache. These are prospective lifetimes, not outcome-based dropping or deduplication. Failed comparisons preserve complete operands; successful KV closure retirement stays unchanged.

The live-donor post-return source/processing branch also includes matched-current fields and the complete source comparison occurrence upper. KV-success post-return failure has retired its current variant's reference; the earlier full-output/two-reference bound dominates its extra completed snapshot. Native/interpolation failures before actual model-end still have no returned-current readout. All scopes and arithmetic are explicit in the route JSON.

## Conditional route bounds

| Branch | Raw array upper B | Buffered NPZ upper B | Buffered RSS upper B | Frozen-header direct-stream RSS upper B |
|---|---:|---:|---:|---:|
| Post-closure suffix | 622,660,472 | 632,505,815 | 3,085,985,948 | 1,765,774,200 |
| PC donor suffix, refs live | 632,287,786 | 641,683,765 | 3,116,263,650 | 1,775,401,514 |
| Orth donor suffix, refs live | 694,246,402 | 703,815,975 | **3,318,019,738** | **1,837,360,130** |
| Live donor post-return source/processing | 656,129,830 | 665,653,793 | 3,194,038,257 | 1,799,243,558 |

The3GiB cap is3,221,225,472bytes. The new orthogonal donor-suffix branch **exceeds the buffered conditional RSS cap by96,794,266bytes**. Therefore the prior3.192GB buffered figure is not a universal certificate. At this branch's symbolic upper, complete preservation requires a direct stream; actual current-RSS admission still decides the route for every real tree.

Largest complete stream NPZ conditional upper is**712,210,612bytes**, below1GiB by361,531,212bytes. Its metadata term conservatively uses the compression bound of16MiB raw metadata rather than assuming the buffered8MiB compressed-metadata check also applies to streaming. Normal evidence remains bounded conditionally at513,184,538bytes under512MiB. No caps, numerical tolerances or raw evidence fields change.

## Admission and new compatible header path

R2 now checks observed non-evidence RSS≤896MiB and conservative whole-failure header feasibility before model loading, after loading/before the first forward, and before every subsequent forward. The source header raw allowance760,300,838bytes covers the new largest raw occurrence upper694,246,402bytes. It is deliberately a header-admission allowance, not a buffered encoding certificate.

Buffered route retains the frozen `dumps_npz(codec_tree(...))`. On archive-buffer refusal, header-admitted direct streaming retains the frozen header encoder. If header/normalization is refused, the new bounded path incrementally hashes C-order bytes and writes an explicit C-order NPY header/data in1MiB chunks. It uses the same public codec schema, tensor dtype/shape/hash metadata and separateaNmembers; **this is a new implementation, not previously runtime-validated frozen `_encode`**. Its semantics for contiguous, noncontiguous, Fortran, negative-stride, scalar, empty and dtype/signed-zero/subnormal cases require an explicitly authorized tiny codec witness.

Conditional maximum bounded-header stream RSS is1,701,927,938bytes, using896MiB non-evidence plus raw occurrence upper plus64MiB workspace and1MiB chunk. The frozen-header stream maximum is1,837,360,130bytes, including the103MB largest identity temporary and96MiB workspace/normalization. Those figures still assume the declared scalar, metadata, copy, workspace and ownership contract. Source logs unexpected admission breaches as failed resource provenance while attempting complete bounded preservation; it does not relabel them admitted. They are not permission to exceed3GiB or evidence of model readiness.

The later exact whole-tree witness must exercise the **live orthogonal-donor suffix** branch and both stream-header routes, not merely the previous generic/closure branch. It must enumerate actual member/scalar/dtype/shape/byte universes, validate actual observed RAM ownership/buffer peaks and same-schema decoding, and preserve every failure. The small changed-ABI fixture is a separate earlier gate. No source import, fake/model forward, checkpoint/tokenizer access, input selection, deletion or Git action occurred for this addendum.
