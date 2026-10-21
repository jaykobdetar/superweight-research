## What this experiment changes

The earlier gate study found that removing a leading calibration direction could
raise loss on Wiki while lowering it on Du Bois. This study asks where those
changes occur: at one layer, or through a combination of layers. The earlier
panel motivated the question and is development evidence. No layer was selected for this run
using its outputs.

The checkpoint is GPT-2 `75e09b43581151bd1d9ef6700faa605df408979f`, with
FP32/default SDPA. All indices are zero-based. Every forward has exactly 512
tokens. The native model is reference C. Reference P sets the source value at
position 0 to zero at layers 4–11; all intervention arms retain that same
boundary condition, including layers whose gate is untouched.

For each head and query, let a be the attention probability allocated to
non-source tokens and b its logit. With source V zero, the context is a times
the conditional non-source value mixture. A patch changes b to b′ and delivers
`c′ = c × sigmoid(b′) / a`. This preserves the current conditional mixture
at that boundary, while allowing later computations to respond. The expression
is established attention algebra, not a newly discovered mechanism.

The original eight natural-start calibration passages provide a mean α and
leading direction u for the centered residual `b − log(q)`, separately at
each layer. These maps are frozen; there are no new calibration forwards.
Let `s = (b − log(q) − α) · u`. The three intervention families are:

- PC1 removal: `b′ = b − s u`.
- Orthogonal control: `b′ = b − s v`, where v is a frozen unit vector orthogonal
  to u, derived from the original seeded QR basis.
- Position replacement: `b′ = α + log(q)`.

Each family acts at one of L4…L11 or jointly at all eight layers. Native C,
zero-value P and two exact-copy controls bring the total to 31 conditions.
At a singleton layer, actual incoming logits, masses, coefficients and contexts
must equal P exactly. In a joint run, earlier interventions change the state
reaching later layers. The control therefore matches the hypothetical PC1
removal norm **at its own current state**, not across different joint runs.
Matching logit norms does not match mass, context, or output changes. One
orthogonal control is not a distribution of random controls.

![The intervention boundary and matched control](figures12/localization_mechanism.png)

## Inputs, controls and uncertainty

The held-out panel has four Wiki validation passages, four Du Bois passages,
four Hobbes passages and three optimizer-code passages. Natural/BOS pairs use
the same tokenized passage, replacing token 0 only. All 15 natural first-token
identities and BOS are absent from the original calibration starts. This is
research-exposure holdout, not a claim about unseen pretraining text.

Tokenizer-only preflight checked historical GPT-2, Llama and shared-source OLMo
windows, plus full calibration/development extraction buffers. It originally
failed to find four code windows. An exact interval audit found a maximum of
three nonoverlapping eligible windows on the frozen 1,001-fraction grid. The
revision retains all 15 successful passages, keeps every exclusion and uses
actual n=3 for code. It does not prove that no fourth window could exist at
any character start. The code windows lie late in the joined SGD/Adam/AdamW
source; corpus coverage is limited. The failed preflight is preserved.

Targets 2–511 are scored. Early targets 2–127 and later targets 128–511 are
views of the same execution, not separate context-length experiments. Each
condition also gets a changed-suffix control: tokens 128–511 are rotated while
the first 128 tokens remain identical. Prefix outputs, gates, source states
and contexts must remain exactly equal. Changed suffix targets are never
scored in these controls. Source query/key/value identities, query-0 preservation,
untouched contexts and exact-copy closures are checked separately.

The planned accounting is 938 main records, including eight native anchors,
and 930 suffix controls: 1,868 forwards, zero new calibration forwards.
All 27 intervention arms are reported. The primary paired contrasts are PC1
removal versus P, PC1 removal versus orthogonal control, and position replacement
versus P, at every singleton and jointly. Joint change minus the sum of
singleton changes is descriptive nonadditivity; it is not a mediation estimate
or an identification of feedback.

Intervals use 2,000 paired passage bootstrap draws, seed 20261021, with actual
n=4 prose/n=3 code. They condition on the fixed calibration and this small,
nonrandom panel. Natural/BOS and early/later views are correlated. No correction
for the large number of comparisons is applied; layer rankings are exploratory
and require new confirmation. Lower teacher-forced loss is distinct from
preserving native predictions.
