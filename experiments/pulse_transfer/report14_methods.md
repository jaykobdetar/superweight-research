## Intervention and evaluation protocol

The checkpoint is `openai-community/gpt2`, revision
`75e09b43581151bd1d9ef6700faa605df408979f`, FP32 with the recorded SDPA backend.
All layer numbers are zero-based. Parameters are unchanged. The native model
is C; P sets the source-position value vector to zero in each attention layer
4–11. Query/key projections and non-source value vectors are not directly
patched, but they may change downstream. Gate edits preserve query zero
relative to P; P's source-value zeroing itself can change query zero versus C.

For each layer and query q>0, define non-source attention mass a and
`b = logit(a) = logsumexp(non-source scores) − source score`.
Calibration centers `r = b − log(q)` and fits its centered 12-dimensional
covariance/SVD. Each fit pools 510 queries from each of four calibration
passages, with covariance denominator 2,039. These queries are correlated;
they are not 2,040 independent samples. Leave-one-passage-out diagnostics use
the same fixed procedure and denominator 1,529. No direction or rank is
selected using the new evaluation losses.

For center α and normalized leading direction u, let
`s = (b − log(q) − α) · u`. PC1 removal sets `b′ = b − s u`.
Its matched control sets `b′ = b − s v`, where v is orthogonal to u,
constructed deterministically from the fixed QR seed. Both have the same
gate-logit perturbation norm at their own incoming state. The hook scales
the actual context by `sigmoid(b′)/a`, preserving its conditional non-source
mixture locally. Downstream computation remains live. All L10-only gate
intervention arms share P's incoming L10 state; joint-layer conditions need not.

There are 24 conditions: C, P, two identity controls, eight Wiki-calibration
PC1-removal arms, eight orthogonal arms, and four M8-anchor arms. The eight
arms per family cross four center/direction choices with L10-only or joint
L4–L11 application. Current/full-rank identities, native repeats and the
writer identity control are checked exactly against their references.

The four fresh calibration passages are Wiki-test windows selected before
evaluation. Sixteen exposure-separated evaluation passages comprise four
each from Wiki validation, Mill's Utilitarianism, Spinoza's Ethics, and joined
Adadelta/Adamax/ASGD source code. Corpus files, processed text, extraction
offsets, rejection history, token hashes and checkpoint identities are pinned
in the manifest. Natural starts and BOS replacement use the same 512 tokens
except for token zero; nothing is inserted or retokenized. The starts are not
matched across corpora or calibration panels. Two current natural starts were
seen in M8, none in O or F; one fresh-calibration start was seen in earlier
evaluation. The complete identity accounting remains local and reproducible.

Every forward executes exactly 512 positions. Main scores cover targets
2–511. Early targets 2–127 and later targets 128–511 are secondary views of
the same execution, not experiments at different context lengths. Each main
condition has a suffix control rotating positions 128–511; unchanged-prefix
logits, hidden states, attention/gate records and per-target quantities must
match exactly. The two earlier cross-length failures remain preserved and
are not reclassified as successful tests.

Uncertainty resamples the four paired passages in each corpus, with 2,000
shared bootstrap draws and seed 20261021. Intervals condition on the two
fixed calibrations; they estimate neither population calibration uncertainty
nor representative corpus uncertainty. The passages are not randomly sampled
from a defined corpus population. The many arms, starts and score views are
correlated, and no multiple-comparison adjustment is applied. All 576 group
summaries and 912 paired contrasts are retained, not just favorable ones.

## Failed branch and numerical audit correction

The original proposal required four fresh Wiki and four fresh Hume calibration
passages. Its tokenizer-only preflight failed before any pretrained forward:
the 212 Hume exclusion records merge to 20 intervals; the largest remaining
gap is 5,492 characters, shorter than the required 8,192-character buffer.
An independent full-gap audit confirmed that this was not merely a grid-search
failure. No exclusion was relaxed. The revised Wiki4-versus-Wiki4 design was
reviewed and frozen before its first model forward. The failure, original
proposal, retained Wiki windows, capacity proof and revised protocol remain
available.

After the successful model run, the original saved-state auditor failed on
exact direction reconstruction. The lossless NPZ representation preserved
matrix values but not tensor strides. Replaying the QR-column calculation
with the decoded row-major layout changed 13 FP64 control vectors by at most
1.1102230246251565e−16; their FP32 forms were all exactly unchanged.
Independent diagnosis showed that restoring only the 24 original QR matrix
layouts reproduced every saved direction, hash, scalar and geometry field
exactly. A separate, reviewed adapter performs that value-preserving layout
restoration before invoking the unchanged frozen auditor. No tolerance,
intervention, model output or original source was changed. The failed audit
and its log remain preserved.
