# T20 input-adapter clarification

Prospective clarification for independent review, 21 October 2026. It binds
`GPT2_POST_WRITE_NORM_MATCHING_PROPOSAL.md`, SHA256
`e562fd57097228fb3348d8a7d564010456651c9dd6bde4540496a11bae338aff`.
The proposal remains unchanged. This addendum does not authorize a model
load, execution, or selection before the input adaptation is reviewed.

## Preserve traversal; extend only duplicate acceptance

The original hash-pinned `select_panel` function checks and adds only the
natural window hash. The proposal additionally requires checking both fresh
natural and BOS hashes. That changes admissibility and must be explicit;
do not describe the adapted function as behaviorally identical to the old
function on all candidates.

Permit a narrow source adaptation of the original selector, SHA256
`810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a`.
Keep its immutable original source. Preserve preferred grid positions,
candidate enumeration and distance/tie order, source/component bounds,
buffer size, tokenizer invocation, offset handling, used-character span,
existing overlap/length/natural-duplicate checks and all preceding rejection
precedence. Modify only the duplicate-hash acceptance and accepted-hash-add
branch, plus explicit metadata for the additional check.

At that branch, derive the BOS identity by cloning the first 512 natural
token IDs and replacing token zero with the already frozen BOS literal.
Do not insert or retokenize. Apply the natural duplicate check first. Then
reject a BOS hash found in the inherited history or any previously accepted
candidate's natural/BOS set, recording an explicit BOS-duplicate reason and
whether the hit was historical or already selected. Continue with the next
candidate in the unchanged enumeration. On acceptance, add both identity
hashes to the membership union and retain separate natural/BOS attestations.
The selected natural character span remains the original tokenizer-derived
span; a synthetic BOS token has no new source-text character interval.

The two identities of the *same candidate* are not duplicates of a previous
candidate. If a natural first token already equals BOS, their hashes coincide:
allow that pair unless its hash hits inherited/previously accepted history,
record its identical-start status, and add one set member with two identity
attestations. Do not introduce first-token balancing, a special-token
exclusion, or a new seen/unseen sampling rule. The scheduled natural/BOS
forwards remain counted and a no-change pair is reported honestly.

Before selection, review the adapted source and a complete AST/text change
proof that all traversal/span/tokenizer logic is unchanged outside the
whitelisted branch. Synthetic selector fixtures must cover natural-history,
BOS-history, BOS-versus-previous-natural, natural-versus-previous-BOS and
BOS-versus-previous-BOS collisions; identical within-candidate starts;
unchanged tie order; overlap rejection; component crossing; and retained
8192-character buffer exclusion. Reconstruct the actual complete candidate
ledger independently after preparation. No loss is involved in these checks.

## Historical membership and privacy

Carry all 32 attested T19 natural/BOS identity hashes and all 16 natural
used-character spans, alongside every older union and complete calibration
buffer. Alias deduplication for set membership must retain source-manifest
hashes and origin records. Bind the actual attestation/scan files by SHA256
before selection; no silent default to natural-only history is permitted.

Historical first-token membership in F/T18 remains prospectively omitted,
as already stated in the proposal. Mark it unavailable/not used. No old raw
token tensor, pretrained model, decoding of historical first-token IDs or
new membership service is needed for that optional diagnostic. Natural/BOS
pair identity and all token-window hashes remain required. Public exports
must preserve hash/provenance checks without releasing restricted raw IDs,
text, checkpoint tensors or host paths.

Everything else in the 776-forward protocol, including matching arithmetic,
both RMS domains, signed-byte controls, input audit, full writer/failure
fixtures and separate model-launch review, remains unchanged. This is a
prospective acceptance-rule clarification, not a relaxation of exposure
exclusions or permission to overwrite any historical source or evidence.
