# Frozen token-history experiment after fresh later-use discovery

Fresh later-use traces show10 natural-ID large later pulses, all on first single
newline token187; one underscore exception appears in a period-source identity
variant. Tokenizer encodes newline as187 and double-newline as535. Association is
not causality. Test prior occurrence and token identity while holding positions,
token count, and later targets fixed; do not retokenize edited IDs.

Use the16 fresh later-use512-token windows. Replace IDs187/535 only within
positions0..128 with space209, set source0=The510. Set probe positionsp=64,q=128.
At p choose187,535,186(tab),209(space),510(The),15(period),64(underscore); at q
choose187 or535. Preserve every ID after q. These constructed prefixes are
intervention inputs, not a natural-corpus perplexity benchmark.

Record all14 identity-pair baselines. For p in{space,187,535}, both qIDs, delete
the target-weight contribution only at p, only q, orboth, and delete the whole
neuron column atq as a matched scope control. Score only predictorsq+1 onward
(targetIDs130..511), identical across all edited inputs. Compare each deletion
to its own edited-input baseline. Preserve native pulse trajectories and H12
contexts/clipped values at0,p,q.

For the p187,q187 case, remove only attention edge(q,p) in L0, L1, orboth, with a
matched edge(q,p-1) control inL0. Standard explicit causal masks in all baselines;
head-shared singleedge deletion preserves all other edges. This directly tests
history transmission without changing token IDs, but it is not a uniquely
localized head mechanism. Record boundary pulse and output effects.

No fits or selection by loss. Expected672forwards:14 baselines+24 deletion
conditions+4edge interventions perwindow. Paired-window intervals are exploratory
on reused fresh text; first occurrence among edited tokenIDs is the controlled
factor, not a general claim about rendered newline characters or semantics.
