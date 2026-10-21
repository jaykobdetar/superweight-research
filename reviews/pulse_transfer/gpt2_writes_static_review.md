# GPT-2 writes static review

No launch-blocking defect found in `gpt2_writes.py` and `fresh_inputs_v2.py`. CPU evidence: `gpt2_writes_cpu_test.py/.json`, peak761,012 KiB; no model instantiated or GPU used.

E modifies the source residual afterL2. A and M intercept native attention/MLP c_proj outputs atL3..11 before residual addition; dropout is inactive in evaluation mode. Edits are source0-only and limited to frozen coordinates138/447 (or the specified random pair), with later-token exact locality assertions. All other computation remains endogenous. Native tuple output handling is consistent with GPT-2 blocks. The all-three-on condition must preserve zero coordinates through every later residual addition; ten synthetic closures pass. Rescale preserves the remaining direction and original source norm to FP32 tolerance, and leaves the erased coordinates zero. Identity and entire-source controls are distinct.

The full factorial has eight conditions plus three controls;32 inputs plus eight anchors imply360 forwards. Scoring inherits predictor1/target2..511. Captures support actual raw-coordinate restoration after E and causal dependence on subsequent writer families, without identifying unique neurons or additive effects.

Independent private-source/tokenizer preparation produced all16 fixed-fraction third-panel inputs and reproduced previous token hashes. Every fresh used512-token character span is disjoint from eight previous spans per corpus and from other new spans. No fraction was changed. Warning about the untruncated extraction buffer exceeding GPT-2's context length is harmless here; the model receives512 tokens only.

One documentation refinement sent to parent: code captures residual vectors for all12 layers but attention/MLP vectors onlyL3..11, whereas the initial design text said every layer/component. Narrow that wording unless early component captures are required. Runtime numerical controls, parameter immutability and saved-result replay remain pending.
