# OLMo-1B native training-state provenance

Addendum to the completed twelve-checkpoint study | 23 October 2026 UTC

## What the follow-up recovered

A credible native dense checkpoint source now exists, but an authentic continuation is not yet ready. An Ai2 maintainer explicitly linked a July-2024 step954000 release in OLMoE issue #32. At that documented public location, config.yaml, model/metadata.json and optim/metadata.json all returned HTTP 200. These three files total 575,590 bytes. No checkpoint tensor payload was downloaded. [1, 2]

The model metadata describes 66 native tensors totaling 1,279,787,008 FP32 parameters: dense feed-forward projections, not the eight-expert upcycled model hosted under the misleadingly similar HF name. Both Adam moments are named for every model tensor. Their shapes match; the nonempty model-shard intervals cover every tensor without gaps or overlaps across a 64-rank layout. This is stronger evidence than a generic statement that optimizer checkpoints were released. It is not numerical proof that these native weights equal a particular HF conversion. [2]

The native config specifies mitchish1, seed 6198, untied embeddings, 4,096-token context, QKV clipping at 8, non-affine LayerNorm, AdamW betas (0.9, 0.95), learning rate 4e-4 and decay 0.1. Its 512-sequence global batch corresponds to 2,097,152 tokens per step. The ordered data manifest has 1,033 paths, including two repeated Wikipedia paths. Do not deduplicate or sort it when reconstructing the dataset. These path counts are not token-mixture proportions. [2]

The native config is semantically identical to the previously recovered config accompanying the upcycled HF artifact. The new evidence is that an independently documented native dense model and optimizer manifest accompany it. Exact code, historical settings, restart state and release identity remain separate questions.

## A real inconsistency, not a resolved recipe

The saved native scheduler is token-based cosine with a 3T-token horizon, warmup of 10,485,760,000 tokens, and minimum-LR multiplier 0.1. At its configured batch size, warmup is 5,000 steps. The OLMoE paper's July-1B comparison table instead says 2,000 warmup steps and 2T pretraining plus 50B annealing tokens, whereas the pinned HF release card and branches extend to about 3.05T. A summary table cannot substitute for the actual sequence of training configs. The native file describes a resume from step878000 and does not certify what settings applied at earlier steps. [2, 3]

The model card also contains passages explicitly about the 7B April model. Its optimizer epsilon or curriculum descriptions should not be silently treated as a complete, exact-run recipe for this 1B release. Public issue #901 still has no substantive answer to the exact-config question. [4]

[PAGEBREAK]

## Access results and remaining uncertainty

The first metadata-only HEAD probe of the documented model shard containing the target scalar returned HTTP 403. The precise target was https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/model/rank_9.safetensors . The probe stopped immediately. No GET, range request, alternate endpoint, credentials or retry was attempted. The planned optimizer, trainer and data-index probes did not execute. This establishes a HEAD access restriction for that shard, not absence of the object, and not a tested denial for all the other artifacts.

The OLMoE paper links a public W&B comparison report. It loads anonymously in the cloud browser, but the exposed runs identify MoE and 7B models, not the July-1B run. The native config's ai2-llm/olmo-small project displays a page-not-found notice mentioning possible login or privacy requirements. No login was attempted. The current public view did not resolve the dense run ID, source-code commit, restart chain or data-order file. This is an access/visibility uncertainty, not proof that the logs were never released. [3, 5]

The older inferred S3 target that returned 403 earlier in the investigation was never retried. The new metadata source was found in a maintainer's explicit public release announcement, not by changing endpoints to bypass that restriction.

## Exact prerequisites for a historical causal experiment

1. Identity and code: verify native step954000 against HF step954000-tokens2000B using mapped tensor values or hashes, and identify the training-code commit, dependencies and conversion recipe. Do not assume the final HF main equals the last numbered branch.

2. Intervention-time checkpoints: obtain matched dense model, optimizer and trainer states before the relevant developmental transitions. Our sampled H12 transition lies between 120k and 256k; H14 newline functionality between 512k and 800k. The newly found 954k checkpoint is already later than both. Alone, it supports later maintenance/continuation questions, not direct historical induction of those transitions.

3. Optimizer and scheduler: obtain first/second moments, per-parameter step counters and serialized parameter-group hyperparameters. The YAML omits explicit epsilon. Recover actual scheduler/trainer token counters and any restart overrides rather than inferring them from a table.

4. Data and RNG: obtain the exact ordered token-file manifest with sizes/hashes, sampler version, global indices or equivalent shuffle state, epoch and cursor, world-size/batch conventions, tokenizer/preprocessing identity and per-rank RNG state. A seed and a Dolma version name do not establish the next batch. Saved config flags restore_dataloader=true and save_data_indices=true indicate intent, not public availability of those files.

5. Baseline replay: reproduce a short, untreated continuation toward an authentic successor checkpoint before interpreting optimizer-basis, decay, boundary-frequency or gradient-pressure interventions. Otherwise observed differences could be restart or data-order artifacts.

[PAGEBREAK]

## Practical next decision

The next useful action is a maintainer-provided public-access clarification and exact-run artifact map, not a large speculative download. Ask for the native dense checkpoint identity mapping, pre-transition checkpoint locations, code commit and restart history, and sampler/trainer/RNG records. No external inquiry has been sent.

Once the owner clarifies access, the cheapest verification is still small: read safetensors headers and selected mapped values from model/optimizer shards, including the serialized optimizer group, without loading a full model. The metadata locates down[1764,1710] and both moments in rank_9; its Adam step scalar is in rank_8, while group settings are in rank_0. This is a future plan, not permission to retry the restricted shard. A numerical match would strengthen identity, but a few matching entries would not prove complete checkpoint equality.

A full native model would contain 5,119,148,032 parameter bytes; two FP32 moments add 10,238,296,064 bytes, before trainer state, gradients and activations. That is outside this study's low-memory partial-model scope. Actual causal training still needs separately approved resources and a reproducible baseline. The completed developmental findings and original twelve-point evidence are unchanged.

## Sources and evidence

[1] Maintainer release announcement, including native July-1B location: https://github.com/allenai/OLMoE/issues/32#issuecomment-2856649755

[2] Public metadata successfully retrieved: https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/config.yaml ; https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/model/metadata.json ; https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/optim/metadata.json

[3] OLMoE paper, Appendix B/Table 10 and Figure 3: https://arxiv.org/html/2409.02060v2

[4] Exact-config issue: https://github.com/allenai/OLMo/issues/901 ; pinned release: https://huggingface.co/allenai/OLMo-1B-0724-hf/tree/d7cbab742d80589e714b1a2d7f838dcd21cbe143

[5] Public comparison report: https://wandb.ai/ai2-llm/olmoe/reports/Plot-OLMoE-1B-7B-vs-OLMo-7B-vs-OLMo-1B--Vmlldzo4OTcyMjEz ; configured source project: https://wandb.ai/ai2-llm/olmo-small

The companion evidence bundle contains saved public metadata, timestamped HTTP receipts, browser-view records, an offline consistency audit and checksums. Network source-document transfer was 1,566,181 bytes, excluding browser assets. Checkpoint tensor and trainer-state payload transfer was zero.
