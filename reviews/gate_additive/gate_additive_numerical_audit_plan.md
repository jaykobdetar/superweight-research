# Prospective independent saved-output audit

Status: audit plan only; no T17 pretrained output has been reviewed. No new model or GPU invocation is authorized for this reviewer.

After a completed manifest and production analyzer pass, verify exact frozen source/input/map/direction identities, 1,992 returned forwards, 1,000 main records, 992 suffix records, 1,800 checks, 15,360 layer guards, 928 hidden trajectories, 1,792 retained main gate states, four attention states and the final six-row storage addendum. Failed or partial output remains partial. Read shards sequentially below 3 GiB RSS.

Independently reduce all saved per-token FP32 losses, both KL baselines and agreement arrays over the declared full/early/later ranges. Recompute all FP64 J/R/N, singleton losses, identity residuals, bootstrap means and intervals with shared indices; retain n=4 and conditional-calibration limitations. Check raw gate formulas, incoming singleton equality, source/suffix evidence and available FP64 QKV/context snapshots. Full successful source/context/logit arrays are absent outside declared snapshots; runtime fingerprints cannot independently reconstruct missing arrays.

For readout, load only the pinned GPT-2 safetensors embedding tensor, verify its FP32 fingerprint against the recorded tied lm_head weight, and never construct a pretrained model. Stream hidden states and query chunks. Compare CPU FP64 hW against the actual saved FP32 output at queries 0,1,126,127,254,510 on four natural window-0 inputs. Report absolute/RMS/relative-L2 errors, observed CE and both directional KL errors, and the prospective FP32 accumulation bound from the approved CPU math fixture. Query 0 is diagnostic only. No sampled-row result proves exact all-query GPU equivalence.

Reconstruct every retained query's readout approximately and compare all saved runtime J/R/N, KL, centered-logit and agreement metrics directly. Report maxima and means of approximation errors, material sign or argmax disagreements and uncertainty relative to those errors. Do not silently replace runtime scientific primaries with CPU reconstructions. The decomposition J+R is an exact algebraic counterfactual identity, not additive causal mediation. R equals LSE(z_add)+7LSE(z_P)-sum LSE(z_single), hence is target-independent; J also contains the target-logit joint-minus-additive term.

Use one weight-loading job, bounded chunks, and no old-model or duplicate inference. Parent owns the experiment and report; raw evidence and all failures remain preserved.
