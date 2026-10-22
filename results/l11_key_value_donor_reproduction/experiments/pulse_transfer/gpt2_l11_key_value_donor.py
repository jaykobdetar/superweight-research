"""Prospective T22 native000 L11 donor-key/value runner. HOLD: literal reviewed launch freeze.

Imports frozen T19 scoring/forward helpers; does not execute the T19 study.
No weights may load until new input, hook, source and full writer reviews pass.
"""
import argparse,contextlib,copy,hashlib,json,platform,resource,shutil,subprocess,time
from importlib.metadata import version
from pathlib import Path
import torch
from gate_native_value import fingerprint
from gate_write_norm import runtime_behavior
from gate_l11_key_value_donor import GateL11KeyValueDonor
from gpt2_gate_native_value import metric_rows,model_forward,F_DIGEST,METADATA_PINS
from l11_key_value_donor_storage import (ARMS,DONORS,RECIPIENTS,L10_PANEL,L11_PANEL,CLOSURE_KEYS,L11_NORM_KEYS,EvidenceWriter,NORMAL_LIMIT,FAILURE_LIMIT,DISK_RESERVE,METADATA_ALLOWANCE,READOUT_ROWS,
 TIMING_ROWS,TIMING_NODES,byte_exact,nested_fp,summarize_diag,empty_payload,trim_gate,trim_gate_norm,failure_bundle,suffix_operands,logit_rows,source_closure,sha,json_write,read_npz)

ROOT=Path(__file__).resolve().parent
T20_RUNNER_ANCESTRY_SHA='43afc0af75d722fed2586866bc0ffc86d3e3998522ba66da63f3ef6e08a927cf'
PROPOSAL_SHA='6238ea83f978705b0069e3b476f3e867ba95f08fbae2864d6e92827b88e2f159'
RESOURCE_PLAN_SHA='ebc7f75070826d7f20bd8e1d0c5faca40241d781cadc2dd557b4a48a7c6b779e'
RESOURCE_INVENTORY_SHA='bce2948f8a4e5d6a67c22f89e8d91b7fd34c133084208e9ce8e88e7aff9dcea7'
DONOR_HOOK_SHA='1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28'
NATIVE_HOOK_SHA='b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd'
T18_MANIFEST_SHA='f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060'
BOUNDARIES=('000',)
SEGMENTS={'A':('wiki_validation','utilitarianism'),'B':('ethics','optimizers_new')}
CORPORA=tuple(c for names in SEGMENTS.values() for c in names)
SNAPSHOT_INPUT=('wiki_validation',0,'natural')
INPUT_HASHES={'data_manifest.json': 'eb172817f41d0c47d2b746c7d074b5331bcce96d49dd7aee2219cc3ac9222227', 'exclusion_inventory.json': '5fcb3ee8386fb2ea34141cbf8a9904375a00db0c39b6e4736ea1d0ee8cec9941', 'known_history_chain.json': '2ac54ea99d9e991434d364e19ceb483440e8bfeb85b7213bf19061df9a70edef', 'known_history_review.json': '453d5caee74c3730c40f8d87616322e55b8bd658800f99fa7510cd3893acfaff', 'preflight.json': '823af4a9919994b7dc31116f1451ca7b908c3d225b89ee554573672ae6b881da', 'start_identity_summary.json': 'b876051f9f9a9b92696a58b91e83b2d68be9d80aa6c85daa1e91ff3da92b6661', 'tokens.pt': '4eba3efdafdd94a9917da5023249aec13fd893c07b56640695d5dc8420cc7c53'}
STORAGE_READY=True
GROUP_ALLOWANCES={'ordinary': 6395951, 'snapshot': 255670845}
PINNED_VERSIONS={'python':'3.12.3','torch':'2.11.0+cu130','transformers':'4.57.6','numpy':'1.26.4','cuda_runtime':'13.0'}

# Metadata-only final freeze candidate. Launch holds above remain untouched.
REVIEWED_DONOR_HOOK_SHA='1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28'
AUDITED_INPUT_HASHES={
 "data_manifest.json": "eb172817f41d0c47d2b746c7d074b5331bcce96d49dd7aee2219cc3ac9222227",
 "exclusion_inventory.json": "5fcb3ee8386fb2ea34141cbf8a9904375a00db0c39b6e4736ea1d0ee8cec9941",
 "known_history_chain.json": "2ac54ea99d9e991434d364e19ceb483440e8bfeb85b7213bf19061df9a70edef",
 "known_history_review.json": "453d5caee74c3730c40f8d87616322e55b8bd658800f99fa7510cd3893acfaff",
 "preflight.json": "823af4a9919994b7dc31116f1451ca7b908c3d225b89ee554573672ae6b881da",
 "start_identity_summary.json": "b876051f9f9a9b92696a58b91e83b2d68be9d80aa6c85daa1e91ff3da92b6661",
 "tokens.pt": "4eba3efdafdd94a9917da5023249aec13fd893c07b56640695d5dc8420cc7c53"
}
REVIEWED_GROUP_ALLOWANCES={'ordinary':6395951,'snapshot':255670845}
REVIEWED_FILE_PINS={'GPT2_L11_KEY_VALUE_DONOR_PROPOSAL_v2.md': '6238ea83f978705b0069e3b476f3e867ba95f08fbae2864d6e92827b88e2f159', 'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v2.md': '95c8e14e55f3662ae30cb80e0d18c913084364a66a966994c17dde4570a50c4b', 'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v3.md': 'e352c6c89d0d11e69efaca48ab1cf6797d02a40b2952940958c5f643f24dfdb5', 'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v4.md': '781dd5cc385adc2483e1f9df651bf2fcfc4fb5527e26bc54ff8b4975b861b89b', 'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v5.md': '990f1d0de8ce9006dfff20b55215f53b89fc82589e3d9c70a1015147263df69f', 'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v6.md': 'ef928ff2a6c4964a9126744b4770e337e0e54bd2a0fda459f1b14c1a2f3effa8', 'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_PLAN_draft.md': 'ebc7f75070826d7f20bd8e1d0c5faca40241d781cadc2dd557b4a48a7c6b779e', 'analyze_l11_key_value_donor.py': '84e789a723521b9866ccd9258ab1e6d3723e301aebbea8b8860decc3f4ed0ee2', 'analyze_query_write_timing.py': '54415feb2fade4da1569f92ba78172285eb955c1a72bccfb43ffc60edc3f760c', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_l11_key_value_donor.py': '1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_native_value_storage.py': 'b4b7f8932a93136fcccc2ad423bd2bc89fd57c84c052273f15ef36cbdd63d78e', 'gate_query_write_timing.py': 'f9f2eb64795b6b40ce19a2b6187e4dbb7263d838ff152bf2d081b06260a0f0ab', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gate_write_norm.py': '62e8b4c67a522043a6cb315b97c7cee34f84dc853cfcb4121679344b4b06445f', 'gpt2_gate_native_value.py': '7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_l11_key_value_donor_run_A/manifest.json': 'c5a29c7c0d129b06cf3f134b7bc17df22000724b908495af5117597521dcf70b', 'gpt2_l11_key_value_donor_run_A_r2/manifest.json': '52b418d965d2a98ce514f6bedf423e97fca433e971dfc386649784913bbd09c9', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'l11_key_value_donor_inputs.py': '26dad5cd0fde2c6ee4c2bfa5c984e825fb50e30a197da4ceb7098539954914f4', 'l11_key_value_donor_storage.py': '132ad35ff5b240eac0a41392a8a04f584d9383d0f3dd82fc4e7e566916a6e7eb', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'merge_l11_key_value_donor.py': '0320773adf7213fecacc3e2315c088803ddc7b832e9318713d9aa7f2ec89ef95', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'native_value_portable_draft.py': '46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4', 'post_write_norm_inputs.py': 'e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48', 'post_write_norm_selector.py': '34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4', 'post_write_norm_storage.py': 'f7d8068b85f4521c7d7409c88de0b1e9a2c6fc582238525db975c2df22ead0d3', 'query_write_timing_portable.py': '39d05580f3cf6e48e0a9ee324b55f11113d0189021ec3064b11384205bb307f1', 'review/l11_key_value_donor_numerical_audit.py': '7c65806c754dc2103e2f2384398905a2710d4508029f8c6e0cc73aba2f1b657a', 'review/l11_key_value_donor_numerical_audit_primary_model_4GiB_candidate_v7.py': 'a128d2d2be37250632776c8db437a4f1baf0fd8c8887d7383ee1cce4890b6ddc', 'review/t22_N_R0_2240MiB_candidate_v8_proof.json': 'b318c07ae5e48da230f5f1bed2df6df4640aae279e7dfdf7a50eb8aba157cb15', 'review/t22_N_primary_model_resource_v7_candidate_proof.json': '989e1e11dbeae5c74cf47b0ee35502cd473744402994730102e781dc7af26f8d', 'review/t22_R0_1152_resource_only_concept_review.json': 'a9369c272edcb4e97d2d732cee2b0d18feeee7b546028f2e70b49cfe1fafb928', 'review/t22_R0_2240MiB_recovery_concept_independent_review_v8.json': '7564d10e64a5ea917f0e95b13759c991a17c565e0a82ab36243a3b780da63679', 'review/t22_actual_input_independent_audit.json': '96df256d5dc90cf7591e20a84a6ab3947986f9d32186e6813865ad27ea0a11ce', 'review/t22_actual_input_independent_audit.py': '7ed40a2f31e684e90c147710ba2a9d789b74245c567f2bd902c159dfca1f11a0', 'review/t22_changed_ABI_CPU_fixture_continuation_1792634550651044869/report.json': 'bafa266aa3b9e3ac9ab5889ba1dc81a9e78d33036dc89bad9a92dadcaef7ead5', 'review/t22_failed_baseline_resource_recovery_independent_review_v1.json': '1396792a3a62adc99b87cb781c4a280cb9cf71ba7df2d6539dbf0db856660038', 'review/t22_first_input_exact_replay_gate_v8.py': 'ca7ba60e348278abb67db0426f0189d386f0655757b55a24053a36f0c1332537', 'review/t22_first_input_exact_replay_gate_v8_SOURCE_READY.json': 'ee3a8b68fe38029b88cb53797742a20b6aa63c9a18a744cf79c716efcffd2464', 'review/t22_first_input_replay_policy_v8.json': '09128d929afe99439dcd1bb99db4907476fa6b913f93957e4b24efec03177a7c', 'review/t22_hook_independent_source_review_v1.json': '1a0d051e4c2ea7717e3d1dd55d800dcc274214a3e55ed96a67dfcad4feb544af', 'review/t22_layout_complete_shape_map_RELEASED_v3_tail.json': '379cc4b5cb0f847664b190b81e4248b197c20f7cbeba2a6656d3d7dbb6817321', 'review/t22_layout_execution_contract_RELEASED_v3_tail.json': 'b684e90ab5a53558e709ee2c95b480682e88697bb9cc19f5699281573788d1be', 'review/t22_primary_model_resource_v7_candidates/GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v7.md': '8db1359395c1df76d13b7f239a926b795dff3b4bc69d8fb58eb46bccf437629b', 'review/t22_primary_model_resource_v7_candidates/gpt2_l11_key_value_donor.py': '873e9d54a34926673a08ed8943d44a787fb59bdc45ea04755e779b33e92ac4e8', 'review/t22_primary_model_resource_v7_candidates/original_sources/l11_key_value_donor_storage.py': '445db76be9121967e93577394b38f0fc809b434c461ed769d5dd91efbb0d2d3d', 'review/t22_primary_model_resource_v7_candidates/resource_inventory_v7.json': '438e74cd76cf3cbff13243e82bcf7475862dc4fe70e8b99a33b872d5e545b299', 'review/t22_primary_model_resource_v7_candidates/source_compatibility_v7.json': '01751ca7b6781c082419f4d00e1b4ccc3c4f2df866672419f41da126fc89f7a9', 'review/t22_primary_model_resource_v7_independent_review.json': '7dcc9501a9157c3feec4d006c90d9be7590d3652b6ec26e7b2171f29afbd82fd', 'review/t22_primary_model_resource_v8_candidates/GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v8.md': '04e8375bc28a93606d1b5c5f89ece26fd0e4615dde7d1dbcf5566ae3750cb6cf', 'review/t22_primary_model_resource_v8_candidates/original_sources/l11_key_value_donor_storage.py': '96bc2ddaeb91660aca8b73501b62f2835af8747eadb51c9d176664221f27b732', 'review/t22_primary_model_resource_v8_candidates/resource_inventory_and_source_proof_v8.json': '55280310120a429d14dc2b5f330d459efe7656533cb1c31208036e405d71eb3c', 'review/t22_primary_model_resource_v8_independent_review.json': '5ba274139ba4b61a7a099b257b96c69462e78baeaa23de9a8e4b52cc1a0c90d1', 'review/t22_root_resource_v7_option_selection.json': 'a0790228a3e93744899a41da5d8a088d8c4bcb95be429283eb53ca9fcea52d2e', 'review/t22_runner_independent_source_review_v3.json': '5cb6bf18e18e77c86fb40a4dbb6cd1485e15e2d29b66dc8d30fcc723ce821f26', 'review/t22_small_ABI_composite_independent_review_v1.json': 'fb130d7bfecb392c67dad017878709b7150306dbb54a36df429a4b860c0307ab', 'review/t22_storage_layout_composite_independent_audit.py': '8515ef18460cb1eb7fcb24fd22e0b0d0dd1e7e6dbccb183f73cf14abf58505a9', 'review/t22_symbolic_resource_inventory.json': 'bce2948f8a4e5d6a67c22f89e8d91b7fd34c133084208e9ce8e88e7aff9dcea7', 'review/t22_symbolic_resource_inventory_v2_additions.json': '73bb998a3014de4a8d5a61622916e25f573ef302ec82cb1fa86285e78aaa52cb', 'review/t22_symbolic_resource_inventory_v3_routes.json': 'b5f616fcdd0e1113da612fa46f8a6c57e6fe6b439937153c42caa7debb2ae8e8', 'review/t22_symbolic_resource_inventory_v4_journal_indices.json': '118284d400de7e0d881f35d6c9ba3f91001841781ba5f46f5527edf413d68e52', 'review/t22_symbolic_resource_inventory_v5_contiguity.json': 'c91243a4095ba7cf8d41a79e1e408cfaa462a825541103b61015bbc2c4e8327e', 'review/t22_symbolic_resource_inventory_v6_R0_retention.json': 'b12aecf2f137cb424cd4709d64306739002f097aecf569480295770d41231502', 'review/t22_tail_v6_source_resource_N_independent_review.json': 'caad7f6e6702fbc0386b5aed5dba43756d1ff5fd2f8bc1620be26e9d62dd7327', 'review/t22_writer_layout_fixture_source_v2_noncontiguous.py': '0493681434cc4fba989328af36d484ef44b28d3c010c002bf53fdddc8c88106e', 'review/t22_writer_layout_fixture_source_v3_tail.py': '1f5114eadf7a98a1f7e62c793ed016b871e400fce74aaa3eb6df56cdfb9d214d', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044', 't22_writer_layout_fixture_run_v1/report.json': '99100e0e69b7478ad3da0d91a9d935be026027c964af4884b6e00e35106699da'}
# Historical v6 witness pins are distinct from the effective v7 resource sources.
HISTORICAL_V6_RESOURCE={'storage_path': 'review/t22_primary_model_resource_v7_candidates/original_sources/l11_key_value_donor_storage.py', 'storage_sha256': '445db76be9121967e93577394b38f0fc809b434c461ed769d5dd91efbb0d2d3d', 'inventory_path': 'review/t22_symbolic_resource_inventory_v6_R0_retention.json', 'inventory_sha256': 'b12aecf2f137cb424cd4709d64306739002f097aecf569480295770d41231502', 'scope': 'Historical 3GiB layout/codec witness; no 4GiB model-process execution claim.'}
HISTORICAL_V7_RESOURCE={'storage_path': 'review/t22_primary_model_resource_v8_candidates/original_sources/l11_key_value_donor_storage.py', 'storage_sha256': '96bc2ddaeb91660aca8b73501b62f2835af8747eadb51c9d176664221f27b732', 'numerical_auditor_path': 'review/l11_key_value_donor_numerical_audit_primary_model_4GiB_candidate_v7.py', 'numerical_auditor_sha256': 'a128d2d2be37250632776c8db437a4f1baf0fd8c8887d7383ee1cce4890b6ddc', 'scope': 'Historical v7 source approval and failed A_r2 under 2GiB R0; not current v8 resource admission or a completed model panel.'}
RESOURCE_RECOVERY_PROVENANCE={'execution_type': 'planned_clean_recovery_after_preserved_resource_admission_failure', 'failed_original_manifest_path': 'gpt2_l11_key_value_donor_run_A/manifest.json', 'failed_original_manifest_sha256': 'c5a29c7c0d129b06cf3f134b7bc17df22000724b908495af5117597521dcf70b', 'preserved_failed_baseline_returns': 1, 'fresh_retained_panel_expected_returns': 776, 'cumulative_actual_returns_if_complete': 803, 'primary_model_RSS_limit_bytes': 4294967296, 'primary_model_R0_limit_bytes': 2348810240, 'pre_model_MemAvailable_minimum_bytes': 6442450944, 'historical_writer_witness_scope': 'Historical 3GiB layout/codec witness; no 4GiB model-process execution claim.', 'original_failed_baseline_is_not_reused': True, 'fresh_segment_output_names': ['gpt2_l11_key_value_donor_run_A_r3', 'gpt2_l11_key_value_donor_run_B_r3'], 'failed_second_manifest_path': 'gpt2_l11_key_value_donor_run_A_r2/manifest.json', 'failed_second_manifest_sha256': '52b418d965d2a98ce514f6bedf423e97fca433e971dfc386649784913bbd09c9', 'preserved_failed_first_input_returns': 26, 'preserved_failed_total_returns': 27, 'prior_failed_shards_are_not_reused': True, 'replay_policy_path': 'review/t22_first_input_replay_policy_v8.json', 'replay_helper_path': 'review/t22_first_input_exact_replay_gate_v8.py', 'replay_hard_gate_stage': 'after_complete_A_r3_before_authorizing_B_r3', 'efficacy_analysis_blinded_until_replay_and_full_panel_complete': True}

# Fill ONLY after the actual witness and independent receipts are reviewed.
# Each record then has exactly path/sha256/expected_status; None fails closed.
FINAL_CERTIFICATE_PINS={
 "whole_writer_composite": {
  "path": "t22_writer_layout_fixture_tail_run_v1/report.json",
  "sha256": "d17b16a9b5585e43376f4313ba7bacbd32d0911f130598b116bada2344c8cfc1",
  "expected_status": "passed_T22_storage_layout_composite_after_preserved_failure"
 },
 "whole_writer_independent_audit": {
  "path": "review/t22_storage_layout_composite_independent_audit.json",
  "sha256": "33945cba32412c51b68b81f1861a0b8badcc187b9a59387df05a304938e41ea5",
  "expected_status": "passed"
 }
}

# The hook owns complete current/cache failure operands. Normal capture retains
# only declared panels/prefixes and immutable references; never clone CPU caches.
def capture(probe,hidden,metrics,*,selected=False):
 assert set(probe.timing_current)==set(TIMING_NODES)
 assert set(probe.l11_norms)==set(L11_NORM_KEYS)
 timing_fingerprints=nested_fp(probe.timing_current)
 l11_fingerprints=nested_fp(probe.l11_current)
 l11_prefix={k:v[:,:128].clone() for k,v in probe.l11_current.items() if k!='final_hidden'}
 raw_l11={k:probe.l11_current[k] for k in ('c_attn_input','original_projection','delivered_qkv','attention_context','attention_write','block_hidden')} if selected and probe.arm in L11_PANEL else None
 timing_samples={'query_rows':torch.tensor(TIMING_ROWS),**{node:probe.timing_current[node][:,list(TIMING_ROWS)].clone() for node in TIMING_NODES}} if selected else None
 bundle={'metrics':metrics,'hidden':hidden,'source':probe.source_states,'gates':probe.gate_cache,
  'gate_norms':probe.localization_norms,'write_norms':probe.write_norms,'l11_norms':probe.l11_norms,
  'write_snapshot':probe.write_snapshot if selected and probe.arm in L10_PANEL else None,
  'l11_mask':probe.l11_effective_mask,'l11_panel':raw_l11,'timing_samples':timing_samples,'l11_prefix':l11_prefix,
  'prefix_contexts':probe.prefix_contexts,'prefix_writes':probe.prefix_writes,'returned_write':probe._returned_write,
  'operations':{'native_diag':summarize_diag(probe.diag),'native_gate_checks':copy.deepcopy(probe.gate_checks),
   'projection_records':copy.deepcopy(probe.projection_records),'context_records':copy.deepcopy(probe.context_records),
   'mask_records':copy.deepcopy(probe.mask_records),'source_fingerprints':nested_fp(probe.source_states),
   'gate_fingerprints':nested_fp(probe.gate_cache),'prefix_context_fingerprints':nested_fp(probe.prefix_contexts),
   'timing_checks':copy.deepcopy(probe.timing_checks),'timing_fingerprints':timing_fingerprints,
   'l11_records':copy.deepcopy(probe.l11_records),'l11_fingerprints':l11_fingerprints,
   'closure_checks':copy.deepcopy(probe.closure_checks),'write_checks':copy.deepcopy(probe.write_checks),
   'write_cache_key':copy.deepcopy(probe.cache_key),
   'projection_checks':probe.projection_checks,'context_checks':probe.context_checks,'query_checks':probe.query_checks}}
 probe.finish()
 return bundle

def pack(payload,arm,bundle,logits,tokens,*,suffix=False):
 metrics=dict(bundle['metrics']);metrics['final_hidden_sha256']=fingerprint(bundle['hidden'])['sha256']
 if suffix:
  payload['suffix'][arm]=metrics;payload['suffix_gates'][arm]=trim_gate(bundle['gates']);payload['suffix_gate_norms'][arm]=trim_gate_norm(bundle['gate_norms'])
  payload['suffix_write_norms'][arm]=bundle['write_norms'];payload['suffix_l11_norms'][arm]=bundle['l11_norms'];payload['suffix_operations'][arm]=bundle['operations']
 else:
  for key,source in (('metrics',metrics),('gates',bundle['gates']),('gate_norms',bundle['gate_norms']),('write_norms',bundle['write_norms']),('l11_norms',bundle['l11_norms']),('operations',bundle['operations'])):payload[key][arm]=source
  if tuple(payload['metadata'][k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT and arm in ARMS:
   if arm in L10_PANEL:payload['snapshots'][arm]=bundle['write_snapshot']
   if arm=='baseline':payload['l11_mask_reference']=bundle['l11_mask'].detach().to('cpu',copy=True)
   if arm in L11_PANEL:
    assert byte_exact(bundle['l11_mask'],payload['l11_mask_reference']),'Actual selected L11 mask differs from saved reference'
    payload['l11_panels'][arm]={**bundle['l11_panel'],'mask_fingerprint':fingerprint(bundle['l11_mask']),'mask_reference_byte_exact':True}
   payload['timing_samples'][arm]=bundle['timing_samples']
   payload['readouts'][arm]={'hidden':bundle['hidden'],'query_rows':torch.tensor(READOUT_ROWS),
    'logits':logits[list(READOUT_ROWS)].detach().cpu().clone(),'targets':tokens[[q+1 for q in READOUT_ROWS]],
    'token_sha256':fingerprint(tokens)['sha256'],'hidden_fingerprint':fingerprint(bundle['hidden'])}

def identity_check(left,right,ll,rr,kind):
 result={'kind':kind,'logits_exact':byte_exact(ll,rr),'hidden_exact':byte_exact(left['hidden'],right['hidden'])}
 for key in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):result[key+'_exact']=byte_exact(left['metrics'][key],right['metrics'][key])
 if left['returned_write'] is not None and right['returned_write'] is not None:result['write_exact']=byte_exact(left['returned_write'],right['returned_write'])
 return result

def suffix_check(left,right,ll,rr):
 check={'kind':'suffix','logits_exact':byte_exact(ll[:128],rr[:128]),'hidden_exact':byte_exact(left['hidden'][:,:128],right['hidden'][:,:128]),
  'source_exact':byte_exact(left['source'],right['source']),'gate_exact':byte_exact(trim_gate(left['gates']),trim_gate(right['gates'])),
  'gate_norm_exact':byte_exact(trim_gate_norm(left['gate_norms']),trim_gate_norm(right['gate_norms'])),
  'write_norm_exact':byte_exact({k:v[:128] for k,v in left['write_norms'].items()},{k:v[:128] for k,v in right['write_norms'].items()}),
  'context_exact':byte_exact(left['prefix_contexts'],right['prefix_contexts']),
  'write_exact':byte_exact(left['prefix_writes'],right['prefix_writes']),
  'l11_prefix_exact':byte_exact(left['l11_prefix'],right['l11_prefix']),
  'l11_norm_exact':byte_exact({k:v[:128] for k,v in left['l11_norms'].items()},{k:v[:128] for k,v in right['l11_norms'].items()}),
  'mask_exact':byte_exact(left['operations']['mask_records'],right['operations']['mask_records'])}
 for key in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):
  n=127 if key=='loss' else 128;check[key+'_exact']=byte_exact(left['metrics'][key][:n],right['metrics'][key])
 return check

def passed(check):return all(v for k,v in check.items() if k.endswith(('_exact','_passed')) and isinstance(v,bool))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('config','inputs','calibration-reference','out'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--segment',choices=tuple(SEGMENTS),required=True);ap.add_argument('--previous-segment',type=Path);a=ap.parse_args()
 assert (a.previous_segment is None)==(a.segment=='A'),'B requires completed A predecessor; A accepts no predecessor'
 for source in (a.inputs,a.calibration_reference,*(() if a.previous_segment is None else (a.previous_segment,))):assert not a.out.resolve().is_relative_to(source.resolve()),'Output inside immutable input/reference/predecessor'
 writer=EvidenceWriter(a.out,outstanding_other_caps=NORMAL_LIMIT*2)
 started=time.time();model=probe=identity_probe=None;attempts=observed_count=returned_count=processed=main_count=suffix_count=anchor_count=0
 active={};last_attempt={};last_returned={};last_processed={};tokens=None;latest_logits=latest_hidden=latest_metrics=None;scoring_status='not_started';payload=None;group_name=None;group_committed=True
 pair_left=pair_right=pair_logits=None
 raw_output_owner=None;last_unhooked=False
 runner_owners={}
 control_counts={k:0 for k in ('current','suffix','donor_later','recipient_upstream','KV_closure','source_q0','anchors')}
 observed_identity={};observed_current=False;observation_origins={'hook_model_end':0,'runner_after_return':0}
 meta={'study':'l11_key_value_donor','status':'preparing','phase':'setup','segment':a.segment,'execution_type':'planned_segment',
  'model_loaded':False,'model_load_attempted':False,'model_object_returned':False,'calibration_forwards':0,'normal_limit_bytes':NORMAL_LIMIT,'failure_limit_bytes':FAILURE_LIMIT,'disk_reserve_bytes':DISK_RESERVE,
  'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},'boundaries':list(BOUNDARIES),'arms':list(ARMS),
  'expected_model_returns':388,'expected_boundary_shards':16,'started_utc':time.strftime('%FT%TZ',time.gmtime())}
 def progress():return {'phase':meta['phase'],'forward_attempts':attempts,'forward_passes':returned_count,'fully_processed_forwards':processed,
  'evaluation_forwards':main_count,'suffix_forwards':suffix_count,'anchor_forwards':anchor_count,'active':active,
  'model_outputs_observed':observed_count,'observation_origins':dict(observation_origins),'last_output_observed':observed_identity,'control_counts':dict(control_counts),'last_attempted':last_attempt,'last_returned':last_returned,'last_processed':last_processed}
 def resource_owners():
  # Explicit existing CPU producer owners, never model parameters/GPU baseline
  # logits and never a cloned second failure cache. Backing intervals are used
  # only for RAM admission; serialization still emits every occurrence.
  owned=[payload,runner_owners,tokens,latest_hidden,latest_metrics,pair_left,pair_right,pair_logits]
  if probe is not None:
   owned.extend(getattr(probe,name,{}) for name in ('failure','write_cache','write_snapshot','write_norms','write_checks',
    'timing_cache','timing_current','timing_snapshot','baseline_l11','donor_cache','closure_references','l11_current','l11_norms'))
   owned.extend(getattr(probe.native,name,{}) for name in ('snapshots','pending','source_states','gate_cache','localization_norms',
    'effective_masks','prefix_contexts','diag','failure'))
  return owned
 writer.set_resource_owners(resource_owners)
 def resources():return {'wall_seconds':time.time()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
  'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}
 def resources_guard(stage):
  observed=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  assert observed<=4*1024**2,('CPU peak RSS budget',stage,observed,4*1024**2)
 def checkpoint(*,final=False,failure=False):
  meta.update(progress(),**resources())
  if not failure:resources_guard('checkpoint')
  writer.checkpoint(progress(),manifest=meta,final=final,failure=failure)
 def check(value,left=None,right=None,ll=None,rr=None,*,comparison='generic'):
  nonlocal pair_left,pair_right,pair_logits
  payload['checks'].append(value)
  if not passed(value):
   if comparison=='suffix':pair_left,pair_right=suffix_operands(left,right),None
   elif comparison=='identity':
    pair_left={'hidden':left['hidden'],'metrics':left['metrics'],'returned_write':left['returned_write']}
    pair_right={'hidden':right['hidden'],'metrics':right['metrics'],'returned_write':right['returned_write']}
   elif value.get('kind')=='source_q0':pair_left,pair_right=left['source'],right['source']
   else:pair_left,pair_right=failure_bundle(left),failure_bundle(right)
   pair_logits=None if ll is None else {'left':ll.detach().cpu().clone(),'right':None if rr is None else rr.detach().cpu().clone(),'scope':comparison,'comparison_query_slice':[0,128] if comparison=='suffix' else [0,512],'right_full_actual_readout':rr is not None and len(rr)==512}
   raise AssertionError(value)
 def ev(x,boundary,arm,c_logits=None,b_logits=None,*,suffix=False,unhooked=False,anchor=False,observe_only=False,selected=False):
  nonlocal attempts,returned_count,processed,last_attempt,last_returned,last_processed,tokens,latest_logits,latest_hidden,latest_metrics,scoring_status,main_count,suffix_count,anchor_count,observed_current,observed_count,observed_identity,raw_output_owner,last_unhooked
  assert x.dtype==torch.int64 and x.shape==(512,);tokens=x.clone();latest_logits=latest_hidden=latest_metrics=None;scoring_status='not_started';observed_current=False;raw_output_owner=None;last_unhooked=unhooked
  writer.guard_known_header('before_forward:'+str(attempts+1))
  writer.journal({'event':'forward_intent','next_attempt':attempts+1,'identity':active,'token_fingerprint':fingerprint(x)})
  if not unhooked:probe.reset(boundary,arm,variant='suffix' if suffix else 'main',token_sha256=fingerprint(x)['sha256'],snapshot=selected and arm in L10_PANEL,observe_only=observe_only)
  def output_observed(event):
   nonlocal observed_count,observed_identity,observed_current
   assert not observed_current,'Duplicate actual model-end observation'
   assert event['arm']==arm and event['variant']==('suffix' if suffix else 'main') and event['actual_token_sha256']==fingerprint(x)['sha256']
   origin='runner_after_return' if unhooked else 'hook_model_end'
   observed_current=True;observed_count+=1;observation_origins[origin]+=1;observed_identity=dict(active)
   writer.journal({'event':'model_output_observed','attempt':attempts,'observed':observed_count,'observer_origin':origin,'identity':active,'hook_event':event})
  if not unhooked:probe.set_output_observer(output_observed)
  def invoked():
   nonlocal attempts,last_attempt
   writer.journal({'event':'forward_attempt','attempt':attempts+1,'identity':active});attempts+=1;last_attempt=dict(active)
  def returned():
   nonlocal returned_count,last_returned,main_count,suffix_count,anchor_count
   returned_count+=1;last_returned=dict(active)
   if anchor:anchor_count+=1
   elif suffix:suffix_count+=1
   else:main_count+=1
  profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if returned_count==0 else contextlib.nullcontext()
  with torch.inference_mode():
   with profile as profiler:
    raw_output_owner=model_forward(model,x,returned,invoked)
    # First actions after actual return bind complete actual owners. Even
    # profiler/journal/allocation failure now has reached-output evidence.
    latest_logits=raw_output_owner.logits[0].detach()
    latest_hidden=raw_output_owner.hidden_states[-1].detach()
   writer.journal({'event':'model_returned','attempt':attempts,'completed':returned_count,'identity':active})
   if unhooked:output_observed({'arm':arm,'variant':'suffix' if suffix else 'main','actual_token_sha256':fingerprint(x)['sha256'],'scope':'Actual output captured by runner AFTER model_returned; no model-end hook installed'})
   else:assert observed_current,'Hooked call returned without owned model-end observation'
   if returned_count==1:
    events=[{'operator':e.key,'count':e.count} for e in profiler.key_averages() if 'scaled_dot_product' in e.key];meta['sdpa_profiler_events']=events
    assert any('_scaled_dot_product' in e['operator'] for e in events),'Expected observed SDPA backend'
   actual_hidden=latest_hidden
   latest_hidden=actual_hidden.to('cpu',copy=True) if unhooked else probe.l11_current['final_hidden']
   assert byte_exact(latest_hidden,actual_hidden);raw_output_owner=None;del actual_hidden
   scoring_status='scoring_started_not_completed';metrics=metric_rows(latest_logits,x,c_logits,b_logits,suffix=suffix);latest_metrics=metrics;scoring_status='completed'
  if unhooked:bundle={'metrics':metrics,'hidden':latest_hidden,'source':{},'gates':{},'gate_norms':{},'write_norms':{},'l11_norms':{},'l11_prefix':{},'snapshots':{},'write_snapshot':{},'prefix_contexts':{},'prefix_writes':None,'returned_write':None,'operations':{}}
  else:bundle=capture(probe,latest_hidden,metrics,selected=selected)
  resources_guard('after_capture_before_processed')
  processed+=1;last_processed=dict(active);writer.journal({'event':'forward_completed','attempt':attempts,'completed':returned_count,'processed':processed,'identity':active,'hidden_fingerprint':fingerprint(latest_hidden),'logits_fingerprint':fingerprint(latest_logits)})
  return bundle,latest_logits
 try:
  checkpoint();assert INPUT_HASHES and STORAGE_READY and GROUP_ALLOWANCES,'Launch blocked: independent T22 input/storage/source freeze incomplete'
  assert DONOR_HOOK_SHA,'Independent hook/source freeze required'
  # Metadata-only admission: no certificate is an inferred model approval.
  assert DONOR_HOOK_SHA==REVIEWED_DONOR_HOOK_SHA and INPUT_HASHES==AUDITED_INPUT_HASHES
  assert GROUP_ALLOWANCES==REVIEWED_GROUP_ALLOWANCES
  assert all(record is not None for record in FINAL_CERTIFICATE_PINS.values()),'Final resource/writer certificates unresolved'
  freeze_file_pins=dict(REVIEWED_FILE_PINS);final_certificates={}
  for role,record in FINAL_CERTIFICATE_PINS.items():
   assert set(record)=={'path','sha256','expected_status'} and all(isinstance(record[k],str) and record[k] for k in record)
   name=record['path'];path=Path(name)
   assert not path.is_absolute() and all(p not in ('','.','..') for p in name.split('/'))
   assert name not in freeze_file_pins,'Certificate path must be one separately reviewed occurrence'
   freeze_file_pins[name]=record['sha256']
   assert (ROOT/name).is_file() and not (ROOT/name).is_symlink() and sha(ROOT/name)==record['sha256']
   final_certificates[role]=json.loads((ROOT/name).read_text())
   assert final_certificates[role]['status']==record['expected_status']
  for name,h in freeze_file_pins.items():
   assert (ROOT/name).is_file() and not (ROOT/name).is_symlink() and sha(ROOT/name)==h
  input_audit=json.loads((ROOT/'review/t22_actual_input_independent_audit.json').read_text())
  assert input_audit['status']=='passed' and input_audit['actual_preparation_sha256']==INPUT_HASHES
  assert input_audit['audited_data_manifest_sha256']==INPUT_HASHES['data_manifest.json']
  resource_review=json.loads((ROOT/'review/t22_tail_v6_source_resource_N_independent_review.json').read_text())
  assert resource_review['status']=='passed_source_only_tail_resource_N_review_not_execution'
  for name in ('l11_key_value_donor_storage.py','GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v6.md','review/t22_symbolic_resource_inventory_v6_R0_retention.json'):
   expected=HISTORICAL_V6_RESOURCE['storage_sha256'] if name=='l11_key_value_donor_storage.py' else REVIEWED_FILE_PINS[name]
   assert resource_review['accepted_sha256'][name]==expected
  writer_receipt=final_certificates['whole_writer_composite']
  assert writer_receipt['source_sha256']==REVIEWED_FILE_PINS['review/t22_writer_layout_fixture_source_v3_tail.py']
  assert writer_receipt['execution_contract_sha256']==REVIEWED_FILE_PINS['review/t22_layout_execution_contract_RELEASED_v3_tail.json']
  assert writer_receipt['model_forwards']==0 and writer_receipt['checkpoint_or_weight_access'] is False
  writer_audit=final_certificates['whole_writer_independent_audit']
  assert writer_audit['status']=='passed'
  assert writer_audit['audited_composite_report_sha256']==FINAL_CERTIFICATE_PINS['whole_writer_composite']['sha256']
  assert writer_audit['audited_storage_source_sha256']==HISTORICAL_V6_RESOURCE['storage_sha256']
  assert writer_audit['audited_resource_inventory_sha256']==HISTORICAL_V6_RESOURCE['inventory_sha256']
  # Current resource approval is source-only; the actual writer certificate above is historical.
  assert REVIEWED_FILE_PINS[HISTORICAL_V6_RESOURCE['storage_path']]==HISTORICAL_V6_RESOURCE['storage_sha256']
  assert REVIEWED_FILE_PINS[HISTORICAL_V6_RESOURCE['inventory_path']]==HISTORICAL_V6_RESOURCE['inventory_sha256']
  recovery_review=json.loads((ROOT/'review/t22_primary_model_resource_v7_independent_review.json').read_text())
  assert recovery_review['status']=='passed'
  for field,name in (
   ('producer_candidate_sha256','review/t22_primary_model_resource_v7_candidates/gpt2_l11_key_value_donor.py'),
   ('storage_candidate_sha256','l11_key_value_donor_storage.py'),
   ('numerical_auditor_candidate_sha256','review/l11_key_value_donor_numerical_audit.py'),
   ('resource_inventory_sha256','review/t22_primary_model_resource_v7_candidates/resource_inventory_v7.json'),
   ('resource_addendum_sha256','review/t22_primary_model_resource_v7_candidates/GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v7.md')):
   expected=HISTORICAL_V7_RESOURCE['storage_sha256'] if field=='storage_candidate_sha256' else HISTORICAL_V7_RESOURCE['numerical_auditor_sha256'] if field=='numerical_auditor_candidate_sha256' else REVIEWED_FILE_PINS[name]
   assert recovery_review[field]==expected
  assert recovery_review['historical_evidence']['audited_storage_source_sha256']==HISTORICAL_V6_RESOURCE['storage_sha256']
  assert recovery_review['historical_evidence']['audited_resource_inventory_sha256']==HISTORICAL_V6_RESOURCE['inventory_sha256']
  assert recovery_review['historical_evidence']['audited_composite_report_sha256']==FINAL_CERTIFICATE_PINS['whole_writer_composite']['sha256']
  assert recovery_review['historical_evidence']['independent_audit_sha256']==FINAL_CERTIFICATE_PINS['whole_writer_independent_audit']['sha256']
  failed_original=json.loads((ROOT/RESOURCE_RECOVERY_PROVENANCE['failed_original_manifest_path']).read_text())
  assert REVIEWED_FILE_PINS[RESOURCE_RECOVERY_PROVENANCE['failed_original_manifest_path']]==RESOURCE_RECOVERY_PROVENANCE['failed_original_manifest_sha256']
  assert failed_original['status']=='failed' and failed_original['segment']=='A'
  assert all(failed_original[k]==1 for k in ('forward_attempts','model_outputs_observed','forward_passes','fully_processed_forwards','evaluation_forwards'))
  assert failed_original['suffix_forwards']==failed_original['anchor_forwards']==0 and all(v==0 for v in failed_original['control_counts'].values())
  assert failed_original['last_processed']['arm']=='baseline' and failed_original['last_processed']['phase']=='main'
  assert failed_original['source_hashes']['l11_key_value_donor_storage.py']==HISTORICAL_V6_RESOURCE['storage_sha256']
  assert recovery_review['recovery_accounting']['failed_original_manifest_sha256']==RESOURCE_RECOVERY_PROVENANCE['failed_original_manifest_sha256']
  # Effective v8 is a separately reviewed R0 literal change; v6/v7 evidence remains historical.
  assert REVIEWED_FILE_PINS[HISTORICAL_V7_RESOURCE['storage_path']]==HISTORICAL_V7_RESOURCE['storage_sha256']
  assert REVIEWED_FILE_PINS[HISTORICAL_V7_RESOURCE['numerical_auditor_path']]==HISTORICAL_V7_RESOURCE['numerical_auditor_sha256']
  effective_resource_review=json.loads((ROOT/'review/t22_primary_model_resource_v8_independent_review.json').read_text())
  assert effective_resource_review['status']=='passed'
  for field,name in (
   ('storage_candidate_sha256','l11_key_value_donor_storage.py'),
   ('numerical_auditor_candidate_sha256','review/l11_key_value_donor_numerical_audit.py'),
   ('resource_inventory_sha256','review/t22_primary_model_resource_v8_candidates/resource_inventory_and_source_proof_v8.json'),
   ('resource_addendum_sha256','review/t22_primary_model_resource_v8_candidates/GPT2_L11_KEY_VALUE_DONOR_RESOURCE_ADDENDUM_v8.md')):
   assert effective_resource_review[field]==REVIEWED_FILE_PINS[name]
  failed_second=json.loads((ROOT/RESOURCE_RECOVERY_PROVENANCE['failed_second_manifest_path']).read_text())
  assert REVIEWED_FILE_PINS[RESOURCE_RECOVERY_PROVENANCE['failed_second_manifest_path']]==RESOURCE_RECOVERY_PROVENANCE['failed_second_manifest_sha256']
  assert failed_second['status']=='failed' and failed_second['segment']=='A'
  assert all(failed_second[k]==26 for k in ('forward_attempts','model_outputs_observed','forward_passes','fully_processed_forwards'))
  assert failed_second['evaluation_forwards']==failed_second['suffix_forwards']==12 and failed_second['anchor_forwards']==2
  assert failed_second['control_counts']=={'current':2,'suffix':12,'donor_later':4,'recipient_upstream':12,'KV_closure':4,'source_q0':22,'anchors':2}
  assert failed_second['source_hashes']['l11_key_value_donor_storage.py']==HISTORICAL_V7_RESOURCE['storage_sha256']
  policy=json.loads((ROOT/RESOURCE_RECOVERY_PROVENANCE['replay_policy_path']).read_text())
  assert policy['status']=='prospective_fixed_exact_replay_policy_before_A_r3' and policy['stage']==RESOURCE_RECOVERY_PROVENANCE['replay_hard_gate_stage']
  assert policy['comparison_is_hard_gate'] is True and policy['efficacy_analysis_blinded_until_replay_and_full_panel_complete'] is True
  assert policy['old_failed_A_r2_manifest']=={'path':RESOURCE_RECOVERY_PROVENANCE['failed_second_manifest_path'],'sha256':RESOURCE_RECOVERY_PROVENANCE['failed_second_manifest_sha256']}
  assert policy['required_comparator_source_relative_path']==RESOURCE_RECOVERY_PROVENANCE['replay_helper_path']
  assert policy['accounting']['prior_failed_returns']==27 and policy['accounting']['retained_planned_forwards']==776 and policy['accounting']['cumulative_if_complete']==803
  assert policy['old_first_shard']['actual_tensor_occurrences_from_metadata']==1366
  meta['resource_recovery']=dict(RESOURCE_RECOVERY_PROVENANCE)
  writer.guard('complete_planned_caps_before_model')
  assert sha(ROOT/'GPT2_L11_KEY_VALUE_DONOR_PROPOSAL_v2.md')==PROPOSAL_SHA
  assert sha(ROOT/'GPT2_L11_KEY_VALUE_DONOR_RESOURCE_PLAN_draft.md')==RESOURCE_PLAN_SHA
  assert sha(ROOT/'review/t22_symbolic_resource_inventory.json')==RESOURCE_INVENTORY_SHA
  assert sha(ROOT/'gate_l11_key_value_donor.py')==DONOR_HOOK_SHA
  assert sha(ROOT/'gate_native_value.py')==NATIVE_HOOK_SHA
  assert all(sha(a.inputs/name)==h for name,h in INPUT_HASHES.items())
  pre=json.loads((a.inputs/'preflight.json').read_text());assert pre['status']=='passed' and pre['model_forwards']==0 and pre['evaluation_passages']==16 and pre['evaluation_inputs']==32
  dm=json.loads((a.inputs/'data_manifest.json').read_text());items=torch.load(a.inputs/'tokens.pt',weights_only=True,map_location='cpu')
  assert len(items)==16 and {(i['corpus'],i['window']) for i in items}=={(c,w) for c in CORPORA for w in range(4)}
  for item in items:
   record=next(r for r in dm['passages'] if (r['corpus'],r['window'])==(item['corpus'],item['window']))
   assert item['tokens'].dtype==torch.int64 and item['tokens'].shape==(512,) and fingerprint(item['tokens'])['sha256']==record['window_token_sha256']
   bos=item['tokens'].clone();bos[0]=dm['literals']['start_special']['id'];assert fingerprint(bos)['sha256']==record['bos_window_token_sha256']
  for name,h in INPUT_HASHES.items():shutil.copyfile(a.inputs/name,a.out/name);assert sha(a.out/name)==h
  ref=a.calibration_reference;assert sha(ref/'manifest.json')==dm['calibration_reference_manifest_sha256']==T18_MANIFEST_SHA
  refmeta=json.loads((ref/'manifest.json').read_text());assert refmeta['status']=='completed' and refmeta['calibration_map_digests']['F']==F_DIGEST
  for name in ('manifest.json','calibration_maps.npz','calibration_directions.npz'):
   if name!='manifest.json':assert sha(ref/name)==dm['calibration_reference_files'][name]
  (a.out/'calibration_reference').mkdir(exist_ok=True)
  for name in ('manifest.json','calibration_maps.npz','calibration_directions.npz'):shutil.copyfile(ref/name,a.out/'calibration_reference'/name);assert sha(a.out/'calibration_reference'/name)==sha(ref/name)
  maps=read_npz(ref/'calibration_maps.npz');directions=read_npz(ref/'calibration_directions.npz')
  from gpt2_gate_subspace import maps_digest
  from run_transfer import modelhash
  from native_prefix_diagnostic import precision_settings
  assert maps_digest(maps['F'])==F_DIGEST;alpha=maps['F'][10]['alpha'];u=directions['F'][10]['u'];v=directions['F'][10]['v']
  closure=source_closure(ROOT,['gpt2_l11_key_value_donor.py','gate_l11_key_value_donor.py','l11_key_value_donor_storage.py','l11_key_value_donor_inputs.py','post_write_norm_selector.py','analyze_l11_key_value_donor.py','GPT2_L11_KEY_VALUE_DONOR_PROPOSAL_v2.md','GPT2_L11_KEY_VALUE_DONOR_RESOURCE_PLAN_draft.md','review/t22_symbolic_resource_inventory.json','gpt2_layer_writes.py','merge_l11_key_value_donor.py']+list(freeze_file_pins))
  assert closure=={**freeze_file_pins,'gpt2_l11_key_value_donor.py':sha(ROOT/'gpt2_l11_key_value_donor.py')},'Unreviewed dependency set/hash'
  for name,h in closure.items():
   dest=a.out/'frozen_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);assert sha(dest)==h
  cfg=json.loads(a.config.read_text())
  for name,h in METADATA_PINS.items():
   assert sha(ROOT/name)==h;dest=a.out/'metadata_dependencies'/name.removeprefix('../').replace('/','__');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);assert sha(dest)==h
  old=json.loads((ROOT/'gpt2_run/manifest.json').read_text());spec=json.loads((ROOT.parent/'superweight-causal-circuits/manifests/transfer_models.json').read_text())['gpt2'];configured=cfg['models']['gpt2'];modelpath=Path(configured['path'])
  assert configured['model_id']==old['model_id']==refmeta['model_id']==spec['model_id'] and configured['revision']==old['revision']==refmeta['revision']==spec['revision']
  for name,entry in spec['checkpoint_files'].items():assert sha(modelpath/name)==entry['sha256']
  torch.set_num_threads(4);torch.manual_seed(20261021);settings=precision_settings();assert settings==refmeta['precision_settings_before']
  runtime={'python':platform.python_version(),'torch':str(torch.__version__),'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda};assert runtime==PINNED_VERSIONS
  arithmetic_runtime=runtime_behavior();freeze=hashlib.sha256(json.dumps({'sources':closure,'inputs':INPUT_HASHES,'metadata':METADATA_PINS,'runtime':runtime,'precision':settings},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  meta.update(source_hashes=closure,input_dependencies=INPUT_HASHES,metadata_dependencies=METADATA_PINS,freeze_digest=freeze,model_id=old['model_id'],revision=old['revision'],dtype='float32',attention='sdpa',runtime_versions=runtime,precision_settings_before=settings,arithmetic_runtime=arithmetic_runtime,raw_panel_contract={'L10':list(L10_PANEL),'L11':list(L11_PANEL),'end':list(ARMS),'readout_rows':list(READOUT_ROWS),'timing_rows':list(TIMING_ROWS)},slice_contract={'earlier':[1,127],'later':[127,512],'QKV_width':768,'layout':'contiguous_Q_K_V'},calibration_map_digest=F_DIGEST,calibration_fit=False,physical_execution_lengths=[512])
  previous_binding=None
  if a.segment=='B':
   from merge_l11_key_value_donor import validate,EQUAL_FIELDS
   previous_manifest,previous_inventory=validate(a.previous_segment,'A')
   for name in EQUAL_FIELDS:assert previous_manifest[name]==meta[name],('A predecessor differs',name)
   assert previous_manifest['model_hash_before']==previous_manifest['model_hash_after']==old['model_hash_before'],'A predecessor parameter identity differs from pinned checkpoint'
   previous_binding={'manifest_sha256':sha(a.previous_segment/'manifest.json'),'numeric_inventory_sha256':sha(a.previous_segment/'numeric_inventory.json')}
   meta['previous_segment_A']=previous_binding
   (a.out/'previous_segment_A').mkdir()
   for name in ('manifest.json','numeric_inventory.json'):shutil.copyfile(a.previous_segment/name,a.out/'previous_segment_A'/name);assert sha(a.out/'previous_segment_A'/name)==previous_binding[name.replace('.json','_sha256')]
   # Only a verified complete predecessor releases its unused future cap.
   writer.outstanding_other_caps=NORMAL_LIMIT
  checkpoint()
  resources_guard('before_model_load');meta['header_admission_before_model']=writer.guard_known_header('before_model_load')
  meta['host_mem_available_before_model_kib']=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
  meta['host_mem_available_minimum_bytes']=6*1024**3
  assert meta['host_mem_available_before_model_kib']*1024>=meta['host_mem_available_minimum_bytes'],'Pre-model host headroom below6GiB'
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not gpu,('Competing GPU computation',gpu)
  meta['gpu_compute_processes_before_load']=gpu;meta['gpu_hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
  from transformers import AutoModelForCausalLM
  torch.cuda.reset_peak_memory_stats();meta['model_load_attempted']=True
  model=AutoModelForCausalLM.from_pretrained(modelpath,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True);meta['model_object_returned']=True;model=model.eval().requires_grad_(False)
  meta['model_loaded']=True;resources_guard('after_model_load');meta['header_admission_after_model']=writer.guard_known_header('after_model_load_before_first_forward')
  before=modelhash(model);assert before==old['model_hash_before'];versions={name:p._version for name,p in model.named_parameters()};meta['model_hash_before']=before
  probe=GateL11KeyValueDonor(model,alpha,u,v,F_DIGEST,freeze)
  for item in items:
   if item['corpus'] not in SEGMENTS[a.segment]:continue
   for identity in ('natural','bos'):
    x=item['tokens'].clone()
    if identity=='bos':x[0]=dm['literals']['start_special']['id']
    base={'corpus':item['corpus'],'window':item['window'],'identity':identity,'length':512,'token_sha256':fingerprint(x)['sha256']};rotated=x.clone();rotated[128:]=x[128:].roll(1);c_bundle=c_logits=None
    for boundary in BOUNDARIES:
     group_name=f"{base['corpus']}__{base['window']}__{identity}__{boundary}";snapshot=tuple(base[k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT
     writer.guard('boundary:'+group_name,GROUP_ALLOWANCES['snapshot' if snapshot else 'ordinary']+METADATA_ALLOWANCE)
     payload=empty_payload({**base,'boundary':boundary});group_committed=False;probe.start_input(base,boundary);own_base=own_logits=own_suffix_base=own_suffix_logits=None;meta['phase']='evaluation'
     for arm in ARMS:
      if arm in ('remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v'):assert set(probe.closure_checks)==set(CLOSURE_KEYS) and all(r['passed'] for r in probe.closure_checks.values()),'Four actual closures required before singles'
      active={**base,'boundary':boundary,'arm':arm,'phase':'main'};trial,ll=ev(x,boundary,arm,c_logits,own_logits,selected=snapshot)
      if arm=='baseline':
       own_base,own_logits=trial,ll;runner_owners['main_baseline']=own_base;payload['source_reference']=trial['source']
       if boundary=='000':c_bundle,c_logits=trial,ll
      runner_owners['main_trial']=trial
      trial['metrics']=metric_rows(ll,x,c_logits,own_logits);pack(payload,arm,trial,ll,x)
      if boundary=='000' and arm=='baseline' and identity=='natural' and item['window']==0:
       saved_active=dict(active)
       for anchor in ('native_repeat','native_identity'):
        active={**base,'boundary':boundary,'arm':anchor,'phase':'anchor'}
        if anchor=='native_repeat':
         with probe.unhooked():candidate,cl=ev(x,'000','baseline',c_logits,c_logits,unhooked=True,anchor=True)
        else:
         from gpt2_layer_writes import LayerWrites
         selection=json.loads((ROOT/'gpt2_run/selection.json').read_text())['selected'];identity_probe=LayerWrites(model,selection,[447,138]);identity_probe.reset_layers(E=False,blocked=[],special='identity');probe.allow_identity_hooks(identity_probe.handles)
         try:candidate,cl=ev(x,'000','baseline',c_logits,c_logits,anchor=True,observe_only=True)
         finally:
          identity_probe.close();identity_probe=None
          if not probe.failed:probe.allow_identity_hooks(())
        pack(payload,anchor,candidate,cl,x);check(identity_check(c_bundle,candidate,c_logits,cl,anchor),c_bundle,candidate,c_logits,cl,comparison='identity');control_counts['anchors']+=1;del candidate,cl
       active=saved_active
      if arm!='baseline':
       check({'kind':'source_q0','arm':arm,'variant':'main','source_exact':byte_exact(trial['source'],own_base['source'])},own_base,trial);control_counts['source_q0']+=1
      if arm=='current':check({'arm':arm,'variant':'main',**identity_check(own_base,trial,own_logits,ll,'current')},own_base,trial,own_logits,ll,comparison='identity');control_counts['current']+=1
      if arm in DONORS:
       assert trial['operations']['timing_checks']['donor_later_passed'];control_counts['donor_later']+=1
      if arm in RECIPIENTS:
       assert trial['operations']['l11_records']['recipient_upstream_passed'];control_counts['recipient_upstream']+=1
      if arm.endswith('_l11_kv'):
       assert probe.closure_checks[('remove' if arm.startswith('remove') else 'orth')+'/main']['passed'];control_counts['KV_closure']+=1
      active={**base,'boundary':boundary,'arm':arm,'phase':'suffix'};altered,al=ev(rotated,boundary,arm,c_logits,own_logits,suffix=True);pack(payload,arm,altered,al,rotated,suffix=True)
      if arm=='baseline':
       own_suffix_base,own_suffix_logits=altered,al;runner_owners['suffix_baseline']=own_suffix_base
      if arm!='baseline':
       check({'kind':'source_q0','arm':arm,'variant':'suffix','source_exact':byte_exact(altered['source'],own_suffix_base['source'])},own_suffix_base,altered);control_counts['source_q0']+=1
      if arm=='current':check({'arm':arm,'variant':'suffix',**identity_check(own_suffix_base,altered,own_suffix_logits,al,'current')},own_suffix_base,altered,own_suffix_logits,al,comparison='identity');control_counts['current']+=1
      if arm in DONORS:
       assert altered['operations']['timing_checks']['donor_later_passed'];control_counts['donor_later']+=1
      if arm in RECIPIENTS:
       assert altered['operations']['l11_records']['recipient_upstream_passed'];control_counts['recipient_upstream']+=1
      if arm.endswith('_l11_kv'):
       assert probe.closure_checks[('remove' if arm.startswith('remove') else 'orth')+'/suffix']['passed'];control_counts['KV_closure']+=1
      check({'arm':arm,**suffix_check(trial,altered,ll,al)},trial,altered,ll[:128],al,comparison='suffix');control_counts['suffix']+=1
      if arm!='baseline':
       runner_owners.pop('main_trial',None);del altered,al,trial,ll
     assert probe.order_index==24 and set(payload['suffix'])==set(ARMS)
     expected=set(ARMS)|({'native_repeat','native_identity'} if boundary=='000' and identity=='natural' and item['window']==0 else set());assert set(payload['metrics'])==expected
     writer.save('inputs/'+group_name+'.npz',payload);group_committed=True;payload=None;runner_owners.clear();own_base=own_logits=own_suffix_base=own_suffix_logits=latest_logits=None;checkpoint();print('L11_KEY_VALUE_DONOR_DONE',group_name,returned_count,flush=True)
     if boundary=='000':c_bundle=None
    c_bundle=c_logits=None
  assert attempts==returned_count==processed==observed_count==388 and main_count==suffix_count==192 and anchor_count==4 and len(writer.inventory)==16
  assert observation_origins=={'hook_model_end':386,'runner_after_return':2}
  assert control_counts=={'current':32,'suffix':192,'donor_later':64,'recipient_upstream':192,'KV_closure':64,'source_q0':352,'anchors':4}
  assert all(sha(a.inputs/name)==h for name,h in INPUT_HASHES.items()) and all(sha(ROOT/name)==h for name,h in closure.items()) and all(sha(ROOT/name)==h for name,h in METADATA_PINS.items())
  assert sha(ref/'manifest.json')==T18_MANIFEST_SHA and all(sha(ref/name)==dm['calibration_reference_files'][name] for name in ('calibration_maps.npz','calibration_directions.npz'))
  assert maps_digest(maps['F'])==F_DIGEST and precision_settings()==settings and runtime_behavior()==arithmetic_runtime
  if previous_binding is not None:
   assert validate(a.previous_segment,'A')==(previous_manifest,previous_inventory)
   assert previous_binding=={'manifest_sha256':sha(a.previous_segment/'manifest.json'),'numeric_inventory_sha256':sha(a.previous_segment/'numeric_inventory.json')}
  after=modelhash(model);assert after==before and versions=={name:p._version for name,p in model.named_parameters()}
  meta.update(status='completed',phase='completed',model_hash_after=after,parameter_versions_unchanged=True,precision_settings_after=precision_settings(),all_final_guards_passed=True,completed_utc=time.strftime('%FT%TZ',time.gmtime()));checkpoint(final=True)
 except BaseException as error:
  error_text=str(error);error_type=type(error).__name__;error.__traceback__=None
  preservation={}
  if probe is not None and probe.failure is None:
   try:probe.capture_failure(error)
   except BaseException as capture_error:preservation['hook_capture_error']=repr(capture_error)
  try:
   if payload is not None and not group_committed:preservation['partial_boundary']=writer.save('inputs/'+group_name+'__partial.npz',payload)
  except BaseException as save_error:preservation['partial_boundary_error']=repr(save_error)
  capsule={'error_type':error_type,'error':error_text,'progress':progress(),'current_attempt_tokens':tokens,'current_input_identity':active,
   'hook_failure':None if probe is None else probe.failure,'left':pair_left,'right':pair_right,
   'comparison_logits':pair_logits,
   'current_actual_readout':None if pair_logits is not None or latest_logits is None else latest_logits.detach().to('cpu',copy=True),
   'current_hidden':latest_hidden.detach().to('cpu',copy=True) if latest_hidden is not None and (probe is None or (last_unhooked and pair_logits is None)) else None,
   'current_metrics':latest_metrics,'scoring_status':scoring_status,
   'uncommitted_boundary':payload if 'partial_boundary_error' in preservation else None}
  try:preservation['failure_capsule']=writer.save('failure_state.npz',capsule,failure=True)
  except BaseException as save_error:preservation['failure_capsule_error']=repr(save_error)
  preservation['journal_preservation_errors']=writer.failure_journal_errors
  meta.update(progress(),**resources(),status='failed',phase='failed',failure_type=error_type,failure_message=error_text,preservation=preservation)
  try:checkpoint(final=True,failure=True)
  except BaseException as save_error:
   if writer.owns():json_write(a.out/'checkpoint_failure.json',{'error':repr(save_error),'known_progress':progress()})
  raise
 finally:
  if probe is not None:probe.close()
  if identity_probe is not None:identity_probe.close()
if __name__=='__main__':main()
