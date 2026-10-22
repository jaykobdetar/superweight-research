"""T23 research-only portable export and reproduction SOURCE HOLD.

Metadata/JSON validation never loads numerical libraries. Optional numerical
reanalysis and tokenizer rebuild are separate explicit routes; no model launch,
checkpoint read, selector search beyond frozen reconstruction, cleanup or upload.
"""
import argparse,ast,copy,hashlib,importlib,json,math,os,re,sys
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parent
BASE_VALIDATOR_SHA='46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4'
assert hashlib.sha256((ROOT/'native_value_portable_draft.py').read_bytes()).hexdigest()==BASE_VALIDATOR_SHA
import native_value_portable_draft as base
assert Path(base.__file__).resolve()==(ROOT/'native_value_portable_draft.py').resolve()
sha,read,put,copy_exact,outside_sources,value_sha,clean,safe=base.sha,base.read,base.put,base.copy_exact,base.outside_sources,base.value_sha,base.clean,base.safe
PHYSICAL=base.PHYSICAL;REFERENCE=base.REFERENCE;OWNED=None
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
CELLS=('PP','PC','CP','CC')
ARMS=('baseline','current')+tuple(c+'_'+op for c in CELLS for op in ('remove_full','orth_full','remove_write_matched_early','orth_write_matched_early'))
MATCHED=tuple(a for a in ARMS if 'write_matched_early' in a)
METRICS=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')
GROUPS=('metrics','suffix','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms')
METADATA_NAMES={'data_manifest.template.json','exclusion_inventory.json','start_identity.template.json','history_attestation.json','preparation.template.json'}
ANALYSIS_NAME='calibration_world_evaluation_analysis.json'
# These remain literal HOLDs until completed N/P/R and finite privacy review.
EXPORT_READY=True
METADATA_STAGE_READY=True
STAGE_BINDING_REVIEW_SHA='99c665f034f4ec7e60d4d5ebb7362de170fe64a0ddca5f5bf1428e3cabd096a8'
CANONICAL={'manifest.json': '1b2e1f8f4c36355b297b45461c3b34143d42b30a348ec438d07a0f3a665c1382', 'numeric_inventory.json': '851bd22f549a205fe53581d5c65b75974565bb8b4ef0814fc3b269d5e8e1bb05', 'calibration_world_evaluation_analysis.json': '8dc01b8911cd95c08c9ba9c46c95a00e0cf5c54177d5bc70f7eda635bd1797b5'}
FROZEN_SOURCE_HASHES={'GPT2_CALIBRATION_WORLD_COMPARISON_PROPOSAL_draft_v2.md': 'a78e9678ffceefc85abcccc2a11cd750754efd1adb7738caa03d508f957c2e7d', 'analyze_calibration_world_evaluation.py': '92618b51a29ecc49c2573b9529b426e3a27dbdc3a86636405fe6bd5262d33c86', 'calibration_world_inputs.py': '638a8f4ff2ea07281602bde1a7882929cf44a6a4a62e2de7e35c5e642089d921', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_calibration_transfer.py': 'decf50e505235bf2f37cc48f14b89e0397eb6f2604a979e2cb8a32bf091c3bb1', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_l11_key_value_donor.py': '1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_native_value_storage.py': 'b4b7f8932a93136fcccc2ad423bd2bc89fd57c84c052273f15ef36cbdd63d78e', 'gate_query_write_timing.py': 'f9f2eb64795b6b40ce19a2b6187e4dbb7263d838ff152bf2d081b06260a0f0ab', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gate_write_norm.py': '62e8b4c67a522043a6cb315b97c7cee34f84dc853cfcb4121679344b4b06445f', 'gpt2_calibration_world_evaluation.py': '9850c438d4fd7af387a72aa1968568e753319e4fe9549b9ba9d36af97bb95b07', 'gpt2_gate_native_value.py': '7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'l11_key_value_donor_storage.py': '132ad35ff5b240eac0a41392a8a04f584d9383d0f3dd82fc4e7e566916a6e7eb', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'merge_calibration_world_evaluation.py': 'd827ae32ed907f44ae542d894f716882445989d47fbe778ddd104d7dd8c23763', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'native_value_portable_draft.py': '46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4', 'post_write_norm_inputs.py': 'e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48', 'post_write_norm_selector.py': '34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4', 'post_write_norm_storage.py': 'f7d8068b85f4521c7d7409c88de0b1e9a2c6fc582238525db975c2df22ead0d3', 'review/t23_evaluation_runtime_contract_v1.json': 'a5b822b2f1b1f051c5393de3c35dd9bf62a566106dbdcfd905bde7906a982c6d', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044'}
CANONICAL_INPUTS={'tokens.pt': '23e11fa6564d4d29cb847432aff97a60d3dc762042fe625535d6db0207f15e9a', 'data_manifest.json': '4696bb86cb666351a00d3b68f9005088a8b9563816a50478a86455f2279f8505', 'exclusion_inventory.json': 'afe1cd68e15848b6c5b619e75b541adb0145060a37601fa4a506a6909af12ba6', 'known_history_chain.json': '55fd60df741c3eadc81487cf77b4fe77e68a4ac59968433af8a577b451be3258', 'known_history_review.json': 'ef971da48a0f3c8e344b27e668146248484d25140c002adb42bb7c3fb1aaf3a0', 'start_identity_summary.json': 'b876051f9f9a9b92696a58b91e83b2d68be9d80aa6c85daa1e91ff3da92b6661', 'preflight.json': '21bd63963ce17bdb3df52ad327abb01790a9bdaccf887d572d8802370048f492'}
AUDITS={'numerical': {'file': 't23_evaluation_numerical_actual_v2.json', 'private_sha256': 'd2f81f3cd07b76792c17ab31ea4c09d1579fc6450668ae4e5a0b78f7c31125cc', 'auditor_sha256': '0d9d8a2a2813c966f49815eaf570adaa1e0a193164f03323144b003dee42722e', 'required_private_bindings': {'audited_manifest_sha256': '1b2e1f8f4c36355b297b45461c3b34143d42b30a348ec438d07a0f3a665c1382', 'raw_inventory_sha256': '851bd22f549a205fe53581d5c65b75974565bb8b4ef0814fc3b269d5e8e1bb05', 'analysis_sha256': '8dc01b8911cd95c08c9ba9c46c95a00e0cf5c54177d5bc70f7eda635bd1797b5', 'audit_source_sha256': '0d9d8a2a2813c966f49815eaf570adaa1e0a193164f03323144b003dee42722e'}}, 'projection': {'file': 't23_selected_projection_actual_v1.json', 'private_sha256': 'b18c4093b987238a08c843670ebe3295d0f991765b2a59f1b886ddd42012e64d', 'auditor_sha256': 'a2d974104890ea7234e3b9994633105cfbae651680b1a7119495cd50eec29c9d', 'required_private_bindings': {'run_manifest_sha256': '1b2e1f8f4c36355b297b45461c3b34143d42b30a348ec438d07a0f3a665c1382', 'inventory_sha256': '851bd22f549a205fe53581d5c65b75974565bb8b4ef0814fc3b269d5e8e1bb05', 'auditor_sha256': 'a2d974104890ea7234e3b9994633105cfbae651680b1a7119495cd50eec29c9d', 'N_pass_sha256': 'd2f81f3cd07b76792c17ab31ea4c09d1579fc6450668ae4e5a0b78f7c31125cc'}}, 'readout': {'file': 't23_selected_readout_actual_v1.json', 'private_sha256': '89b8084cf33b55c25c7c653b6d92c86e7315f280a6c975dc23550da701523f24', 'auditor_sha256': 'a2d974104890ea7234e3b9994633105cfbae651680b1a7119495cd50eec29c9d', 'required_private_bindings': {'run_manifest_sha256': '1b2e1f8f4c36355b297b45461c3b34143d42b30a348ec438d07a0f3a665c1382', 'inventory_sha256': '851bd22f549a205fe53581d5c65b75974565bb8b4ef0814fc3b269d5e8e1bb05', 'auditor_sha256': 'a2d974104890ea7234e3b9994633105cfbae651680b1a7119495cd50eec29c9d', 'N_pass_sha256': 'd2f81f3cd07b76792c17ab31ea4c09d1579fc6450668ae4e5a0b78f7c31125cc'}}}
# Per original source: exact source hash and finite, reviewed public projection.
PUBLIC_SOURCE_HASHES={'GPT2_CALIBRATION_WORLD_COMPARISON_PROPOSAL_draft_v2.md': 'a78e9678ffceefc85abcccc2a11cd750754efd1adb7738caa03d508f957c2e7d', 'analyze_calibration_world_evaluation.py': '92618b51a29ecc49c2573b9529b426e3a27dbdc3a86636405fe6bd5262d33c86', 'calibration_world_inputs.py': '638a8f4ff2ea07281602bde1a7882929cf44a6a4a62e2de7e35c5e642089d921', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_calibration_transfer.py': 'decf50e505235bf2f37cc48f14b89e0397eb6f2604a979e2cb8a32bf091c3bb1', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_l11_key_value_donor.py': '1a4b9ed45d3ae621e76751f5c2f7639bd3db2a4eacc5e9f4b0003a90dec8cf28', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_native_value_storage.py': 'b4b7f8932a93136fcccc2ad423bd2bc89fd57c84c052273f15ef36cbdd63d78e', 'gate_query_write_timing.py': 'f9f2eb64795b6b40ce19a2b6187e4dbb7263d838ff152bf2d081b06260a0f0ab', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gate_write_norm.py': '62e8b4c67a522043a6cb315b97c7cee34f84dc853cfcb4121679344b4b06445f', 'gpt2_calibration_world_evaluation.py': '9850c438d4fd7af387a72aa1968568e753319e4fe9549b9ba9d36af97bb95b07', 'gpt2_gate_native_value.py': '7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'l11_key_value_donor_storage.py': '132ad35ff5b240eac0a41392a8a04f584d9383d0f3dd82fc4e7e566916a6e7eb', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'merge_calibration_world_evaluation.py': 'd827ae32ed907f44ae542d894f716882445989d47fbe778ddd104d7dd8c23763', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'native_value_portable_draft.py': '46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4', 'post_write_norm_inputs.py': 'e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48', 'post_write_norm_selector.py': '34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4', 'post_write_norm_storage.py': 'f7d8068b85f4521c7d7409c88de0b1e9a2c6fc582238525db975c2df22ead0d3', 'review/t23_evaluation_runtime_contract_v1.json': '6992f04e229a805217753bcae21389bed35146210758f1df5022b4a00738abf4', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044'}
CERTIFICATE_LOCATOR_REPLACEMENTS={}
CERTIFICATE_KEY_RENAMES={'review/t23_evaluation_runtime_contract_v1.json': [{'parent_pointer': '/receipt_provenance/independent_audit', 'original_key_sha256': 'a0af9f865bf637e6736817f4ce552e4cdf7b8c36ea75bc254c1d1f0af744b5bf', 'public_key': 'artifact_member'}, {'parent_pointer': '/receipt_provenance/independent_acceptance', 'original_key_sha256': 'a0af9f865bf637e6736817f4ce552e4cdf7b8c36ea75bc254c1d1f0af744b5bf', 'public_key': 'artifact_member'}]}
CERTIFICATE_HASH_EDGES={}
# Finite staged JSON data changes and source assignment data changes, never
# scientific predicate edits. Each source must be a member of current closure.
STAGED_JSON_BINDINGS={'review/t23_evaluation_runtime_contract_v1.json': {'/audited_input_hashes': {'original_value_sha256': '6dd7d5ecbe64baacab9c1c689365610ba09e56caab4a7e68ef2ed1867c714d63', 'binding': {'input_hashes': True}}, '/reviewed_source_hashes': {'original_value_sha256': 'e071590dc4515be57fb9ce8b926f0f8f00e530ad9df5f07201581586444d1903', 'binding': {'source_map_excluding': ['gpt2_calibration_world_evaluation.py', 'review/t23_evaluation_runtime_contract_v1.json']}}}}
STAGED_SOURCE_BINDINGS={'gpt2_calibration_world_evaluation.py': {'INPUT_HASHES': {'input_hashes': True}, 'RESOURCE_CONTRACT_SHA': {'source_sha256': 'review/t23_evaluation_runtime_contract_v1.json'}, 'CALIBRATION_MANIFEST_SHA': {'calibration_sha256': 'calibration_study/manifest.json'}}}
# Exact low-dimensional F/CE assets and sanitized metadata are a separate privacy
# gate. No raw calibration b/mass/token assets may be admitted here.
CALIBRATION_RELEASE_FILES={'calibration_reference/manifest.json': {'source': 'calibration_reference/manifest.json', 'original_sha256': 'f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060', 'public_sha256': 'f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060', 'json_projection': None}, 'calibration_reference/calibration_maps.npz': {'source': 'calibration_reference/calibration_maps.npz', 'original_sha256': 'f75cf7582402ce5b999758c70477e6cf2740f317a9ed8e508b350501397ef14a', 'public_sha256': 'f75cf7582402ce5b999758c70477e6cf2740f317a9ed8e508b350501397ef14a', 'json_projection': None}, 'calibration_reference/calibration_directions.npz': {'source': 'calibration_reference/calibration_directions.npz', 'original_sha256': '53d686645f431a47f5e22ec9c5ec64c5cfb0ed1086c2172b8bb6922a532be8dc', 'public_sha256': '53d686645f431a47f5e22ec9c5ec64c5cfb0ed1086c2172b8bb6922a532be8dc', 'json_projection': None}, 'calibration_study/manifest.json': {'source': 'calibration_study/manifest.json', 'original_sha256': 'f71a3c3b8af737f240b9d559b0786c90501f89dae7ef3a2fba2efacc004f6b2a', 'public_sha256': '99450857fcaa196ecdbee4dde56c12f91be610a0353921efa7f1d3f0cef76853', 'json_projection': {'study': 'T23_calibration_only', 'status': 'completed', 'phase': 'calibration_complete_pending_independent_acceptance', 'model_load_attempted': True, 'model_loaded': True, 'future_evaluation_authorized': False, 'planned_calibration_forwards': 16, 'evaluation_forwards': 0, 'calibration_world_order': ['PE', 'CE', 'PN', 'CN'], 'original_F_digest': '8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8', 'source_hashes': {'GPT2_CALIBRATION_WORLD_COMPARISON_PROPOSAL_draft_v2.md': 'a78e9678ffceefc85abcccc2a11cd750754efd1adb7738caa03d508f957c2e7d', 'fit_calibration_world.py': '6b86a44e53f94c9e7a9217d0e5827d7fdcc7f317393ab3198b94da945b075b6e', 'fit_gate_calibration.py': '9cc6ea695f6c3753f8878a5d5146fa9a371be264685c2e26d8f831cd2b147c7e', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_calibration_world.py': 'fb2c0dd1008b0632a91442588fa4dc3e4f60584330f87127f756b93e71bed959', 'gate_direction_geometry.py': '5ca31e5bb1b5bce8ea2e2af093ffdad15224ba200ee1e852d2695f94af582ea8', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gpt2_calibration_world.py': '4b1406352ecca8fb59dddbf286955cce23dc85bb2fa1ff698ce631c44a38ecd9', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_gate_subspace_v2.py': '8a1b6b08b5d1e0d379247455fda610ebb98a4c3102161d37cc6531c8ba807dda', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'l11_key_value_donor_storage.py': '132ad35ff5b240eac0a41392a8a04f584d9383d0f3dd82fc4e7e566916a6e7eb', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'post_write_norm_storage.py': 'f7d8068b85f4521c7d7409c88de0b1e9a2c6fc582238525db975c2df22ead0d3', 'review/t23_calibration_phase_contract_v1.json': 'f58c2727c3ce1095bee01714c1adcc1c16dacf9676649b8a145594cc4f6e818f', 'review/t23_calibration_world_asset_map_draft_v2.json': '0799c11d23aca737f06c40e9823c8c97382243ef2a6444de124c8b53d2ab13ea', 'review/t23_calibration_world_independent_design_review_v2.json': '5d375094b40a5558ea9266fa506a7a46ff217f943b95d500b8451f318389961b', 'review/t23_root_SOURCE_ONLY_implementation_GO_v1.json': 'ef8d6293ad113025db3a9db8f47f8b6c27c62f0b665118911d978c9eb167d3f8', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044'}, 'asset_dependencies': {'calibration_wiki_inputs/calibration_inputs.json': 'b58e0dfc03cc6125cf618c9d66bd38b375db102e418971bf0a48b4fe2a58df04', 'calibration_wiki_inputs/calibration_tokens.pt': 'dcdc07d717bac319889491f637e0c98fd8f4c05dc663fb58aef994a2ca7faae5', 'gpt2_gate_wiki_calibration_run/manifest.json': 'f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060', 'gpt2_gate_wiki_calibration_run/calibration_maps.npz': 'f75cf7582402ce5b999758c70477e6cf2740f317a9ed8e508b350501397ef14a', 'gpt2_gate_wiki_calibration_run/calibration_directions.npz': '53d686645f431a47f5e22ec9c5ec64c5cfb0ed1086c2172b8bb6922a532be8dc', 'gpt2_gate_wiki_calibration_run/calibration/wiki_test__0.npz': '0f4d1093f73815bee8de63c5345d98365443aba96ac491776120ed283d0c82f4', 'gpt2_gate_wiki_calibration_run/calibration/wiki_test__1.npz': '0537dc0ba7c424fd9d834864b018e80f367a0950911ac494e7bf77be346c0016', 'gpt2_gate_wiki_calibration_run/calibration/wiki_test__2.npz': '786a561338bff68e9cb9c20103111ab419de37e174d29d0fee1f4010b2c02d6c', 'gpt2_gate_wiki_calibration_run/calibration/wiki_test__3.npz': 'c47b37fd7fb70bd2de71a0d1982c15e5d6f4727fd41a7b584e93afa295debe8f'}, 'execution_review_sha256': '7a15f037a650d9170a6c9527a653eb18232acca4c9e1d9a27a433733ec90594e', 'contract_sha256': 'f58c2727c3ce1095bee01714c1adcc1c16dacf9676649b8a145594cc4f6e818f', 'runtime_versions': {'python': '3.12.3', 'torch': '2.11.0+cu130', 'transformers': '4.57.6', 'numpy': '1.26.4', 'cuda_runtime': '13.0'}, 'precision_settings_before': {'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'deterministic_algorithms': False, 'cudnn_deterministic': False, 'cudnn_benchmark': False, 'flash_sdp_enabled': True, 'memory_efficient_sdp_enabled': True, 'math_sdp_enabled': True, 'cudnn_sdp_enabled': True}, 'primary_RSS_limit_bytes': 4294967296, 'review_RSS_limit_bytes': 3221225472, 'host_MemAvailable_before_model_bytes': 10443116544, 'GPU_hardware_before_model': 'NVIDIA GeForce RTX 3060 Ti, UUID withheld, driver 595.99.02, 8192 MiB', 'forward_attempts': 16, 'model_outputs_observed': 16, 'forward_passes': 16, 'fully_processed_forwards': 16, 'calibration_forwards': 16, 'identity': {'world': 'CN', 'window': 3, 'corpus': 'wiki_test', 'length': 512, 'token_sha256': '3d8dc06fd347985d85a97dff5f8bee14cdd67b5b94b7c352ab6a8ecfe0ba9e2e'}, 'wall_seconds': 9.491490364074707, 'peak_process_rss_bytes': 1851559936, 'peak_GPU_allocated_bytes': 827707904, 'peak_GPU_reserved_bytes': 905969664, 'model_hash_before': '2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527', 'model_id': 'openai-community/gpt2', 'revision': '75e09b43581151bd1d9ef6700faa605df408979f', 'PE_continuity_sha256': '23cd2d54fe4e81a5e03db8c1efaf156c012e16249d8dd346769a90211cfac8e2', 'model_hash_after': '2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527', 'fit_records': {'PE': {'file': 'fits/PE.npz', 'sha256': '12714ae35656f3301b2ca4f3118f39934e5ce8e6ce01e92585412d92281669eb', 'map_digest': '8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'}, 'CE': {'file': 'fits/CE.npz', 'sha256': '31c9db95912c8e317f77d0814a5ee94bdd66a25566bea48245495519d2b5ccd9', 'map_digest': '2c26d9c65918f298a3f933986438bd5daf805b539270ffe15b0993e98cbf418d'}, 'PN': {'file': 'fits/PN.npz', 'sha256': 'dde3797736a616b998ba4930ad36af59f9846a8ab5af41924592e8dd05c9dd61', 'map_digest': '8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'}, 'CN': {'file': 'fits/CN.npz', 'sha256': 'd6498af25310a87e165188cb5f1fb3344813c2adc2d03326895d5aec51beece6', 'map_digest': '2c26d9c65918f298a3f933986438bd5daf805b539270ffe15b0993e98cbf418d'}}, 'primary_worlds': ['PE_original_F', 'CE'], 'secondary_worlds': ['PN', 'CN'], 'all_final_guards_passed': True, 'precision_settings_after': {'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'deterministic_algorithms': False, 'cudnn_deterministic': False, 'cudnn_benchmark': False, 'flash_sdp_enabled': True, 'memory_efficient_sdp_enabled': True, 'math_sdp_enabled': True, 'cudnn_sdp_enabled': True}}}, 'calibration_study/PE_continuity.json': {'source': 'calibration_study/PE_continuity.json', 'original_sha256': '23cd2d54fe4e81a5e03db8c1efaf156c012e16249d8dd346769a90211cfac8e2', 'public_sha256': '23cd2d54fe4e81a5e03db8c1efaf156c012e16249d8dd346769a90211cfac8e2', 'json_projection': None}, 'calibration_study/fits/CE.npz': {'source': 'calibration_study/fits/CE.npz', 'original_sha256': '31c9db95912c8e317f77d0814a5ee94bdd66a25566bea48245495519d2b5ccd9', 'public_sha256': '31c9db95912c8e317f77d0814a5ee94bdd66a25566bea48245495519d2b5ccd9', 'json_projection': None}, 'calibration_study/acceptance.json': {'source': 'calibration_study/acceptance.json', 'original_sha256': '87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300', 'public_sha256': '87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300', 'json_projection': None}}
CALIBRATION_PUBLIC_SHA256={'calibration_reference/manifest.json': 'f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060', 'calibration_reference/calibration_maps.npz': 'f75cf7582402ce5b999758c70477e6cf2740f317a9ed8e508b350501397ef14a', 'calibration_reference/calibration_directions.npz': '53d686645f431a47f5e22ec9c5ec64c5cfb0ed1086c2172b8bb6922a532be8dc', 'calibration_study/manifest.json': '99450857fcaa196ecdbee4dde56c12f91be610a0353921efa7f1d3f0cef76853', 'calibration_study/PE_continuity.json': '23cd2d54fe4e81a5e03db8c1efaf156c012e16249d8dd346769a90211cfac8e2', 'calibration_study/fits/CE.npz': '31c9db95912c8e317f77d0814a5ee94bdd66a25566bea48245495519d2b5ccd9', 'calibration_study/acceptance.json': '87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300'}
CALIBRATION_ASSET_NAMES={'calibration_reference/manifest.json','calibration_reference/calibration_maps.npz','calibration_reference/calibration_directions.npz','calibration_study/manifest.json','calibration_study/PE_continuity.json','calibration_study/fits/CE.npz','calibration_study/acceptance.json'}
CALIBRATION_MANIFEST_SHA='f71a3c3b8af737f240b9d559b0786c90501f89dae7ef3a2fba2efacc004f6b2a'
CALIBRATION_AUDIT_SHA='50f41e1e4f31c2ecc123ef9c8f71b119a3512fd1795d1deca54f03cb073eba47'
CALIBRATION_ACCEPTANCE_SHA='87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300'
COUNT_CONTRACT={'scheduled_evaluation':1160,'main':576,'suffix':576,'anchors':8,'calibration':16,'shards':32,'window_rows':1728,'window_scalar_entries':5184,'anchor_rows':24,'summaries':432,'contrasts':2304,'paired_starts':1152,'primary':64,'matching_pairs':256}

def fresh(out,allowance=128*1024**2):
 global OWNED
 base.fresh(out,allowance);st=out.stat();OWNED=(out.resolve(),st.st_dev,st.st_ino)

def no_personal_source(path):assert not re.search(r'/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/',Path(path).read_text()),('Personal host locator in source',Path(path).name)

def tokenizer(package,config):
 from transformers import AutoTokenizer
 dm=read(package/'metadata/data_manifest.template.json');cfg=read(config);spec=read(package/'metadata_dependencies/superweight-causal-circuits__manifests__transfer_models.json')['gpt2'];gc=cfg['models']['gpt2'];assert gc['model_id']==spec['model_id'] and gc['revision']==spec['revision']
 for n,h in dm['tokenizer_sha256'].items():assert sha(Path(gc['path'])/n)==h
 tok=AutoTokenizer.from_pretrained(gc['path'],local_files_only=True);assert tok.is_fast and type(tok.bos_token_id)is int
 literals={name:{'literal':text,'ids':tok.encode(text,add_special_tokens=False)} for name,text in [('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n')]};literals['start_special']={'label':'bos','id':tok.bos_token_id}
 return tok,literals,cfg

def assignment_only(source,replacements):
 tree=ast.parse(source);nodes={}
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:nodes[node.targets[0].id]=node
 assert set(nodes)==set(replacements)
 lines=source.splitlines(keepends=True)
 for name,node in sorted(nodes.items(),key=lambda pair:pair[1].lineno,reverse=True):lines[node.lineno-1:node.end_lineno]=[name+'='+repr(replacements[name])+'\n']
 changed=''.join(lines);restored=ast.parse(changed)
 for node in restored.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in nodes:node.value=copy.deepcopy(nodes[node.targets[0].id].value)
 assert ast.dump(restored,include_attributes=False)==ast.dump(tree,include_attributes=False),'Only declared binding assignments may change; all predicates/functions/schedule/counters/scoring stay exact'
 return changed

def assert_typed(value,schema,path='audit'):
 """Fixed actual-report structural contract; unknown fields fail closed."""
 if isinstance(schema,dict):
  assert isinstance(value,dict) and set(value)==set(schema),('Typed fields',path)
  for name,child in schema.items():assert_typed(value[name],child,path+'/'+name)
 elif isinstance(schema,list):
  assert isinstance(value,list) and schema[0]=='list' and len(value)==len(schema)-1,('Typed list',path)
  for i,(child,kind) in enumerate(zip(value,schema[1:])):assert_typed(child,kind,path+'/'+str(i))
 else:
  assert type(value).__name__==schema,('Typed scalar',path,schema)
  if type(value)is float:assert math.isfinite(value)
def require_release(*,metadata=False):
 assert EXPORT_READY and CANONICAL and len(FROZEN_SOURCE_HASHES)==30 and len(CANONICAL_INPUTS)==7,'HOLD: completed frozen T23 source/run/NPR review required'
 assert set(AUDITS)=={'numerical','projection','readout'}
 if metadata:assert METADATA_STAGE_READY and STAGE_BINDING_REVIEW_SHA is not None,'HOLD: finite metadata/fit privacy and staging review required'

def backend_module(code=ROOT):
 sys.path.insert(0,str(code));backend=importlib.import_module('analyze_calibration_world_evaluation')
 assert Path(backend.__file__).resolve()==(code/'analyze_calibration_world_evaluation.py').resolve()
 assert sha(backend.__file__)==FROZEN_SOURCE_HASHES['analyze_calibration_world_evaluation.py']
 assert tuple(backend.ARMS)==ARMS and tuple(backend.CORPORA)==CORPORA
 return backend

def canonical_run(run,*,shards=False):
 require_release()
 for name,h in CANONICAL.items():assert sha(run/name)==h,('Canonical T23 identity',name)
 m=read(run/'manifest.json');assert m['status']=='completed' and m['execution_type']=='planned_sequential_segments'
 for key,count in (('forward_attempts',1160),('model_outputs_observed',1160),('forward_passes',1160),('fully_processed_forwards',1160),('evaluation_forwards',576),('suffix_forwards',576),('anchor_forwards',8),('calibration_forwards',0)):assert m[key]==count
 assert m['boundaries']==['000'] and m['arms']==list(ARMS)
 assert m['source_hashes']==FROZEN_SOURCE_HASHES and m['input_dependencies']==CANONICAL_INPUTS and m['all_final_guards_passed'] is True
 assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True and m['precision_settings_before']==m['precision_settings_after']
 assert m['calibration_acceptance_sha256']==CALIBRATION_ACCEPTANCE_SHA
 for name,h in FROZEN_SOURCE_HASHES.items():base.relative_member(name);assert sha(run/'frozen_sources'/name)==h
 for name,h in CANONICAL_INPUTS.items():assert sha(run/name)==h
 for marker in ('merge_failure.json','checkpoint_failure.json','failure_state.npz'):assert not (run/marker).exists()
 inventory=read(run/'numeric_inventory.json');expected={f'inputs/{c}__{w}__{i}__000.npz' for c in CORPORA for w in range(4) for i in ('natural','bos')}
 assert set(inventory)==expected
 if shards:
  assert base.scan_package(run/'inputs',set())=={Path(n).name for n in expected}
  for name,value in inventory.items():assert value=={'sha256':sha(run/name),'bytes':(run/name).stat().st_size}
 return m,inventory

def fingerprint(value,shape,dtype='torch.float32'):
 assert set(value)=={'sha256','shape','dtype'} and base.hash_value(value['sha256'])
 assert type(value['dtype']) is str and value['dtype']==dtype and value['shape']==list(shape) and all(type(x)is int for x in value['shape'])
 return copy.deepcopy(value)

def expected_checks(anchor):
 metrics={name+'_exact':True for name in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary')}
 identity={'logits_exact':True,'hidden_exact':True,**metrics}
 suffix={name+'_exact':True for name in ('logits','hidden','source','gate','gate_norm','write_norm','context','write','mask')};suffix.update(metrics)
 result=[]
 for arm in ARMS:
  if arm=='baseline' and anchor:result.extend(({'kind':'native_repeat',**identity},{'kind':'native_identity',**identity,'write_exact':True}))
  for variant in ('main','suffix'):
   if arm!='baseline':result.append({'kind':'source_q0','arm':arm,'variant':variant,'source_exact':True})
   if arm=='current':result.append({'arm':arm,'variant':variant,'kind':'current',**identity,'write_exact':True})
  result.append({'arm':arm,'kind':'suffix',**suffix})
 return result

def metadata_identity(metadata):
 assert set(metadata)=={'corpus','window','identity','token_sha256','boundary','length'}
 assert metadata['corpus'] in CORPORA and type(metadata['window'])is int and metadata['window'] in range(4)
 assert metadata['identity'] in ('natural','bos') and metadata['boundary']=='000' and type(metadata['length'])is int and metadata['length']==512 and base.hash_value(metadata['token_sha256'])
 return metadata['window']==0 and metadata['identity']=='natural'

def parameter_record(value,cell):
 assert set(value)=={'cell','center_world','basis_world','center_map_digest','basis_map_digest','vectors','parameter_digest'}
 assert value['cell']==cell and value['center_world']==('PE' if cell[0]=='P' else 'CE') and value['basis_world']==('PE' if cell[1]=='P' else 'CE')
 for name in ('center_map_digest','basis_map_digest','parameter_digest'):assert base.hash_value(value[name])
 assert set(value['vectors'])=={'alpha','u','v'}
 for z in value['vectors'].values():fingerprint(z,(12,),'torch.float64')
 bare={k:v for k,v in value.items() if k!='parameter_digest'}
 assert hashlib.sha256(json.dumps(bare,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==value['parameter_digest']
 return value

def retirement_view(raw):
 result=[]
 for value in raw:
  cell=value['cell'];assert set(value)=={'cell','order_index','parameter_record','variants','retirement_scope'}
  assert cell in CELLS and value['order_index']==12+8*CELLS.index(cell)
  parameter_record(value['parameter_record'],cell);variants={}
  assert set(value['variants'])=={'main','suffix'}
  for variant,z in value['variants'].items():
   assert set(z)=={'score','full_candidates','matching_commitment','delivered'}
   fingerprint(z['score'],(511,));assert set(z['full_candidates'])=={'remove_full','orth_full'} and set(z['delivered'])=={'0','1'}
   for fp in (*z['full_candidates'].values(),*z['delivered'].values()):fingerprint(fp,(1,512,768))
   assert base.hash_value(z['matching_commitment']);variants[variant]=copy.deepcopy(z)
  result.append({'cell':cell,'order_index':value['order_index'],'parameter_record':copy.deepcopy(value['parameter_record']),'variants':variants})
 assert [x['cell'] for x in result]==list(CELLS)
 return result

def assert_public_ledger(value):
 assert set(value)=={'metadata','checks','operations','suffix_operations','final_hidden_fingerprints','suffix_final_hidden_fingerprints','retirement_receipts'}
 anchor=metadata_identity(value['metadata']);main=set(ARMS)|({'native_repeat','native_identity'} if anchor else set())
 assert value['checks']==expected_checks(anchor)
 for actual,wanted in zip(value['checks'],expected_checks(anchor)):
  assert set(actual)==set(wanted) and all(type(actual[k]) is type(v) for k,v in wanted.items())
 for group,variant,hashes,arms in (('operations','main','final_hidden_fingerprints',main),('suffix_operations','suffix','suffix_final_hidden_fingerprints',set(ARMS))):
  assert set(value[group])==set(value[hashes])==arms
  for arm,row in value[group].items():
   assert base.hash_value(value[hashes][arm])
   if arm=='native_repeat':assert row=={};continue
   assert set(row)=={'write_checks','timing_fingerprints','cell_record','cache_key'}
   assert set(row['timing_fingerprints'])=={'post_attention','mlp_output','block_output','l11_qkv'}
   for node,fp in row['timing_fingerprints'].items():fingerprint(fp,(1,512,2304 if node=='l11_qkv' else 768))
   cell=arm.split('_',1)[0] if arm not in ('baseline','current','native_identity') else None
   record=row['cell_record'];assert set(record)=={'cell','parameter_record','score_origin','score'} and record['cell']==cell
   parameter_record(record['parameter_record'],cell or 'PP')
   if cell is None:assert record['score_origin'] is record['score'] is None
   else:
    assert record['score_origin']=='same_variant_first_actual_cell_full_remove_ff_map';fingerprint(record['score'],(511,))
   cache=row['cache_key'];assert set(cache)=={'cell','variant','actual_token_sha256','parameter_digest','fit_digests','freeze_digest','calibration_acceptance_sha256'}
   assert cache['cell']==cell and cache['variant']==variant and cache['parameter_digest']==record['parameter_record']['parameter_digest']
   assert set(cache['fit_digests'])=={'PE','CE'} and all(base.hash_value(h) for h in cache['fit_digests'].values())
   for k in ('actual_token_sha256','freeze_digest','calibration_acceptance_sha256'):assert base.hash_value(cache[k])
   wc=row['write_checks']
   if arm in MATCHED:
    assert set(wc)=={'support','active_write_bytes_exact','inactive_write_bytes_exact','cell','target_scope','paired_actual_delivery_complete','counterpart_norm_scope'}
    assert wc['support']=='early' and wc['cell']==cell and wc['active_write_bytes_exact'] is wc['inactive_write_bytes_exact'] is True
    assert wc['target_scope']=='same_frozen_full_recipe_target_on_active_queries_zero_elsewhere'
    assert wc['paired_actual_delivery_complete'] is arm.endswith('orth_write_matched_early')
    assert wc['counterpart_norm_scope']==('actual_delivered_same_support' if arm.endswith('orth_write_matched_early') else 'expected_CPU_rounded_not_yet_delivered_same_support')
   else:assert wc=={}
 # Public retirement scope is an enum expressed by schema, not copied prose.
 assert len(value['retirement_receipts'])==4
 for cell,receipt in zip(CELLS,value['retirement_receipts']):
  assert set(receipt)=={'cell','order_index','parameter_record','variants'} and receipt['cell']==cell and receipt['order_index']==12+8*CELLS.index(cell)
  parameter_record(receipt['parameter_record'],cell)
  assert set(receipt['variants'])=={'main','suffix'}
  for row in receipt['variants'].values():
   assert set(row)=={'score','full_candidates','matching_commitment','delivered'};fingerprint(row['score'],(511,));assert base.hash_value(row['matching_commitment'])
   assert set(row['full_candidates'])=={'remove_full','orth_full'} and set(row['delivered'])=={'0','1'}
   for fp in (*row['full_candidates'].values(),*row['delivered'].values()):fingerprint(fp,(1,512,768))
 safe(value);return value

def scalar_ledger(raw):
 anchor=metadata_identity(raw['metadata'])
 result={'metadata':dict(raw['metadata']),'checks':copy.deepcopy(raw['checks']),'operations':{},'suffix_operations':{},'final_hidden_fingerprints':{},'suffix_final_hidden_fingerprints':{},'retirement_receipts':retirement_view(raw['retirement_receipts'])}
 keys=('support','active_write_bytes_exact','inactive_write_bytes_exact','cell','target_scope','paired_actual_delivery_complete','counterpart_norm_scope')
 for group,metric_group,hashes in (('operations','metrics','final_hidden_fingerprints'),('suffix_operations','suffix','suffix_final_hidden_fingerprints')):
  for arm,z in raw[metric_group].items():result[hashes][arm]=z['final_hidden_sha256']
  for arm,op in raw[group].items():
   if arm=='native_repeat':assert op=={};result[group][arm]={};continue
   cr=op['cell_records'];cache=op['write_cache_key']
   result[group][arm]={'write_checks':{k:op['write_checks'][k] for k in keys} if arm in MATCHED else {},'timing_fingerprints':copy.deepcopy(op['timing_fingerprints']),'cell_record':{'cell':cr['cell'],'parameter_record':copy.deepcopy(cr['parameter_record']),'score_origin':cr.get('score_origin'),'score':copy.deepcopy(cr.get('score'))},'cache_key':{k:copy.deepcopy(cache[k]) for k in ('cell','variant','actual_token_sha256','parameter_digest','fit_digests','freeze_digest','calibration_acceptance_sha256')}}
 return assert_public_ledger(json.loads(json.dumps(result)))

def safe_arrays(raw,backend):
 import numpy as np
 assert tuple(backend.ARMS)==ARMS
 arrays={};anchor=metadata_identity(raw['metadata'])
 for group in GROUPS:
  expected=set(ARMS)|({'native_repeat','native_identity'} if anchor and not group.startswith('suffix') else set())
  assert set(raw[group])==expected
  for arm,values in raw[group].items():
   if group in ('metrics','suffix'):
    extras={'pred','final_hidden_sha256'}|({'target_indices','logit_query_indices'} if group=='suffix' else set())
    if raw.get('schema')=='calibration_world_evaluation_explicit_v1':
     assert set(values)==set(METRICS)|extras and base.hash_value(values['final_hidden_sha256'])
     if group=='suffix':assert values['target_indices'].dtype==values['logit_query_indices'].dtype==np.int64 and backend.exact(values['target_indices'],np.arange(1,128,dtype=np.int64)) and backend.exact(values['logit_query_indices'],np.arange(128,dtype=np.int64))
    else:assert set(values)==set(METRICS)
    for name in METRICS:
     a=values[name];n=127 if group=='suffix' and name=='loss' else 128 if group=='suffix' else 511
     assert a.shape==(n,) and a.dtype==(np.bool_ if name.startswith('agreement') else np.float32) and np.isfinite(a).all()
     arrays[group+'__'+arm+'__'+name]=a
   else:
    if arm=='native_repeat':assert values=={};continue
    gate=group in ('gate_norms','suffix_gate_norms')
    if gate:assert set(values)=={10};values=values[10];keys=backend.GATE_KEYS
    else:keys=backend.WRITE_KEYS['baseline' if arm=='native_identity' else arm]
    assert set(values)==keys
    for name,a in values.items():
     n=(12 if name=='source_v_norm' else 127 if group=='suffix_gate_norms' else 511) if gate else 512
     assert a.shape==(n,) and a.dtype==(np.float32 if gate and name=='s' else np.float64) and np.isfinite(a).all()
     arrays[group+'__'+arm+('__10__' if gate else '__')+name]=a
 return arrays
def public_audit(raw,kind):
 spec=AUDITS[kind];assert raw['status']=='passed' and base.hash_value(spec['private_sha256']) and base.hash_value(spec['auditor_sha256'])
 assert CANONICAL['manifest.json'] in spec['required_private_bindings'].values()
 assert CANONICAL[ANALYSIS_NAME] in spec['required_private_bindings'].values() if kind=='numerical' else CANONICAL['numeric_inventory.json'] in spec['required_private_bindings'].values()
 # Only these already-pinned scalar bindings are published. Entire private audit
 # trees, resource paths, selected row IDs and raw comparisons never traverse.
 for field,wanted in spec['required_private_bindings'].items():assert raw[field]==wanted
 result={'schema':'T23_public_'+kind+'_acceptance_v1','status':'passed','private_report_sha256':spec['private_sha256'],'auditor_sha256':spec['auditor_sha256'],'run_manifest_sha256':CANONICAL['manifest.json'],'raw_inventory_sha256':CANONICAL['numeric_inventory.json'],'analysis_sha256':CANONICAL[ANALYSIS_NAME],'scope':'Original private execution acceptance commitment; omitted activations and distributions are not reconstructed by this public view.'}
 assert_public_audit(result,kind);return result

def assert_public_audit(value,kind):
 schema={'schema':'str','status':'str','private_report_sha256':'str','auditor_sha256':'str','run_manifest_sha256':'str','raw_inventory_sha256':'str','analysis_sha256':'str','scope':'str'}
 assert_typed(value,schema)
 assert value['schema']=='T23_public_'+kind+'_acceptance_v1' and value['status']=='passed'
 assert value['private_report_sha256']==AUDITS[kind]['private_sha256'] and value['auditor_sha256']==AUDITS[kind]['auditor_sha256']
 assert value['run_manifest_sha256']==CANONICAL['manifest.json'] and value['raw_inventory_sha256']==CANONICAL['numeric_inventory.json'] and value['analysis_sha256']==CANONICAL[ANALYSIS_NAME]
 assert value['scope']=='Original private execution acceptance commitment; omitted activations and distributions are not reconstructed by this public view.'
 for key in ('private_report_sha256','auditor_sha256','run_manifest_sha256','raw_inventory_sha256','analysis_sha256'):assert base.hash_value(value[key])
 safe(value)

def execution_view(m):
 result={'schema':'T23_public_execution_v1','canonical_private_manifest_sha256':CANONICAL['manifest.json'],'retained_scheduled_evaluation_forwards':1160,'main':576,'suffix':576,'anchors':8,'separate_completed_calibration_forwards':16,'calibration_manifest_sha256':CALIBRATION_MANIFEST_SHA,'calibration_audit_sha256':CALIBRATION_AUDIT_SHA,'calibration_acceptance_sha256':CALIBRATION_ACCEPTANCE_SHA,'source_hashes':FROZEN_SOURCE_HASHES,'original_segment_manifests':copy.deepcopy(m['original_segments']),'model_hash':m['model_hash_before'],'runtime_versions':copy.deepcopy(m['runtime_versions']),'precision_settings':copy.deepcopy(m['precision_settings_before']),'fresh_model_replay':'unperformed'}
 safe(result);return result

def numeric_members():
 names={f'{c}__{w}__{i}__000' for c in CORPORA for w in range(4) for i in ('natural','bos')}
 return {'export.json','execution.json','index.json',ANALYSIS_NAME,'numeric_export_inventory.json'}|{f'audits/{kind}.json' for kind in AUDITS}|{f'inputs/{name}.{ext}' for name in names for ext in ('npz','json')}

def export_numeric(a):
 require_release();import numpy as np
 backend=backend_module();outside_sources(a.out,a.run,a.audits);m,inventory=canonical_run(a.run,shards=True)
 analysis=read(a.run/ANALYSIS_NAME);assert analysis['scope']['analyzer_sha256']==FROZEN_SOURCE_HASHES['analyze_calibration_world_evaluation.py'] and analysis['scope']['raw_manifest_sha256']==CANONICAL['manifest.json'] and analysis['scope']['raw_inventory_sha256']==CANONICAL['numeric_inventory.json'];safe(analysis)
 assert analysis['scope']['source_hashes']=={Path(n).name:x['sha256'] for n,x in inventory.items()}
 audit_views={}
 for kind,spec in AUDITS.items():
  path=a.audits/spec['file'];assert sha(path)==spec['private_sha256'];audit_views[kind]=public_audit(read(path),kind)
 fresh(a.out,256*1024**2);put(a.out/ANALYSIS_NAME,analysis);put(a.out/'execution.json',execution_view(m))
 for kind,value in audit_views.items():put(a.out/'audits'/(kind+'.json'),value)
 index=[];count=0
 for name,entry in sorted(inventory.items()):
  raw=backend.load_npz(a.run/name);assert raw['schema']=='calibration_world_evaluation_explicit_v1'
  arrays=safe_arrays(raw,backend);ledger=scalar_ledger(raw);dest=a.out/'inputs'/Path(name).name;dest.parent.mkdir(exist_ok=True)
  with dest.open('xb') as f:np.savez_compressed(f,**arrays)
  with np.load(dest,allow_pickle=False) as saved:assert len(saved.files)==len(set(saved.files)) and set(saved.files)==set(arrays) and all(backend.exact(saved[k],v) for k,v in arrays.items())
  put(dest.with_suffix('.json'),ledger);index.append({'file':dest.name,'export_sha256':sha(dest),'ledger_sha256':sha(dest.with_suffix('.json')),'raw_sha256':entry['sha256'],'metadata':dict(raw['metadata']),'arrays':sorted(arrays)})
  count+=len(arrays);del raw,arrays
 put(a.out/'index.json',index)
 put(a.out/'export.json',{'schema':'T23_restricted_numeric_v1','canonical_run_sha256':CANONICAL['manifest.json'],'analysis_sha256':CANONICAL[ANALYSIS_NAME],'raw_inventory_sha256':CANONICAL['numeric_inventory.json'],'audit_private_commitments':{k:v['private_sha256'] for k,v in AUDITS.items()},'adapter_sha256':sha(__file__),'raw_shards':32,'main_records':584,'suffix_records':576,'numeric_arrays':count,'count_contract':COUNT_CONTRACT,'omitted':'Input/prediction/target/mismatch IDs,text,literal tokens,full gates or masses,masks,QKV,contexts,hidden,vocabulary rows,timing activation samples,weights,private locators,failure payloads. Signed s is a projection trace; agreement is an observation, not regenerated from omitted IDs.','model_forwards':0,'new_full_model_replay':'unperformed'})
 expected=numeric_members();actual=base.scan_package(a.out,base.expected_dirs(expected));assert actual==expected-{'numeric_export_inventory.json'}
 put(a.out/'numeric_export_inventory.json',{'schema':'T23_restricted_numeric_inventory_v1','files':{n:{'sha256':sha(a.out/n),'bytes':(a.out/n).stat().st_size} for n in sorted(actual)}})
 assert sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file())<=256*1024**2

def pointer(value,path,replacement=None,*,write=False):
 assert isinstance(path,str) and path.startswith('/')
 parts=[s.replace('~1','/').replace('~0','~') for s in path[1:].split('/')];owner=value
 for part in parts[:-1]:owner=owner[int(part)] if isinstance(owner,list) else owner[part]
 key=int(parts[-1]) if isinstance(owner,list) else parts[-1]
 previous=owner[key]
 if write:owner[key]=replacement
 return previous

def json_bytes(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()

def public_sources(run):
 assert set(PUBLIC_SOURCE_HASHES)==set(FROZEN_SOURCE_HASHES)
 result={};projections={}
 for name,h in FROZEN_SOURCE_HASHES.items():
  source=run/'frozen_sources'/name;assert sha(source)==h;data=source.read_bytes()
  edits=CERTIFICATE_LOCATOR_REPLACEMENTS.get(name,[]);renames=CERTIFICATE_KEY_RENAMES.get(name,[])
  if edits or renames:
   assert Path(name).suffix=='.json';value=json.loads(data)
   for edit in edits:
    assert set(edit)=={'pointer','original_value_sha256','public_value'}
    old=pointer(value,edit['pointer']);assert hashlib.sha256(json_bytes(old)).hexdigest()==edit['original_value_sha256'];safe(edit['public_value']);pointer(value,edit['pointer'],edit['public_value'],write=True)
   for edit in renames:
    assert set(edit)=={'parent_pointer','original_key_sha256','public_key'}
    owner=pointer(value,edit['parent_pointer']);assert isinstance(owner,dict);matches=[k for k in owner if hashlib.sha256(k.encode()).hexdigest()==edit['original_key_sha256']]
    assert len(matches)==1 and edit['public_key'] not in owner;old=matches[0];items=[(edit['public_key'] if k==old else k,v) for k,v in owner.items()];owner.clear();owner.update(items)
   safe(value);data=json_bytes(value)
  if Path(name).suffix=='.json':safe(json.loads(data))
  assert not re.search(rb'/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/',data)
  assert hashlib.sha256(data).hexdigest()==PUBLIC_SOURCE_HASHES[name]
  result[name]=data;projections[name]={'original_sha256':h,'public_sha256':PUBLIC_SOURCE_HASHES[name],'locator_value_replacements':edits,'locator_key_replacements':renames}
 return result,projections

def metadata_objects(run):
 dm=read(run/'data_manifest.json');inv=read(run/'exclusion_inventory.json');chain=read(run/'known_history_chain.json');review=read(run/'known_history_review.json');pre=read(run/'preflight.json')
 assert len(set(inv['history_token_hashes']))==841 and pre['status']=='passed'
 assert sha(run/'known_history_chain.json')==CANONICAL_INPUTS['known_history_chain.json']
 assert len(inv['retained_calibration_buffers'])==4
 for z in inv['retained_calibration_buffers']:assert z['char_end']-z['char_start']==8192 and any(v['char_start']<=z['char_start']<z['char_end']<=v['char_end'] for v in inv['exclusions'][z['corpus']])
 assert len(dm['passages'])==16 and sum(map(len,dm['rejections'].values()))==1074
 template=clean(dm);template.pop('literals');safe(template)
 # Preserve reviewed safe history/exclusion structure without calling ancestry
 # scanner or cal16 again. Each is validated before any public write.
 exclusion=clean(inv);safe(exclusion);safe(chain)
 history={'schema':'T23_inherited_recorded_history_v1','canonical_known_history_chain':chain,'canonical_chain_sha256':CANONICAL_INPUTS['known_history_chain.json'],'canonical_review_sha256':CANONICAL_INPUTS['known_history_review.json'],'canonical_exclusions_sha256':CANONICAL_INPUTS['exclusion_inventory.json'],'history_guard_count':841,'canonical_prior_hashes':809,'auxiliary_suffix_hashes':32,'calibration_observations':16,'calibration_unique_preexcluded_contexts':4,'added_calibration_hashes':0,'calibration_manifest_sha256':CALIBRATION_MANIFEST_SHA,'calibration_audit_sha256':CALIBRATION_AUDIT_SHA,'calibration_acceptance_sha256':CALIBRATION_ACCEPTANCE_SHA,'recorded_workflow_only_not_hostwide_or_pretraining':True}
 preparation={'schema':'T23_portable_preparation_commitment_v1','canonical_preflight_sha256':CANONICAL_INPUTS['preflight.json'],'canonical_input_hashes':CANONICAL_INPUTS,'accepted':16,'rejected':1074,'paired_starts':32,'historical_first_token_membership':'unavailable_not_used','input_adapter_sha256':FROZEN_SOURCE_HASHES['calibration_world_inputs.py']}
 objects={'data_manifest.template.json':template,'exclusion_inventory.json':exclusion,'start_identity.template.json':clean(read(run/'start_identity_summary.json')),'history_attestation.json':history,'preparation.template.json':preparation}
 for value in objects.values():safe(value)
 return objects,value_sha(dm['literals'])

def metadata_members(identity):
 assert set(identity)=={'schema','canonical_run_manifest_sha256','canonical_private_input_sha256','frozen_source_sha256','public_source_sha256','portable_source_sha256','metadata_sha256','certificate_projections_sha256','calibration_assets','metadata_dependencies','private_literals_commitment','historical_calibration','model_forwards','scope'}
 assert identity['schema']=='T23_portable_metadata_v1' and identity['canonical_run_manifest_sha256']==CANONICAL['manifest.json'] and identity['frozen_source_sha256']==FROZEN_SOURCE_HASHES and identity['public_source_sha256']==PUBLIC_SOURCE_HASHES and identity['canonical_private_input_sha256']==CANONICAL_INPUTS
 assert set(identity['metadata_sha256'])==METADATA_NAMES and set(identity['portable_source_sha256'])=={'calibration_world_portable.py','audit_calibration_world_public.py'}
 assert identity['historical_calibration']=={'manifest_sha256':CALIBRATION_MANIFEST_SHA,'audit_sha256':CALIBRATION_AUDIT_SHA,'acceptance_sha256':CALIBRATION_ACCEPTANCE_SHA,'actual_forwards':16,'unique_preexcluded_contexts':4}
 expected={'identity.json':None,'certificate_projections.json':identity['certificate_projections_sha256']}
 for name,h in {**identity['public_source_sha256'],**identity['portable_source_sha256']}.items():base.relative_member(name);expected['experiments/pulse_transfer/'+name]=h
 for name,h in identity['metadata_sha256'].items():expected['metadata/'+name]=h
 assert set(identity['calibration_assets'])==CALIBRATION_ASSET_NAMES and identity['calibration_assets']==CALIBRATION_PUBLIC_SHA256
 for name,h in identity['calibration_assets'].items():base.relative_member(name);expected[name]=h
 for name,spec in identity['metadata_dependencies'].items():
  assert spec=={'file':name.removeprefix('../').replace('/','__'),'sha256':base.METADATA_DEPENDENCY_HASHES[name]};expected['metadata_dependencies/'+spec['file']]=spec['sha256']
 assert set(identity['metadata_dependencies'])==set(base.METADATA_DEPENDENCY_HASHES)
 for name,h in expected.items():base.relative_member(name);assert h is None or base.hash_value(h)
 safe(identity);return expected

def validate_metadata(package):
 require_release(metadata=True);identity=read(package/'identity.json');expected=metadata_members(identity)
 assert base.scan_package(package,base.expected_dirs(expected))==set(expected)|{'package_inventory.json'}
 inventory=read(package/'package_inventory.json');assert set(inventory)=={'schema','files'} and inventory['schema']=='T23_portable_metadata_inventory_v1' and set(inventory['files'])==set(expected)
 for name,entry in inventory['files'].items():
  assert entry=={'sha256':sha(package/name),'bytes':(package/name).stat().st_size}
  if expected[name] is not None:assert entry['sha256']==expected[name]
 projections=read(package/'certificate_projections.json');assert set(projections)==set(FROZEN_SOURCE_HASHES)
 for name,row in projections.items():assert row=={'original_sha256':FROZEN_SOURCE_HASHES[name],'public_sha256':PUBLIC_SOURCE_HASHES[name],'locator_value_replacements':CERTIFICATE_LOCATOR_REPLACEMENTS.get(name,[]),'locator_key_replacements':CERTIFICATE_KEY_RENAMES.get(name,[])}
 for name in METADATA_NAMES:safe(read(package/'metadata'/name))
 return identity

def export_metadata(a):
 require_release(metadata=True);outside_sources(a.out,a.run);m,_=canonical_run(a.run)
 assert set(CALIBRATION_RELEASE_FILES)==set(CALIBRATION_PUBLIC_SHA256)==CALIBRATION_ASSET_NAMES,'HOLD: lowdimensional calibration bytes/privacy review'
 objects,literals=metadata_objects(a.run);sources,projections=public_sources(a.run)
 # Source dictionaries list exact reviewed assets, never recursive historical
 # raw calibration, token, full-state or checkpoint directories.
 assets={}
 for destination,spec in CALIBRATION_RELEASE_FILES.items():
  assert set(spec)=={'source','original_sha256','public_sha256','json_projection'};base.relative_member(destination);base.relative_member(spec['source'])
  src=a.run/spec['source'];assert sha(src)==spec['original_sha256'];data=src.read_bytes()
  if spec['json_projection'] is not None:
   value=copy.deepcopy(spec['json_projection']);safe(value);data=json_bytes(value)
  assert hashlib.sha256(data).hexdigest()==spec['public_sha256']==CALIBRATION_PUBLIC_SHA256[destination];assets[destination]=data
 fresh(a.out,128*1024**2)
 for name,data in sources.items():
  p=a.out/'experiments/pulse_transfer'/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
 for name in ('calibration_world_portable.py','audit_calibration_world_public.py'):no_personal_source(ROOT/name);copy_exact(ROOT/name,a.out/'experiments/pulse_transfer'/name,sha(ROOT/name))
 for name,value in objects.items():put(a.out/'metadata'/name,value)
 put(a.out/'certificate_projections.json',projections)
 for name,data in assets.items():
  p=a.out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
 deps={}
 for name,h in m['metadata_dependencies'].items():
  filename=name.removeprefix('../').replace('/','__');safe(read(a.run/'metadata_dependencies'/filename));copy_exact(a.run/'metadata_dependencies'/filename,a.out/'metadata_dependencies'/filename,h);deps[name]={'file':filename,'sha256':h}
 identity={'schema':'T23_portable_metadata_v1','canonical_run_manifest_sha256':CANONICAL['manifest.json'],'canonical_private_input_sha256':CANONICAL_INPUTS,'frozen_source_sha256':FROZEN_SOURCE_HASHES,'public_source_sha256':PUBLIC_SOURCE_HASHES,'portable_source_sha256':{name:sha(ROOT/name) for name in ('calibration_world_portable.py','audit_calibration_world_public.py')},'metadata_sha256':{name:sha(a.out/'metadata'/name) for name in objects},'certificate_projections_sha256':sha(a.out/'certificate_projections.json'),'calibration_assets':CALIBRATION_PUBLIC_SHA256,'metadata_dependencies':deps,'private_literals_commitment':literals,'historical_calibration':{'manifest_sha256':CALIBRATION_MANIFEST_SHA,'audit_sha256':CALIBRATION_AUDIT_SHA,'acceptance_sha256':CALIBRATION_ACCEPTANCE_SHA,'actual_forwards':16,'unique_preexcluded_contexts':4},'model_forwards':0,'scope':'Inherited accepted recorded exclusions and calibration commitments; no fresh calibration, tokenizer/model replay or omitted raw-primitive reconstruction.'}
 put(a.out/'identity.json',identity);expected=metadata_members(identity);assert base.scan_package(a.out,base.expected_dirs(expected))==set(expected)
 put(a.out/'package_inventory.json',{'schema':'T23_portable_metadata_inventory_v1','files':{name:{'sha256':sha(a.out/name),'bytes':(a.out/name).stat().st_size} for name in sorted(expected)}});validate_metadata(a.out)
def portable_input_objects(package,d,private_dm):
 md=package/'metadata';history=read(md/'history_attestation.json');chain=history['canonical_known_history_chain']
 assert hashlib.sha256(json_bytes(chain)).hexdigest()==history['canonical_chain_sha256']==d['canonical_private_input_sha256']['known_history_chain.json']
 exclusion=read(md/'exclusion_inventory.json');assert len(set(exclusion['history_token_hashes']))==841
 review={'status':'passed','schema':'T23_portable_inherited_recorded_history_review_v1','kind':'inherited_review_not_new_private_tree_scan','known_history_chain_sha256':history['canonical_chain_sha256'],'canonical_private_review_sha256':history['canonical_review_sha256'],'attestation_sha256':sha(md/'history_attestation.json'),'historical_calibration_acceptance_sha256':CALIBRATION_ACCEPTANCE_SHA,'no_unaccounted_model_experiments':'Inherited historical coordinator attestation; not a new claim about the reproducer host.'}
 dm=copy.deepcopy(private_dm);dm['exclusion_inventory_sha256']=sha(md/'exclusion_inventory.json')
 dm['T23_calibration_manifest_sha256']=d['calibration_assets']['calibration_study/manifest.json'];dm['T23_calibration_acceptance_sha256']=d['calibration_assets']['calibration_study/acceptance.json']
 pre={'status':'passed','phase':'completed','kind':'T23_portable_tokenizer_only_reconstruction_v1','model_forwards':0,'pretrained_loaded':False,'gpu_used':False,'calibration_fitted':False,'evaluation_passages':16,'evaluation_inputs':32,'history_hash_count':841,'canonical_history_hash_count':809,'auxiliary_suffix_hash_count':32,'historical_exposure_is_inherited':True,'historical_calibration_forwards':16,'calibration_reexecuted':False,'metadata_identity_sha256':sha(package/'identity.json'),'canonical_private_preflight_sha256':d['canonical_private_input_sha256']['preflight.json'],'current_sources_checked':list(PHYSICAL),'all_candidates_rechecked':True}
 return {'data_manifest.json':dm,'known_history_chain.json':chain,'known_history_review.json':review,'preflight.json':pre,'start_identity_summary.json':read(md/'start_identity.template.json')}

def rebuild(a):
 outside_sources(a.out,a.package);d=validate_metadata(a.package);os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 import torch
 from datasets import Dataset
 code=a.package/'experiments/pulse_transfer';sys.path.insert(0,str(code));selector=importlib.import_module('post_write_norm_selector');assert Path(selector.__file__).resolve()==(code/'post_write_norm_selector.py').resolve()
 torch.set_num_threads(1);dm=read(a.package/'metadata/data_manifest.template.json');inv=read(a.package/'metadata/exclusion_inventory.json');tok,literals,cfg=tokenizer(a.package,a.config);assert value_sha(literals)==d['private_literals_commitment'];texts={}
 for c in PHYSICAL:
  p=Path(cfg['sources'][c]);spec=dm['sources'][c];assert sha(p)==spec['file_sha256']
  if c=='wiki_validation':text='\n\n'.join(Dataset.from_file(str(p))['text'])
  elif c in ('utilitarianism','ethics'):
   raw=p.read_text();lo=raw.find('*** START OF');hi=raw.rfind('*** END OF');assert lo>=0 and hi>lo;text=raw[raw.find('\n',lo)+1:hi]
  else:text=p.read_text()
  assert len(text)==spec['characters'] and selector.textsha(text)==spec['processed_text_sha256'];texts[c]=text
 texts['optimizers_new']='\n\n'.join(texts[c] for c in ('adadelta','adamax','asgd'));assert selector.textsha(texts['optimizers_new'])==dm['sources']['optimizers_new']['processed_text_sha256'];components=dm['sources']['optimizers_new']['components'];cursor=0
 for c,component in zip(('adadelta','adamax','asgd'),components):assert component=={'source':c,'char_start':cursor,'char_end':cursor+len(texts[c]),'file_sha256':dm['sources'][c]['file_sha256']};cursor+=len(texts[c])+2
 progress={};history=set(inv['history_token_hashes']);assert len(history)==841;selector.select_panel(CORPORA,texts,tok,inv['exclusions'],history,components,False,progress,bos_id=tok.bos_token_id)
 assert len(progress['items'])==16 and clean(progress['selected'])==dm['passages'] and progress['rejections']==dm['rejections'];assert sum(map(len,progress['rejections'].values()))==1074
 private_dm=copy.deepcopy(dm);private_dm['passages']=progress['selected'];private_dm['literals']=literals;objects=portable_input_objects(a.package,d,private_dm)
 assert validate_metadata(a.package)==d
 for c in PHYSICAL:assert sha(Path(cfg['sources'][c]))==dm['sources'][c]['file_sha256']
 fresh(a.out,32*1024**2);torch.save(progress['items'],a.out/'tokens.pt');assert sha(a.out/'tokens.pt')==d['canonical_private_input_sha256']['tokens.pt']
 copy_exact(a.package/'metadata/exclusion_inventory.json',a.out/'exclusion_inventory.json')
 for n,v in objects.items():put(a.out/n,v)
 put(a.out/'input_bindings.json',{'schema':'T23_portable_input_binding_v1','original':d['canonical_private_input_sha256'],'regenerated':{n:sha(a.out/n) for n in d['canonical_private_input_sha256']},'metadata_identity_sha256':sha(a.package/'identity.json'),'tokens_byte_exact':True,'historical_exposure_is_inherited':True,'model_forwards':0})

def validate_private_inputs(a,d):
 import torch
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 expected=set(d['canonical_private_input_sha256'])|{'input_bindings.json'};assert base.scan_package(a.inputs,set())==expected
 binding=read(a.inputs/'input_bindings.json');assert set(binding)=={'schema','original','regenerated','metadata_identity_sha256','tokens_byte_exact','historical_exposure_is_inherited','model_forwards'} and binding['schema']=='T23_portable_input_binding_v1' and binding['original']==d['canonical_private_input_sha256'] and binding['metadata_identity_sha256']==sha(a.package/'identity.json') and binding['tokens_byte_exact'] is True and binding['historical_exposure_is_inherited'] is True and binding['model_forwards']==0
 assert set(binding['regenerated'])==set(d['canonical_private_input_sha256'])
 for n,h in binding['regenerated'].items():assert sha(a.inputs/n)==h
 assert sha(a.inputs/'tokens.pt')==d['canonical_private_input_sha256']['tokens.pt'] and sha(a.inputs/'exclusion_inventory.json')==sha(a.package/'metadata/exclusion_inventory.json')
 tok,literals,_=tokenizer(a.package,a.config);assert value_sha(literals)==d['private_literals_commitment'];items=torch.load(a.inputs/'tokens.pt',weights_only=True,map_location='cpu');dm=read(a.package/'metadata/data_manifest.template.json');assert len(items)==len(dm['passages'])==16
 for item,record in zip(items,dm['passages']):
  assert set(item)=={'corpus','window','tokens'} and (item['corpus'],item['window'])==(record['corpus'],record['window']);t=item['tokens'];assert t.dtype==torch.int64 and t.shape==(512,) and hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()==record['window_token_sha256'];bos=t.clone();bos[0]=tok.bos_token_id;assert hashlib.sha256(bos.numpy().tobytes()).hexdigest()==record['bos_window_token_sha256'];record['first_token_id']=int(t[0])
 dm['literals']=literals;objects=portable_input_objects(a.package,d,dm)
 for n,v in objects.items():assert read(a.inputs/n)==v,('Portable scientific metadata altered',n)
 return binding
def staged_source_bytes(package,binding):
 require_release(metadata=True)
 assert STAGED_SOURCE_BINDINGS and STAGED_JSON_BINDINGS,'HOLD: exact consumed T23 certificate/data assignment maps not reviewed'
 original={name:(package/'experiments/pulse_transfer'/name).read_bytes() for name in FROZEN_SOURCE_HASHES}
 assert all(hashlib.sha256(data).hexdigest()==PUBLIC_SOURCE_HASHES[name] for name,data in original.items())
 assert binding['original']==CANONICAL_INPUTS and set(binding['regenerated'])==set(CANONICAL_INPUTS)
 assert binding['regenerated']['tokens.pt']==CANONICAL_INPUTS['tokens.pt'] and binding['regenerated']['known_history_chain.json']==CANONICAL_INPUTS['known_history_chain.json']
 derived={};visiting=set();changes={}
 def resolve(spec):
  assert isinstance(spec,dict) and len(spec)==1
  kind,arg=next(iter(spec.items()))
  if kind=='input_hashes':assert arg is True;return copy.deepcopy(binding['regenerated'])
  if kind=='input_sha256':assert arg in CANONICAL_INPUTS;return binding['regenerated'][arg]
  if kind=='calibration_sha256':assert arg in CALIBRATION_PUBLIC_SHA256;return CALIBRATION_PUBLIC_SHA256[arg]
  if kind=='source_sha256':assert arg in original;return hashlib.sha256(visit(arg)).hexdigest()
  if kind=='source_map_excluding':
   assert isinstance(arg,list) and set(arg)<=set(original)
   return {name:hashlib.sha256(visit(name)).hexdigest() for name in original if name not in arg}
  raise AssertionError('Unreviewed staging binding kind')
 def visit(name):
  if name in derived:return derived[name]
  assert name not in visiting,'Finite certificate DAG cycle';visiting.add(name);data=original[name];edits=[]
  edges=CERTIFICATE_HASH_EDGES.get(name,{})
  if edges or name in STAGED_JSON_BINDINGS:
   assert Path(name).suffix=='.json';value=json.loads(data)
   for location,target in edges.items():
    assert pointer(value,location)==FROZEN_SOURCE_HASHES[target];new=hashlib.sha256(visit(target)).hexdigest();pointer(value,location,new,write=True);edits.append({'pointer':location,'target':target,'staged_sha256':new})
   for location,spec in STAGED_JSON_BINDINGS.get(name,{}).items():
    assert set(spec)=={'original_value_sha256','binding'};old=pointer(value,location);assert hashlib.sha256(json_bytes(old)).hexdigest()==spec['original_value_sha256'];new=resolve(spec['binding']);pointer(value,location,new,write=True);edits.append({'pointer':location,'binding':spec['binding']})
   safe(value);data=json_bytes(value)
  if name in STAGED_SOURCE_BINDINGS:
   assert Path(name).suffix=='.py';replacements={key:resolve(spec) for key,spec in STAGED_SOURCE_BINDINGS[name].items()}
   data=assignment_only(data.decode(),replacements).encode();edits.append({'assignments':sorted(replacements)})
  visiting.remove(name);derived[name]=data
  if edits:changes[name]=edits
  return data
 assert set(CERTIFICATE_HASH_EDGES)|set(STAGED_JSON_BINDINGS)|set(STAGED_SOURCE_BINDINGS)<=set(original)
 for name in original:visit(name)
 receipt={'schema':'T23_portable_certificate_binding_v1','status':'prepared_not_executed','stage_binding_review_sha256':STAGE_BINDING_REVIEW_SHA,'canonical_original_sha256':FROZEN_SOURCE_HASHES,'public_view_sha256':PUBLIC_SOURCE_HASHES,'staged_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in derived.items()},'changed_bindings':changes,'scientific_functions_guard_predicates_order_scoring_AST_exact':True,'scope':'Reviewed metadata-binding derivatives; original manifests/audits attest original execution only. Optional1160model replay and fresh provenance-aware N/P/R comparators unperformed.','model_forwards':0,'calibration_forwards':0}
 safe(receipt);return derived,receipt

def stage_replay(a):
 require_release(metadata=True);outside_sources(a.out,a.package,a.inputs);d=validate_metadata(a.package)
 binding=validate_private_inputs(a,d);staged,receipt=staged_source_bytes(a.package,binding)
 fresh(a.out,128*1024**2);dest=a.out/'pulse_transfer';dest.mkdir()
 for name,data in staged.items():
  path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(data)
 for name,value in d['metadata_dependencies'].items():copy_exact(a.package/'metadata_dependencies'/value['file'],dest/name,value['sha256'])
 for name,h in d['calibration_assets'].items():copy_exact(a.package/name,a.out/name,h)
 acceptance='review/t23_calibration_actual_result_independent_acceptance_v1.json'
 copy_exact(a.package/'calibration_study/acceptance.json',dest/acceptance,d['calibration_assets']['calibration_study/acceptance.json'])
 put(a.out/'portable_certificate_bindings.json',receipt)
 common=[sys.executable,'-B',str((dest/'gpt2_calibration_world_evaluation.py').resolve()),'--config',str(a.config.resolve()),'--inputs',str(a.inputs.resolve()),'--calibration-reference',str((a.out/'calibration_reference').resolve()),'--calibration-study',str((a.out/'calibration_study').resolve())]
 commands={'A':common+['--segment','A','--out',str((a.out/'segment_A').resolve())],'B':common+['--segment','B','--previous-segment',str((a.out/'segment_A').resolve()),'--out',str((a.out/'segment_B').resolve())],'merge':[sys.executable,'-B',str((dest/'merge_calibration_world_evaluation.py').resolve()),'--segment-a',str((a.out/'segment_A').resolve()),'--segment-b',str((a.out/'segment_B').resolve()),'--out',str((a.out/'composite').resolve())],'analysis':[sys.executable,'-B',str((dest/'analyze_calibration_world_evaluation.py').resolve()),'--out',str((a.out/'composite').resolve())]}
 put(a.out/'replay_preparation.json',{'status':'prepared_only','scheduled_evaluation_forwards':1160,'segment_forwards':580,'calibration_forwards_to_execute':0,'historical_completed_calibration_forwards':16,'model_forwards':0,'canonical_runner_sha256':FROZEN_SOURCE_HASHES['gpt2_calibration_world_evaluation.py'],'staged_runner_sha256':sha(dest/'gpt2_calibration_world_evaluation.py'),'commands':commands,'scope':'No model subprocess is invoked. Accepted originalF/CE assets reused; freshcal16 and optional1160evaluation replay/comparators remain unperformed. Staging is not launch approval or independent replication.'})

def main():
 global OWNED
 OWNED=None;ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
 for action,names in (('export-metadata',('run','out')),('validate-metadata',('package','report')),('rebuild',('package','config','out')),('stage-replay',('package','inputs','config','out')),('export-numeric',('run','audits','out')),('audit-public',('package','report'))):
  cmd=sub.add_parser(action)
  for name in names:cmd.add_argument('--'+name,type=Path,required=True)
  if action=='audit-public':cmd.add_argument('--raw',type=Path)
 a=ap.parse_args()
 try:
  if a.action=='validate-metadata':
   value=validate_metadata(a.package);put(a.report,{'status':'passed','schema':'T23_JSON_only_metadata_validation_v1','identity_sha256':sha(a.package/'identity.json'),'model_forwards':0,'numerical_reanalysis':False})
  elif a.action=='audit-public':
   from audit_calibration_world_public import audit
   put(a.report,audit(a.package,ROOT,a.raw))
  else:{'export-metadata':export_metadata,'rebuild':rebuild,'stage-replay':stage_replay,'export-numeric':export_numeric}[a.action](a)
 except BaseException as error:
  if hasattr(a,'out') and OWNED is not None and a.out.is_dir() and OWNED==(a.out.resolve(),a.out.stat().st_dev,a.out.stat().st_ino):put(a.out/'portable_failure.json',{'status':'failed','action':a.action,'error_type':type(error).__name__,'error':str(error),'model_forwards':0})
  raise

if __name__=='__main__':main()
