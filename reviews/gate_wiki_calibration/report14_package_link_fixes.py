"""Prose-only repository adaptations; frozen sources and original reviews untouched."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];repo=ROOT/'superweight-causal-circuits';rd=repo/'reviews/gate_wiki_calibration'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
omitted={x['file']:x for x in json.loads((repo/'manifests/wiki_calibration_review_export.json').read_text())['private_only']}
fixes={'gate_calibration_factorial_failure_review.md':[('audit evidence','gate_calibration_factorial_failure_audit.json'),('audit script','gate_calibration_factorial_failure_audit.py')],'wiki_replay_author_review.md':[('wiki_replay_author_tests.py','wiki_replay_author_tests.py')]}
records=[]
for name,pairs in fixes.items():
 original=ROOT/'pulse_transfer/review'/name;p=rd/name;s=original.read_text();sub=[]
 for label,target in pairs:
  old=f'[{label}]({target})';assert old in s
  new=f'{label} (private-only file `{target}`, SHA256 `{omitted[target]["sha256"]}`; see the [omission manifest](../../manifests/wiki_calibration_review_export.json))'
  s=s.replace(old,new);sub.append({'original':old,'replacement':new})
 p.write_text(s);records.append({'file':str(p.relative_to(repo)),'original_sha256':sha(original),'exported_sha256':sha(p),'substitutions':sub})
(repo/'manifests/wiki_calibration_review_link_adaptations.json').write_text(json.dumps({'scope':'Prose-only link adaptation of exported review copies; private originals and frozen experiment sources unchanged.','files':records},indent=2)+'\n')
p=repo/'docs/GPT2_GATE_WIKI_CALIBRATION_REPRODUCING.md';s=p.read_text();heading='## Frozen source-relative review links'
assert heading not in s
s+='''\n## Frozen source-relative review links\n\nFrozen protocols retain the original local layout so their recorded hashes remain valid. Their two source-relative review links resolve in this repository as follows; the protocols themselves have not been edited.\n\n| Frozen source and original relative link | Repository review |\n| --- | --- |\n| `GPT2_GATE_WIKI_CALIBRATION_REVISION.md` → `review/gate_calibration_factorial_failure_review.md` | [Failure review](../reviews/gate_wiki_calibration/gate_calibration_factorial_failure_review.md) |\n| `GPT2_WIKI_CALIBRATION_REPLAY_PROPOSAL.md` → `review/gate_wiki_replay_contract.json` | [Replay contract](../reviews/gate_wiki_calibration/gate_wiki_replay_contract.json) |\n\nSome historical audit programs and evidence contain local paths or token identities and remain private. Exported review prose identifies those filenames and original hashes through the [omission manifest](../manifests/wiki_calibration_review_export.json). The [link adaptation record](../manifests/wiki_calibration_review_link_adaptations.json) preserves original and exported review hashes and every changed link. Their absence does not affect the standalone NumPy audit or the reviewed replay source closure, but the omitted historical checks are not independently runnable from this package alone.\n'''
p.write_text(s)
print(json.dumps({'adapted_reviews':len(records),'replaced_links':sum(len(x['substitutions']) for x in records),'frozen_sources_changed':False}))
