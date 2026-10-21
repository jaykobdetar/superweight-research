"""Assemble allowlisted T18 sources/report/metadata after numerical review.

This is a local export, not publication. The release must pass separate review.
"""
from pathlib import Path
import hashlib,json,shutil,re
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent/'superweight-causal-circuits'
RUN=ROOT/'gpt2_gate_wiki_calibration_run'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
EXTRA=('audit_gate_wiki_results_layout.py','build_report14.py','build_report14_figures.py',
       'report14_findings.md','report14_methods.md','report14_prior.md',
       'export_report14_numeric.py','export_report14_package.py',
       'prepare_wiki_calibration_replay_inputs.py','reproduce_wiki_calibration.py',
       'GPT2_WIKI_CALIBRATION_REPLAY.md','GPT2_WIKI_CALIBRATION_REPLAY_PROPOSAL.md')
def main():
    assert j(ROOT/'review/gate_wiki_numerical_audit.json')['status']=='passed'
    assert j(REPO/'manifests/wiki_calibration_numeric_export.json')['main_records']==776
    manifest=j(RUN/'manifest.json');source=j(REPO/'manifests/transfer_source_export.json')
    closure=j(ROOT/'review/wiki_replay_source_manifest.json')['files']
    files={**manifest['source_hashes'],**closure,**{n:sha(ROOT/n)for n in EXTRA}}
    for n,h in files.items():
        assert Path(n).name==n and sha(ROOT/n)==h
        if n in manifest['source_hashes']:assert sha(RUN/'frozen_sources'/n)==h
        assert not re.search(r'/home/[A-Za-z0-9_.-]+/',(ROOT/n).read_text()),('Unexpected personal path in source',n)
        dst=REPO/'experiments/pulse_transfer'/n
        if dst.exists() and n in source['files']:assert sha(dst)==h,('Historical exported source would change',n)
        shutil.copyfile(ROOT/n,dst);source['files'][n]=h
    dump(REPO/'manifests/transfer_source_export.json',source)
    reproduction=REPO/'results/gate_wiki_calibration_reproduction';reproduction.mkdir(exist_ok=True)
    md=reproduction/'metadata';md.mkdir(exist_ok=True);identity=j(ROOT/'wiki_replay_metadata/identity.json')
    expected={**identity['metadata_sha256'],'identity.json':sha(ROOT/'wiki_replay_metadata/identity.json')}
    assert set(p.name for p in(ROOT/'wiki_replay_metadata').iterdir())==set(expected)
    for n,h in expected.items():
        assert Path(n).name==n and sha(ROOT/'wiki_replay_metadata'/n)==h
        shutil.copyfile(ROOT/'wiki_replay_metadata'/n,md/n)
    assert set(p.name for p in md.iterdir())==set(expected)
    parent=reproduction/'parent_metadata';parent.mkdir(exist_ok=True)
    for n,h in {'manifest.json':'60909cdc2af2240ba99bf103c3cbf62146bddffd43c4f8f6810d092632fb5515','selection.json':'56bbb1076fc342ae1f504bf8bca0d72b521ff1b69269e51575375a144eba50ae'}.items():
        assert sha(ROOT/'gpt2_run'/n)==h;shutil.copyfile(ROOT/'gpt2_run'/n,parent/n)
    shutil.copyfile(ROOT/'review/wiki_replay_source_manifest.json',reproduction/'source_inventory.json')
    assert set(p.name for p in reproduction.iterdir())=={'metadata','parent_metadata','source_inventory.json'}
    reviews=REPO/'reviews/gate_wiki_calibration';reviews.mkdir(exist_ok=True);omitted=[]
    for p in sorted((ROOT/'review').iterdir()):
        if not p.name.startswith(('gate_wiki','gate_calibration_factorial','gate_calibration_wiki','wiki_replay','report14','export_report14'))or p.suffix not in('.py','.md','.json','.log'):continue
        text=p.read_text()
        # Keep original review evidence immutable. Path/token-bearing evidence
        # is retained locally and committed by hash; no silent redaction.
        if '/home/'in text or '/tmp/'in text or 'first_token_id'in text:
            omitted.append({'file':p.name,'sha256':sha(p),'reason':'Local-path or token-identity-bearing diagnostic; original preserved locally'});continue
        shutil.copyfile(p,reviews/p.name)
    dump(REPO/'manifests/wiki_calibration_review_export.json',{'private_only':omitted})
    figs=REPO/'figures/gate_wiki_calibration';figs.mkdir(exist_ok=True)
    for p in(ROOT/'figures14').iterdir():
        assert p.suffix in('.png','.svg');shutil.copyfile(p,figs/p.name)
    (REPO/'docs/GPT2_GATE_WIKI_CALIBRATION.md').write_text((ROOT/'REPORT14.md').read_text().replace('](figures14/','](../figures/gate_wiki_calibration/'))
    shutil.copyfile(ROOT/'GPT2_WIKI_CALIBRATION_REPLAY.md',REPO/'docs/GPT2_GATE_WIKI_CALIBRATION_REPRODUCING.md')
    dump(REPO/'manifests/wiki_calibration_source_export.json',{'files':files,'production_source_hashes':manifest['source_hashes'],
        'source_freeze_unchanged':True,'scope':'Frozen production closure plus separately versioned exact-layout audit adapter, report builder, safe numerical exporter and metadata-only preparation/replay wrapper.'})
    print('EXPORTED_T18_PACKAGE',len(files),'source files')
if __name__=='__main__':main()
