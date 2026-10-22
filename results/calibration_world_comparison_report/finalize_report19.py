"""Finite presentation-only finalization of the independently reviewed Report19.

Rebuild candidate with its supplied builder first, then invoke --input CANDIDATE
--out FRESH. No scientific calculations, arrays, models or publication actions.
"""
import argparse, base64, hashlib, json
from pathlib import Path

PINS = {
    'candidate.md': '9c3a775d907ea3b42f5b1032f196134b6fac6a6a73d0d082da14f82839f72da1',
    'candidate.html': 'e86dde6cbe76a26f623a48926548b486b4cacbb2482c78406e7fe396902be591',
    'methods.md': 'c7fcf97050abb0c8556ab418cf64017dfcea5701c352df34dc1365c048b7f198',
    'claims.json': 'fd785db34ac81ea94e8bee37e1da914020fed4e13ba0ab1536d41f6179b5f124',
}
REVIEW_SHA = '41f621674c2f35438cc92a17ef3e157b3f3492c3e919a60c0052328973574ad9'
OLD_COMPARISON = 'byte-identical for 128 stored allocation-logit/mass arrays in total'
NEW_COMPARISON = 'byte-identical in all 128 paired stored-array comparisons of allocation logits and mass'
OLD_DRAFT = 'DRAFT — scientific interpretation and presentation pending final review.'
NEW_REVIEWED = 'Independently reviewed; finite study and reproduction limits are stated below.'
OLD_ACCOUNT = 'This report is a draft account of the completed T23 study'
NEW_ACCOUNT = 'This report describes the completed T23 study'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    src, out = args.input.resolve(), args.out.resolve()
    assert src.is_dir() and not out.exists() and not out.is_relative_to(src)
    assert all(not p.is_symlink() for p in args.input.rglob('*'))
    for name, expected in PINS.items():
        assert sha(src/name) == expected, name
    originals = {p.relative_to(src).as_posix(): sha(p) for p in src.rglob('*') if p.is_file()}
    out.mkdir()
    changed = set(PINS) | {'file_inventory.json'}
    for name in originals:
        if name not in changed:
            write(out/name, (src/name).read_bytes())
    methods = (src/'methods.md').read_text().replace('— DRAFT', '— reviewed')
    write(out/'methods.md', methods.encode())
    for original, final in [('candidate.md', 'report19.md'), ('candidate.html', 'report19.html')]:
        text = (src/original).read_text()
        for old, new in [(OLD_COMPARISON, NEW_COMPARISON), (OLD_DRAFT, NEW_REVIEWED), (OLD_ACCOUNT, NEW_ACCOUNT)]:
            assert text.count(old) == 1, (original, old)
            text = text.replace(old, new)
        if final.endswith('.html'):
            assert text.count('<title>Report19 draft</title>') == 1
            text = text.replace('<title>Report19 draft</title>', '<title>Super-weight research — Report 19</title>')
            old_methods = base64.b64encode((src/'methods.md').read_bytes()).decode()
            assert text.count(old_methods) == 1
            text = text.replace(old_methods, base64.b64encode(methods.encode()).decode())
        write(out/final, text.encode())
    claims = json.loads((src/'claims.json').read_text())
    assert claims['status'] == 'DRAFT_descriptive_claims_pending_final_report_review'
    claims['status'] = 'independently_reviewed_finite_descriptive_claims'
    write(out/'claims.json', (json.dumps(claims, indent=2, allow_nan=False)+'\n').encode())
    write(out/'finalize_report19.py', Path(__file__).read_bytes())
    receipt = {
        'status': 'reviewed_report19_presentation_only_finalization',
        'independent_candidate_review_sha256': REVIEW_SHA,
        'reviewed_candidate_sha256': PINS,
        'finalizer_sha256': sha(Path(__file__)),
        'changes': ['128 paired comparisons wording', 'draft labels to reviewed', 'embedded methods label', 'candidate filenames to report19'],
        'scientific_numbers_tables_figures_analysis_unchanged': True,
        'reproduction_status_at_report_time': 'aggregate presentation available; public numeric/model routes pending separate acceptance',
        'model_calls': 0, 'numerical_recomputations': 0,
    }
    write(out/'finalization_receipt.json', (json.dumps(receipt, indent=2)+'\n').encode())
    assert all(sha(src/name) == h for name,h in originals.items())
    inventory = {p.relative_to(out).as_posix(): {'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'file_inventory.json', (json.dumps(inventory, indent=2)+'\n').encode())
    print(json.dumps({'status':receipt['status'], 'HTML_sha256':sha(out/'report19.html'), 'MD_sha256':sha(out/'report19.md'), 'files':len(inventory)+1}))

if __name__ == '__main__':
    main()
