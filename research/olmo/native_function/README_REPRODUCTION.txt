OLMo native computation: numeric-only publication export, 23 October 2026.

This directory is the completed descriptive reanalysis of archived recorded
outputs. It is NOT the later fresh full-model factorial. The adjacent
validated_continuation directory documents only implementation validation and
completed synthetic-prompt baseline controls at its explicit cutoff.

OFFLINE NUMERIC CHECKS (no tokenizer, corpus, downloads or model required):
  OPENBLAS_NUM_THREADS=1 python analyze_native_dependence.py
  OPENBLAS_NUM_THREADS=1 python source_attention_estimands.py
Python 3.12 and NumPy suffice. Commands rewrite only local analysis outputs.

SOURCE-TEXT-FREE EXPORT:
The 42 original numerical result files remain exact. NPZ arrays contain only
floating-point losses and KL divergences. Analysis tables retain numeric losses,
coarse token classes/repetition categories, positions and input hashes, but omit
token IDs and decoded text. Input hashes/starts are in input_manifest.json and
inputs/recovered_window_manifest.json. No tokenizer, corpus token arrays or
source excerpts are included. Historical original-package manifests are kept
under historical/ and do NOT describe this export; EXPORT_MANIFEST.json does.

OPTIONAL LOCAL CORPUS REBUILD:
Acquire the pinned model's tokenizer.json yourself under its original terms,
and place it at inputs/tokenizer.json. Acquire the public inputs listed in
input_manifest.json under their source terms and filenames. Exact source bytes
matter: hashes are mandatory, and current changed Gutenberg files may fail.
For WikiText, supply inputs/wiki_validation_rows.json with ordered validation
rows (the corpus construction joins rows with two newline characters).
  OLMO_ALLOW_LOCAL_TEXT_JOINS=1 OPENBLAS_NUM_THREADS=1 python reconstruct_tokenizer.py
This writes *_ids.npy only after exact full corpus text/token hashes match.
Then optional local-only token joins may run:
  OLMO_ALLOW_LOCAL_TEXT_JOINS=1 OPENBLAS_NUM_THREADS=1 python analyze_recovered_targets.py
  OLMO_ALLOW_LOCAL_TEXT_JOINS=1 OPENBLAS_NUM_THREADS=1 python summarize_discriminating_findings.py
These optional commands intentionally recreate text-bearing local outputs.
Do not redistribute them without independently checking the original licenses.
A missing input or changed hash is not permission to substitute another corpus.

ATTRIBUTION / SOURCE LINKS (original terms remain controlling):
Research outputs: https://github.com/jaykobdetar/superweight-research
Commit: 0bf1747a8d1b4e6a0adfa3aab72005f7b13df096
Model/tokenizer: https://huggingface.co/allenai/OLMo-1B-0724-hf
Revision: d7cbab742d80589e714b1a2d7f838dcd21cbe143
WikiText: https://huggingface.co/datasets/Salesforce/wikitext
Du Bois: https://www.gutenberg.org/ebooks/408
Hobbes: https://www.gutenberg.org/ebooks/3207
Optimizer code: https://github.com/pytorch/pytorch/tree/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/optim
This export is not a legal determination or a license grant for source works.
