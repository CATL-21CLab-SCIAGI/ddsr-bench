# HiPhO evaluator provenance

`hipho.py`, `hipho_verifier.py`, and `hipho_prompt_inference.py` are copied
unchanged from `open-compass/VLMEvalKit` commit
`ed3e918edf4a62e0bf42a2768655e02d45312fef` under Apache-2.0.
`hipho_verifier.py` is Copyright 2025 Garena Online Private Limited.

`shim.py` is copied unchanged from `bytedance/axion1_0` commit
`96ce1830d87399f20577790f405fe2407da34156` under Apache-2.0. It exposes only
VLMEvalKit's per-problem scorer without installing the full toolkit.

Pinned SHA-256 digests:

- `hipho.py`: `ceae7b2e9776b7d6c02c8303d732bf7204b1c0aeba032bbcf416b0623d7effed`
- `hipho_verifier.py`: `e9419d8f460efa7eb700334b3b03085463cb208f301644cac11848448a18430f`
- `hipho_prompt_inference.py`: `217f1f4ddc6e2b5c580c392f815df7b2baffe7e26349f41b5bec7815a87121b1`

Do not format or edit these files. DDSR-Bench-owned adaptation belongs in the
parent `evaluation` package.
