# T1 — starter repo falsification report (brief-306)

Gate for T2: verify no `new-theory-research/newt-starter-*` repo imports `modal`, `fastapi`, `uvicorn`,
`sshtunnel`, or `ml_collections` without declaring it itself, and none relies transitively on newt-python's
own declaration of those deps (i.e. none touches newt's `[hardware]` extra path — `_client/run_robot_client.py`
or `_client/imitation_mirror/*`).

**Method:** `gh repo list new-theory-research --limit 50` to enumerate the org, filtered to `newt-starter-*`
(public repos only — 4 found, no others). Shallow-cloned each (`git clone --depth 1`) to a scratch dir, then:
- `grep -rniE 'import (modal|fastapi|uvicorn|sshtunnel|ml_collections)|from (modal|fastapi|uvicorn|sshtunnel|ml_collections)' <repo> --include="*.py"` — code-level import check.
- `grep -rniE 'modal|fastapi|uvicorn|sshtunnel|ml_collections' <repo> --exclude-dir=.git` — broad check (strings, comments, subprocess, config).
- Full read of each repo's `pyproject.toml` dependency declarations.

Repos found in org (public, `newt-starter-*` prefix):
- `newt-starter-trossen-widowx`
- `newt-starter-yam`
- `newt-starter-so101`
- `newt-starter-yam-bimanual`

(`rebot-dev` and `yam-dev` are private dev-integration repos, not `newt-starter-*` — out of scope per brief.)

## Per-repo verdicts

### newt-starter-trossen-widowx — SAFE
- No `import modal|fastapi|uvicorn|sshtunnel|ml_collections` anywhere in the repo (zero `.py` hits).
- Broad grep: one hit, `.github/workflows/install-golden.yml:134`, a comment referring to "modal cold-start"
  (the New Theory *cloud service* colloquially called "Modal", not the `modal` Python package) — not code, not
  an import.
- `pyproject.toml:6-9` dependencies: `newt`, `lerobot` (from `lerobot-nt` fork), `numpy`, `opencv-python-headless`.
  `pyproject.toml:15-18` hardware extra: `pyrealsense2`, `trossen-arm`. None of the five pruned/moved deps
  appear anywhere. Hardware path uses `lerobot`/`trossen-arm` directly, not newt's `_client/run_robot_client.py`
  or `imitation_mirror` subtree.
- **Verdict: does not import or declare any of the five deps. Not affected by the prune.**

### newt-starter-yam — SAFE
- No `import modal|fastapi|uvicorn|sshtunnel|ml_collections` anywhere in the repo (zero `.py` hits).
- Broad grep: zero hits, no matches at all.
- `pyproject.toml:6-7` dependencies: `newt`, `numpy`. No optional-dependencies block exists at all.
- **Verdict: does not import or declare any of the five deps. Not affected by the prune.**

### newt-starter-so101 — SAFE
- No `import modal|fastapi|uvicorn|sshtunnel|ml_collections` anywhere in the repo (zero `.py` hits).
- Broad grep: one hit, `.github/workflows/install-golden.yml:133`, a comment referring to "Modal cold-start"
  (same cloud-service usage as trossen-widowx, not the Python package) — not code, not an import.
- `pyproject.toml:6-9` dependencies: `newt`, `lerobot` (from `lerobot-nt` fork), `numpy`, `opencv-python-headless`.
  `pyproject.toml:16-18` hardware extra: `lerobot[feetech]`. None of the five pruned/moved deps appear anywhere.
  Hardware path uses `lerobot` directly for Feetech servo control, not newt's robot-client subtree.
- **Verdict: does not import or declare any of the five deps. Not affected by the prune.**

### newt-starter-yam-bimanual — SAFE
- No `import modal|fastapi|uvicorn|sshtunnel|ml_collections` anywhere in the repo (zero `.py` hits).
- Broad grep: five hits — `run.py:108`, `README.md:71`, `README.md:117`, `yam/config.py:32`, `yam/warm.py:1,5,8,26`
  — all prose/comments referring to "Modal cold-start" / "IPv6 → Modal" (the New Theory cloud inference
  backend, MolmoAct-2-on-Modal), never the Python `modal` package, never an import statement.
- `pyproject.toml:6-16` dependencies: `numpy`, `opencv-python-headless`, `newt`, `i2rt`, `pyrealsense2`.
  `pyproject.toml:18-20` `mac-can` extra: `python-can[gs_usb]`. None of the five pruned/moved deps appear
  anywhere. Hardware path (CAN bus control) uses `i2rt` directly, not newt's robot-client subtree.
- **Verdict: does not import or declare any of the five deps. Not affected by the prune.**

## Gate result

All four `newt-starter-*` repos: **SAFE.** Zero imports of `modal`, `fastapi`, `uvicorn`, `sshtunnel`, or
`ml_collections` in any starter's Python code; zero declarations of these packages in any starter's
`pyproject.toml`; no starter consumes newt-python's `[hardware]` extra or `_client/run_robot_client.py` /
`_client/imitation_mirror/*` subtree (each starter drives hardware through its own `lerobot`/`i2rt` dependency
instead). Every prose match for "Modal" across all four repos refers to the New Theory cloud inference service,
not the Python `modal` package.

T2 (dependency prune) is clear to proceed.
