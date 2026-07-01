# Troubleshooting

Running log of known failure shapes and how to diagnose them. Append entries; don't rewrite history.

## 2026-07-01 — brief-306 dependency prune: what to look for if behavior gets weird

**What changed:** the base install dropped `modal`, `fastapi`, `uvicorn` (never imported anywhere — skeleton cruft from the original repo scaffold), and `sshtunnel` + `ml_collections` were removed entirely (their only consumers, `src/newt/_client/run_robot_client.py` and `src/newt/_client/imitation_mirror/`, were deleted by brief-246). Base deps are now exactly: `msgpack`, `numpy`, `httpx`, `websockets`, `pillow`.

**If something breaks after this, the shape to look for is `ModuleNotFoundError` for one of the five removed packages** — coming NOT from `newt` itself, but from:

- **A user script or notebook that did `pip install newt` and then `import modal` / `import fastapi`** — it was free-riding on our bloat. Correct fix: that project declares the dep itself. Don't re-add it here.
- **An older starter, smoke script, or CI job pinned to a pre-306 install path** — e.g. anything assuming `.[hardware]` provides `sshtunnel`/`ml_collections`. The `[hardware]` extra now contains only the pinned `lerobot` fork.
- **Old docs/snippets** floating around that show a serve-side flow (uvicorn/modal) "just working" in a newt env.

**How to confirm it's this and not something else:** `pip show modal fastapi uvicorn sshtunnel ml_collections` in the failing env — if absent and the traceback's import site is outside `src/newt/`, it's a downstream free-rider, not a newt bug. If the import site is *inside* `src/newt/`, that's a real regression: someone added a consumer without declaring the dep, and deptry in CI should have caught it — check why the gate didn't fire before patching.

**Guard added:** deptry runs in CI (`.github/workflows/test.yml`) and fails on declared-but-unused or used-but-undeclared base deps, so aspirational deps can't silently regrow. Verified to bite on 2026-07-01 (dummy dep → `DEP002` failure → revert → green).

**Receipts:** brief-306 card in portal (`wiki/briefs/cards/brief-306-newt-sdk-dep-prune/`), `T1-starter-report.md` (all four public starters verified clean of these imports), validator run wf_5cd2a82c-fbd (APPROVE, 0 fix passes).
