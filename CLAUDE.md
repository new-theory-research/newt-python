# newt-python — agent instructions

## What this repo is

The Python SDK for the New Theory inference API. Distribution name: `newt`; import name: `newt` (`import newt`). Install: `pip install git+ssh://git@github.com/new-theory-research/newt-python.git`. Private repo — PyPI publish is deferred.

## Repo map

- **newt-python** (this repo) — SDK surface. The only thing developers import.
- **portal** (github.com/new-theory-research/portal) — product surface: console, docs, API contracts. Engineering specs live here.
- **nt-runway** (github.com/new-theory-research/nt-runway) — server-only: inference serve layer, Modal infra. Internal; developers never see it.
- **newt-starter-trossen-widowx / newt-starter-yam** — starter repos that consume this SDK.

## Naming

The import name is `newt`, never `nt`. `nt` shadows `ntpath` and causes circular import errors.

## Rule 10 — Fail loud; never invent plausible values

**Never invent plausible values for missing inputs** — no identity matrices, zero-fills, or shaped-right defaults standing in for data the caller didn't send. A substitution is either a deliberate, declared decision the caller can see, or an error — never a silent courtesy. *(Receipt: NT0's extrinsics identity-fill served confidently-wrong geometry with zero warning, 2026-06-12.)*
