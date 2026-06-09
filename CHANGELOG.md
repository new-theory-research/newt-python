# Changelog

## Unreleased

Default bootstrap URL flipped from the NT0-FP3 Modal serve URL to the always-on registry at https://nt-registry-production.up.railway.app (merge `f59a993`). First `Robot()` now resolves routing in about a second even when the GPU container is cold, and no longer raises `RegistryUnavailable` on a cold container. `NT_BOOTSTRAP_URL` and `NT_INFERENCE_URL` overrides are unchanged.

## v0.0.1 — 2026-05-29

Extracted from `nt-runway/src/newt` at commit `efc9b38`. History for SDK files preserved via `git filter-repo --path src/newt`.
