# Changelog

## Unreleased

**New verb `newt run <tag>`** — the CLI hero verb: take a model tag, authenticate, load a real bundled observation, and call the model **once against prod**, then print the resolved model, the round-trip latency, and the action-chunk shape. Built on the parts that already ship — `Robot(model=tag)`'s in-constructor tag→registry resolve, the callback-free one-shot `infer(obs)`, and the self-describing `newt.snapshots`. Flags: `--snapshot <name>` (default `cup_stacking`; also `pour_coffee_beans`), `--prompt "..."` to override the snapshot's recorded prompt, and `--json` for the machine-readable mirror. **v1 is hardware-free by design** — no robot is connected and nothing moves; the human output says so plainly so a live inference is never mistaken for a robot demo. Errors render in the house shape (`newt: <problem> — <hint>`, no-key → `newt login`), and a server detail carrying a pending/dead own-model `model_status` is surfaced **verbatim**. `--embodiment` and the streaming loop against real hardware are a separate future phase.

New verb `newt promote <job-handle> --band <token>` — keep a fine-tune's checkpoint band and serve it from the terminal, the CLI twin of the console's promote button. The band is registered as a model born `pending`; the normal admission chain takes it live, and the output points at `newt models` to watch it. Every server refusal prints its plain reason verbatim — a band whose eval hasn't landed, a checkpoint whose location the pipeline hasn't reported yet, a run already promoted (with its existing model's uid) — never a generic failure. `--band` is passed verbatim; `--json` emits the route's response body on stdout. Reaching the promote route from the CLI required teaching it to accept a Bearer key beside its browser session (a portal-side change).

**Breaking behavior change:** `Robot()` credential resolution is now env-first — `NT_API_KEY` environment variable wins over `~/.nt/credentials` file. Previously the SDK read the file first, while CLI verbs were already env-first; this unifies both to the 12-factor convention. Only observable when both sources are set and disagree.

Default bootstrap URL flipped from the NT0-FP3 Modal serve URL to the always-on registry at https://nt-registry-production.up.railway.app (merge `f59a993`). First `Robot()` now resolves routing in about a second even when the GPU container is cold, and no longer raises `RegistryUnavailable` on a cold container. `NT_BOOTSTRAP_URL` and `NT_INFERENCE_URL` overrides are unchanged.

## v0.0.1 — 2026-05-29

Extracted from `nt-runway/src/newt` at commit `efc9b38`. History for SDK files preserved via `git filter-repo --path src/newt`.
