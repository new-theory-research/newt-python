# newt-python

Python SDK for the [New Theory](https://newtheory.ai) inference API. Connects your robot to cloud-hosted action policies.

## Quickstart

```bash
# 1. Install the newt CLI globally (works in every shell — nothing to activate)
uv tool install "git+ssh://git@github.com/new-theory-research/newt-python.git"

# 2. Log in (one browser confirm; key persists to ~/.nt/credentials)
newt login

# 3. Confirm you can reach the API
newt models
```

The library installs into a project, exactly when you have Python to write:

```bash
# 4. Add the library to a project, then connect from Python
uv init my-robot && cd my-robot
uv add "newt @ git+ssh://git@github.com/new-theory-research/newt-python.git"
uv run python -c "from newt import Robot; print(Robot())"
# so101 · contract received · (30,6) · 6 labeled axes
```

`Robot()` reads the credentials `newt login` created — no second login, no shell export — and fetches the model contract from the registry. That printed line confirms you reached the API; you're done with milestone 1.

## Recording (alpha)

Record demonstrations as NT-format episodes — the native training intake — with
one optional install and one verb. Recording is an extra so plain `import newt`
stays light for everyone who only runs inference:

```bash
uv pip install "newt[recording]"
```

The library is the moat; the CLI is hospitality. `newt.recording.Session` holds
all the behavior — the capture loop, atomic episode writes (a discarded or killed
episode leaves no directory), dropped-frame accounting, and a descriptive
preflight. `newt record` is a thin keyboard skin on it:

```bash
# Exercise the rhythm with a simulated arm — no hardware needed.
newt record --task "stack the cups" --simulate
#   SPACE start/stop · ENTER keep · D discard · R redo · Ctrl+H kill

# Validate any episode you recorded.
newt episodes validate ./episodes/episode_<id>
```

Episodes are written in **one** format only (NT v0.0.3); every episode carries
its task prompt and declared provenance (author/license, marked unverified for
local-first capture). Agents drive the same Session over `--json` (line-delimited
events + stdin commands); a missing extra produces a lantern naming
`pip install "newt[recording]"`, not a stack trace.

To record from your own rig, pass a `RecordingSource` (an object with a
`descriptor` and `read_state()`) to `newt.recording.Session(...)` directly.

Recording is alpha — the surface and the format version may move.

## Full guide

**[Getting started →](https://newtheory-docs.vercel.app/docs/getting-started)** — install, auth, first inference call, no robot required.

For hardware setup, use a [starter kit](https://newtheory-docs.vercel.app/docs/starters). See [CHANGELOG.md](CHANGELOG.md) for version history.
