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
# nt0-fp3 · contract received · (50,8) · 8 labeled axes
```

`Robot()` reads the credentials `newt login` created — no second login, no shell export — and fetches the model contract from the registry. That printed line confirms you reached the API; you're done with milestone 1.

## Full guide

**[Getting started →](https://newtheory-docs.vercel.app/docs/getting-started)** — install, auth, first inference call, no robot required.

For hardware setup, use a [starter kit](https://newtheory-docs.vercel.app/docs/starters). See [CHANGELOG.md](CHANGELOG.md) for version history.
