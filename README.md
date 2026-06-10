# newt-python

Python SDK for the [New Theory](https://newtheory.ai) inference API. Connects your robot to cloud-hosted action policies.

## Quickstart

```bash
# 1. Install
uv pip install "git+ssh://git@github.com/new-theory-research/newt-python.git"

# 2. Log in (one browser confirm; key persists to ~/.nt/credentials)
newt login

# 3. Confirm you can reach the API
newt models
```

```python
# 4. Connect from Python
from newt import Robot

robot = Robot()
print(robot)
# nt0-fp3 · contract received · (50,8) · 8 labeled axes
```

`Robot()` fetches the model contract from the registry. `print(robot)` confirms you reached the API — you're done with milestone 1.

## Full guide

**[Getting started →](https://nt-docs-eight.vercel.app/docs/getting-started)** — install, auth, first inference call, no robot required.

For hardware setup, use a [starter kit](https://nt-docs-eight.vercel.app/docs/starters). See [CHANGELOG.md](CHANGELOG.md) for version history.
