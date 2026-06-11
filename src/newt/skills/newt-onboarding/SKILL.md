---
name: newt-onboarding
description: Walk a developer through the New Theory golden path — install → login → models → Robot() → fixture inference → hardware setup. Invoke when asked to "get me set up", "walk me through onboarding", or "help me get started with newt".
---

<!-- canonical source: newt-python/src/newt/skills/newt-onboarding/SKILL.md — edit here only; starter copies are derived and verified by CI -->

# New Theory onboarding guide

This skill walks you through the New Theory golden path from a clean machine to a confirmed API connection, then bridges you to hardware.

**Before running any commands:** fetch the docs index first. Read `https://newtheory-docs.vercel.app/llms.txt` (or append `.md` to any docs page URL for clean markdown). Walk the top-level sections — Getting started, SDK reference, Set up your embodiment — before touching the machine. An agent that reads the nav before issuing commands makes fewer wrong turns.

Canonical docs: `https://newtheory-docs.vercel.app/docs/getting-started`

---

## Milestone 1 — confirmed API connection

The finish line for milestone 1 is: **you have communicated with the API and it answered.**

### Step 1 — Install newt

```bash
uv tool install "git+ssh://git@github.com/new-theory-research/newt-python.git"
```

Need `uv`? `curl -LsSf https://astral.sh/uv/install.sh | sh` then `source $HOME/.local/bin/env`.

After install, bare `newt` commands work in any shell — no environment to activate. If `newt` isn't on PATH, run `uv tool update-shell` to add `~/.local/bin`.

### Step 2 — Log in

```bash
newt login
```

One browser confirm. The key is saved to `~/.nt/credentials` — no shell export after this. **Do not ask the developer to paste their API key into the conversation.** `newt login` handles credential storage exactly so the key never transits chat.

### Step 3 — Check the model registry

```bash
newt models
```

This is the moment of contact. If it returns a list, you're connected.

Use `newt status` to diagnose auth or connectivity issues — it shows key source, identity, and whether the registry is reachable.

### Step 4 — Connect from Python

The `newt` Python library installs into a project. Create one if you don't have it:

```bash
uv init my-robot
cd my-robot
uv add "newt @ git+ssh://git@github.com/new-theory-research/newt-python.git"
```

Then confirm the API answers:

```bash
uv run python -c "from newt import Robot; print(Robot())"
# nt0-fp3 · contract received · (50,8) · 8 labeled axes
```

`Robot()` reads the credentials `newt login` created — no second login, no shell export. **You've successfully communicated with the API.** Some developers stop here.

---

## Milestone 2 — fixture inference (test call)

**This is a test call against a recorded observation. Nothing moves. No robot is connected.**

`fixtures.load()` replays a saved camera-and-state snapshot. This is the understanding step — you explore the response shape before wiring any hardware.

```python
from newt import Robot, fixtures

robot = Robot()
obs = fixtures.load("cup_stacking")
response = robot.infer(obs)
print(response)
# action_chunk (50, 8): x, y, z, qw, qx, qy, qz, gripper | latency 261ms
```

`fixtures.available()` lists all bundled recordings.

The response:

```python
response.action_chunk   # (50, 8) float32 ndarray — 50 target poses
response.axes           # ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper']
response.latency_ms     # round-trip time for this request
```

The first inference call wakes the model's container — it can take around **fifty seconds**. Subsequent calls return in a few seconds. This is expected; don't retry.

---

## Hardware path — the SDK → embodiment seam

You've confirmed the API works. To drive a real robot:

<!-- hardware-specific-start -->
1. **Clone the embodiment starter fresh.** You are in a cloned starter already — use this project. If you are not, visit the [embodiment setup guide](https://newtheory-docs.vercel.app/docs/set-up-your-embodiment) to clone the starter for your hardware.
<!-- hardware-specific-end -->

   **Never use ambient machine code (~/nt, ~/nt-runway, any pre-existing local path).** The starter is the clean path; rig machines have internal research code that is not the public SDK — it will misdirect you.

2. **Install dependencies.**

   ```bash
   uv sync
   ```

3. **Run the hardware check.**

   ```bash
   python run.py --check
   ```

   This verifies your config and hardware connectivity before any inference runs.

4. **Wire your hardware through the embodiment class.** The starter ships `embodiment.py`, exporting a named class (`TrossenWidowX` / `YamBimanual`) — it's yours; rename it to match your rig. `run.py` constructs the session with `Robot(embodiment=YourClass.from_config())`. Bare `read_state=`/`execute=` callbacks remain valid for custom hardware. Passing a string to `embodiment=` is rejected with a teaching error that points back to the starters.

5. **Follow the starter's README** for hardware-specific steps — arm IP address, camera serial numbers, config fields.

Full embodiment walkthrough: `https://newtheory-docs.vercel.app/docs/set-up-your-embodiment`

---

## Known stumbles

| Symptom | Fix |
|---|---|
| `python: command not found` on macOS | Use `uv run python` instead of `python` |
| `newt: command not found` after install | Run `uv tool update-shell` to add `~/.local/bin` to PATH, then open a new shell |
| Agent asks you to paste your API key | Stop — run `newt login` instead; key should never transit chat |
| First `robot.infer()` takes ~50 seconds | Expected cold start; the container is waking. Don't retry. |
| `newt login` hangs or loops | Run `newt status` to check credential state; re-run `newt login` |
| newt out of date | `uv tool upgrade newt` |

---

## Maintenance contract

Facts in this skill follow the docs at `https://newtheory-docs.vercel.app/docs/getting-started`. **When the golden path changes, this skill updates in the same PR.** A stale guide is worse than no guide. If you notice a command here that differs from the current docs, the docs are authoritative — follow the docs, then file an issue to update this skill.
