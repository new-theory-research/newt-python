# Troubleshooting

Known failure shapes and how to diagnose them.

## `ModuleNotFoundError` for `modal`, `fastapi`, `uvicorn`, `sshtunnel`, or `ml_collections`

The base install of `newt` depends only on `msgpack`, `numpy`, `httpx`, `websockets`, and `pillow`. It does **not** pull in `modal`, `fastapi`, `uvicorn`, `sshtunnel`, or `ml_collections` — earlier builds shipped some of those by accident, and a project that did `import modal` (etc.) alongside `newt` was free-riding on that bloat.

If one of those imports now fails after installing `newt`, the fix is to declare the package in your own project — don't expect `newt` to provide it:

```bash
uv add modal   # or fastapi, uvicorn, etc. — whatever your code actually imports
```

**How to confirm it's this and not a `newt` bug:** run `pip show modal fastapi uvicorn sshtunnel ml_collections` in the failing environment. If the packages are absent and the traceback's import site is in *your* code (not inside the installed `newt` package), it's a missing dependency in your project, not a `newt` regression.
