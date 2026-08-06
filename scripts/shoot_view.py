"""Point a real browser at a live view and write down what it saw.

The measuring instrument, not part of the product. It exists because the only
honest answer to "does the page render the cameras and the robot, and is it the
lean layout rather than Rerun's application" is a picture somebody looked at.

    python scripts/shoot_view.py --url http://localhost:9099/ --out /tmp/shots --shots 3

Out comes one PNG per shot plus every console message and page error the browser
emitted, verbatim — a paraphrase of a browser error is not a receipt.

**Run it on the machine serving the view.** `http://localhost` is a secure
context, which is what lets the page decode camera video; the same page fetched
over a bare LAN address is not, and a screenshot taken from another machine would
be measuring the browser's rules rather than the rig's output.

**Screenshots of a wasm canvas need real time to exist.** The viewer instantiates
46 MB of WebAssembly, compiles shaders, opens the stream and paints. On a software
rasterizer that is several seconds, and a shot taken early is an empty frame that
looks exactly like a failure. `--settle` is that wait and it is long on purpose.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Headless Chromium has no GPU, so wgpu falls back to a software rasterizer. It
# renders correctly and slowly. Without these flags it does not render at all.
SWIFTSHADER = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]


def drive(url: str, out: Path, shots: int, settle: float, interval: float) -> int:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    console: list[dict] = []
    errors: list[str] = []
    taken: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=SWIFTSHADER)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: console.append({"type": m.type, "text": m.text}))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(settle)
        for i in range(shots):
            path = out / f"shot-{i + 1}.png"
            before = time.time()
            page.screenshot(path=str(path))
            taken.append({"shot": i + 1, "path": str(path), "t0": before, "t1": time.time()})
            if i + 1 < shots:
                time.sleep(interval)
        state = page.evaluate(
            "() => ({ state: document.getElementById('state')?.textContent,"
            " note: document.getElementById('note')?.textContent,"
            " overlay: !!document.getElementById('overlay'),"
            " canvases: document.querySelectorAll('canvas').length })"
        )
        browser.close()

    report = {"url": url, "shots": taken, "page": state, "console": console, "errors": errors}
    (out / "shots.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"page": state, "errors": errors, "console_errors": [
        c for c in console if c["type"] == "error"
    ]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--settle", type=float, default=35.0)
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    return drive(args.url, args.out.resolve(), args.shots, args.settle, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
