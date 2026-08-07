# Live-view bench proof — the pictures somebody looked at

Three screenshots of `newt record --view` running on a real rig, taken by a headless
browser on that rig, while an episode was recording. They exist because the only honest
answer to "does the page draw the cameras and the robot, and is it the lean layout rather
than Rerun's application" is a picture. A picture nobody can find is not a receipt.

| file | what it shows |
| --- | --- |
| `shot-1.png` | timeline at `18:59:44.859657Z`, viewer readout 37.6 s |
| `shot-2.png` | timeline at `18:59:51.161247Z`, readout 39.7 s |
| `shot-3.png` | the third of the three, four seconds after `shot-2` |

Read the badge in the top-right of every one: **`recording · 5f74a3f6`**. The page did not
pause, blank, or drop a pane while the writer was working. That is the claim, and these
are the frames it is made of.

## What is in the frame

Three camera panes carrying live RealSense output (`d405-230422271405`,
`d455f-419222301574`, `d455f-352222302988`), the arm drawn from the kit's URDF, a joint
plot, and a provenance pane naming the description file the drawing came from.

## What these three do not show, and the run that does

These were taken when the layout fence was one line — `collapse_panels=True` — and that
line bought less than it was described as buying. Look at the top of any of the three:
the Rerun wordmark and its menu, a Share button, a notification bell, three panel toggles
and a fullscreen toggle. Look at the bottom: a full transport bar, with play, pause, step,
loop, a speed control and a scrubber. Every one of those belongs to Rerun's application,
and the earlier version of this file claimed they were gone. They were not; only the
blueprint and selection panels were.

The fence is four named panels now — blueprint, selection, top and time, each `Hidden`,
each asserted in `tests/test_live_view.py`. What that actually looks like is a different
set of pictures, taken on the same rig on 2026-08-06 through a page that embeds this view
in an iframe, and kept beside that page rather than here. In those, the wordmark, the
Share button, the toggles and the transport bar are gone, and what is left of Rerun is the
help, visibility and expand icons in each view's own title bar.

These three stay because they are the only frames of the view drawing a *real robot* from
the kit's URDF, which the passive bench source in the newer run does not declare. Read
them for the cameras, the body and the recording badge. Do not read them for the chrome.

## The run these came from

The shell wrapper that drove it was never committed and is gone, so this is not a
pointer to a file — it is the command line, read back off the run's own receipts. Each
flag below is pinned by something in this directory:

| flag | what pins it |
| --- | --- |
| `--task "live view bench proof - nothing is being driven"` | the page header, legible in all three screenshots |
| `--source recording_source:cameras_only` | the footer: `Source: LIVE CAMERAS (3) + SIMULATED JOINTS` |
| `--dest /tmp/view-episodes` | the `stopped` event's `path` below |
| `--json` | the event lines below are `--json`'s own records, verbatim |
| `--view` (no `--view-port`) | the shot was taken against `localhost:9099`, which is the default |

```
newt record \
    --task "live view bench proof - nothing is being driven" \
    --source recording_source:cameras_only \
    --dest /tmp/view-episodes \
    --view --json
```

`--json` reads commands from stdin, so the wrapper fed it `{"cmd": "start"}`, waited
while the screenshotter worked, then `{"cmd": "stop", "keep": true}` and
`{"cmd": "close"}`. Anything the receipts do not pin — `--hz`, `--author`, `--license`,
`--target` — is left out rather than guessed at; their absence here is not a claim that
they were unset.

Run today, that command's stdout differs from the log this run produced in one way, and
it is worth knowing which: `--view` now announces itself as a `{"event": "view", …}`
record. On the run above it printed three bare `[view] …` lines onto the same stdout,
ahead of the first JSON record — which is exactly the NDJSON contract this branch went
back and fixed.

Session events, verbatim from the run log:

```
{"event": "started", "episode_id": "5f74a3f6"}
{"event": "stopped", "kept": true, "path": "/tmp/view-episodes/episode_5f74a3f6", "state_count": 4170, "dropped_state": 0, "frame_counts": {"d405-230422271405": 2153, "d455f-419222301574": 2153, "d455f-352222302988": 2153}, "dropped_frames": {}, "kept_total": 1}
```

The episode written underneath the picture, checked afterward:

```
$ newt episodes validate /tmp/view-episodes/episode_5f74a3f6
[PASS] /tmp/view-episodes/episode_5f74a3f6
  ok  episode_json_present: present and parses
  ok  format_version: format_version is '0.0.3'
  ok  data_mcap_present: present
  ok  data_mcap_readable: readable
  ok  robot_state_channel: robot_state/widowx_250/leader with 4170 messages
  ok  timestamps_monotonic: state timestamps non-decreasing
  ok  frame_count[d405-230422271405]: 2153 MCAP markers vs 2153 video frames
  ok  frame_count[d455f-419222301574]: 2153 MCAP markers vs 2153 video frames
  ok  frame_count[d455f-352222302988]: 2153 MCAP markers vs 2153 video frames
exit=0
```

Browser output over the same window: 16,633 console messages, 0 errors, 0 page errors.
All five warnings are the software rasterizer announcing itself (`No available adapters`,
`Software rasterizer detected`) plus two egui viewport commands the web backend does not
implement. Headless Chromium has no GPU, which is why the render is slow and why
`--settle` is long.

## No arm was driven or energized

`cameras_only` constructs no leader, no follower, and no driver handle. The factory that
does reach the arms is deliberately not called: lerobot's `connect()` sets position mode
and commands the home pose, which moves them. The joint traces in the plot are simulated,
and the page says so in its own footer, in the same sentence as the camera count.

## Reproducing

The source is committed now — `scripts/bench/passive_source.py`, which opens a rig's
cameras and nothing else, so a run of this kind no longer has to be read back off its own
output the way the command line above was:

```
PYTHONPATH=scripts/bench NT_BENCH_CAMERAS=4,10,14 \
    newt record --source passive_source:cameras_only \
    --task "..." --dest ~/episodes --view --json
```

Then, on the machine serving the page:

```
uv run --with playwright python scripts/shoot_view.py \
    --url http://localhost:9099/ --out /tmp/view-shots --shots 3 --settle 40 --interval 4
```

Run it on the machine serving the view. `http://localhost` is a secure context, which is
what lets the page decode camera video. The same page fetched over a bare LAN address is
not, so a shot taken from another machine measures the browser's rules rather than the
rig's output.

The full `shots.json`, every console message verbatim, stays on the rig at
`/tmp/view-shots/shots.json`. It is 3.8 MB and almost entirely wasm chatter, so it is not
committed here; the counts above are what it says.
