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

What is *not* in the frame is the point of the layout fence: no blueprint panel, no
selection panel, a simplified scrubber. This is the hackathon's lean page built on Rerun's
embedded viewer, and it is not Rerun's application.

## The run these came from

`bench_proof.sh` on the rig, `--source recording_source:cameras_only`. Session events,
verbatim from the run log:

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
