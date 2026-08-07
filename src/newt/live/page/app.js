// The page's whole behaviour: mount the viewer against this rig's own stream and
// say plainly what the session is doing.
//
// Adapted from web/static/app.js in
// https://github.com/mission-robotics-ai/so100-hackathon (a fork of
// rerun-io/so100-hackathon), Copyright Rerun Technologies AB, dual-licensed
// MIT OR Apache-2.0. The focus patch, the VIEWER_OPTIONS literal and its
// width/height warning, and the same-origin `/viewer/index.js` import are theirs.
// Both licence texts ship with this package — see NOTICE and licenses/ at the
// repository root, and newt/NOTICE inside an installed wheel.

// The viewer (eframe/egui underneath) grabs keyboard focus in ways that scroll the
// page. Its "text agent" is a hidden 1px <input> parked at the top of the body to
// capture IME input; eframe marks it autofocus and calls .focus() on it whenever
// the canvas is touched, and focusing an element at the top of a page scrolls the
// window there. The canvas gets the same treatment on boot and on fullscreen exit.
// Nothing on this page calls .focus() itself, so patch globally, once, at load:
//   1. every programmatic .focus() gets preventScroll (Tab-key navigation is
//      browser-internal, does not go through this method, and keeps its scrolling);
//   2. autofocus is neutered outright — the browser honours it on insertion through
//      a path .focus() patching cannot reach.
{
  const original = HTMLElement.prototype.focus;
  HTMLElement.prototype.focus = function (opts) {
    original.call(this, { ...opts, preventScroll: true });
  };
  Object.defineProperty(HTMLElement.prototype, "autofocus", {
    get() { return false; },
    set() {},
  });
}

// width/height MUST stay "" — the canvas is sized by CSS (.viewer-box canvas).
// "100%" would set inline styles, and the viewer's fullscreen exit wipes those with
// removeAttribute("style"), leaving an unsized canvas and a wasm panic.
//
// The panels are not listed here. The blueprint and selection panels are collapsed
// from the Python side (collapse_panels=True), which is the supported route; the
// panel_state_overrides that would do it from here is marked @private in the
// viewer's own types.
const VIEWER_OPTIONS = {
  width: "",
  height: "",
  hide_welcome_screen: true,
  allow_fullscreen: true,
};

const box = document.getElementById("viewer");
const overlay = document.getElementById("overlay");
const overlayNote = document.getElementById("overlay-note");
const stateEl = document.getElementById("state");
const taskEl = document.getElementById("task");
const noteEl = document.getElementById("note");

function refuse(title, detail) {
  overlay.classList.add("bad");
  overlay.querySelector(".overlay-title").textContent = title;
  overlayNote.textContent = detail;
  stateEl.textContent = "not connected";
  stateEl.className = "state bad";
  console.error("newt live view: " + title + " — " + detail);
}

let session = null;

async function readSession() {
  const response = await fetch("/session.json", { cache: "no-store" });
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}

// The status line, and the only thing on this page that polls. It reads the same
// session the stream comes from, so "recording" appears here because an episode is
// open — not because the page guessed from the picture.
async function pollStatus() {
  try {
    const now = await readSession();
    taskEl.textContent = now.task;
    if (now.recording) {
      stateEl.textContent = "recording · " + now.episode_id;
      stateEl.className = "state rec";
    } else {
      stateEl.textContent = "live";
      stateEl.className = "state live";
    }
    noteEl.textContent = now.note;
  } catch (err) {
    stateEl.textContent = "session ended";
    stateEl.className = "state bad";
    noteEl.textContent =
      "This page can no longer reach the session that serves it (" + err +
      "). The rig is not being asked for anything; the command that started the " +
      "session has stopped or the machine went away.";
  }
  setTimeout(pollStatus, 1000);
}

try {
  session = await readSession();
} catch (err) {
  refuse(
    "this page cannot reach its own session",
    "The page loaded but /session.json did not answer (" + err + "). Ours: the " +
    "server that handed you this page is the one that failed to answer. Restart " +
    "the session command on the rig."
  );
}

if (session) {
  taskEl.textContent = session.task;
  noteEl.textContent = session.note;
  pollStatus();
  try {
    const { WebViewer } = await import("/viewer/index.js");
    const viewer = new WebViewer();
    viewer.on("recording_open", () => overlay.remove());
    // The session names its stream with a {host} placeholder rather than an
    // address: the same page is opened from the rig's own browser and from a
    // laptop beside it, and a URL naming 127.0.0.1 is true for only one of them.
    const stream = session.stream.replace("{host}", location.hostname);
    await viewer.start(stream, box, VIEWER_OPTIONS);
    // The viewer emits no failure event of any kind — its whole event list is
    // play/pause/time_update/timeline_change/selection_change/recording_open — so
    // the absence of recording_open is the only signal an embedding page gets.
    // Say what that absence means rather than leaving a black rectangle.
    setTimeout(() => {
      if (!document.getElementById("overlay")) return;
      overlayNote.textContent =
        "The viewer started and " + session.stream.replace("{host}", location.hostname) +
        " has not delivered a " +
        "recording yet. The stream is on this same rig, so this is the session " +
        "not publishing rather than a network between you and it.";
    }, 20000);
  } catch (err) {
    refuse(
      "the embedded viewer would not start",
      "Nothing was asked of the session: " + err + ". This browser or the cached " +
      "viewer build, not the rig. Check that this browser supports WebGPU or " +
      "WebGL2, then reload."
    );
  }
}
