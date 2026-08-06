"""newt teleop — the keyboard frontend on a teleoperation session.

Like ``newt record``, this module is a skin. It parses arguments, refuses to run
where there is no kill key, arms Ctrl+H, loads the developer's source, and
renders what the session returns. The tick loop, the pass-through, and the exit
choreography live in the library — if you find them creeping in here, they are
in the wrong file (the layering invariant ``record`` set).

What the verb owns and what the rig owns:
- the verb decides *when* and *in what order* — the tick, the kill, which exit
  the session is on, and the refusal to start without a keyboard;
- the rig decides *what* — which hardware, how it is reached, what an action
  is, what "put yourself away" means for it. All of that arrives through
  ``--source MODULE:FACTORY`` and none of it is known here.

The kill key. Ctrl+H de-energizes and exits 130. It does not move anything
first: motion after the panic key is the thing the panic key exists to prevent,
and it is the same meaning Ctrl+H has under ``newt record``. Ctrl+C takes the
other exit — the embodiment is put away by its own declared rest, then
de-energized, and the process exits 0. Every exit path ends de-energized. What
differs is whether it was put somewhere first.

Exit codes:
  0    the session ended on Ctrl+C and everything confirmed it was de-energized
  1    a usage error, or the source refused to come up, or a halt went
       unconfirmed
  2    stood down before anything connected — no TTY, no kill key, or a rate
       that is not a rate
  130  Ctrl+H fired, or Ctrl+C landed during bring-up
"""
from __future__ import annotations

import select
import sys
import termios
import threading
import tty

from newt._cli._source_spec import SourceNotResolved, load_source, resolve_spec

# Ctrl+H. The same byte `newt record` reads for the same meaning.
_CTRL_H = "\x08"

_DEFAULT_RATE_HZ = 30.0


def _usage() -> None:
    print("Usage: newt teleop [options]")
    print("")
    print("  Drive one embodiment from another's motion. Every tick the source")
    print("  produces an action and the sink is sent it unchanged — no scaling,")
    print("  no filtering, no interpolation. Ctrl+H kills (de-energize, no move")
    print("  first); Ctrl+C ends the session (put away, then de-energize).")
    print("")
    print("Options:")
    print("  --source SPEC   Which TeleopSource to run — either a short name your")
    print("                  kit declares, or a full MODULE:FACTORY import path")
    print("                  (e.g. mypkg.rig:make_teleop), which needs no")
    print("                  declaration. Optional on a configured rig: with no")
    print("                  flag the verb reads [sources].teleop from your site")
    print("                  config ($NT_SITE_CONFIG, else ~/.config/nt/nt.toml),")
    print("                  else the one your kit declares. The flag always wins.")
    print("  --rate HZ       Ticks per second (default: 30). The source is built")
    print("                  at the rate it is driven at — one decision, not two.")
    print("")
    print("  Needs a real terminal: with no keyboard there is no Ctrl+H, and this")
    print("  verb moves hardware. Nothing connects until the kill key is armed.")


def _rate(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"--rate {value!r} is not a number — pass Hz, e.g. --rate 30"
        ) from None


def _parse(args: list[str]) -> dict:
    opts = {
        "source": None,
        "rate": _DEFAULT_RATE_HZ,
    }
    # option -> (key, converter)
    valued = {
        "--source": ("source", str),
        "--rate": ("rate", _rate),
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a in valued:
            key, conv = valued[a]
            i += 1
            if i >= len(args):
                raise ValueError(f"{a} expects a value")
            opts[key] = conv(args[i])
        else:
            raise ValueError(f"unknown option {a!r}")
        i += 1
    return opts


# --------------------------------------------------------------------------- #
# The kill key — arming, listening, restoring
# --------------------------------------------------------------------------- #

class KillKey:
    """Ctrl+H on a background thread, exposed as an event the loop can wait on.

    A thread rather than a per-tick poll because a tick spends most of its time
    inside a driver call or asleep. The loop sleeps on ``wait()``, so a Ctrl+H
    pressed mid-sleep wakes it immediately instead of at the next tick boundary.
    """

    def __init__(self) -> None:
        self.fired = threading.Event()
        self._saved_tty = None
        self._thread: threading.Thread | None = None

    def arm(self) -> bool:
        """Put the terminal in cbreak and start listening. False if it could not.

        Returns False rather than raising: the caller decides what an unarmable
        kill key means, and for this verb it means standing down before
        anything connects."""
        if not sys.stdin.isatty():
            return False
        try:
            self._saved_tty = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError) as exc:
            print(f"[newt teleop] could not set cbreak on this terminal: {exc}", file=sys.stderr)
            self._saved_tty = None
            return False
        self._thread = threading.Thread(
            target=self._listen, name="newt-teleop-kill-key", daemon=True
        )
        self._thread.start()
        return True

    def _listen(self) -> None:
        while not self.fired.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            except (ValueError, OSError):
                return  # stdin closed under us; the session's own exit takes over
            if not ready:
                continue
            try:
                key = sys.stdin.read(1)
            except (ValueError, OSError):
                return
            if key == _CTRL_H:
                print(
                    "\r\n[newt teleop] EMERGENCY STOP — Ctrl+H pressed; de-energizing now "
                    "(no move; it will settle under gravity)…",
                    flush=True,
                )
                self.fired.set()
                return

    def restore(self) -> None:
        """Idempotent. Safe to call from a finally that may run twice."""
        saved, self._saved_tty = self._saved_tty, None
        if saved is None:
            return
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        except (termios.error, ValueError, OSError):
            pass


# --------------------------------------------------------------------------- #
# The two stand-downs — both exit 2, both before anything connects
# --------------------------------------------------------------------------- #

def _stand_down_no_tty() -> int:
    print(
        "[newt teleop] stdin is not a TTY — there is no keyboard, so there is no Ctrl+H, "
        "and this loop moves hardware.",
        file=sys.stderr,
    )
    print(
        "        Fix: run newt teleop in a real terminal. To drive it from a script\n"
        "        anyway, bound the session before it starts:\n"
        "          timeout --signal=INT 30 newt teleop --source mypkg.rig:make_teleop\n"
        "        SIGINT lands where Ctrl+C lands: the embodiment is put away first, then\n"
        "        de-energized. It is an exit, not a kill — it moves.",
        file=sys.stderr,
    )
    return 2


def _stand_down_unarmed() -> int:
    print(
        "[newt teleop] stdin is a terminal but the Ctrl+H listener could not arm on it "
        "(the reason is printed above), so this session would move hardware with no kill key.",
        file=sys.stderr,
    )
    print(
        "        Fix: run newt teleop from an ordinary interactive shell — not under a\n"
        "        wrapper that owns the terminal (tmux send-keys, an editor's run pane, a\n"
        "        CI runner). Nothing has been connected and nothing has moved.",
        file=sys.stderr,
    )
    return 2


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

def cmd_teleop(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    try:
        opts = _parse(args)
    except (ValueError, IndexError) as exc:
        print(f"newt teleop: {exc}", file=sys.stderr)
        print("Run 'newt teleop --help' for usage.", file=sys.stderr)
        return 1

    # Resolved here, where the old refusal was: before the rate check and before
    # the kill key, so a rig that declares nothing is turned away with nothing
    # armed and nothing connected.
    try:
        spec = resolve_spec("teleop", opts["source"], "mypkg.rig:make_teleop")
    except SourceNotResolved as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if opts["rate"] <= 0:
        print(
            f"[newt teleop] --rate {opts['rate']:g} is not a rate; it is how many times per "
            "second the sink is told where the source is.",
            file=sys.stderr,
        )
        print(
            "        Fix: pass a positive number of Hz, e.g. --rate 30.",
            file=sys.stderr,
        )
        return 2

    # The kill key comes before the source, not after: the factory is what
    # connects and energizes, and a session that cannot be killed must never
    # reach it. Both stand-downs below leave nothing connected.
    if not sys.stdin.isatty():
        return _stand_down_no_tty()

    kill_key = KillKey()
    if not kill_key.arm():
        return _stand_down_unarmed()

    try:
        try:
            source = load_source(spec)
        except KeyboardInterrupt:
            # Bring-up is where the factory connects, and on real hardware that
            # is seconds of blocking vendor motion — long enough for an operator
            # to change their mind. 130, unlike the Ctrl+C that ends a live
            # session (0): that one is the documented way to finish, this one is
            # an abort before the session started. Putting away whatever came up
            # is the factory's job; it is the only thing holding those handles.
            print(
                "\n[newt teleop] bring-up interrupted (Ctrl+C) — the session never "
                "started. Check what the source reported about anything it had already "
                "connected.",
                file=sys.stderr,
            )
            return 130
        except Exception as exc:
            # The factory's own refusal — a missing address, a missing driver.
            # Surface it, don't trace it.
            print(f"[newt teleop] {exc}", file=sys.stderr)
            return 1

        from newt.teleop import run_session

        return run_session(source, rate_hz=opts["rate"], kill=kill_key.fired)
    finally:
        kill_key.restore()
