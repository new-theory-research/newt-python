"""The arms lease — who owns the hardware while a session runs.

The arms are single-client. Two doors reach for them: a CLI verb, and the
collect page the rig serves. This module is the only thing either door asks,
and the answer is a file they both read.

**Why a written artifact and not a lock.** Two systems that execute
independently can share exactly one thing: state they can both read. An
in-process mutex is invisible to the other surface by construction — the CLI
could only learn who holds the arms by asking the page, which is entanglement
wearing a lock's clothes. So the lease is a file, and both doors write it
before they touch a motor.

**Three states, and they never share a string.**

``free``
    Nothing on disk. Nobody has the arms. This is also what a clean exit
    leaves behind, which is why release removes the file rather than marking
    it: "it told us it was leaving" and "we stopped hearing from it" are
    different states and must not share a string.

``held``
    A lease whose holder is still checking in. The refusal names which surface
    holds them, since when, and the one move that ends it — and that move is on
    the *other* surface, because this process cannot make another one stop
    moving an arm.

``expired``
    A lease whose holder stopped checking in. Distinct from ``free`` on
    purpose: taking it over is something the user does, never something that
    happens to them. Three sub-reasons, three strings — a process we can prove
    is gone, a process still running but silent, and silence we cannot check.

A file that will not parse is a fourth condition and gets its own refusal. It
must never read as ``free``: that is the silent steal this module exists to
prevent.

**The stamped fields, and the three this build added.** ``holder``,
``surface``, ``since`` and ``heartbeat`` are the mechanism as stamped. Three
more are recorded alongside them, declared here rather than slipped in:
``command`` (Rule 12 — a refusal owes the next step, and the next step is
phrased in what the holder is running), ``machine`` and ``pid`` (a killed
holder and a silent one are different problems, and a pid is the only thing
that can tell them apart — see ``REASON_PROCESS_GONE``).
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

LEASE_DIR = Path.home() / ".nt"
LEASE_PATH = LEASE_DIR / "arms.lease"
LOCK_PATH = LEASE_DIR / "arms.lease.lock"

# Which door a holder came through.
SURFACE_PAGE = "page"
SURFACE_CLI = "cli"

# The three states.
FREE = "free"
HELD = "held"
EXPIRED = "expired"

# Why an expired lease is expired. Three causes, three strings (Rule 12).
REASON_PROCESS_GONE = "process-gone"
REASON_HOLDER_HUNG = "holder-hung"
REASON_SILENT = "silent"

HEARTBEAT_EVERY_SECONDS = 10.0
# Three missed beats. Long enough that a rig swapping to a busy USB bus does not
# lose its own arms; short enough that a killed session does not lock a bench for
# a coffee break.
STALE_AFTER_SECONDS = 30.0


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Lease:
    """One holder's claim on the arms, as it sits on disk."""

    holder: str       # identifier for the session that took the arms
    surface: str      # SURFACE_PAGE or SURFACE_CLI — which door it came through
    since: str        # ISO-8601 with offset; when it was taken
    heartbeat: str    # ISO-8601 with offset; last time the holder said it was there
    command: str      # what the holder is running, in the words the user typed
    machine: str      # the host that wrote it; a pid only means something on its own machine
    pid: int | None   # the holder's process, when it has one

    def since_time(self) -> str:
        """The time of day a human reads back — "20:31"."""
        return _parse(self.since).strftime("%H:%M")

    def heartbeat_time(self) -> str:
        return _parse(self.heartbeat).strftime("%H:%M")


class LeaseError(Exception):
    """Base for every refusal this module raises. Carries a printable message."""


class ArmsHeld(LeaseError):
    """Somebody is driving. ``lease`` names who, and the message names the move."""

    def __init__(self, message: str, lease: Lease) -> None:
        super().__init__(message)
        self.lease = lease


class ArmsLeaseExpired(LeaseError):
    """A lease with nobody home. Takeable — by the user, explicitly, never silently."""

    def __init__(self, message: str, lease: Lease, reason: str) -> None:
        super().__init__(message)
        self.lease = lease
        self.reason = reason


class LeaseUnreadable(LeaseError):
    """The file is there and will not parse. Not free — unknown, which is worse."""


class LeaseMoved(LeaseError):
    """A take-over lost its race: the lease changed between being shown and being taken."""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def _now() -> datetime:
    return datetime.now().astimezone()


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def new_holder_id(surface: str) -> str:
    """An identifier for one session.

    The pid alone is not enough: pids are recycled, and a recycled pid would let
    a dead session's lease be mistaken for a live one's.
    """
    return f"{surface}-{os.getpid()}-{secrets.token_hex(3)}"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, which means it exists.
        return True
    except OSError as exc:  # pragma: no cover — platform oddity, not a real branch
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def read() -> Lease | None:
    """The lease as it sits on disk, or None if there is none.

    Raises ``LeaseUnreadable`` for a file that exists and will not parse. A
    caller that treats that as None has stolen the arms from whoever wrote it.
    """
    try:
        raw = LEASE_PATH.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LeaseUnreadable(
            f"newt: the arms lease at {LEASE_PATH} could not be read ({exc.strerror}).\n"
            "\n"
            f"        Fix the file's permissions, or remove it: rm {LEASE_PATH}\n"
            "\n"
            "Until it reads, nothing here can tell whether somebody is driving this "
            "rig, and this command will not guess."
        ) from exc

    try:
        data = json.loads(raw)
        return Lease(
            holder=str(data["holder"]),
            surface=str(data["surface"]),
            since=str(data["since"]),
            heartbeat=str(data["heartbeat"]),
            command=str(data["command"]),
            machine=str(data["machine"]),
            pid=None if data.get("pid") is None else int(data["pid"]),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise LeaseUnreadable(
            f"newt: the arms lease at {LEASE_PATH} is there but does not parse ({exc}).\n"
            "\n"
            f"        If no session is running on this rig, remove it: rm {LEASE_PATH}\n"
            "\n"
            "A lease that will not parse is not the same as no lease. This command "
            "refuses rather than assume the arms are free, because assuming that is "
            "how two sessions end up driving one arm."
        ) from exc


def state(lease: Lease | None, *, now: datetime | None = None) -> str:
    """FREE, HELD or EXPIRED. The one place the three states are decided."""
    if lease is None:
        return FREE
    return HELD if _reason(lease, now=now) is None else EXPIRED


def _reason(lease: Lease, *, now: datetime | None = None) -> str | None:
    """Why this lease is expired, or None if it is still held."""
    moment = now or _now()
    same_machine = lease.machine == socket.gethostname()

    if same_machine and lease.pid is not None and not _process_alive(lease.pid):
        # We can prove nobody is home. This does not wait for the heartbeat: a
        # killed session should not lock a bench for thirty seconds.
        return REASON_PROCESS_GONE

    if _parse(lease.heartbeat) + timedelta(seconds=STALE_AFTER_SECONDS) >= moment:
        return None

    if same_machine and lease.pid is not None:
        # Still running, and it stopped saying so. Taking the arms will probably
        # fail on a serial port this process still has open, so the move differs.
        return REASON_HOLDER_HUNG
    return REASON_SILENT


# --------------------------------------------------------------------------- #
# The strings — the refusals both surfaces render from
# --------------------------------------------------------------------------- #

def headline(lease: Lease, *, now: datetime | None = None) -> str:
    """One sentence: who holds the arms, through which door, since when.

    Both surfaces render from here rather than each writing its own. Symmetry is
    the point — a developer must never have to guess which surface is winning,
    and two hand-written sentences drift apart the week after they are written.
    """
    reason = _reason(lease, now=now)

    if reason is None:
        if lease.surface == SURFACE_PAGE:
            return (
                f"The collect page holds the arms — taken at {lease.since_time()} "
                f"on {lease.machine}, still checking in."
            )
        return (
            f"A `{lease.command}` session holds the arms — taken at "
            f"{lease.since_time()} on {lease.machine}, still checking in."
        )

    if reason == REASON_PROCESS_GONE:
        return (
            f"The arms are still leased to `{lease.command}` from "
            f"{_door(lease)}, taken at {lease.since_time()} — and that process "
            f"(pid {lease.pid}) is gone. Nobody is home."
        )
    if reason == REASON_HOLDER_HUNG:
        return (
            f"The arms are still leased to `{lease.command}` from "
            f"{_door(lease)}, taken at {lease.since_time()} — its process "
            f"(pid {lease.pid}) is still running but stopped checking in at "
            f"{lease.heartbeat_time()}."
        )
    return (
        f"The arms are still leased to `{lease.command}` from {_door(lease)} on "
        f"{lease.machine}, taken at {lease.since_time()} — and nothing has "
        f"checked in since {lease.heartbeat_time()}."
    )


def _door(lease: Lease) -> str:
    return "the collect page" if lease.surface == SURFACE_PAGE else "a terminal"


def next_move(lease: Lease, *, verb: str, now: datetime | None = None) -> str:
    """The one move that ends this situation, for a CLI caller running ``verb``.

    A live holder is not offered ``--take``. This process cannot make another one
    stop moving an arm, so the move is on the surface that can — which is also
    what makes ``held`` and ``expired`` different states rather than two spellings
    of the same one.
    """
    reason = _reason(lease, now=now)

    if reason is None:
        if lease.surface == SURFACE_PAGE:
            return "Release them in the cockpit, or stop the `newt collect` serving it."
        return f"Finish or Ctrl-C the `{lease.command}` session, then run this again."

    if reason == REASON_HOLDER_HUNG:
        return (
            f"Stop the hung process first — kill {lease.pid} — then: newt {verb} --take"
        )
    return f"newt {verb} --take"


def refusal(lease: Lease, *, verb: str, now: datetime | None = None) -> str:
    """The full lantern a CLI verb prints when it cannot have the arms."""
    reason = _reason(lease, now=now)
    move = next_move(lease, verb=verb, now=now)

    if reason is None:
        tail = (
            "Nothing here reached for a motor. The arms are single-client, and the "
            "lease is how the two doors avoid grabbing at once."
        )
    elif reason == REASON_PROCESS_GONE:
        tail = (
            "Taking them over is yours to do rather than something that happens to "
            "you, which is why this is a refusal and not a shrug."
        )
    elif reason == REASON_HOLDER_HUNG:
        tail = (
            "Taking the arms while that process is up will fail on a port it still "
            "holds, so it goes first."
        )
    else:
        tail = (
            "Silence is not the same as free. If that session really is gone, take "
            "the arms deliberately."
        )

    return (
        f"newt {verb}: {headline(lease, now=now)}\n"
        "\n"
        f"        {move}\n"
        "\n"
        f"{tail}"
    )


def describe(lease: Lease | None, *, verb: str = "collect", now: datetime | None = None) -> dict:
    """The same truth, shaped for a page rather than a terminal.

    The page renders these strings; it does not write its own. Behaviour 3 of the
    ruling is "the page finding a CLI lease says who holds them, **in the same
    terms**" — the only way to hold that over time is one source for the words.
    """
    if lease is None:
        return {"state": FREE, "headline": "The arms are free.", "next_move": None}

    return {
        "state": state(lease, now=now),
        "reason": _reason(lease, now=now),
        "holder": lease.holder,
        "surface": lease.surface,
        "since": lease.since,
        "since_time": lease.since_time(),
        "heartbeat": lease.heartbeat,
        "command": lease.command,
        "machine": lease.machine,
        "headline": headline(lease, now=now),
        "next_move": next_move(lease, verb=verb, now=now),
    }


# --------------------------------------------------------------------------- #
# Writing — every transition goes through the same lock
# --------------------------------------------------------------------------- #

@contextmanager
def _transaction() -> Iterator[None]:
    """Serialize read-modify-write across processes on this machine.

    ``flock`` and not a lock *file* we create and delete: a lock the OS releases
    when the process dies cannot be left behind by a kill -9, and a lease with a
    race in it is worse than no lease because it is trusted.

    The lock is only for the transition. The lease file stays the readable truth,
    so the other surface never has to ask this process anything.
    """
    LEASE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _write(lease: Lease) -> None:
    """Replace the lease atomically. Readers see the old one or the new one."""
    tmp = LEASE_PATH.parent / f"{LEASE_PATH.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(asdict(lease), indent=2) + "\n")
    os.replace(tmp, LEASE_PATH)


def acquire(*, surface: str, command: str, now: datetime | None = None) -> Lease:
    """Take the arms, or refuse and say who has them.

    Taking the arms means writing the lease first — this is the only door to the
    hardware either surface has, and it never silently takes over an expired one.
    """
    moment = now or _now()
    with _transaction():
        existing = read()
        if existing is not None:
            reason = _reason(existing, now=moment)
            if reason is None:
                raise ArmsHeld(headline(existing, now=moment), existing)
            raise ArmsLeaseExpired(headline(existing, now=moment), existing, reason)

        stamp = _stamp(moment)
        lease = Lease(
            holder=new_holder_id(surface),
            surface=surface,
            since=stamp,
            heartbeat=stamp,
            command=command,
            machine=socket.gethostname(),
            pid=os.getpid(),
        )
        _write(lease)
        return lease


def take_over(
    expected: Lease, *, surface: str, command: str, now: datetime | None = None
) -> Lease:
    """Take an expired lease, having shown the user the expired state first.

    ``expected`` is the lease the user was shown. If the file moved underneath —
    the old holder woke up, or somebody else took it in the same second — this
    refuses rather than overwrite, because the thing the user agreed to take is
    no longer the thing on disk.
    """
    moment = now or _now()
    with _transaction():
        current = read()
        if current is None:
            # Nobody home and nothing on disk: the takeover is just an acquire.
            pass
        elif current.holder != expected.holder:
            raise LeaseMoved(
                f"newt: the arms lease changed while you were reading it — it now "
                f"belongs to {current.holder}, not {expected.holder}.\n"
                "\n"
                "        Run this command again to see who has them now.\n"
                "\n"
                "Nothing was taken. A take-over applies to the session you were "
                "shown, not to whichever one happens to be there when it lands."
            )
        elif _reason(current, now=moment) is None:
            raise ArmsHeld(headline(current, now=moment), current)

        stamp = _stamp(moment)
        lease = Lease(
            holder=new_holder_id(surface),
            surface=surface,
            since=stamp,
            heartbeat=stamp,
            command=command,
            machine=socket.gethostname(),
            pid=os.getpid(),
        )
        _write(lease)
        return lease


def beat(lease: Lease, *, now: datetime | None = None) -> Lease:
    """Say the holder is still there. Returns the lease with its new heartbeat.

    A holder whose lease was taken from underneath it is told so rather than
    quietly re-writing itself back in.
    """
    moment = now or _now()
    with _transaction():
        current = read()
        if current is None or current.holder != lease.holder:
            raise LeaseMoved(
                "newt: this session no longer holds the arms — the lease was taken "
                "over while it was running.\n"
                "\n"
                "        Stop this session; something else is driving.\n"
                "\n"
                "Two claimants on one arm is the thing the lease exists to prevent, "
                "and continuing to drive after losing it would be exactly that."
            )
        beaten = replace(current, heartbeat=_stamp(moment))
        _write(beaten)
        return beaten


def release(lease: Lease) -> None:
    """Give the arms back on a clean exit.

    Removing the file rather than marking it released is deliberate: it makes
    "it told us it was leaving" indistinguishable from "it was never here" *on
    the rig* — which is correct, because both mean the arms are free — while the
    console's presence row keeps those two apart, which is where the difference
    actually matters to a reader.
    """
    with _transaction():
        current = read_quietly()
        if current is not None and current.holder != lease.holder:
            # Somebody else's lease. Deleting it would be the silent steal.
            return
        try:
            LEASE_PATH.unlink()
        except FileNotFoundError:
            pass


def read_quietly() -> Lease | None:
    """``read`` for the release path, where an unparseable file is not ours to judge."""
    try:
        return read()
    except LeaseUnreadable:
        return None


def claim(*, verb: str, command: str, take: bool, surface: str = SURFACE_CLI) -> Lease:
    """Take the arms for one verb, or raise a ``LeaseError`` that is ready to print.

    Every verb that touches hardware wants the same four lines at the same point
    in its flow, and the shape it wants is "hand me the arms or hand me the
    sentence to print". Callers do::

        try:
            arms = _lease.claim(verb="teleop", command="newt teleop", take=opts["take"])
        except _lease.LeaseError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    The refusal is rendered here rather than at each call site so that four verbs
    cannot drift into four dialects of the same sentence.
    """
    try:
        return acquire(surface=surface, command=command)
    except ArmsHeld as exc:
        raise ArmsHeld(refusal(exc.lease, verb=verb), exc.lease) from None
    except ArmsLeaseExpired as exc:
        if not take:
            raise ArmsLeaseExpired(
                refusal(exc.lease, verb=verb), exc.lease, exc.reason
            ) from None
        return take_over(exc.lease, surface=surface, command=command)


@contextmanager
def held(*, surface: str, command: str, take: bool = False) -> Iterator[Lease]:
    """Hold the arms for the length of a session, and give them back after.

    ``take`` is the user having said so — it is never inferred. Without it an
    expired lease refuses exactly like a held one, differing only in the move it
    hands over.
    """
    try:
        lease = acquire(surface=surface, command=command)
    except ArmsLeaseExpired as expired:
        if not take:
            raise
        lease = take_over(expired.lease, surface=surface, command=command)

    try:
        yield lease
    finally:
        release(lease)
