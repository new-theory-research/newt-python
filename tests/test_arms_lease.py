"""The arms lease — the mechanism that stops two doors grabbing one arm.

These tests encode *why* the lease exists, not just that it writes a file. The
load-bearing ones:

* the three states never share a string, because a developer walking up to a busy
  rig has to be able to tell "somebody is driving" from "somebody died holding
  the arms" from "they're free" — and one shared sentence collapses all three;
* nothing takes an expired lease without the user saying so, because the silent
  steal is the failure that makes a lease worse than no lease;
* two claimants racing really do produce one winner, because a lease with a race
  in it is trusted and wrong, which is worse than untrusted and wrong.
"""
from __future__ import annotations

import json
import socket
import threading
from datetime import datetime, timedelta

import pytest

from newt._cli import _lease
from newt._cli._lease import (
    EXPIRED,
    FREE,
    HELD,
    REASON_HOLDER_HUNG,
    REASON_PROCESS_GONE,
    REASON_SILENT,
    SURFACE_CLI,
    SURFACE_PAGE,
    ArmsHeld,
    ArmsLeaseExpired,
    Lease,
    LeaseMoved,
    LeaseUnreadable,
)

# Fixed stamps, and fixed in the *past* on purpose. Some paths under test take an
# injected ``now`` and some (``held``, ``acquire`` inside it) read the wall clock,
# so a fixture that is only stale relative to ``NOW`` would read as live to the
# second kind. Anchoring behind both is what keeps one helper honest for both.
NOW = datetime.fromisoformat("2026-08-05T20:44:00+00:00")
TAKEN_AT = datetime.fromisoformat("2026-08-05T20:31:00+00:00")


def _now_stamp(delta_seconds: float = 0.0) -> str:
    return (datetime.now().astimezone() + timedelta(seconds=delta_seconds)).isoformat(
        timespec="seconds"
    )


def _make(**over) -> Lease:
    """An *expired* lease on disk — stale against the wall clock and against NOW.

    ``machine`` defaulting to something that is not this host is deliberate: it
    isolates the heartbeat rule from the process-liveness rule, so a test that
    means to exercise one is never quietly answered by the other.
    """
    fields = dict(
        holder="cli-4821-a3f9c1",
        surface=SURFACE_CLI,
        since=TAKEN_AT.isoformat(timespec="seconds"),
        heartbeat=TAKEN_AT.isoformat(timespec="seconds"),
        command="newt teleop",
        machine="some-other-rig",
        pid=None,
    )
    fields.update(over)
    return Lease(**fields)


def _fresh(**over) -> Lease:
    """A *held* lease — taken at 20:31 and still checking in as of right now.

    The heartbeat is stamped off the wall clock so this reads as held to the
    paths that do not take an injected ``now``; ``since`` stays at TAKEN_AT so
    the time-of-day a reader gets back is a stable "20:31".
    """
    over.setdefault("heartbeat", _now_stamp(-1))
    return _make(**over)


def _put(lease: Lease) -> None:
    _lease.LEASE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _lease.LEASE_PATH.write_text(json.dumps(lease.__dict__, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Card test 1 — the three states never share a string
# --------------------------------------------------------------------------- #

def test_the_three_states_never_share_a_string():
    """Held-by-page, held-by-CLI, expired. The most important assertion on the card.

    A developer who walks up to a rig somebody else is driving is owed a
    different sentence from the one who walks up to a rig whose last session was
    killed. If these two ever collapse into "the arms are busy", the lease has
    stopped being a lantern and become a scold with extra steps.
    """
    by_page = _fresh(surface=SURFACE_PAGE, command="newt collect")
    by_cli = _fresh(surface=SURFACE_CLI, command="newt teleop")
    expired = _make()  # heartbeat is thirteen minutes stale at NOW

    assert _lease.state(by_page, now=NOW) == HELD
    assert _lease.state(by_cli, now=NOW) == HELD
    assert _lease.state(expired, now=NOW) == EXPIRED

    lines = [_lease.headline(x, now=NOW) for x in (by_page, by_cli, expired)]
    assert len(set(lines)) == 3, f"two states share a string: {lines}"

    # And each one says who and since when — a refusal that names neither is a
    # busy error wearing better clothes.
    for line, lease in zip(lines, (by_page, by_cli, expired)):
        assert lease.since_time() in line, f"no since-time in: {line}"
    assert "collect page" in lines[0]
    assert "newt teleop" in lines[1]


def test_the_three_expired_reasons_never_share_a_string(monkeypatch):
    """Gone, hung, and silent are three problems with three different next moves.

    Rule 12: two causes must never produce one string. Taking the arms from a
    process that is still running fails on a port it still holds, so telling a
    user that case is the same as a dead holder sends them into a confusing
    failure we could have named.
    """
    host = socket.gethostname()
    gone = _make(machine=host, pid=4821)
    hung = _make(machine=host, pid=4822)
    silent = _make()  # another machine — liveness is not checkable from here

    monkeypatch.setattr(_lease, "_process_alive", lambda pid: pid == 4822)

    assert _lease._reason(gone, now=NOW) == REASON_PROCESS_GONE
    assert _lease._reason(hung, now=NOW) == REASON_HOLDER_HUNG
    assert _lease._reason(silent, now=NOW) == REASON_SILENT

    lines = [_lease.headline(x, now=NOW) for x in (gone, hung, silent)]
    assert len(set(lines)) == 3, f"two expired causes share a string: {lines}"

    moves = [_lease.next_move(x, verb="teleop", now=NOW) for x in (gone, hung, silent)]
    # The hung holder's move is not the other two's: kill the process first.
    assert moves[0] == moves[2] == "newt teleop --take"
    assert moves[1] != moves[0] and "4822" in moves[1]


def test_a_dead_holder_expires_without_waiting_for_the_heartbeat(monkeypatch):
    """A killed session must not lock a bench for the staleness window.

    The heartbeat is the contract; a pid is a shortcut we can only take on our
    own machine. When we *can* prove nobody is home, waiting thirty seconds to
    say so is a lantern that arrives after the developer has given up.
    """
    just_now = (NOW - timedelta(seconds=1)).isoformat(timespec="seconds")
    dead = _make(machine=socket.gethostname(), pid=4821, heartbeat=just_now)
    monkeypatch.setattr(_lease, "_process_alive", lambda pid: False)

    assert _lease.state(dead, now=NOW) == EXPIRED
    assert _lease._reason(dead, now=NOW) == REASON_PROCESS_GONE


# --------------------------------------------------------------------------- #
# Card test 3 — an expired lease is takeable, and nothing takes it silently
# --------------------------------------------------------------------------- #

def test_acquire_refuses_an_expired_lease_rather_than_taking_it():
    """No path takes the arms without rendering the expired state first.

    This is the silent steal, and it is the reason ``expired`` is a state rather
    than a synonym for ``free``. If ``acquire`` ever returns a lease here, a
    developer's arms changed owner without anybody deciding.
    """
    _put(_make())

    with pytest.raises(ArmsLeaseExpired) as caught:
        _lease.acquire(surface=SURFACE_CLI, command="newt teleop", now=NOW)

    assert caught.value.reason == REASON_SILENT
    assert caught.value.lease.holder == "cli-4821-a3f9c1"
    # And the arms did not change hands on the way out.
    assert _lease.read().holder == "cli-4821-a3f9c1"


def test_holding_with_take_takes_an_expired_lease_and_gives_it_back():
    _put(_make())

    with _lease.held(surface=SURFACE_CLI, command="newt teleop", take=True) as mine:
        on_disk = _lease.read()
        assert on_disk.holder == mine.holder
        assert on_disk.holder != "cli-4821-a3f9c1"

    # Clean exit gives the arms back — the next session sees free, not expired.
    assert _lease.read() is None
    assert _lease.state(None) == FREE


def test_holding_without_take_refuses_an_expired_lease():
    """``take`` is the user having said so. It is never inferred from convenience."""
    _put(_make())

    with pytest.raises(ArmsLeaseExpired):
        with _lease.held(surface=SURFACE_CLI, command="newt teleop"):
            pytest.fail("the arms were taken without anybody deciding to take them")


def test_a_live_lease_refuses_even_with_take():
    """``--take`` is for a lease with nobody home, not for evicting a live session."""
    _put(_fresh(surface=SURFACE_PAGE, command="newt collect"))

    with pytest.raises(ArmsHeld):
        with _lease.held(surface=SURFACE_CLI, command="newt teleop", take=True):
            pytest.fail("a live holder was evicted")


# --------------------------------------------------------------------------- #
# The refusal, as a reader meets it
# --------------------------------------------------------------------------- #

def test_a_live_lease_refuses_as_a_lantern_not_a_busy_error():
    """Cause, owner, and one move — with the move alone on its own line.

    The move for a live holder is on the *other* surface on purpose: this process
    cannot make another one stop moving an arm, so offering ``--take`` here would
    be a control we cannot honour.
    """
    page = _fresh(surface=SURFACE_PAGE, command="newt collect")
    text = _lease.refusal(page, verb="teleop", now=NOW)

    assert text.startswith("newt teleop: ")
    assert "20:31" in text                      # since when
    assert "collect page" in text               # which door
    move = [ln for ln in text.splitlines() if ln.startswith("        ")]
    assert len(move) == 1, f"expected exactly one move line, got {move}"
    assert "--take" not in text, "a live session must not be offered as takeable"


def test_an_expired_lease_hands_over_the_take_move():
    text = _lease.refusal(_make(), verb="teleop", now=NOW)
    move = [ln.strip() for ln in text.splitlines() if ln.startswith("        ")]
    assert move == ["newt teleop --take"]


def test_the_page_and_the_terminal_say_the_same_thing():
    """Behaviour 3: the page names the holder *in the same terms* as the CLI.

    Asserted as one source rather than two similar strings, because two
    hand-written sentences drift apart the week after they are written and
    nothing fails when they do.
    """
    cli_holder = _fresh(surface=SURFACE_CLI, command="newt teleop")
    seen_by_page = _lease.describe(cli_holder, now=NOW)
    seen_by_cli = _lease.refusal(cli_holder, verb="collect", now=NOW)

    assert seen_by_page["headline"] in seen_by_cli
    assert seen_by_page["state"] == HELD
    assert seen_by_page["since_time"] == "20:31"


def test_a_free_bench_is_described_without_a_holder():
    described = _lease.describe(None)
    assert described["state"] == FREE
    assert described["next_move"] is None


# --------------------------------------------------------------------------- #
# Unreadable is not free
# --------------------------------------------------------------------------- #

def test_a_lease_that_does_not_parse_is_not_free():
    """Refuse rather than assume. Assuming is how two sessions drive one arm."""
    _lease.LEASE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _lease.LEASE_PATH.write_text("{ this is not json")

    with pytest.raises(LeaseUnreadable) as caught:
        _lease.read()
    assert str(_lease.LEASE_PATH) in str(caught.value)

    with pytest.raises(LeaseUnreadable):
        _lease.acquire(surface=SURFACE_CLI, command="newt teleop", now=NOW)


def test_a_lease_missing_a_field_does_not_parse_into_a_default():
    """Rule 10: no shaped-right default standing in for a field nobody wrote."""
    _lease.LEASE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _lease.LEASE_PATH.write_text(json.dumps({"holder": "x", "surface": "cli"}))

    with pytest.raises(LeaseUnreadable):
        _lease.read()


# --------------------------------------------------------------------------- #
# The race — a lease with a race in it is worse than no lease
# --------------------------------------------------------------------------- #

def test_eight_claimants_race_and_exactly_one_gets_the_arms():
    """Two processes reaching at once must produce one winner and seven refusals.

    ``flock`` is held per open file description, so eight threads each opening
    the lock file really do contend the same way eight processes would — this is
    a real race, not a simulated one.
    """
    winners: list[Lease] = []
    refused: list[Exception] = []
    start = threading.Barrier(8)

    def claim() -> None:
        start.wait()
        try:
            winners.append(_lease.acquire(surface=SURFACE_CLI, command="newt teleop"))
        except ArmsHeld as exc:
            refused.append(exc)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, f"{len(winners)} sessions all think they hold the arms"
    assert len(refused) == 7
    assert _lease.read().holder == winners[0].holder


def test_a_takeover_that_lost_its_race_refuses_rather_than_overwrites():
    """The user agreed to take *that* session's arms, not whoever is there now."""
    shown = _make()
    _put(shown)
    # Somebody else took it in the meantime.
    _put(_fresh(holder="cli-9999-bbbbbb", command="newt record"))

    with pytest.raises(LeaseMoved):
        _lease.take_over(shown, surface=SURFACE_CLI, command="newt teleop", now=NOW)

    assert _lease.read().holder == "cli-9999-bbbbbb"


def test_a_holder_whose_lease_was_taken_is_told_on_its_next_beat():
    """Continuing to drive after losing the lease is exactly the two-claimant case."""
    mine = _lease.acquire(surface=SURFACE_CLI, command="newt teleop")
    _put(_fresh(holder="page-1-cccccc", surface=SURFACE_PAGE, command="newt collect"))

    with pytest.raises(LeaseMoved):
        _lease.beat(mine)


def test_beating_keeps_a_lease_alive_and_leaves_since_alone():
    """The heartbeat moves; when it was taken does not. Since-time is what a
    human reads back, and a value that crept forward would be a quiet lie."""
    mine = _lease.acquire(surface=SURFACE_CLI, command="newt teleop", now=TAKEN_AT)
    assert _lease.state(mine, now=TAKEN_AT + timedelta(seconds=90)) == EXPIRED

    beaten = _lease.beat(mine, now=TAKEN_AT + timedelta(seconds=90))
    assert beaten.since == mine.since
    assert _lease.state(beaten, now=TAKEN_AT + timedelta(seconds=91)) == HELD


def test_release_does_not_delete_somebody_elses_lease():
    """The mirror image of the silent steal, and just as bad."""
    mine = _lease.acquire(surface=SURFACE_CLI, command="newt teleop")
    theirs = _fresh(holder="page-1-cccccc", surface=SURFACE_PAGE, command="newt collect")
    _put(theirs)

    _lease.release(mine)

    assert _lease.read().holder == "page-1-cccccc"


def test_acquire_writes_the_four_stamped_fields():
    """holder, surface, since, heartbeat — the mechanism as stamped."""
    mine = _lease.acquire(surface=SURFACE_PAGE, command="newt collect", now=TAKEN_AT)
    on_disk = json.loads(_lease.LEASE_PATH.read_text())

    assert on_disk["holder"] == mine.holder
    assert on_disk["surface"] == SURFACE_PAGE
    assert on_disk["since"] == TAKEN_AT.isoformat(timespec="seconds")
    assert on_disk["heartbeat"] == on_disk["since"]
    assert mine.since_time() == "20:31"
