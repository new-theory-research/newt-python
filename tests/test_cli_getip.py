"""Offline unit tests for `newt getip`.

Every test drives the verb with a fabricated interface table — no getifaddrs(3)
call, no UDP route probe, no network. What each one encodes is why the developer
standing at the rig can trust the line they paste:

  - nothing that a browser could reach is hidden from them (Rule 10),
  - the two ways this can fail read as two different problems (Rule 12),
  - the printed host still parses once they append the port their publisher named,
  - a refusal never leaves stdout looking like an address.
"""
from __future__ import annotations

import io
import json
import re
import sys

import pytest

from newt._cli import getip as getip_mod
from newt._cli.getip import cmd_getip, select_addresses

# The Move mirror's own parser, transcribed from
# portal:apps/console/app/preview/_viewer/rig-address.ts (HOST_PORT). Kept as a
# literal copy on purpose: if the pane's accepted shape changes, this test file is
# where the divergence should surface, not a developer's paste at the bench.
_HOST_PORT = re.compile(r"^([A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+\]):(\d{1,5})$")


# A machine with more going on than one cable: the LAN it routes through, a second
# wired interface, a tunnel, a ULA, and the loopback/link-local noise the OS always
# reports alongside them.
_MULTI = [
    {"interface": "lo0", "family": "ipv4", "address": "127.0.0.1", "up": True, "loopback": True},
    {"interface": "lo0", "family": "ipv6", "address": "::1", "up": True, "loopback": True},
    {"interface": "en0", "family": "ipv4", "address": "192.168.1.42", "up": True, "loopback": False},
    {"interface": "en0", "family": "ipv6", "address": "fe80::1", "up": True, "loopback": False},
    {"interface": "en0", "family": "ipv6", "address": "fd00::42", "up": True, "loopback": False},
    {"interface": "en1", "family": "ipv4", "address": "10.0.5.9", "up": True, "loopback": False},
    {"interface": "utun6", "family": "ipv4", "address": "100.64.1.5", "up": True, "loopback": False},
]

_MULTI_ROUTE = "192.168.1.42"

_SINGLE = [
    {"interface": "lo0", "family": "ipv4", "address": "127.0.0.1", "up": True, "loopback": True},
    {"interface": "eth0", "family": "ipv4", "address": "192.168.7.20", "up": True, "loopback": False},
]

# Offline laptop, or a container on the default isolated network: interfaces exist,
# none of them reach anywhere.
_NO_INTERFACE = [
    {"interface": "lo0", "family": "ipv4", "address": "127.0.0.1", "up": True, "loopback": True},
    {"interface": "en0", "family": "ipv6", "address": "fe80::c1a0", "up": True, "loopback": False},
]


def _run(args: list[str], monkeypatch, entries, default_route=None):
    """Run cmd_getip against a fabricated interface table, capturing both streams."""
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(getip_mod, "_enumerate_interfaces", lambda: entries)
    monkeypatch.setattr(getip_mod, "_default_route_address", lambda: default_route)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    exit_code = cmd_getip(args)
    return exit_code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Golden: the multi-interface machine sees every candidate, best guess first
# ---------------------------------------------------------------------------

def test_multi_interface_lists_every_reachable_candidate(monkeypatch):
    """A machine with four usable addresses prints four lines — none collapsed away.

    Picking one and hiding the rest is the Rule 10 failure this verb exists to
    avoid: the developer whose rig answers on the *second* interface would be told,
    confidently, an address that reaches nothing.
    """
    exit_code, out, err = _run([], monkeypatch, _MULTI, _MULTI_ROUTE)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    for address in ("192.168.1.42", "10.0.5.9", "100.64.1.5", "fd00::42"):
        assert address in out, f"candidate {address} was dropped from the output: {out!r}"


def test_multi_interface_labels_every_candidate_by_interface(monkeypatch):
    """Each address is labeled with the interface it came from.

    Four bare addresses are a guessing game; `en0` / `utun6` is how the person at
    the bench knows which one is the cable they just plugged in.
    """
    exit_code, out, _ = _run([], monkeypatch, _MULTI, _MULTI_ROUTE)

    assert exit_code == 0
    for interface in ("en0", "en1", "utun6"):
        assert interface in out, f"interface label {interface} missing: {out!r}"


def test_default_route_address_is_the_best_guess(monkeypatch):
    """The address the kernel actually routes from is ranked first and marked.

    Everything else in the ordering is a heuristic. The route probe is the one
    genuinely-known signal, so it outranks them — and the marker is what makes the
    ordering legible instead of arbitrary.
    """
    exit_code, out, _ = _run([], monkeypatch, _MULTI, _MULTI_ROUTE)

    assert exit_code == 0
    first_line = out.strip().splitlines()[0]
    assert "192.168.1.42" in first_line, f"default-route address must lead: {out!r}"
    assert "best guess" in first_line, f"the leading line must say so: {first_line!r}"


def test_route_probe_outranks_the_physical_interface_heuristic(monkeypatch):
    """When the routed address sits on a tunnel, the tunnel still wins.

    A rig reached over Tailscale is the case where ranking physical-before-virtual
    as gospel would put the wrong line first. Known beats guessed.
    """
    _, out, _ = _run([], monkeypatch, _MULTI, "100.64.1.5")

    first_line = out.strip().splitlines()[0]
    assert "100.64.1.5" in first_line, f"routed tunnel address must lead: {out!r}"
    assert "best guess" in first_line


def test_loopback_and_link_local_never_printed(monkeypatch):
    """127.0.0.1, ::1 and fe80:: addresses are filtered, not ranked last.

    They are the addresses a developer is most likely to grab by hand off
    `ifconfig` and the ones guaranteed to fail from another machine — printing
    them at all, even at the bottom, invites the paste.
    """
    _, out, _ = _run([], monkeypatch, _MULTI, _MULTI_ROUTE)

    for unusable in ("127.0.0.1", "::1", "fe80::1"):
        assert unusable not in out, f"unusable address {unusable} was printed: {out!r}"


def test_down_interface_is_listed_but_ranked_below_up_ones(monkeypatch):
    """An interface that is down still appears — labeled as down, never first.

    Hiding it strands the developer who is looking at exactly that cable and
    wondering why it isn't in the list; ranking it first sends everyone else to a
    dead address.
    """
    entries = list(_SINGLE) + [
        {"interface": "en5", "family": "ipv4", "address": "192.168.9.9", "up": False, "loopback": False},
    ]
    _, out, _ = _run([], monkeypatch, entries, "192.168.7.20")

    assert "192.168.9.9" in out, f"down interface must still be listed: {out!r}"
    assert "down" in out, f"down interface must be labeled as such: {out!r}"
    assert out.strip().splitlines()[0].startswith("192.168.7.20")


# ---------------------------------------------------------------------------
# Golden: one interface, one obvious answer
# ---------------------------------------------------------------------------

def test_single_interface_prints_one_marked_candidate(monkeypatch):
    """The ordinary machine prints its one address, marked, and exits clean."""
    exit_code, out, err = _run([], monkeypatch, _SINGLE, "192.168.7.20")

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    assert "192.168.7.20" in out
    assert "eth0" in out
    assert out.count("best guess") == 1, f"exactly one best guess: {out!r}"


def test_no_port_is_ever_invented(monkeypatch):
    """The verb hands over the port question instead of answering it.

    `newt` cannot know what a rig's motion publisher bound to. A plausible-looking
    `:9096` would open the wrong thing and blame the rig for it, so the output
    carries the placeholder and says where the real number comes from.
    """
    _, out, _ = _run([], monkeypatch, _SINGLE, "192.168.7.20")

    assert "<port>" in out, f"the port must be handed over explicitly: {out!r}"
    assert "9096" not in out, f"no port number may appear as fact: {out!r}"
    assert re.search(r"\d+\.\d+\.\d+\.\d+:\d", out) is None, (
        f"no address may be printed with a port attached: {out!r}"
    )


# ---------------------------------------------------------------------------
# Golden: two failures, two causes, two strings (Rule 12)
# ---------------------------------------------------------------------------

def test_no_interface_refusal_names_cause_and_next_move(monkeypatch):
    """Nothing reachable → say what was noticed, then hand over one move.

    "no address found" tells the developer nothing they didn't already know. This
    string has to name the observation (everything here is loopback or link-local)
    and the thing to do about it (bring up Wi-Fi/Ethernet, or bridge the container).
    """
    exit_code, out, err = _run([], monkeypatch, _NO_INTERFACE)

    assert exit_code != 0, "a machine with nothing to report must exit non-zero"
    assert out == "", f"a refusal must not print to stdout: {out!r}"
    lowered = err.lower()
    assert "loopback" in lowered or "link-local" in lowered, (
        f"the refusal must name what it noticed: {err!r}"
    )
    assert "wi-fi" in lowered or "ethernet" in lowered or "bridged" in lowered, (
        f"the refusal must hand over a next move: {err!r}"
    )
    assert lowered.count("no address") == 0, f"reprimand phrasing, not a lantern: {err!r}"


def test_unreadable_platform_refusal_is_a_distinct_string(monkeypatch):
    """"We could not look" and "there is nothing there" are different answers.

    They share an exit code and nothing else. A developer on an unsupported
    platform who reads the offline-machine wording goes off checking cables that
    were never the problem.
    """
    _, _, no_interface = _run([], monkeypatch, _NO_INTERFACE)
    exit_code, out, unreadable = _run([], monkeypatch, None)

    assert exit_code != 0
    assert out == "", f"a refusal must not print to stdout: {out!r}"
    assert unreadable != no_interface, "two causes must never share one string"
    assert "getifaddrs" in unreadable, (
        f"must name what it tried to read: {unreadable!r}"
    )
    assert sys.platform in unreadable, (
        f"must name the platform it could not read: {unreadable!r}"
    )

    # No shared sentence between the two refusals — near-identical wording is how
    # two causes quietly become one message in a reader's head.
    shared = set(_sentences(no_interface)) & set(_sentences(unreadable))
    assert not shared, f"refusals share wording: {shared!r}"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 12]


def test_refusals_go_to_stderr_so_command_substitution_stays_honest(monkeypatch):
    """`ADDR=$(newt getip)` must capture an address or nothing — never a refusal.

    Refusal text on stdout is the silent-wrong-value failure in its most literal
    form: a shell variable holding a paragraph of English that then gets pasted
    into a config as if it were a host.
    """
    for entries in (_NO_INTERFACE, None):
        exit_code, out, err = _run([], monkeypatch, entries)
        assert exit_code != 0
        assert out == "", f"stdout must stay empty on refusal: {out!r}"
        assert err.strip(), "the refusal has to be said somewhere"


# ---------------------------------------------------------------------------
# Golden: the agent door is additive, and the same list
# ---------------------------------------------------------------------------

def test_json_carries_the_same_candidates_in_the_same_order(monkeypatch):
    """`--json` is the same answer in another shape — not a smaller one.

    An agent that reads fewer candidates than the human sees would make exactly
    the pick-and-hide mistake the human path refuses to make.
    """
    exit_code, out, err = _run(["--json"], monkeypatch, _MULTI, _MULTI_ROUTE)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    data = json.loads(out)
    assert data["error_kind"] is None
    addresses = [entry["address"] for entry in data["addresses"]]
    assert addresses == ["192.168.1.42", "10.0.5.9", "100.64.1.5", "fd00::42"], (
        f"json order must match the ranked human order: {addresses!r}"
    )


def test_json_entries_carry_every_field_an_agent_ranks_on(monkeypatch):
    """Each entry names its address, interface, family, link state, and the pick."""
    _, out, _ = _run(["--json"], monkeypatch, _MULTI, _MULTI_ROUTE)

    data = json.loads(out)
    for entry in data["addresses"]:
        for field in ("address", "interface", "family", "up", "best_guess"):
            assert field in entry, f"required field '{field}' missing: {entry!r}"
    best = [entry for entry in data["addresses"] if entry["best_guess"]]
    assert len(best) == 1, f"exactly one best_guess: {data!r}"
    assert best[0]["address"] == _MULTI_ROUTE


@pytest.mark.parametrize(
    "entries,expected_kind",
    [(_NO_INTERFACE, "no_interface"), (None, "unreadable_platform")],
)
def test_json_names_which_failure_it_was(monkeypatch, entries, expected_kind):
    """The two refusals stay distinguishable through the machine door too.

    An agent branching on `error_kind` decides between "wait and retry once the
    network is up" and "this platform will never answer" — collapsing them to one
    code makes that branch unwritable.
    """
    exit_code, out, err = _run(["--json"], monkeypatch, entries)

    assert exit_code != 0
    data = json.loads(out)
    assert data["error_kind"] == expected_kind
    assert data["addresses"] == []
    assert err == "", "json mode says it once, in the payload"


def test_human_path_is_complete_without_the_json_flag(monkeypatch):
    """Bare `newt getip` stands on its own — the agent door adds, never carries.

    Everything `--json` reports (every address, its interface, which one is the
    pick) has to be readable in the plain output too.
    """
    _, human, _ = _run([], monkeypatch, _MULTI, _MULTI_ROUTE)
    _, machine, _ = _run(["--json"], monkeypatch, _MULTI, _MULTI_ROUTE)

    for entry in json.loads(machine)["addresses"]:
        assert entry["address"] in human, f"{entry['address']} missing from human output"
        assert entry["interface"] in human, f"{entry['interface']} missing from human output"
    assert "best guess" in human


# ---------------------------------------------------------------------------
# Golden: what gets printed still parses on the other end
# ---------------------------------------------------------------------------

def test_every_printed_address_parses_once_a_port_is_appended(monkeypatch):
    """The paste has to survive the pane's parser, not just look like an address.

    The Move mirror refuses a bare host by design (it won't pick a port either), so
    the contract this verb owes is that `<printed>:<port>` matches HOST_PORT — for
    every candidate, including the IPv6 ones.
    """
    _, out, _ = _run(["--json"], monkeypatch, _MULTI, _MULTI_ROUTE)

    for entry in json.loads(out)["addresses"]:
        paste = getip_mod._paste_form(entry)
        assert _HOST_PORT.match(f"{paste}:9096"), (
            f"{paste!r} does not parse as a rig address once a port is added"
        )


def test_ipv6_is_printed_bracketed_because_raw_v6_does_not_parse(monkeypatch):
    """`[fd00::42]`, not `fd00::42` — and the reason is mechanical, not stylistic.

    Append `:9096` to a raw v6 literal and the pane sees one more colon-separated
    group, not a port. Invisible until someone on a v6-only bench pastes a line
    that silently fails.
    """
    _, out, _ = _run([], monkeypatch, _MULTI, _MULTI_ROUTE)

    assert "[fd00::42]" in out, f"IPv6 must print bracketed: {out!r}"
    assert not _HOST_PORT.match("fd00::42:9096"), (
        "if raw v6 started parsing, the bracketing rationale needs rechecking"
    )
    assert _HOST_PORT.match("[fd00::42]:9096")


# ---------------------------------------------------------------------------
# Golden: help text, and why this verb exists at all
# ---------------------------------------------------------------------------

def test_help_exits_clean_and_enumerates_nothing(monkeypatch):
    """`-h` prints usage and performs no work — the uniform guard every verb keeps."""
    def explode():
        raise AssertionError("--help must not enumerate interfaces")

    out = io.StringIO()
    monkeypatch.setattr(getip_mod, "_enumerate_interfaces", explode)
    monkeypatch.setattr(sys, "stdout", out)

    assert cmd_getip(["--help"]) == 0
    assert "newt getip" in out.getvalue()


def test_help_carries_the_manual_baton_framing():
    """The help text says out loud that this verb is a stopgap the bridge replaces.

    A developer who meets `getip` on day one should not build a habit around a step
    that is scheduled to disappear — and a future reader of the source should not
    have to find this card to learn that.
    """
    out = io.StringIO()
    stdout = sys.stdout
    sys.stdout = out
    try:
        cmd_getip(["-h"])
    finally:
        sys.stdout = stdout

    helptext = out.getvalue().lower()
    assert "bridge" in helptext, f"help must name what supersedes this verb: {helptext!r}"
    assert "diagnostic" in helptext or "stopgap" in helptext, (
        f"help must say this is an interim step: {helptext!r}"
    )
    assert "bridge" in getip_mod.__doc__.lower(), (
        "the module docstring carries the same framing for a source reader"
    )


# ---------------------------------------------------------------------------
# The ranking function, driven directly
# ---------------------------------------------------------------------------

def test_select_addresses_is_pure_and_orders_without_a_route_probe():
    """With no route signal, the heuristics alone still produce a defensible order.

    IPv4 before IPv6, physical before virtual, private before public — the machine
    where the UDP probe fails (no route at all) still gets a ranked list rather
    than whatever order the kernel happened to report.
    """
    ranked = select_addresses(_MULTI, default_route=None)

    assert [entry["address"] for entry in ranked] == [
        "192.168.1.42",   # physical, private, v4 — en0 sorts ahead of en1 at equal rank
        "10.0.5.9",       # the other physical v4
        "100.64.1.5",     # virtual tunnel, ranked under both physical ones
        "fd00::42",       # v6 last
    ]
    assert ranked[0]["best_guess"] is True
    assert all(entry["best_guess"] is False for entry in ranked[1:])


def test_select_addresses_drops_nothing_it_merely_ranks():
    """A tunnel-only machine gets its tunnel address, not a refusal.

    Ranking virtual interfaces down is a guess about which cable a browser uses.
    Dropping them would make that guess load-bearing, and strand the developer
    whose only route to the rig *is* the tunnel.
    """
    tunnel_only = [
        {"interface": "lo0", "family": "ipv4", "address": "127.0.0.1", "up": True, "loopback": True},
        {"interface": "utun4", "family": "ipv4", "address": "100.64.9.9", "up": True, "loopback": False},
    ]
    ranked = select_addresses(tunnel_only, default_route=None)

    assert [entry["address"] for entry in ranked] == ["100.64.9.9"]
    assert ranked[0]["best_guess"] is True
