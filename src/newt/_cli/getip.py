"""newt getip — this machine's LAN address, for the console's rig address field.

A hand-carried baton: today a developer reads the address off this verb and types it
into the Move mirror pane themselves. Once the bridge verb's heartbeat carries the rig's
address to the console on its own, this stops being a step in anyone's path and becomes
a diagnostic — what you run when the automatic path has gone dark.

What this verb will not do: pick a port. The port is whatever the rig's motion publisher
printed when it started, and `newt` has no way to know it. A shaped-right guess here
would open the wrong thing and blame the rig for it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import ipaddress
import json
import socket
import sys

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_GRAY = "\033[90m"


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Reading the machine's interfaces — POSIX getifaddrs(3) through ctypes
# ---------------------------------------------------------------------------
#
# ctypes rather than netifaces/psutil on purpose: the SDK's core dependency list is
# kept featherweight (`import newt` stays cheap), and one verb printing an IP does not
# earn a new wheel on every install.


class _Ifaddrs(ctypes.Structure):
    pass


# Only the leading fields are declared — they are laid out identically on Linux and the
# BSDs (macOS included), and nothing past ifa_netmask is read. Declaring the tail would
# mean tracking two platform layouts for data we never touch.
_Ifaddrs._fields_ = [
    ("ifa_next", ctypes.POINTER(_Ifaddrs)),
    ("ifa_name", ctypes.c_char_p),
    ("ifa_flags", ctypes.c_uint),
    ("ifa_addr", ctypes.c_void_p),
    ("ifa_netmask", ctypes.c_void_p),
]

_IFF_UP = 0x1
_IFF_LOOPBACK = 0x8

# BSD-derived kernels put a length byte ahead of the address family in `struct sockaddr`;
# Linux starts with a 2-byte family. Same struct, two headers.
_BSD_SOCKADDR = sys.platform == "darwin" or "bsd" in sys.platform


def _sockaddr_family(ptr: int) -> int:
    head = ctypes.string_at(ptr, 2)
    if _BSD_SOCKADDR:
        return head[1]
    return int.from_bytes(head, sys.byteorder)


def _sockaddr_address(ptr: int) -> tuple[str, str] | None:
    """(family label, printable address) for an IP sockaddr; None for anything else.

    Offsets are stable across Linux and BSD: two header bytes, then the port, then the
    address itself. socket.AF_INET6 is read from Python rather than hardcoded because
    its numeric value differs by platform (10 on Linux, 30 on macOS).
    """
    family = _sockaddr_family(ptr)
    if family == socket.AF_INET:
        return "ipv4", socket.inet_ntop(socket.AF_INET, ctypes.string_at(ptr, 8)[4:8])
    if family == socket.AF_INET6:
        return "ipv6", socket.inet_ntop(socket.AF_INET6, ctypes.string_at(ptr, 24)[8:24])
    return None


def _enumerate_interfaces() -> list[dict] | None:
    """Every IP address the OS reports, per interface, unfiltered.

    Returns None — not an empty list — when the platform's interfaces cannot be read at
    all. The two are different answers ("nothing is connected" vs. "we cannot look") and
    they get different refusals downstream.
    """
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return None
    try:
        libc = ctypes.CDLL(libc_name, use_errno=True)
        getifaddrs = libc.getifaddrs
        freeifaddrs = libc.freeifaddrs
    except (OSError, AttributeError):
        return None

    getifaddrs.restype = ctypes.c_int
    getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(_Ifaddrs))]
    freeifaddrs.argtypes = [ctypes.POINTER(_Ifaddrs)]

    head = ctypes.POINTER(_Ifaddrs)()
    if getifaddrs(ctypes.byref(head)) != 0:
        return None

    found: list[dict] = []
    try:
        node = head
        while node:
            entry = node.contents
            if entry.ifa_addr:
                parsed = _sockaddr_address(entry.ifa_addr)
                if parsed is not None:
                    family, address = parsed
                    found.append(
                        {
                            "interface": entry.ifa_name.decode("utf-8", "replace"),
                            "family": family,
                            "address": address,
                            "up": bool(entry.ifa_flags & _IFF_UP),
                            "loopback": bool(entry.ifa_flags & _IFF_LOOPBACK),
                        }
                    )
            node = entry.ifa_next
    finally:
        freeifaddrs(head)
    return found


def _default_route_address() -> str | None:
    """The address this machine would source from to reach another host, or None.

    A UDP connect() only fixes a route in the kernel — no packet leaves. 192.0.2.1 is
    RFC 5737 documentation space, so even a stray datagram reaches nothing real. This is
    the one genuinely-known signal about which interface is "the" one; everything else in
    the ordering below is a heuristic and is ranked underneath it.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


# ---------------------------------------------------------------------------
# Choosing and ordering — pure, so the tests can drive every case
# ---------------------------------------------------------------------------

# Interfaces that exist for tunnels, containers, VMs, and peer-to-peer radios. A browser
# on the same network usually does not reach a rig through one of these. They are ranked
# down, never dropped — the machine that needs one is exactly the machine that would be
# stuck if we hid it.
_VIRTUAL_PREFIXES = (
    "awdl", "llw", "utun", "gif", "stf", "bridge", "vmnet", "ap",
    "docker", "br-", "veth", "virbr", "vbox", "tun", "tap", "wg", "zt", "tailscale",
)


def _is_virtual(interface: str) -> bool:
    return interface.startswith(_VIRTUAL_PREFIXES)


def select_addresses(entries: list[dict], default_route: str | None = None) -> list[dict]:
    """Reachable candidates, best guess first, nothing dropped that a browser could use.

    Filters only what genuinely cannot be pasted anywhere useful — loopback, link-local,
    multicast, the unspecified address. Everything that survives is returned and labeled;
    a machine with four candidates prints four lines (Rule 10).
    """
    candidates: list[dict] = []
    for entry in entries:
        if entry.get("loopback"):
            continue
        try:
            ip = ipaddress.ip_address(entry["address"])
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            continue
        candidates.append(
            {
                "address": entry["address"],
                "interface": entry["interface"],
                "family": entry["family"],
                "up": bool(entry.get("up", True)),
            }
        )

    def rank(candidate: dict) -> tuple:
        return (
            0 if candidate["address"] == default_route else 1,
            0 if candidate["up"] else 1,
            0 if candidate["family"] == "ipv4" else 1,
            0 if not _is_virtual(candidate["interface"]) else 1,
            0 if ipaddress.ip_address(candidate["address"]).is_private else 1,
            candidate["interface"],
            candidate["address"],
        )

    candidates.sort(key=rank)
    for position, candidate in enumerate(candidates):
        candidate["best_guess"] = position == 0
    return candidates


def _paste_form(candidate: dict) -> str:
    """The address as the console's rig field wants to read it.

    IPv6 goes in brackets so that appending `:<port>` still parses as one address.
    """
    if candidate["family"] == "ipv6":
        return f"[{candidate['address']}]"
    return candidate["address"]


# ---------------------------------------------------------------------------
# Refusals — one cause, one string, one next move
# ---------------------------------------------------------------------------

_REFUSALS: dict[str, tuple[str, str]] = {
    "no_interface": (
        (
            "Every interface on this machine is loopback or link-local, so it can currently "
            "reach only itself — there is nothing here a browser elsewhere on your network "
            "could open."
        ),
        (
            "Bring up Wi-Fi or Ethernet and run this again. Inside a container, that means "
            "starting it on a bridged network rather than the default isolated one."
        ),
    ),
    "unreadable_platform": (
        (
            f"This verb reads interfaces through POSIX getifaddrs(3), and {sys.platform} did "
            "not hand it over — so nothing was enumerated, rather than nothing being present."
        ),
        (
            "Read the address out of your operating system's network settings for now, and "
            "open an issue on newt-python naming your platform so this verb learns it."
        ),
    ),
}


def _usage() -> None:
    print("Usage: newt getip [options]")
    print()
    print("  Print this machine's LAN address(es) — what to paste into the console's rig")
    print("  address field, best guess first. A hand-carried stopgap: once the bridge verb")
    print("  announces the rig's address itself, this is only a diagnostic.")
    print()
    print("Options:")
    print("  --json  Emit machine-readable JSON")


def cmd_getip(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    as_json = "--json" in args

    entries = _enumerate_interfaces()
    if entries is None:
        candidates: list[dict] = []
        error_kind: str | None = "unreadable_platform"
    else:
        candidates = select_addresses(entries, _default_route_address())
        error_kind = None if candidates else "no_interface"

    if as_json:
        print(json.dumps({"addresses": candidates, "error_kind": error_kind}))
        return 0 if error_kind is None else 1

    if error_kind is not None:
        cause, next_move = _REFUSALS[error_kind]
        print(_c(_RED, cause), file=sys.stderr)
        print(next_move, file=sys.stderr)
        return 1

    _render(candidates)
    return 0


def _render(candidates: list[dict]) -> None:
    width = max(len(_paste_form(c)) for c in candidates)
    for candidate in candidates:
        shown = _paste_form(candidate).ljust(width)
        label = candidate["interface"]
        if not candidate["up"]:
            label += "  (interface is down)"
        if candidate["best_guess"]:
            print(f"{_c(_GREEN, shown)}  {label}  <- best guess")
        else:
            print(f"{shown}  {_c(_GRAY, label)}")

    print()
    print("Paste one into the console's rig address field, followed by the port your motion")
    print(f"publisher printed when it started: {_paste_form(candidates[0])}:<port>")
    print(_c(_GRAY, "Nothing here knows that port, so nothing here fills it in."))
