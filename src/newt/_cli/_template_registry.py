"""The starter-template registry — the one place in ``newt`` that knows a robot's name.

``newt create`` dispenses starter kits. Everything it needs to do that is here: a
name, whether that name is public, and — for public names only — the repository
and the exact commit to fetch. Nothing else. There is no camera count in this
file, no driver pin, no arm topology, no prompt schema. Adding the next robot is
a row.

**The console is the registry; this table is the offline fallback.** The
authoritative mapping lives console-side, so a new starter kit becomes available
without waiting for an SDK release, and so a private kit can be served to a
developer whose ``nt_`` key says they may have it — the console is the only thing
holding the credential that can read a private repository. The CLI asks it first.

**Why the fallback carries public templates only.** This package is
world-readable. A public row publishes nothing that ``git clone`` would not
already hand out. A private row would publish a repository path that only the
console is supposed to be able to read, so private templates appear here as a
name and a visibility and nothing more: enough to tell a developer "this one
needs your key" instead of "no such thing," which is the whole difference
between a useful refusal and a misleading one.

**Refs are commits, not branches.** A pinned SHA is what makes running the same
command next month produce the same directory. "The tip of main" is a different
product, and it is not the one being shipped.
"""
from __future__ import annotations

from dataclasses import dataclass

PUBLIC = "public"
PRIVATE = "private"


@dataclass(frozen=True)
class Template:
    """One registry row.

    ``repo`` and ``ref`` are populated for templates this client can fetch on its
    own — public ones. For a private template they are ``None``: the console
    resolves and serves it, and the client never learns where it lives.
    """

    name: str
    visibility: str
    repo: str | None = None
    ref: str | None = None

    @property
    def is_private(self) -> bool:
        return self.visibility == PRIVATE


# Ordered — this is the order a developer sees when the CLI lists templates.
FALLBACK_TEMPLATES: tuple[Template, ...] = (
    Template(
        name="so101",
        visibility=PUBLIC,
        repo="new-theory-research/newt-starter-so101",
        ref="a2ae79a618831e6bb883c48031fa4b2871509303",
    ),
    Template(
        name="yam",
        visibility=PUBLIC,
        repo="new-theory-research/newt-starter-yam",
        ref="6a7fdf9acad4494d3383fcfba9997fda82d47ad7",
    ),
    Template(
        name="yam-bimanual",
        visibility=PUBLIC,
        repo="new-theory-research/newt-starter-yam-bimanual",
        ref="ecd9fc7c99294c96d1b4266d1a8ff6eacaa841e7",
    ),
    Template(name="trossen-widowx", visibility=PRIVATE),
)


def parse_templates(payload: object) -> list[Template]:
    """Turn a console ``/api/cli/templates`` response into rows.

    Raises ``ValueError`` on anything that is not the documented shape. A
    malformed registry is a contract violation to surface, never an empty list to
    quietly fall through on — an empty list reads as "you have access to nothing,"
    which is a different and wrong answer.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    rows = payload.get("templates")
    if not isinstance(rows, list):
        raise ValueError("no 'templates' array in the response")

    out: list[Template] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"template entry is not an object: {row!r}")
        name = row.get("name")
        visibility = row.get("visibility")
        if not isinstance(name, str) or not name:
            raise ValueError(f"template entry has no name: {row!r}")
        if visibility not in (PUBLIC, PRIVATE):
            raise ValueError(
                f"template {name!r} has visibility {visibility!r}; "
                f"expected {PUBLIC!r} or {PRIVATE!r}"
            )
        repo = row.get("repo")
        ref = row.get("ref")
        out.append(
            Template(
                name=name,
                visibility=visibility,
                repo=repo if isinstance(repo, str) and repo else None,
                ref=ref if isinstance(ref, str) and ref else None,
            )
        )
    return out


def find(templates: list[Template] | tuple[Template, ...], name: str) -> Template | None:
    for t in templates:
        if t.name == name:
            return t
    return None


def names(templates: list[Template] | tuple[Template, ...]) -> list[str]:
    return [t.name for t in templates]
