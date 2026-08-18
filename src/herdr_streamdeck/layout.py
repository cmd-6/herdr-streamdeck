"""Mapping from deck keys to herdr panes.

The deck is a grid -- 3 rows by 5 columns on an MK.2. Each **column** is a
group (a workspace, or a tab within one workspace) and each **row** is one of
that group's panes.

There is no stored layout and no pinning: **herdr's current arrangement is the
source of truth**, and the deck mirrors it. Two orderings make that work, both
verified against herdr 0.7.5:

* ``workspace.list`` returns workspaces in sidebar order -- the same order as
  ``session.json``, carrying an explicit 1-based ``number``. It is what
  ``workspace.move``'s ``insert_index`` rearranges, so it is the user's own
  ordering, not an accident of allocation.
* ``pane.list`` returns panes in a depth-first walk of the tab's split tree.
  For a tab laid out ``Split(Pane 5, Split(Pane 6, Pane 10))`` it returns
  ``p1, p2, p6`` in that order, so row order already matches what is on screen.

So neither ordering needs to be computed or remembered -- only preserved.
Nothing here may sort: sorting would silently substitute our opinion for
herdr's, which is precisely what mirroring must not do.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .protocol import JSONObject

logger = logging.getLogger(__name__)


class GroupingMode(StrEnum):
    """What a column represents."""

    WORKSPACE = "workspace"
    """Each column is a workspace; rows are its panes."""

    TAB = "tab"
    """Each column is a tab of one workspace; rows are its panes."""

    AGENT = "agent"
    """No columns at all: the busiest agents, laid out row-major.

    Grouping by workspace spends a whole column on a workspace holding one
    pane, so a deck of five columns showed five agents out of twelve. This
    mode drops the grouping and fills keys left-to-right instead."""


BADGE_LENGTH = 8
"""Characters a badge can show.

Measured against the rendered badge rather than guessed: at the 10px badge
font there are 56px of usable width, and eight characters of realistic text
come in under that (``ENG-4521`` is 50.6px, ``refactor`` 39.3px). Strings of
uniformly wide glyphs (``mmmmmmmm``, 77.9px) still overflow and fall back to
the renderer's ellipsis, which is an acceptable edge for pane names."""

_WORKSPACE_ID = re.compile(r"^w([0-9A-Fa-f]+)$")
"""herdr's workspace ids: `w` and a hexadecimal counter -- w9, wA, wB, wC."""

_SPINNER = re.compile(r"^[\s\u2000-\u3000\u25a0-\u25ff\u2700-\u27bf\u2800-\u28ff*]+")
"""Leading spinner glyphs an agent writes into its terminal title.

herdr's ``terminal_title_stripped`` removes the marks it knows about, but not
all of them -- Claude Code cycles a quarter-circle glyph that arrives intact.
Left in they are worse than ugly: the glyph changes frame to frame, so the
title changes with it, and every frame re-renders and rewrites the key."""

_TICKET = re.compile(r"^[A-Za-z]{3}-(\d+)(?:-(.+))?$")
"""A ticket-style name: three letters, a hyphen, a number, optionally more."""


def abbreviate(name: str, limit: int = BADGE_LENGTH) -> str:
    """Shorten a pane name to what fits on a key.

    Ticket-style names are the interesting case, because their first four
    characters are the project prefix -- identical across every pane, and so
    useless for telling them apart. The number, or better a trailing
    description, actually distinguishes them:

    =========================  ============  =============================
    name                       badge         why
    =========================  ============  =============================
    ``ENG-4521``               ``ENG-4521``  fits whole, so keep it whole
    ``ENG-45211``              ``45211``     too long: the number identifies
    ``ENG-4521-refactor``      ``refactor``  a description beats a number
    ``ENG-4521-authenticate``  ``authenti``  trimmed to the limit
    ``reviewer``               ``reviewer``  not a ticket, and it fits
    ``build-and-deploy-all``   ``build-an``  not a ticket: leading characters
    =========================  ============  =============================

    Case is preserved: ticket prefixes are conventionally upper and
    descriptions lower, and flattening either loses a legibility cue.
    """
    trimmed = name.strip()
    if not trimmed:
        return ""

    # Whole name first. Abbreviating something that already fits throws away
    # information for nothing -- "ENG-4521" is more use than "4521".
    if len(trimmed) <= limit:
        return trimmed

    ticket = _TICKET.match(trimmed)
    if ticket is None:
        return trimmed[:limit]

    number, rest = ticket.group(1), ticket.group(2)
    if rest:
        # Only up to the next hyphen: "4521-refactor-client" describes itself
        # as "refa", not as a slice spanning a separator.
        return rest.split("-", 1)[0][:limit]
    return number[:limit]


@dataclass(frozen=True, slots=True)
class Pane:
    """The bits of a pane record that reach the deck."""

    pane_id: str
    workspace_id: str = ""
    tab_id: str = ""
    label: str = ""
    title: str = ""
    terminal_title: str = ""
    agent: str = ""
    display_agent: str = ""
    status: str = "unknown"
    cwd: str = ""
    state_change_seq: int = 0
    """herdr's server-wide counter as of this agent's last status change.

    The closest thing the protocol has to "when did this last do something":
    there are no timestamps on any record. It is global rather than per-agent
    -- a pane with `revision` 7 carries seq 143, and a pane cannot have changed
    state more often than it has been revised -- so comparing it across panes
    orders them by recency of activity. It reaches us only through the
    snapshot's `agents` listing; `pane.list` omits it."""

    @classmethod
    def from_record(cls, record: JSONObject) -> Pane | None:
        pane_id = record.get("pane_id")
        if not isinstance(pane_id, str):
            return None

        def text(key: str) -> str:
            value = record.get(key)
            return value if isinstance(value, str) else ""

        def number(key: str) -> int:
            value = record.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        return cls(
            pane_id=pane_id,
            workspace_id=text("workspace_id"),
            tab_id=text("tab_id"),
            label=text("label"),
            title=text("title"),
            # `_stripped` has the agent's own status glyph removed -- the raw
            # `terminal_title` reads "✳ orchestrator", and that leading
            # mark is both redundant with the one we draw and wide enough to
            # cost two of the eight badge characters.
            terminal_title=_SPINNER.sub(
                "", text("terminal_title_stripped") or text("terminal_title")
            ).strip(),
            agent=text("agent"),
            display_agent=text("display_agent"),
            status=text("agent_status") or "unknown",
            cwd=text("cwd"),
            state_change_seq=number("state_change_seq"),
        )

    @property
    def mark_key(self) -> str:
        """Which agent's mark to draw.

        ``display_agent`` wins because it is a free-form override, unlike
        ``agent`` which herdr constrains to its detection enum. That is how an
        agent herdr cannot detect -- qwencode, say -- still gets its own mark.
        """
        return self.display_agent or self.agent

    @property
    def given_name(self) -> str:
        """The name a person chose for this pane, or empty if nobody has.

        Distinct from ``badge``: this is only the deliberate name, never the
        terminal title. A pane the user bothered to rename is telling us what
        that pane is *for*, which outranks anything we or the agent infer.

        At protocol 19 ``pane.list`` carries no ``title`` key at all and
        ``label`` only on panes that have been renamed, so this is empty far
        more often than not. That is the expected shape, not a fault.
        """
        return (self.title or self.label).strip()

    @property
    def creation_key(self) -> tuple[int, str]:
        """Sort key placing older panes before newer ones.

        herdr hands out workspace ids sequentially, so `w1` predates `wC`, and
        the numeric part of the id is the only creation-order signal the
        protocol carries -- no record on any method has a timestamp. Sorting on
        it rather than on `workspace.number` is deliberate: `number` follows
        sidebar *position*, so dragging a workspace would reshuffle the deck
        and "oldest" would quietly stop meaning oldest.

        Ids are documented as opaque, so this reads them defensively: anything
        that does not parse sorts last, by id, rather than raising.
        """
        digits = _WORKSPACE_ID.match(self.workspace_id)
        if digits is None:
            return (1 << 30, self.workspace_id)
        return (int(digits.group(1), 16), self.pane_id)

    @property
    def badge(self) -> str:
        """Text for the corner badge: the most specific name, abbreviated.

        ``terminal_title`` is what herdr actually populates -- ``title`` and
        ``label`` are absent from every pane record observed at protocol 17.
        They are kept ahead of it because they are the explicit, user-set
        names when they do appear, whereas the terminal title is whatever the
        shell or agent last wrote.
        """
        return abbreviate(self.title or self.label or self.terminal_title or "")


@dataclass(frozen=True, slots=True)
class GroupKey:
    """An ordered column candidate, as herdr reports it."""

    id: str
    label: str


@dataclass(frozen=True, slots=True)
class Group:
    """A column: one group and the panes visible in it."""

    id: str
    label: str
    panes: tuple[Pane, ...] = ()


@dataclass(frozen=True, slots=True)
class Grid:
    """The deck's key geometry. Keys are numbered row-major from top-left."""

    rows: int
    columns: int

    @property
    def key_count(self) -> int:
        return self.rows * self.columns

    def index(self, row: int, column: int) -> int:
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise IndexError(f"({row}, {column}) outside {self.rows}x{self.columns}")
        return row * self.columns + column

    def position(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.key_count:
            raise IndexError(f"key {index} outside 0..{self.key_count - 1}")
        return divmod(index, self.columns)

    def pane_at(self, columns: Sequence[Group | None], index: int) -> Pane | None:
        """The pane occupying a key, or None for an empty key."""
        if not 0 <= index < self.key_count:
            return None
        row, column = divmod(index, self.columns)
        group = columns[column] if column < len(columns) else None
        if group is None or row >= len(group.panes):
            return None
        return group.panes[row]


def build_columns(
    panes: Sequence[Pane],
    order: Sequence[GroupKey],
    grid: Grid,
    mode: GroupingMode = GroupingMode.WORKSPACE,
    *,
    workspace_id: str = "",
) -> list[Group | None]:
    """Lay panes out into columns, mirroring herdr's order.

    ``panes`` must already be in herdr's order and ``order`` in sidebar order;
    both are preserved exactly. Anything past the edge of the grid is dropped
    and logged -- a deck with five columns cannot show a sixth workspace, and
    that should be visible in the log rather than silently invisible.

    In TAB mode ``workspace_id`` restricts the columns to one workspace's tabs,
    since a tab column would otherwise mean something different in each column.
    """
    buckets: dict[str, list[Pane]] = {}
    for pane in panes:
        if mode is GroupingMode.WORKSPACE:
            key = pane.workspace_id
        else:
            if workspace_id and pane.workspace_id != workspace_id:
                continue
            key = pane.tab_id
        if key:
            buckets.setdefault(key, []).append(pane)

    columns: list[Group | None] = [None] * grid.columns
    overflow_groups: list[str] = []
    truncated: list[str] = []

    for position, group_key in enumerate(order):
        members = buckets.get(group_key.id, [])
        if position >= grid.columns:
            if members:
                overflow_groups.append(group_key.label)
            continue
        if len(members) > grid.rows:
            truncated.append(f"{group_key.label} ({len(members)} panes)")
        columns[position] = Group(
            id=group_key.id,
            label=group_key.label,
            panes=tuple(members[: grid.rows]),
        )

    if overflow_groups:
        logger.info(
            "%d column(s) do not fit and are not shown: %s",
            len(overflow_groups),
            ", ".join(overflow_groups),
        )
    if truncated:
        logger.info("showing only the first %d panes of: %s", grid.rows, ", ".join(truncated))

    return columns


AGENT_KEY_LIMIT = 10
"""How many agents the flat layout shows.

Two full rows of a 3x5 deck, leaving the bottom row for action keys. Not a
technical ceiling -- the grid would hold fifteen -- but the point of the row is
that it stays free."""


def build_agent_columns(
    panes: Sequence[Pane],
    grid: Grid,
    limit: int = AGENT_KEY_LIMIT,
) -> list[Group | None]:
    """The most recently active agents, oldest first, filled row-major.

    Two different orderings, doing two different jobs. *Which* agents make the
    cut is recency -- the ones that did something most recently are the ones
    worth a key. *Where* they sit is creation order, oldest first, because a
    deck whose keys reshuffle every time an agent speaks is unusable: you
    reach for position, and position has to hold still while you do.

    Returns the same column structure the grouped modes return, so key lookup
    and drawing are shared. Each column here is a slice of the flat run rather
    than a workspace: with five columns, keys 0 and 5 are column 0's two rows.
    """
    capacity = min(limit, grid.key_count)
    # Ties on seq keep herdr's own pane order, which `sorted` guarantees.
    ranked = sorted(panes, key=lambda pane: pane.state_change_seq, reverse=True)
    shown = sorted(ranked[:capacity], key=lambda pane: pane.creation_key)

    dropped = len(panes) - len(shown)
    if dropped > 0:
        logger.info(
            "%d agent(s) beyond the %d most recent are not shown: %s",
            dropped,
            capacity,
            ", ".join(pane.pane_id for pane in ranked[capacity:]),
        )

    columns: list[Group | None] = [None] * grid.columns
    for position in range(grid.columns):
        members = tuple(shown[position :: grid.columns])
        columns[position] = (
            Group(id=f"agents:{position}", label="", panes=members) if members else None
        )
    return columns
