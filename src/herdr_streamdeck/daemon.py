"""Wires the herdr event stream to the button surface.

Model: one key per pane that has a detected agent, coloured by agent status.
Pressing a key focuses that pane.

Two threading notes drive the structure here:

* The Stream Deck library delivers key callbacks on its own reader thread, so
  presses are handed to the event loop with ``call_soon_threadsafe`` rather
  than touched directly.
* Agent status arrives **only** on ``pane.agent_status_changed``, which is
  pane-scoped. ``pane.updated`` carries an ``agent_status`` field in its
  payload but does not fire when status changes -- verified by driving
  transitions with ``pane.report_agent`` and watching both. Relying on the
  global event meant status only refreshed on the 60s reconcile.

  So a subscription is held per pane, rebuilt whenever the pane set changes.
  That needs a new connection each time (see HerdrSession.resubscribe), which
  is why it is done on restructure rather than per event.

Layout mirrors herdr rather than storing anything: columns follow
``workspace.list`` (sidebar order) and rows follow ``pane.list`` (a depth-first
walk of the tab's split tree). Events are split into *structural* ones, which
can reorder things and so trigger a re-read, and *cosmetic* ones, which only
change a pane in place. See ``STRUCTURAL_EVENTS``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import logging
import platform
import re
import signal
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Protocol

from .actions import DeckAction, default_config_path, load_actions, open_action
from .animation import EMPTY_ANIMATION, Animation, animation_for, frame_index
from .client import HerdrSession
from .deck import (
    EMPTY_BACKGROUND,
    RGB,
    ButtonFace,
    ButtonSurface,
    DeckDisconnected,
    KeyFrames,
    open_surface,
    plan_preview,
)
from .icons import mark_for, resolve_override
from .instance import AlreadyRunning, SingleInstance, lock_path, stop_running
from .layout import (
    AGENT_KEY_LIMIT,
    Grid,
    Group,
    GroupingMode,
    GroupKey,
    Pane,
    build_agent_columns,
    build_columns,
)
from .lock import POLL_SECONDS, LockWatcher, NoLock, lock_watcher
from .protocol import Event, HerdrError, JSONObject, subscription
from .summary import PaneSummary, Reply, Summariser
from .summary import build as build_summariser

logger = logging.getLogger("herdr_streamdeck")

# Events that can change *which* pane sits where, as opposed to merely
# restyling one. These trigger a re-read of herdr's ordering, since neither
# workspace order nor split-tree order can be derived from a pane record.
STRUCTURAL_EVENTS = frozenset(
    {
        "pane.created",
        "pane.closed",
        "pane.exited",
        "pane.moved",
        "pane.agent_detected",
        "tab.created",
        "tab.closed",
        "tab.moved",
        "workspace.created",
        "workspace.closed",
        "workspace.moved",
        "workspace.renamed",
        "workspace.reordered",
        "workspace.focused",
        "layout.updated",
    }
)

OPTIONAL_SUBSCRIPTIONS = ("workspace.reordered", "workspace.renamed")
"""Structural kinds that older servers reject outright.

herdr validates ``events.subscribe`` as a whole: one unrecognised kind fails
the entire call, so subscribing blindly to a kind the server has never heard of
leaves the daemon with *no* events -- a dead deck, not a degraded one. herdr
0.7.5 answers `unknown variant `workspace.reordered`` and subscribes to
nothing at all. So the server is asked first; see
HerdrSession.supports_subscription.

Version numbers are no help here: ``workspace.reordered`` was added while
PROTOCOL_VERSION was already 18 and shipped in 19, so the number does not
say whether a given server has it. Asking does.

``workspace.renamed`` is probed the same way for the same reason. It matters
because a space's name is what its keys are called: without it a rename waits
for the 60-second reconcile, which is exactly the delay that made a renamed
space look broken. It is classified structural so the rename triggers a re-read
of ``workspace.list``, which is where the new label lives.

It accompanies ``workspace.move_block``, herdr 0.8.0's atomic worktree-group
reorder and what a sidebar drag uses for a workspace inside a worktree group.
Unlike ``workspace.move`` it shifts nothing else, so without this a reorder
reaches us as nothing at all."""

COSMETIC_SUBSCRIPTIONS = ("pane.updated", "pane.focused", "tab.focused")
"""Restyle one key in place; no re-read needed."""

SUBSCRIPTIONS = tuple(
    sorted((STRUCTURAL_EVENTS - set(OPTIONAL_SUBSCRIPTIONS)) | set(COSMETIC_SUBSCRIPTIONS))
)
"""Derived, not listed, so the two cannot drift apart.

Classifying a kind as structural is worthless unless something subscribes to
it, and that gap is invisible: the deck simply catches up on the next reconcile
instead of at once. Two kinds sat in STRUCTURAL_EVENTS unsubscribed for a long
time this way. ``workspace.moved`` was hidden because moving a single workspace
also shifts the focused one, so ``workspace.focused`` arrived instead; and
closing a tab or workspace removes its panes **without emitting
``pane.closed``** -- measured, not assumed -- so the deck kept drawing keys for
panes that no longer existed."""

# Drawn as a dot in the top-right, so the key field itself stays neutral.
STATUS_COLORS: dict[str, RGB] = {
    "working": (217, 132, 24),  # amber -- busy
    "blocked": (204, 44, 44),  # red   -- needs input
    "done": (34, 168, 82),  # green -- finished, unseen
    "idle": (82, 82, 91),  # grey  -- waiting, seen
}


def worth_summarising(before: str, after: str) -> bool:
    """Whether a status transition is worth paying a model to explain.

    The trigger is **leaving `working`**, not arriving anywhere in particular.
    That is a correction: an earlier version keyed on arriving at `blocked` or
    `done`, and fired almost never. herdr's schema lists `done` in its
    `AgentStatus` enum, but a pane never reaches it -- driving a real agent
    through a complete turn emits exactly `working` then `idle`, 1.5s apart.

    Leaving `working` is also the better semantic anyway: it means "the agent
    stopped", which is when its last message became worth reading, whatever
    status it landed on. `blocked` is included from any state because arriving
    there is always worth explaining.
    """
    if before == after:
        return False
    return before == "working" or after == "blocked"


HOLD_SECONDS = 0.45
"""How long a key must be held before its reply options appear.

Long enough that a normal press-to-focus never trips it, short enough that the
options feel like part of the same gesture rather than a separate mode."""

MENU_BACKGROUND: RGB = (62, 64, 74)
"""The lit columns: choices on the left, controls on the right.

Brighter than anything else on the deck while the menu is open, so the two
columns you can actually press are obvious and the block between them reads as
a display rather than as nine more buttons."""

PREVIEW_BACKGROUND: RGB = (8, 8, 10)
"""Near black behind the preview, for the same reason."""
SELECTED_BORDER: RGB = (236, 236, 241)
ACCEPT_COLOR: RGB = (34, 168, 82)
BACK_COLOR: RGB = (120, 120, 130)

REPLY_COLORS: dict[str, RGB] = {
    "affirmative": (34, 168, 82),
    "negative": (204, 44, 44),
    "proceed": (217, 132, 24),
    "alternative": (96, 200, 240),
}

RECONNECT_SECONDS = 3.0
"""How often to look for a deck that went away.

Frequent enough that plugging it back in feels immediate, sparse enough that an
absent deck costs nothing -- enumeration is the only work, and it is cheap."""

HEARTBEAT_SECONDS = 5.0
"""How often to ask a deck we believe in whether it is still there.

Every other route to that answer costs a write, and ``tick`` writes only keys
whose level changed -- so a deck showing nothing but idle panes is never
written to, and an undock would otherwise go unnoticed until some key happened
to change. The probe itself is a locked attribute read; five seconds of it is
free next to a single key write."""

DEEP_PROBE_EVERY = 3
"""Beats between the pricier probe that re-enumerates the USB bus."""

ANIMATION_FPS = 20
"""Frame rate for pulsing and blinking.

A 15-key refresh measured 25 ms (1.34 ms a key), so 20 fps leaves ample
headroom even in the worst case where every key animates -- and writes are
skipped when the level is unchanged, so the usual cost is far lower."""

ACTIVATE_TIMEOUT = 2.0
TASK_CHECK_SECONDS = 60.0
"""How often to notice that a pane's task changed without an event saying so.

Not a refresh cadence: a pane whose task is unchanged costs nothing here. It
used to re-summarise every working pane on this timer, which is what made the
keys rewrite themselves while you were reading them."""

_TASK_FILLER = frozenset({"code", "for", "pr", "pull", "request", "review", "with"})


def working_label(title: str, limit: int = 24) -> str:
    """Turn a terminal title into an immediate, model-free task label."""
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", title.strip())
    pr = re.search(r"\bPR\s*#?(\d+)\b", cleaned, flags=re.IGNORECASE)
    prefix = f"PR {pr.group(1)}" if pr else ""
    without_number = re.sub(r"\bPR\s*#?\d+\b", "", cleaned, flags=re.IGNORECASE)
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", without_number)
        if word.lower() not in _TASK_FILLER
    ]
    candidates = ([prefix] if prefix else []) + words
    phrase = ""
    for word in candidates:
        proposed = f"{phrase} {word}".strip()
        if len(proposed) > limit:
            break
        phrase = proposed
    return phrase or cleaned[:limit].rstrip()


def activate_application(name: str) -> None:
    """Bring a named macOS application, and its Space, to the foreground."""
    subprocess.run(
        ("open", "-a", name),
        capture_output=True,
        text=True,
        timeout=ACTIVATE_TIMEOUT,
        check=True,
    )


@dataclass(frozen=True, slots=True)
class MenuLayout:
    """Which key means what while the reply menu is open."""

    options: dict[int, int]
    """Key index -> index into the reply list."""

    preview: tuple[int, ...]
    """Key indices holding the selected reply's text, in reading order."""

    accept: int
    back: int


def menu_layout(grid: Grid, count: int) -> MenuLayout:
    """Lay the menu out across the whole deck.

    Choices down the left, controls down the right, and the text you are about
    to send filling everything between. The held key is not preserved: once the
    menu is open the question is which reply, not which pane, and reserving a
    key to answer a question nobody is asking wastes the widest part of the
    display.
    """
    last = grid.columns - 1
    options = {row * grid.columns: row for row in range(min(count, grid.rows))}
    preview = tuple(
        row * grid.columns + column for row in range(grid.rows) for column in range(1, last)
    )
    return MenuLayout(
        options=options,
        preview=preview,
        accept=last,
        back=(grid.rows - 1) * grid.columns + last,
    )


@dataclass(frozen=True, slots=True)
class ReplyMenu:
    """An open menu of suggested replies for one pane."""

    pane_id: str
    replies: tuple[Reply, ...]
    selected: int = 0

    @property
    def choice(self) -> Reply | None:
        if 0 <= self.selected < len(self.replies):
            return self.replies[self.selected]
        return None


def option_face(reply: Reply, *, selected: bool) -> ButtonFace:
    return ButtonFace(
        summary=reply.label,
        status_color=REPLY_COLORS.get(reply.kind),
        background=MENU_BACKGROUND,
        border=SELECTED_BORDER if selected else None,
    )


def control_face(label: str, color: RGB) -> ButtonFace:
    return ButtonFace(summary=label, status_color=color, background=MENU_BACKGROUND)


class HerdrLike(Protocol):
    """The slice of the client the controller actually needs.

    Depending on this rather than the concrete client keeps the controller
    testable with a stub, without casts or type suppressions at the seam.
    """

    async def request(self, method: str, params: JSONObject | None = None) -> JSONObject: ...

    async def snapshot(self) -> JSONObject: ...

    async def subscribe(self, subscriptions: Sequence[JSONObject]) -> None: ...

    async def supports_subscription(self, kind: str) -> bool: ...

    async def resubscribe(self, subscriptions: Sequence[JSONObject]) -> None: ...

    def events(self) -> AsyncIterator[Event]: ...


class DeckController:
    """Keeps the button surface in sync with herdr's pane state."""

    def __init__(
        self,
        client: HerdrLike,
        surface: ButtonSurface,
        *,
        mode: GroupingMode = GroupingMode.AGENT,
        reconcile_interval: float = 60.0,
        agent_limit: int = AGENT_KEY_LIMIT,
        summariser: Summariser | None = None,
        lock: LockWatcher | None = None,
        lock_interval: float = POLL_SECONDS,
        activate: Callable[[], None] | None = None,
        actions: dict[int, DeckAction] | None = None,
        action_runner: Callable[[DeckAction], None] = open_action,
        task_check_interval: float = TASK_CHECK_SECONDS,
    ) -> None:
        self._client = client
        self._surface = surface
        self._mode = mode
        self._lock = lock or NoLock()
        self._lock_interval = lock_interval
        self._activate = activate
        self._actions = actions or {}
        self._action_runner = action_runner
        self._locked = False
        self._summariser = summariser
        self._summaries: dict[str, PaneSummary] = {}
        self._summary_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_check_interval = task_check_interval
        self._reconcile_interval = reconcile_interval
        self._agent_limit = agent_limit
        # Insertion order matters: it is herdr's pane order, and sorting it
        # would replace herdr's arrangement with ours.
        self._panes: dict[str, Pane] = {}
        self._order: list[GroupKey] = []
        self._columns: list[Group | None] = []
        self._focused_workspace = ""
        # workspace_id -> label, kept for every mode: the name a person gave a
        # space is what its keys are called, whether or not columns are spaces.
        self._workspace_labels: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dirty = asyncio.Event()
        self._restructure = False
        # Prebuffered frames and the animation driving each key.
        self._frames: dict[int, KeyFrames] = {}
        self._animations: dict[int, Animation] = {}
        self._shown: dict[int, int] = {}
        # One clock for the whole deck: phase must not depend on when a pane
        # started working, or keys pulse out of step with each other.
        self._epoch = 0.0
        # The reply menu, once a key has been held long enough to open it. It
        # stays until dismissed -- there is no timeout to race.
        self._menu: ReplyMenu | None = None
        self._holding: int | None = None
        self._hold_task: asyncio.Task[None] | None = None
        # False from the moment a write fails until the device is reacquired.
        self._connected = True
        self._reconnect_task: asyncio.Task[None] | None = None
        # Resolved once against the running server, which cannot change
        # version underneath a live connection. None until asked; then the
        # subset of OPTIONAL_SUBSCRIPTIONS this server actually recognises.
        self._optional_supported: frozenset[str] | None = None

    @property
    def grid(self) -> Grid:
        rows, columns = self._surface.key_layout
        return Grid(rows=rows, columns=columns)

    # ------------------------------------------------------------------- state

    def _upsert(self, record: JSONObject) -> bool:
        pane = Pane.from_record(record)
        if pane is None:
            return False
        existing = self._panes.get(pane.pane_id)
        if existing is not None and not pane.state_change_seq:
            # Only the snapshot's `agents` listing carries state_change_seq;
            # `pane.list` and every `pane.updated` payload omit it. Letting a
            # cosmetic update zero it would drop that pane to the back of the
            # recency ranking and shuffle the deck under your hand.
            pane = replace(pane, state_change_seq=existing.state_change_seq)
        if existing == pane:
            return False
        self._panes[pane.pane_id] = pane
        return True

    def _mark_active(self, pane_id: str) -> int:
        """Stamp a pane as the most recently active one.

        herdr only reports state_change_seq in a snapshot, so between the
        60-second reconciles the ranking would freeze and a just-woken agent
        would not climb into the visible ten until the next one. Continuing
        herdr's own counter locally keeps that live, and the next reconcile
        overwrites it with the server's authoritative value.
        """
        highest = max((pane.state_change_seq for pane in self._panes.values()), default=0)
        return highest + 1

    def _remove(self, pane_id: str) -> bool:
        self._cancel_summary(pane_id)
        self._summaries.pop(pane_id, None)
        menu = self._menu
        if menu is not None and menu.pane_id == pane_id:
            self._menu = None
        return self._panes.pop(pane_id, None) is not None

    def _rebuild_columns(self) -> None:
        """Recompute columns from the current model, preserving herdr's order."""
        panes = list(self._panes.values())
        if self._mode is GroupingMode.AGENT:
            self._columns = build_agent_columns(panes, self.grid, self._agent_limit)
            return
        self._columns = build_columns(
            panes,
            self._order,
            self.grid,
            self._mode,
            workspace_id=self._focused_workspace,
        )

    # ------------------------------------------------------------------ drawing

    def deliberate_space_name(self, pane: Pane) -> str:
        """The space's name, if a person chose it, otherwise empty.

        Renaming a space in herdr's sidebar is the rename people actually
        perform -- it is the one the UI offers -- and it writes
        ``workspace.label``. But that field is not blank by default: herdr
        seeds it with the checkout's directory name, so six spaces opened on
        the same repo all read ``v2-dos``, which identifies nothing. Taking the
        label unconditionally would replace nine useful auto-titles with nine
        copies of a repo name.

        A rename is detectable precisely because it stops matching the
        directory. Measured across a twelve-space session, this picks out
        exactly the three spaces that were renamed by hand and rejects all
        nine repo-derived ones.
        """
        label = self._workspace_labels.get(pane.workspace_id, "").strip()
        if not label or not pane.cwd:
            return ""
        return "" if label == PurePath(pane.cwd).name else label

    def name_for(self, pane: Pane) -> str:
        """What a key calls its pane, most deliberate name first.

        A name someone chose says what the pane is *for*, and nothing inferred
        beats it, so it holds even while the agent works and the model never
        displaces it. The space name leads because renaming a space is the
        gesture herdr's sidebar actually offers; ``pane.rename`` exists but is
        a different, rarely-used command, and having it outrank the sidebar
        meant a space you had just renamed kept showing an old pane label.
        Failing both, the model's title says what the pane is doing right now,
        and failing that the terminal title -- whatever the agent last wrote,
        usually serviceable and always better than a blank key.
        """
        summary = self._summaries.get(pane.pane_id)
        return (
            self.deliberate_space_name(pane)
            or pane.given_name
            or (summary.display if summary else "")
            or pane.terminal_title
        )

    def face_for(self, pane: Pane | None) -> ButtonFace:
        """The face for one key: the name in the body, the mark in a corner.

        Weighted this way because every key in a column carries the *same*
        agent glyph, so between them the glyph distinguishes nothing, while the
        name is the only thing that says which pane this is. The mark keeps a
        chip in the top-left, which is enough to tell agents apart in a column
        that mixes them.
        """
        if pane is None:
            return ButtonFace(background=EMPTY_BACKGROUND)
        mark = mark_for(pane.mark_key)
        return ButtonFace(
            mark=mark.glyph,
            mark_color=mark.color,
            mark_scale=mark.scale,
            # A user PNG in the plugin config dir replaces the glyph.
            icon=resolve_override(pane.mark_key),
            title=self.name_for(pane),
            title_style=True,
            status_color=STATUS_COLORS.get(pane.status),
        )

    def repaint(self) -> None:
        """Re-render keys whose content changed, then show the current frame.

        Rendering is the expensive part (~1.26 ms a key), so a face that has
        not changed keeps its existing prebuffered frames even when its
        animation is running.
        """
        self._rebuild_columns()
        if not self._connected or self._locked:
            # The model stays current so the deck is right the moment it
            # returns; there is just nothing to draw on.
            return
        grid = self.grid

        menu = self._menu
        menu_faces = self._menu_faces(menu, grid) if menu is not None else {}
        for index in range(self._surface.key_count):
            if index in menu_faces:
                face = menu_faces[index]
                # Steady and full: keys you are choosing between must not pulse.
                animation = EMPTY_ANIMATION
            elif index in self._actions:
                face = self._actions[index].face
                animation = EMPTY_ANIMATION
            else:
                pane = grid.pane_at(self._columns, index)
                face = self.face_for(pane)
                animation = animation_for(pane.status) if pane else EMPTY_ANIMATION
            self._animations[index] = animation

            cached = self._frames.get(index)
            if cached is None or cached.face != face:
                try:
                    self._frames[index] = self._surface.render(face)
                except DeckDisconnected:
                    self._note_disconnect()
                    return
                except Exception:
                    logger.warning("could not render key %d", index, exc_info=True)
                    continue
                # Force a write: the cached level now refers to old content.
                self._shown.pop(index, None)

        self.tick()

    def tick(self, now: float | None = None) -> int:
        """Advance every key to its frame for the current instant.

        Writes only where the level actually changed -- a USB write measured
        1.34 ms, so a full 15-key refresh is 25 ms and blind rewriting would
        cap the deck at ~40 fps for no benefit. Returns the number of writes.
        """
        if not self._connected or self._locked:
            return 0
        if now is None:
            loop = self._loop
            now = loop.time() if loop is not None else 0.0
        elapsed = now - self._epoch
        levels = self._surface.levels
        writes = 0

        for index, frames in self._frames.items():
            animation = self._animations.get(index, EMPTY_ANIMATION)
            level = frame_index(animation, elapsed, levels)
            if self._shown.get(index) == level:
                continue
            try:
                self._surface.write(index, frames, level)
            except DeckDisconnected:
                self._note_disconnect()
                return writes
            except Exception:
                logger.warning("could not write key %d", index, exc_info=True)
                continue
            self._shown[index] = level
            writes += 1
        return writes

    # ---------------------------------------------------------------- summaries

    def _request_summary(self, pane_id: str, *, expected_status: str) -> None:
        """Kick off a summary without blocking the caller.

        Fire-and-forget on purpose: the key repaints immediately with its status
        animation, and the words arrive a beat later if they arrive at all. The
        deck never waits on the network to draw.
        """
        if self._summariser is None:
            return
        existing = self._summary_tasks.get(pane_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._summarise(pane_id, expected_status=expected_status),
            name=f"summary-{pane_id}",
        )
        self._summary_tasks[pane_id] = task

        def finished(done: asyncio.Task[None]) -> None:
            if self._summary_tasks.get(pane_id) is done:
                self._summary_tasks.pop(pane_id, None)

        task.add_done_callback(finished)

    def _cancel_summary(self, pane_id: str) -> None:
        task = self._summary_tasks.pop(pane_id, None)
        if task is not None:
            task.cancel()

    async def _summarise(self, pane_id: str, *, expected_status: str) -> None:
        summariser = self._summariser
        if summariser is None:
            return
        try:
            response = await self._client.request(
                "pane.read", {"pane_id": pane_id, "source": "recent", "lines": 60}
            )
        except HerdrError as exc:
            logger.debug("could not read %s for a summary: %s", pane_id, exc)
            return
        except Exception:
            logger.warning("could not read %s for a summary", pane_id, exc_info=True)
            return

        read = response.get("read")
        text = read.get("text") if isinstance(read, dict) else None
        if not isinstance(text, str):
            return

        summary = await summariser.summarise(text)
        if summary is None:
            return
        # A periodic working summary must never land after the agent stops and
        # overwrite its completion summary with stale progress.
        pane = self._panes.get(pane_id)
        if pane is None or pane.status != expected_status:
            return
        if expected_status == "working":
            # Suggestions are useful once an agent stops, not while it is busy.
            summary = PaneSummary(phrase=summary.phrase, waiting=False)
        summary = replace(summary, subject=self._subject_of(pane))

        # The name holds still; only the replies move. Re-reading a thread
        # mid-task yields a differently-worded name for the same task, and a
        # key whose words change every minute cannot be used to find anything
        # -- you re-read all ten instead of reaching for the one you know.
        previous = self._summaries.get(pane_id)
        if (
            previous is not None
            and previous.phrase
            and not previous.provisional
            and previous.subject == summary.subject
        ):
            summary = replace(summary, phrase=previous.phrase)
        self._summaries[pane_id] = summary
        logger.info(
            "summary for %s: %s%s",
            pane_id,
            summary.display,
            f"  (+{len(summary.replies)} replies)" if summary.replies else "",
        )
        self._dirty.set()

    @staticmethod
    def _subject_of(pane: Pane) -> str:
        """What the thread is working on, as far as the pane can say.

        The agent writes its own task into the terminal title and rewrites it
        when given a new one, so this is the available signal for "the task
        changed" as distinct from "the agent did something".
        """
        return " ".join((pane.given_name or pane.terminal_title).split()).casefold()

    def _resummarise_if_the_task_changed(self) -> None:
        """Re-read a pane only when what it is working on has changed.

        This replaces a timer that re-summarised every working pane once a
        minute. That timer was the thing rewriting the keys under you: it
        asked a model what was happening *now*, so the words tracked activity
        by construction, and paid for a model call per working pane per minute
        to do it.
        """
        for pane in self._panes.values():
            summary = self._summaries.get(pane.pane_id)
            subject = self._subject_of(pane)
            if summary is None or summary.provisional:
                # No name yet, or only the model-free stand-in. Note this does
                # not require a subject: a pane whose title is empty still
                # deserves a name, it just cannot be told when to renew it.
                self._request_summary(pane.pane_id, expected_status=pane.status)
            elif subject and summary.subject and summary.subject != subject:
                self._request_summary(pane.pane_id, expected_status=pane.status)

    def _seed_working_labels(self) -> None:
        """Give active panes useful words before the first model refresh."""
        for pane in self._panes.values():
            if pane.status != "working" or pane.pane_id in self._summaries:
                continue
            label = working_label(pane.terminal_title)
            if label:
                self._summaries[pane.pane_id] = PaneSummary(
                    phrase=label,
                    waiting=False,
                    provisional=True,
                    subject=self._subject_of(pane),
                )

    def replies_for(self, pane_id: str) -> tuple[Reply, ...]:
        """Suggested replies for a pane, if any were offered."""
        summary = self._summaries.get(pane_id)
        return summary.replies if summary else ()

    # ------------------------------------------------------------- connection

    def _note_disconnect(self) -> None:
        """Say it once, then stop talking to a device that is not there.

        Every path that touches the deck funnels through here, so the first
        failure is reported and the next few thousand are not.
        """
        if not self._connected:
            return
        self._connected = False
        self._menu = None
        logger.warning("deck disconnected")
        loop = self._loop
        if loop is None:
            return
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop(), name="reconnect")

    async def _heartbeat_loop(self) -> None:
        """Ask the deck whether it is still there, since nothing else will.

        A disconnect is otherwise discovered only by a write failing, and an
        idle deck is written to never: dock and undock while every pane is
        quiet and the deck comes back on its power-on logo, with the daemon
        still believing it drew the right thing. Probing costs a locked
        attribute read, so this can run far more often than it needs to.
        """
        beat = 0
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            if not self._connected:
                # Already known to be gone; _reconnect_loop owns it from here.
                continue
            beat += 1
            try:
                alive = self._surface.alive(deep=beat % DEEP_PROBE_EVERY == 0)
            except Exception:
                logger.debug("heartbeat probe failed", exc_info=True)
                continue
            if not alive:
                logger.warning("deck went away quietly; noticed by heartbeat")
                self._note_disconnect()

    async def _reconnect_loop(self) -> None:
        """Watch for the deck coming back, and redraw it when it does."""
        while not self._connected:
            await asyncio.sleep(RECONNECT_SECONDS)
            try:
                back = self._surface.reopen()
            except Exception:
                logger.debug("reopen failed", exc_info=True)
                back = False
            if not back:
                continue
            logger.info("deck reconnected")
            self._connected = True
            # The frames were encoded against the previous device handle, and
            # its key format is only *probably* the same one. Re-render rather
            # than trust that.
            self._frames.clear()
            self._shown.clear()
            self._surface.set_press_handler(self._on_press)
            # A deck that comes back mid-lock comes back at full brightness
            # showing whatever open() drew, so the lock has to be reapplied
            # rather than waiting for the next transition -- there may not be
            # one until the screen is unlocked.
            if self._locked:
                self._blank()
            else:
                self.repaint()

    # ------------------------------------------------------------------ locking

    async def _lock_loop(self) -> None:
        """Follow the screen lock, keeping the deck dark for as long as it holds."""
        while True:
            await asyncio.sleep(self._lock_interval)
            try:
                locked = await asyncio.to_thread(self._lock.locked)
            except Exception:
                # ScreenLock decides for itself what an unreadable state means;
                # anything that still escapes is a bug in the watcher, and
                # tearing down the deck over it would be the wrong answer.
                logger.debug("lock check failed", exc_info=True)
                continue
            if locked != self._locked:
                self.set_locked(locked)

    def set_locked(self, locked: bool) -> None:
        """Take the deck dark, or bring it back."""
        self._locked = locked
        if locked:
            logger.info("screen locked; blanking the deck")
            # You walked away mid-choice. Coming back to a half-made decision
            # still sitting on the keys invites finishing it by reflex.
            self._menu = None
            self._holding = None
            if self._hold_task is not None:
                self._hold_task.cancel()
                self._hold_task = None
            self._blank()
        else:
            logger.info("screen unlocked; restoring the deck")
            self._restore()

    def _blank(self) -> None:
        """Black out every key, then kill the backlight.

        In that order, and the order is the whole point. Dimming alone leaves
        the summaries sitting in the deck's own key buffers, where anything
        that puts the brightness back -- another program, a crash and a fresh
        daemon, the deck's power-on default -- puts them back on screen too.
        Writing black first means there is nothing left to reveal.
        """
        if not self._connected:
            return
        try:
            frames = self._surface.render(ButtonFace(background=EMPTY_BACKGROUND))
            for index in range(self._surface.key_count):
                self._surface.write(index, frames, self._surface.levels - 1)
            self._surface.set_brightness(0)
        except DeckDisconnected:
            self._note_disconnect()
            return
        except Exception:
            logger.warning("could not blank the deck", exc_info=True)
            return
        # Every key now shows something other than what these levels claim, so
        # the next tick has to write rather than skip.
        self._shown.clear()

    def _restore(self) -> None:
        if not self._connected:
            return
        try:
            self._surface.set_brightness(self._surface.brightness)
        except DeckDisconnected:
            self._note_disconnect()
            return
        self.repaint()

    # ------------------------------------------------------------------ presses

    def _on_press(self, index: int, pressed: bool) -> None:
        """Invoked on the deck's reader thread -- hop to the event loop."""
        if self._locked:
            # A dark deck is still a live one. Without this, a bag set down on
            # it while you are away sends a canned reply into a running agent.
            return
        loop = self._loop
        if loop is None:
            # Only reachable if a press arrives before run() starts. Logged
            # rather than dropped silently: "buttons do nothing" is otherwise
            # indistinguishable from the handler never being installed.
            logger.warning("key %d pressed before the controller was running", index)
            return
        logger.debug("key %d %s", index, "down" if pressed else "up")
        loop.call_soon_threadsafe(self._key_down if pressed else self._key_up, index)

    def _key_down(self, index: int) -> None:
        menu = self._menu
        if menu is not None:
            self._menu_press(menu, index)
            return

        if index in self._actions:
            self._holding = index
            return

        pane = self.grid.pane_at(self._columns, index)
        if pane is None:
            logger.debug("key %d pressed but maps to no pane", index)
            return
        self._holding = index
        self._hold_task = asyncio.create_task(
            self._hold(index, pane.pane_id), name=f"hold-{index}"
        )

    def _menu_press(self, menu: ReplyMenu, index: int) -> None:
        layout = menu_layout(self.grid, len(menu.replies))
        if index in layout.options:
            chosen = layout.options[index]
            if chosen != menu.selected:
                self._menu = replace(menu, selected=chosen)
                self.repaint()
            return
        if index == layout.accept:
            reply = menu.choice
            if reply is not None:
                self._send_reply(menu.pane_id, reply)
            self._close_menu()
            return
        if index == layout.back:
            self._close_menu()
            return
        # The preview is a display, not a control. Pressing it does nothing --
        # it is the text you are about to send, and nudging it should not send.

    def _key_up(self, index: int) -> None:
        if self._menu is not None:
            # Releasing the key that opened the menu must not choose anything;
            # the menu stays until Send or Back.
            self._holding = None
            return
        if self._holding != index:
            return
        self._holding = None
        if self._hold_task is not None:
            self._hold_task.cancel()
            self._hold_task = None
        action = self._actions.get(index)
        if action is not None:
            logger.info("key %d tapped -> %s", index, action.label)
            task = asyncio.create_task(self._run_action(action), name=f"action-{index}")
            task.add_done_callback(lambda _: None)
            return
        # A tap, not a hold. Focus happens on release so the two gestures stay
        # distinguishable -- focusing on press would fire before a hold could
        # be recognised, and every hold would drag you into the pane.
        pane = self.grid.pane_at(self._columns, index)
        if pane is None:
            return
        logger.info("key %d tapped -> focusing %s", index, pane.pane_id)
        task = asyncio.create_task(self._focus(pane.pane_id), name=f"focus-{pane.pane_id}")
        task.add_done_callback(lambda _: None)

    async def _run_action(self, action: DeckAction) -> None:
        try:
            await asyncio.to_thread(self._action_runner, action)
        except Exception:
            logger.warning("action %r failed", action.label, exc_info=True)

    async def _hold(self, index: int, pane_id: str) -> None:
        """Open the reply menu if the key stays down long enough."""
        await asyncio.sleep(HOLD_SECONDS)
        if self._holding != index:
            return
        replies = self.replies_for(pane_id)
        if not replies:
            logger.debug("held key %d but %s has no replies to offer", index, pane_id)
            return
        self._menu = ReplyMenu(pane_id=pane_id, replies=replies)
        logger.info(
            "key %d held -> reply menu for %s (%d options)", index, pane_id, len(replies)
        )
        self.repaint()

    def _menu_faces(self, menu: ReplyMenu, grid: Grid) -> dict[int, ButtonFace]:
        """Every key the menu occupies, which is all of them."""
        layout = menu_layout(grid, len(menu.replies))
        faces: dict[int, ButtonFace] = {}
        for key, option in layout.options.items():
            faces[key] = option_face(menu.replies[option], selected=option == menu.selected)
        faces[layout.accept] = control_face("Send", ACCEPT_COLOR)
        faces[layout.back] = control_face("Back", BACK_COLOR)

        reply = menu.choice
        cells, size = plan_preview(
            reply.text if reply else "",
            grid.rows,
            max(0, grid.columns - 2),
            self._surface.key_size,
        )
        for key, chunk in zip(layout.preview, cells, strict=True):
            faces[key] = ButtonFace(
                summary=chunk,
                summary_size=size,
                background=PREVIEW_BACKGROUND,
            )
        # Whatever the menu does not name stays dark rather than showing a pane
        # that pressing it would not focus.
        for index in range(self._surface.key_count):
            faces.setdefault(index, ButtonFace(background=EMPTY_BACKGROUND))
        return faces

    def _close_menu(self) -> None:
        if self._menu is None:
            return
        self._menu = None
        self._holding = None
        if self._hold_task is not None:
            self._hold_task.cancel()
            self._hold_task = None
        self.repaint()

    def _send_reply(self, pane_id: str, reply: Reply) -> None:
        logger.info("sending %r to %s: %s", reply.label, pane_id, reply.text)
        task = asyncio.create_task(self._prompt(pane_id, reply.text), name=f"reply-{pane_id}")
        task.add_done_callback(lambda _: None)

    async def _prompt(self, pane_id: str, text: str) -> None:
        try:
            await self._client.request("agent.prompt", {"target": pane_id, "text": text})
        except HerdrError as exc:
            logger.warning("could not send a reply to %s: %s", pane_id, exc)
        except Exception:
            logger.warning("could not send a reply to %s", pane_id, exc_info=True)

    async def _focus(self, pane_id: str) -> None:
        try:
            await self._client.request("pane.focus", {"pane_id": pane_id})
        except HerdrError as exc:
            if exc.code == "pane_not_found":
                # Our model outlived the pane. Drop it and repaint rather than
                # leaving a key that fails every press.
                logger.info("pane %s is gone; dropping it", pane_id)
                if self._remove(pane_id):
                    self._dirty.set()
                return
            logger.warning("failed to focus %s: %s", pane_id, exc)
        except Exception:
            logger.warning("failed to focus %s", pane_id, exc_info=True)
        else:
            if self._activate is None:
                return
            try:
                await asyncio.to_thread(self._activate)
            except Exception:
                logger.warning(
                    "focused %s but failed to activate the configured app",
                    pane_id,
                    exc_info=True,
                )

    # --------------------------------------------------------------------- run

    async def drain_replay(self, quiet: float = 0.4, limit: float = 5.0) -> int:
        """Swallow the backlog herdr replays when a subscription starts.

        Subscribing does not begin a live-only stream: herdr immediately
        replays historical events, and **not in causal order** -- a
        ``pane.closed`` for a pane can arrive before its ``pane.created``.
        Applying that backlog resurrects long-dead panes, whose keys then fail
        every press.

        So the backlog is discarded and the snapshot taken afterwards is
        treated as the truth. Returns once no event has arrived for ``quiet``
        seconds, or after ``limit`` seconds regardless.
        """
        discarded = 0
        deadline = asyncio.get_running_loop().time() + limit
        events = self._client.events()

        while asyncio.get_running_loop().time() < deadline:
            try:
                await asyncio.wait_for(events.__anext__(), timeout=quiet)
            except TimeoutError:  # asyncio.TimeoutError is an alias on 3.11+
                break
            except StopAsyncIteration:
                break
            discarded += 1

        if discarded:
            logger.debug("discarded %d replayed events", discarded)
        return discarded

    async def read_order(self) -> tuple[list[GroupKey], str]:
        """Ask herdr for its column order, and which workspace is focused.

        Neither is derivable from a pane record: sidebar order lives only in
        ``workspace.list`` (the order ``workspace.move`` rearranges), and tab
        order only in ``tab.list``.
        """
        listing = await self._client.request("workspace.list")
        workspaces = listing.get("workspaces")
        focused = ""
        order: list[GroupKey] = []

        if isinstance(workspaces, list):
            for item in workspaces:
                if not isinstance(item, dict):
                    continue
                ws_id = item.get("workspace_id")
                if not isinstance(ws_id, str):
                    continue
                if item.get("focused"):
                    focused = ws_id
                label = item.get("label")
                # Collected in every mode, not just WORKSPACE: the space's name
                # names its keys even when columns are not spaces.
                self._workspace_labels[ws_id] = label if isinstance(label, str) else ""
                if self._mode is GroupingMode.WORKSPACE:
                    order.append(
                        GroupKey(id=ws_id, label=label if isinstance(label, str) else ws_id)
                    )

        if self._mode is GroupingMode.TAB:
            tabs = (await self._client.request("tab.list", {"workspace_id": focused})).get(
                "tabs"
            )
            if isinstance(tabs, list):
                for item in tabs:
                    if not isinstance(item, dict):
                        continue
                    tab_id = item.get("tab_id")
                    if not isinstance(tab_id, str):
                        continue
                    label = item.get("label")
                    order.append(
                        GroupKey(id=tab_id, label=label if isinstance(label, str) else tab_id)
                    )

        return order, focused

    def _subscription_set(self, optional: frozenset[str] | None = None) -> list[JSONObject]:
        """Global subscriptions plus one status subscription per live pane.

        ``optional`` is the subset of OPTIONAL_SUBSCRIPTIONS to include; None
        means all of them, which is the right assumption before asking.
        """
        extra = OPTIONAL_SUBSCRIPTIONS if optional is None else optional
        kinds = (*SUBSCRIPTIONS, *(k for k in OPTIONAL_SUBSCRIPTIONS if k in extra))
        subs = [subscription(kind) for kind in kinds]
        subs.extend(
            subscription("pane.agent_status_changed", pane_id=pane_id)
            for pane_id in self._panes
        )
        return subs

    async def _install_subscriptions(
        self, send: Callable[[Sequence[JSONObject]], Awaitable[None]]
    ) -> None:
        """Subscribe, leaving out the kinds this server does not recognise.

        Asked before subscribing rather than discovered by failing: an unknown
        kind rejects the whole set, and the connection is spent by the attempt,
        so there is nothing left to retry on. See
        HerdrSession.supports_subscription.
        """
        if self._optional_supported is None:
            supported: set[str] = set()
            for kind in OPTIONAL_SUBSCRIPTIONS:
                # Each kind is asked about separately. They shipped in
                # different herdr releases, so one being unknown says nothing
                # about the other -- treating them as a single yes/no dropped
                # both whenever the older server lacked either.
                if await self._client.supports_subscription(kind):
                    supported.add(kind)
                else:
                    logger.info(
                        "this herdr does not know %s, so it is left out; "
                        "what it reports will reach the deck on the next "
                        "reconcile rather than at once",
                        kind,
                    )
            self._optional_supported = frozenset(supported)
        await send(self._subscription_set(self._optional_supported))

    async def run_subscriptions(self) -> None:
        """Establish the initial subscription set (globals plus per pane)."""
        await self.prime()
        await self._install_subscriptions(self._client.subscribe)

    async def prime(self) -> None:
        """Re-read herdr's arrangement. Snapshot and listing are the truth.

        The pane map is rebuilt rather than patched, so its iteration order is
        exactly the snapshot's -- which is herdr's split-tree order. Patching
        would leave panes wherever they were first seen.
        """
        order, focused = await self.read_order()
        snapshot = await self._client.snapshot()

        before = set(self._panes)
        self._panes = {}
        for record in _iter_panes(snapshot):
            self._upsert(record)
        self._order = order
        self._focused_workspace = focused
        self._seed_working_labels()

        # Status subscriptions are per pane, so the set has to follow the panes.
        if set(self._panes) != before:
            await self._install_subscriptions(self._client.resubscribe)

        self.repaint()

    def handle(self, event: Event) -> None:
        """Fold one event into the model.

        Structural events only *schedule* a re-read: ordering cannot be
        recovered from the event payload, so the model is rebuilt from herdr
        rather than guessed at here.
        """
        if event.kind in STRUCTURAL_EVENTS:
            self._restructure = True
            self._dirty.set()
            return

        if event.kind == "pane.agent_status_changed":
            # The only event that fires on a status transition.
            pane_id = event.data.get("pane_id")
            status = event.data.get("agent_status")
            if isinstance(pane_id, str) and isinstance(status, str):
                existing = self._panes.get(pane_id)
                if existing is not None and existing.status != status:
                    self._cancel_summary(pane_id)
                    self._panes[pane_id] = replace(
                        existing,
                        status=status,
                        state_change_seq=self._mark_active(pane_id),
                    )
                    if status == "working":
                        # The name describes the task, and starting work does
                        # not change the task -- so it survives. Only a pane
                        # with no name yet gets the model-free stand-in, to
                        # carry it until the first summary lands.
                        if pane_id not in self._summaries:
                            label = working_label(existing.terminal_title)
                            if label:
                                self._summaries[pane_id] = PaneSummary(
                                    phrase=label,
                                    waiting=False,
                                    provisional=True,
                                    subject=self._subject_of(existing),
                                )
                        else:
                            # Replies answered a message that is now answered.
                            self._summaries[pane_id] = replace(
                                self._summaries[pane_id], waiting=False, replies=()
                            )
                        menu = self._menu
                        if menu is not None and menu.pane_id == pane_id:
                            self._menu = None
                    if worth_summarising(existing.status, status):
                        self._request_summary(pane_id, expected_status=status)
                    self._dirty.set()
            return

        if event.kind == "pane.updated":
            record = event.data.get("pane")
            if isinstance(record, dict) and self._upsert(record):
                self._dirty.set()

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._epoch = self._loop.time()
        self._surface.set_press_handler(self._on_press)
        # Stated explicitly at startup: presses only work once run() installs
        # this, so a script that drives the controller directly has a live
        # display and dead keys. The log should make that obvious.
        logger.info("press handler installed; keys are live")

        # Order matters: subscribe first so no live change is missed, then
        # throw away the replayed backlog, then snapshot. Snapshotting before
        # the backlog drains would just be overwritten by stale events.
        await self._install_subscriptions(self._client.subscribe)
        await self.drain_replay()
        await self.prime()

        painter = asyncio.create_task(self._paint_loop(), name="painter")
        animator = asyncio.create_task(self._animate_loop(), name="animator")
        reconciler = asyncio.create_task(self._reconcile_loop(), name="reconciler")
        locker = asyncio.create_task(self._lock_loop(), name="locker")
        heartbeat = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
        working_summaries = (
            asyncio.create_task(self._task_change_loop(), name="task-changes")
            if self._summariser is not None and self._task_check_interval > 0
            else None
        )
        try:
            async for event in self._client.events():
                self.handle(event)
            # The stream ended, so the server is gone. Returning here (rather
            # than reconnecting) is deliberate: herdr does NOT reap plugin
            # startup processes -- probes outlived their server, reparented to
            # init. A daemon that retried forever would survive a herdr restart
            # still holding the Stream Deck, and lock out its own replacement,
            # since hidapi opens the device exclusively. Exiting makes this
            # self-reaping.
            logger.info("event stream closed; herdr is gone, shutting down")
        finally:
            for task in (
                painter,
                animator,
                reconciler,
                locker,
                heartbeat,
                working_summaries,
                self._reconnect_task,
                *self._summary_tasks.values(),
            ):
                if task is None:
                    continue
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._surface.set_press_handler(None)

    async def _animate_loop(self) -> None:
        """Drive pulsing and blinking from the shared clock.

        Sleeps to the next frame boundary rather than a fixed interval, so
        phase does not drift when a tick runs long -- drift would gradually
        desynchronise keys, which is the one thing this must not do.
        """
        loop = asyncio.get_running_loop()
        interval = 1.0 / ANIMATION_FPS
        while True:
            if any(a.animated for a in self._animations.values()):
                self.tick(loop.time())
            # Align to the grid so ticks stay in phase with the epoch.
            now = loop.time()
            await asyncio.sleep(max(0.0, interval - ((now - self._epoch) % interval)))

    async def _reconcile_loop(self) -> None:
        """Re-snapshot periodically so the model cannot drift indefinitely.

        A safety net, not the main mechanism: events keep the deck responsive,
        this keeps it honest if one is ever missed or dropped under load.
        """
        while True:
            await asyncio.sleep(self._reconcile_interval)
            try:
                await self.prime()
            except Exception:
                logger.warning("reconcile failed", exc_info=True)

    async def _task_change_loop(self) -> None:
        """Catch a task change that arrived without an event to announce it."""
        while True:
            await asyncio.sleep(self._task_check_interval)
            self._resummarise_if_the_task_changed()

    async def _paint_loop(self) -> None:
        """Coalesce bursts of events into at most one update per interval."""
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            if self._restructure:
                self._restructure = False
                try:
                    await self.prime()
                except Exception:
                    logger.warning("could not re-read layout", exc_info=True)
            else:
                self.repaint()
            await asyncio.sleep(0.05)


def _iter_panes(snapshot: JSONObject) -> list[JSONObject]:
    """Pull pane records out of a session.snapshot response.

    The snapshot nests panes under workspaces and tabs; the exact shape has
    changed across protocol versions, so this walks defensively rather than
    indexing a fixed path.
    """
    # Keyed by pane_id because the snapshot lists the same pane under both
    # `agents` and `panes`; without this every agent pane is visited twice.
    found: dict[str, JSONObject] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            pane_id = node.get("pane_id")
            if isinstance(pane_id, str) and "terminal_id" in node:
                # Merge rather than pick a winner. The two listings carry
                # *different* fields, not more and less of the same: only
                # `agents` has state_change_seq, only `panes` has scroll, and
                # at protocol 19 both records are exactly fourteen keys wide --
                # so choosing by size silently dropped whichever arrived
                # second, and which one that is depends on key order.
                existing = found.get(pane_id)
                if existing is None:
                    found[pane_id] = dict(node)
                else:
                    existing.update({k: v for k, v in node.items() if v is not None})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(snapshot)
    return list(found.values())


def probe_devices() -> int:
    """List attached Stream Decks and exit.

    Cheap enough to run fork-per-invoke, so it is exposed as a plugin action --
    it answers "does this machine see the device at all?", which is the first
    question whenever the deck goes dark.
    """
    try:
        from StreamDeck.DeviceManager import DeviceManager
    except ImportError as exc:
        print(f"streamdeck package unavailable: {exc}")
        return 1

    try:
        decks = DeviceManager().enumerate()
    except Exception as exc:
        print(f"enumeration failed: {exc}")
        print("On Linux check the udev rule; under WSL check that usbipd has attached it.")
        return 1

    if not decks:
        print("no Stream Deck found")
        print("macOS: quit the Elgato app, which claims the device exclusively.")
        print("Linux/WSL: check /dev/hidraw* ownership and the usbipd attachment.")
        return 1

    for deck in decks:
        deck.open()
        try:
            print(
                f"{deck.deck_type()}  serial={deck.get_serial_number()}  "
                f"keys={deck.key_count()}"
            )
        finally:
            deck.close()
    return 0


async def amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.probe:
        return probe_devices()

    if args.activate_app and platform.system() != "Darwin":
        logger.error("--activate-app is only supported on macOS")
        return 2

    if args.stop:
        stopped = stop_running(args.serial)
        print(f"stopped pid {stopped}" if stopped else "no daemon was running")
        return 0

    instance = SingleInstance(lock_path(args.serial))
    try:
        displaced = instance.acquire(takeover=not args.no_takeover)
    except AlreadyRunning as exc:
        logger.error("%s", exc)
        return 1
    if displaced is not None:
        logger.info("took the deck over from pid %d", displaced)

    surface = open_surface(
        use_device=not args.no_device,
        serial=args.serial,
        virtual=args.virtual,
    )

    # One reply per row: the overlay is a single column, so asking for more
    # would generate options the deck cannot show.
    rows = surface.key_layout[0] or 3
    summariser = None if args.no_summaries else build_summariser(max_replies=rows)
    if summariser is None and not args.no_summaries:
        logger.info("no OPENAI_API_KEY found; running without pane summaries")

    try:
        surface.open()
    except Exception:
        instance.release()
        raise

    actions_path = Path(args.actions_config) if args.actions_config else default_config_path()
    try:
        actions = load_actions(actions_path, key_count=surface.key_count)
    except ValueError as exc:
        surface.close()
        instance.release()
        logger.error("%s", exc)
        return 2
    if actions:
        logger.info("loaded %d custom action(s) from %s", len(actions), actions_path)

    # Two connections -- herdr resets a connection that both subscribes and
    # issues requests. See HerdrSession.
    client = HerdrSession(Path(args.socket) if args.socket else None)
    await client.connect()

    controller = DeckController(
        client,
        surface,
        mode=GroupingMode(args.mode),
        agent_limit=args.agent_limit,
        summariser=summariser,
        lock=lock_watcher(enabled=not args.no_screen_lock),
        activate=(
            functools.partial(activate_application, args.activate_app)
            if args.activate_app
            else None
        ),
        actions=actions,
        task_check_interval=args.task_check_seconds,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    runner = asyncio.create_task(controller.run(), name="controller")
    waiter = asyncio.create_task(stop.wait(), name="stop")
    try:
        await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        runner.cancel()
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        await client.close()
        surface.close()
        instance.release()

    if client.dropped_events:
        logger.warning("dropped %d events while repainting", client.dropped_events)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="herdr-streamdeck")
    parser.add_argument("--socket", help="path to herdr.sock (default: $HERDR_SOCKET_PATH)")
    parser.add_argument("--serial", help="target a specific Stream Deck by serial number")
    parser.add_argument(
        "--no-device",
        action="store_true",
        help="run against an in-memory surface; no hardware required",
    )
    parser.add_argument(
        "--virtual",
        action="store_true",
        help=(
            "serve a Hammerspoon virtual deck and mirror it to USB whenever a physical "
            "deck is attached"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in GroupingMode],
        default=GroupingMode.AGENT.value,
        help=(
            "what the keys show: 'agent' (the most recently active agents, "
            "oldest first, ignoring workspaces), 'workspace' (a column per "
            "workspace, in sidebar order) or 'tab' (a column per tab of the "
            "focused workspace). Default: agent"
        ),
    )
    parser.add_argument(
        "--agent-limit",
        type=int,
        default=AGENT_KEY_LIMIT,
        metavar="N",
        help=(
            f"how many agents the 'agent' layout shows (default: "
            f"{AGENT_KEY_LIMIT}, two rows of a 3x5 deck, leaving the bottom "
            "row for action keys)"
        ),
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help=(
            "skip the three-word pane summaries even if a key is configured. "
            "They are already skipped when OPENAI_API_KEY is absent"
        ),
    )
    parser.add_argument(
        "--task-check-seconds",
        type=float,
        default=TASK_CHECK_SECONDS,
        metavar="SECONDS",
        help=(
            "how often to check whether a pane's task changed and needs a new "
            "name; 0 disables it, leaving names to status transitions "
            "(default: 60)"
        ),
    )
    parser.add_argument(
        "--no-screen-lock",
        action="store_true",
        help=(
            "keep the deck lit while the screen is locked. On macOS the deck is "
            "blanked and its backlight turned off for as long as the screen is "
            "locked, because the summaries are the agents' own words. No other "
            "platform publishes a lock state to follow, so this changes nothing "
            "off macOS"
        ),
    )
    parser.add_argument(
        "--actions-config",
        metavar="PATH",
        help=(
            "custom button configuration (default: "
            "$HERDR_STREAMDECK_ACTIONS_CONFIG or ~/.config/herdr-streamdeck/actions.toml)"
        ),
    )
    parser.add_argument(
        "--activate-app",
        metavar="NAME",
        help=(
            "on macOS, bring this application and its Space to the foreground "
            "after a key successfully focuses a pane (for example: iTerm)"
        ),
    )
    parser.add_argument(
        "--no-takeover",
        action="store_true",
        help=(
            "fail instead of replacing a daemon that is already running. "
            "By default a second invocation takes the deck over, so restarting "
            "is just running the command again"
        ),
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="stop a running daemon and exit",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="list attached Stream Decks and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(amain(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
