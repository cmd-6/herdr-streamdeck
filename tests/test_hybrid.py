"""Liveness: how a deck that left quietly is noticed and taken back.

The failure these cover is not a deck that refuses a write -- that one
announces itself. It is a deck that is unplugged and plugged back in while
every pane is idle, so nothing is ever written and nothing ever finds out. The
deck comes back showing the logo it draws at power-on and stays there.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from herdr_streamdeck.deck import (
    ButtonFace,
    DeckDisconnected,
    KeyFrames,
    NullSurface,
    StreamDeckSurface,
)
from herdr_streamdeck.hybrid import HybridSurface


@dataclass
class FakeDevice:
    """The handful of DeckDevice methods a liveness probe touches."""

    open_: bool = True
    present: bool = True
    closed: int = 0
    raises: bool = False

    def is_open(self) -> bool:
        if self.raises:
            raise OSError("transport is confused")
        return self.open_

    def connected(self) -> bool:
        if self.raises:
            raise OSError("transport is confused")
        return self.present

    def close(self) -> None:
        self.closed += 1
        self.open_ = False


def surface_holding(device: FakeDevice) -> StreamDeckSurface:
    surface = StreamDeckSurface()
    surface._deck = device  # type: ignore[assignment]
    return surface


def test_a_live_deck_answers_yes_without_enumerating() -> None:
    device = FakeDevice()
    surface = surface_holding(device)

    device.present = False  # would only show up in a deep probe
    assert surface.alive() is True
    assert surface.alive(deep=True) is False


def test_a_handle_closed_underneath_us_is_not_alive() -> None:
    """The exact shape of the bug.

    The StreamDeck package's reader thread closes the handle itself when a
    transport read fails, on its own thread and without telling anyone. We are
    left holding a device object that will refuse the next write, whenever
    that turns out to be.
    """
    device = FakeDevice()
    surface = surface_holding(device)
    device.open_ = False

    assert surface.alive() is False
    assert surface.connected is False, "a dead probe should let go of the device"


def test_a_probe_that_throws_counts_as_gone() -> None:
    surface = surface_holding(FakeDevice(raises=True))
    assert surface.alive() is False


def test_no_device_at_all_is_not_alive() -> None:
    assert StreamDeckSurface().alive() is False


@dataclass
class FakePhysical(NullSurface):
    """A physical half that can be yanked out from under the hybrid surface."""

    plugged: bool = True
    opens: int = 0
    fail_on_reattach_write: bool = False

    def open(self) -> None:
        self.opens += 1
        if not self.plugged:
            raise RuntimeError("no Stream Deck found")
        super().open()

    def alive(self, *, deep: bool = False) -> bool:
        if deep:
            self.deep_probes += 1
        return self.plugged

    def write(self, index: int, frames: KeyFrames, level_index: int) -> None:
        if self.fail_on_reattach_write:
            self.fail_on_reattach_write = False
            self.plugged = False
            raise DeckDisconnected("yanked mid-redraw")
        if not self.plugged:
            raise DeckDisconnected("gone")
        super().write(index, frames, level_index)


@dataclass
class QuietVirtual(NullSurface):
    """Stands in for the virtual half, which is never the thing that leaves."""

    _handler_seen: list[object] = field(default_factory=list)

    def alive(self, *, deep: bool = False) -> bool:
        return True


def hybrid_with(physical: FakePhysical) -> HybridSurface:
    return HybridSurface(physical=physical, virtual=QuietVirtual())  # type: ignore[arg-type]


def settle(check: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Wait for a background thread to reach a state, or give up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return False


def test_the_beat_notices_a_physical_deck_that_left_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("herdr_streamdeck.hybrid.RECONNECT_SECONDS", 0.01)
    physical = FakePhysical(key_count_=15, key_layout_=(3, 5))
    deck = hybrid_with(physical)
    deck.open()
    try:
        assert settle(lambda: deck._physical_connected)
        deck.set_face(0, ButtonFace(badge="one"))

        physical.plugged = False
        assert settle(lambda: not deck._physical_connected), (
            "the deck left and the beat never noticed"
        )

        physical.plugged = True
        assert settle(lambda: deck._physical_connected), "it never came back"
    finally:
        deck.close()

    assert physical.faces, "the returning deck was redrawn"


def test_a_deck_yanked_mid_reattach_does_not_strand_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconnect thread is the only thing that can bring the deck back.

    It used to redraw outside any try, so a deck that went away again during
    those fifteen writes killed the thread outright -- with the surface still
    marked connected, which meant nothing retried and nothing ever drew again.
    """
    monkeypatch.setattr("herdr_streamdeck.hybrid.RECONNECT_SECONDS", 0.01)
    physical = FakePhysical(key_count_=15, key_layout_=(3, 5), plugged=False)
    deck = hybrid_with(physical)
    deck.open()
    try:
        deck.set_face(0, ButtonFace(badge="one"))
        physical.plugged = True
        physical.fail_on_reattach_write = True

        # The reattach throws, unplugging itself on the way out.
        assert settle(lambda: physical.opens >= 1)
        assert settle(lambda: not physical.plugged)

        physical.plugged = True
        assert settle(lambda: deck._physical_connected), (
            "the reconnect thread died and took the deck with it"
        )
        assert deck._thread is not None and deck._thread.is_alive()
    finally:
        deck.close()


def test_the_beat_stops_with_the_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("herdr_streamdeck.hybrid.RECONNECT_SECONDS", 0.01)
    deck = hybrid_with(FakePhysical(key_count_=15, key_layout_=(3, 5)))
    deck.open()
    thread = deck._thread
    deck.close()

    assert thread is not None
    assert settle(lambda: not thread.is_alive()), "the beat outlived the surface"
    assert threading.active_count() >= 1
