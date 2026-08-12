"""Drive the virtual deck continuously and attach USB whenever it is present."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from .deck import ButtonFace, ButtonSurface, DeckDisconnected, KeyFrames, PressHandler
from .virtual import VirtualSurface

logger = logging.getLogger(__name__)

RECONNECT_SECONDS = 3.0


@dataclass
class HybridSurface:
    """A virtual surface with an optional, hot-pluggable physical mirror."""

    physical: ButtonSurface
    virtual: VirtualSurface = field(default_factory=VirtualSurface)
    _handler: PressHandler | None = None
    _physical_connected: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _current: dict[int, tuple[KeyFrames, int]] = field(default_factory=dict)
    _physical_frames: dict[ButtonFace, KeyFrames] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key_count(self) -> int:
        return self.virtual.key_count

    @property
    def key_layout(self) -> tuple[int, int]:
        return self.virtual.key_layout

    @property
    def key_size(self) -> tuple[int, int]:
        return self.virtual.key_size

    @property
    def levels(self) -> int:
        return self.virtual.levels

    @property
    def connected(self) -> bool:
        return True

    @property
    def brightness(self) -> int:
        return self.virtual.brightness

    def open(self) -> None:
        self.virtual.open()
        self._try_physical_open()
        self._thread = threading.Thread(
            target=self._reconnect_loop,
            name="stream-deck-reconnect",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=RECONNECT_SECONDS + 1)
        self.virtual.close()
        self.physical.close()

    def reopen(self) -> bool:
        return True

    def set_press_handler(self, handler: PressHandler | None) -> None:
        self._handler = handler
        self.virtual.set_press_handler(handler)
        if self._physical_connected:
            self.physical.set_press_handler(handler)

    def set_brightness(self, percent: int) -> None:
        self.virtual.set_brightness(percent)
        if self._physical_connected:
            try:
                self.physical.set_brightness(percent)
            except DeckDisconnected:
                self._physical_connected = False

    def set_face(self, index: int, face: ButtonFace) -> None:
        self.write(index, self.render(face), self.levels - 1)

    def render(self, face: ButtonFace) -> KeyFrames:
        return self.virtual.render(face)

    def write(self, index: int, frames: KeyFrames, level_index: int) -> None:
        self.virtual.write(index, frames, level_index)
        with self._lock:
            self._current[index] = (frames, level_index)
            if not self._physical_connected:
                return
            try:
                physical = self._physical_frames.get(frames.face)
                if physical is None:
                    physical = self.physical.render(frames.face)
                    self._physical_frames[frames.face] = physical
                self.physical.write(index, physical, level_index)
            except DeckDisconnected:
                self._physical_connected = False
                logger.warning("physical deck disconnected; virtual deck remains available")

    def _reconnect_loop(self) -> None:
        while not self._stop.wait(RECONNECT_SECONDS):
            if not self._physical_connected:
                self._try_physical_open()

    def _try_physical_open(self) -> None:
        try:
            self.physical.open()
        except Exception:
            logger.debug("physical deck is not available")
            return
        if (
            self.physical.key_count != self.key_count
            or self.physical.key_layout != self.key_layout
        ):
            logger.error("physical and virtual deck layouts do not match")
            self.physical.close()
            return
        with self._lock:
            self._physical_connected = True
            self._physical_frames.clear()
            self.physical.set_press_handler(self._handler)
            for index, (frames, level) in self._current.items():
                physical = self.physical.render(frames.face)
                self._physical_frames[frames.face] = physical
                self.physical.write(index, physical, level)
        logger.info("physical Stream Deck attached")
