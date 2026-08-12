"""A loopback-only virtual Stream Deck for the Hammerspoon overlay."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from .animation import LEVELS
from .deck import ButtonFace, KeyFrames, PressHandler, key_frames

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

VIRTUAL_HOST = "127.0.0.1"
VIRTUAL_PORT = 17373


def default_state_dir() -> Path:
    return Path.home() / "Library" / "Caches" / "herdr-streamdeck"


def default_token_path() -> Path:
    return Path.home() / ".config" / "herdr-streamdeck" / "virtual-token"


def load_or_create_token(path: Path) -> str:
    """Return a stable local API secret, creating it with user-only access."""
    try:
        token = path.read_text().strip()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(32)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(token + "\n")
    if not token:
        raise ValueError(f"virtual deck token is empty: {path}")
    return token


def _png_frames(size: tuple[int, int], face: ButtonFace, levels: int) -> Iterator[bytes]:
    from io import BytesIO

    for image in key_frames(size, face, levels):
        output = BytesIO()
        image.save(output, format="PNG")
        yield output.getvalue()


@dataclass
class VirtualSurface:
    """A 3x5 surface rendered to files and pressed through a local HTTP API."""

    state_dir: Path = field(default_factory=default_state_dir)
    token_path: Path = field(default_factory=default_token_path)
    host: str = VIRTUAL_HOST
    port: int = VIRTUAL_PORT
    key_count_: int = 15
    key_layout_: tuple[int, int] = (3, 5)
    _handler: PressHandler | None = None
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None
    _revision: int = 0
    _key_revisions: list[int] = field(default_factory=lambda: [0] * 15)
    _brightness: int = 60
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key_count(self) -> int:
        return self.key_count_

    @property
    def key_layout(self) -> tuple[int, int]:
        return self.key_layout_

    @property
    def key_size(self) -> tuple[int, int]:
        return (72, 72)

    @property
    def levels(self) -> int:
        return LEVELS

    @property
    def connected(self) -> bool:
        return self._server is not None

    @property
    def brightness(self) -> int:
        return self._brightness

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    def open(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        token = load_or_create_token(self.token_path)
        surface = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/health":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()

            def do_POST(self) -> None:
                if self.headers.get("Authorization") != f"Bearer {token}":
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) != 3 or parts[0] != "keys" or parts[2] not in {"down", "up"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    index = int(parts[1])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not 0 <= index < surface.key_count:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                handler = surface._handler
                if handler is not None:
                    handler(index, parts[2] == "down")
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("virtual deck: " + format, *args)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="virtual-deck-http",
            daemon=True,
        )
        self._thread.start()
        self._write_state()
        logger.info("virtual deck ready at http://%s:%d", self.host, self.port)

    def close(self) -> None:
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def reopen(self) -> bool:
        return self.connected

    def set_press_handler(self, handler: PressHandler | None) -> None:
        self._handler = handler

    def set_brightness(self, percent: int) -> None:
        self._brightness = percent

    def set_face(self, index: int, face: ButtonFace) -> None:
        self.write(index, self.render(face), self.levels - 1)

    def render(self, face: ButtonFace) -> KeyFrames:
        return KeyFrames(face=face, frames=tuple(_png_frames(self.key_size, face, self.levels)))

    def write(self, index: int, frames: KeyFrames, level_index: int) -> None:
        if not 0 <= index < self.key_count:
            raise IndexError(f"key {index} out of range (0..{self.key_count - 1})")
        image = frames.at(level_index)
        with self._lock:
            self._atomic_write(self.state_dir / f"key-{index}.png", image)
            self._revision += 1
            self._key_revisions[index] = self._revision
            self._write_state()

    def _write_state(self) -> None:
        document = {
            "revision": self._revision,
            "rows": self.key_layout[0],
            "columns": self.key_layout[1],
            "keys": [
                str(self.state_dir / f"key-{index}.png") for index in range(self.key_count)
            ],
            "key_revisions": self._key_revisions,
        }
        self._atomic_write(
            self.state_path,
            (json.dumps(document, separators=(",", ":")) + "\n").encode(),
        )

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
