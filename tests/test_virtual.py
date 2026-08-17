"""The virtual surface exposes exact images and authenticated press events."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from herdr_streamdeck.deck import ButtonFace, NullSurface
from herdr_streamdeck.hybrid import HybridSurface
from herdr_streamdeck.virtual import VirtualSurface, load_or_create_token


def request(url: str, token: str | None = None) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    call = urllib.request.Request(url, data=b"", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(call) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code


def test_token_is_created_once_with_user_only_permissions(tmp_path: Path) -> None:
    path = tmp_path / "config" / "token"

    first = load_or_create_token(path)
    second = load_or_create_token(path)

    assert first == second
    assert path.stat().st_mode & 0o777 == 0o600


def test_rendered_keys_and_revisions_are_published_atomically(tmp_path: Path) -> None:
    surface = VirtualSurface(
        state_dir=tmp_path / "state", token_path=tmp_path / "token", port=0
    )
    surface.open()
    try:
        frames = surface.render(ButtonFace(mark="❯", badge="codex"))
        surface.write(3, frames, surface.levels - 1)

        state = json.loads(surface.state_path.read_text())
        image = Path(state["keys"][3])
        assert state["key_revisions"][3] == state["revision"]
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert not list(surface.state_dir.glob("*.tmp"))
    finally:
        surface.close()


def test_only_the_token_holder_can_press_a_virtual_key(tmp_path: Path) -> None:
    surface = VirtualSurface(
        state_dir=tmp_path / "state", token_path=tmp_path / "token", port=0
    )
    events: list[tuple[int, bool]] = []
    surface.set_press_handler(lambda index, pressed: events.append((index, pressed)))
    surface.open()
    try:
        server = surface._server
        assert server is not None
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/keys/7/down"

        assert request(url) == 401
        assert request(url, surface.token_path.read_text().strip()) == 204
        assert events == [(7, True)]
    finally:
        surface.close()


@pytest.mark.parametrize("path", ["/keys/nope/down", "/keys/99/up", "/wrong"])
def test_invalid_press_targets_are_rejected(tmp_path: Path, path: str) -> None:
    surface = VirtualSurface(
        state_dir=tmp_path / "state", token_path=tmp_path / "token", port=0
    )
    surface.open()
    try:
        server = surface._server
        assert server is not None
        token = surface.token_path.read_text().strip()
        assert request(f"http://127.0.0.1:{server.server_address[1]}{path}", token) == 404
    finally:
        surface.close()


def test_hybrid_surface_mirrors_the_same_face_to_virtual_and_physical(tmp_path: Path) -> None:
    physical = NullSurface()
    virtual = VirtualSurface(
        state_dir=tmp_path / "state", token_path=tmp_path / "token", port=0
    )
    surface = HybridSurface(physical=physical, virtual=virtual)
    surface.open()
    try:
        face = ButtonFace(mark="✳", badge="claude")
        surface.set_face(4, face)

        assert physical.faces[4] == face
        state = json.loads(virtual.state_path.read_text())
        assert Path(state["keys"][4]).read_bytes().startswith(b"\x89PNG")
    finally:
        surface.close()


def test_blanking_does_not_forget_the_hybrid_surface_brightness(tmp_path: Path) -> None:
    physical = NullSurface(brightness_=73)
    virtual = VirtualSurface(
        state_dir=tmp_path / "state", token_path=tmp_path / "token", port=0
    )
    surface = HybridSurface(physical=physical, virtual=virtual)
    surface.open()
    try:
        surface.set_brightness(0)
        assert surface.brightness == 73

        surface.set_brightness(surface.brightness)
        assert physical.brightness_written == 73
    finally:
        surface.close()


def test_virtual_surface_starts_when_usb_is_absent(tmp_path: Path) -> None:
    class MissingDeck(NullSurface):
        def open(self) -> None:
            raise RuntimeError("no Stream Deck found")

    virtual = VirtualSurface(
        state_dir=tmp_path / "state", token_path=tmp_path / "token", port=0
    )
    surface = HybridSurface(physical=MissingDeck(), virtual=virtual)
    surface.open()
    try:
        surface.set_face(14, ButtonFace(builtin_icon="microphone"))
        assert virtual.connected
        assert virtual.state_path.exists()
    finally:
        surface.close()
