"""User-configured buttons that live alongside Herdr panes."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .deck import RGB, ButtonFace

ACTION_BACKGROUND: RGB = (82, 48, 128)
ACTION_TIMEOUT = 2.0
SUPPORTED_ICONS = frozenset({"microphone"})


@dataclass(frozen=True, slots=True)
class DeckAction:
    """One reserved physical key and the URL opened when it is tapped."""

    key: int
    """Zero-based key index used internally by the Stream Deck library."""

    label: str
    icon: str
    url: str

    @property
    def face(self) -> ButtonFace:
        return ButtonFace(
            builtin_icon=self.icon,
            badge=self.label[:8],
            background=ACTION_BACKGROUND,
        )


def default_config_path() -> Path:
    """The per-user actions file, overridable for services and tests."""
    override = os.environ.get("HERDR_STREAMDECK_ACTIONS_CONFIG")
    if override:
        return Path(override).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "herdr-streamdeck" / "actions.toml"


def load_actions(path: Path, *, key_count: int) -> dict[int, DeckAction]:
    """Read and validate actions, returning them keyed by zero-based index."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"could not read actions config {path}: {exc}") from exc

    records = document.get("action", [])
    if not isinstance(records, list):
        raise ValueError(f"{path}: 'action' must be an array of tables")

    actions: dict[int, DeckAction] = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: action {position} must be a table")
        action = _parse_action(record, path, position, key_count)
        if action.key in actions:
            raise ValueError(f"{path}: key {action.key + 1} is configured more than once")
        actions[action.key] = action
    return actions


def _parse_action(
    record: dict[object, object], path: Path, position: int, key_count: int
) -> DeckAction:
    key = record.get("key")
    if not isinstance(key, int) or isinstance(key, bool) or not 1 <= key <= key_count:
        raise ValueError(f"{path}: action {position} key must be between 1 and {key_count}")

    label = record.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{path}: action {position} label must be a non-empty string")

    icon = record.get("icon")
    if not isinstance(icon, str) or icon not in SUPPORTED_ICONS:
        choices = ", ".join(sorted(SUPPORTED_ICONS))
        raise ValueError(f"{path}: action {position} icon must be one of: {choices}")

    url = record.get("url")
    if not isinstance(url, str) or not urlsplit(url).scheme:
        raise ValueError(f"{path}: action {position} url must include a scheme")

    return DeckAction(key=key - 1, label=label.strip(), icon=icon, url=url)


def open_action(action: DeckAction) -> None:
    """Open an action URL without bringing its handler to the foreground."""
    subprocess.run(
        ("open", "-g", action.url),
        capture_output=True,
        text=True,
        timeout=ACTION_TIMEOUT,
        check=True,
    )
