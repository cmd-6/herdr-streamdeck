"""Custom buttons: strict config, rendering, and macOS URL dispatch."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from herdr_streamdeck.actions import DeckAction, load_actions, open_action


def write_config(path: Path, text: str) -> None:
    path.write_text(text)


def test_missing_config_means_no_reserved_keys(tmp_path: Path) -> None:
    assert load_actions(tmp_path / "missing.toml", key_count=15) == {}


def test_actions_use_human_friendly_one_based_key_numbers(tmp_path: Path) -> None:
    path = tmp_path / "actions.toml"
    write_config(
        path,
        """
[[action]]
key = 15
label = "Dictate"
icon = "microphone"
url = "superwhisper://record"
""",
    )

    assert load_actions(path, key_count=15) == {
        14: DeckAction(
            key=14,
            label="Dictate",
            icon="microphone",
            url="superwhisper://record",
        )
    }


@pytest.mark.parametrize(
    "body, message",
    [
        ("key = 0\nlabel = 'x'\nicon = 'microphone'\nurl = 'x://y'", "between 1 and 15"),
        ("key = 1\nlabel = ''\nicon = 'microphone'\nurl = 'x://y'", "non-empty"),
        ("key = 1\nlabel = 'x'\nicon = 'camera'\nurl = 'x://y'", "microphone"),
        ("key = 1\nlabel = 'x'\nicon = 'microphone'\nurl = 'missing'", "scheme"),
    ],
)
def test_invalid_actions_fail_loudly(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "actions.toml"
    write_config(path, f"[[action]]\n{body}\n")

    with pytest.raises(ValueError, match=message):
        load_actions(path, key_count=15)


def test_a_physical_key_cannot_be_assigned_twice(tmp_path: Path) -> None:
    path = tmp_path / "actions.toml"
    write_config(
        path,
        """
[[action]]
key = 1
label = "one"
icon = "microphone"
url = "one://toggle"

[[action]]
key = 1
label = "two"
icon = "microphone"
url = "two://toggle"
""",
    )

    with pytest.raises(ValueError, match="configured more than once"):
        load_actions(path, key_count=15)


def test_open_action_keeps_the_current_app_in_front(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command: tuple[str, ...], **options: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("herdr_streamdeck.actions.subprocess.run", run)
    action = DeckAction(14, "Dictate", "microphone", "superwhisper://record")

    open_action(action)

    assert calls == [
        (
            ("open", "-g", "superwhisper://record"),
            {"capture_output": True, "text": True, "timeout": 2.0, "check": True},
        )
    ]
