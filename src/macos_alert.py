from __future__ import annotations

import subprocess
from typing import Optional


def mac_notify(title: str, message: str) -> None:
    # Uses AppleScript notifications
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], check=False)


def mac_play_sound(sound_path: str) -> None:
    # Use afplay (bundled with macOS)
    subprocess.run(["afplay", sound_path], check=False)


def alert(title: str, message: str, sound_path: Optional[str] = None) -> None:
    mac_notify(title, message)
    if sound_path:
        mac_play_sound(sound_path)
