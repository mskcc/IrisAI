"""Persistence utilities — user settings and data directory helpers.

Handles:
- User data directory resolution
- User settings (get/save/update work dir)

No Chainlit or LLM dependencies. Pure file I/O functions.
"""
import os
import json
from pathlib import Path
from typing import Optional


# ── User data directory helper ──────────────────────────────────────────────────
IRISAI_APP_NAME = os.environ.get("IRISAI_APP_NAME", "IrisAIdev")


def get_user_data_dir(username: str) -> Path:
    """Get the user's IrisAI data directory.
    Uses IRISAI_APP_NAME env var set by before.sh.erb, defaults to IrisAIdev."""
    safe_username = Path(username).name
    if not safe_username or safe_username != username or '/' in username or '\\' in username:
        raise ValueError(f"Invalid username: {username!r}")
    return Path(f"/home/{safe_username}/{IRISAI_APP_NAME}")


# ── User settings ──────────────────────────────────────────────────────
def get_user_settings(username: str) -> dict:
    """Load user settings from persistent JSON file.

    Args:
        username: User identifier

    Returns:
        Settings dict, or empty dict if not found
    """
    base_dir = get_user_data_dir(username)
    file_path = base_dir / "usersettings.json"
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load settings for {username}: {e}")
    return {}


def save_user_settings(username: str, settings: dict) -> bool:
    """Save user settings to persistent JSON file.

    Args:
        username: User identifier
        settings: Settings dict to save

    Returns:
        True on success, False on failure
    """
    base_dir = get_user_data_dir(username)
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / "usersettings.json"
    try:
        with open(file_path, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save settings for {username}: {e}")
        return False


def get_work_dir(username: str = None) -> str:
    """Get the authoritative work_dir.

    Priority: WORK_DIR env var → settings file → empty string.
    Env var takes precedence because:
    - It's set at container launch (intended runtime value)
    - set_user_work_directory() updates both env var + settings file
    - In tests, monkeypatch overrides env var for isolation
    Settings file is the cross-session/cross-container fallback when env var unset.

    Args:
        username: User identifier. If None, resolved from environment.

    Returns:
        Absolute path string, or empty string if not configured.
    """
    env_wd = os.environ.get("WORK_DIR", "")
    if env_wd:
        return env_wd
    if username is None:
        username = os.environ.get("USER", "")
        if not username:
            import pwd
            username = pwd.getpwuid(os.getuid()).pw_name
    settings = get_user_settings(username)
    return settings.get("work_dir", "")


def bootstrap_work_dir_from_env(username: str) -> Optional[str]:
    """One-time bootstrap: seed settings file from WORK_DIR env var.

    Only writes to settings if the file doesn't already have a work_dir entry.
    This preserves the user's explicit choice via set_user_work_directory().

    Args:
        username: User identifier

    Returns:
        The work directory path string, or None if env var not set
    """
    env_work_dir = os.environ.get("WORK_DIR")
    if not env_work_dir:
        return None
    settings = get_user_settings(username)
    if settings.get("work_dir"):
        return settings["work_dir"]
    settings["work_dir"] = env_work_dir
    if save_user_settings(username, settings):
        print(f"[INFO] Bootstrapped work directory from env: {env_work_dir}")
        return env_work_dir
    else:
        print(f"[ERROR] Failed to bootstrap work directory: {env_work_dir}")
        return None


def update_work_dir_from_env(username: str) -> Optional[str]:
    """DEPRECATED: Use bootstrap_work_dir_from_env() instead.
    Kept for backward compatibility during transition."""
    return bootstrap_work_dir_from_env(username)
