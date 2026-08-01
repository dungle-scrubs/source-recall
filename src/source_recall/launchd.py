"""launchd plist generation and lifecycle management."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from source_recall.daemon_config import DaemonConfig

PLIST_LABEL = "dev.source-recall.daemon"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"
LOG_PATH = Path("/tmp/source-recall-daemon.log")


def _find_sr_binary() -> str:
    """Locate the sr binary.

    @returns: Absolute path to the sr executable.
    """
    import shutil

    local_venv_sr = Path.cwd() / ".venv" / "bin" / "sr"
    if local_venv_sr.exists():
        return str(local_venv_sr)
    sr = shutil.which("sr")
    if sr:
        return sr
    # Fallback to common uv tool install location.
    home_bin = Path.home() / ".local" / "bin" / "sr"
    if home_bin.exists():
        return str(home_bin)
    return "sr"


def generate_plist(config: DaemonConfig) -> str:
    """Generate a launchd plist XML for the daemon.

    The plist invokes ``sr daemon run`` with the config path.
    ExitTimeOut is set to shutdown_timeout_s + 5 (D-005).

    @param config: Daemon configuration.
    @returns: Plist XML string.
    """
    sr_bin = _find_sr_binary()
    exit_timeout = config.shutdown_timeout_s + 5

    args = [sr_bin, "daemon", "run"]
    if config.config_path:
        args.extend(["--config", str(config.config_path)])

    args_xml = "\n".join(f"        <string>{a}</string>" for a in args)

    # Build PATH from current environment.
    path_dirs = os.environ.get("PATH", "/usr/bin:/bin")

    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ExitTimeOut</key>
    <integer>{exit_timeout}</integer>

    <key>StandardOutPath</key>
    <string>{LOG_PATH}</string>

    <key>StandardErrorPath</key>
    <string>{LOG_PATH}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path_dirs}</string>
    </dict>
</dict>
</plist>
"""


def install_plist(config: DaemonConfig) -> Path:
    """Write the plist and load it via launchctl.

    @param config: Daemon configuration.
    @returns: Path to the written plist file.
    """
    # Unload first if already loaded.
    unload_plist()

    plist_content = generate_plist(config)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist_content)

    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
        check=True,
        capture_output=True,
    )
    return PLIST_PATH


def unload_plist() -> None:
    """Unload the daemon plist via launchctl. No-op if not loaded."""
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(PLIST_PATH)],
        capture_output=True,
    )


def is_loaded() -> bool:
    """Check if the daemon plist is loaded.

    @returns: True if launchctl reports the service.
    """
    uid = os.getuid()
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{PLIST_LABEL}"],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0
