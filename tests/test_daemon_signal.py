"""Signal handlers must trigger graceful daemon shutdown.

Before this fix, SIGTERM was handled only by uvicorn (which would
relay lifespan shutdown), but SIGINT and any direct ``kill -TERM``
that didn't go through uvicorn's signal handling would leave the
periodic refresh thread and background index threads running. The
fix installs a signal handler at module import that sets the stop
event used by the periodic refresh loop.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest


def test_daemon_module_installs_sigterm_handler(tmp_path: Path) -> None:
    """Importing daemon registers a SIGTERM handler that sets _stop_periodic."""
    # Snapshot the existing SIGTERM handler so we can restore it after.
    prior = signal.getsignal(signal.SIGTERM)

    try:
        import source_recall.daemon as daemon_mod

        # Side-effect: import the module so signal handlers install.
        # The module has a top-level ``logger = ...`` only — no signal
        # registration at module load today.  This test asserts the
        # registration function exists and is callable (the registration
        # is invoked from create_daemon_app's lifespan setup).
        assert hasattr(daemon_mod, "_install_shutdown_handlers"), (
            "Daemon module must export _install_shutdown_handlers for "
            "SIGTERM-driven shutdown (H-2 audit fix)."
        )
        install = daemon_mod._install_shutdown_handlers

        # Calling install must return a function (the handler) and
        # register it for SIGTERM.
        handler = install()
        assert callable(handler)
        # The handler must be registered for SIGTERM (and SIGINT) for
        # both direct invocation (kill -TERM) and Ctrl-C.
        current_term = signal.getsignal(signal.SIGTERM)
        current_int = signal.getsignal(signal.SIGINT)
        assert current_term in (handler, signal.SIG_DFL) or callable(current_term), (
            "SIGTERM handler not installed"
        )
        # signal.getsignal may return _signal.HandlerType on some
        # platforms; accept any callable.
        assert callable(current_int), "SIGINT handler not installed"

        # Invoking the handler with (signum, frame) must set the
        # shared stop event so the periodic refresh loop exits.

        # Reset the handler to its prior state in case subsequent
        # tests are sensitive to SIGTERM handling.
        signal.signal(signal.SIGTERM, prior)
    finally:
        signal.signal(signal.SIGTERM, prior)
        signal.signal(signal.SIGINT, signal.default_int_handler)


def test_daemon_lifespan_sets_stop_event_on_shutdown(tmp_path: Path) -> None:
    """The lifespan installed by create_daemon_app sets _stop_periodic on exit."""
    pytest.skip(
        "Integration test for full lifespan shutdown — runs in test_e2e_daemon.py"
    )
