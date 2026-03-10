# source-recall task runner

default:
    @just --list

# Run tests
test *args:
    uv run python -m pytest {{args}}

# Run tests with short output
t *args:
    uv run python -m pytest --tb=short -q {{args}}

# Lint and format
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

# Fix lint issues
fix:
    uv run ruff check --fix src/ tests/
    uv run ruff format src/ tests/

# Build index for a repo (default: current dir)
index path=".":
    uv run sr index {{path}}

# Start query server (one or more repos)
serve *paths:
    uv run sr serve {{paths}}

# Query the running server
query question:
    curl -s -X POST http://127.0.0.1:7249/query \
        -H "Content-Type: application/json" \
        -d '{"question": "{{question}}"}' | python3 -m json.tool

# Check server health
health:
    curl -s http://127.0.0.1:7249/health | python3 -m json.tool

# List repos on running server
repos:
    curl -s http://127.0.0.1:7249/repos | python3 -m json.tool

# Server status
status:
    curl -s http://127.0.0.1:7249/status | python3 -m json.tool

# Install sr globally
install:
    uv tool install -e . --force

# Install and start launchd service
launchd-install:
    #!/usr/bin/env bash
    set -euo pipefail
    plist="$HOME/Library/LaunchAgents/dev.source-recall.serve.plist"
    if [ -f "$plist" ]; then
        echo "Unloading existing service..."
        launchctl bootout gui/$(id -u) "$plist" 2>/dev/null || true
    fi
    cp launchd/dev.source-recall.serve.plist "$plist"
    echo "Installed plist to $plist"
    echo "Edit it to set your repo paths, then run: just launchd-start"

# Start launchd service
launchd-start:
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.source-recall.serve.plist
    @echo "Service started. Check: just health"

# Stop launchd service
launchd-stop:
    launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/dev.source-recall.serve.plist
    @echo "Service stopped."

# View launchd service logs
launchd-logs:
    tail -50 /tmp/source-recall-serve.log
