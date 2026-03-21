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

# Build index and output JSON summary
index-json path=".":
    uv run sr index {{path}} --json

# List all indexed repos
list:
    uv run sr list

# List all indexed repos (JSON)
list-json:
    uv run sr list --json

# Show index status
status path=".":
    uv run sr status {{path}}

# Show index status (JSON)
status-json path=".":
    uv run sr status {{path}} --json

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
server-status:
    curl -s http://127.0.0.1:7249/status | python3 -m json.tool

# Remove orphaned indexes
clean:
    uv run sr clean

# Remove orphaned indexes (dry run)
clean-dry:
    uv run sr clean --dry-run

# Show resolved config
config path=".":
    uv run sr config {{path}}

# Install sr globally
install:
    uv tool install -e . --force

# Start daemon in foreground (from repos.toml)
daemon-run *args:
    uv run sr daemon run {{args}}

# Start daemon via launchd (background)
daemon-start *args:
    uv run sr daemon start {{args}}

# Stop daemon
daemon-stop:
    uv run sr daemon stop

# Daemon status
daemon-status:
    uv run sr daemon status

# Daemon logs
daemon-logs *args:
    uv run sr daemon logs {{args}}

# Add a repo to the daemon
add path:
    uv run sr add {{path}}

# Remove a repo from the daemon
remove name:
    uv run sr remove {{name}}

# List daemon repos
daemon-repos:
    uv run sr repos

# Legacy launchd recipes (deprecated — use daemon-* instead)

# Install and start launchd service (DEPRECATED)
launchd-install:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "DEPRECATED: Use 'just daemon-start' instead"
    plist="$HOME/Library/LaunchAgents/dev.source-recall.serve.plist"
    if [ -f "$plist" ]; then
        echo "Unloading existing service..."
        launchctl bootout gui/$(id -u) "$plist" 2>/dev/null || true
    fi
    cp launchd/dev.source-recall.serve.plist "$plist"
    echo "Installed plist to $plist"
    echo "Edit it to set your repo paths, then run: just launchd-start"

# Start launchd service (DEPRECATED)
launchd-start:
    @echo "DEPRECATED: Use 'just daemon-start' instead"
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.source-recall.serve.plist
    @echo "Service started. Check: just health"

# Stop launchd service (DEPRECATED)
launchd-stop:
    @echo "DEPRECATED: Use 'just daemon-stop' instead"
    launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/dev.source-recall.serve.plist
    @echo "Service stopped."

# View launchd service logs (DEPRECATED)
launchd-logs:
    @echo "DEPRECATED: Use 'just daemon-logs' instead"
    tail -50 /tmp/source-recall-serve.log
