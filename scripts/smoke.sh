#!/usr/bin/env bash
set -euo pipefail

repo="${1:-$(pwd)}"
repo="$(cd "$repo" && pwd)"
tmp="$(mktemp -d)"
server_pid=""

cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT

export HOME="$tmp/home"
mkdir -p "$HOME"

python - <<'PY' > "$tmp/port"
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
port="$(cat "$tmp/port")"

echo "== smoke: index own source =="
uv run sr index "$repo" --no-embed --json > "$tmp/index.json"
python - "$repo" "$tmp/index.json" <<'PY'
import json
import sys

repo = sys.argv[1]
with open(sys.argv[2]) as fh:
    data = json.load(fh)
assert data["repo_path"] == repo, data
assert data["file_count"] > 0, data
assert data["chunk_count"] > 0, data
assert data["vector_count"] == 0, data
print(f"indexed {data['file_count']} files / {data['chunk_count']} chunks")
PY

echo "== smoke: status =="
uv run sr status "$repo" --json > "$tmp/status.json"
python - "$tmp/status.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
assert data["file_count"] > 0, data
assert data["chunk_count"] > 0, data
print("status ok")
PY

echo "== smoke: cli query =="
uv run sr ask "index directory" "$repo" --no-embed --json > "$tmp/ask.json"
python - "$tmp/ask.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
assert data, "expected at least one query result"
assert any(item["file_path"] == "src/source_recall/store.py" for item in data), data[:5]
print(f"query returned {len(data)} results")
PY

echo "== smoke: http server =="
uv run sr serve "$repo" --no-embed --host 127.0.0.1 --port "$port" > "$tmp/server.log" 2>&1 &
server_pid="$!"
python - "$port" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
base = f"http://127.0.0.1:{port}"

for _ in range(50):
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=0.5) as resp:
            if resp.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(0.1)
else:
    raise SystemExit("server did not become healthy")

with urllib.request.urlopen(f"{base}/status", timeout=5) as resp:
    status = json.load(resp)
if "repos" in status:
    assert status["repos"], status
else:
    assert status["file_count"] > 0, status
    assert status["chunk_count"] > 0, status

payload = json.dumps({"question": "index directory", "top_k": 5}).encode()
req = urllib.request.Request(
    f"{base}/query",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=5) as resp:
    body = json.load(resp)
results = body.get("results", body)
assert results, body
print("server query ok")
PY

echo "smoke passed"
