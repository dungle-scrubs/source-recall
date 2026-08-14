# CLAUDE.md

This file imports the shared guidance in [`AGENTS.md`](./AGENTS.md).

> @AGENTS.md

## Claude-specific mechanics

- Keep responses concise and direct; avoid superlatives and emotional framing.
- When referencing code, use clickable local-file links with `file_path:line_number`.
- For privacy-restricted work (tokens, `.env`, prod output), delegate to a local model via `pi --provider lmstudio`.
- Use the tool-proxy `github` app for GitHub operations when available; fall back to `gh` CLI only when needed.
