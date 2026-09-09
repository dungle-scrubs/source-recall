# RFCs

Design decisions that are specified but not yet built. Each RFC covers
one change, in RFC 2119 language, and carries a `status` field in its
frontmatter.

Anything not `Implemented` or `Withdrawn` is open work. To list what is
open:

```bash
grep -H '^status:' docs/rfc/*.rfc.md
```

| RFC | Status | Summary |
|-----|--------|---------|
| [01](./01_shared-repo-resolution-and-error-contract.rfc.md) — Shared repo resolution and error contract | Draft | One resolution seam and one error taxonomy for the daemon and the query server, replacing four copies of the same branch ladder. Five open questions block Accepted. |

## Statuses

| Status | Meaning |
|--------|---------|
| Draft | Open for major changes. |
| Review | Requesting feedback; structure is stable. |
| Accepted | Approved for implementation. |
| Implemented | Code matches the spec. |
| Superseded | Replaced by another RFC, which it links to. |
| Withdrawn | Abandoned, with the reason recorded in the RFC. |

## Adding one

Use the `draft-rfc` skill, which numbers the file and writes the section
stubs for the RFC type. Add a row to the table above when the file
lands, and update the row's status when it changes.

Historical design rationale that predates this directory lives in
`docs/history/`, including the audits that several of these RFCs answer.
