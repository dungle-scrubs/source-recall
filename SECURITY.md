# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in source-recall, please report it
responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Open a **private** [GitHub Security Advisory][ghsa] using the
   "Report a vulnerability" button on the Security tab, **or**
3. Email the maintainer directly.

[ghsa]: https://github.com/dungle-scrubs/source-recall/security/advisories/new

## Disclosure Timeline

- You will receive an acknowledgement within **72 hours**.
- We will investigate and aim to provide an initial assessment within
  **7 days**.
- A fix or mitigation will be coordinated with you before public disclosure.
- We ask for a **90-day** responsible disclosure window before any public
  details are shared.

## Scope

This policy covers the `source-recall` package and its first-party CLI (`sr`).
Vulnerabilities in third-party dependencies should be reported upstream to the
relevant project and, if applicable, as a GitHub dependency advisory so we can
patch our pinned versions.

## Supported Versions

Only the latest release line receives security updates.

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |
| < 0.1   | no        |
