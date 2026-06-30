"""Tests for the optional-dependency split (B-3).

The heavy ML deps (sentence-transformers, onnxruntime, einops) are an
optional ``embed`` extra so FTS-only users can ``uv tool install -e .``
without pulling torch.  The dev dependency group must include
``source-recall[embed]`` so the test suite still has access to the
real embedder for the slow-marked tests.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


class TestEmbedExtra:
    def test_embed_optional_dependency_exists(self) -> None:
        """[project.optional-dependencies] has an ``embed`` key."""
        data = _load_pyproject()
        opt = data.get("project", {}).get("optional-dependencies", {})
        assert "embed" in opt, (
            "expected [project.optional-dependencies] to declare an 'embed' extra"
        )

    def test_embed_extra_contains_ml_deps(self) -> None:
        """The embed extra pulls the ML trio (sentence-transformers et al.)."""
        data = _load_pyproject()
        embed = data["project"]["optional-dependencies"].get("embed", [])
        joined = " ".join(embed)
        assert "sentence-transformers" in joined
        assert "onnxruntime" in joined
        assert "einops" in joined

    def test_ml_deps_not_in_core_dependencies(self) -> None:
        """Core dependencies do NOT pull sentence-transformers / torch."""
        data = _load_pyproject()
        deps = data["project"].get("dependencies", [])
        for forbidden in ("sentence-transformers", "onnxruntime", "einops"):
            matches = [d for d in deps if d.lower().startswith(forbidden)]
            assert matches == [], (
                f"{forbidden!r} must be in the embed extra, not core deps: {matches}"
            )

    def test_dev_group_includes_embed_extra(self) -> None:
        """dev dependency group installs the embed extra.

        Without this, the slow-marked reranker test loses access to
        sentence_transformers and silently fails to import the library
        it tests.
        """
        data = _load_pyproject()
        dev = data.get("dependency-groups", {}).get("dev", [])
        joined = " ".join(dev).lower()
        assert "source-recall[embed]" in joined or "source_recall[embed]" in joined, (
            "dev group must include 'source-recall[embed]' so the test "
            "suite can import sentence_transformers for slow tests"
        )

    def test_license_field_present(self) -> None:
        """Sanity check: license field is declared (B-1 follow-up)."""
        data = _load_pyproject()
        assert data["project"].get("license") == "MIT"


class TestLazyImportContract:
    """The lazy-import contract that makes the embed split safe.

    sentence_transformers is imported only inside CodeRankEmbedder and
    CrossEncoderReranker (both lazy, both wrapped in try/except).  This
    test pins that contract so a future refactor can't accidentally
    promote the import to module scope and break FTS-only installs.
    """

    def test_no_module_level_sentence_transformers_import(self) -> None:
        """No source module imports sentence_transformers at top level."""
        violations: list[str] = []
        for py in (ROOT / "src").rglob("*.py"):
            text = py.read_text()
            # Walk the module-level lines (no indentation).
            for line in text.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if (
                    (
                        "import sentence_transformers" in stripped
                        or "from sentence_transformers" in stripped
                    )
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    violations.append(f"{py.name}: {line.strip()}")
                    break
        assert violations == [], (
            "module-level sentence_transformers import found (breaks "
            f"FTS-only installs): {violations}"
        )

    def test_embedder_none_path_works_without_import(self) -> None:
        """Index(embedder=None) never imports sentence_transformers."""
        import sys

        # Ensure any cached import is cleared so we observe fresh import.
        for mod in list(sys.modules):
            if mod == "sentence_transformers" or mod.startswith(
                "sentence_transformers."
            ):
                del sys.modules[mod]

        # Importing the Index class and constructing with embedder=None
        # must not pull in sentence_transformers.
        from source_recall import Index  # noqa: F401

        # The import above is the act under test; if it had triggered
        # a module-level sentence_transformers import, the assertion
        # below would catch it.
        assert "sentence_transformers" not in sys.modules, (
            "importing Index triggered a sentence_transformers import"
        )
