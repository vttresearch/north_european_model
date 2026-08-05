"""Which GAMS installation ``gams.transfer`` binds to.

A bare ``gt.Container()`` binds to whatever the machine-global setting names --
on Windows the registry key ``HKCU\\Software\\Classes\\gams.location``, i.e.
whichever GAMS installer ran last.  On a machine with several GAMS versions that
is usually not the one ``gamsapi`` is pinned to, and every container then warns:

    UserWarning: The GAMS version (53.5.0) differs from the API version (47.4.1)

Reads and writes still succeed, which is exactly why it goes unnoticed: the
binding is arbitrary rather than chosen.  ``resolve_gams_system_directory``
picks it deliberately, honouring the same ``BB_GAMS_API`` variable the parent
Backbone repo uses so one setting covers both.
"""

from pathlib import Path

import pytest

import src.GDX_exchange as gdx


class TestResolution:
    def test_bb_gams_api_wins(self, monkeypatch):
        monkeypatch.setenv("BB_GAMS_API", r"C:\GAMS\99")
        assert gdx.resolve_gams_system_directory() == r"C:\GAMS\99"

    def test_an_explicit_pin_is_honoured_even_if_it_does_not_exist(self, monkeypatch):
        # Deliberate: if someone pins a path, a silent fallback to a different
        # GAMS is worse than the failure they will see from gams.transfer.
        monkeypatch.setenv("BB_GAMS_API", r"Z:\nonexistent\GAMS")
        assert gdx.resolve_gams_system_directory() == r"Z:\nonexistent\GAMS"

    def test_falls_back_to_the_install_matching_the_installed_gamsapi(self, monkeypatch):
        monkeypatch.delenv("BB_GAMS_API", raising=False)
        monkeypatch.setattr(gdx, "_installed_gamsapi_major", lambda: 47)
        monkeypatch.setattr(Path, "is_dir", lambda self: str(self) == "C:\\GAMS\\47")

        assert gdx.resolve_gams_system_directory() == r"C:\GAMS\47"

    def test_returns_none_when_no_matching_install_exists(self, monkeypatch):
        # None means "let gams.transfer auto-discover" -- the previous behaviour.
        # An unresolvable binding is a noisy warning, not a reason to fail a build.
        monkeypatch.delenv("BB_GAMS_API", raising=False)
        monkeypatch.setattr(gdx, "_installed_gamsapi_major", lambda: 47)
        monkeypatch.setattr(Path, "is_dir", lambda self: False)

        assert gdx.resolve_gams_system_directory() is None

    def test_returns_none_when_gamsapi_version_is_unknown(self, monkeypatch):
        monkeypatch.delenv("BB_GAMS_API", raising=False)
        monkeypatch.setattr(gdx, "_installed_gamsapi_major", lambda: None)
        assert gdx.resolve_gams_system_directory() is None

    def test_ignores_a_gamsapi_too_old_for_the_transfer_api(self, monkeypatch):
        # gams.transfer needs GAMS 45+; pointing at an older install would fail
        # in a far less obvious way than not binding at all.
        monkeypatch.delenv("BB_GAMS_API", raising=False)
        monkeypatch.setattr(gdx, "_installed_gamsapi_major", lambda: 24)
        assert gdx.resolve_gams_system_directory() is None


class TestNoBareContainers:
    def test_every_container_goes_through_new_container(self):
        """Guard the rule rather than trusting people to remember it.

        One bare ``gt.Container(...)`` anywhere reintroduces the arbitrary
        binding for that call site only, which is the hardest kind of instance
        to notice.
        """
        import re

        repo_root = Path(__file__).resolve().parents[2]
        offenders = []
        for path in repo_root.rglob("*.py"):
            parts = set(path.parts)
            if parts & {"tests", ".git", "__pycache__", "dev", "scratch"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"\bgt\.Container\(", text):
                line = text[: match.start()].count("\n") + 1
                # The definition of new_container is the one legitimate site.
                if path.name == "GDX_exchange.py":
                    continue
                offenders.append(f"{path.relative_to(repo_root)}:{line}")

        assert not offenders, (
            "bare gt.Container(...) found at "
            + ", ".join(offenders)
            + " -- use GDX_exchange.new_container() so the GAMS binding is chosen "
            "rather than inherited from a machine-global registry key."
        )


@pytest.mark.gams
class TestAgainstTheRealApi:
    def test_a_container_is_created_without_a_version_mismatch_warning(self, recwarn):
        """The end result: no warning on this machine's real installation.

        Skipped when the conftest stub is standing in for gams.transfer, since
        there is then no real binding to get wrong.
        """
        gdx.new_container()
        mismatches = [
            w for w in recwarn if "differs from the API version" in str(w.message)
        ]
        assert not mismatches, [str(w.message) for w in mismatches]
