"""Session-wide test setup.

Three responsibilities, in this order:

1. Put the repo root on ``sys.path`` so ``import src.<...>`` resolves however
   pytest was invoked.
2. Install a stub for ``gams.transfer`` when the real GAMS Python API is absent,
   so that the bulk of the suite runs on a machine without a GAMS install.
3. Register the handful of session fixtures. Everything else is an imported
   function -- see ``tests/_common/__init__.py`` for why.
"""

import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class RefusingModule(types.ModuleType):
    """A module that satisfies an ``import`` and refuses everything after it.

    Module-level (not nested inside the installer) so that ``tests/meta/`` can
    exercise it even on a machine where the real GAMS API is present and the
    stub therefore never gets installed.  A safety net that only runs on other
    people's machines is one that rots unnoticed.
    """

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"gams.transfer is STUBBED in this test session, but something "
            f"reached gams.transfer.{name}. Either mark the test "
            f"@pytest.mark.gams (it will skip without GAMS), or stop the code "
            f"under test from writing GDX."
        )


def _install_gams_stub() -> bool:
    """Install a deliberately strict ``gams.transfer`` stub if the real one is absent.

    ``src/GDX_exchange.py`` imports ``gams.transfer`` at module level, and
    ``timeseries_pipeline.py`` / ``timeseries_processor.py`` import it in turn --
    so ``import build_input_data`` itself requires the GAMS Python API.  Guarding
    every affected test with ``importorskip`` would silently skip the whole
    timeseries tier, the ``_patch_gams_file_content`` tests and the entrypoint
    tier on any machine without GAMS: a large, invisible-green hole.

    The stub therefore satisfies the *import*, and nothing more.  Touching any
    attribute on it raises, naming the attribute, so a test can never quietly
    pass against a mock instead of the real API.  Tests that genuinely need GDX
    carry ``@pytest.mark.gams`` and are skipped while the stub is active.

    Returns True if the stub was installed (i.e. the real API is missing).
    """
    try:
        import gams.transfer  # noqa: F401

        return False
    except Exception:
        pass

    gams_pkg = types.ModuleType("gams")
    transfer_mod = RefusingModule("gams.transfer")
    gams_pkg.transfer = transfer_mod
    sys.modules.setdefault("gams", gams_pkg)
    sys.modules.setdefault("gams.transfer", transfer_mod)
    return True


GAMS_IS_STUBBED = _install_gams_stub()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def src_files_dir() -> Path:
    """The real ``src_files/``.

    Test input folders reuse the genuine ``indexSheet.xlsx`` and ``GAMS_files/``
    from here rather than faking them, so that the code paths reading them are
    exercised as they run in production.
    """
    return _REPO_ROOT / "src_files"


@pytest.fixture
def fake_logger():
    """Convenience only -- ``FakeLogger()`` at the call site is equally fine."""
    from tests._common.fixtures import FakeLogger

    return FakeLogger()


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.gams`` tests while the GAMS stub is standing in."""
    if not GAMS_IS_STUBBED:
        return
    skip = pytest.mark.skip(reason="real gams.transfer unavailable; conftest stub active")
    for item in items:
        if "gams" in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config):
    """State which tier actually ran, so a green run cannot hide a stubbed one."""
    state = "STUBBED (gams tests skipped)" if GAMS_IS_STUBBED else "real API"
    return f"gams.transfer: {state}"
