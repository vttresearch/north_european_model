"""Tests for the ``gams.transfer`` stand-in installed by ``tests/conftest.py``.

On a machine with the real GAMS Python API the stub is never installed, so
these tests exercise ``HostileModule`` directly.  Without them the stub would
only ever be exercised on machines nobody is looking at.
"""

import pytest

from tests.conftest import GAMS_IS_STUBBED, HostileModule


def test_the_stub_satisfies_an_import_statement():
    # `import gams.transfer` only needs the module object to exist; it must not
    # trip the __getattr__ guard.
    module = HostileModule("gams.transfer")
    assert module.__name__ == "gams.transfer"


@pytest.mark.parametrize("attribute", ["Container", "Parameter", "Set", "anything"])
def test_touching_any_attribute_raises_and_names_it(attribute):
    """The whole point: nothing may quietly pass against a mock.

    If code under test reaches into gams.transfer while stubbed, it must fail
    loudly and say which attribute it wanted -- silently returning a Mock would
    let a GDX-writing test report success without writing anything.
    """
    module = HostileModule("gams.transfer")
    with pytest.raises(RuntimeError) as excinfo:
        getattr(module, attribute)

    message = str(excinfo.value)
    assert attribute in message
    assert "@pytest.mark.gams" in message  # tells the reader what to do about it


def test_gams_marked_tests_are_skipped_exactly_when_the_stub_is_active():
    """Consistency check between the marker policy and the reported header.

    This test does not assert which state the machine is in -- it asserts the
    two signals agree, so a green run can never misreport the tier it ran.
    """
    assert isinstance(GAMS_IS_STUBBED, bool)

    try:
        import gams.transfer  # noqa: F401

        real_api_present = True
    except Exception:
        real_api_present = False

    assert GAMS_IS_STUBBED is not real_api_present


@pytest.mark.gams
def test_marker_reaches_the_real_api_when_one_is_present():
    """Sentinel for the @pytest.mark.gams path.

    Skipped automatically while stubbed (see pytest_collection_modifyitems);
    when it runs, it proves the marker does not block a genuine API.
    """
    import gams.transfer as gt

    assert gt.Container is not None
