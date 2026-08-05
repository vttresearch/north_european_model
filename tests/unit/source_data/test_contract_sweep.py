"""Contract sweeps over the source-data loader functions.

Three properties, each applied to every function in
``tests/_common/loader_cases.py`` rather than written per function:

1. whatever a user typed into a cell, the output still satisfies the dtype /
   NA contract;
2. an all-``pd.NA`` ``object`` column -- the "assume nothing" state -- is
   tolerated wherever ``Float64`` would normally appear;
3. the source stage never collapses ``pd.NA`` into ``0``.

Property 2 is the cascade bug, as a regression test: empty string columns and
empty float columns both become all-NA, and code that assumes a dtype crashes.
The fix was to type them ``object``; the obligation that creates is on every
consumer, and this is where that obligation is checked.

These sweeps are marked ``contract`` so they can be run alone:
``python -m pytest -m contract``
"""

import pandas as pd
import pytest

from tests._common.contracts import (
    NASTY_CELLS,
    assert_no_na_became_zero,
    assert_normalized,
    frame_with_blank_column,
    frame_with_cell,
    nasty_id,
)
from tests._common.fixtures import FakeLogger
from tests._common.loader_cases import LOADER_CASES, SWEEP_COLUMNS
from src.source_data.source_data_loader import normalize_dataframe

pytestmark = pytest.mark.contract


def _check(case, df: pd.DataFrame, *, where: str) -> None:
    """Apply the contract, honouring a case's known-violation exemption.

    A known violation is downgraded to xfail rather than removed from the
    sweeps: the function keeps being exercised (so a *new*, different breach
    still surfaces as a hard failure), while the register in
    ``test_known_contract_violations.py`` carries the strict tripwire that fires
    when the underlying issue is fixed.
    """
    try:
        assert_normalized(df, where=where)
    except AssertionError:
        if case.known_contract_violation:
            pytest.xfail(case.known_contract_violation)
        raise


def _normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Put a frame through the gatekeeper.

    Most loader functions document a precondition of normalized input
    (``merge_row_by_row``'s docstring states it outright at :875).  Feeding them
    raw frames would test a situation the pipeline never produces.
    """
    return normalize_dataframe(df, "sweep-setup", FakeLogger())


@pytest.mark.parametrize("cell", NASTY_CELLS, ids=nasty_id)
def test_normalize_dataframe_output_satisfies_the_contract(cell):
    """The gatekeeper must produce the canonical shape from anything.

    Everything downstream is entitled to assume this, so if it fails here the
    contract is not merely violated -- it never held in the first place.
    """
    raw = frame_with_cell(SWEEP_COLUMNS, cell)
    out = normalize_dataframe(raw, "sweep", FakeLogger())
    assert_normalized(out, where=f"normalize_dataframe given {cell!r}")


@pytest.mark.parametrize("case", LOADER_CASES, ids=str)
@pytest.mark.parametrize("cell", NASTY_CELLS, ids=nasty_id)
def test_loader_output_satisfies_the_contract(case, cell):
    """Every loader function preserves the canonical shape."""
    raw = frame_with_cell(SWEEP_COLUMNS, cell)
    df = _normalized(raw) if case.needs_normalized_input else raw

    out = case.call(df, FakeLogger())

    _check(case, out, where=f"{case.name} given {cell!r}")


@pytest.mark.parametrize("case", LOADER_CASES, ids=str)
@pytest.mark.parametrize("blank_column", SWEEP_COLUMNS)
def test_all_na_object_column_is_tolerated(case, blank_column):
    """The cascade bug, as a property.

    An all-``pd.NA`` column is typed ``object`` precisely so that no assumption
    is baked in about what it holds (utils.py:90-91).  The price is that every
    consumer must cope with seeing ``object`` where it expected ``Float64``:
    handle it, or reject it with a logged message -- but never crash, and never
    silently coerce it into something else.

    A crash here is a real bug in the function under test, not a test artefact.
    """
    raw = frame_with_blank_column(SWEEP_COLUMNS, blank_column)
    df = _normalized(raw) if case.needs_normalized_input else raw
    logger = FakeLogger()

    out = case.call(df, logger)

    _check(case, out, where=f"{case.name} with {blank_column!r} all-NA")


@pytest.mark.parametrize("case", LOADER_CASES, ids=str)
def test_source_stage_never_collapses_na_to_zero(case):
    """Boundaries 1-2: ``pd.NA`` and ``0`` stay distinct through the source stage.

    ``method=replace`` has to be able to overwrite a value with a genuine zero.
    If NA and 0 are conflated before the merge runs, that distinction is already
    gone.  ``0 = NA`` is a GAMS convention and belongs at the inputData.xlsx /
    GDX boundary, not upstream of it.
    """
    # The full sweep schema, so that every case finds the columns it needs;
    # only 'capacity' carries the NA under test.
    raw = frame_with_cell(SWEEP_COLUMNS, "FI", rows=2, filler="SE")
    raw["grid"] = ["elec", "elec"]
    raw["generator_id"] = ["gen1", "gen1"]
    raw["scenario"] = ["test", "test"]
    raw["year"] = [2030, 2030]
    raw["method"] = ["replace", "replace"]
    raw["capacity"] = [pd.NA, 5.0]
    df = _normalized(raw) if case.needs_normalized_input else raw

    out = case.call(df, FakeLogger())

    assert_no_na_became_zero(df, out, where=case.name, columns=["capacity"])


def test_the_sweep_table_covers_the_public_loader_surface():
    """Guard against a new transform silently escaping the sweeps.

    If someone adds a public function to source_data_loader.py, it should either
    be swept or be deliberately excluded here with a reason. Without this check
    the sweeps quietly stop being comprehensive.
    """
    import inspect

    import src.source_data.source_data_loader as loader

    # read_input_excels reads from disk rather than transforming a frame, so it
    # cannot take part in a frame-in/frame-out sweep; it is covered separately
    # in test_read_input_excels.py.
    deliberately_excluded = {"read_input_excels"}

    public = {
        name
        for name, fn in inspect.getmembers(loader, inspect.isfunction)
        if fn.__module__ == loader.__name__ and not name.startswith("_")
    }
    swept = {case.name for case in LOADER_CASES}

    missing = public - swept - deliberately_excluded
    assert not missing, (
        f"loader function(s) {sorted(missing)} are not in LOADER_CASES. "
        f"Add an entry to tests/_common/loader_cases.py, or list them in "
        f"deliberately_excluded with a reason."
    )
