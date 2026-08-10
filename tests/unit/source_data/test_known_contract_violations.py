"""Register of known dtype / NA contract violations -- **currently empty**.

An entry here is a **strict xfail**: expected to fail while the issue exists,
and pytest reports a hard error the moment it starts passing.  That is the
point -- a known issue that gets fixed must not keep its exemption silently.
Each entry pins the *exact* scenario that breaches the contract, so it doubles
as the reproduction case, and the matching loader case in
``tests/_common/loader_cases.py`` carries a ``known_contract_violation`` string
pointing back here, which downgrades its sweep failures to xfail and keeps the
sweep output readable.

To add one: write the tripwire here, then name it on the loader case.
To retire one: fix the code, watch the tripwire XPASS, then delete both.

What is left in this file is what retiring an entry leaves behind. A tripwire
asserts that something is broken, so it cannot simply lose its marker -- it has
to be rewritten into the positive statement of what the code now does, and it
stays here beside the register that explains why it exists.

The entry retired most recently was ``merge_unittypedata_into_unitdata``
leaving unmatched unittype columns all-NA but typed ``Float64``, which reached
the contract sweep as 48 xfail cases from one root cause.
"""

import pandas as pd

from src.source_data.source_data_loader import merge_row_by_row, merge_unittypedata_into_unitdata
from tests._common.contracts import assert_normalized
from tests._common.fixtures import FakeLogger


def test_merge_row_by_row_creates_the_key_columns_it_says_it_will():
    """Regression: a message that promised something the code did not do.

    Handed frames without one of its ``key_columns``, ``merge_row_by_row`` logs
    "Some key_columns not found and will be created as <NA>". It duly added them
    to each frame -- and then rebuilt the result from ``cols_union``, which had
    been computed before the columns existed, so they were dropped again.

    Not cosmetic: ``create_p_userconstraint`` believed the message, and a
    userconstraint sheet using only the 1st and 2nd dimension killed the build
    with a bare KeyError.
    """
    frame = pd.DataFrame(
        {
            "group": pd.Series(["g1"], dtype="object"),
            "parameter": pd.Series(["coefficient"], dtype="object"),
            "value": pd.Series([1.0], dtype="Float64"),
            "method": pd.Series(["replace"], dtype="object"),
        }
    )
    logger = FakeLogger()

    merged = merge_row_by_row(
        [frame, frame],
        logger,
        key_columns=["group", "1st dimension", "parameter"],
    )

    logger.assert_logged("will be created as <NA>", level="warn")
    assert "1st dimension" in merged.columns
    assert merged["1st dimension"].isna().all()


def test_merge_reports_and_retypes_when_nothing_matches():
    """A unitdata file whose generator_ids match nothing in unittypedata.

    ``merge_unittypedata_into_unitdata`` left-joins the type defaults onto the
    unit rows.  When *no* row matches, every column contributed by the unittype
    side comes through entirely NA -- and used to keep the ``Float64`` dtype it
    had in the unittype frame, which is the shape the all-NA-is-object rule
    exists to prevent: an empty float column and an empty text column must be
    indistinguishable so that no consumer can assume a dtype.

    Two things were wrong and both are asserted here, because fixing only the
    dtype would have left the worse half in place: a generator_id matching
    nothing is a misconfiguration, and it used to pass in complete silence.
    """
    unitdata = pd.DataFrame(
        {
            "country": pd.Series(["FI"], dtype="object"),
            "generator_id": pd.Series(["absent_from_unittypedata"], dtype="object"),
            "method": pd.Series(["replace"], dtype="object"),
        }
    )
    unittypedata = pd.DataFrame(
        {
            "generator_id": pd.Series(["gen1"], dtype="object"),
            "eff00": pd.Series([0.4], dtype="Float64"),
            "method": pd.Series(["replace"], dtype="object"),
        }
    )
    logger = FakeLogger()

    merged = merge_unittypedata_into_unitdata(unitdata, unittypedata, logger)

    # Named, not merely counted: the whole point is that the user can find the
    # typo without reading the code.
    logger.assert_logged("absent_from_unittypedata", level="warn")

    assert merged["eff00"].isna().all()
    assert str(merged["eff00"].dtype) == "object"

    assert_normalized(merged, where="merge_unittypedata_into_unitdata, no join match")


def test_merge_stays_quiet_and_typed_when_everything_matches():
    """Negative control for the warning and for the retype.

    A column the join *did* fill must keep its numeric dtype -- the retype is
    meant to reach all-NA columns only.
    """
    unitdata = pd.DataFrame(
        {
            "country": pd.Series(["FI"], dtype="object"),
            "generator_id": pd.Series(["gen1"], dtype="object"),
            "method": pd.Series(["replace"], dtype="object"),
        }
    )
    unittypedata = pd.DataFrame(
        {
            "generator_id": pd.Series(["gen1"], dtype="object"),
            "eff00": pd.Series([0.4], dtype="Float64"),
            "method": pd.Series(["replace"], dtype="object"),
        }
    )
    logger = FakeLogger()

    merged = merge_unittypedata_into_unitdata(unitdata, unittypedata, logger)

    assert merged["eff00"].tolist() == [0.4]
    assert str(merged["eff00"].dtype) == "Float64"
    logger.assert_not_logged("No unittypedata found")
