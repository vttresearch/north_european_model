"""Register of known dtype / NA contract violations.

Every test here is a **strict xfail**: it is expected to fail while the issue
exists, and pytest reports a hard error the moment it starts passing.  That is
the point -- a known issue that gets fixed must not keep its exemption silently.

Each entry pins the *exact* scenario that breaches the contract, so the entry
also serves as the reproduction case.  The corresponding loader case in
``tests/_common/loader_cases.py`` carries a ``known_contract_violation`` string
pointing back here, which downgrades its sweep failures to xfail and keeps the
sweep output readable.

To retire an entry: fix the code, watch this test XPASS, then delete both the
test and the ``known_contract_violation`` on the loader case.

These are judgment calls, not clear-cut slips -- fixing them changes behaviour
in ways that deserve a decision rather than a drive-by edit.
"""

import pandas as pd
import pytest

from src.source_data.source_data_loader import merge_unittypedata_into_unitdata
from tests._common.contracts import assert_normalized


@pytest.mark.xfail(
    strict=True,
    reason=(
        "merge_unittypedata_into_unitdata leaves unmatched unittype columns "
        "all-NA but typed Float64, breaking the all-NA-is-object rule"
    ),
)
def test_merge_leaves_all_na_float64_columns():
    """A unitdata file whose generator_ids match nothing in unittypedata.

    ``merge_unittypedata_into_unitdata`` (source_data_loader.py:616) left-joins
    the type defaults onto the unit rows.  When *no* row matches, every column
    contributed by the unittype side comes through entirely NA -- but keeps the
    ``Float64`` dtype it had in the unittype frame.

    That is the shape the all-NA-is-object rule exists to prevent: an empty
    float column and an empty string column must be indistinguishable so that
    no consumer can assume a dtype.  Here a consumer *can*, and will be wrong as
    soon as the same config runs against data where the join does match.

    Why this is a judgment call rather than a fix:
      - the cheap repair is to re-run ``standardize_df_dtypes`` after the merge,
        which costs a full frame copy on a hot path;
      - a narrower repair (retype only all-NA columns) is cheap, but silently
        papering over a total join failure may be worse than the dtype trap --
        a generator_id that matches nothing is a misconfiguration the user
        probably wants told about, loudly.

    Decide which, then delete this test.
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

    merged = merge_unittypedata_into_unitdata(unitdata, unittypedata)

    # Demonstrates the breach concretely before the contract call, so the
    # failure output shows *what* went wrong rather than only *that* it did.
    assert merged["eff00"].isna().all()
    assert str(merged["eff00"].dtype) == "Float64"

    assert_normalized(merged, where="merge_unittypedata_into_unitdata, no join match")
