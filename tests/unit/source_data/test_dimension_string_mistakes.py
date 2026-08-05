"""What a mistyped dimension string does to an overlay.

Overlays work by key: a later row overrides an earlier one when their key
columns match. Dimension strings -- ``node_suffix``, ``unit_name_prefix``, the
transfer suffixes -- are part of that key *and* are baked into the generated
node and unit names before the merge runs.

So a suffix that is added, omitted or mistyped does not fail to override. It
**instantiates a second node or unit**, silently, and the value the author meant
to change stays as it was. There is no warning, because nothing invalid
happened: the pipeline cannot tell an intended new node from a typo.

Case is the sharp edge. The project already lowercases the fields it joins on
(``scenario``, ``generator_id``, ``method`` -- ``normalize_dataframe``'s
``lowercase_col_values``) and reconstructs the first-seen form for presentation
(``compile_domain_df``). ``node_suffix`` and ``unit_name_prefix`` are not in
that list, so ``dh`` and ``DH`` produce two distinct labels -- while GAMS
considers them one, and refuses the GDX outright:

    Exception: Encountered data errors with symbol `node`. Possible causes are
    from duplicate records ... GDX file was not created successfully.

These tests pin the behaviour rather than assert it is right. Where it is
clearly wrong -- the workbook GAMS cannot read -- there is an xfail.
"""

import pandas as pd
import pytest

import src.source_data.source_data_loader as loader
from tests._common.fixtures import FakeLogger

UNITTYPES = [
    {"generator_id": "chp", "unittype": "CHPbio", "grid_output1": "elec", "method": "replace"}
]


def _nodedata(*rows) -> pd.DataFrame:
    """Run the nodedata chain in pipeline order, up to and including the merge."""
    logger = FakeLogger()
    df = loader.normalize_dataframe(pd.DataFrame(list(rows)), "nodedata", logger)
    df = loader.drop_underscore_values(df, "nodedata", logger)
    df = loader.build_node_column(df, logger)
    if df.empty:
        return df, logger
    merged = loader.merge_row_by_row(
        [df], logger, key_columns=["country", "grid", "node"]
    )
    return merged, logger


def _node_row(suffix=None, limit=None, method="replace"):
    row = {"country": "FI", "grid": "heat", "method": method, "upwardlimit": limit}
    if suffix is not None:
        row["node_suffix"] = suffix
    return row


def _nodes(df):
    return list(df["node"]) if "node" in df.columns else []


class TestAMatchingSuffixOverrides:
    def test_the_intended_case_still_works(self):
        merged, _ = _nodedata(_node_row("dh", 100), _node_row("dh", 200))
        assert _nodes(merged) == ["FI_heat_dh"]
        assert merged.iloc[0]["upwardlimit"] == 200

    def test_surrounding_whitespace_is_forgiven(self):
        # normalize_dataframe strips string columns, so a padded cell -- the
        # commonest spreadsheet accident -- still matches.
        merged, _ = _nodedata(_node_row("dh", 100), _node_row("  dh  ", 200))
        assert _nodes(merged) == ["FI_heat_dh"]
        assert merged.iloc[0]["upwardlimit"] == 200


class TestAMistypedSuffixInstantiatesInstead:
    """Each of these creates a second node. The override does not happen."""

    @pytest.mark.parametrize(
        "suffix, expected_node",
        [
            pytest.param("dhh", "FI_heat_dhh", id="typo"),
            pytest.param("", "FI_heat", id="omitted"),
            pytest.param("dh2", "FI_heat_dh2", id="different-suffix"),
        ],
    )
    def test_a_replace_overlay_adds_a_node_rather_than_editing_one(
        self, suffix, expected_node
    ):
        merged, logger = _nodedata(_node_row("dh", 100), _node_row(suffix, 200))

        assert set(_nodes(merged)) == {"FI_heat_dh", expected_node}
        # The value the author meant to change is untouched...
        original = merged[merged["node"] == "FI_heat_dh"].iloc[0]
        assert original["upwardlimit"] == 100
        # ...and nothing warned, because nothing invalid happened.
        logger.assert_no_errors()

    def test_an_add_overlay_does_the_same(self):
        # 'add' on an unmatched key initialises a new record rather than
        # accumulating onto the intended one.
        merged, _ = _nodedata(_node_row("dh", 100), _node_row("dhh", 50, method="add"))
        values = dict(zip(merged["node"], merged["upwardlimit"]))
        assert values == {"FI_heat_dh": 100, "FI_heat_dhh": 50}

    def test_a_suffix_containing_an_underscore_loses_the_row_entirely(self):
        """Different failure, and the only one that is reported.

        Underscore separates the parts of a node name, so a value containing one
        is dropped by drop_underscore_values -- with a warning. The override is
        lost, but at least loudly.
        """
        merged, logger = _nodedata(_node_row("dh", 100), _node_row("_dh", 200))

        assert _nodes(merged) == ["FI_heat_dh"]
        assert merged.iloc[0]["upwardlimit"] == 100
        logger.assert_logged("Underscores detected", level="warn")


class TestUnitPrefixesBehaveTheSameWay:
    def test_a_mistyped_unit_prefix_instantiates_a_second_unit(self):
        logger = FakeLogger()
        unittypes = loader.normalize_dataframe(pd.DataFrame(UNITTYPES), "u", logger)
        units = loader.normalize_dataframe(
            pd.DataFrame(
                [
                    {"country": "FI", "generator_id": "chp", "unit_name_prefix": "a",
                     "capacity_output1": 100, "method": "replace"},
                    {"country": "FI", "generator_id": "chp", "unit_name_prefix": "aa",
                     "capacity_output1": 200, "method": "replace"},
                ]
            ),
            "u",
            logger,
        )
        units = loader.build_unittype_unit_column(units, unittypes, logger)
        merged = loader.merge_row_by_row(
            [units], logger,
            key_columns=["country", "generator_id", "unit_name_prefix"],
        )

        assert sorted(merged["unit"]) == ["FI_a_CHPbio", "FI_aa_CHPbio"]
        assert sorted(merged["capacity"]) == [100, 200]


class TestCaseIsTheDangerousOne:
    """``dh`` and ``DH`` are two labels here and one label to GAMS."""

    def test_a_case_difference_produces_two_distinct_nodes(self):
        merged, logger = _nodedata(_node_row("dh", 100), _node_row("DH", 200))

        assert set(_nodes(merged)) == {"FI_heat_dh", "FI_heat_DH"}
        logger.assert_no_errors()

    def test_uppercase_suffixes_are_preserved_because_they_are_meaningful(self):
        """Not a defect: real suffixes are uppercase on purpose.

        Production data uses city codes -- HKI, TKU, TRE, ESP, VAN, OUL, JKL --
        for district-heating subregions. Folding suffixes to lower case would
        rename 20 live nodes and break every downstream reference to them, so
        the fields deliberately sit outside ``lowercase_col_values``.

        Which means the collision below cannot be solved by normalising. It has
        to be *detected*.
        """
        merged, _ = _nodedata(_node_row("HKI", 100))
        assert _nodes(merged) == ["FI_heat_HKI"]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "merge_row_by_row compares keys case-sensitively, so labels that "
            "GAMS considers identical are merged as two rows"
        ),
    )
    def test_labels_differing_only_in_case_merge_into_one_first_form_wins(self):
        """Open: the merge key does not follow the rule the domains follow.

        The established convention is case-insensitive identity with the
        first-presented form preserved -- which is exactly what
        ``compile_domain_df`` does when it builds the ``node`` sheet. The merge
        key does not: it compares raw strings, so ``dh`` and ``DH`` are two keys
        here and one label in GAMS.

        The damage is asymmetric, which is what makes it hard to trace. The
        ``node`` domain sheet dedupes case-insensitively and lists one node,
        while ``p_gn`` keeps both rows, so GDXXRW receives two records for a
        single label. Writing the same pair through gams.transfer fails with
        nothing more useful than::

            Encountered data errors with symbol `node`. Possible causes are from
            duplicate records ... GDX file was not created successfully.

        Lowercasing the fields would not do: production suffixes are uppercase
        city codes and folding them would rename 20 live nodes. Applying the
        domain rule to the merge key would -- the two rows become one, carrying
        the first form, and an overlay written ``DH`` against a base written
        ``dh`` does what its author meant instead of silently forking the node.

        Real data has no collisions today, so this is currently a no-op there.
        Not made here: it changes key comparison in the most failure-prone
        function in the repo, and that deserves a decision of its own.
        """
        merged, _ = _nodedata(_node_row("dh", 100), _node_row("DH", 200))

        assert _nodes(merged) == ["FI_heat_dh"]        # first form preserved
        assert merged.iloc[0]["upwardlimit"] == 200    # the overlay applied
