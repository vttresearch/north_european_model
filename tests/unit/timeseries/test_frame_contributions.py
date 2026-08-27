"""Boundary 8, from the processor's side: what a contribution may be.

``tests/unit/source_data/test_contributions.py`` covers the rules themselves.
This file covers the gate as ``ProcessorRunner`` applies it -- with a real
misbehaving processor per path, the same way the main_result contract is tested,
because a guard nobody drives is a guard nobody can rely on.

Two things separate this boundary from the main_result one. A bad contribution
costs the *contribution*, not the GDX: the time series is still correct and still
written, and only the claim about the source data is dropped. And the reason is
always logged naming the processor, because a processor author is the only person
who can act on it.
"""

import pandas as pd
import pytest

from tests._common.processor_contract import hourly_frame, run_fake_processor

#: A well-formed nodedata contribution, as a source expression for the template.
GOOD_NODEDATA = (
    '{"nodedata": pd.DataFrame({'
    '"grid": ["elec"], "node": ["FI00_elec"], "influx": [-5.0]})}'
)


def run_with(tmp_path, frames: str, **kwargs):
    """A processor whose time series is fine and whose contribution is `frames`."""
    return run_fake_processor(
        tmp_path, hourly_frame(nodes=("FI00_elec",)), frames=frames, **kwargs
    )


class TestWhatIsRefused:
    def test_a_name_that_is_not_a_source_table(self, tmp_path):
        run = run_with(tmp_path, '{"nonsense": pd.DataFrame({"a": [1]})}')

        run.logger.assert_logged("not one of the source data tables", level="warn")
        assert not run.result.frames

    def test_something_that_is_not_a_dataframe(self, tmp_path):
        run = run_with(tmp_path, '{"nodedata": "a string"}')

        run.logger.assert_logged("expected pd.DataFrame", level="error")
        assert not run.result.frames

    def test_a_frame_missing_its_key_column(self, tmp_path):
        run = run_with(tmp_path, '{"nodedata": pd.DataFrame({"node": ["FI00_elec"]})}')

        run.logger.assert_logged("key column", level="error")
        assert not run.result.frames

    def test_a_blank_value_inside_a_key_column(self, tmp_path):
        """A blank key becomes the GAMS set element ''.

        Same reasoning as a blank dimension in main_result: it is a broken key
        rather than a gap, so it is refused instead of filled.
        """
        run = run_with(
            tmp_path,
            '{"nodedata": pd.DataFrame({"grid": ["elec", None], '
            '"node": ["FI00_elec", "SE00_elec"]})}',
        )

        run.logger.assert_logged("blank", level="error")
        assert not run.result.frames

    def test_an_empty_frame(self, tmp_path):
        # Not silent: the processor put the frame there on purpose, so producing
        # nothing means its own build came up empty.
        run = run_with(
            tmp_path, '{"nodedata": pd.DataFrame(columns=["grid", "node"])}'
        )

        run.logger.assert_logged("empty", level="warn")
        assert not run.result.frames

    def test_the_message_names_the_processor(self, tmp_path):
        # The fix is in the processor's code, so the author has to be able to
        # tell which of them it is.
        run = run_with(tmp_path, '{"nonsense": pd.DataFrame({"a": [1]})}')

        assert run.logger.matching("FakeProcessor", level="warn")


class TestARefusalCostsOnlyTheContribution:
    @pytest.mark.gams
    def test_the_gdx_is_still_written(self, tmp_path):
        """The time series was never in question.

        Refusing it too would turn a wrong claim about one node into no data for
        every node the processor covers.
        """
        run = run_with(tmp_path, '{"nonsense": pd.DataFrame({"a": [1]})}')

        assert run.gdx_files

    def test_one_bad_contribution_does_not_take_a_good_one_with_it(self, tmp_path):
        run = run_with(
            tmp_path,
            '{"nonsense": pd.DataFrame({"a": [1]}), '
            '"nodedata": pd.DataFrame({"grid": ["elec"], "node": ["FI00_elec"]})}',
        )

        assert set(run.result.frames) == {"nodedata"}


class TestWhatIsAccepted:
    def test_a_well_formed_contribution_comes_back(self, tmp_path):
        run = run_with(tmp_path, GOOD_NODEDATA)

        assert set(run.result.frames) == {"nodedata"}
        run.logger.assert_no_errors()

    def test_it_is_cached_under_the_spec_name_not_the_processor_name(self, tmp_path):
        """Three specs share VRE_PECD, so the processor's name is not an identity.

        Caching by it had the three overwrite each other, and only the last one's
        answer survived into the next build.
        """
        run = run_fake_processor(
            tmp_path,
            hourly_frame(nodes=("FI00_elec",)),
            frames=GOOD_NODEDATA,
            name="SharedProcessor",
        )

        assert set(run.cache_manager.processor_frames) == {"SharedProcessor"}

    def test_contributing_nothing_is_the_ordinary_case(self, tmp_path):
        # Most processors have nothing to add: the nodes they build series for
        # are in the workbooks already, which is how they found them.
        run = run_with(tmp_path, "{}")

        assert run.result.frames == {}
        run.logger.assert_no_errors()


class TestValuesTheModelDoesNotKnow:
    """The check that replaced the domain sheets.

    A processor's dimension values used to be added to the workbook's node, grid
    and flow sheets automatically. That existed to stop GAMS aborting when a
    processor emitted every country regardless of the model's topology, and
    processors filter by country_codes now -- so it was doing nothing, while a
    genuine typo in a node name became a new node rather than a complaint.
    """

    def _with_nodedata(self, tmp_path, nodes, **kwargs):
        return run_fake_processor(
            tmp_path,
            hourly_frame(nodes=nodes),
            source_data={"nodedata": pd.DataFrame({
                "grid": ["elec"], "node": ["FI00_elec"],
            })},
            **kwargs,
        )

    def test_a_node_no_table_has_is_named(self, tmp_path):
        run = self._with_nodedata(tmp_path, ("FI00_elec", "TYPO_elec"))

        run.logger.assert_logged("TYPO_elec", level="warn")

    def test_a_node_the_workbook_has_is_not(self, tmp_path):
        # Negative control: the warning must not fire on the ordinary path.
        run = self._with_nodedata(tmp_path, ("FI00_elec",))

        run.logger.assert_not_logged("does not have")

    def test_a_node_the_processor_contributes_itself_is_not(self, tmp_path):
        """Contributing the node is exactly how a processor introduces one.

        Warning about it would make the two halves of the same statement
        contradict each other.
        """
        run = self._with_nodedata(
            tmp_path,
            ("FI00_elec", "NEW_elec"),
            frames='{"nodedata": pd.DataFrame({"grid": ["elec"], "node": ["NEW_elec"]})}',
        )

        run.logger.assert_not_logged("NEW_elec")

    def test_nothing_is_checked_when_the_tables_were_not_loaded(self, tmp_path):
        """An empty frame means the source excels were skipped, not that the
        model has no nodes. Checking against it would warn about everything."""
        run = run_fake_processor(tmp_path, hourly_frame(nodes=("ANYTHING",)))

        run.logger.assert_not_logged("does not have")

    @pytest.mark.gams
    def test_the_series_is_still_written(self, tmp_path):
        # A warning, not a refusal: the model is the one the user's data
        # describes, and a series for a node Backbone lacks is inert.
        run = self._with_nodedata(tmp_path, ("FI00_elec", "TYPO_elec"))

        assert run.gdx_files
        run.logger.assert_no_errors()
