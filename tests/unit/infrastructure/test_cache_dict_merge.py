"""What survives a merge into a shared cache file, and what overwrites it.

``all_ts_domains.json`` and ``all_ts_domain_pairs.json`` are written by every
processor that runs and read back whole by the next phase. On a partial rerun
only some processors contribute, so the merge is the only thing keeping the
others' values in the file.

It kept the domains and lost the pairs. ``merge_dict_to_cache`` matched a pair
collection by its container, and the round trip through JSON is not
type-preserving: the pipeline passes a set of tuples, ``load_dict_from_cache``
hands back a list of tuples. Requiring a list on both sides matched the two ends
of that round trip against each other, so the branch never fired and the
overwrite fallback took it -- silently, because a shorter file still parses and
``_collect_gn_pairs`` unions what is left with nodedata, demanddata and
p_gnu_io.

Scalars were never affected: they come back as a set, which is what goes in.
"""

from __future__ import annotations

from pathlib import Path

from src.infrastructure.cache_manager import CacheManager
from tests._common.fixtures import FakeLogger, make_config

DOMAINS = "all_ts_domains.json"
PAIRS = "all_ts_domain_pairs.json"


def make_manager(tmp_path: Path) -> CacheManager:
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "data_files").mkdir(parents=True, exist_ok=True)
    return CacheManager(
        input_folder=tmp_path / "input",
        output_folder=output,
        config=make_config(),
        logger=FakeLogger(),
    )


class TestPairsAccumulate:
    """The regression: a set of tuples merged onto what a set of tuples became."""

    def test_a_second_merge_keeps_the_first_ones_pairs(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.merge_dict_to_cache({"grid_node": {("dheat", "FI00_dheat")}}, PAIRS)
        manager.merge_dict_to_cache({"grid_node": {("elec", "FI00_elec")}}, PAIRS)

        assert set(manager.load_dict_from_cache(PAIRS)["grid_node"]) == {
            ("dheat", "FI00_dheat"),
            ("elec", "FI00_elec"),
        }

    def test_one_processor_rerunning_alone_keeps_every_other_pair(self, tmp_path):
        """The build-shaped statement of the same thing.

        Three processors write grid_node on a full run; one of them re-runs on
        its own afterwards. The file has to still describe the whole model.
        """
        manager = make_manager(tmp_path)
        full_run = {
            ("dheat", "FI00_dheat"),
            ("elec", "FI00_elec"),
            ("elec", "SE01_elec"),
            ("water", "FI00_reservoir"),
        }
        manager.merge_dict_to_cache({"grid_node": set(full_run)}, PAIRS)

        # Only the district heating processor runs this time.
        manager.merge_dict_to_cache({"grid_node": {("dheat", "FI00_dheat")}}, PAIRS)

        assert set(manager.load_dict_from_cache(PAIRS)["grid_node"]) == full_run

    def test_a_pair_is_not_stored_twice(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.merge_dict_to_cache({"flow_node": {("PV", "FI00_elec")}}, PAIRS)
        manager.merge_dict_to_cache({"flow_node": {("PV", "FI00_elec")}}, PAIRS)

        assert manager.load_dict_from_cache(PAIRS)["flow_node"] == [("PV", "FI00_elec")]

    def test_merging_the_same_pairs_leaves_the_file_alone(self, tmp_path):
        """Order is kept, so an unchanged run does not churn the cache file."""
        manager = make_manager(tmp_path)
        pairs = {("elec", "SE01_elec"), ("dheat", "FI00_dheat"), ("elec", "FI00_elec")}
        manager.merge_dict_to_cache({"grid_node": set(pairs)}, PAIRS)

        written = (manager.cache_folder / PAIRS).read_text(encoding="utf-8")
        manager.merge_dict_to_cache({"grid_node": set(pairs)}, PAIRS)

        assert (manager.cache_folder / PAIRS).read_text(encoding="utf-8") == written


class TestDomainsStillAccumulate:
    """The half that always worked -- the fix must not cost it."""

    def test_scalar_domains_union(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.merge_dict_to_cache({"node": {"FI00_dheat"}}, DOMAINS)
        manager.merge_dict_to_cache({"node": {"FI00_elec"}}, DOMAINS)

        assert manager.load_dict_from_cache(DOMAINS)["node"] == {"FI00_dheat", "FI00_elec"}

    def test_a_new_key_is_taken_as_it_is(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.merge_dict_to_cache({"grid": {"elec"}}, DOMAINS)
        manager.merge_dict_to_cache({"flow": {"PV"}}, DOMAINS)

        loaded = manager.load_dict_from_cache(DOMAINS)
        assert loaded == {"grid": {"elec"}, "flow": {"PV"}}


class TestEverythingElseOverwrites:
    """Only collections accumulate; a flag records the latest answer."""

    def test_a_flag_takes_the_new_value(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.merge_dict_to_cache({"bb_excel_succesfully_built": True}, "general_flags.json")
        manager.merge_dict_to_cache({"bb_excel_succesfully_built": False}, "general_flags.json")

        flags = manager.load_dict_from_cache("general_flags.json")
        assert flags["bb_excel_succesfully_built"] is False

    def test_an_unrelated_flag_is_left_alone(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.merge_dict_to_cache(
            {"source_excel_run_successfully": True, "timeseries_run_successfully": True},
            "general_flags.json",
        )
        manager.merge_dict_to_cache({"timeseries_run_successfully": False}, "general_flags.json")

        flags = manager.load_dict_from_cache("general_flags.json")
        assert flags["source_excel_run_successfully"] is True
        assert flags["timeseries_run_successfully"] is False
