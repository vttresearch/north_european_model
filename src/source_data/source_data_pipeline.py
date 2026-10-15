"""
Source Data Pipeline
====================

Purpose
-------
SourceDataPipeline is the first and only stage of the pipeline that is aware of
the scenario context. It reads source Excel files, filters them to the current
(scenario, year, build) combination, merges them row-by-row, and exposes one
normalized DataFrame per data category. Downstream stages (time series processing,
BBExcelPipeline) receive scenario-neutral data and do not need to repeat scenario
filtering.

Scenario, year and build
------------------------
build_input_data.py drives execution by iterating over scenarios x scenario_years x
the builds of the config's `variants`. For each combination a new pipeline instance
is created and run(). A config without `variants` has one build, '', which chooses
every variant's default and adds no folder-name segment.

Within each data category the filtering works in three steps:

1. apply_whitelist() retains only rows whose 'scenario' column matches the current
   scenario or the catch-all value 'all'. The 'year' column is similarly filtered to
   the target year or the catch-all value 1. This means a single source Excel file
   can contain rows for multiple scenarios and years; only the relevant rows survive.

2. The variant selection (source_data_variants.py) keeps untagged rows, the ':all'
   rows of a variant wherever the country has an option, and the rows of the option
   the build chose for that country. Rows of other options are not read.

3. merge_row_by_row() processes the kept rows in two passes: the base (untagged and
   ':all' rows) in file load order, then the chosen options' rows in file load order.
   Each row carries a 'method' value (replace / replace-partial / add / multiply /
   remove / ...) that controls how it interacts with previously seen rows for the
   same key. Base data can be overwritten or supplemented in further input data
   files, and an option overrides the base whatever the file order.


Interface
---------
Input dataframes are user given data and need to be handled very carefully and methodologically.

All output DataFrames expose exactly three column dtypes: 
- Float64 (pandas nullable float), 
- object,  
- string. 

Each empty cell is converted to pd.NA as a sign of no data. Zero means data.

Data conventions
-----------------------------------------
- Column names: lowercase, whitespace-stripped.
- String values in key and categorical columns: lowercase, whitespace-stripped.
- Numeric columns: pd.NA for missing values. Zero is meaningful data.
- String/object columns: pd.NA for missing values. Empty string is not used.
- Blacklists: apply_blacklist() is applied before apply_whitelist() for datasets that
  support exclude_grids and exclude_nodes (demanddata, nodedata, unitdata,
  transferdata). Rows whose grid/node matches an excluded value are dropped before any
  further scenario or country filtering takes place.
- Metadata columns (_source_file, _source_sheet, method): dropped by merge_row_by_row.
- Scenario and year columns: retained in most output DataFrames. Downstream stages can
  drop them; they carry no further information because filtering has already been
  applied. Exception: df_userconstraintdata drops 'scenario', 'year', and 'country'
  before merging because its key structure makes them redundant.
- Topology columns (node, from_node, to_node): derived from component columns (country,
  grid, suffix) by data_loader helpers during run(); the component columns are also
  retained.


Output DataFrames
-----------------
After run() the following attributes are populated (empty DataFrame if no source
files are configured):

  df_nodedata            node parameters (fuel costs, emissions, storage)  key: country, grid, node
  df_emissiondata        emission factors                key: emission, group
  df_demanddata          demand parameters               key: country, grid, node
  df_unitdata            unit capacity parameters        key: country, unittype, unit_name_prefix
                         NOTE: df_unitdata is the MERGED result — type-level defaults
                         from df_unittypedata are incorporated here via
                         merge_unittypedata_into_unitdata().  df_unittypedata is
                         NOT exposed as a public attribute after run().
  df_transferdata        interconnector parameters       key: from_country, from_suffix, to_country, to_suffix, grid
  df_userconstraintdata  custom constraint parameters    key: group, 1st dimension,
                         2nd dimension, 3rd dimension, 4th dimension, parameter
  df_boundarydata        node state boundaries, long     key: grid, node, param_gnboundarytypes
                         NOTE: derived, not read. build_boundarydata() melts
                         nodedata's wide boundary columns (upwardlimit, maxSpill,
                         ...) into one row per type. A timeseries processor adds
                         to the same table, which is why it is long: a boundary
                         that comes from a series states useTimeseries there.
"""

import src.source_data.source_data_contributions as source_data_contributions
import src.source_data.source_data_loader as data_loader
import src.source_data.source_data_variants as source_data_variants
import src.utils as utils
import pandas as pd
from src.source_data.source_data_inputs import SourceDataPipelineInputs


class SourceDataPipeline:
    """
    SourceDataPipeline handles reading, merging, filtering, and validating all input Excel files.
    """

    def __init__(self, inputs: SourceDataPipelineInputs):
        self.config = inputs.config
        self.input_folder = inputs.input_folder
        self.data_folder = inputs.input_folder / "data_files"

        self.scenario = inputs.scenario
        self.scenario_year = inputs.scenario_year
        self.variant = inputs.variant
        self.country_codes = inputs.country_codes
        self.logger = inputs.logger

        self.df_demanddata = pd.DataFrame()
        self.df_transferdata = pd.DataFrame()
        self._df_unittypedata = pd.DataFrame()  # internal; merged into df_unitdata before run() ends
        self.df_unitdata = pd.DataFrame()
        self.df_nodedata = pd.DataFrame()
        self.df_emissiondata = pd.DataFrame()
        self.df_userconstraintdata = pd.DataFrame()
        # Derived from df_nodedata at the end of run(), never read from a
        # workbook of its own -- see build_boundarydata.
        self.df_boundarydata = pd.DataFrame()

        # Where a value was written down: {dimension: {value: {"file:sheet"}}}.
        # Filled while the per-sheet frames still carry their provenance columns,
        # which merge_row_by_row drops. Both this phase and the Excel builder
        # report against it, so a message about a node or a unittype can name the
        # sheet to open -- see collect_origins.
        self.origins: dict[str, dict[str, set[str]]] = {}

        # What this build chose for each declared variant: variant -> country ->
        # option, None where the variant is dormant. And what the build fell back
        # on, which build_input_data.py puts in a PARTIAL banner.
        self.variant_choices: dict[str, dict[str, str | None]] = {}
        self.partial_causes: list[str] = []

    #: What each table merges on. The variant checks merge every option on the
    #: same keys, so the two cannot disagree about what one record is.
    KEY_COLUMNS = {
        'unittypedata': ['unittype'],
        # (emission, group), not emission alone: create_ts_emissionPriceChange
        # keys its output on the pair, so a narrower key here would merge two
        # groups of one emission into a single row and drop the second group
        # without a word.
        'emissiondata': ['emission', 'group'],
        'nodedata': ['country', 'grid', 'node'],
        'demanddata': ['country', 'grid', 'node'],
        'unitdata': ['country', 'unittype', 'unit_name_prefix'],
        'transferdata': ['from_country', 'from_suffix', 'to_country', 'to_suffix', 'grid'],
        'userconstraintdata': ['group', '1st dimension', '2nd dimension',
                               '3rd dimension', '4th dimension', 'parameter'],
    }

    def _read(self, table: str, *, drop_underscores: bool = True) -> list:
        """One table's sheets: read, variant tags checked, normalised.

        The tags are checked first, on the rows as written: before the
        underscore rule could drop a tag like 'bio_price:high' without saying
        why, and before any filter, so a malformed tag is reported in every run.
        """
        dfs = data_loader.read_input_excels(self.data_folder, self.config[f'{table}_files'],
                                            table, self.logger)
        dfs = self.variants.check_tags(dfs, table)
        dfs = [data_loader.normalize_dataframe(df, table, self.logger) for df in dfs]
        if drop_underscores:
            dfs = [data_loader.drop_underscore_values(df, table, self.logger) for df in dfs]
        return dfs

    def _selected(self, dfs: list, table: str) -> list:
        """This build's rows of a table, in merge order: the base pass, then the options."""
        base, chosen = self.variants.select(dfs, table)
        return [*base, *chosen]

    def _no_files(self, table: str) -> None:
        self.logger.log_status(
            f"No Excel files for '{table}_files' defined in the config file",
            level="info"
        )

    def run(self):
        """
        Run the full pipeline: load, process, and filter all input DataFrames.

        In six steps, because a country's variant option depends on rows in
        every table, and every table must follow the same choice:

        0. read every workbook's variant declarations;
        1. read and merge the tables without a country, first because units
           are named from their unittype;
        2. prepare every country table -- read, expand, name, filter -- without
           merging any of them;
        3. resolve each country's option per variant from all of them;
        4. check every option of every variant;
        5. select each table's rows for this build and merge them, the base
           first and the chosen options after it.
        """
        scenario_filter = {'scenario': [self.scenario], 'year': [self.scenario_year]}
        country_filter = {**scenario_filter, 'country': self.country_codes}
        exclude_grids = self.config['exclude_grids']
        exclude_nodes = self.config['exclude_nodes']

        # --- 0. Variant declarations, from every workbook the config lists ---
        self.variants = source_data_variants.Variants.from_workbooks(
            self.data_folder, self.config, self.variant, self.country_codes, self.logger
        )

        # --- 1. Tables without a country ---
        unittype_dfs, emission_dfs = [], []
        if self.config['unittypedata_files']:
            unittype_dfs = [data_loader.apply_whitelist(df, scenario_filter, self.logger, 'unittypedata')
                            for df in self._read('unittypedata')]
        else:
            self._no_files('unittypedata')

        if self.config['emissiondata_files']:
            emission_dfs = [data_loader.apply_whitelist(df, scenario_filter, self.logger, 'emissiondata')
                            for df in self._read('emissiondata')]
        else:
            self._no_files('emissiondata')

        # They take the build's 'all' choice or the default, which needs no
        # country table, so they can merge now -- and unittypedata has to.
        self.variants.resolve_countryless({'unittypedata': unittype_dfs,
                                           'emissiondata': emission_dfs})

        # unittypedata (internal — will be merged into df_unitdata before run() ends)
        if self.config['unittypedata_files']:
            dfs = self._selected(unittype_dfs, 'unittypedata')
            data_loader.collect_origins(dfs, {'unittype'},
                                        into=self.origins.setdefault('unittype', {}))
            data_loader.collect_origins(dfs, {'grid'},
                                        into=self.origins.setdefault('grid', {}))
            self._df_unittypedata = data_loader.merge_row_by_row(
                dfs, self.logger, key_columns=self.KEY_COLUMNS['unittypedata'])

        if self.config['emissiondata_files']:
            dfs = self._selected(emission_dfs, 'emissiondata')
            self.df_emissiondata = data_loader.merge_row_by_row(
                dfs, self.logger, key_columns=self.KEY_COLUMNS['emissiondata'])

        # --- 2. Prepare every country table, merging none of them ---
        # nodedata
        node_dfs = []
        if self.config['nodedata_files']:
            dfs = self._read('nodedata')
            dfs = [data_loader.expand_all_country(df, self.country_codes) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_node_column(df, self.logger) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, {'node': exclude_nodes}) for df in dfs]
            node_dfs = [data_loader.apply_whitelist(df, country_filter, self.logger, 'nodedata')
                        for df in dfs]
        else:
            self._no_files('nodedata')

        # demanddata
        demand_dfs = []
        if self.config['demanddata_files']:
            dfs = self._read('demanddata')
            dfs = [data_loader.expand_all_country(df, self.country_codes) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_node_column(df, self.logger) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, {'node': exclude_nodes}) for df in dfs]
            demand_dfs = [data_loader.apply_whitelist(df, country_filter, self.logger, 'demanddata')
                          for df in dfs]
        else:
            self._no_files('demanddata')

        # unitdata
        unit_dfs, unit_dfs_before_exclusion = [], []
        if self.config['unitdata_files']:
            dfs = self._read('unitdata')
            dfs = [data_loader.expand_all_country(df, self.country_codes) for df in dfs]
            # Before merge_row_by_row, and that ordering is the point: the merge
            # keys on 'unittype', so every sheet's spelling has to have been
            # respelled from unittypedata first. dfs_before_exclusion is captured
            # after both builders for the same reason -- count_units_without_exclusions
            # counts the same keys the real merge will.
            dfs = [data_loader.canonicalize_unittype_and_build_unit(df, self._df_unittypedata, self.logger) for df in dfs]
            dfs = [data_loader.build_unit_grid_and_node_columns(df, self._df_unittypedata, self.logger) for df in dfs]
            unit_dfs_before_exclusion = dfs
            dfs = [data_loader.apply_unit_grids_blacklist(d, exclude_grids) for d in dfs]
            dfs = [data_loader.apply_unit_nodes_blacklist(d, exclude_nodes) for d in dfs]
            unit_dfs = [data_loader.apply_whitelist(df, country_filter, self.logger, 'unitdata')
                        for df in dfs]
        else:
            self._no_files('unitdata')

        # transferdata
        transfer_dfs = []
        if self.config['transferdata_files']:
            dfs = self._read('transferdata')
            dfs = [data_loader.apply_blacklist(df, {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_from_to_columns(df, self.logger) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, {'from_node': exclude_nodes}) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, {'to_node': exclude_nodes}) for df in dfs]
            # Both ends in one call: apply_whitelist already ANDs its filters, so
            # a link survives only if both countries are modelled. Two calls said
            # the same thing and repeated every scenario and year comparison, and
            # any warning either raised, twice.
            transfer_dfs = [data_loader.apply_whitelist(df, {**scenario_filter,
                                                             'from_country': self.country_codes,
                                                             'to_country': self.country_codes},
                                                        self.logger, 'transferdata')
                            for df in dfs]
        else:
            self._no_files('transferdata')

        # userconstraintdata: no underscore rule, because its selectors are node
        # and unit names, which are built with underscores.
        constraint_dfs = []
        if self.config['userconstraintdata_files']:
            dfs = self._read('userconstraintdata', drop_underscores=False)
            constraint_dfs = [data_loader.apply_whitelist(df, country_filter, self.logger,
                                                          'userconstraintdata')
                              for df in dfs]
        else:
            self._no_files('userconstraintdata')

        country_tables = {
            'nodedata': node_dfs, 'demanddata': demand_dfs, 'unitdata': unit_dfs,
            'transferdata': transfer_dfs, 'userconstraintdata': constraint_dfs,
        }

        # --- 3. Resolve each country's option, from rows in every table ---
        self.variants.resolve_countries(country_tables)

        # --- 4. Check every option of every variant, whatever this build chose ---
        self.variants.check({'unittypedata': unittype_dfs, 'emissiondata': emission_dfs,
                             **country_tables}, self.KEY_COLUMNS)

        # --- 5. Select this build's rows and merge, in the order the tables were read ---
        if self.config['nodedata_files']:
            dfs = self._selected(node_dfs, 'nodedata')
            data_loader.collect_origins(dfs, {'node'}, into=self.origins.setdefault('node', {}))
            data_loader.collect_origins(dfs, {'grid'}, into=self.origins.setdefault('grid', {}))
            self.df_nodedata = data_loader.merge_row_by_row(
                dfs, self.logger, key_columns=self.KEY_COLUMNS['nodedata'])

        if self.config['demanddata_files']:
            dfs = self._selected(demand_dfs, 'demanddata')
            data_loader.collect_origins(dfs, {'node'}, into=self.origins.setdefault('node', {}))
            data_loader.collect_origins(dfs, {'grid'}, into=self.origins.setdefault('grid', {}))
            self.df_demanddata = data_loader.merge_row_by_row(
                dfs, self.logger, key_columns=self.KEY_COLUMNS['demanddata'])

        if self.config['unitdata_files']:
            unit_keys = self.KEY_COLUMNS['unitdata']
            dfs = self._selected(unit_dfs, 'unitdata')
            data_loader.collect_origins(dfs, {'node'}, into=self.origins.setdefault('node', {}))
            data_loader.collect_origins(dfs, {'unittype'},
                                        into=self.origins.setdefault('unittype', {}))
            self.df_unitdata = data_loader.merge_row_by_row(dfs, self.logger, key_columns=unit_keys)

            # An excluded grid or node removes the entire unit connected to it, not
            # just that connection -- intended, and not what "exclude a node"
            # sounds like, so the count is said out loud. Units rather than rows,
            # because several sheets can describe one unit and a row count would
            # overstate what left the model. A count rather than names: this is
            # expected and handled, and docs/source-data.md carries the reasoning.
            # The count selects as the real merge does, so a unit of an option this
            # build did not choose is not counted as excluded.
            units_dropped = data_loader.count_units_without_exclusions(
                unit_dfs_before_exclusion, country_filter, unit_keys,
                select=lambda kept: self._selected(kept, 'unitdata'),
            ) - len(self.df_unitdata)
            if units_dropped:
                self.logger.log_status(
                    f"Excluded grids and nodes removed {units_dropped} unit(s).",
                    level="none"
                )

            # Merge type-level technical parameters (LP/MIP, flow, emission_group*,
            # capacity defaults, etc.) from _df_unittypedata into each unit row.
            # Unit-specific values take priority; type-level values fill NAs only.
            self.df_unitdata = data_loader.merge_unittypedata_into_unitdata(
                self.df_unitdata, self._df_unittypedata, self.logger,
                origins=self.origins.get('unittype', {}),
            )

        if self.config['transferdata_files']:
            dfs = self._selected(transfer_dfs, 'transferdata')
            data_loader.collect_origins(dfs, {'from_node', 'to_node'},
                                        into=self.origins.setdefault('node', {}))
            data_loader.collect_origins(dfs, {'grid'}, into=self.origins.setdefault('grid', {}))
            self.df_transferdata = data_loader.merge_row_by_row(
                dfs, self.logger, key_columns=self.KEY_COLUMNS['transferdata'])

            # Deprecation check: old bidirectional format used export_capacity / import_capacity
            # per row; the new format uses one row per direction with transferCap instead.
            # Remove this check once all source Excel files have been updated.
            _deprecated_cols = [c for c in self.df_transferdata.columns
                                 if c in ('export_capacity', 'import_capacity')]
            if _deprecated_cols:
                self.logger.log_status(
                    f"transferdata contains deprecated column(s) {_deprecated_cols}. "
                    "The old bidirectional format (one row with export_capacity + import_capacity) "
                    "is no longer supported. Replace each bidirectional row with two directional rows "
                    "using 'transferCap' as the capacity column.",
                    level="warn",
                    add_empty_line_before=True,
                    add_empty_line_after=True,
                )

        if self.config['userconstraintdata_files']:
            # Selected first, because the choice is by country.
            dfs = self._selected(constraint_dfs, 'userconstraintdata')
            # errors='ignore': a sheet may lack any of the three, and a sheet that
            # normalized to empty has no columns at all -- normalize_dataframe returns
            # a bare DataFrame for empty input. Raising here would abort the phase,
            # which the error policy in CLAUDE.md forbids after the logger exists.
            dfs = [df.drop(columns=['scenario', 'year', 'country'], errors='ignore') for df in dfs]
            self.df_userconstraintdata = data_loader.merge_row_by_row(
                dfs, self.logger, key_columns=self.KEY_COLUMNS['userconstraintdata'])

        self.variant_choices = self.variants.choices
        self.partial_causes = self.variants.partial_causes

        # Both node-bearing frames are now filtered the same way -- same
        # blacklists, same whitelists -- so what they say about the node set can
        # be compared.
        data_loader.report_node_disagreements(
            self.df_nodedata, self.df_demanddata, self.logger
        )

        # Convert all empty cells to pd.NA across every output DataFrame.
        # Object columns may contain empty strings from normalization; replace them.
        # Float64 NaN values are already represented as pd.NA by the nullable dtype.
        for attr in ('df_nodedata', 'df_emissiondata', 'df_demanddata',
                     'df_unitdata', 'df_transferdata',
                     'df_userconstraintdata'):
            df = getattr(self, attr)
            if not df.empty:
                for col in df.columns:
                    if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
                        df[col] = df[col].replace('', pd.NA)
                        df[col] = df[col].where(df[col].notna(), pd.NA)

        # Node state boundaries, long. nodedata writes them wide -- one column
        # per boundary type -- because one row per node is what a spreadsheet is
        # good at; p_gnBoundaryPropertiesForStates has the type as a dimension.
        # Built last, so it sees a nodedata that is already merged and cleaned.
        self.df_boundarydata = source_data_contributions.build_boundarydata(
            self.df_nodedata, self.logger
        )