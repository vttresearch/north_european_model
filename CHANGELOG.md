# Changelog -- North European Energy System Model

Grouped by subject rather than by date.

## Timeseries

Every processor now proves its input data before using it, names what it could
not build and why, and has a documentation page of its own.

- How the build turns an hourly source into Backbone input: [docs/timeseries.md](docs/timeseries.md).
- hydro: built only for the hydro nodes `nodedata` carries; reservoir sizes come
  from `nodedata` `upwardLimit` and `PECD-hydro-capacities.csv` is removed; inflow
  and the seasonal fill limits are continuous over the year change; several
  zero-inflow and zero-limit periods from partial source data fixed; AT00 has its
  seasonal limits back. [docs/hydro.md](docs/hydro.md)
- electricity demand: every scenario year reads the 2030 profiles, because the
  2040 ones contain hours of negative demand. [docs/elec-demand-timeseries.md](docs/elec-demand-timeseries.md)
- district heating: [docs/dh-demand-timeseries.md](docs/dh-demand-timeseries.md)
- wind and solar: [docs/vre-timeseries.md](docs/vre-timeseries.md)
- `VRE_MAF2019` and `hydro_mingen_limits_MAF2019` removed. A config still naming
  either writes no timeseries for it; switch to `VRE_PECD`.
- Demand grids with no processor of their own get a constant `influx` instead of
  a flat timeseries; `ts_influx_other_demands.gdx` is no longer written.
- A processor contributes to the source data tables instead of returning a
  secondary result; the `secondary_output_name` spec field is retired.
  [docs/timeseries.md](docs/timeseries.md)
- A `node`, `grid` or `flow` a processor builds data for that the source data
  does not have is reported.
- wind and solar: capacity factors are built only for the nodes `unitdata`
  attaches a unit of that flow to. [docs/vre-timeseries.md](docs/vre-timeseries.md)
- The build reports what needs acting on rather than what happened: an absence
  the source workbooks already state is silent, and what the rules handled is a
  line of counts. [docs/timeseries.md](docs/timeseries.md)
- `is_input_data_dependent` is retired and a timeseries processor is never copied
  between scenario folders; a config still setting it is ignored, and a
  multi-scenario build costs about a minute more per scenario.
- A processor reruns when the input it is given changed or its own code changed,
  and the build names the value that changed. Editing a cost in `unitdata` no
  longer rebuilds the wind and solar timeseries.
  [docs/timeseries.md](docs/timeseries.md)
- `requires_source_data` names the columns of each table a processor reads, and
  a processor receives only those. The older tuple-of-table-names form still
  works and delivers the whole frame.
  [docs/timeseries.md](docs/timeseries.md)
- `reads_input_files` declares the files a processor opens, so a replaced PECD
  download is noticed. A processor declaring none is rerun every build.
  [docs/timeseries.md](docs/timeseries.md)
- The annual summary CSVs are removed; the `annual_summary` spec field is
  retired and a config still setting it is ignored.
- A cold build is about a third faster.

## Source workbooks

- `generator_ID` is removed. A `unitdata` row names its `unittype` directly, and
  `unittypedata` is keyed on `unittype` with a free-text `## Description` column
  in place of the old name.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- `transferdata` is unidirectional: one row per line per direction, with
  `transferCap` in place of `export_capacity` and `import_capacity`. Either old
  column is named by the build and contributes nothing.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- `##` in a cell skips the row, `##` in a column header skips the column.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- A build no longer holds every source workbook open for its whole run, which
  made them read-only in Excel while it ran.
- Excel error values now reach the report at all. pandas turns every error cell
  into an empty one before anything can look, so no `#REF!` or `#N/A` had ever
  been named.
- Malformed cells are reported once per workbook, naming the worst three columns,
  instead of once per column.
- Malformed numbers (`1,000.0`, `100 MW`, `#REF!`) are reported and treated as
  not set, in source excels and timeseries files alike.
- Warnings about a unittype or a node name the workbook and sheet it was written
  in, and say when another sheet spells the same name differently.
- A row with no unittype is identified by its country, scenario and year.
- A blank key column, or a year that is neither a real year nor the `1` meaning
  every year, is an error naming the spreadsheet row. Such a row used to be
  dropped in silence.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- Source workbooks renamed for what they hold: `dheat_balticData.xlsx`,
  `dheat_unitdata_PL_DE_AT.xlsx`, `dheat_unitdata_SE_DK.xlsx`. The unused
  `unitdata_TYNDP-2020.xlsx` and `unitdata_additional-*.xlsx` are removed, and
  some content moved between the remaining ones.
- Unit types and several fuel nodes renamed.
- Finnish city units and district heating transfer links no longer share
  identical costs, and VRE costs are slightly higher.
- Three AT00 units are deactivated with `method = remove` rather than a year that
  matched nothing; the sub-10 MW rule drops one further unit.
- Blank rows, unnamed columns and repeated headers inside a table are reported
  rather than silently dropped.
- A node that only one of `nodedata` and `demanddata` knows about is reported.
- `merge_row_by_row`: column titles compared case-insensitively, first spelling kept.
- Excluding a grid or node says how many units it removed.
- A column no stage reads is reported by file and sheet.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- What the source data phase does with a workbook:
  [docs/source-data.md](docs/source-data.md)
- A demand written as `0` is kept instead of being deleted as an empty row.
- `multiply` leaves a value unchanged when either side is missing, instead of
  writing `0`.
- An `add` or `multiply` row whose key matches nothing is reported; `multiply`
  no longer creates a record.
- A row with no `scenario` or `year` is reported before it is dropped.
- A sheet that overwrites its own earlier row with `replace` is reported.
- A sheet dropped for a missing `country`, `grid` or `unittype` says how
  many rows that cost.
- A user constraint sheet without a `country` column no longer stops the build.
- `emissiondata` is merged on `emission` and `group` together.
- `transferdata` is filtered once for both link ends rather than twice.
- Sheet hashes use the full category prefix, so `unittypedata` is no longer
  hashed as `unitdata` too.
- Config file lists name `Finland_dheat_and_industry.xlsx` with the case the
  folder uses.

## Input excel builder

- How the builder turns the source data tables into `inputData.xlsx`:
  [docs/input-excel.md](docs/input-excel.md).
- p_gn/p_gnn/p_gnu_io/p_unit/p_gnBoundaryPropertiesForStates: empty parameter
  columns are dropped, always keeping one.
- storage starts: a node with no determinable start level is reported. It was
  already left unbounded, but `boundStart=1` and a 0 reference made it look bound.
- Node state boundaries are read from one table, and it states whether each one
  is a constant or a timeseries instead of that following from where it came
  from. [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- The hydro reservoir start level is written by `changes.inc` alone; the input
  excel's provisional value is no longer meant to be the one used.
- `useTimeSeries` and `storageValueUseTimeSeries` spelled `useTimeseries` and
  `storageValueUseTimeseries`, following Backbone.
- `boundStart` is dropped when no node has a storage start level, rather than
  written as a column of zeros.
- Warnings name the first three offenders and then count the rest, instead of
  one line each. [docs/timeseries.md](docs/timeseries.md)

## Test suite

- A pytest suite, and 20+ latent minor bugs found and fixed with it. Current
  scenarios unimpacted.

## Tools

- `tools/input_data_summary.py` describes one built folder as a `report.md` with
  embedded figures: capacity, demand, storage, interconnection and prices by country
  and carrier, a net-load duration curve, and what the climate years do. Replaces the
  untracked `analyze_ts.py` draft and its ~874 per-node figures.
- The report said battery and heat storage carry no energy capacity anywhere. They
  carry 1.06 TWh, as `upperLimitCapacityRatio` in `p_gnu_io` -- a duration per grid,
  now reported as one and shown in the storage figure's legend.
- `build_input_summary.py` at the repository root: a thin wrapper on
  `tools/input_data_summary.py`, so building a folder and reading it are one command
  each. Same arguments, same exit codes.
- The carrier map tinted only what was demanded, so hydrogen -- nodes in 22 bidding
  zones, demand in none -- tinted nothing and the TYNDP map came out identical to a
  scenario with no hydrogen in it. It now tints what is modelled, in four fills, and
  says beside the key which carrier has no sink. "Where the model is" is two counts,
  the map, and a list of what is missing; the 16-row presence table is gone.
- Hydro storage was read as the annual maximum of `upwardLimit` alone: 159.6 TWh in
  NT2030. Both bounds are seasonal series, and the usable volume is 80.3 TWh. Reported
  as usable, seasonal envelope and nameplate, from one forecast branch.
- Storage is two nested electricity groups -- battery and closed-loop pumped hydro,
  then plus inflow hydro -- derived from the grids, not listed. Heat storage is in
  neither and says why.
- "On what timescale" compared nothing: it reported unservable TWh/yr against storage
  measured in TWh, and left hydro inflow out, so Norway read as the area least able to
  cover itself. It now reports the store depth the residual needs at each timescale,
  with the installed fleet on the same axis, and counts inflow as supply.
- `dheat` and `steam` were storage grids, because a `balancePenalty` row was read as a
  state. A state is now `energyStoredPerUnitOfState`, and an unknown storage grid is
  named rather than assumed to be pumped storage.
- The fuel table listed seven of the ten fuels and was sorted by a column the sentence
  did not name; it now carries all ten, sorted by the one it points at, and says that
  ordering fuels is not ordering plants.
- The correlation threshold is the level the number of climate years can resolve,
  0.33 at 35, rather than a round 0.3 that sits below it.
- Two maps: which carriers each area models and whether anything demands them, and the
  transfer corridors, which replace the capacity heatmap. `--no-neighbours` drops the
  grey ring of unmodelled countries, and a missing asset skips the maps and leaves the
  report otherwise whole.
- One map asset per level, both from `tools/prepare_zone_geometry.py`:
  `country_shapes.geojson` and `zone_shapes.geojson`. Every border is Natural Earth's
  (public domain); the ENTSO-E layer (Mopo, CC BY 4.0) now only says which zone a piece
  of land belongs to. Its Norway outline was 31% sea. A country's zones cover exactly
  its country outline, so the two maps agree. No geometry library.
- Both maps are always drawn per bidding zone, with country borders over them, whatever
  level the tables use. A country report was drawing the Nordic ring as four dots while
  the text above it counted 46 corridors.
- The carrier map tints an area only for carriers something actually demands, and the
  three-cell chips are gone; what they carried is a table under the map, which also
  reads at publication size. The map now separates what the country tables roll
  together -- DKE1 has district heat demand and DKW1 does not.
- Demand counts the constant `influx` in `p_gn`, not just the `ts_influx` families. The
  two are alternatives, never a sum: the model overrides the constant wherever a series
  exists, and a check names any node carrying both. Industrial steam was the whole of
  it -- 495 TWh/yr, more than district heat -- and it now has a section rather than one
  bullet. Its number survives a run that can read no GDX file at all.
- Timeseries are read per bidding zone and rolled up for a country report, so one read
  serves both levels.
- `--no-timeseries` is removed. Every report reads the climate years; a missing GAMS
  install or GDX file still degrades the same way, with the reason printed.
- The net-load table does the subtraction it promised -- firm, storage and demand
  response, imports, and what is left of the peak -- adds potential VRE energy as a
  share of demand, and a system row summed hour by hour rather than peak by peak.
- A new section decomposes residual demand by the storage duration that could remove
  it, and a new one counts what the build lets a unit do: no ramp limits, availability
  1 everywhere, no unit commitment.
- Carrier figures pair the absolute panel with a normalised one, because six of the
  sixteen countries were a hairline beside Germany. Raw grid identifiers used as table
  headers are defined, with their durations derived from the data.
- `GDX_exchange.read_gdx_parameter_over_files` reads a parameter across many files on
  one container, the way the write path already does. Reading a build's 245 per-year
  files takes seconds rather than minutes; `read_gdx_parameter` builds a container per
  call and should not be called in a loop.

## Running the model

- `%init_file%` command line parameter, for switching between schedule and invest.
- changes_loop.inc: tighter `vq_userconstraint.up` limits, to speed up the solver.
- `config_OT2030-continuous5y.ini`, an example of `bb_timeseries_length`
  expressions (365*5).
- environment.yml: added matplotlib.
- Toolbox wrapper import fixed by moving the CLI arg parser to `src/utils.py`.
