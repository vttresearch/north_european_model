# Identified gaps

Backbone can express more than this project writes, and a few of the rules this
project does write are known to be provisional. Both are recorded here so that the
next person to work on the pipeline finds them in one place rather than
rediscovering them one docstring at a time.

**This is a staging page, not a reference page.** An entry leaves it in one of two
ways: the gap is closed, or the entry moves to the page that properly owns it —
which is why every one of them carries a *where it would live* line. Nothing here
is a proposal. An entry says what the state is and where the code says so, and
takes no position on whether it should change.

## One minute summary

Six kinds of gap, and they are not equally interesting:

- **Slots declared here that nothing feeds** — the plumbing exists and no source
  fills it. Looks like a bug and is not.
- **Checks that cannot be made yet** — a question worth asking that nothing can
  currently answer.
- **Rules known to be provisional** — the code is deliberate, and its author
  already knows it is not the final answer.
- **Deferred by the source data pass** — found while measuring, deliberately not
  acted on. Alternatives being scenario names is the one with teeth.
- **Backbone parameters this build does not write** — derived from
  `../inc/1a_definitions.gms`. Mostly investment, reserves and part-load
  efficiency.
- **Backbone symbols with no sheet at all** — `src_files/indexSheet.xlsx` declares
  63, the builder writes 21, and three more arrive as GDX. Reserves, group
  policies and unit constraints are the substance of what is left.

The last two are inventory. The first four are the ones that change what someone
designs next.

**One entry has been closed rather than moved.** "The source workbook side" asked
what the workbooks hold that nothing reads; the build now reports it on every
run, so nobody has to measure it again.

## Contents

- [Slots declared here that nothing feeds](#slots-declared-here-that-nothing-feeds)
- [Checks that cannot be made yet](#checks-that-cannot-be-made-yet)
- [Rules known to be provisional](#rules-known-to-be-provisional)
- [Backbone parameters this build does not write](#backbone-parameters-this-build-does-not-write)
- [Backbone symbols with no sheet at all](#backbone-symbols-with-no-sheet-at-all)
- [The source workbook side](#the-source-workbook-side)
- [Deferred by the source data pass](#deferred-by-the-source-data-pass)

## Slots declared here that nothing feeds

### slackCost

It is one of the four `param_gnBoundaryProperties`, it is coerced to a number with
the rest of them, and the workbook writer keeps its column whenever any row sets
one — `test_storage_starts.py` pins both that and the drop when no row does.
Nothing in this repo ever sets one, so it is a slot waiting for a source rather
than a value being lost somewhere.

Backbone uses it to price a violation of a state boundary. Without it a bound is
hard: the solver cannot buy its way past a `downwardLimit` at any price, and an
infeasible hydro year is infeasible rather than expensive.

*Where it would live:* [Input Excel builder](input-excel.md), beside the boundary
properties, once something writes one.

### A boundary table with no workbook

`df_boundarydata` is the long table `p_gnBoundaryPropertiesForStates` is built
from, and it is the only source data table with no workbook of its own.
`build_boundarydata` derives every row of it from `nodedata`'s wide boundary
columns, and a timeseries processor adds `useTimeseries` rows through the
contribution merge.

The consequence is that precedence runs one way only. A processor cannot overwrite
a constant a workbook wrote, because the merge fills only where the source said
nothing — but a workbook cannot say "use the constant, not the series" either,
because there is no boundary row for it to write `useTimeseries = 0` on. The merge
would honour it. There is simply no cell.

*Where it would live:* [Source workbook conventions](source-workbook-conventions.md),
as a sheet, if boundaries ever get one.

## Checks that cannot be made yet

`ProcessorRunner` reports a `grid`, `node` or `flow` value that no source table
declares, which is the pipeline's only defence against a mistyped dimension value.
Three things it cannot extend to, all of them argued in `source_workbook_shape.py`
under "What is not here":

- **`emission` and `group` have no declaring table.** An emission is the suffix of
  a `nodedata` `emission_XX` column, and a group is assembled by the Excel builder
  out of emissions, user constraints and unit groups. There is nothing to check a
  value against without inventing a declaration for it.
- **`p_userconstraint`'s four selector slots** do refer to values declared
  elsewhere, but what each slot *means* depends on the row's own `parameter` —
  `../docs/dictionary.md` gives a dimension contract per parameter. The check needs
  that contract in machine-readable form before it can exist.
- **`restype` has no source in this repo at all**, so there is nothing to declare
  and nothing to check against. See the reserves entry below.

*Where they would live:* [Timeseries](timeseries.md), in "What the runner checks
before it writes", as each becomes answerable.

## Rules known to be provisional

### The hydro storage start level

`add_storage_starts` writes a provisional starting level, and `changes.inc` then
recomputes the reference of every `psOpen` and `reservoir` node from the maximum
of that node's own `upwardLimit` series. So for those nodes the workbook value
only has to be above zero; what it actually is never reaches the solved model.

Two things follow, and both are wanted rather than tolerated:

- A node whose `upwardLimit` comes only from a series, with no `nodedata`
  constant, gets **no** start level and a warning naming it. That is correct: the
  data really is partial, and partial data warns. `changes.inc` will bound the
  node anyway, but the build cannot see that far and should not pretend to.
- The rule `add_storage_starts` applies — 0.7 of the node's own upward limit —
  cannot express a run that starts and ends in summer, which its own docstring
  says.

Redoing the hydro rules properly is what closes this, and taking the `changes.inc`
patch back out is part of that work.

*Where it would live:* [Hydro data](hydro.md), which already carries the
`changes.inc` paragraph.

### A zero written where Backbone reads it as not-set

A `$`-gated parameter treats a written `0` as absent, so writing one does nothing
at all — the cell is not a zero, it is a wasted cell. `param_gnBoundaryProperties`
`multiplier` is the case `backbone_params.py` already names, and it is left out for
exactly this reason rather than by oversight. Whether any other `0` this build
writes is in the same position has not been swept.

*Where it would live:* [Input Excel builder](input-excel.md), in "Zero is not a
number here", once the sweep has been done.

## Backbone parameters this build does not write

Re-derive rather than trusting this table: Backbone's own vocabulary moves. Last
derived **2026-09-07**, against the `param_*` set declarations in
`../inc/1a_definitions.gms` rather than by hand — see
[How to re-derive it](#how-to-re-derive-it) below.

| Sheet | Written here | In Backbone, not written |
|---|---|---|
| `p_gn` | 17 of 20 | `maxInvest`, `invCost`, `annuityFactor` — node-level investment |
| `p_gnn` | 12 of 18 | `transferCapBidirectional`, `boundStateMaxDiff`, `unitSize`, `portion_of_transfer_to_reserve`, `useTimeseriesAvailability`, `useTimeseriesLoss` |
| `p_gnu_io` | 32 of 34 | `profitMargin`, `maxTsDelay` |
| `p_unit` | 26 of 79 | `eff02`–`eff12` and `op02`–`op12`, the whole `hr*` / `hrop*` heat-rate family, `section`, `hrsection`, `outputCapacityTotal`, `unitOutputCapacityTotal`, `lastStepNotAggregated` |
| `param_gnBoundaryTypes` | 6 of 46 | `minSpill`, `upwardSlack01`–`upwardSlack20`, `downwardSlack02`–`downwardSlack20` |
| `param_gnBoundaryProperties` | 4 of 5 | `multiplier` — deliberate, see above |

Two denominators moved when this was derived rather than counted by hand, both
for the same reason: GAMS declares `eff02*12` and `upwardSlack01*20` as ranges,
so counting the written lines undercounts the members. `p_unit` has 79 members,
not 32, and `param_gnBoundaryTypes` 46, not 8. The lists of missing names were
right; only the totals were wrong, which flattered the coverage considerably.

The check also runs the other way, and that direction is clean: **every name in
`backbone_params.py` is a member of the Backbone set it claims to belong to.**
Nothing had verified that before.

Two of these are more than a missing column. **The efficiency curve stops at two
points**: a unit gets `eff00` / `eff01` and `op00` / `op01`, so every part-load
efficiency in the model is a single straight segment, and the `hr*` heat-rate form
is not available at all. And **investment is expressible per unit but not per
node**, since `p_gn`'s three investment parameters are the three that are missing.

*Where it would live:* [Input Excel builder](input-excel.md) for the parameters
themselves, [Source workbook conventions](source-workbook-conventions.md) for the
columns that would carry them.

### How to re-derive it

The Backbone repository parses its own dictionary, and the parser is usable from
here. It must not become a dependency — nothing under `src/` or `tests/` imports
it, and this table stays a document rather than something generated — so this is
a thing you run by hand when you want to know whether the table has drifted:

```python
import sys; sys.path.insert(0, "../scripts/docs")
import dictionary as d
members = [m for m in d.parse_param_sets()["param_unit"] if not m.commented_out]
expanded = [n for m in members for n in (d.expand_range(m) if m.range_end else [m.name])]
```

`parse_param_sets` reads `../inc/1a_definitions.gms`, skips the commented-out
members that mean NOT IMPLEMENTED, and `expand_range` turns `eff02*12` into its
eleven names. Compare against the `PARAM_*` lists in `src/backbone_params.py`.

## Backbone symbols with no sheet at all

`src_files/indexSheet.xlsx` is this project's own statement of what a Backbone
workbook can carry: 63 symbols. The builder writes 21 of them, plus the `index`
sheet itself — the 22 that [Input Excel builder](input-excel.md) lists. Of the
rest, 15 are `ts_*` and would arrive as GDX rather than as a sheet, and this
project produces three of those: `ts_cf`, `ts_influx` and `ts_node`. That leaves
**27 declared in the index sheet and written by no route at all**:

- **Reserves, entirely** — `p_gnuReserves`, `p_gnnReserves`, `p_gnuRes2Res`,
  `p_groupReserves`, `p_groupReserves3D`, `p_groupReserves4D`, `restypeDirection`,
  `restypeReleasedForRealization`, `restype_inertia`. The visible edge of this is
  the `restype` sheet, which the builder writes empty every run.
- **Group policies** — `p_groupPolicy`, `p_groupPolicyUnit`,
  `p_groupPolicyEmission`, and the group memberships `uGroup`, `gnuGroup`,
  `gn2nGroup`, `sGroup`. `gnGroup` is the one group sheet that is written.
- **Unit constraints** — `p_unitConstraint`, `p_unitConstraintNode`. A user
  constraint can express some of the same things through `p_userconstraint`, which
  is written; the unit-constraint form is not.
- **The rest** — `p_storageValue`, `p_uStartupfuel`, `p_gnuBoundaryProperties`,
  `unitUnitEffLevel`, `utAvailabilityLimits`, `unit_fail`, `gnss_bound`,
  `uss_bound`, `t_invest`.

`p_gnuEmission` is a further case: it is in `../docs/dictionary.md` but not even in
this project's index sheet, so per-unit emission factors have no route into the
workbook at all. Emissions reach the model only through `p_nEmission`, per node.

*Where it would live:* [Input Excel builder](input-excel.md), as sheets, one at a
time.

## The source workbook side

**Closed.** This was the largest entry and the one that had not been measured:
what do the source workbooks hold that nothing reads? It is no longer a question
anyone has to ask, because the build answers it on every run — an unrecognised
column is reported by file and sheet rather than silently ignored. See
[A column nothing reads](source-workbook-conventions.md#a-column-nothing-reads)
and [The source data phase](source-data.md#columns-nothing-reads).

What closed it, kept because it says what "clean" looks like: **every column on
every prefix-matching sheet the four shipped configs name is recognised**. The
only unread ones found were four in `unittypedata_compilation.xlsx`, disabled by
renaming them `disabled-maxRampUp` and so on; they are marked `##` now, which is
what the builder can see. Two of them, `rampUpCost` and `rampDownCost`, are live
again — the rename had taken ramp costs away from every thermal, CHP and
heat-only unit, leaving them only on hydro. The figure is deliberately not written
down as a count — a tally of sheets and headers goes stale the first time a
workbook is consolidated, and silently. Run a build and read `summary.log`.

What stays open is the mirror half, which is still an inventory rather than a
defect: every parameter in the table above needs a workbook column before it can
be written, so that table is also a list of columns that do not exist yet.

## Deferred by the source data pass

Found while measuring the above, deliberately not acted on.

### Alternatives are scenario names

`scenario_alternatives` through `scenario_alternatives4` are four axes of a
Cartesian product, and an alternative is not a thing in a workbook: it is a value
in the ordinary `scenario` column, which `apply_whitelist` adds to the list of
scenarios this run accepts.

So **an alternative carries no precedence**. A base row in a later file overwrites
an alternative row in an earlier one, and the only lever is config file order —
the same lever everything else uses. Expressing "this alternative overrides the
base" therefore means ordering files rather than saying so.

All four shipped configs set `scenario_alternatives = []`, and no test exercises
the axes. That is deliberate: pinning the current behaviour would make it harder
to change, and reworking alternatives is its own piece of work.

*Where it would live:* [The source data phase](source-data.md), once alternatives
have a mechanism of their own.

### `country = 'all'` is expanded for three tables of five

`expand_all_country` runs for `nodedata`, `demanddata` and `unitdata`. It does not
run for `userconstraintdata`, which drops `country` after filtering, and it is
vacuous for `transferdata`, which has no `country` column at all — it has
`from_country` and `to_country`, and neither is expanded. No shipped sheet writes
`all` in a table that would not expand it.

*Where it would live:* [The source data phase](source-data.md), if the two ever
need it.

### A sheet that normalises to nothing loses its column names

`normalize_dataframe` returns a bare `DataFrame` — no rows and no columns — for
an empty input, so a sheet whose every row is marked `##` arrives downstream with
its schema gone rather than empty. That is the mechanism behind the `KeyError`
the userconstraint block used to raise; that caller is guarded, and the general
rule is unchanged because it reaches the contract sweep.

*Where it would live:* `tests/README.md`, in the boundary map, if the rule
changes.

### A `remove` that removes nothing cannot be verified

Dozens of `remove` rows on each shipped config match no earlier row, and every
one is deliberate — removing offshore wind from
landlocked countries by writing the row for every country and letting it miss.
A misspelled `remove` looks exactly the same and nothing in the sheet
distinguishes them. This is why the build reports an unmatched `add` or
`multiply` and not an unmatched `remove`.

*Where it would live:* [The source data phase](source-data.md), if a `remove`
ever gains a way to say it meant to match something.

### Five workbooks no config names

`H2 heavy.xlsx`, `transferdata_TYNDP2020.xlsx`,
`unittypedata_nuclear-lwr-smr.xlsx`, `transferdata_additional1.xlsx` and
`demanddata_other.xlsx` sit in `src_files/data_files/` and no shipped config
lists any of them. Their columns are genuinely unread, and listing one now says
so rather than staying silent — which is the useful part, because
`transferdata_additional1.xlsx` is a long-format `parameter`/`value` sheet the
transferdata reader does not support at all and would read as a table of
nothing.

*Where it would live:* nowhere. They are either deleted or listed.

### A processor could declare the columns it reads

`source_workbook_shape.DERIVATION_INPUTS` names the three columns a later phase
reads by name — `lp/mip`, `twh/year`, `constant_share` — because nothing can
discover them. Two of the three belong to timeseries processors, and
`BaseProcessor.reads_source_columns` lets a processor state its own; a test holds
the two equal rather than one importing the other, because the source phase runs
first and must not depend on which processors a config enables.

Making the processors the authority would need the source phase to read
declarations without importing processor modules. Worth doing if the list grows;
at three entries it is not.

*Where it would live:* [Timeseries](timeseries.md), beside the other processor
declarations.

### A unit's grid is `unittypedata`'s, and a `unitdata` sheet is not told

`build_unit_grid_and_node_columns` assigns `grid_<put>` from the unittype's row
unconditionally, so a `grid_output1` written on a `unitdata` sheet is overwritten
— and set to blank when the unittype does not define that connection at all.
Nothing is logged, and `grid` is an accepted `unitdata` column, so the
unrecognised-column check waves it through as well.

The behaviour is right: one authority per connection is what keeps a unit's grids
consistent across every sheet that mentions it. What is open is whether writing
the column should say something, or stop being accepted on that table.

*Where it would live:* [Source workbook conventions](source-workbook-conventions.md),
which now states the rule; this entry is about the silence, not the rule.

### Silent steps that change what reaches the model

Four of them, none wrong, none visible:

- **`expand_all_country` says nothing.** One `all` row becomes one row per
  country, and no count is reported for a step that can multiply a sheet
  thirtyfold.
- **A second `unittypedata` row for one unittype is ignored**, first wins, in
  three separate functions. A duplicated unittype is not reported anywhere.
- **An empty `unittypedata` leaves `unitdata` untouched** — no `unit` column at
  all — and every later stage sees a table that looks merely empty rather than
  unbuilt.
- **`method` defaults to `replace` for a blank cell, and a numeric `*_output1`
  column is renamed to its base name.** Both are the documented behaviour; a
  reader comparing the sheet with the merged table sees neither happen.

Each is a candidate for a count rather than a warning, on the rule that what is
expected and handled costs counts, not names.

*Where it would live:* [The source data phase](source-data.md), if any of them
starts speaking.

### Excluding a node is counted for units only

`14d2c33` answered "how much of the model left with this exclusion" for
`unitdata`, where a unit is dropped whole and the loss is surprising. The rows
`exclude_grids` and `exclude_nodes` drop from `nodedata`, `demanddata` and
`transferdata` are still dropped in silence. Those losses are one row per node
and much less surprising, which is why they were left — but the reader who wants
to know what an exclusion cost gets half an answer.

*Where it would live:* [The source data phase](source-data.md), beside the unit
count.

### A unit name can contain `<NA>`

A `unitdata` row whose `unittype` is blank is reported by
`canonicalize_unittype_and_build_unit`, and the row is left in place with a
`unit` string built from the blank — `FI_<NA>`. The report is the actionable
half; the malformed name travels further than it should, and nothing downstream
treats it as special.

*Where it would live:* [The source data phase](source-data.md), if the row is
ever dropped instead of reported.

## See also

- [Input Excel builder](input-excel.md) — what the builder does write, and why a
  parameter column is missing whenever nothing set it
- [Source workbook conventions](source-workbook-conventions.md) — the sheets and
  columns that exist today
- [Hydro data](hydro.md) — the storage start level and the `changes.inc` patch in
  their own context
- `docs/dictionary.md` in the Backbone repository — the authority for every
  parameter named here
