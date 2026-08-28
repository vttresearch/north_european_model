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

## In one minute

Five kinds of gap, and they are not equally interesting:

- **Slots declared here that nothing feeds** — the plumbing exists and no source
  fills it. Looks like a bug and is not.
- **Checks that cannot be made yet** — a question worth asking that nothing can
  currently answer.
- **Rules known to be provisional** — the code is deliberate, and its author
  already knows it is not the final answer.
- **Backbone parameters this build does not write** — measured against
  `../docs/dictionary.md`. Mostly investment and reserves.
- **Backbone symbols with no sheet at all** — `src_files/indexSheet.xlsx` declares
  63, the builder writes 21, and three more arrive as GDX. Reserves, group
  policies and unit constraints are the substance of what is left.

The last two are inventory. The first three are the ones that change what someone
designs next.

## Contents

- [Slots declared here that nothing feeds](#slots-declared-here-that-nothing-feeds)
- [Checks that cannot be made yet](#checks-that-cannot-be-made-yet)
- [Rules known to be provisional](#rules-known-to-be-provisional)
- [Backbone parameters this build does not write](#backbone-parameters-this-build-does-not-write)
- [Backbone symbols with no sheet at all](#backbone-symbols-with-no-sheet-at-all)
- [The source workbook side](#the-source-workbook-side)

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

Measured against the parameter tables in `../docs/dictionary.md`. Re-derive rather
than trusting this table: Backbone's own vocabulary moves.

| Sheet | Written here | In Backbone, not written |
|---|---|---|
| `p_gn` | 17 of 20 | `maxInvest`, `invCost`, `annuityFactor` — node-level investment |
| `p_gnn` | 12 of 18 | `transferCapBidirectional`, `boundStateMaxDiff`, `unitSize`, `portion_of_transfer_to_reserve`, `useTimeseriesAvailability`, `useTimeseriesLoss` |
| `p_gnu_io` | 32 of 34 | `profitMargin`, `maxTsDelay` |
| `p_unit` | 26 of 32 | `eff02`–`eff12` and `op02`–`op12`, the whole `hr*` / `hrop*` heat-rate family, `section`, `hrsection`, `outputCapacityTotal`, `unitOutputCapacityTotal`, `lastStepNotAggregated` |
| `param_gnBoundaryTypes` | 6 of 8 | `minSpill`, `upwardSlack01`–`upwardSlack20`, `downwardSlack02`–`downwardSlack20` |
| `param_gnBoundaryProperties` | 4 of 5 | `multiplier` — deliberate, see above |

Two of these are more than a missing column. **The efficiency curve stops at two
points**: a unit gets `eff00` / `eff01` and `op00` / `op01`, so every part-load
efficiency in the model is a single straight segment, and the `hr*` heat-rate form
is not available at all. And **investment is expressible per unit but not per
node**, since `p_gn`'s three investment parameters are the three that are missing.

*Where it would live:* [Input Excel builder](input-excel.md) for the parameters
themselves, [Source workbook conventions](source-workbook-conventions.md) for the
columns that would carry them.

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

The largest entry, and the one that has not been measured. The sections above ask
what Backbone can hold that the builder does not write. The mirror question is
what the source workbooks hold that nothing reads, and answering it needs a pass
over `source_data_loader` and the sheets in `src_files/data_files/` that has not
been done.

Two things are known without it:

- A column the builder does not recognise is carried through the source stage and
  then ignored, silently. `_coerce_numeric_dtypes` warns about exactly one shape of
  mistake — a `param_unit` name wearing a connection suffix — and about nothing
  else.
- Every parameter in the table above needs a workbook column before it can be
  written, so that table is also a list of columns that do not exist yet.

*Where it would live:* [Source workbook conventions](source-workbook-conventions.md),
once measured.

## See also

- [Input Excel builder](input-excel.md) — what the builder does write, and why a
  parameter column is missing whenever nothing set it
- [Source workbook conventions](source-workbook-conventions.md) — the sheets and
  columns that exist today
- [Hydro data](hydro.md) — the storage start level and the `changes.inc` patch in
  their own context
- `docs/dictionary.md` in the Backbone repository — the authority for every
  parameter named here
