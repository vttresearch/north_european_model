# Input Excel builder

The last phase of a build. `BBExcelPipeline` reads the merged source data tables
and writes `inputData.xlsx` — 22 sheets, one per Backbone symbol — into the
scenario's output folder, where the GAMS side converts it to GDX.

It is the point where this project's data conventions become Backbone's. Python
has kept `0` and "empty" apart all the way here; from this phase on they are the
same thing, and most of what the builder does that is not copying is a
consequence of that. See [Source workbook conventions](source-workbook-conventions.md)
for what the sheets it reads look like, and [Timeseries](timeseries.md) for the
phase that runs before it.

## One minute summary

- **The source data tables are the only input.** Seven `df_*` frames off
  `SourceDataPipeline`, and nothing else. Whatever the timeseries phase had to
  say about a node has already been merged into them, so no rule here depends on
  which phase a row came from.
- **`0` = NA = "not set".** Backbone treats an absent parameter and an explicit
  zero identically, so an all-zero parameter column states nothing and is
  dropped. A sheet therefore carries **only the parameters some row set** — see
  [Zero is not a number here](#zero-is-not-a-number-here).
- **Nodes are classified, not just copied.** `usePrice`, `nodeBalance` and
  `energyStoredPerUnitOfState` are deduced when the workbook does not state them.
  This is where most of the builder's warnings come from — see
  [How a node is classified](#how-a-node-is-classified).
- **Five sheets carry a second header row.** Blank in the dimension columns,
  repeating the parameter name elsewhere. It is a GDXXRW requirement, applied
  only on the way out.
- **Some values are derived.** A missing capacity on one side of a unit, and a
  storage node's starting level. Both are conservative and both are documented
  below.

| Sheet group | Sheets | Second header row |
|---|---|---|
| node | `grid`, `node`, `p_gn`, `p_gnBoundaryPropertiesForStates` | `p_gn`, `p_gnBoundary…` |
| transfer | `p_gnn` | `p_gnn` |
| unit | `unit`, `unittype`, `unitUnittype`, `flowUnit`, `effLevelGroupUnit`, `p_gnu_io`, `p_unit`, `p_userconstraint` | `p_gnu_io`, `p_unit` |
| emission | `p_nEmission`, `ts_emissionPriceChange`, `gnGroup` | — |
| domains | `group`, `flow`, `emission`, `restype` | — |
| other | `index`, `add_scen_tags` | — |

Sizes below are from the Observed Trends 2030 build (`config_OT2030.ini`):
361 `p_gn` rows, 971 `p_gnu_io` rows, 503 units, 95 transfer links.

## Contents

- [The second header row](#the-second-header-row)
- [Units it drops](#units-it-drops)
- [Capacities it fills in](#capacities-it-fills-in)
- [Zero is not a number here](#zero-is-not-a-number-here)
- [Storage start levels](#storage-start-levels)
- [How a node is classified](#how-a-node-is-classified)
- [What a build says](#what-a-build-says)
- [Known open items](#known-open-items)
- [Where the input Excel builder is defined](#where-the-input-excel-builder-is-defined)

## The second header row

Five sheets are written with a **fake MultiIndex**: an extra row under the header
that is blank in the dimension columns and repeats the parameter name in every
other one. GDXXRW reads it as the column dimension, and `src_files/indexSheet.xlsx`
declares the split — `p_gn` as Rdim=2/Cdim=1, `p_gnn` as Rdim=3/Cdim=1, `p_gnu_io`
as Rdim=4/Cdim=1.

| Sheet | Dimension columns (blank in row 2) |
|---|---|
| `p_gn` | `grid`, `node` |
| `p_gnBoundaryPropertiesForStates` | `grid`, `node`, `param_gnBoundaryTypes` |
| `p_gnn` | `grid`, `from_node`, `to_node` |
| `p_gnu_io` | `grid`, `node`, `unit`, `input_output` |
| `p_unit` | `unit` |

It is a **write format, not a working one**. The builders hold ordinary frames
from first row to last, and `write_workbook` applies the transform once per sheet
on the way out; nothing reads a sheet back, so there is no inverse. That matters
if you are editing the builder: a `create_*` function never sees the marker row,
and adding a sheet to `SHEET_DIMENSIONS` is all it takes to give it one.

Because the row holds parameter *names* — text — every parameter column in a
written sheet is `object` dtype whatever it held before. That is the one place in
this project where `object` does not mean "no assumption has been made".

## Units it drops

A unit is removed from `p_gnu_io` and `p_unit` when it can neither run nor be
built:

- every `capacity` for it is zero or empty, **and**
- `invCosts` is zero or empty in `p_gnu_io`, **and**
- `maxUnitCount` is zero or empty in `p_unit`.

All three matter. Zero capacity alone is not redundant — a unit the model is
allowed to invest in is meaningful with no capacity today, which is the whole
point of an investment run. The build reports the count and names the first few;
on Observed Trends 2030 it drops 4 units, on National Trends 2040 it drops 6.

Dropping is the intended way to switch a unit off for a scenario: give it zero
capacity and no investment path in the source workbook, and it leaves the model
rather than sitting in it doing nothing.

## Capacities it fills in

Where one side of a unit has a capacity and the other does not, the builder
derives the missing one from the unit's best efficiency (the largest `eff*` in
`p_unit`). Two rules, both deliberately narrow:

1. **One input, one output.** The empty side is derived from the full one —
   `capacity / efficiency` for a missing input, `capacity * efficiency` for a
   missing output.
2. **One input, several outputs, no `cv`.** The input is derived from the *sum*
   of the outputs, again over the efficiency.

Results are rounded **up** to one decimal (`math.ceil(x * 10) / 10`), so a derived
capacity never under-states what the unit needs.

Nothing is derived when the answer would be ambiguous: when both sides are
already set, when neither is, when the efficiency is zero or empty, or — in rule
2 — when any output carries a `cv`. A `cv` means the outputs trade against each
other rather than adding up, so their sum is not the input.

## Zero is not a number here

Everywhere upstream, `0` and an empty cell are different: `0` is a value someone
wrote, and NA means nobody wrote anything. `method: replace` depends on that
distinction. **In this phase they collapse.** Backbone reads an absent parameter
and an explicit `0` identically for every parameter whose default is 0, so the
builder does too, and `fill_numeric_na` / `fill_all_na` are where a frame crosses.

The visible consequence is that **a parameter column exists only if some row set
it**. An all-zero column says nothing that its absence does not, so it is dropped
before the sheet is written. Measured on two builds:

| Sheet | OT2030 | NT2040 | of |
|---|---|---|---|
| `p_gnu_io` | 9 | 17 | 32 `param_gnu` |
| `p_unit` | 7 | 11 | 26 `param_unit` |
| `p_gn` | 7 | 7 | 17 `param_gn` |
| `p_gnn` | 5 | 5 | 12 `param_gnn` |

One parameter per sheet is kept even when empty — `capacity`, `isActive` or
`useConstant` — because a sheet with no parameter column at all is a GDXXRW
dimension error rather than an empty sheet.

If you are writing code that reads one of these frames, **guard the column**:

```python
if 'upperLimitCapacityRatio' in p_gnu_io.columns:
```

A column goes missing only when *no row in the whole model* set it, which is
exactly the case a populated test fixture does not have. Two of the three readers
of `upperLimitCapacityRatio` were guarded; the third crashed a build.

## Storage start levels

A node with a state variable needs a level to start from, or the solver may start
it full and generate energy the model never bought. For each storage node the
builder looks for a maximum, in order:

1. the node's **`upwardLimit` constant** in the boundary data, if above zero;
2. otherwise `capacity * upperLimitCapacityRatio` of the first unit on the node
   that sets a ratio.

Given one, it writes `boundStart = 1` and a `reference` constant of **70% of that
maximum**, rounded to a whole unit. Given neither, it writes nothing and names
the node — which is the case to act on, because the fix is in the data: give the
node an `upwardLimit`, or give one of its units an `upperLimitCapacityRatio`.

Two things worth knowing about that number. **For hydro it is provisional**:
`changes.inc` recomputes the reference of every `psOpen` and `reservoir` node from
the maximum of its `upwardLimit` *series*, gated on `boundStart = 1` and a
reference above zero — so what matters for those nodes is that both gates are
passed, not what the value is. The 70% is what the other storages actually get:
batteries, closed pumped hydro, gas tanks. And **it cannot express a run that
starts and ends in summer**; a start level that follows the modelled period is
work still to do.

Writing nothing rather than a zero is deliberate. Backbone gates the bound on the
reference constant's own value, where `0` is indistinguishable from absent, so a
`boundStart = 1` beside a zero reference bound nothing while looking in the
workbook as though it did.

## How a node is classified

Every `(grid, node)` pair in the model gets a `p_gn` row, and each node is either
a **price node** (`usePrice = 1`, buys and sells at a price) or a **balance node**
(`nodeBalance = 1`, its energy must balance), optionally with **state**
(`energyStoredPerUnitOfState > 0`). The workbook may say so outright; where it
does not, the builder deduces.

Explicit values in `nodedata` are taken first, then:

| Property | Deduced when | From |
|---|---|---|
| `usePrice` | not set | a `price` above zero in `nodedata` |
| `nodeBalance` | not set | the node appears in `demanddata` |
| `energyStoredPerUnitOfState` | not set, and not a price node | an `upwardLimit`, `downwardLimit` or `reference` boundary on the node — constant or timeseries, it makes no difference — or an `upperLimitCapacityRatio` above zero on any of its units |
| `nodeBalance` | still not set, node has state | having state implies a balance |

A deduced storage node gets `energyStoredPerUnitOfState = 1`; price nodes and
non-storage balance nodes get 0.

`maxSpill` and `balancePenalty` deliberately do **not** imply state. They say what
may leave a node and what an imbalance costs, not how much it holds.

Two combinations are reported as contradictions rather than resolved: `usePrice`
with `nodeBalance`, and `usePrice` with `energyStoredPerUnitOfState`. A node that
reaches the end as neither price nor balance is reported too — the model has the
node and nothing says what it is.

Measured on Observed Trends 2030: of 361 `p_gn` rows, 220 are price nodes, 141
balance nodes, and 78 of those carry state.

## What a build says

The builder is quiet on a good build. What it writes follows the rule in
[Timeseries — What a build says](timeseries.md#what-a-build-says): a warning asks
the reader to change something, and it **names the first three offenders and then
counts the rest** rather than printing a line each. A missing input file can leave
a hundred units with partial data, and the count is what tells you that is what
happened.

| Message | What to do |
|---|---|
| `N unit connection(s) name a grid but no node` | a `grid_<put>` column without its `node_<put>`; check spelling in unitdata |
| `N unit(s) have no row in the unit data` | a unit in `unitUnittype` that unitdata does not have; check spelling |
| `Dropped N unit(s) with zero capacity and no investment parameters` | expected when switching units off; check the names are the ones you meant |
| `N node(s) set both 'usePrice' and 'nodeBalance'` | contradictory; pick one in nodedata |
| `N node(s) set 'usePrice' together with 'energyStoredPerUnitOfState'` | a price node cannot hold a state; pick one |
| `N node(s) are neither price nor balance nodes` | give the node a price, a demand, or an explicit flag |
| `No storage start level could be determined for N node(s)` | see [Storage start levels](#storage-start-levels) |
| `df_unitdata has no emission_group* columns` | no unit will produce emissions; check the unittype files |
| `N unit data column(s) name a unit-level parameter with a connection suffix` | e.g. `availability_output1`; `param_unit` columns are per unit, drop the suffix |
| `p_userconstraint has N row(s) with an empty 'group'/'parameter'` | Backbone cannot resolve either; fill them in |
| `The Backbone input excel file … is currently open` | close Excel and rerun |

## Known open items

- **The storage start rule cannot follow the modelled period.** 70% of the
  maximum is a whole-year assumption; a run starting and ending in summer wants a
  different level. See [Storage start levels](#storage-start-levels).
- **`restype` is written empty every build.** Nothing produces reserve types yet,
  so the sheet exists only so the symbol is defined.
- **Nothing checks the values against anything.** Whether a capacity or a
  transfer limit is plausible is a question this phase does not ask; it checks
  form, not magnitude.

## Where the input Excel builder is defined

- `src/bb_excel/bb_excel_pipeline.py` — `BBExcelPipeline`, the `create_*`
  functions that build each sheet, and `run()`, which reads as the recipe.
- `src/bb_excel/bb_excel_writer.py` — the workbook format: the second header row,
  `SHEET_DIMENSIONS`, sheet order, column widths, the index sheet.
- `src/bb_excel/bb_excel_tables.py` — the frame helpers the builders share.
- `src/backbone_params.py` — which parameters exist per sheet and their non-zero
  defaults. The source-data and timeseries phases read it too.
- `src/utils.py` — `standardize_df_dtypes`, `fill_numeric_na`, `fill_all_na` and
  `drop_empty_parameter_columns`: the zero/NA crossing.
- `src_files/indexSheet.xlsx` — the Rdim/Cdim declaration GDXXRW reads.

## See also

- [Source workbook conventions](source-workbook-conventions.md) — the sheets this
  phase reads, and what a row in them is allowed to say
- [Timeseries](timeseries.md) — the phase before this one, what a processor may
  contribute to the source data tables, and the full build-log rule
- [Hydro data](hydro.md) — where the storage nodes and their boundaries come from,
  and why the reservoir reference is recomputed in `changes.inc`
- [Identified gaps](identified-gaps.md) — which Backbone parameters and sheets this
  phase does not write, and which of its rules are known to be provisional. The
  place to look when a parameter you expected is not on any sheet
- `tests/README.md` — the NA/zero boundary map, for anyone changing this phase
