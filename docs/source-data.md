# The source data phase

The first of the three build phases. It reads the Excel workbooks a config lists,
keeps the rows this run is about, merges them into one table per data type, and
hands those tables to everything downstream.

[Source workbook conventions](source-workbook-conventions.md) is the page for
someone *writing* a sheet. This one is for someone who wants to know what the
phase does with it — which order things happen in, why that order, and what the
phase says when something is wrong.

## One minute summary

Seven data types, each read the same way. For one build:

- a config names **9 distinct workbooks across 21 file slots**, and one workbook
  can feed four data types at once;
- every sheet whose name starts with the data type's name is read, so one entry
  in `unitdata_files` can contribute four sheets;
- **44 sheets, 363 column headers**. 54 of those headers are marked `##` and 9
  are named `note`, so **300 columns reach the pipeline**, spelling **101
  distinct (table, column) pairs**;
- rows are filtered to this scenario, year and country, then merged key by key
  in the order the config lists the files.

Three things about it are easy to get wrong and are the reason this page exists:
precedence depends on Excel tab order as well as file order; excluding a node
deletes the units connected to it, whole; and a column nothing recognises is
reported rather than silently dropped.

## Contents

- [The order, and why it is that order](#the-order-and-why-it-is-that-order)
- [Precedence: file order, then tab order](#precedence-file-order-then-tab-order)
- [What the blacklists remove](#what-the-blacklists-remove)
- [Removing something that is not there](#removing-something-that-is-not-there)
- [Columns nothing reads](#columns-nothing-reads)
- [What the phase produces](#what-the-phase-produces)
- [Where the source data phase is defined](#where-the-source-data-phase-is-defined)
- [See also](#see-also)

## The order, and why it is that order

Every data type runs the same chain, and the order is load-bearing rather than
incidental:

```
read_input_excels  ->  normalize_dataframe  ->  drop_underscore_values
   ->  expand_all_country  ->  blacklists  ->  build the topology columns
   ->  apply_whitelist  ->  merge_row_by_row
```

**Form is judged before relevance.** Everything that decides whether a row is
*well written* — the `##` marker, blank rows, repeated headers, malformed
numbers, unrecognised columns — happens in `read_input_excels`, before
`apply_whitelist` decides whether the row is *wanted*. If it ran the other way,
a bad cell in a row belonging to another scenario would disappear as merely
irrelevant, and its author would meet it only when they ran that scenario. One
build tells you about every mistake in the workbook, not only the ones this run
happens to touch.

**`country = 'all'` is expanded before the blacklists**, so an expanded row is
subject to the same exclusions as one written out by hand. It is expanded for
`nodedata`, `demanddata` and `unitdata`. `transferdata` has no `country` column
at all — it has `from_country` and `to_country` — and `userconstraintdata` drops
`country` after filtering, so neither expands.

**Topology columns are built before the node blacklist**, because a node name
has to exist before it can be excluded.

## Precedence: file order, then tab order

Several files and sheets can describe the same thing, and the last statement
wins unless the row's `method` says otherwise. The order that decides "last" has
two levels, and only the first is visible in the config:

1. **the order the config lists the files** — this is the intended lever, and
   the three-tier layering the shipped configs use depends on it: a pan-European
   base workbook, then single-topic overlays, then country overlays last;
2. **the order the sheets sit in the workbook** — Excel tab order, for the
   sheets one file contributes to one data type.

The second is worth knowing about because nothing on screen suggests it.
`ObservedTrends.xlsx` contributes four `unitdata*` sheets, so dragging one of
its tabs reorders which of them wins a key the others also mention. Row order
inside a sheet is the third and last level.

## What the blacklists remove

`exclude_grids` and `exclude_nodes` are applied before the scenario filter, in
that order, and they are matched case-insensitively whatever the config comment
used to say.

For `nodedata` and `demanddata` a row is one node, so excluding a node drops
that row and nothing else.

**A `unitdata` row is a whole unit, and it is dropped whole.** A unit declares up
to ten connections, and if any of their nodes is excluded the entire row goes —
not just that connection. This is deliberate: a unit whose heat output has no
node cannot be represented as it stands, and silently converting it into a
different unit would be a worse answer than removing it.

The build says how many units went, and the count is exact: what the merged
table would have held, minus what it holds. Measured on the shipped configs:

| Config | Units removed |
|---|---|
| `config_OT2030.ini` | 0 |
| `config_NT2030.ini`, `config_NT2040.ini` | 1 — `ES00 / solar thermal` |

That unit's only real connection is `ES00_dheat`, which the configs exclude, so
nothing beyond the excluded node is lost. It is worth re-measuring rather than
trusting this table if you add a unit to a country whose district heating is
excluded, because a CHP plant is exactly the case where dropping the unit whole
would also remove its electricity capacity.

It stays one line at any scale. Excluding all eighteen Spanish nodes from
`config_OT2030.ini` — a real way to shorten a run — reports 18 units, once.

Counting rows would overstate it three ways, which is why it does not: several
sheets can describe one unit, a row may belong to another scenario this run
never wanted, and a `remove` row may have been going to delete the unit anyway.
The Spanish case is 30 rows, 19 keys and 18 units.

Taking a country out that way and taking it out of `country_codes` produce the
same model. Measured on `config_OT2030.ini` without Spain, both routes give 283
nodedata rows, 60 demanddata, 488 unitdata and 92 transferdata, and neither
warns about anything.

## Removing something that is not there

`method = remove` deletes the record a key already has. A `remove` row whose key
matches nothing does nothing, and **that is not an error** — 49 of them run on
`config_OT2030.ini` and 21 on the National Trends configs, every one of them
deliberate. `ObservedTrends.xlsx` removes offshore wind from landlocked
countries by writing a `remove` row for every country and letting it miss the
ones that never had any, which is a reasonable way to write the sheet.

The cost is that a **misspelled** `remove` is indistinguishable from a
deliberately idempotent one, and nothing can tell them apart. The other
arithmetic methods are different: `add` and `multiply` change a value that
already exists, so a key matching nothing there is reported.

## Columns nothing reads

A header is free text and underscore is a node-name separator, so nothing about
a wrong column name looks wrong. `capacty` sits in a sheet looking exactly like
`capacity`, and the unit built from that row has no capacity.

Every sheet is therefore checked against the vocabulary of column names its data
type can carry, and anything else is reported by file and sheet, spelled the way
it was typed. The vocabulary comes from three places, because a source column has
three possible consumers:

- **the parameters** come from `backbone_params.py`, per table, so they cannot
  drift from what the builder writes;
- **the dimension columns** belong to this phase;
- **the derivation inputs** — `lp/mip`, `twh/year`, `constant_share` — are read
  by name at a call site in a later phase and are declared by hand, with a test
  that greps for each so a declaration cannot outlive its reader.

The message offers three remedies because there are three reasons a column can
be unread and nothing can tell them apart: it is misspelled, it is the author's
own working material and should say so with `##`, or it names a real Backbone
parameter this build does not write yet. See
[Identified gaps](identified-gaps.md) for that last list.

Silent on every sheet the four shipped configs name, and a test sweeps them to
keep it that way. The eight workbooks in `src_files/data_files/` that no config
lists are a different matter: their columns genuinely are unread, and listing one
of them will say so.

## What the phase produces

Seven tables, held on the pipeline object rather than returned:

| Table | Key |
|---|---|
| `df_nodedata` | country, grid, node |
| `df_demanddata` | country, grid, node |
| `df_unitdata` | country, generator_id, unit_name_prefix |
| `df_transferdata` | from_country, from_suffix, to_country, to_suffix, grid |
| `df_emissiondata` | emission, group |
| `df_userconstraintdata` | group, the four dimensions, parameter |
| `df_boundarydata` | grid, node, param_gnBoundaryTypes |

`df_unitdata` is the merged result: `merge_unittypedata_into_unitdata` folds the
type-level defaults into it, and `df_unittypedata` is not exposed afterwards.
`df_boundarydata` is derived rather than read — `build_boundarydata` melts
`nodedata`'s wide boundary columns — and is the one table with no workbook of its
own.

Every one of them obeys the source-side conventions: three dtypes only
(`Float64`, `object`, `string`), an all-NA column is `object`, and **`pd.NA` and
`0` are different things**. Only the Excel builder collapses them.

The timeseries phase may add rows to these tables afterwards, through the
contribution merge, and the workbook always wins where both have something to
say. See [Timeseries](timeseries.md).

## Where the source data phase is defined

- `src/source_data/source_data_pipeline.py` — `SourceDataPipeline`, the chain
  above, one block per data type
- `src/source_data/source_data_loader.py` — every step of it
- `src/source_workbook_shape.py` — which table declares a dimension's values, and
  which column names a table can carry
- `src/backbone_params.py` — the parameter vocabulary, shared with the other two
  phases
- `src/source_data/source_data_contributions.py` — `build_boundarydata`, and the
  merge the timeseries phase uses

## See also

- [Source workbook conventions](source-workbook-conventions.md) — the same
  subject from the sheet author's side: what to write and what the builder makes
  of it
- [Input Excel builder](input-excel.md) — what becomes of these seven tables
- [Timeseries](timeseries.md) — what a processor may contribute to them
- [Identified gaps](identified-gaps.md) — what the workbooks cannot say yet
- `tests/README.md` — the NA/zero boundary map, for anyone changing the phase
