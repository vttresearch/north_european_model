# The source data phase

The first of the three build phases. It reads the Excel workbooks a config lists,
keeps the rows this run is about, merges them into one table per data type, and
hands those tables to everything downstream.

[Source workbook conventions](source-workbook-conventions.md) is the page for
someone *writing* a sheet. This one is for someone who wants to know what the
phase does with it — which order things happen in, why that order, and what the
phase says when something is wrong.

## One minute summary

- **Seven data types, each read the same way**: `unittypedata`, `emissiondata`,
  `nodedata`, `demanddata`, `unitdata`, `transferdata`, `userconstraintdata`, in
  that order, because a unit needs its unittype before it can be named.
- **One workbook can feed several types at once**, and every sheet whose name
  starts with the type's name is read — so one entry in `unitdata_files` can
  contribute four sheets.
- **Rows are filtered to this scenario, year and country, then merged key by key**
  in the order the config lists the files.
- **To see the actual inventory, read a config.** `src_files/config_OT2030.ini`'s
  seven `*_files` lists are the whole of what a build opens, and `summary.log` in
  the output folder says what that particular run made of them.

Three things about it are easy to get wrong and are the reason this page exists:
precedence depends on Excel tab order as well as file order; excluding a node
deletes the units connected to it, whole; and a column nothing recognises is
reported rather than silently dropped.

## Contents

- [The order, and why it is that order](#the-order-and-why-it-is-that-order)
- [How the types differ](#how-the-types-differ)
- [Precedence: file order, then tab order](#precedence-file-order-then-tab-order)
- [A key written twice in one sheet](#a-key-written-twice-in-one-sheet)
- [Removing something that is not there](#removing-something-that-is-not-there)
- [What the blacklists remove](#what-the-blacklists-remove)
- [Columns nothing reads](#columns-nothing-reads)
- [Where a message points, and why that has to be collected](#where-a-message-points-and-why-that-has-to-be-collected)
- [What the phase produces](#what-the-phase-produces)
- [Where the source data phase is defined](#where-the-source-data-phase-is-defined)
- [See also](#see-also)

## The order, and why it is that order

Every data type runs the same chain, and the order is load-bearing rather than
incidental. For `nodedata` and `demanddata`:

```
read_input_excels  ->  normalize_dataframe  ->  drop_underscore_values
   ->  expand_all_country  ->  exclude_grids  ->  build_node_column
   ->  exclude_nodes  ->  apply_whitelist  ->  collect_origins
   ->  merge_row_by_row
```

`unitdata` differs in the middle, because a unit's name and its connections both
come from the unittype:

```
   ...  ->  expand_all_country  ->  canonicalize_unittype_and_build_unit
   ->  build_unit_grid_and_node_columns  ->  exclude_grids  ->  exclude_nodes
   ->  apply_whitelist  ->  collect_origins  ->  merge_row_by_row
   ->  merge_unittypedata_into_unitdata
```

**Form is judged before relevance.** Everything that decides whether a row is
*well written* — the `##` marker, blank rows, repeated headers, malformed
numbers, blank key columns, unrecognised columns — happens in
`read_input_excels`, before `apply_whitelist` decides whether the row is
*wanted*. If it ran the other way, a bad cell in a row belonging to another
scenario would disappear as merely irrelevant, and its author would meet it only
when they ran that scenario. One build tells you about every mistake in the
workbook, not only the ones this run happens to touch.

**`country = 'all'` is expanded before the blacklists**, so an expanded row is
subject to the same exclusions as one written out by hand.

**Node names are built before the node blacklist**, because a node name has to
exist before it can be excluded — which is why the grid blacklist and the node
blacklist sit on either side of the topology step rather than together.

## How the types differ

The chain above is the common case. What each type does with it:

| Table | Filtered by | `country = all` expanded | `_` rows dropped | Provenance collected |
|---|---|---|---|---|
| `unittypedata` | scenario, year | no | yes | unittype, grid |
| `emissiondata` | scenario, year | no | yes | none |
| `nodedata` | scenario, year, country | yes | yes | node, grid |
| `demanddata` | scenario, year, country | yes | yes | node, grid |
| `unitdata` | scenario, year, country | yes | yes | node, unittype |
| `transferdata` | scenario, year, both countries | no | yes | node, grid |
| `userconstraintdata` | scenario, year, country | no | **no** | none |

`transferdata` has `from_country` and `to_country` rather than `country`, and both
ends are judged in **one** filtering pass: two passes would re-apply the shared
scenario and year filters and raise every warning they produce twice.

`userconstraintdata` is the one table exempt from `drop_underscore_values`,
because its four dimension slots hold node and unit names, which contain
underscores by construction. It also drops `scenario`, `year` and `country` after
filtering, since its own key makes all three redundant — and drops them
tolerantly, because a sheet legitimately omits `country`, and a sheet whose every
row is marked `##` arrives with no columns at all.

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

## A key written twice in one sheet

Overriding a value across files is what that file order is for. Two rows for one
key inside **one** sheet is a different statement: the earlier row is applied and
then discarded, so it is dead text, and the build says so.

It runs at merge time, after the whitelist, which is what keeps it quiet on a
sheet holding six years of `replace` rows for one unit: only this run's year is
still there by then.

The check is also narrower than "a key seen twice", deliberately. `add` and
`multiply` accumulate by definition — `dheat_unitdata_PL_DE_AT.xlsx` writes
`DE00 / gasCCGTpresent2` twice as `add-non-negative`, two deliberate reductions
that stack — so only `replace` and `replace-partial`, which discard, are
reported. A check that speaks every run about something correct is not strict, it
is broken.

## Removing something that is not there

`method = remove` deletes the record a key already has. A `remove` row whose key
matches nothing does nothing, and **that is not an error** — the shipped configs
run dozens of them, every one deliberate. `ObservedTrends.xlsx` removes offshore
wind from landlocked countries by writing a `remove` row for every country and
letting it miss the ones that never had any, which is a reasonable way to write
the sheet.

The cost is that a **misspelled** `remove` is indistinguishable from a
deliberately idempotent one, and nothing can tell them apart. The other
arithmetic methods are different: `add` and `multiply` change a value that
already exists, so a key matching nothing there is reported. They then diverge —
`add` creates the record from zero, `multiply` creates nothing at all, because
writing a multiplier as though it were a quantity is how a `0.5` meant to halve a
capacity becomes a capacity of `0.5`.

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
| `config_NT2030.ini`, `config_NT2040.ini` | 1 — `ES00 / solarThermal` |

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
nodedata rows, 60 demanddata, 487 unitdata and 92 transferdata, and neither
warns about anything.

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

Keeping the vocabularies per table rather than pooling them costs nothing on the
shipped data and is what catches an emission factor written on a unit sheet, or a
`country` column on one of the three tables that never filter by it.

The message offers three remedies because there are three reasons a column can
be unread and nothing can tell them apart: it is misspelled, it is the author's
own working material and should say so with `##`, or it names a real Backbone
parameter this build does not write yet. See
[Identified gaps](identified-gaps.md) for that last list.

Silent on every sheet the four shipped configs name, and a test sweeps them to
keep it that way. The workbooks in `src_files/data_files/` that no config lists
are a different matter: their columns genuinely are unread, and listing one of
them will say so. [Identified gaps](identified-gaps.md) names which ones those
currently are.

## Where a message points, and why that has to be collected

`merge_row_by_row` keeps the values and drops where they came from — the
`_source_file` and `_source_sheet` columns `read_input_excels` stamps on every
frame. So any check running after the merge can say a node is unusable but not
which sheet to open.

`collect_origins` is the answer: it indexes each node, grid and unittype against
the sheets that wrote it, while the per-sheet frames still exist, and hands that
index to whoever reports. It is keyed case-insensitively but keeps the spelling,
because the two together are what diagnoses a rename — a sheet holding the same
name in a different case is not an unrelated sheet, it is the one that was
missed.

Two consumers use it today: this phase, for a unittype no `unittypedata`
declares, and the Excel builder, for a node it cannot classify.

## What the phase produces

Seven tables, held on the pipeline object rather than returned:

| Table | Key |
|---|---|
| `df_nodedata` | country, grid, node |
| `df_demanddata` | country, grid, node |
| `df_unitdata` | country, unittype, unit_name_prefix |
| `df_transferdata` | from_country, from_suffix, to_country, to_suffix, grid |
| `df_emissiondata` | emission, group |
| `df_userconstraintdata` | group, the four dimensions, parameter |
| `df_boundarydata` | grid, node, param_gnBoundaryTypes |

`df_emissiondata`'s key is both columns together because a group is what carries
a price: two groups pricing one emission are two rows, and merging on the
emission alone collapsed them into a silent override.

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
