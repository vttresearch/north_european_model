# Source workbook conventions

The Excel files in `src_files/data_files/` are what says the model exists: which
countries, which nodes, which units, and every number attached to them. This page
is for someone filling in a sheet — what to write, and what the builder makes of
it. `inputData.xlsx` is not one of these files; the builder writes that one.

## One minute summary

- **A config file lists workbooks per data type, and every sheet whose name starts
  with that type's name is read.** A scratch sheet only needs a name that does not.
- **A row has dimension columns saying which thing it is about and parameter
  columns holding numbers.** Parameter names are Backbone's own; the full list is
  `docs/dictionary.md` in the Backbone repository.
- **Several sheets may describe one thing, and the later one wins.** `method` on
  the later row says what "wins" means — replace, add, multiply, remove.
- **`##` marks what is yours rather than the model's**, in a cell or in a header,
  and it is removed before anything is checked.
- **A fully empty row ends the sheet.** It is not a spacer.
- **A name typed in a cell is often also a formula's lookup key.** Renaming the
  data does not break `SUMIF` or `VLOOKUP`; it makes them return the wrong row,
  and no build check can see it.

| You write | The builder does |
|---|---|
| `##` in a cell, or as a header | ignores that row, or that column |
| a fully empty row | stops reading the sheet there |
| `all` as scenario or country, `1` as year | the row applies to every one of them |
| a blank `scenario` or `year` | reports the row and drops it |
| `method` | decides what a later row does to an earlier one |
| anything the builder cannot read | reports it and reads it as not set |

Read [Writing a sheet](#writing-a-sheet) once. The rest is there for when a build
log says something you did not expect. The full lookup table is
[Quick reference](#quick-reference), at the end.

## Contents

- [Writing a sheet](#writing-a-sheet)
- [Naming things](#naming-things)
- [Combining rows](#combining-rows)
- [What each table needs](#what-each-table-needs)
- [When a build complains](#when-a-build-complains)
- [Renaming a value a formula depends on](#renaming-a-value-a-formula-depends-on)
- [Editing a workbook reruns processors](#editing-a-workbook-reruns-processors)
- [Why the rules are what they are](#why-the-rules-are-what-they-are)
- [Quick reference](#quick-reference)
- [See also](#see-also)

---

## Writing a sheet

### Which sheets are read

A config file lists workbooks per data type. From `config_OT2030.ini`:

```ini
unitdata_files = ['ObservedTrends.xlsx',
                  'dheat_unitdata_SE_DK.xlsx',
                  'industrialCHP.xlsx',
                  ...
                  ]
nodedata_files = ['ObservedTrends.xlsx',
                  'industrialCHP.xlsx',
                  'hydroUpd-v1.xlsx',
                  ...
                  ]
```

Inside each listed workbook the builder takes every sheet whose name **starts
with** the matching prefix, case-insensitively. `ObservedTrends.xlsx` appears
under `unitdata_files`, so its `unitdata_VRE`, `unitdata_battery`,
`unitdata_nuclearThermal` and `unitdata_demandRed` sheets are all read as unit
data. The same workbook is listed under `nodedata_files` too, and there its
`nodedata` sheet is read.

The prefixes are `unitdata`, `unittypedata`, `nodedata`, `transferdata`,
`demanddata`, `emissiondata` and `userconstraintdata`. Each is a whole category
name: `unittypedata` does not start with `unitdata`, so the two never collect
each other's sheets. A sheet whose name matches no prefix is never opened.

The order of the file list matters — see [Combining rows](#combining-rows).

### What a row says: dimensions and parameters

Every sheet has two kinds of column. `unitdata_VRE` in `ObservedTrends.xlsx`:

| Country | unittype | Scenario | Year | capacity_output1 | vomCosts | method |
|---|---|---|---|---|---|---|
| AT00 | solarPV | Observed Trends | 2015 | 937 | 1.69 | |
| BE00 | solarPV | Observed Trends | 2015 | 3132 | 1.68 | |

**Dimension columns** say *which thing* the row is about: `Country`, `unittype`,
`Scenario`, `Year`. They hold labels, and together they are the key the builder
uses to recognise that two rows describe the same unit.

**Parameter columns** hold the numbers: `capacity_output1`, `vomCosts`. Their
names are Backbone parameter names — the full list is `docs/dictionary.md` in the
Backbone repository. A cell in one of these has to be a number; see
[Cells that should be numbers](#cells-that-should-be-numbers) for what happens
when it is not.

`method` is neither; it is an instruction, described under
[Combining rows](#combining-rows).

Column names are matched case-insensitively, so `Country` and `country` are the
same column.

### Connection suffixes

A unit can have several inputs and outputs, and a parameter belonging to one of
them carries a suffix: `capacity_output1`, `grid_input2`. Valid suffixes run
`_input1`–`_input5` and `_output1`–`_output5`.

Writing the bare name is the common case and means `_output1`: the `capacity`
column above is `capacity_output1`, which is why the sheet spells it out.

Some parameters belong to the unit as a whole rather than to one connection —
`availability`, `unitCount`, `eff00`. Giving one of those a connection suffix is
reported and the column ignored. `_output1` is the exception, and only because it
is the one suffix that means "no suffix": it is stripped, and the column is read.

### Rows that apply to everything

Three dimension columns have a value meaning *any*:

| Column | Write | Meaning |
|---|---|---|
| `scenario` | `all` | every scenario the config runs |
| `year` | `1` | every scenario year |
| `country` | `all` | every country in `country_codes` |

`year = 1` is the one nobody guesses. It is not year 1 and not a placeholder; it
is how a row says the number does not change between 2030 and 2040, which saves
writing one block of rows per year. Most sheets in the shipped workbooks use it;
`unittypedata_compilation.xlsx` writes `all` and `1` on every row it has.

**A blank in either cell is not "every run".** A row that never says which
scenario or year it belongs to is reported and dropped, because reading a blank
as `all` would promote a half-finished row into every scenario the workbook
holds. The catch-alls exist so you can say "every run" on purpose.

`country = all` differs from the other two in when it acts: it is *expanded* into
one row per country before the exclusions run, so an expanded row can then be
excluded like any other. `scenario` and `year` are matched at filter time and
never expanded. See [The source data phase](source-data.md).

A row you want for one scenario or one year writes that value instead, and both
kinds can sit in the same sheet — the specific row and the `all`/`1` row are
merged in file order like everything else, so the later one wins.

### Marking what is not input: `##`

A workbook is a working document as well as a data source. `##` is how you say a
part of it is yours rather than the model's:

- **`##` in any cell of a data row** ignores that row. `ObservedTrends.xlsx` uses
  this for section headings — `## PV`, `## Onshore` and `## Offshore` in the
  `Country` column of `unitdata_VRE`, and `## Tier 1`, `## Tier 2`, `## Tier 3`
  in `unitdata_demandRed`.
- **`##` as a column header** ignores that column. Use it for the helper table you
  keep beside the real one. `unitdata_battery` and `unitdata_nuclearThermal` in
  the same workbook do this.

A bare `##` works as the header of every helper column — Excel is happy with
duplicate headers, and so is the builder.

`unittypedata` uses this for `## Description`, where it says in words what the row
is: `windOnshore` is an onshore wind turbine. Nothing reads it, which is the point
— a unittype is a name the model uses, and the sentence explaining it belongs
beside it rather than in someone's head.

Marked rows and columns are removed **before anything is checked**, so a
half-finished formula, a `#DIV/0!` or a pasted `1,000.0` sitting in your working
area is never reported as a problem. Nothing is logged: you said what you meant.

Columns are removed before rows are judged. That ordering means a `##` you typed
as free text out in the helper area cannot delete the row of the real table it
happens to sit beside.

### Where a sheet ends

**A fully empty row ends the sheet.** Everything below it is ignored. A row counts
as empty when every cell is blank or whitespace. This is how you stop the table;
it is not a spacer you can put in the middle of one. If real rows do follow it,
they are dropped and you get a warning saying how many.

**A column with no header is ignored.** Past the last named column that is the
scratch area and nothing is said — several shipped sheets rely on it. A column
with no header sitting *inside* the table is different: its values are dropped
and you get a warning, because that is usually a header someone deleted rather
than a decision. Marking the column `##` says "working material" deliberately and
keeps it quiet.

**Leave no blank row above the header.** The reader takes the first row as the
header, so a blank one makes every column unnamed and there is nothing left to
identify. That is reported as an error and the sheet is skipped.

---

## Naming things

### Underscore is the node-name separator

Node names are built as `{country}_{grid}`, or `{country}_{grid}_{node_suffix}`
when a suffix is given, so an `_` inside a text cell would produce a name nobody
can take apart again. Any row containing one is dropped, with a warning naming the
column and showing examples.

`userconstraintdata` is the one table exempt, and has to be: its four dimension
slots hold node and unit names, which contain underscores by construction.

### Case is folded, first spelling wins

`scenario` and `method` values are lower-cased, as are column names. Rows match
case-insensitively while the spelling used first is what reaches the output. GAMS
treats `dh` and `DH` as one set element and refuses a GDX containing both, so this
is not a nicety.

It also means an overlay cannot rename anything by changing its case. A later row
written `DH` matches a record established as `dh` and edits it; the label stays
`dh`.

### A unit is named after its unittype

The name is `{country}_{unittype}`, or `{country}_{unit_name_prefix}_{unittype}`
with a prefix, so the `unittype` column cannot be case-folded like the others —
that would rename every unit in the model. Instead your sheet's spelling is
*replaced* by the one `unittypedata` uses: write `chpbio` against a `CHPbio` row
and the unit is `FI_CHPbio`.

A `unittype` no `unittypedata` sheet declares keeps the spelling you wrote, gets
no grids, no nodes and no type-level defaults, and so reaches the model as nothing
at all. Every one of those is reported by name, with the sheet it was written in
and how many units it costs.

### A mistyped suffix makes a new node

`node_suffix` and `unit_name_prefix` are part of the merge key and are built into
the node and unit name. A typo does not fail to override an earlier row — it
quietly creates a second node, and the value you meant to change stays as it was.
Nothing can detect this, because an intended new node looks exactly the same.

---

## Combining rows

### The methods

Several files and sheets can describe the same thing. They are applied in the
order the config lists them, and `method` on a later row says what that row does
to what came before:

| `method` | Effect |
|---|---|
| `replace` | overwrite the whole row; empties and zeros included |
| `replace-partial` | overwrite only the columns you filled in; zero counts as filled |
| `add` | add into the parameter columns |
| `add-non-negative` | as `add`, but never below zero |
| `multiply` | multiply the parameter columns |
| `remove` | delete the earlier row for this key |

An empty `method` cell means `replace`. An unrecognised value is reported and
treated as `replace`.

`add`, `add-non-negative` and `multiply` touch **only the columns your row fills
in**. A blank leaves the earlier value alone, so two arithmetic rows give the same
answer in either order — and `add-non-negative` cannot clamp a column your row
never mentioned.

Where one side is missing: adding to a missing value treats the missing one as
`0.0`, multiplying by a missing new value leaves the old one unchanged, and
missing on both sides stays missing. A missing value is not a zero here.

### Which order is "later"

**"Later" has three levels, and only the first is visible in the config.** The
file order decides between files; within one file, the order its sheets sit in
decides between them; within one sheet, row order.

The middle one is worth knowing about because nothing on screen suggests it:
`ObservedTrends.xlsx` contributes four `unitdata*` sheets, so dragging one of its
tabs changes which of them wins a key the others also mention.

### A key written twice in one sheet

Overriding a value is what the file order is for. Two rows for one key inside
**one** sheet is a different thing: the earlier row is read, applied, and then
discarded, so it is dead text. That is reported, naming the key and the sheet.

`add` and `multiply` are exempt, because stacking is exactly what they are for —
two `add-non-negative` rows for one unit are two deliberate reductions, not a
mistake. Only `replace` and `replace-partial` discard what the sheet already said.

Rows for other years do not count either: this is judged after the scenario and
year filter, so a sheet holding one block of rows per year says nothing.

### `add` and `multiply` need something to change

An arithmetic row is an instruction to change a value that already exists, so a
key matching nothing is nearly always a misspelled key. Both are reported, and
they then differ: `add` creates the record from zero, `multiply` creates nothing
at all — writing the multiplier as though it were a quantity is how a `0.5` meant
to halve a capacity becomes a capacity of `0.5`.

A `remove` matching nothing is silent and deliberate; see
[The source data phase](source-data.md) for why.

---

## What each table needs

### Where a grid or a node comes from

Four sheets can bring one into being, and none of them is more official than the
others:

- a `nodedata` row, one node;
- a `demanddata` row, one node;
- a `unitdata` row, **one per connection** — a battery charger declares a
  `battery` output, which creates that grid and its `XX00_battery` node, and
  every fuel grid arrives the same way;
- either end of a `transferdata` link.

**Which grid a unit connection has is `unittypedata`'s to say, not `unitdata`'s.**
`grid_output1` is read from the unittype's row and written over whatever the
unitdata sheet holds in that column — so a grid written on a unitdata sheet is
discarded, without a word, and the cell is left blank when the unittype does not
define that connection at all. What a unitdata row contributes to the node name is
the country and the optional `node_suffix_output1`.

So there is no sheet you must declare a node in before using it elsewhere. The
cost of that convenience is that a mistyped name is indistinguishable from a new
node — it simply appears in the model, connected to whatever the typo was in.
The one place the build can tell is a **timeseries processor** naming a node none
of the four sheets has, and it reports that; between the sheets themselves it
cannot, and does not pretend to.

### `nodedata` columns that stand for a value

Two `nodedata` column families name a value rather than a parameter, so the set
of them is open-ended and no list has to be kept up to date.

`emission_CO2`, `emission_CH4` and so on give the node's emission factor per
MWh. The part after the underscore is the emission's name, and whatever you write
there becomes an emission in the model.

`upwardLimit`, `downwardLimit`, `reference`, `maxSpill` and `balancePenalty` give
the node's state boundaries — one column per boundary type, because one row per
node is what a spreadsheet is good at. Backbone indexes them the other way round,
so the build turns each non-blank cell into a row saying "this node's upwardLimit
is this constant". That matters when something else has an opinion about the same
boundary: a timeseries processor can say a node's limit follows a seasonal
profile instead, and then the profile is used and the column is not. Your value
wins wherever you wrote one — a processor can only fill a gap, never overwrite.

### Excluding a grid or a node

`exclude_grids` and `exclude_nodes` in the config remove a grid or a node from the
model, and they match whatever case the workbook spelled it in. For a `nodedata`
or `demanddata` row that is one row, and it is the only thing that goes.

A `unitdata` row is a whole unit, and it is **removed whole**. A unit declares up
to ten connections, and if any one of their nodes is excluded the entire unit
goes, not just that connection. A unit whose heat output has no node cannot be
represented as it stands, and quietly turning it into a different unit would be
a worse answer than removing it. The build says how many units left with the
exclusion, once, as a count.

Worth checking when you exclude a district heating node in a country that has
CHP, because the plant's electricity capacity leaves with it.

### `transferdata` is one row per direction

A link is unidirectional: every line and every direction needs its own row, with
`transferCap` as the capacity column. `export_capacity` and `import_capacity` are
the old bidirectional format, where one row carried both. They reach no Backbone
parameter now, and the build names the columns once at the end of the merge — not
the sheet, because by then it no longer knows which sheet they came from.

A link survives only if the model carries both of its countries, and each end is
judged on its own.

### `country`, and the three tables that ignore it

`unittypedata`, `emissiondata` and `transferdata` are not filtered by country —
`transferdata` has `from_country` and `to_country` instead, and the other two are
global. A `country` column on any of the three is reported as a column nothing
reads, because an author who writes one believes they made something
country-specific and did not.

`userconstraintdata` is filtered by country and then drops the column, along with
`scenario` and `year`: its own key makes all three redundant. A sheet that omits
`country` because the constraint is not about one country is fine.

`emissiondata` is keyed on `emission` **and** `group` together, because a group is
what carries a price. Two groups pricing the same emission are two rows.

---

## When a build complains

### Where the message points you

A message about a node, a grid or a unittype names the workbook and sheet the
name was written in. The builder has to collect that while the per-sheet tables
still exist, because merging them keeps the values and not where they came from.

When two sheets spell one name differently the message says so — "spelled
`CHPbio` in ObservedTrends.xlsx:unitdata_VRE" — because that is the whole
diagnosis of a rename that reached one sheet and not the other.

### A row that identifies nothing

A blank in a column that says *which thing* the row is about — `country`, `grid`,
`unittype` and their equivalents on the other tables — leaves a row describing
nothing. It is reported with its spreadsheet row number, and so is a year that
could not be a year, `0` most often, which matches no run at all.
`unit_name_prefix` and the node suffixes are exempt: blank is their normal state.

Both are at **error** level: the build finishes and still writes its output, but
it is marked as failed and re-runs from scratch next time.

A row for another scenario, year or country is a different thing entirely: it is
*dropped* without a word, because that is what those sheets are for. Its cells
are still read and still checked, so a bad number in it is reported all the same.

### Excel error values

`#REF!`, `#N/A`, `#DIV/0!`, `#VALUE!`, `#NAME?`, `#NUM!` and their relatives are
reported wherever they appear, in dimension columns as much as parameter columns,
and read as not set. None is ever a value anyone meant to write. `#REF!` in
particular is what Excel leaves behind when a column another sheet pointed at is
deleted, so it usually means the workbook has quietly lost a reference.

### A header used twice

Excel lets two columns carry the same header. The builder reads the first and
warns about the rest, naming the file, the sheet, the header and how many values
are being passed over. Give the second column its own name, or mark it `##` if it
is working material.

One pair looks like a duplicate and is reported differently. A bare parameter
name means `_output1`, so `vomCosts` and `vomCosts_output1` in the same sheet are
two spellings of the same column. That is reported as a rename collision, and the
suffixed column is left as it is rather than one silently overwriting the other.
Use one spelling or the other.

### Cells that should be numbers

A parameter column has to hold numbers. The builder does not try to interpret one
that does not:

`1,000.0` · `1 000` · `1'000` · `12,345,678` · `1.000,5` · `1_000` · `100 MW` ·
`100MW` · `5%` · `€100` · `(500)` · `−5`

Each is read as **not set**, which the model treats as zero, and reported: one
message per workbook, naming the three columns with the most bad cells and
counting the rest. Per column would be right for one stray cell and unreadable
for a workbook whose export changed format, and that case is the one that buries
every other warning in the build.

The report is at **error** level, and the message is repeated in the summary at
the end. Look in `summary.log` in the output folder.

**Rows this run does not use are checked too.** A row for another scenario, year
or country is reported before it is filtered out, so one build tells you about
every bad cell in the workbook rather than only the ones this scenario happens to
touch.

Text that never looked like a number — `unknown` in a capacity column — is not
caught. Nothing in the cell distinguishes it from a label, and a rule aggressive
enough to catch it would eat identifiers like `chp1`.

### A column nothing reads

A header is free text, so nothing about a wrong one looks wrong. `capacty` sits
in a sheet looking exactly like `capacity`, and the unit built from that row
simply has no capacity.

Every sheet is checked against the column names its data type can carry, and
anything else is reported by file and sheet, spelled the way you typed it. There
are three reasons a column can be here and the builder cannot tell them apart,
so the message offers all three:

- **it is misspelled** — the usual case, and the one this exists for;
- **it is yours, not the model's** — put `##` at the start of the header, and
  nothing more is said about it;
- **it names a real Backbone parameter this build does not write yet** —
  `profitMargin`, `invCost`, the `hr*` heat-rate family and about a hundred
  others. [Identified gaps](identified-gaps.md) has the list. The column still
  reaches nothing, which is why it is reported rather than quietly accepted.

Nothing is dropped on account of this. The column is carried exactly as before;
the only change is that the build now says it is going nowhere.

The set of names is not a fixed list you have to keep up with. Parameter names
come from Backbone's own vocabulary, `emission_<name>` on a node and
`emission_group<n>` on a unit are open-ended families, and a connection suffix is
understood wherever the bare name is — `capacity_input3` is recognised because
`capacity` is.

The check only covers the workbooks a config lists. A workbook sitting in
`src_files/data_files/` that no config names is read by nothing at all, so
nothing is said about its columns either.

### A node only one table knows about

`nodedata` and `demanddata` both bring nodes into being, and a node that one of
them has and the other does not is usually half of an edit. Every such node is
named, per grid — and only for a grid the two tables mostly agree about, so a
grid that legitimately lives in one of them stays quiet. `elec` and `hydrogen`
carry no `nodedata` rows at all and are therefore never reported.

### Timeseries input files

The CSV and Excel files a timeseries processor reads are held to a stricter rule:
a malformed number makes the processor refuse the file and write no output, rather
than blanking the cell.

These files are machine-generated, and a generator does not make isolated typos.
One bad number means whatever produced it changed format, so blanking would
manufacture a column of zeros indistinguishable from real data. In a
comma-delimited file an unquoted `1,000.0` is worse again: the comma is the
delimiter, so the row gains a field and every column after it shifts — the node
label becomes a number, and nothing further down could tell.

If your source genuinely uses a missing-value marker of its own, the processor can
declare it. `NA`, `N/A`, `n/a`, `NULL`, `NaN`, `None` and `#N/A` are recognised
already.

---

## Renaming a value a formula depends on

This is the one hazard on this page that no build check can reach, and the one
that has cost the most.

A name typed in a cell is often also a **lookup key**. `SUMIF`, `VLOOKUP` and
`COUNTIF` criteria, and the helper tables they point at, refer to unittypes, node
names and country codes by their text. None of them fails when the data is
renamed. They return `0`, or the value from a neighbouring row, and the sheet
still looks finished.

Renaming the unittypes in one pass broke six such formulas. One of them silently
deleted every Finnish district-heating heat pump and electric boiler from the
model. Not one was found by the build; all six were found by comparing numbers
before and after.

So when you rename anything a formula might key on:

- Use Excel's Find & Replace with **Within: Workbook** and **Match entire cell
  contents**. Workbook scope is what moves the helper tables in the same pass as
  the data they key; whole-cell matching is what stops `Nuclear` from eating
  `Nuclear-flex`.
- Re-read any formula that classified rows by the text of a name. A criterion
  like `"*heat pump*"` stops matching when `Ground source heat pump` becomes
  `GSHP`.
- Check the result by number, not by eye. `tools/compare_source_workbooks.py`
  compares two versions of the workbooks row by row on their dimension columns,
  and `--git-ref` takes the earlier version straight from git. It is also the
  only readable diff a binary `.xlsx` has.
- `python tools/check_unittype_columns.py src_files/data_files` is the cheaper
  name-level check: every `unitdata` unittype declared by some `unittypedata`
  sheet, and with `--legacy-names`, no cell anywhere still holding an old name.

## Editing a workbook reruns processors

A timeseries processor can declare that it reads a source table, and `VRE_PECD`
declares `unitdata` — it needs to know which nodes have a unit of its flow. So
editing one `unitdata` sheet reruns the PV and both wind processors, and a build
that starts three weather processors after a workbook edit is doing the right
thing rather than the wrong one. `unittypedata` counts as `unitdata` here, because
the two are folded into one table before any processor sees it.

The build names the processors it reruns and what woke each one.
[Timeseries](timeseries.md) has the full rule.

---

## Why the rules are what they are

### Why the marker is two hashes

A single `#` used to mark a comment row, and it was the wrong character. Every
Excel error value starts with one — `#REF!`, `#N/A`, `#DIV/0!` — so a formula that
broke silently deleted the row it sat in, and a broken reference could remove a
power plant from the model without a word.

A single `#` no longer marks anything. No Excel error value begins with `##`, and
unlike `=`, `+`, `-` and `@`, a hash does not make Excel treat the cell as a
formula.

### Why a bad number is not repaired

`1.000` is a thousand to an author writing in one locale and one to an author
writing in another, and the cell carries nothing that says which. A builder that
guessed would put a confidently wrong number into the model, where nothing would
ever reveal it. Blanking the cell and naming it means you fix it once, at the
source.

### Why a repeated header is not merged

Nothing in the sheet says which value should win or how two of them should
combine, and a wrong answer chosen automatically is worse than a question asked
out loud.

### `note` columns

A column named exactly `note` is dropped. This predates `##` and is kept because
shipped workbooks still use it. Prefer `##`: it is not limited to one name or one
column per sheet, and a second `note` column arrives as `note.1`, which the old
rule never matched.

---

## Quick reference

| You write | The builder does |
|---|---|
| `##` in any cell of a row | ignores the whole row |
| `##` as a column header | ignores the whole column |
| `## Description` on `unittypedata` | free text saying what the unittype is |
| a fully empty row | stops reading the sheet there; warns if rows follow |
| a column with no header, past the table | ignores that column, silently |
| a column with no header, inside the table | ignores it, and warns |
| `1,000.0`, `100 MW`, `(500)` in a parameter column | reports it and reads it as not set |
| `#REF!`, `#DIV/0!` anywhere | reports it and reads it as not set |
| the same header on two columns | reads the first, warns about the rest |
| `_` anywhere in a text cell | drops the row, with a warning |
| a column name nothing recognises | reports it and reads nothing from it |
| `all` as a scenario or country | the row applies to every one of them |
| `1` as a year | the row applies to every year |
| a blank `scenario` or `year` | reports the row and drops it |
| a blank `country`, `grid` or `unittype` | reports the row and reads nothing from it |
| a year that is not a year, `0` say | reports the row; it matches no run |
| a `unittype` no `unittypedata` declares | reports it; the unit reaches the model as nothing |
| two rows for one key in one sheet, `replace` | reports it; the first row is dead text |
| `add` or `multiply` on a key nothing established | reports it; `multiply` writes nothing |
| `remove` on a key nothing established | nothing, and no message |
| `grid_output1` on a `unitdata` sheet | discards it; `unittypedata` says which grid |
| a node in `exclude_nodes` | drops it, and every unit connected to it, whole |

## See also

- [The source data phase](source-data.md) — the same subject from the builder's side:
  what happens to a sheet after it is read, in what order, and why that order
- [District heating demand timeseries](dh-demand-timeseries.md) — what `TWh/year` in a
  demand sheet means: a weather-normalised normal year, never a realised one, and the
  same holds for the electricity demand table
- [Hydro data](hydro.md) — which file supplies which hydro number, and in what unit
- [Input Excel builder](input-excel.md) — what becomes of these sheets: which of their
  columns reach `inputData.xlsx`, and which values are deduced rather than copied
- [Identified gaps](identified-gaps.md) — Backbone parameters that have no column in
  any of these sheets yet, and the one table with no workbook of its own
- [Migration guide](Migration%20guide.md) — what to change in a workbook when an input
  format changes
- `tests/README.md` — the NA/zero boundary map, for anyone changing the pipeline
- `docs/dictionary.md` in the Backbone repository — what each parameter means
