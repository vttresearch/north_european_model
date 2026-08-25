# Source workbook conventions

How the input builder reads the Excel files in `src_files/data_files/`, and the
conventions a workbook author needs to know.

Everything here applies to the **source workbooks** — the files listed in a
`config_*.ini` under `unitdata_files`, `nodedata_files` and the rest. It does not
apply to `inputData.xlsx`, which the builder writes.

Read [Writing a sheet](#writing-a-sheet) once. The rest is reference for when
something in a build log does not look right.

## Quick reference

| You write | The builder does |
|---|---|
| `##` in any cell of a row | ignores the whole row |
| `##` as a column header | ignores the whole column |
| a fully empty row | stops reading the sheet there; warns if rows follow |
| a column with no header, past the table | ignores that column, silently |
| a column with no header, inside the table | ignores it, and warns |
| `1,000.0`, `100 MW`, `(500)` in a parameter column | reports it and reads it as not set |
| `#REF!`, `#DIV/0!` anywhere | reports it and reads it as not set |
| the same header on two columns | reads the first, warns about the rest |
| `_` anywhere in a text cell | drops the row, with a warning |

---

## Writing a sheet

### Which sheets are read

A config file lists workbooks per data type. From `config_OT2030.ini`:

```ini
unitdata_files = ['ObservedTrends.xlsx',
                  'unitdata_chp-units.xlsx',
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
`demanddata`, `emissiondata` and `userconstraintdata`. A sheet whose name matches
no prefix is never opened, so a scratch sheet only needs a name that does not
start with one of them.

The order of the file list matters — see [Combining rows](#combining-rows-the-method-column).

### What a row says: dimensions and parameters

Every sheet has two kinds of column. `unitdata_VRE` in `ObservedTrends.xlsx`:

| Country | Generator_ID | Scenario | Year | capacity_output1 | vomCosts | method |
|---|---|---|---|---|---|---|
| AT00 | Solar PV | Observed Trends | 2015 | 937 | 0.56 | |
| BE00 | Solar PV | Observed Trends | 2015 | 3132 | 0.56 | |

**Dimension columns** say *which thing* the row is about: `Country`,
`Generator_ID`, `Scenario`, `Year`. They hold labels, and together they are the
key the builder uses to recognise that two rows describe the same unit.

**Parameter columns** hold the numbers: `capacity_output1`, `vomCosts`. Their
names are Backbone parameter names — the full list is `docs/dictionary.md` in the
Backbone repository. A cell in one of these has to be a number; see
[Cells that should be numbers](#cells-that-should-be-numbers) for what happens
when it is not.

`method` is neither; it is an instruction, described next.

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
reported and the column ignored.

### Combining rows: the `method` column

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

For `add` and `multiply`, a missing value is not the same as zero: adding to a
missing value treats the missing one as `0.0`, multiplying by a missing new value
leaves the old one unchanged, and missing on both sides stays missing.

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

## When something looks wrong

### Cells that should be numbers

A parameter column has to hold numbers. The builder does not try to interpret one
that does not:

`1,000.0` · `1 000` · `1'000` · `12,345,678` · `1.000,5` · `1_000` · `100 MW` ·
`100MW` · `5%` · `€100` · `(500)` · `−5`

Each is reported — naming the file, sheet, column and value — and read as **not
set**, which the model treats as zero.

None of them is repaired, and that is deliberate. `1.000` is a thousand to an
author writing in one locale and one to an author writing in another, and the cell
carries nothing that says which. A builder that guessed would put a confidently
wrong number into the model, where nothing would ever reveal it. Blanking the cell
and naming it means you fix it once, at the source.

The report is at **error** level: the build still finishes and still writes its
output, but it is marked as failed, it re-runs from scratch next time, and the
message is repeated in the summary at the end. Look in `summary.log` in the output
folder.

**Rows this run does not use are checked too.** A row for another scenario, year
or country is reported before it is filtered out, so one build tells you about
every bad cell in the workbook rather than only the ones this scenario happens to
touch.

Text that never looked like a number — `unknown` in a capacity column — is not
caught. Nothing in the cell distinguishes it from a label, and a rule aggressive
enough to catch it would eat identifiers like `chp1`.

### Excel error values

`#REF!`, `#N/A`, `#DIV/0!`, `#VALUE!`, `#NAME?`, `#NUM!` and their relatives are
reported wherever they appear, in dimension columns as much as parameter columns,
and read as not set. None is ever a value anyone meant to write. `#REF!` in
particular is what Excel leaves behind when a column another sheet pointed at is
deleted, so it usually means the workbook has quietly lost a reference.

### A header used twice

Excel lets two columns carry the same header. The builder reads the first and
warns about the rest, naming the file, the sheet, the header and how many values
are being passed over.

It does not merge them, deliberately: nothing in the sheet says which value
should win or how two of them should combine, and a wrong answer chosen
automatically is worse than a question asked out loud. Give the second column its
own name, or mark it `##` if it is working material.

One pair looks like a duplicate and is reported differently. A bare parameter
name means `_output1`, so `vomCosts` and `vomCosts_output1` in the same sheet are
two spellings of the same column. That is reported as a rename collision, and the
suffixed column is left as it is rather than one silently overwriting the other.
Use one spelling or the other.

### Identifiers

**Underscore is the node-name separator.** Node names are built as
`{country}_{grid}`, or `{country}_{grid}_{node_suffix}` when a suffix is given, so
an `_` inside a text cell would produce a name nobody can take apart again. Any
row containing one is dropped, with a warning naming the column and showing
examples.

**Case is folded, first spelling wins.** `scenario`, `generator_id` and `method`
values are lower-cased, as are column names. Rows match case-insensitively while
the spelling used first is what reaches the output. GAMS treats `dh` and `DH` as
one set element and refuses a GDX containing both, so this is not a nicety.

**A mistyped suffix makes a new node.** `node_suffix` and `unit_name_prefix` are
part of the merge key and are built into the node and unit name. A typo does not
fail to override an earlier row — it quietly creates a second node, and the value
you meant to change stays as it was. Nothing can detect this, because an intended
new node looks exactly the same.

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

### Why the marker is two hashes

A single `#` used to mark a comment row, and it was the wrong character. Every
Excel error value starts with one — `#REF!`, `#N/A`, `#DIV/0!` — so a formula that
broke silently deleted the row it sat in, and a broken reference could remove a
power plant from the model without a word.

A single `#` no longer marks anything. No Excel error value begins with `##`, and
unlike `=`, `+`, `-` and `@`, a hash does not make Excel treat the cell as a
formula.

### `note` columns

A column named exactly `note` is dropped. This predates `##` and is kept because
shipped workbooks still use it. Prefer `##`: it is not limited to one name or one
column per sheet, and a second `note` column arrives as `note.1`, which the old
rule never matched.

## See also

- [District heating demand timeseries](dh-demand-timeseries.md) — what `TWh/year` in a
  demand sheet means: a weather-normalised normal year, never a realised one, and the
  same holds for the electricity demand table
- [Hydro data](hydro.md) — which file supplies which hydro number, and in what unit
- `tests/README.md` — the NA/zero boundary map, for anyone changing the pipeline
- `docs/dictionary.md` in the Backbone repository — what each parameter means
