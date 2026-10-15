# Variants

A source workbook can hold several versions of the same data side by side — district
heating as one national node, as grouped cities, or as every city; fuel prices at a
reference and a high level — and each build chooses one version per country. The
versions live in the data, where they can be checked against each other; the config
only says which one a build takes.

## One minute summary

- **Why.** More detail costs solve time, so the data can carry every level of detail
  while a build picks the one it can afford, per country.
- **A variant is a family, an option one member of it.** `dh-level` is a variant;
  `national`, `grouped` and `cities` are its options.
- **A workbook declares its variants** on a `variantdata` sheet: the options, which one
  is the default, and how they are checked against each other.
- **A row joins an option** through a `variant` column in any data sheet. An empty cell
  is in no variant and is read in every build, as before.
- **A config names builds** in its `variants` key. Each build chooses options per country
  and becomes one output folder; anything it leaves out takes the default.
- **The chosen option overrides the base whatever the file order**, because the options'
  rows merge after everything else.
- **Mistakes warn, and the build carries on with the default.** Such a build is marked
  PARTIAL at the top of its `summary.log`.

| Where | What you write | Example |
|---|---|---|
| a data sheet's `variant` column | which option the row belongs to | `dh-level:cities` |
| the workbook's `variantdata` sheet | the variant, its options, default and check | `dh-level`, `national`, `yes`, `totals` |
| `variants` in `src_files/config_*.ini` | builds, each choosing per country | `{'': {}, 'fi-cities': {'dh-level': {'FI00': 'cities'}}}` |
| `build_input_data.py ... --variant <build>` | build one build only | `--variant fi-cities` |

## Contents

- [Tagging a row](#tagging-a-row)
- [Declaring a variant](#declaring-a-variant)
- [Choosing in the config](#choosing-in-the-config)
- [Which rows a build reads](#which-rows-a-build-reads)
- [Base first, options after](#base-first-options-after)
- [The checks](#the-checks)
- [What a build says](#what-a-build-says)
- [Folders and `add_scen_tags`](#folders-and-add_scen_tags)
- [Where variants are defined](#where-variants-are-defined)
- [See also](#see-also)

## Tagging a row

Any data sheet may carry a `variant` column. What a cell in it says:

| `variant` cell | The row is |
|---|---|
| empty | in no variant: read in every build |
| `dh-level:cities` | in the option `cities` only |
| `dh-level:grouped,cities` | in both of those options |
| `dh-level:all` | in every option of `dh-level` — part of the variant, whichever option is chosen |

Empty options as in `dh-level:` is not allowed. Every options is written e.g. `dh-level:all`, similarly as `all` means every
scenario or every country elsewhere in a workbook. 

Names are letters, digits and `-`. No `_`, which the builder reads as a node-name
separator, and `all` is reserved. Case and the spaces around `:` and `,`
do not matter.

`:all` rows are what make a row's membership visible. For an example a price that every option builds on
is tagged `bio-price:all`.

## Declaring a variant

A workbook may use only the variants and options declared on its own `variantdata` sheet.
A typo in a tag is then caught against that sheet, and the row is dropped rather than read
as untagged:

| variant | option | default | check | description |
|---|---|---|---|---|
| dh-level | national | yes | totals | one heat node per country |
| dh-level | grouped | | totals | capital region, large cities, rest |
| dh-level | cities | | totals | every city its own node |

- **`default`** is `yes` on the option a build gets when it says nothing, and empty on the
  others.
- **A variant may have no default** — three alternative new units, say, none of them right
  unless a build asks for one. Such a variant is **dormant** wherever no build chooses it:
  none of its rows are read, `:all` rows included.
- **`check`** is `totals`, `coverage` or `none`; see [The checks](#the-checks).
- **Several workbooks may declare one variant**, as when district heating spans a units
  workbook and a demand workbook. Their options are pooled. The default and the check must
  agree; where they do not, the first workbook in the config's order wins and the build is
  marked PARTIAL.

## Choosing in the config

`variants` in `src_files/config_*.ini` names the builds. Each key is a build, and each
build is one output folder:

```ini
variants = {'':          {},
            'fi-cities': {'dh-level': {'FI00': 'cities'}},
            'bio-high':  {'bio-price': 'high'}
            }
```

A config without `variants` has one build, `''`, which takes every default.

Per variant, a build gives either one option for every country (`'high'`), or a dict keyed
by country. Where several keys could apply, the most specific wins:

1. an exact country code, `FI00`;
2. a two-letter prefix, `SE`, which is shorthand for writing every Swedish code out;
3. `all`, which reaches **only the countries that have rows for that option** — never
   every code in `country_codes`;
4. otherwise the declared default, or dormant where the variant has none.

`'high'` alone means `{'all': 'high'}`: countries without `high` rows quietly keep the
default, and `summary.log` says which countries the `all` reached.

A country named by code or prefix must have rows of its own for the option, in any table.
If it has none — usually a typo such as `FI00: tallinn` — the build warns and that country
gets the default. `:all` rows never count here, or that typo would pass on Finland's
`dh-level:all` rows and leave Finland with no district heating at all.

The config is checked before anything is read: a build name that cannot be a folder name,
a key that matches no code in `country_codes`, a malformed name, or a leftover
`scenario_alternatives` key stops the build with a message saying what to write instead.

## Which rows a build reads

- **A country follows one choice in every table.** `cities` rows in the demand data alone
  are enough for Finland to choose `cities`, and the node data then drops its `national`
  row too. This is why every country table is read before any is merged.
- **A row written for `country = all` is copied to every code first**, so an option
  written once for `all` has rows in every country.
- **A link** in `transferdata` takes an option only when both its ends chose it. Ends that
  disagree take the default. With no default, the link's rows in that variant are not
  read, and the build warns. A variant never creates a link: to extend only some of the
  links between three countries, give each corridor a variant of its own.
- **`unittypedata` and `emissiondata` have no country**, so their tagged rows take the
  build's `all` choice, or the default. Choosing such a variant per country is warned
  about: data without a country belongs in a variant of its own.
- **Untagged rows are read in every build** and never judged. The builder cannot know
  whether an untagged row "should have been" in a variant: if an untagged row and an option
  describe the same plant under two different names, the build has it twice. That is the
  workbook author's to avoid.

## Base first, options after

A build merges in two passes. First the untagged and `:all` rows, in the usual file, tab
and row order; then the chosen options' rows, in the same order among themselves. An
option therefore overrides the base whatever file it sits in, and every `method` works on
top of the base:

| Base row | Option row in `high` | `high` gives |
|---|---|---|
| `bio-price:all`, price 40 | `replace`, 55 | 55 |
| `bio-price:all`, price 40 | `multiply`, 1.3 | 52 |
| `demand:all`, 7 TWh | `add`, 0.5 | 7.5 |
| `nuclear:all`, a unit | `remove` | no unit |

Every other build gets 40, 7 TWh and the unit.

The "written twice in one sheet" warning knows the passes: an option row replacing its own
sheet's `:all` row is what an option is for, and two different options never meet in one
build. Two rows that both reach the chosen option are still dead text, and still reported.

## The checks

A variant's `check` compares its options with each other, in every build, whichever option
the build chose — so a broken `grouped` shows in a `national` build. A check sees each
option as the variant's `:all` rows plus that option's rows, merged with their methods.
**Untagged rows never enter a check**: they are never judged, and they never make up for
anything an option lacks.

- **`totals`**, for aggregation such as district heating by city. In each country, the
  default and every option that has rows there must add up to the same annual demand per
  grid, and the same unit capacity per grid, in and out, within 0.1 %. Its job is to catch
  a missing or doubled row. A country is only compared against the options it has, so
  Finland's options are never compared with an Estonia-only one.
- **`coverage`**, for the same things at different values, such as fuel prices. Every key
  the variant touches — a node, a unit, a link — must be resolved by every option, from its
  own rows or from `:all` rows. An explicit `remove` counts as resolving it. A `multiply` on
  an untagged price resolves nothing, so tag the base price `bio-price:all`.
- **`none`**, for scenario-like variants such as with or without new reactors. Nothing is
  compared; `summary.log` gives each option's rows and the workbooks they come from.

A failed check warns, naming the first few places and counting the rest, and marks every
build of that config PARTIAL.

## What a build says

- **The choice.** One line per variant in `summary.log`: which option each country got.
  One more line names the dormant variants, if any.
- **What `all` reached**, whenever a build uses an `all` key.
- **Mistakes in the data**, each naming the workbook, sheet and row: a tag that cannot be
  read or is not declared (the row is dropped), a declaration that is malformed or uses
  `all` (dropped), two workbooks declaring one variant differently.
- **Fallbacks**, each naming the variant and the countries: an option a country has no
  rows for, an option or variant no workbook declares, link ends that disagree with no
  default, an `all` that reached nothing.
- **PARTIAL.** A build with a fallback or a failed check opens `summary.log` with a banner
  listing every cause, and says PARTIAL on its closing line. Later warnings may follow from
  those, so fix the banner's first. A rerun that finds nothing changed keeps the banner.

## Folders and `add_scen_tags`

A build writes `<output_folder_prefix>_<scenario>_<year>_<build>`, so the build
`fi-cities` of `config_OT2030.ini` writes `input_ObservedTrends_2030_fi-cities`. The build
`''` adds no suffix. Each folder has its own cache, and changing one build's choice
rebuilds that build and no other.

`inputData.xlsx` names its build in the `add_scen_tags` sheet, one row: `scenario`, `year`,
`variant` (the build's name, empty for `''`) and `name` (`ObservedTrends_2030_fi-cities`,
different for every build). Backbone does not read it; Spine Toolbox workflows do, by cell
position, so new cells go on the end of the row.

## Where variants are defined

- `src/source_data/source_data_variants.py` — declarations, tags, choices, checks and the
  row selection
- `src/source_data/source_data_pipeline.py` — the order: declarations, tables without a
  country, every country table prepared, choices, checks, then the merges
- `src/infrastructure/config_reader.py` — `parse_variants`, the folder names and the
  `add_scen_tags` row
- `build_input_data.py` — one pass per scenario, year and build, `--variant`, the PARTIAL
  banner
- `tests/expectations/variants.md` — the behaviour the tests pin, row by row

## See also

- [Source workbook conventions](source-workbook-conventions.md) — writing a sheet, and the
  `variant` column among the other columns every sheet may carry
- [The source data phase](source-data.md) — the order the builder reads and merges in, and
  why
- [Input Excel builder](input-excel.md) — what becomes of the merged tables
