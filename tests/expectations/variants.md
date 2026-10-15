# Variants: expected behaviour

The behaviour the variant tests pin, one row per test or per assertion. Each test
names its row (A3, E15 …), so this table is what to read when a test's intent is
unclear. What variants are, and why the rules are what they are, is in
[docs/variants.md](../../docs/variants.md).

| Group | Subject | Tests |
|---|---|---|
| A | Reading tags and declarations | `unit/source_data/test_variant_tags.py` |
| B | The config | `unit/infrastructure/test_config_variants.py` |
| C | Which rows a build keeps | `unit/source_data/test_variant_selection.py` |
| D | Checking that the chosen option has rows | `unit/source_data/test_variant_selection.py` |
| E | Merging: precedence and methods | `unit/source_data/test_variant_merge.py` |
| F | Checks | `unit/source_data/test_variant_checks.py` |
| G | Partial builds, outputs, command line | `unit/test_build_partial_banner.py` (G1), `unit/source_data/test_variant_selection.py` (G2), `unit/infrastructure/test_config_variants.py` (G3–G8), `route/test_route_variants.py` (G4, G5), `unit/test_utils.py` (G6, G8), `unit/test_run_model.py` (G9), `unit/infrastructure/test_cache_rerun_decisions.py` (G10, G11) |

Helpers and the stable message substrings are in `_common/variants.py`.

## Words

- **Base:** untagged rows, plus the `:all` rows (`bio-price:all`) of countries that have an
  option for that variant, merged in the usual order. This is pass 1 of a build.
- **Option rows:** rows tagged with one or more options (`bio-price:high`), merged on top of
  the base. This is pass 2 of a build.
- **Choice:** the option a build gives a country, from the first of these that applies:
  1. an exact country code;
  2. a two-letter key, which is shorthand for listing every code it starts;
  3. `all`, which reaches only the countries that have rows for that option;
  4. the declared default.

  When none of these gives an option, the country is **dormant** for that variant, and none
  of the variant's rows are read there.
- **Check view of an option:** the variant's `:all` rows plus that option's rows, merged with
  their methods. Untagged rows never enter it; see the last section.
- **Three `all`s, in different places:**
  - **Country `all` in a data row** is a copy: one row per code in `country_codes`, made
    before node names are built.
  - **`:all` in a variant tag** is a condition: the row is read where the country has an
    option.
  - **`all` as a config key** reaches the countries with rows for the chosen option.

  An empty cell never means all.

**Fixture countries** used in the examples: FI00, SE01, SE02, EE00.

## A. Reading tags and declarations (Excel)

| # | Given | Expected |
|---|---|---|
| A1 | A sheet has no `variant` column | Every row is untagged |
| A2 | Variant cell empty | Untagged: in no variant, so read in every build and never judged |
| A3 | `bio-price:all` | In every option: read wherever the country has an option for this variant, chosen or default. Never where it is dormant |
| A4 | `dh-level:grouped,cities` | In those two options only |
| A5 | `Bio-Price:HIGH`, or spaces around `:` and `,` | Same as `bio-price:high`: Excel cells are matched case-insensitively and trimmed |
| A6 | `bio-price` (no options), `bio-price:`, `bio-price:all,high`, `:high`, `a:b:c`, `bio price:high`, `bio_price:high` | Warning naming sheet and row; the row is dropped. Every option is written `:all`, never left empty. `_` gets this message rather than a silent drop by the underscore rule |
| A7 | `dh-level:cities; fuel:high` | As A6: one variant per cell |
| A8 | A tag its workbook's `variantdata` sheet doesn't declare (typo in variant or option) | Warning naming sheet, row and tag; the row is dropped, never read as untagged |
| A9 | Tagged rows in a workbook with no `variantdata` sheet | As A8, for every tagged row |
| A10 | The same variant declared in two workbooks with different options | Options are pooled |
| A11 | The same variant with a different `default` or `check` in two workbooks | The first encountered wins (config file order, then tab order); loud warning naming both workbooks; the build is partial |
| A12 | Two `yes` defaults for one variant | As A11: the first encountered is the default |
| A13 | `check` not one of `totals`, `coverage`, `none` (e.g. `totalss`) | Warning; that declaration row is dropped. Rows tagged with what it declared are then undeclared (A8) |
| A14 | A declared non-default option with no rows in any workbook | Warning. A build choosing it gets nothing from it (D2, C19) |
| A15 | A declared default with no rows | Silent: a default may be empty |
| A16 | Malformed tags in rows of other scenarios or years | Reported too: form is judged before relevance |
| A17 | `all` as a variant or option name in a `variantdata` sheet | Warning; that declaration row is dropped, and rows tagged with it are then undeclared (A8). `all` is reserved for "every option" |

## B. The config

| # | Given | Expected |
|---|---|---|
| B1 | No `variants` key | One build `''` with every default; the folder has no build suffix |
| B2 | `'': {}` | A build with no folder suffix |
| B3 | Build names `sens1` and `Sens1` | Raise: build names are unique, case-insensitively |
| B4 | Build name with a space, `/`, `:` or `_` | Raise: it can't be a folder segment, and `_` already separates the folder name's parts |
| B5 | `{'bio-price': 'high'}` | Same as `{'bio-price': {'all': 'high'}}`: reaches the countries with `high` rows (C16) |
| B6 | `{'bio-price': {'FI00': 'high'}}` | FI00 `high`; the rest the default |
| B7 | `{'bio-price': {'SE': 'high'}}` | SE01 and SE02 `high`. Shorthand for listing each code, so each one is checked as an exact code (D7) |
| B8 | `{'bio-price': {'all': 'low', 'SE': 'high', 'SE02': 'ref'}}` | SE01 `high`, SE02 `ref`; `low` for the others that have `low` rows: the most specific key wins |
| B9 | Key `XX`, or `DE00` not in `country_codes` | Raise, naming the key |
| B10 | Any of the keys `scenario_alternatives` … `scenario_alternatives4` present | Raise, pointing to `variants` |
| B11 | `variants` not a dict of dicts whose values are strings or dicts | Raise, showing the expected shape |
| B12 | A name with spaces, characters outside letters, digits and `-`, or an empty string as a variant or option. Also `all` as a variant name or as an option value | Raise. No trimming in the config, unlike Excel (A5). No `_`: the source tables drop text values containing it. `all` is reserved; as a country key it means every country |
| B13 | A variant or option name that no workbook declares | Not raised, because the config is read before the data; handled in D4 and D5 |
| B14 | Two scenarios × one year × three `variants` entries | Six builds, six folders |

## C. Which rows a build keeps

| # | Given | Expected |
|---|---|---|
| C1 | Untagged rows; `:all` rows | Untagged: every build, every country. `:all`: every country with an option for the variant, which with a default is every country |
| C2 | `bio-price:high`, country `all`; build `{'FI00': 'high'}` | Copied to every country, kept for FI00 only |
| C3 | `dh-level:grouped,cities` row in FI00; FI00 chooses `cities` | Kept |
| C4 | Tagged rows of another scenario or year | Gone before anything else, filtered like any row |
| C5 | Tagged row on an excluded node or grid | Gone, as an untagged row would be |
| C6 | Tagged rows in unittypedata or emissiondata (no country) | Take the `all` choice, or the default |
| C7 | As C6, with a per-country choice for that variant | Those rows take `all` or the default; warning; partial. Country-less data belongs in its own variant |
| C8 | Transfer FI00–SE01 tagged `ext`; build `{FI: ext, SE: ext}` | Kept: both ends chose `ext` |
| C9 | Transfer FI00–EE00 tagged `ext`; same build | Dropped; that link takes the default's rows |
| C10 | Transfer FI00–EE00; build `{EE: ext, FI: ext}` | Kept |
| C11 | A transfer whose ends disagree, in a variant with no default | The link has no option: its tagged rows, `:all` included, are dropped; warning naming the link; partial. Both ends unnamed is dormant: nothing, silently |
| C12 | Transfer written SE01→FI00 instead of FI00→SE01 | Same result: a link's option depends on its two ends, not their order |
| C13 | userconstraintdata rows with a country | Per-country choice, as nodedata |
| C14 | A country `all` row tagged `new-units:all`; build `{'new-units': {'FI00': 'smr'}}` | Copied to every code before node names are built; kept for FI00 only. `smr`'s units and nodes exist in FI00 and nowhere else |
| C15 | Transfers, `{'ic': 'ext'}` | The countries with `ext` rows at either end reach `ext`. A link takes `ext` only where it has `ext` rows and both ends chose `ext`. Links outside the variant are untouched, no link is created, and nothing warns about pairs without rows. To extend only some links within a group of countries, use one variant per corridor |
| C16 | `{'new-units': {'all': 'smr'}}`, with `smr` rows for FI00 and SE01 | `smr` in FI00 and SE01 only. SE02 and EE00 are dormant, with no warning. `all` never means every code in `country_codes` |
| C17 | `{'bio-price': 'high'}`, with `high` rows only for FI00 and `:all` rows everywhere | FI00 `high`; the others take the default `ref`. No warning, not partial |
| C18 | Any `all` key | An info line names the variant, the option, the countries it reached, and those that took the default or stayed dormant |
| C19 | `all` reaching no country (the option has no rows) | Warning, besides A14's; partial |
| C20 | An `smr` row written with country `all` | Defines `smr` for every code, so `{'all': 'smr'}` reaches every code: "has rows" is judged after country `all` is copied |
| C21 | `cities` rows only in demanddata; nodedata has only a `national` row; FI00 chooses `cities` | No warning: rows in any table and any workbook are evidence. Every table uses the same choice, so nodedata drops its `national` row |

## D. Checking that the chosen option has rows

| # | Given | Expected |
|---|---|---|
| D1 | FI00 chooses `cities`; FI00 has rows tagged `cities` | Fine. "Has rows" is judged per exact country code, in any table. A transfer row counts for both its ends; `:all` rows never count |
| D2 | FI00 chooses `tallinn`; only EE00 has `tallinn` rows; FI00 has `dh-level:all` rows | Warning (a clear typo); FI00 falls back to the default `national`; partial. Finnish district heating is not empty. This applies to exact codes and two-letter keys, never to countries `all` reached (C16, C17) |
| D3 | FI00 chooses the default, which has no rows of its own | Fine: a default may be empty |
| D4 | A chosen option no workbook declares | That country takes the default; warning; partial |
| D5 | A variant the config names that no workbook declares | Ignored; warning; partial. Not an abort: one build should report every mistake at once |
| D6 | A variant with no default, not chosen for a country (e.g. three alternative new units, where none is right by default) | **Dormant** there: none of its rows, `:all` included. No warning, not partial. `summary.log` lists all dormant variants on one line |
| D7 | `{'new-units': {'SE': 'smr'}}`, with `smr` rows only for SE01 | SE01 `smr`. SE02 warns (D2) and, with no default, stays dormant |

## E. Merging: precedence and methods

| # | Given | Expected |
|---|---|---|
| E1 | Untagged price 40 in a *later* file; `bio-price:high` replace 55 in an earlier one; FI00 `high` | FI00 55: option rows merge in pass 2, after the whole base |
| E2 | `bio-price:all` 40; `bio-price:high` replace 55 | `high` 55; `ref` and `low` (no rows of their own) 40 |
| E3 | `bio-price:all` 40; `bio-price:high` multiply 1.3 | `high` 52, others 40 |
| E4 | As E3, country `all` | Every country choosing `high` gets 1.3 × its own base |
| E5 | `bio-price:high` multiply 1.3, with no base row for that key | The usual unmatched-arithmetic message; nothing is created |
| E6 | Two `bio-price:high` multiply rows, 1.3 and 1.1 | 40 × 1.43, in either order |
| E7 | `bio-price:high` multiply 1.3, then a later `bio-price:high` replace 60 | 60: file order inside pass 2 |
| E8 | `demand:high` add +0.5 TWh on a base demand of 7 | 7.5 |
| E9 | As E8, with no base row | The usual unmatched-arithmetic message; the record is created as 0.5 |
| E10 | `demand:low` add-non-negative −10 on 7 | 0 |
| E11 | `nuclear:without` remove on a base unit | The unit is gone in `without`, present in other options |
| E12 | `nuclear:without` remove on a key nothing established | Silent, as any `remove` matching nothing |
| E13 | `bio-price:high` replace-partial filling only `price`, on a base row with `price` and `emission_CO2` | `price` replaced, `emission_CO2` kept |
| E14 | An `:all` row with method multiply | Applied in pass 1, wherever the variant has an option, like an untagged row |
| E15 | `bio-price:low` and `bio-price:high` replace rows for one key in one sheet | No "written twice" message: they never merge together |
| E16 | An `:all` row and a `bio-price:high` replace row for one key in one sheet | No "written twice" message: different passes |
| E17 | `dh-level:grouped,cities` and `dh-level:cities` replace rows for one key in one sheet | "Written twice" in builds choosing `cities`: one of them is dead text there |
| E18 | Two untagged replace rows for one key in one sheet | The usual "written twice" message |

## F. Checks

| # | Given | Expected |
|---|---|---|
| F1 | Any build | Every option of every declared variant is checked, not only the chosen ones: a broken `grouped` shows in a `national` build |
| F2 | What the checks merge | Each option's **check view**: `:all` rows plus that option's rows, with add, multiply, replace-partial and remove applied. Untagged rows never enter |
| F3 | `totals`, unitdata | Per country: the default and every option that has rows there (Finland is not compared against Estonia's `tallinn`). Capacity summed per grid, inputs and outputs separately, equal within 0.1 % |
| F4 | `totals`, demanddata | Per country: annual demand summed per grid, equal across options within 0.1 % |
| F5 | `totals`: `national` FI00 heat demand 10 TWh/year; `cities` rows sum to 9 (one city's row missing) | Warning naming FI00, dheat, both options and both sums |
| F6 | `totals` in a country where every option is identical | Silent |
| F7 | `coverage` scope | Every key of every row tagged with the variant (`:all` or any option). `:all` rows are the same for every option, and the rest must match |
| F8 | `coverage`: `bio-price:all` biomass 40, `high` multiply 1.3, `low` replace 30, `ref` no rows of its own | Silent |
| F9 | `coverage`: `high` replaces a gas key, with no gas row anywhere else | Warning: gas missing from `low` and `ref` |
| F10 | As F9, but an untagged gas row exists | Same warning: untagged rows are not checked and don't cover anything |
| F11 | `coverage`: `high` multiplies an **untagged** base price | Warning: the key is missing in every option, since a multiply establishes nothing in the check view. Tag the base row `bio-price:all` to fix it |
| F12 | `coverage`: `low` removes a key that `:all` rows established | Silent: an explicit `remove` covers its key. Only a key an option says nothing about is missing |
| F13 | `totals` with a `remove` in one option | The removal counts: the sums differ and the warning names it |
| F14 | `none` | Never a check message. `summary.log` gives each option's rows in this run and the workbooks they come from |
| F15 | Check messages | Name the first three offenders and count the rest (`utils.summarise`) |
| F16 | Merges done only for checking | Silent; merge messages come from the build's own merge, once |
| F17 | A failed check | Reported in every build; every build is flagged partial, naming the variant |

## G. Partial builds, outputs, command line

| # | Given | Expected |
|---|---|---|
| G1 | A partial build | First lines of `summary.log`: "PARTIAL", each cause, and that later warnings may follow from them. The closing status line says PARTIAL |
| G2 | Any build | `summary.log` lists the resolved choice per variant and country |
| G3 | Folder | `<prefix>_<scenario>_<year>_<build>`; `''` gives no suffix |
| G4 | `add_scen_tags` cells (0, 0), (0, 1), (0, 2) | `scenario`, `year`, `variant`: the build name, empty for `''`. No `alternative*` columns |
| G5 | `add_scen_tags` cell (0, 3) | `name`: the folder name without its prefix, e.g. `ObservedTrends_2030_sens1`, or `ObservedTrends_2030` for `''`. A distinct Spine alternative per build. Readers of cells (0, 0) and (0, 1) are unaffected |
| G6 | `build_input_data.py ... --variant sens1` | Builds only `sens1`, across every scenario and year |
| G7 | `--variant nope` | Raise, listing the build names |
| G8 | No `--variant` | Every build |
| G9 | `run_model.py` | The argument is the built input folder (`run_model.py input_ObservedTrends_2030_sens1 --year 2015 --days 7`): a name in the project, or a path inside the Backbone checkout. A folder without `inputData.xlsx`, or outside the checkout, is refused. A scenario name is answered with every folder its config builds. `--list` lists built input folders, whether a run holds each, and which config and build wrote it. The default results tag is `<folder>-<year>-<days>d` |
| G10 | A build's choice changes, same build name | Full rebuild of that build |
| G11 | Another build's choice changes | This build's folder and cache are untouched |

## How the checks see the data

A check builds each option from the variant's **own** rows only: the `:all` rows plus that
option's rows, merged with their methods. Untagged rows never enter a check, so variant rows
can sit inside large sheets of ordinary rows without those rows being judged. That leaves a
typo in an untagged row silent, which is accepted, because the two checks catch most typos in
tagged rows.

Three consequences to keep in mind when writing tests:

- **A patch variant tags the row it patches** (F11). `bio-price:high` multiplying an untagged
  price warns. Tagging the base row `bio-price:all` makes it part of the variant, and the
  check passes.
- **`remove` is a statement about a key** (F12). It counts as coverage, because removing a
  unit is a legitimate scenario. It still changes `totals` (F13).
- **Builds and checks differ in what they merge.** A build merges untagged rows too (E1, E2);
  a check doesn't. The same row can therefore be fine in the build and a coverage warning in
  the check (F11), and that is intended.
