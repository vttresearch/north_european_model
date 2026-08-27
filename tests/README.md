# Test suite

## Running

```
conda run -n northEuropeanModel python -m pytest              # everything
conda run -n northEuropeanModel python -m pytest tests/unit   # fast tier
conda run -n northEuropeanModel python -m pytest -m contract  # the dtype/NA sweeps alone
conda run -n northEuropeanModel python -m pytest -k merge_row_by_row
```

Use `python -m pytest`, not bare `pytest`, so the repo root lands on `sys.path`.

The run header prints `gams.transfer: real API` or `gams.transfer: STUBBED`. Check it.
A green run on a machine without GAMS has skipped every `@pytest.mark.gams` test,
and the header is the only thing that says so.

## Layout

```
tests/
  _common/     shared infrastructure. Leading underscore + no test_*.py => never collected
  meta/        tests for the test infrastructure itself
  unit/        mirrors src/, one module per source module
  route/       full source-xlsx -> inputData.xlsx
  _common/workbooks/   the .wb.txt fixture library
```

Which folder does a new test go in?

| Question it answers | Folder |
|---|---|
| Does one function behave correctly? | `unit/` + the mirror of its `src/` path |
| Does source data reach `inputData.xlsx` correctly? | `route/` |
| Does a test helper work? | `meta/` |

Helpers are **functions imported at the call site**, not fixtures. Arguments are then
visible in the test that uses them, and functions compose where fixtures do not.
`conftest.py` holds only `repo_root`, `src_files_dir` and `fake_logger`.

---

## The NA / zero boundary map

GAMS has no NaN, and plain `0` **is** empty — efficient, and hard to hold in your head
consistently. Python is precise about types; GAMS is not. Nearly every bug in this
codebase's history lives at the seam. This table is what the suite is organised around.

| # | Boundary | Left of it | Right of it | Conversion sits in | Status |
|---|---|---|---|---|---|
| 1 | Excel cell → source DataFrame | blank cell | `pd.NA`; all-NA column → `object` | `read_input_excels` → `normalize_dataframe` → `standardize_df_dtypes` | specified |
| 2 | within source merging | `pd.NA` ≠ `0` | same | `merge_row_by_row` truth table (`:884-909`) | specified |
| 3 | source DataFrames → BB Excel builder | `pd.NA` ≠ `0` | `0 = NA = None = not set` | `fill_all_na` / `fill_numeric_na` (`utils.py:107,119`) | specified |
| 4 | BB builder → `inputData.xlsx` | `0 = empty` | GAMS reads it | `is_col_empty` drops all-zero columns (`bb_excel_pipeline.py:332`) | specified |
| 4b | BB frame → fake-MultiIndex sheet | dtype is meaningful | every parameter column is `object` | `create_fake_MultiIndex` inserts a text header row | specified |
| 5 | processor `process()` → `main_result` | NaN = no data | NaN = no data | validation in `ProcessorRunner` | specified |
| 6 | `main_result` → climate windows → forecasts | NaN = no data | NaN = no data | nothing — gaps pass through | specified |
| 7 | window DataFrame → GDX | NaN = no data | GAMS: `0 = empty` | `GDX_exchange.prepare_values_for_gdx` | specified |
| 8 | processor `frames` → source `df_*` | `pd.NA` ≠ `0` | `pd.NA` ≠ `0`; all-NA column → `object` | `merge_contribution` (`source_data_contributions.py`) | specified |

**`0 = NA = None` governs written GDX files too, not only `inputData.xlsx`.** The GAMS
convention begins at boundary 7 and nowhere earlier.

### The time axis is part of boundary 5

A missing **value** is a gap and is legal all the way to boundary 7. A missing **row** is
not the same thing at all, and is rejected at boundary 5. `split_timeseries_to_climate_windows`
labels by row position, so an absent row does not leave a hole in the labels — it pulls every
later hour of that group one label earlier, for the rest of the window, and nothing downstream
can tell. `ProcessorRunner` therefore proves, per parameter, that every group is one complete
hourly grid and that all groups cover the same span
(`timeseries_helpers.find_time_axis_defects`). Holes, repeats, sub-hourly rows and ragged
spans are all errors with no config escape hatch.

This subsumed a `duplicated()` check and is strictly stronger — `00:00` and `00:15` are
distinct timestamps and survived that one. `assert_processor_conforms` (with
`check_coverage=True` by default) is the author-facing mirror and delegates to the same
function, so the two cannot drift apart.

Cost, because an earlier version of this document argued the opposite from an unmeasured
figure: 66 ms on a 8.9 M-row parameter, against 1.8 s for the `duplicated()` it replaced. It
is nearly free because it reuses the ordering the labeller needs anyway. **Profile before
concluding something is too expensive to do properly.**

Boundary 7 is the **only** place NaN becomes 0 on the timeseries route. Do not add a
`fillna(0)` upstream of it. Three used to exist — in
`ProcessorRunner` right after validation, as a side effect of `cutoff_below`, and inside
`calculate_climatological_forecasts` — and each one silently made a gap in the source data
indistinguishable from a genuine zero. Filling before the quantile step was the worst of the
three: pandas' `quantile` skips NaN, so a missing hour counted as a real zero and dragged the
whole climatology down.

The conversion is **silent** during a normal build. A gap in a source timeseries is not
something the person running a build can act on, and warning about it every time trains people
to skim past the warnings their own inputs *do* cause. `prepare_values_for_gdx` counts the
converted entries and reports them only under `report_missing=True`, which the timeseries data
verifier turns on. The audience for that number is whoever writes or checks a processor.

Dimension values are different from measurements: a missing one is a **broken key**, not a gap.
It would become the GAMS set element `''`, so it is rejected rather than filled — and that *is*
reported, because it means something upstream is broken rather than merely incomplete.

### Boundary 8: what a processor may add to a source table

A processor returns `main_result` plus **contributions** — frames named after the
source data tables — and they are merged into those tables once the timeseries phase
is over, so `BBExcelPipeline` reads one set of tables and never asks which stage
produced a row.

This used to be three routes with three cache formats, and it was unspecified because
nothing said what could travel them: a `secondary_result` was whatever a processor
felt like returning, re-keyed by a config field and picked back up by a string prefix
match. What is specified now:

- **the same conventions on both sides.** `pd.NA` ≠ `0` — this is the *source* side of
  the pipeline, not the GAMS side — and an all-NA column is `object`.
- **the workbook wins.** A contribution fills only where the source frame said nothing,
  the same precedence `merge_unittypedata_into_unitdata` uses. That is what lets a
  workbook override a processor rather than the other way round.
- **nothing else moves.** A column no contribution mentions keeps the exact dtype the
  source stage gave it, and existing rows keep their order. Only the columns actually
  written to are re-typed, because re-standardising the whole frame would re-decide
  dtypes on data nothing touched.
- **a key is a key.** A blank value in a key column is refused, not filled — it would
  become the GAMS set element `''`, same reasoning as a blank dimension at boundary 5.
- **only raw contributions are cached.** The cache holds what a processor returned and
  never a merged or melted table; every derivation is recomputed from
  `workbooks + cache` on each build, so a partial rerun reads the same as a full one.

A refusal costs the contribution alone. The time series was never in question, so the
GDX is still written and the run continues.

### Boundary 4b: after the fake MultiIndex, dtype means nothing

`create_fake_MultiIndex` pushes a row of parameter **names** into the sheet, so every
parameter column below it is `object` whatever it held before — and so is every `*_flat`
frame derived from it. This is the one place where `object` does not mean "no assumption
has been made" and does not mean something unparseable got in.

It is also why the numeric gate is not applied there: a column that legitimately mixes a
name with numbers is precisely the shape the gate reports, so running it would flag every
sheet in the workbook.

### Read, check, then process

Every source data type runs in that order: `read_input_excels` (which gates the sheet),
`normalize_dataframe`, `drop_underscore_values` — and only then `expand_all_country`, the
blacklists, `apply_whitelist` and `merge_row_by_row`.

The order is the point, not an accident. The whitelist drops rows for a scenario, year or
country this run does not cover. If it ran first, a malformed cell in one of those rows
would disappear as merely irrelevant and never be reported, and the author would discover
it only when they ran the scenario that does use it. Judging a row's *form* before its
*relevance* is what makes a build's complaints complete.

This masking is not hypothetical: a route test asserting that a `#`-prefixed row was
treated as a comment went on passing after the marker moved to `##`, because the row's
country was `#FI` and the whitelist rejected it anyway. Nothing in the test could tell the
two reasons apart, so it would have stayed green indefinitely — it surfaced only because a
neighbouring assertion about warnings failed. `TestFormalityIsJudgedBeforeRelevance` in
`route/test_route_reader_rules.py` now pins the pipeline order behaviourally.

The one check that deliberately runs late is `merge_unittypedata_into_unitdata`'s unmatched
`generator_id` report: it is a cross-reference rather than a property of the row, so it
should only speak about rows the run actually uses.

### `##` is what the author declares is not input

One marker, two placements: in a **data row** it skips the row, in a **column header** it
skips the column. Both are dropped in `read_input_excels` *before* the numeric gate, so a
helper table beside the real one — half-built formulas, `#DIV/0!`, pasted values — is never
validated as input.

Order matters and is asserted: columns are dropped first, so `##` written as free text in a
helper column cannot delete the row of the real table it happens to line up with.

The marker used to be a single `#`. Every Excel error value starts with one, so a broken
formula read as a comment and deleted its own row silently.

### Boundary 1 rejects a cell that should be a number and is not

`read_input_excels` runs `utils.gate_xlsx_frame` over every sheet. A column holding **both**
real numbers and values that look like numbers but do not parse (`1,000.0`, `100 MW`,
`(500)`) has those cells reported at `level="error"` and set to `pd.NA`.

Blanking rather than interpreting is the point: `1.000` is a thousand to one author and one
to another, and the cell does not say which. Blanking also lets `standardize_df_dtypes` type
the column `Float64` by itself, which is what stops one bad cell demoting a whole column to
`object` — the failure that dropped every row of a sheet through
`filter_nonzero_numeric_rows`, and stopped the `_output1` rename from firing.

The rule is *starts with a digit after any sign or currency symbol*, not *contains a digit*:
the weaker test blanked `chp1` out of an identifier column. A purely textual value in a
numeric column (`unknown` in `capacity`) is deliberately **not** caught — no content-only
rule can tell it from a label.

Generated input takes the opposite decision. `BaseProcessor.read_input_csv` refuses the file
outright, because a generator does not make isolated typos and blanking would manufacture a
column of zeros. It also treats a `ParserWarning` as fatal, which is the only way to catch an
unquoted thousands separator shifting every column of a CSV.

### The all-NA rule

An all-`pd.NA` column is `object`, **never** `Float64`. This is deliberate: it is the fix
for a cascade bug where empty string columns and empty float columns both became all-NA,
both were inferred to float, and code expecting a dtype crashed. `object` on an all-NA
column means *no assumption has been made*.

The obligation that creates sits on **consumers**: every function must tolerate an all-NA
`object` column where it would normally see `Float64` — handle it, or reject it with a
logged message, but never crash and never silently coerce. That property is swept in
`unit/source_data/test_contract_sweep.py`.

---

## The rules

- **R1** Never compare a whole sheet or DataFrame to anything. No `assert_frame_equal`
  against a literal, no golden file, no `to_csv()` snapshot.
- **R2** Never assert on a full row. A row dict pins every parameter that happens to be
  in the row today.
- **R3** Never assert an exact column set; use a superset check. Three exceptions, where
  the exact set *is* the contract: the processor output validation, the fake-MultiIndex
  round-trip, and the writer's sheet list.
- **R4** Never assert a whole-sheet row count. Count rows matching a key.
- **R5** **Keys may be pinned; parameters may not.**
  `("elec", "FI00_elec", "FI_u1_chp", "output")` is a coordinate — pinning it is how you
  name the thing under test. `capacity == 500` is a parameter. If you cannot tell which
  you are writing, it is a parameter.
- **R6** Prefer, in order: contract > relational > provenance > derivational > delta >
  counting > pinned.
- **R7** Never write a `{column: dtype}` map, in tests or in `src/`. It needs editing on
  every schema change and re-creates the assumption the all-NA rule exists to avoid.
  State dtype rules as *properties*.

### When pinning IS correct — the whole list

1. **Pure string transforms with a fixed contract** — `_patch_gams_file_content`. Pin the
   substituted substring, not the document.
2. **Arithmetic that is the contract** — `_safe_eval_int("365*5") == 1825`,
   `t_max = ceil((L*24 + 10920) / 1000) * 1000`, TWh/yr → MWh/h.
3. **Documented truth tables** — `merge_row_by_row`'s six methods, whose docstring at
   `source_data_loader.py:884-909` *is* the specification. Cite the line in a comment.
4. **Format contracts** — the fake-MultiIndex first row, the `_output1` strip,
   `input_output ∈ {input, output}`, the 21 sheet names.
5. **Error-message identity** — a stable substring, never the whole message.

Not pinnable: any `PARAM_*` membership, any `*_DEFAULTS` value, any capacity/cost/
efficiency in an output sheet, any row or sheet count.

### Why there are no golden files

Both sides of a delta comparison are produced by the code under test, in the same process,
in the same run. Add a column to `PARAM_GNU` and it appears in both with identical values
→ zero delta → no test edit. Change a default → both sides move → still zero. A golden
file cannot do this: one side is frozen at record time and every schema change moves the
other.

There is deliberately **no `--regenerate` flag and no `generate_goldens.py`**. Adding one
is a change to this document and needs an argument for why the delta approach stopped
working.

(The parent Backbone repo *does* use goldens, for a good reason: it compares against a
solver — an external oracle that cannot be re-run per assertion. Here every baseline is
recomputable in-process in about a second.)

---

## The contract sweeps

`tests/_common/contracts.py` holds `assert_normalized`, `assert_no_na_became_zero` and
`NASTY_CELLS` — a curated catalogue of ~33 pathological cell values crossed over every
function in `tests/_common/loader_cases.py`.

**Adding a function to the sweeps is one entry in `loader_cases.py`.** That is the point:
nobody has to remember to write a dtype assertion for a new transform, so the coverage
does not decay. `test_contract_sweep.py` also asserts that every public loader function
*is* in the table, so a new one cannot slip past unnoticed.

**When a live bug is found, add the value that caused it to `NASTY_CELLS`.** That is how
each incident becomes permanently regression-tested.

### Known violations

A function that is *known* to breach the contract carries a `known_contract_violation`
reason on its loader case, which downgrades its sweep failures to `xfail` so the rest of
the signal stays readable. Every such entry must also have a precise **strict-xfail**
tripwire in `unit/source_data/test_known_contract_violations.py`, which fails loudly the
moment the underlying issue is fixed — otherwise a fixed bug would keep its exemption
silently.

To retire one: fix the code, watch the tripwire XPASS, then delete both the tripwire and
the `known_contract_violation`. A tripwire asserts that something is broken, so it cannot
merely lose its marker — rewrite it into the positive statement of what the code now does.

**There are currently no entries.** The suite has no xfails at all: the register is empty
and every `pytest.mark.xfail` is gone. That is worth keeping true — an xfail is a decision
deferred, and the two rounds that cleared them both found the deferral had been resting on
something that was never checked.

Round one, when the time-axis gate landed: two xfails, both parked on the belief that the
check cost ~2 s per parameter and therefore belonged in a separate tool. Measured, it costs
66 ms. **An xfail whose stated blocker is a cost is only as good as the measurement behind
it**, and neither had one.

Round two cleared the remaining six. The lesson there is the mirror image: **an xfail parked
as a formatting preference is worth re-deriving from the consumer's spec.** Three of the six
were filed as "should the workbook show 0 or a blank?", which reads like a matter of taste —
but `p_userconstraint` is Rdim=6 and `inc/1e_inputs.gms` carries 21 aborts demanding that an
unused `uc` slot hold exactly `'-'`. The blank was a build failure, not a style choice, and
the framing had kept it parked for weeks. Two others in that batch turned out to be reasoning
about the wrong mechanism entirely — the fill was not missing, it was blind to all-NA columns
— which no amount of re-reading the xfail would have shown, only running the code.

When retiring an xfail, delete its reasoning too: a stale justification outlives the test and
gets quoted back as fact.

---

## How to add a test

**Pure unit.** Find the mirror module under `unit/`, import the function, build inputs as
DataFrame literals, pass `FakeLogger()`. Assert with the contract helpers plus
`logger.assert_logged(...)`. Most of these functions *log* instead of raising (the error
policy in `CLAUDE.md`: after logger init, never raise), so assert on the logger, not
`pytest.raises`.

**Route.** Write the workbook inline as a `.wb.txt` string. Call
`run_route(tmp_path, workbooks={"data.xlsx": TEXT})`. The first two assertions are always
`r.logger.assert_no_errors()` and `assert_workbook_consistent(r.sheets)`. A run takes about a
second, so use a module-scoped fixture when several tests read the same result.

**Sprawl rule:** a workbook enters `_common/workbooks/` the moment a *second* test needs it.
Until then it lives as a string literal in the one test that uses it. Every library fixture
carries a header comment saying what it exists to cover.

**Delta.** Start from a fixture, produce the variant with `workbook_text_with` — never by
hand-editing a copy, which drifts, and never by writing the variant out in full, which hides
what changed. Run both, `assert_delta`. **Omission is the assertion**: list what moved, and
everything unmentioned is asserted unchanged. Prefer the 3-tuple `(sheet, key, column)` over
the 4-tuple with a value — it says "this had to move, by how much is not my business" and
survives a fixture edit.

`assert_delta` refuses to assert nothing: passing no expected changes requires
`expect_no_change=True`, so a variant that silently failed to differ cannot pass vacuously.

---

## When a test breaks after an intentional change

You have exactly three options:

1. **Fix the code.** The test found a real regression. The common case, and the point.
2. **Change the test's INPUT.** You changed what a fixture *means* — renamed a source
   column, changed a sheet prefix. Update the fixture. Do not touch the assertion.
3. **Delete the assertion and replace it with a weaker, more durable one**, with a
   one-line comment saying what was given up.

You may **not**: re-record a number, loosen a tolerance, append to an "expected" list, or
add a branch for the new behaviour. If your instinct is *"just update the expected
value"*, the assertion was pinning a parameter and should never have existed.

**A contract failure is never option 3.** Weakening a contract to make a test pass
reintroduces exactly the bug class the contract exists to prevent. Fix the code, or change
the contract deliberately and write down why.

If a change makes more than three or four tests fail *and* every failure wants option 3:
stop. A whole module is pinned too tightly — rewrite the module, not the assertions one at
a time.
