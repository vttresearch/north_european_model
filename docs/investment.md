# Investment and unit lifetimes

What to put in the source workbooks so that a Backbone model can build new
capacity. That means units, transfer links and storage, and when an existing unit
starts or stops running. This project only prepares the data. Whether anything
gets built is decided by the model run, and an investment run needs an init file
that this project does not ship.

## One minute summary

- **There are three things to invest in, and each has its own source table.** A
  unit's columns go on `unittypedata` or `unitdata`, a transfer link's on
  `transferdata`, and a storage node's energy capacity on `nodedata`. The build
  copies them into the Backbone sheets in the table below. Nothing is derived.
- **`maxUnitCount` is what makes a unit investable.** Backbone offers a unit for
  investment only when `maxUnitCount` is set. `invCosts` without it keeps a
  zero-capacity unit in the workbook, but the model has no way to build it.
- **`becomeAvailable` and `becomeUnavailable` say when a unit exists.** Each holds
  a time step number, where `1` is the first hour of the climate window, and the
  build writes it as a row of `utAvailabilityLimits`. Use them for commissioning,
  decommissioning, a maintenance break, or two vintages of one technology. They
  work in schedule runs too.
- **`annuityFactor` is yours to calculate.** It turns an investment cost into a
  yearly one, from the asset's lifetime and the discount rate. The build copies it
  and does not compute it.
- **The run is not set up here.** `src_files/GAMS_files/` ships `scheduleInit.gms`
  only. An investment run needs an init file of its own, chosen with `init_file`
  in `modelsInit.gms`. `../docs/running-backbone/investment-runs.md` says what goes
  in one.

| To invest in | Source table | Columns | Backbone sheet |
|---|---|---|---|
| a unit | `unittypedata` or `unitdata` | `maxUnitCount`, `minUnitCount`, `investMIP` | `p_unit` |
| | | `invCosts`, `annuityFactor`, `fomCosts`, `unitSize`, `invEnergyCost` | `p_gnu_io` |
| a transfer link | `transferdata` | `transferCapInvLimit`, `invCost`, `annuityFactor`, `investMIP` | `p_gnn` |
| storage energy capacity | `nodedata` | `maxInvest`, `invCost`, `annuityFactor` | `p_gn` |
| when a unit exists | `unittypedata` or `unitdata` | `becomeAvailable`, `becomeUnavailable` | `utAvailabilityLimits` |

`invCosts` is plural on a unit and `invCost` singular on a link or a node. That
is Backbone's spelling, and a header spelled the other way round is reported as
a column nothing reads.

## Contents

- [What the build does with these columns](#what-the-build-does-with-these-columns)
- [When a unit exists](#when-a-unit-exists)
- [Running an investment model](#running-an-investment-model)
- [Where this is defined](#where-this-is-defined)

## What the build does with these columns

It copies them, with the same rules as every other parameter
([Input Excel builder](input-excel.md)). `0` and an empty cell both mean "not
set", and a column reaches the sheet only if some row sets it.

- **A unit's cost columns belong to one connection.** `invCosts`, `annuityFactor`,
  `fomCosts`, `unitSize` and `invEnergyCost` are `p_gnu_io` parameters, so they
  may carry a connection suffix. A bare name means `_output1`, as it does for
  `capacity` ([Connection suffixes](source-workbook-conventions.md#connection-suffixes)).
  `maxUnitCount`, `minUnitCount` and `investMIP` belong to the unit as a whole,
  and a suffix on them is reported.
- **A candidate with no capacity is kept.** The build drops a unit only when it
  has no capacity, no `invCosts` and no `maxUnitCount`
  ([Units it drops](input-excel.md#units-it-drops)). A pure investment candidate
  can therefore start at zero.
- **A unittype row applies to every unit of that type.** A cost written once on
  `unittypedata` reaches every country's unit of the type, and a `unitdata` row
  can override it for one unit.

## When a unit exists

Two columns on `unittypedata` or `unitdata`, both counted in hours from the
start of the climate window:

- **`becomeAvailable`**: the unit is offline before this time step.
- **`becomeUnavailable`**: the unit is offline from this time step on.

The number is the hour, counted the way `t_start` is in `scheduleInit.gms`. `1`
is `t000001`, the first hour of every climate window, and `13` is `t000013`. A
window starts on `bb_timeseries_start` and is `bb_timeseries_length` days long,
both set per config in `src_files/config_*.ini`. A time step is a position in the
window, not a date, so the same number means the same hour of whichever climate
year a run picks.

| You want | Write |
|---|---|
| a unit commissioned on day 100 of the window | `becomeAvailable` = 2377 (99 × 24 + 1) |
| a unit retired after day 200 | `becomeUnavailable` = 4801 (200 × 24 + 1) |
| a maintenance break over days 150–160 | `becomeUnavailable` = 3577, `becomeAvailable` = 3841 (the start of day 161) |
| two vintages of one technology | two `unitdata` rows with different `unit_name_prefix`, each with its own `becomeAvailable` |

A `becomeUnavailable` before the `becomeAvailable` is read by Backbone as a
break: the unit runs until the first and again from the second. Writing both on
the same step makes the unit never available.

**What the build checks.** A value has to be a whole number from 1 to the
window's length in hours. Anything else, a fraction, a negative or a step past
the end of the window, is reported by unit and left out, because past the window
it would only mean "never" or "always". `0` and an empty cell mean no limit. Only
units that reach `p_unit` get a row: Backbone checks each row's unit against the
`unit` set, and a row for a unit the build dropped would stop the run.

**Why a set and not `p_unit`.** Backbone has `p_unit` parameters with the same two
names, but it calculates them from `utAvailabilityLimits` ("Unit lifetime" in
`../inc/1e_inputs.gms`). A value written straight into `p_unit` skips Backbone's
checks on the set, and the investment-energy cost term reads the set alone. So
the build writes the set and leaves `p_unit` alone.

**In an investment run.** A unit investment is decided once for the whole run,
and `becomeAvailable` is how a later vintage enters it. Transfer links are the
exception: they are built at the `t_invest` steps, which this build does not
write (see below).

## Running an investment model

The build writes the same input folder for schedule and investment runs. What
makes a run an investment run is its init file, the GAMS file that sets the time
structure and the model type. The two are run differently;
[Schedule runs and investment runs](running-the-model.md#schedule-runs-and-investment-runs)
sets them side by side.

- **Choose it with `init_file`.** `src_files/GAMS_files/modelsInit.gms` includes
  `%init_file%` and defaults to `scheduleInit.gms`. Pass another one after a bare
  `--`, for example `python run_model.py <input folder> -- --init_file=investInit.gms`.
  Backbone reads it from the input folder. A build copies every file in
  `src_files/GAMS_files/` there, so that is where one goes. This project does not
  ship one.
- **Give it more solver threads.** `1_options.gms` gives the solver one thread,
  which wins in schedule runs only
  ([Solver threads](running-the-model.md#solver-threads)). An investment run is
  one large solve that can use more:
  `python run_model.py <input folder> --threads -1 -- --init_file=investInit.gms`
  gives it every core but one, for that run only. Started with another
  `--init_file` and no `--threads`, `run_model.py` prints a note saying so.
- **Run it on its own, unpinned.** Its solve already uses several cores and holds
  the whole problem in memory, so another investment run or a batch of schedule
  runs beside it takes cores and memory from it. Leave out `--cpus`: held on the
  performance cores, a many-thread solve loses the others.
- **What goes in it** is Backbone's subject:
  `../docs/running-backbone/investment-runs.md` covers the single solve,
  representative periods and the weights that go with them, and
  `../docs/running-backbone/investment-then-schedule.md` covers operating an
  invested system over the full year. `../docs/features/investment-modelling.md`
  explains the cost terms.
- **One run sees one window.** Each climate year is its own window from
  `t000001` ([Timeseries](timeseries.md)). An investment run over one window
  therefore chooses capacity for that climate year's weather alone.

Three Backbone inputs an investment model may want are not written here
([Identified gaps](identified-gaps.md)):

- **`t_invest`**, the steps at which transfer links may be built. When it is empty
  and some link has `transferCapInvLimit`, Backbone uses the first modelled step,
  so a single-step build-out needs nothing more.
- **`p_gnn` `unitSize`**, the size of one integer link investment. Backbone
  divides `transferCapInvLimit` by it for a link with `investMIP`, so leave
  `investMIP` empty on `transferdata` and invest in links continuously.
- **Group investment limits** (`p_groupPolicy`, `p_groupPolicyUnit`), such as a cap
  on wind and solar together.

## Where this is defined

- `src/backbone_params.py`: the `PARAM_*` lists that say which of these columns
  are read, and `UNIT_AVAILABILITY_LIMITS`.
- `src/bb_excel/bb_excel_pipeline.py`: `create_utAvailabilityLimits`, and
  `drop_redundant_units` for which units are kept.
- `tests/unit/bb_excel/test_unit_availability_limits.py` and
  `TestUnitAvailabilityLimits` in `tests/route/test_route_features.py`.

## See also

- [Input Excel builder](input-excel.md): every sheet the build writes, and what
  it does with a zero
- [Source workbook conventions](source-workbook-conventions.md): how a column is
  named, suffixed and combined across files
- [Running the model](running-the-model.md): the one command that starts a run
- `docs/dictionary.md` in the Backbone repository: what each parameter means
