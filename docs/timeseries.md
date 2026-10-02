# Timeseries

Everything hourly that Backbone reads — demand, hydro inflow, wind and solar
availability, storage limits — is built by a **processor**: one small program per
data source, run by a shared pipeline that does whatever every source has in
common. This page is about the shared part. What any individual source *means*
has its own page, and those pages come and go as the data sources do.

## One minute summary

- **One processor per source; the pipeline does the rest.** A processor reads its
  own files and returns one long table of dimensions, time and value. Labelling
  the hours, cutting climate windows, building forecast branches, writing GDX and
  telling GAMS about it all happen once, in shared code, identically for every
  source.
- **A climate year is a weather year, not a scenario year.** The shipped configs
  build 1982–2016: thirty-five versions of the same scenario year, each with a
  different year's weather. A Backbone run uses one of them at a time.
- **Forecast branches are the climate, counted in energy.** `f01` gives every
  series its mean energy in a typical year's shape; `f02`, `f03`, … a stated low or
  high over their own length, set by `energy_quantiles` in the config.
- **A zero is not a value.** Backbone reads `0` as "not set", so the pipeline
  keeps "no data" and "zero" apart until the last possible moment, and counts
  what it converts.
- **One bad source costs one source.** A processor that fails is reported and
  writes no GDX; everything else in the build still runs.
- **The build says what it did** — what was built, what was not, and why — once
  per run, in the log.

| | |
|---|---|
| what a processor returns | long format: dimensions + `time` + `value` |
| what the pipeline writes | `{bb_parameter}_{gdx_name_suffix}_{year}.gdx` |
| which weather years | `climate_data`, in `src_files/config_*.ini` |
| how much of each year | `bb_timeseries_start`, `bb_timeseries_length`, same file |
| which sources run, and how each behaves | `timeseries_specs`, same file |

## Contents

- [The pages for individual sources](#the-pages-for-individual-sources)
- [What a build produces](#what-a-build-produces)
- [The shape of the pipeline](#the-shape-of-the-pipeline)
- [Climate years and the window](#climate-years-and-the-window)
- [Forecast branches](#forecast-branches)
- [Why zero is the hard case](#why-zero-is-the-hard-case)
- [What the runner checks before it writes](#what-the-runner-checks-before-it-writes)
- [What a build says](#what-a-build-says)
- [What a processor contributes besides the series](#what-a-processor-contributes-besides-the-series)
- [What is cached, and what forces a rebuild](#what-is-cached-and-what-forces-a-rebuild)
- [Adding a source](#adding-a-source)
- [Where the timeseries code lives](#where-the-timeseries-code-lives)

## The pages for individual sources

This page deliberately explains no particular dataset. Each source has its own
page, and **that set is a snapshot rather than a fixture**: a page is deleted
when its source is retired, and a new one is added when a better source arrives.
Nothing here should be read as a claim about which sources exist.

Today:

- [Hydro data](hydro.md)
- [District heating demand timeseries](dh-demand-timeseries.md)
- [Electricity demand timeseries](elec-demand-timeseries.md)
- [Wind and solar timeseries](vre-timeseries.md)

If a page you expected is missing, its source has probably been retired.
`timeseries_specs` in your config is the list that actually decides what runs,
and it is the one to check first.

## What a build produces

Into the output folder, per source:

- **One GDX per climate year**, `{bb_parameter}_{gdx_name_suffix}_{year}.gdx`, or
  a single `{bb_parameter}_{gdx_name_suffix}.gdx` when the run has only one year.
  Both halves of the name are that source's own `bb_parameter` and
  `gdx_name_suffix` from `timeseries_specs`.
- **A line in `import_timeseries.inc`**, the GAMS include file that loads them.
  For the per-year case it reads the year from `%climateYear%`, so Backbone picks
  the window at run time rather than at build time.

Two things do not belong to any one source. Demand grids that appear in the
demand tables but have no processor of their own get a **constant** demand rather
than a time series: the annual energy spread evenly over the hours, which is the
honest thing to write when nothing knows the shape. And a processor may
**contribute to the source data tables** — see below — instead of, or as well as,
writing a GDX.

## The shape of the pipeline

```
config: timeseries_specs
        │
        ▼
  TimeseriesPipeline      decides, per source: run it or leave last run's output
        │
        ▼
  ProcessorRunner         once per source
        │
        ├── the processor   reads its files, returns one long table
        ├── check           columns, dimensions, values, time axis
        ├── round, cut off  rounding_precision, cutoff_below
        ├── label + slice   t-labels, one climate window per year
        ├── forecasts       branches from the climate windows, energy per series
        └── write           GDX + a line in import_timeseries.inc
                            and whatever the processor contributed to the
                            source data tables, for the input Excel
```

The split is the point. **A processor knows about its source and nothing else:**
where the files are, what their columns mean, which repairs are honest, what a
zero means in that data. **The runner knows about Backbone and nothing about any
source:** how an hour becomes a `t` label, what a climate window is, what a GDX
needs.

So a processor never filters to a window, never assigns a `t` or an `f`, and
never writes a file. It returns every hour of the configured range and stops —
and a new source inherits everything after that for free.

## Climate years and the window

All three settings below live in your run's `src_files/config_*.ini`, under
`# --- Timeseries -------`, and they are **global**: they apply to every source
alike, because a window that meant different hours for demand than for wind would
not be one window. The per-source settings sit inside that source's own
`timeseries_specs` entry instead, and the comment block above it documents every
field.

`climate_data` picks the weather years — `1982-2016` in every shipped config, and
that is also the range the current sources cover. Each becomes its own GDX, and
Backbone reads one per run.

A window need not be a calendar year:

- `bb_timeseries_start` is the day each window opens, `MM-DD`, default `01-01`.
- `bb_timeseries_length` is how many days it runs, default `365`. It accepts
  expressions, so `365*5` is a five-year window and `365*35+9` is the whole
  climate range as one continuous series.

**Five years is the practical ceiling.** A window that long exists to be run as
one multi-year run, and Backbone slows sharply between three and five years;
beyond five a run does not finish today. The build warns about any window longer
than five years, and the forecast branches below are a second reason: they thin
out as windows grow.

Three consequences worth knowing. A window that does not start on 1 January puts
the **calendar year change inside the sample**, where the solver has to absorb
whatever discontinuity is there — sources whose data is naturally annual care
about this, and their pages say so. A window longer than the data left after
its start year simply cannot be built for the last few years: those years are
dropped, and the build says which and why rather than writing a short one
silently.

And a window that is **not a whole number of years joins two seasons**.
Backbone's look-ahead runs past the end of the data and carries on from the
window's first hour, unsmoothed — so an 800-day window from `01-01` goes from
10 March straight back to 1 January. The build warns when the length is more
than three days from a whole number of calendar years, leap days counted —
`365`, `365*5` and `365*35+9` are all whole — and says how long a run can be
before it meets the join: the window minus the horizon.

The horizon is `bb_horizon_weeks` in the same block, 52 weeks (364 days) by
default. It is not a timeseries setting, but it decides how far each solve's
look-ahead reaches past the window, and the build sizes the model's `t` set to
the window plus one horizon. [Running the model](running-the-model.md#horizon-and-forecast-discount)
covers what it does to a solve.

## Forecast branches

Every daily solve has to plan beyond the day it realizes: how much water to keep,
which plants to keep warm. An optimisation model knows nothing it is not given —
not that Januaries are cold, not that spring brings the snowmelt — so the build
gives it the climate, as **forecast branches** on Backbone's `f` index, all made
from the climate record:

- **`f00` is the realized weather**: the climate window being run, exactly as the
  processor produced it.
- **`f01` is the central forecast**: the climate's average. Every series carries
  its mean energy, in the shape of a typical year.
- **`f02`, `f03`, … are side forecasts**: a dry or calm spell, a wet or windy one,
  each carrying a stated low or high share of the climate's energy.

### What you set

All three live in the global block of `src_files/config_*.ini`:

| setting | says | default |
|---|---|---|
| `energy_quantiles` | how much energy each branch carries | `{'f01': 0.5, 'f02': 0.1, 'f03': 0.9}` |
| `forecast_weights` | how likely each branch is | `{'f01': 0.6, 'f02': 0.2, 'f03': 0.2}` |
| `forecast_branches` | how long each side branch lasts, and how it ends | 149 days, then cut |

An empty `energy_quantiles` is the deterministic mode: no branches, no forecast
file. A timeseries spec may carry an `energy_quantiles` of its own, which replaces
the global value for its series alone — a dry branch can be one-in-ten for hydro
inflow and milder for wind.

### What an energy quantile means

The number is a point in the climate years, counted in energy:

| value | the branch carries |
|---|---|
| `0.5` | the mean energy of the climate years |
| `0.1` | a low: one climate year in ten has less. A dry branch for hydro, a calm one for wind |
| `0.9` | a high: one climate year in ten has more |
| `0` | the lowest of the climate years at that time of year: the calmest, the driest |

Two things make it concrete:

- **It is measured over the branch's own length.** A 252-day branch at `0.1` is a
  dry 252 days; a 5-day branch at `0.1` is a calm 5 days. Those are different
  things — short spells swing much further from the mean than long ones — so the
  same number makes a deeper branch when the branch is short.
- **It holds for every series on its own.** Every country's wind, every hydro
  node, every demand node gets the energy its number asks for from its own
  climate years. `0.5` is the mean for each of them, however different their
  weather.

An example. A hydro node whose inflow averages 20 TWh a year, and whose driest
year in ten brings 17, gets a whole-year branch of 20 TWh at `0.5` and of about
17 TWh at `0.1`.

Demand is a negative inflow in Backbone, so for demand `0.1` is the
**high**-demand branch. A low value is the hard direction for every series.

### How a branch is built

Each branch has the **shape** of the climate years and the **energy** its number
asks for:

- **Shape.** At every hour, the branch takes a value from the spread of that same
  hour across the climate years. Winter hours stay winter-like and nights
  night-like, and a branch has no step the realized years do not have.
- **Energy.** Which point of that spread it takes — low, middle or high — is
  chosen for each series separately, so that the branch carries the energy asked
  for. The point differs by country, because some weather is more lopsided than
  other: most hours are calm and a few are very windy, so the middle hour of wind
  is well below its average hour, and a wind branch at `0.5` takes points above
  the middle.

The build log says which points were used, one line per source: *Per-hour
quantiles that carry the energy quantiles, over N series: f01 0.5 -> 0.47-0.70,
…*. `python tools/forecast_branch_energy.py <built_folder>` reads a built folder
and states what each branch carries against the climate years (`--by-node` per
series).

**The exact rule**, for whoever needs it: over every window of the branch's
length, from every start day of the climate window, the branch carries
`Q_p + (mean − Q_0.5) × (1 − |1 − 2p|)` of the climate years' energy in that
window, where `Q_p` is their p-quantile. At `0.5` that is the mean exactly; towards
`0` and `1` the step from the median up to the mean fades out, so a low value is
the climate years' own low. That matters for short wind branches: a few very windy
spells pull the mean well above the median, and without the fade even `0` would
stay far from the calmest spell on record. `calculate_climatological_forecasts` in
`src/timeseries/timeseries_helpers.py` has the derivation.

### What a central branch cannot carry

`f01` is the climate's average, day by day — and an average has no weather. In a
real year the north can be windy while the south is calm, or a calm week can also
be a cold one; across many years those cancel out, so in `f01` every region
simply follows the seasons together. That is by design: no single forecast can
hold the weather of every year. The realized branch carries the year's own
weather, and the side branches carry how far from the average it can go.

### How long a branch lasts, and how it ends

`forecast_branches` states, for each branch beside the central one, how many days
it carries its own data and what happens after that:

```
forecast_branches = {
    'f01': 'central',
    'f02': {'length_days': 252, 'end': 'continue', 'blend_days': 28},
    'f03': {'length_days': 252, 'end': 'continue', 'blend_days': 28},
    'f04': {'length_days': 5, 'end': 'cut'},
    }
```

| `end` | after its length the branch | suits |
|---|---|---|
| `cut` | stops, and the remaining branches share its probability | a short event the model has to get through |
| `bound` | ends at the central branch's storage levels | a short-term forecast that converges back |
| `continue` | runs on to the horizon on the central branch's data, keeping its own storage levels and its probability | a long deviation such as a dry year |

`blend_days` eases a continuing branch from its own data into the central
branch's. A branch the key does not name is 149 days long and cut. `'f01':
'central'` is there for the reader: f01 always reaches the horizon and takes no
settings, and no other branch can be central, because `scheduleInit.gms` and
`changes.inc` name f01.

Four things follow from how Backbone reads these:

- **The length is part of the branch's data.** Its energy quantile is measured
  over it, so a branch made longer also changes what it carries.
- **A cut branch values nothing past its cut.** What its storages hold at the
  end is worth nothing to it, so it spends what the storage limits let it.
- **A length counts to the start of a model time step.** From day 15 of a solve
  the steps are a week long, so a length inside a week holds until that week
  ends: 250 days acts as 252. `blend_days` is sampled the same way.
- **A branch costs what its length costs.** Every branch adds its own variables
  for every step it lasts; past day 15 a week is one step, so a long branch costs
  less per day than its first two weeks do.

Near the solve every branch is also pulled towards the realized weather, over
five days for wind, solar and most series, seven for demand and fourteen for
hydro inflow (`scheduleInit.gms`; a branch shorter than seven or fourteen days
keeps the five). A branch is fully its own only after that, so one shorter than
its pull carries only part of its deviation.

### What a change rebuilds

| changed | rebuilt |
|---|---|
| `energy_quantiles`, `forecast_weights` | everything |
| a branch's `length_days` | everything: the length is part of the branch's data |
| a branch's `end` or `blend_days` | the GAMS files only |
| a spec's own `energy_quantiles` | that source |

### Long windows

The branches are built from the spread across the climate windows, and windows
longer than a year overlap: a window of N years leaves 36 − N of them in
1982–2016 — 31 for five years, 16 for twenty, and one for `365*35+9`, where every
branch would be the realized window itself. That is the other reason the build
warns above five years. A source with fewer than two climate windows cannot have
branches at all, and is told so.

## Why zero is the hard case

GAMS has no NaN, and a plain `0` **is** absent. A node whose demand is zero for
one hour is indistinguishable, downstream, from a node whose demand was never
built — and most ways a processor can fail produce the second while looking like
the first.

The pipeline's answer is to keep the two apart for as long as possible:

1. A gap in the source stays `NaN` through the processor and everything after it.
2. `GDX_exchange.prepare_values_for_gdx` is the **single** place it becomes `0`,
   at the GDX boundary, and it counts what it converted.
3. Nothing upstream may fill early. Filling makes a source gap indistinguishable
   from a real zero — and because the forecast branches skip `NaN` but not `0`,
   an early fill also drags every branch downward.

Two per-source settings make zeros of their own, after the processor has
returned: `rounding_precision` rounds the value, and `cutoff_below` sends small
magnitudes to zero to keep tiny coefficients out of the LP. Both sit in that
source's `timeseries_specs` entry, and a processor that checks its own output for
zeros has to test what those two will leave *written*, not what it holds.

**What a zero means is a property of the source, not of the pipeline.** Zero
demand in a heat network is impossible; zero wind is an ordinary calm hour. So
the pipeline takes no position, and each source's page states its own rule.

## What the runner checks before it writes

Everything below is refused with a message naming the processor, and costs that
one source its GDX. The rest of the build carries on.

| check | what it catches |
|---|---|
| exact columns | a processor returning more or fewer than the spec's dimensions plus `time` and `value` |
| no blank dimension value | a blank where a GAMS set element belongs |
| numeric `value` | text that survived the read |
| `time` is datetime | a column that cannot be dated |
| **time axis** | see below |

The time axis check is the one worth understanding, because it guards against a
failure nothing downstream can see. **t-labels are assigned by row position**, so
a missing hour does not leave a hole — it pulls every later hour of that series
one label earlier, for the rest of the window. The numbers stay entirely
plausible and are simply attached to the wrong hours, and for a model whose value
is largely the correlation between countries, an undetected one-hour offset
between two of them is not a small error.

So the runner proves two things: within each series, consecutive rows are exactly
one hour apart; across series, every one covers the same span. Repeats, holes,
sub-hourly rows and ragged spans are errors, with no config override.

Separately, a processor may **declare** what its output should look like —
`value_range`, `value_sign` — and the runner checks the declaration against the
data on every run. Those are warnings rather than refusals: an out-of-range value
may be a real feature of the source, where a broken time axis cannot be.

One more warning is worth knowing about, because it is the only thing standing
between a mistyped node name and silence. The runner checks every `node`, `grid`
and `flow` value a processor produced against the source data tables, and names
any the model does not have. Backbone will not read a series for a node nothing
else refers to, so the usual cause is a spelling mistake or a workbook missing
rows the processor expected.

"The model does not have it" is a wider question than it looks, and
`src/source_workbook_shape.py` is where the answer lives. A grid or a node can be
declared by four tables: `nodedata` and `demanddata` one per row, `unitdata` one
per unit connection — which is how every battery, heat store and fuel grid enters
the model without a `nodedata` row of its own — and both ends of a `transferdata`
link. That is the same union the input Excel builds its `grid` and `node` sheets
from, so a value this warning names is genuinely one nothing else in the model
mentions.

## What a build says

A build log is read by someone who wants to know whether anything needs their
attention. Everything else in it is cost, and a page of standing text is worse
than cost: it is what teaches a reader to skip the line that is new.

So the rule, for every message a build writes -- the timeseries pipeline, the
source data phase and the input Excel builder alike:

1. **A warning asks the reader to change something.** If there is nothing they
   can do about it, it is not a warning, whatever its tone.
2. **Absence is not a defect.** The source workbooks are the statement of what
   the model contains. A country with no district heating, a zone with no
   reservoir, a landlocked country with no offshore wind — the build says nothing
   at all. Spain has no district heating and never will; a line saying so every
   run only makes its reader wonder what they did wrong.
3. **Partial or contradictory data earns a line, and the line names names.** The
   model has the node or the unit, and the data for it is missing or
   inconsistent: a node in `nodedata` but not in `demanddata`, a hydro node with
   no inflow anywhere in the source, a unit whose `flow` has no capacity factor
   series. That is the case worth interrupting someone for, and it is the case
   the checks are shaped around.

   One line, naming the **first three offenders and then counting the rest** --
   `summarise` in `utils.py` is exactly that, and at three or fewer it names them
   all. A bare count is not enough: the reader's next question after "1 node has
   no price data" is always *which node*, and the log already knows. But nor is a
   line per offender: a missing input file leaves a hundred units with partial
   data, and a hundred lines is a hundred lines nobody reads. This is the one
   place a warning spends names, which is why clause 4 is strict about the rest.
4. **Expected and handled costs counts, not names, and never reasons.** A repair
   the rules made, a decision taken once and recorded in code, a check that found
   what it always finds — one short line per processor, and the names and the
   reasoning stay in this documentation where they do not have to be retyped into
   every log. `Gaps interpolated at 7 node(s), 3 large year change(s) left as
   they are.` is a whole build's worth of hydro repairs.
5. **Progress lines carry the run.** `Validating and curing processor output...`
   is the shape to copy: it says all normal and nothing else, and it is a fine
   summary of a thousand lines of code that found nothing to report.

The same rule governs what a *check* is worth adding. A test that fires on
correct data every run is not a strict check, it is a broken one — the isolated
capacity-factor dropout test has a threshold precisely so that it stays silent on
weather and speaks on dropped values.

## What a processor contributes besides the series

Most processors return the time series and nothing else. A node, a grid or a flow
the model already has needs no announcing — it is in the source workbooks, which
is how the processor found it in the first place.

What does need saying is a fact about the data that only the processor knows and
nothing downstream can work out. Today there is exactly one: the hydro storage
limits are a **time series** rather than a constant, and the input Excel has to
say so or Backbone uses the node's constant and never opens the GDX.

A processor says it by filling `self.frames` with tables named after the source
data ones — `nodedata`, `boundarydata`, `demanddata` and the rest, the same names
`requires_source_data` uses to ask for them. They are merged into those tables
after the timeseries phase, and the input Excel is built from the result.

Two rules govern the merge. **The workbook wins**: a contribution fills only
where the source data said nothing, so a value written by hand is never
overwritten by a processor. And a contribution is checked before it is accepted —
an unknown table name, a missing key column or a blank key is reported naming the
processor and dropped. That costs the contribution alone: the time series is
unaffected and its GDX is still written.

## What is cached, and what forces a rebuild

**A processor is rerun when the input it is given changed, or its own code
changed.** Nothing else, and its previous output stands otherwise.

The decision is taken in two stages, because the two questions can be answered
at different times.

**What might need rerunning** is `CacheManager`'s, before any data is read. It
is deliberately generous: a `timeseries_specs` entry that moved, a processor
file whose hash changed, a source workbook a processor declared, a demand
workbook when the spec has a `demand_grid`, an input file that is no longer what
it was, or `force_full_rerun`. A sheet hash cannot tell which *cell* moved, so
this stage answers "a workbook you read was edited, somewhere".

**What actually needs rerunning** is `TimeseriesPipeline`'s, once the source
data has been read and the input exists. It compares what the processor would be
handed now against what it was handed last time — the record described below —
and spares the ones whose input is identical. This is why a cost tweak in
`unitdata` no longer rebuilds 742 MB of PECD: `VRE_PECD` is given the `flow` and
node columns of `unitdata` and nothing else, so a `vomCosts` cell is not part of
its input and cannot move its record.

### The record

Per spec, in `cache/processor_inputs/`, and readable on purpose: it is what the
build quotes when it says why something reran. It holds the source-data frames
the processor was handed, narrowed to the columns its `requires_source_data`
declares; the spec and config values it was called with; and the size and
modification time of the input files its `reads_input_files` declares.

The recorded frame **is** the delivered frame, not a description of one, so the
two cannot drift apart. That is the reason `ProcessorRunner` narrows the
delivery rather than only the comparison.

Two "cannot tell" answers both mean rerun, and both are deliberate:

- **A processor that declares no `reads_input_files` reruns every build.** There
  is no comparing files nobody named, and a replaced PECD download touches no
  workbook and no config — so a cache that assumed "no files" would serve the
  old GDX with nothing saying so.
- **A processor that names a table without naming columns** gets the whole frame
  and compares the whole frame. It still works, which is what a processor
  written outside this repo will do; it just cannot tell an edit it reads from
  one it does not.

The build says what it found. `CacheManager` prints the run plan before the
first phase; the timeseries phase then prints which processors it spared and, for
those it did not, the value that actually changed — `demanddata country=FI
twh/year 100.0 -> 101.0` rather than "the demand data changed".

What is kept between runs is what each processor *returned*: its GDX files, and
its contributions to the source data tables exactly as it produced them. Nothing
merged is ever cached, so the input Excel is rebuilt from the source workbooks
plus those contributions every time — which is what makes a partial rerun, where
most sources did not execute, describe the same model as a full one.

Every processor is scenario-dependent, and a multi-scenario build runs each of
them per scenario. A build used to copy the weather-driven ones from the first
output folder on the grounds that weather does not care which scenario year is
modelled — but what gets *built* does: the VRE processors read `unitdata` to
learn which nodes have a unit of their flow, and that is filtered per scenario,
year and country. A copy from another scenario's folder would be that scenario's
answer wearing this one's name. The copying, its `is_input_data_dependent` spec
key and the checks that guarded it are gone; a second scenario costs about a
minute more.

## Adding a source

1. Write `src/timeseries/processors/<Name>.py` with a class `<Name>` — the file
   and the class must share the name — subclassing `BaseProcessor`.
2. Implement `process()`. Return a long table of the spec's dimensions (without
   `t` and `f`) plus `time` and `value`, covering every hour of the configured
   range. Read files through `read_input_csv` / `read_input_excel`, which refuse
   input whose numbers or field count are wrong.
3. Declare what the output must always be: `value_range`, `value_sign`, and
   `requires_source_data` if you need a merged source-data frame. Declarations
   are checked against the real data on every run and are versioned with the
   file, so they cannot go stale.
   Fill `self.frames` only if the input Excel needs to be told something the
   source workbooks cannot say — see above. Most sources need nothing here.
4. Add an entry to `timeseries_specs` in the config. The comment block above it
   documents every field.
5. Write a page in `docs/`, link it from `README.md`, and add it to the list on
   this page.

Report per-node problems rather than raising. An exception is caught at
whole-processor level, so one bad cell that raises costs every node in the run
its time series; the same cell reported costs one node and names it.

## Where the timeseries code lives

| | |
|---|---|
| `src/timeseries/timeseries_pipeline.py` | decides what runs, copies or is skipped; handles demand grids with no processor |
| `src/timeseries/timeseries_processor.py` | runs one processor, checks its output, writes the GDX. Carries the processor contract |
| `src/timeseries/timeseries_helpers.py` | labelling, the time-axis check, climate windows, forecast branches, gap filling |
| `src/timeseries/processors/base_processor.py` | the base class, the declarations, and the file readers |
| `src/timeseries/processors/*.py` | one file per source |
| `src/GDX_exchange.py` | the GDX boundary, and the one NaN-to-zero conversion |
| `src/source_data/source_data_contributions.py` | what a processor may add to a source data table, and how it is merged in |
| `src/source_workbook_shape.py` | which table declares which dimension, for the check above |
| `src_files/config_*.ini` | `timeseries_specs`, `climate_data`, the window and the forecast branches |

## See also

- [Source workbook conventions](source-workbook-conventions.md) — how the Excel
  files behind the annual figures are read and combined
- [Input Excel builder](input-excel.md) — the phase after this one, and what a
  contribution to a source data table ends up looking like in `inputData.xlsx`
- `tests/README.md` — the NA and zero boundary map, for anyone changing the
  pipeline rather than the data
