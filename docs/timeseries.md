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
- **An annual summary CSV**, for a source whose spec sets `annual_summary` — the
  annual mean or sum of each series, which is the quickest way to see whether one
  is the right size.

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
        ├── forecasts       quantile branches across climate years
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

Two consequences worth knowing. A window that does not start on 1 January puts
the **calendar year change inside the sample**, where the solver has to absorb
whatever discontinuity is there — sources whose data is naturally annual care
about this, and their pages say so. And a window longer than the data left after
its start year simply cannot be built for the last few years: those years are
dropped, and the build says which and why rather than writing a short one
silently.

## Forecast branches

Backbone can carry uncertainty as several forecast branches on the `f` index.
The pipeline fills them from the climate record itself:

- **`f00` is the realized weather** — the climate year being run, exactly as the
  processor produced it.
- **`f01`, `f02`, … are quantiles** taken across all climate years at each hour
  of the year, so `f01: 0.5` is the median year at that hour and `f02: 0.1` the
  low decile. The branches are written once into a `_forecasts.gdx`, because they
  are the same for every window.

`forecast_quantiles` names them, in the same global config block as the window
settings above — `{'f01': 0.5, 'f02': 0.1, 'f03': 0.9}` by default. Leaving it
empty is the deterministic mode: no branches, no forecast file. A source with
fewer than two climate years cannot have branches either, and is told so.

`forecast_weights` beside it is the probability of each branch. It is written
into the **GAMS files** at the end of the build, alongside the branch count, and
never into a timeseries GDX.

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
   from a real zero — and because the forecast quantiles skip `NaN` but not `0`,
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

A processor is rerun when its `timeseries_specs` entry changed, when its own
source file changed (the runner hashes it), or when `force_full_rerun = True` at
the top of the config. Otherwise its previous output stands.

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
| `src/timeseries/timeseries_helpers.py` | labelling, the time-axis check, climate windows, forecast quantiles, gap filling |
| `src/timeseries/processors/base_processor.py` | the base class, the declarations, and the file readers |
| `src/timeseries/processors/*.py` | one file per source |
| `src/GDX_exchange.py` | the GDX boundary, and the one NaN-to-zero conversion |
| `src/source_data/source_data_contributions.py` | what a processor may add to a source data table, and how it is merged in |
| `src/source_workbook_shape.py` | which table declares which dimension, for the check above |
| `src_files/config_*.ini` | `timeseries_specs`, `climate_data`, the window and the forecast branches |

## See also

- [Source workbook conventions](source-workbook-conventions.md) — how the Excel
  files behind the annual figures are read and combined
- `tests/README.md` — the NA and zero boundary map, for anyone changing the
  pipeline rather than the data
