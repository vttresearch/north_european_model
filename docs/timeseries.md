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
demand tables but have no processor of their own get a **flat** series in
`ts_influx_other_demands.gdx`: the annual energy spread evenly over the window,
which is the honest thing to write when nothing knows the shape. And some
processors return a **secondary result** — a small table about what they built —
which is cached for the Excel build rather than written to GDX.

## The shape of the pipeline

```
config: timeseries_specs
        │
        ▼
  TimeseriesPipeline      decides, per source: run it, copy it, or skip it
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

## What is cached, and what forces a rebuild

A processor is rerun when its `timeseries_specs` entry changed, when its own
source file changed (the runner hashes it), or when `force_full_rerun = True` at
the top of the config. Otherwise its previous output stands.

A processor may also be **copied instead of run**. A source that does not depend
on the scenario — weather does not care which scenario year is being modelled —
takes `is_input_data_dependent: false` in its spec, and a multi-scenario build
then copies its GDX, its cached results and its hash from the first output folder
rather than repeating the work.

With one override: a processor that declares it needs merged source data is
scenario-dependent whatever the config says, because the frames it receives are
filtered per scenario, year and country. A copy from another scenario's folder
would be that scenario's answer wearing this one's name. The code takes the
processor's declaration over the config, and says so when the two disagree.

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
| `src_files/config_*.ini` | `timeseries_specs`, `climate_data`, the window and the forecast branches |

## See also

- [Source workbook conventions](source-workbook-conventions.md) — how the Excel
  files behind the annual figures are read and combined
- `tests/README.md` — the NA and zero boundary map, for anyone changing the
  pipeline rather than the data
