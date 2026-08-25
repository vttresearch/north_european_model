# Electricity demand timeseries

Hourly `ts_influx` for the `elec` grid, built by `elec_demand_TYNDP2024` by
scaling a TYNDP 2024 demand profile to each node's annual energy. One profile per
country, 35 climate years in every shipped config, negative MWh/h.

This describes what the build does **today**, with the TYNDP 2024 workbook it
reads today. The choice of source, and the decision to use only its 2030 sheets,
are current positions rather than settled ones — both are open questions below.
See [Timeseries](timeseries.md) for the parts shared by every source.

## One minute summary

- **`TWh/year` is a normal-year demand.** The build matches it as a multi-year
  mean; individual climate years land between 95% and 107% of it. That is the
  point of running 35 of them, not a bug.
- **`Constant_share` is blank for every electricity node, deliberately.** The
  TYNDP profile is already a demand projection, so a separate flat term would
  double-count.
- **A zero hour is a defect.** Electricity demand does not stop, and there is no
  season it could be confused with. The build alarms about any node that comes out
  empty; on the data that ships it says nothing, and the smallest hour anywhere is
  419 MWh/h.
- **Every scenario year reads the 2030 workbook**, 2040 included, because the 2040
  profiles contain negative demand. Read that section before using a 2040 run.
- **The workbook is read once.** It takes over a minute, so everything checkable
  about it is proved before a parquet cache is written and recorded in that
  cache. Later runs read the receipt, not the workbook.

| Quantity | Comes from | Unit |
|---|---|---|
| annual energy per node | `demanddata` `TWh/year` | TWh/a, normal year |
| flat share of that energy | `demanddata` `Constant_share` | 0–1, blank today |
| hourly profile | `src_files/timeseries/elec_2030_National_Trends.xlsx` | one sheet per country |
| which countries are read | `elec_demand_TYNDP2024.ALLOWED_COUNTRIES` | — |

Output is `ts_influx` on grid `elec`, negative, in MWh/h.

## Contents

- [The calculation](#the-calculation)
- [What `TWh/year` means](#what-twhyear-means)
- [Why `Constant_share` is blank](#why-constant_share-is-blank)
- [Which countries and which climate years](#which-countries-and-which-climate-years)
- [Zero hours](#zero-hours)
- [Only the 2030 workbook is used](#only-the-2030-workbook-is-used)
- [The source has no leap years](#the-source-has-no-leap-years)
- [The parquet cache](#the-parquet-cache)
- [What this does not model](#what-this-does-not-model)
- [Where the electricity demand timeseries is defined](#where-the-electricity-demand-timeseries-is-defined)

## The calculation

Two steps, per country, then per node.

1. **Normalise** the country's profile so the *mean* climate year sums to 1.
   Across the whole range, not within each year.

2. **Scale to each node's energy**, split into a profile-driven and a flat part:

   ```
   demand(t) = A · profile(t) + B
   A = TWh/year · 1e6 · (1 − Constant_share)
   B = TWh/year · 1e6 · Constant_share / 8760
   ```

With `Constant_share` blank, `B` is zero and the profile carries everything.

## What `TWh/year` means

**A normal-year figure — not the energy any particular climate year consumes.**
Because the input is a normal year, the built demand matches it as a multi-year
mean while individual years run above or below by weather:

| | share of that node's table figure |
|---|---|
| 35-year mean | 100% |
| coldest year of any node | 106.9% (NOS0) |
| mildest year of any node | 95.5% (NOS0) |

Measured over 1982–2016 from a built `ts_influx_elec_summary.csv`, across the 22
electricity nodes that ship today. Each node spans 2.4% (NL00) to 11.4% (NOS0)
between its own mildest and coldest year — far narrower than district heating's
21%–26%, because only part of electricity demand follows the weather. Every build
prints the range it produced.

Normalising per year instead would force every climate year to the same total and
delete exactly the variation that running 35 of them is for.

## Why `Constant_share` is blank

The TYNDP profile is a projection matched to ENTSO-E demands, used as the baseline
for the scenario years this model runs. It already contains whatever flat
component belongs in it, so adding a separate constant term on top would
double-count.

The case for setting one is a load the projection does not know about — a large
datacenter build-out, say — where a flat addition is the honest way to express it.

## Which countries and which climate years

**Countries.** `ALLOWED_COUNTRIES` in the processor is the list the cache is built
for — a hard gate, not a preference. The workbook holds many more zones, and this
is the set that has been checked by hand. A configured country outside it is
reported at warning level and its node is alarmed about as empty; the rest of the
build is unaffected. Widening the tuple is a deliberate act, and it invalidates
the cache so the new country is actually read.

**Climate years.** Coverage is **ragged, per country and per workbook**. In the
2030 workbook most countries run 1982–2019 while AT00, ES00, FR00 and SE01–SE04
stop at 2016; the 2040 workbook disagrees about which. `climate_data = 1982-2016`
in every shipped config is exactly the intersection.

That raggedness is data, not a defect, so it is judged only against what a run
asks for. A country missing 2017 is nothing to a 1982–2016 build. A country
missing a year *inside* the requested range is not built at all, and is named —
because the alternative is a full year of fabricated zeros that every downstream
check passes.

## Zero hours

Backbone reads a zero as "not set", so a node whose demand is zero for an hour is
indistinguishable from a node whose demand was never built. Every way this
processor can fail produces the second while looking like the first: a country
outside `ALLOWED_COUNTRIES`, a climate year the workbook does not cover, a
`TWh/year` that is not a number.

So the build **alarms about any hour of any node that comes out empty**, naming
the node and the hour count. On the data that ships it says nothing: the smallest
hourly value is 419 MWh/h (LV00, the smallest node at 7.2 TWh/year) and the lowest
hour of any profile is 22% of that profile's mean. Anything it does say is real.

It grades what it finds. A node whose source profile goes negative has to pass
through zero to get there, so those hours are the data saying what it says and are
reported as a warning. Anything else is an error, which counts against the run and
forces a full rerun next build — the right price for a fabricated zero, and the
wrong one for a property of the source that will still be there tomorrow.

The check tests what will actually be *written*: the output is rounded to whole
MWh afterwards, so a node whose every hour is below 0.5 MWh/h would reach GAMS as
nothing at all.

A zero in the *source* is treated the same way. A climate year whose every hour is
zero is a gap the workbook spelled with a number instead of a blank, and it is
refused rather than scaled.

## Only the 2030 workbook is used

Including for 2040. `elec_2040_National_Trends.xlsx` exists and is **deliberately
not read**. Every scenario year takes the 2030 profile *shape* and scales it to
its own `TWh/year`, so a 2040 run gets 2040 magnitudes on 2030 hourly shapes.

The reason is that the 2040 workbook contains hours of **negative** demand, which
the 2030 one does not. A negative demand becomes a *positive* `ts_influx` —
Backbone reads it as generation injected into the node for nothing.

Measured over 1982–2016, scaled to the National Trends 2040 annual figures:

| node | negative hours/year | energy/year | of that node's demand | lowest hour |
|---|---|---|---|---|
| SE04 | 168 | 74.3 GWh | 0.238% | −2 032 MWh/h |
| UK00 | 36 | 68.4 GWh | 0.012% | −8 685 MWh/h |
| DKW1 | <1 | ~0 | 0.000% | −111 MWh/h |

143 GWh/year in total against a system demand of about 3 800 TWh/year — 0.0038%.
Small in aggregate; not nothing for SE04, where it is a quarter of a percent of a
31 TWh zone and reaches the size of a large power plant for a few hundred hours.

The likely reading is that these are **net** demand profiles with embedded
generation already subtracted, so a sunny low-load hour goes below zero. That is
not what the rest of this model assumes a demand series is: the embedded
generation would be counted twice, once inside the profile and once as the units
and VRE capacity attached to the node.

**This is flagged for users rather than settled here.** A negative demand is a red
flag for the approach, there may be a good reason for it, and finding out belongs
with the data's authors. Neither option available without them is honest: clipping
to zero replaces a known quantity with "not set", and clipping to a positive floor
invents 74 GWh/year of demand SE04 does not have. Using the shape that has no
negatives, and saying so, is what the build does instead — and it is the current
answer, not the intended final one.

The trade is that a 2040 run does not get 2040's *shape* — the electrification and
demand-profile changes TYNDP projects between the two years are not represented.
Annual energy, which is where most of the difference lives, is correct.

## The source has no leap years

Every sheet is a standardised **365-day, 8760-hour calendar**. A 366-day year does
not exist anywhere in the data, and everything below follows from that.

A leap year's output is therefore *built* from a non-leap year's input:

- standard day 60 becomes Feb 29
- standard days 61–365 shift to Mar 1 – Dec 30
- standard day 365 is used twice, also filling Dec 31

So a leap year has 8784 output hours drawn from 8760 source hours. Measured on
FI00 in 2016, it receives **0.282%** more energy than its source year holds —
slightly above the 24/8760 the duplication implies, because late December sits
above the annual mean.

The flat term divides by a nominal 8760 for the same reason. Left alone
deliberately: both effects are far inside the precision of the demand projections
themselves. `DH_demand_fromTemperature` uses the identical divisor, so the two
would have to change together.

## The parquet cache

Reading the workbook takes over a minute; reading the cache takes under a second.
That difference is what makes a thorough check affordable: it runs **once**.

The cache sits beside the workbook, named after it, and is keyed on nothing else —
not the config, not the country set, not the climate range — which is what lets
one cache serve every config. It is gitignored.

**Proved before the cache is written**, per sheet: the header row is where it is
read from; the month, day and hour columns hold the ranges they should; every
column that is not one of those three is a climate year, and all of them are
taken; the calendar is 8760 distinct hours and every sheet agrees about it; no
value is negative; and each (country, year) is recorded as complete, partial or
empty. A sheet that fails is excluded and *why* is recorded, so a run asking for
that country is told the reason rather than nothing.

**Recorded in the cache's own schema metadata** — a receipt, so it cannot be
separated from the data it describes: the contract version, the workbook's name,
size and modification time, and the per-country year coverage.

**The cache is rebuilt** when it is missing, carries no readable receipt, was
written by a different contract version, does not account for every name in
`ALLOWED_COUNTRIES`, or when the workbook or this processor has changed since. A
shortfall of climate *years* is never a reason to rebuild — the receipt was
written by a reader that had the workbook open, so re-reading will not produce a
year it says is not there.

**The workbook need not be present** if the cache is trustworthy. That is what
lets a test or demo folder ship a 69 MB cache without a 282 MB workbook; the build
says once that it could not confirm freshness. With neither present, electricity
demand is skipped with a warning and the rest of the build continues.

## What this does not model

Each of these is a limit of the current source and method rather than a decision
to defend. Any of them would be worth revisiting given a better source.

- **No sub-country profiles.** One profile per bidding zone; a country split into
  regions gives each the same shape.
- **No 2040 profile shape**, for the reason above — the largest known gap here.
- **No price response** and no demand-side flexibility here — that belongs to the
  units and storages connected to the node.
- **No efficiency or electrification trend within a climate range.** `TWh/year` is
  per scenario year; the 35 climate years around it all use the same figure.
- **No correlation with the weather that drives VRE beyond what the source
  carries.** The profile's climate year and the wind and solar climate year are
  aligned by label, and the source decides how well they actually correspond.

## Where the electricity demand timeseries is defined

- `src_files/data_files/demanddata_elec_own_projection.xlsx` — `demanddata_elec`
  (annual energy per node and scenario year). `TYNDP-2024_National_Trends.xlsx`
  supplies the same for the National Trends configs.
- `src_files/timeseries/elec_2030_National_Trends.xlsx` — the hourly profiles, one
  sheet per zone, downloaded separately (see the README). `elec_2040_...` may sit
  beside it and is not read.
- `src/timeseries/processors/elec_demand_TYNDP2024.py` — the processor,
  `ALLOWED_COUNTRIES`, and the cache contract.
- `src_files/config_*.ini` — the `electricity demand` entry in `timeseries_specs`,
  plus `country_codes` and `climate_data`.

## See also

- [Timeseries](timeseries.md) — the shared pipeline: climate years, windows,
  forecast branches, and what is checked before anything is written
- [District heating demand timeseries](dh-demand-timeseries.md) — the same
  calculation driven by temperature, and the opposite zero problem
- [Source workbook conventions](source-workbook-conventions.md) — how the demand
  sheets are read and combined
