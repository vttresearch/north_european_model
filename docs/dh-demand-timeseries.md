# District heating demand timeseries

Hourly `ts_influx` for the `dheat` grid, built from outdoor temperature by
`DH_demand_fromTemperature`. The calculation is short enough to state in full,
and everything surprising about the result follows from how simple it is.

This describes what the build does **today**. The temperature-driven approach is
the current one rather than a settled one — it is crude on purpose, and a better
source or a better method would replace it and this page together. See
[Timeseries](timeseries.md) for the parts shared by every source.

## One minute summary

- **`TWh/year` in the input data is the demand for a weather-normalised year.** The
  build matches it as a multi-year mean; each climate year lands above or below.
  Getting this wrong biases every scenario built from that row, and nothing
  downstream can detect it.
- **`Constant_share` is iterated to 0.3** based on how well the formula repeats annual demands and known public hourly profiles.
- **Four steps, no more:** smooth the temperature over a 24h window, subtract it from a
  balance point, normalise, scale to each node's annual energy.
- **A zero hour is a defect, not a summer.** Hot water and network losses do not
  stop, so the build alarms about any node that comes out empty.
- **The physics is one national temperature per country.** No per-city weather,
  no hot-water seasonality, no wind or solar gain.
- **Only countries `Temperature.csv` covers can be built**, and the column is
  found by the country code's first two letters.

| Quantity | Comes from | Unit |
|---|---|---|
| annual energy per node | `demanddata` `TWh/year` in `demanddata_DH_own_projection.xlsx` | TWh/a, normal year |
| flat share of that energy | `demanddata` `Constant_share` | 0–1 |
| hourly outdoor temperature | `src_files/timeseries/Temperature.csv` | °C, one column per country |
| balance point | `DH_demand_fromTemperature.BALANCE_POINT_C` | °C |
| smoothing window | `DH_demand_fromTemperature.SMOOTHING_HOURS` | hours |

Output is `ts_influx` on grid `dheat`, negative, in MWh/h.

## Contents

- [The calculation](#the-calculation)
- [What `TWh/year` means](#what-twhyear-means)
- [Which countries can be built](#which-countries-can-be-built)
- [Constant share](#Constant-share)
- [Leap years](#leap-years)
- [Node names and exclusions](#node-names-and-exclusions)
- [What this does not model](#what-this-does-not-model)
- [Where the district heating timeseries is defined](#where-the-district-heating-timeseries-is-defined)

## The calculation

Four steps, per country, then per node.

1. **Smooth the temperature.** A trailing mean over `SMOOTHING_HOURS` = 24 hours.
   This is the entire building-physics model: a crude stand-in for thermal mass,
   so that demand responds to yesterday's weather rather than to this instant's.

2. **Subtract from the balance point.** `profile = max(0, 17 °C − mean)`. Above
   17 °C no space heating is needed and the weather-driven term is zero.
   `BALANCE_POINT_C` = 17.0 is a modelling choice, not a tuning constant — it and
   the window between them decide the shape of every heat demand series in the
   model.

3. **Normalise** so that the *mean* climate year sums to 1. Across the whole
   range of climate years, not within each one. See below.

4. **Scale to each node's energy**, splitting it into a weather-driven and a flat
   part:

   ```
   demand(t) = A · profile(t) + B
   A = TWh/year · 1e6 · (1 − Constant_share)
   B = TWh/year · 1e6 · Constant_share / 8760
   ```

   `B` is hot water, network losses and everything else that does not care what
   the weather is doing. Every node shipped today uses `Constant_share = 0.3`.

Several nodes can share one country's profile: the eight Finnish heat nodes all
use the same national temperature series and differ only in `TWh/year`.

## What `TWh/year` means

**A weather-normalised "normal year" figure — not the energy any particular year
consumed.** That holds for a historical row too: a 2015 entry must be a
weather-corrected 2015 demand, not what the meters recorded in 2015.

This is the one thing worth getting right before typing a number into the demand
table, because nothing downstream can detect a realised figure entered by
mistake. It would simply bias every scenario built from that row.

The normalisation follows from it. Because the input is a normal year, the built
demand matches it as a **multi-year mean** while individual climate years run
above or below by weather:

| | share of that node's table figure |
|---|---|
| 35-year mean | 100% |
| that node's coldest year | 111% – 114% |
| that node's mildest year | 88% – 92% |

Measured over 1982–2016 across the eighteen heat nodes that ship today. Each node
spans 21.5% (AT00) to 26.0% (DE00) between its own mildest and coldest year. So
**a single-climate-year build is not meant to reproduce `TWh/year`**, and finding
that it does not is the expected result rather than a bug. Every build prints the
range it produced.

Normalising per year instead would force every climate year to the same total and
delete exactly the variation that running 35 of them is for.

## Which countries can be built

The temperature column for a country is its code's **first two letters** —
`FI00` → `FI`, `NOS0` → `NO`, `DKW1` → `DK`. A rule rather than a lookup table,
so that splitting a country into regions costs nothing: `EE00` becoming `EE01`
and `EE02` needs no code change at all.

`Temperature.csv` carries AT, BE, CH, DE, DK, EE, ES, FI, FR, GB, LT, LV, NL, NO,
PL, SE. Two consequences:

- **`UK00` finds no column**, because the file says `GB`. There is deliberately no
  alias table: an alias is a modelling assumption the reader cannot see, and it
  would not help `ITN1`, `ITCN`, `ITCS` or `PT00`, which have no temperature data
  under any name. A demand row for such a country is reported at warning level,
  its node is alarmed about as empty, and the rest of the build is unaffected.
- A country with no demand rows at all is normal — most of a run's countries have
  no district heating — and is reported at info level, never as a warning.

Widening this means finding temperature data for the missing countries, not
editing the rule.

## Constant share

For part of every year the 24-hour mean sits at or above 17 °C, and the
weather-driven term is exactly zero. Measured over 1982–2016:

| | share of hours with a zero weather term |
|---|---|
| NO | 2.1% |
| GB | 6.0% |
| FI | 6.8% |
| SE | 7.4% |
| DK | 9.1% |
| EE | 9.8% |
| LV | 10.4% |
| LT | 11.9% |
| NL | 14.3% |
| BE | 14.7% |
| DE | 16.1% |
| AT | 16.3% |
| CH | 17.5% |
| PL | 19.1% |
| FR | 22.4% |
| ES | 39.3% |

With `Constant_share = 0.3` those hours sit at 30% of the node's average demand,
which is where hot water and losses belong. With `Constant_share` blank or zero
they would be **exactly zero**, for a fifth of the year in Poland.

That matters more here than anywhere else in the pipeline, because Backbone reads
a zero as "not set". A node whose demand is zero for July is indistinguishable
from a node whose demand was never built — and every way this processor can fail
produces the second while looking like the first.

So the build **alarms about any hour of any node that comes out empty**, naming
the node and the hour count. The check passes on the data that ships today: there are no zero demand hours. The smallest hourly value is 42 MWh/h.

The check tests what will actually be *written*, not what the processor holds: the
output is rounded to whole MWh afterwards, so a node whose every hour is below
0.5 MWh/h would reach GAMS as nothing at all.

## Leap years

The flat term divides by a nominal 8760 hours regardless of how long the year
actually is, so a leap year receives 24 hours' worth more of its constant part
than nominal — with `Constant_share = 0.3`, about 0.02% of the annual total.

Left alone deliberately, and worth stating alongside the electricity case, which
errs the same way for a different reason: the TYNDP source has no leap years at
all, so a leap year's output is built by reusing a standardised 365-day year and
comes out about 0.28% high. See
[Electricity demand timeseries](elec-demand-timeseries.md#the-source-has-no-leap-years).
Both are far inside the precision of the demand projections themselves.
`elec_demand_TYNDP2024` divides by the same constant, so the two would have to
change together.

## Node names and exclusions

A node is `country_grid[_node_suffix]`, built from three separate cells. A wrong
cell therefore does not produce an error — it produces a *different node*, spelled
plausibly, which quietly takes the demand meant for another one.

`exclude_nodes` matches the **whole node string**, case-insensitively. So
`'NOS0_dheat'` excludes `NOS0_dheat` and does **not** exclude `NOS0_dheat_HKI`.

The builder reports a heat node that appears in `demanddata` but not in
`nodedata`, or the reverse, for any grid the other table describes. That is what
catches a mistyped country cell, and it is why a one-off industrial node needs a
row in both.

## What this does not model

Each of these is a known limit of the current method rather than a decision to
defend. Any of them would be worth revisiting given a better source.

- **No hot water seasonality.** The flat share is flat: same in January as in July.
- **No per-city temperature.** The eight Finnish nodes share one national series,
  so Helsinki and Rovaniemi have identically shaped demand and differ only in size.
- **No wind, humidity or solar gain.** Dry-bulb temperature only.
- **No behavioural response** to price, and no demand-side flexibility here — that
  belongs to the units and storages connected to the node.
- **No efficiency trend within a climate range.** `TWh/year` is per scenario year;
  the 35 climate years around it all use the same figure.
- **No measured load data anywhere in the chain.** The whole series is derived
  from temperature and one annual figure, and nothing validates it against what
  a network actually consumed.

## Where the district heating timeseries is defined

- `src_files/data_files/demanddata_DH_own_projection.xlsx` — `demanddata_dh`
  (annual energy and constant share per node and scenario year) and `nodedata`
  (balance penalties, spill). The per-country sheets behind it are working
  material; only the sheets whose names start with `demanddata` or `nodedata` are
  read.
- `src_files/timeseries/Temperature.csv` — hourly temperature, 1980-01-01 to
  2019-12-31, one column per country.
- `src/timeseries/processors/DH_demand_fromTemperature.py` — the processor, and
  the home of the two modelling constants.
- `src_files/config_*.ini` — the `District heating demand` entry in
  `timeseries_specs`, plus `country_codes` and `exclude_nodes`.

## See also

- [Timeseries](timeseries.md) — the shared pipeline: climate years, windows,
  forecast branches, and what is checked before anything is written
- [Electricity demand timeseries](elec-demand-timeseries.md) — the same
  calculation driven by a TYNDP profile instead of temperature
- [Source workbook conventions](source-workbook-conventions.md) — how the demand
  sheets are read and combined
- [Wind and solar timeseries](vre-timeseries.md) — where a zero is ordinary rather
  than an alarm, and why the rule differs from this one
