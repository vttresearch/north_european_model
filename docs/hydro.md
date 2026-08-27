# Hydro data

How hydro reaches the model: what the four types mean, which file supplies which
number, what is known to be missing, and what the builder does about it. Two
processors are involved — `hydro_inflow_MAF2019` for inflow and
`hydro_storage_limits_MAF2019` for seasonal storage limits.

This describes what the build does **today**, from the sources available today.
Hydro is the part of this model where the data is least settled: several shapes
in the processors exist only to fit the quirks of whichever database was
available, `hydroUpd` v2 is coming, and this page will change with them. See
[Timeseries](timeseries.md) for the parts shared by every source.

## One minute summary

- **Pan-European data simplifies hydro to four types** — reservoir,
  run-of-river, open-loop and closed-loop pumped storage. The simplification is
  accepted, not believed.
- **`nodedata` decides what exists.** A hydro node absent from it is absent from
  the model, and the build says nothing about it. Both processors gate on that.
- **Seasonal storage limits exist for two of the four types**, and the rest run on a
  flat `upwardLimit` deliberately rather than by mistake. The build names them
  every run.
- **The PECD files have holes, and the builder repairs or refuses them by rule:**
  one missing week or day is interpolated, anything longer needs a person, and
  the decisions already taken are listed in the code with their magnitudes.
- **`capacity` always means unit power in MW.** Reservoir *size* is an energy, in
  MWh, and lives in `nodedata.upwardLimit`.

| Quantity | Comes from | Unit |
|---|---|---|
| reservoir size | `nodedata` `upwardLimit` in `hydroUpd-v1.xlsx` | MWh |
| turbining / pumping power | `unitdata` `capacity_output1` | MW |
| weekly inflow | `PECD-hydro-weekly-inflows.csv` | GWh/week → MWh/h |
| daily run-of-river | `PECD-hydro-daily-ror-generation.csv` | GWh/day → MWh/h |
| seasonal fill limits | `PECD-hydro-weekly-reservoir-levels.csv` | ratio 0–1, scaled to MWh |
| Norwegian psOpen limits | `PEMMDB_NO*_Hydro Inflow_SOR 20.xlsx` | ratio 0–1, scaled to MWh |

## Contents

- [The four types](#the-four-types)
- [Which nodes get built](#which-nodes-get-built)
- [What is not built, and why](#what-is-not-built-and-why)
- [Where the PECD files come from](#where-the-pecd-files-come-from)
- [Gaps in the source data, and what the builder does](#gaps-in-the-source-data-and-what-the-builder-does)
- [Minimum generation](#minimum-generation)
- [Known open items](#known-open-items)
- [Some caveats from cross-checking the data](#some-caveats-from-cross-checking-the-data)
- [Where hydro is defined](#where-hydro-is-defined)

## The four types

Pan-European datasets simplify hydro to four types, and this project follows that
convention:

| Type | Inflow | Storage | Pumping | Grid |
|---|---|---|---|---|
| Reservoir | yes | yes | no | `reservoir` |
| Run-of-river | yes | no | no | `ror` |
| Pumped storage, open loop | yes | yes | yes | `psOpen` |
| Pumped storage, closed loop | no | yes | yes | `psClosed` |

The simplification is accepted, not believed. Run-of-river really does hold
somewhere between a few hours and a couple of days of storage; the convention
gives it none.

**National data does not sit still inside these four boxes.** Norwegian hydro is
psOpen in one database, reservoir in another, and a mixture of psOpen, psClosed
and reservoir in a third. Several shapes in the processors exist only to fit the
quirks of the source that was available, and neither side of that bargain was
written down until this page. If something in `hydro_inflow_MAF2019` or
`hydro_storage_limits_MAF2019` looks arbitrary, that is usually why.

`hydroUpd-v1.xlsx` is a more complete database than PECD in several respects, and
a v2 is coming.

## Which nodes get built

`nodedata` is the statement of what the model has. By the time a processor sees
the frame, the source workflow has already applied its scenario, year and country
filtering, so a node absent from it is absent from the model — not missing, not
broken, and not something to report. Both hydro processors gate on that.

The gate is **presence of the row**, and for inflow it is only presence. Inflow
describes water arriving, which happens whether or not the workbook records a
reservoir size, so an `upwardLimit` left blank or zero by oversight must not
cascade into a node reaching GAMS with no inflow at all. The storage limits are
the opposite case and do additionally require a usable `upwardLimit`, because a
fill limit is a fraction of a size and there is nothing to express without one.
That distinction is the whole reason the two are separate tests.

Before the gate, `hydro_inflow_MAF2019` built the cross product of every country
code and every hydro type and then reported on all of it: 37 lines about 35 nodes
that do not exist, burying the two that do. It now names what it built, and the
only nodes it reports are the ones the model has and the source cannot fill —
`CH00_psOpen` and `FR00_psOpen` on the shipped configuration.

## What is not built, and why

Seasonal fill limits exist for two of the four types:

- **`reservoir`** for every zone with rows in the weekly levels CSV.
- **`psOpen`** for the three Norwegian zones only, from the PEMMDB workbooks.
- **`psOpen` elsewhere, and `psClosed` anywhere: not built.** PECD carries no
  weekly levels for them.

Those nodes are not broken and are not being skipped by mistake. They use the
constant `upwardLimit` from `nodedata`, which is a flat bound rather than a
seasonal profile. `hydro_storage_limits_MAF2019` names them in the build log each
run so their absence from the time series is stated rather than discovered.

How a node ends up on one side or the other is written into the input Excel
rather than inferred. `nodedata`'s `upwardLimit` and `downwardLimit` columns are
constants, and the processor states separately which nodes got a series instead;
the input Excel writes `useTimeseries` for those and `useConstant` for the rest.
Backbone reverses the first decision by itself for a series that turns out to be
flat — `changes.inc` converts it back to a constant, because a constant is much
faster to carry — but nothing reverses it the other way, so the processor's claim
is what makes the seasonal profiles reachable at all.

The **starting level** of each reservoir is not decided here. The input Excel
writes a provisional one, and `changes.inc` then recomputes it for every `psOpen`
and `reservoir` node from the maximum of that node's own upwardLimit series. That
rule needs a rewrite of its own — as written it cannot express a run starting and
ending in summer — so nothing in the build depends on the provisional number
beyond it being above zero.

Whether that is the right treatment is discussed in
[the caveats](#some-caveats-from-cross-checking-the-data), which is also where the
evidence against conjuring a profile for them sits.

## Where the PECD files come from

The three `PECD-hydro-*.csv` files are used **unmodified** from:

> De Felice, M. (2020). *ENTSO-E Hydropower modelling data (PECD) in CSV format*
> (Version 4) [Data set]. Zenodo. <https://zenodo.org/records/3985078>

That dataset ships six CSVs. Three are read here; the other three are not, and
each absence is a decision rather than an oversight:

| File | Status |
|---|---|
| `PECD-hydro-weekly-inflows.csv` | read, weeks 1–52; see [the year change](#the-year-change) |
| `PECD-hydro-daily-ror-generation.csv` | read |
| `PECD-hydro-weekly-reservoir-levels.csv` | read |
| `PECD-hydro-capacities.csv` | **removed** — duplicated `nodedata` and `unitdata` exactly, in GWh where the workbook uses MWh |
| `PECD-hydro-weekly-reservoir-min-max-generation.csv` | **removed** — unusable, see [minimum generation](#minimum-generation) |
| `PECD-hydro-weekly-reservoir-min-max-uniform-levels.csv` | never imported |

Keeping the files byte-identical to the citable source is deliberate. This repo
previously carried a `PECD-hydro-weekly-inflows-corrected.csv`, a hand-patched
copy that filled twenty-six of SE01's missing winter weeks by holding the
previous week's value. Its provenance was not recorded and became unrecoverable;
worse, its one headline edit — SE01 week 6 of 1991, 42.84 → 55 — was appended as
a *duplicate row* rather than a replacement, and the reader keeps the first
occurrence, so that correction never once took effect. Those gaps are now filled
by the rules below, in code, with the reasoning written down.

**`capacity` in this project always means unit power in MW, and nothing else.**
Reservoir *size* is an energy, it lives in `nodedata.upwardLimit`, and it is in
MWh. The two used to be confused because a now-deleted
`PECD-hydro-capacities.csv` held both under one `variable` column, in GWh, and
the storage-limits processor scaled its ratios by that file rather than by the
node's own `upwardLimit` — the same number, maintained twice, with nothing able
to tell whether the two had drifted apart.

## Gaps in the source data, and what the builder does

The PECD files have holes. Left alone they become zero inflow in the shipped
dataset — a zero nobody checks, and one that makes no sense to whoever eventually
finds it. The builder's job is to do the best that can be done with the data and
say so, rather than to leave the same work for every user.

### The rule

Every hydro series is completed **at its own resolution** — weekly for reservoir
and pumped-storage inflow and for the fill limits, daily for run-of-river — and
only then cast to hourly. That order matters: at weekly resolution a missing week
sits one step from its neighbours, while on an hourly axis it is 168 steps and
whether it gets bridged depends on an interpolation limit. Completing first makes
the hourly step mechanical, and makes the grid easy to check before anything
downstream can be affected by it.

Then:

- **A gap of one week or one day is interpolated**, without ceremony. It is a
  repair.
- **A longer run is never interpolated automatically.** Two consecutive missing
  weeks is an invention rather than a repair, so the build warns and someone
  decides. Decisions already taken are recorded in `ACCEPTED_LONG_RUNS` on the
  processor, each with its reason and its magnitude.
- **Anything still missing afterwards warns.**

`ACCEPTED_LONG_RUNS` doubles as a register of where the source data is bad. A
data refresh that introduces a new gap will warn rather than be quietly absorbed.
What is in it today:

| Series | Gap | Decision |
|---|---|---|
| `PL00_psOpen` | up to 9 weeks | ~1.2 GWh/week against ~3400 GWh of weekly Polish demand |
| `CH00_reservoir` | 2 weeks × 4 years | source stops at week 51; ~110 GWh/year, 0.55% |
| `SE02_reservoir` | 2 weeks in 1985 | between 176.6 and 112.2 GWh; ~289 GWh, 0.75% |
| `SE01_reservoir` | winter weeks in 10 years, longest 5 | ~744 GWh over 35 years, ~0.1% of a 20.2 TWh year |
| `NOS0_psOpen` | 5 weeks in 1995 | [neighbouring zones contradict the zeros](#the-norwegian-pump-storage-zeros); ~100 GWh |
| `NON1_psOpen` | 4 weeks, 1984 and 1987 | as above; ~100 GWh |
| `SE04_reservoir downwardLimit` | 2 weeks | the only multi-week zero run in the level data |

### The year change

A climate window does not have to start on 1 January. `bb_timeseries_start` takes
any `MM-DD` and `bb_timeseries_length` any number of days, so a summer-to-summer
or three-year window puts the calendar-year seam **in the middle of a sample**,
where the solver has to absorb whatever is there. With the default `01-01` it sits
at the window edge, where nobody meets it. Both hydro processors used to ship a
step there.

**Inflow: week 53 is not read.** The year is 52 whole weeks and a remainder of one
or two days, and PECD says different things about that remainder depending on the
zone. Ten of the twenty-eight reservoir zones repeat week 52 verbatim, so the cell
carries nothing. AT00 reports the remainder day itself — its week 53 is 0.136 to
0.145 of its week 52 in all 36 years — and dividing that by 168 like a whole week
put a one-day cliff at every New Year, collapsing AT00 inflow from 51.7 to
14.2 MWh/h and back up to 68 by 4 January. Dropping the cell costs at most 0.33%
of one node's annual inflow (AT00_psOpen, 18.8 GWh of 5.67 TWh) and under 0.07%
everywhere else.

What that buys is geometry, not levels: the year change is now one straight
eight-day span, interpolated at the same weekly resolution as the rest of the
year. How far apart week 52 and week 1 are is weather, and it is left alone. It is
also unremarkable — the change exceeds the node's own 95th-percentile weekly
change in **none** of the 34 year changes for AT00, FI00, SE01–04 or the Norwegian
zones. Where it is genuinely large the build says so and changes nothing, which is
the same job [`ACCEPTED_LONG_RUNS`](#the-rule) does for gaps.

**Fill limits: the tail of the pattern is blended.** The weekly ratios are
climatological and get replicated onto every calendar year, so week 52 wraps back
onto week 1 — and the profile is not cyclic. AT00's minimum steps 0.504 → 0.244
there, twenty-six percentage points of reservoir. Week 1 is trusted and never
moved; the blend walks back from week 52 only as far as it must for the step into
week 1 to be no larger than one the profile already makes inside the year, at the
95th percentile so a single outlying week cannot license a seam as large as
itself. On the shipped data five of the twenty-four series need it at all:

| Series | Blend |
|---|---|
| `AT00_reservoir` `downwardLimit` | 2 weeks |
| `NOM1_psOpen` `downwardLimit` | 3 weeks |
| `NOS0_psOpen` `downwardLimit` | 3 weeks |
| `SE02_reservoir` `downwardLimit` | 1 week |
| `SE03_reservoir` `downwardLimit` | 1 week |

The rest, upper bounds included, are already within their own normal variation.

### A recorded zero is a gap

In every hydro series the builder reads, an exact `0` is treated as missing
rather than as a value:

| Series | Why a zero is a gap |
|---|---|
| reservoir inflow | no real zero occurs anywhere in 35 years × 11 zones |
| pumped-storage inflow | see [the Norwegian zeros](#the-norwegian-pump-storage-zeros) |
| run-of-river generation | a river does not stop for exactly one day |
| `upwardLimit` | no zone asks for a reservoir kept empty |
| `downwardLimit` | meaningful in principle; never actually said here |

`downwardLimit` is the one worth explaining, because zero *is* a sensible thing
for it to say: the reservoir may run dry. This source simply never says it. The
whole shipped level dataset contains two zero runs, both in SE04's minimum — one
week at 15, two weeks at 46–47 — and SE04 expresses "essentially empty" constantly
in a different way, with a smallest non-zero of 0.00044 and eleven weeks below
0.01. The lone zero sits between 0.0015 and 0.0145; the pair sits between 0.0268
and 0.0386. A column with that vocabulary would have written 0.0004. They are
dropped values, and they are treated as such.

If a future dataset genuinely means zero here, the multi-week run will warn and
the decision gets made deliberately — which is the point of the register above.

### The Norwegian pump-storage zeros

NOS0 records exactly zero natural inflow for five weeks of 1995, and NON1 for
four weeks across 1984 and 1987. They are interpolated like any other gap, under
the stated exceptions above. The reasoning is recorded once, in
`ACCEPTED_LONG_RUNS`, because the conclusion is not the obvious one and nobody
should have to re-derive it.

Those weeks were genuinely dry. 1995 is NOS0's lowest inflow year of the 35
(62,252 GWh against a 99,101 mean), the weeks fall in February and March, and
every Norwegian zone is far below its seasonal normal at the time. So the obvious
theory — frozen catchment, melt not started — fits.

It fails its own test. If cold drove inflow to nothing it would do so across the
region, and **NOM1 does not record a single zero in 35 years**. In those same
weeks NOM1 sits at 5–25% of its median and NON1 at 17–77%. Cold takes a Norwegian
catchment to a fraction of normal, not to nothing. Nor is it a rounding artefact:
NON1's smallest non-zero is 0.0286 GWh/week, so the source can say "almost
nothing" when it means it.

So they are dropped values in genuinely dry weeks. Interpolating across the
zone's own neighbouring weeks keeps the drought — 13.0 and 16.6 GWh between real
weeks of 9.3 and 20.3, then a smooth ramp of 21.4, 32.1, 42.9 into the melt at
53.6 — and adds roughly 100 GWh to each affected year. Far too little to move any
conclusion about those years; they remain dry and remain difficult. What it
removes is a discontinuity the solver has no reason to be handed.

## Minimum generation

Hydro minimum generation comes from hand-written rows in
`hydroUpd-v1.xlsx :: userconstraintdata`, **not** from PECD.

PECD's `min-max-generation` file was tried and abandoned, and it is worth saying
why so that nobody tries again. Its required minimum generation exceeds the
reservoir inflow of the same zone and week in **24.5% of SE02's weeks and 23% of
SE01's**, with runs of 22 and 20 consecutive weeks — five months during which the
model would be told to generate more than the water arriving. The worst single
week asks for 302.6 GWh against 17.3 GWh of inflow. A constraint like that either
drains the reservoir or makes the model infeasible, and no amount of gap-filling
repairs it. The file has been removed from this repo.

The hand-written rows that replaced it are an early cut of the data published in
[Kiehle et al. (2026)](https://www.sciencedirect.com/science/article/pii/S0306261926009785).
The final dataset from that article is what `hydroUpd` v2 will carry.

## Known open items

- **`hydroUpd` v2.** The article above is out and its final data supersedes the
  early cut currently in `hydroUpd-v1.xlsx`, both for minimum generation and more
  widely. v2 is the intended route for that.
- **Pumped storage outside Norway has no seasonal profile**, and is left that
  way deliberately. Thirteen nodes, 9.6 TWh, run on a flat `upwardLimit` with an
  `Eps` floor. Only `AT00_psOpen` (1.72 TWh) looks like a case where a seasonal
  ceiling would represent something physical; for the rest a flat bound is either
  correct or unknowable. See the caveats below.

## Some caveats from cross-checking the data

Findings from comparing the sources against each other. They are recorded because
each one looked like a fixable defect until the numbers came in.

- **Norwegian psOpen is functionally reservoir, so it cannot validate anything
  about pumped storage.** Pumping is 0% of turbining capacity in NON1, 1.8% in
  NOM1 and 4.4% in NOS0, and their storage-to-annual-inflow ratios (0.41–0.81
  years) sit inside the Nordic reservoir band (0.24–0.72). Continental plants run
  74–100%. Any comparison that uses Norway as the pumped-storage reference is
  comparing reservoirs with reservoirs.

- **Some psOpen nodes are batteries with a hydrological label.** `ES00_psOpen`
  holds **14.4 years** of its own natural inflow; `FR00_psOpen` has **no inflow at
  all**. `PL00_psOpen` receives **81 GWh/year** in total, a median of 1.21 GWh in a
  non-zero week and 0.00135 GWh in its smallest — about 0.008 MWh/h, which at the
  default `rounding_precision` of 0 rounds to nothing in **10 492 hours**, 3.4% of
  the series. That is faithful, not a defect: the source has no blanks for this
  zone, the distribution is smooth, and the rounding costs 0.05% of the annual
  total because it cancels. Polish open-loop pumped storage is effectively closed
  loop. Their state follows electricity prices, not rainfall, and a flat bound
  is the honest representation. Imposing an inflow-shaped season on them would
  add an error rather than remove one.

  It also rules out a per-hour zero alarm here. A district heating node cannot
  have zero demand in June, so for that data "any zero hour is an error" is a
  sound rule; applied to hydro it would fire on ten thousand correct hours every
  build and teach people to ignore warnings. What hydro reports instead is a
  *node* that comes out empty, which is the case that actually means something.

- **An upper bound transfers across a hydrological regime; a corridor width does
  not.** Upper-bound shapes correlate +0.81 to +0.99 among the snowmelt zones and
  −0.73 to −0.83 for Iberian ES00, which is the physically correct inversion.
  Corridor widths correlate at −0.84 to +0.64 — noise. If a bound is ever
  borrowed, it should be the ceiling alone.

- **SE04's level data is unreliable, on four independent counts.** Its only
  isolated zero, its only multi-week zero run, a width profile that correlates
  with nothing, and a reservoir holding just 2.6 weeks of inflow — barely a
  seasonal store at all. The node is small (71.7 GWh), so it has not been worth
  chasing.

**Conclusion: no seasonal bounds are conjured for the nodes that lack them.**
Understanding what European pumped-storage systems physically are took weeks of
work during the article above and ended without a general answer. An assumption
here might happen to be right, but there is not enough knowledge behind it to
tell — and a plausible invented profile is harder to catch later than an
obviously flat one.

## Where hydro is defined

- `src_files/data_files/hydroUpd-v1.xlsx` — `nodedata` (reservoir sizes, spill,
  balance penalties), `unitdata` (turbining and pumping power), and
  `userconstraintdata` (minimum-generation constraints). Listed after
  `TYNDP-2024_National_Trends.xlsx` in the config, so its rows win.
- `src_files/data_files/unittypedata_compilation.xlsx` — maps each hydro
  `Generator_ID` to its unit type, grids and efficiency, which is what turns
  `AT00 / Run-of-River` into unit `AT00_rorTurbine` on `AT00_ror` → `AT00_elec`.
- `src/timeseries/processors/hydro_inflow_MAF2019.py` — inflow for all three
  inflow-bearing types.
- `src/timeseries/processors/hydro_storage_limits_MAF2019.py` — seasonal fill
  limits, scaled by `nodedata.upwardLimit`.

## See also

- [Timeseries](timeseries.md) — the shared pipeline: climate years, windows,
  forecast branches, and what is checked before anything is written
- [Wind and solar timeseries](vre-timeseries.md) — the other PECD-fed part of the
  model, and where a zero is ordinary rather than a gap
- [District heating demand timeseries](dh-demand-timeseries.md) — where a zero
  hour is an alarm, and why the rule differs from this one
- [Source workbook conventions](source-workbook-conventions.md) — how
  `hydroUpd-v1.xlsx` and its neighbours are read and combined
