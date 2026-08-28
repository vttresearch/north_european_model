# Wind and solar timeseries

Hourly `ts_cf` for onshore wind, offshore wind and solar PV, built by `VRE_PECD`
from a folder of PECD files you download yourself. One processor serves all
three; they differ only in which folder they read and which flow they write.

This describes what the build does **today**, with the PECD releases available
today. The dataset is actively changing under us — 4.1 has been deprecated and
4.2 is not the same numbers — so treat this page as a snapshot of the current
arrangement rather than a fixed contract. See [Timeseries](timeseries.md) for the
parts shared by every source.

## One minute summary

- **The CSV does not say which download it came from.** Roughly ten choices are
  made when fetching, and none is written into the data. The file name is the
  only record, so the build reads it and prints the selection it found.
- **Read [What a download decides](#what-a-download-decides) before fetching new
  data.** It is the part no file records and the part the build cannot check.
- **One folder, one download.** Mixing two produces a series joined at a step of
  tens of percent, with no other symptom. The build warns; overlapping hours it
  refuses outright.
- **Where several PECD zones match a node, the highest-output one wins.** That is
  a modelling decision, not a lookup, and the build prints its size every run.
- **A zero capacity factor is ordinary** — midnight for PV, a calm hour for wind
  — so there is no per-hour zero alarm here, unlike district heating.

| Quantity | Comes from | Unit |
|---|---|---|
| hourly capacity factor | `*.csv` in the folder named by `input_sub_folder` | MW/MW, 0–1 |
| which countries | `country_codes` in `src_files/config_*.ini` | — |
| which climate years | `start_year`, `end_year` | — |
| node name | `country_code` + `_` + `attached_grid` | — |
| flow written | `custom_column_value` `flow` | — |
| mean shift, optional | `scaling_factor` | multiplier, default 1 |

Output is `ts_cf` on flows `PV`, `onshore` and `offshore`, one series per node.
`src_files/timeseries/PECD-PV`, `PECD-onshore` and `PECD-offshore` each hold 40
files, one per climate year 1980–2019.

## Contents

- [What a download decides](#what-a-download-decides)
- [Which zone a node gets](#which-zone-a-node-gets)
- [Which countries can be built](#which-countries-can-be-built)
- [Zeros and holes](#zeros-and-holes)
- [What this does not model](#what-this-does-not-model)
- [Known open items](#known-open-items)
- [Where the VRE timeseries is defined](#where-the-vre-timeseries-is-defined)

## What a download decides

**The CSV does not say which download it came from.** Roughly ten choices are
made when fetching from the PECD portal — technology variant, turbine class,
regridding, physical model, hub height — and none of them is written into the
data. Two files from different downloads carry the *same columns on the same
hourly index* and differ only in their values.

The file name is the only record. The shipped 4.1 files and a 4.2 download of
the same year differ in four of the name's twenty-two fields and in nothing else:

| Field | Shipped (4.1) | A 4.2 download |
|---|---|---|
| `technology_variant` | `NA-` | `COM` |
| `regridding` | `NA---` | `ReGrB` |
| `physical_model` | `PhM02` | `PhM04` |
| `pecd_version` | `PECD4.1` | `PECD4.2` |

So the build reads the name and the file's comment block, and **prints the
selection it found, once per folder, every run**. That line is what makes a
build reproducible from its log, because the file names alone do not fit in one.

### Why it matters more than a version bump

PECD4.1 was deprecated on 29 October 2025, so everyone will be moving to 4.2.
The two are not interchangeable numbers. For the same climate year (2013,
onshore), values move by a median of **+25.5%** across zones:

| | 4.1 → 4.2, 2013 onshore, same column |
|---|---|
| CH00 | +176% |
| SE02 | +75% |
| SE01 | +56% |
| NON1 | +45% |
| NOM1 | +42% |
| NL00 | −7% |
| DKE1 | −3% |

That is a technology-assumption change, not a reanalysis tweak: the 4.2 file is
titled *Onshore Existing technologies* where 4.1 said only *Wind Power Onshore*.

**Neither is wrong.** Which one is right depends on what you are modelling — a
historical 2015 run wants the fleet that existed, a 2040 run wants future
turbines. The build has no way to know which, so it never judges the values. It
only says what it read.

### One folder, one download

Because the selection is currently chosen by pointing a spec at a folder, a
folder holding two downloads is under-specified rather than wrong. The build
**warns and keeps going**:

- Topping up a 4.1 folder with 4.2 files for later years overlaps on no hour at
  all. The series is continuous, complete, and joined from two downloads with a
  step of tens of percent in the middle of it. Nothing else would say so.
- Two files covering the **same** hours are refused outright, and no GDX is
  written. There is no reading under which you meant both, and which one won
  would be decided by the order the folder happened to be listed in. The usual
  cause is a download unpacked into the wrong folder.

A file whose name does not parse is skipped and named. There is deliberately no
"include it in case it is fine": the name is the only record of provenance, so a
name that cannot be read is a file of unknown origin.

## Which zone a node gets

PECD splits countries into zones finer than this model's nodes. `FI00` matches a
column exactly; `FR00` matches nothing, and has to be resolved by prefix —
`FR00` → `FR0` → FR01…FR09.

**Where several zones match, the one with the highest output wins.** That is a
modelling decision rather than a lookup, and it is not a small one:

| | onshore | offshore |
|---|---|---|
| codes resolved by prefix | 10 of 22 | 9 of 20 |
| chosen zone vs the mean of the zones it beat | +2% to +68% | +1% to +42% |
| largest | AT00 +68%, NOS0 +37%, ES00 +32% | ES00 +42%, FR00 +16%, UK00 +13% |

It is kept because capacity is built at good sites rather than average ones, so
the best zone is the closer approximation. It is still an approximation, and the
build prints the chosen zone and the size of the lift every run so that it is a
decision someone can see rather than one buried inside a prefix.

Three consequences worth knowing:

- **The choice is made over the whole configured window**, not from one year.
  It used to come from whichever file sorted first, which meant moving
  `bb_timeseries_start` could silently move French onshore wind by up to 9%:
  FR00 is FR09 in 17 of the 35 years, FR04 in 11 and FR03 in 7, whose 35-year
  means are 0.2838, 0.2808 and 0.2614.
- **A prefix is arithmetic, not a choice.** `FR0` cannot reach FR10–FR15, so six
  of France's fifteen onshore zones are candidates for nothing. Onshore leaves
  10 such zones unreachable (ES10–ES12, FR10–FR15, UKNI), offshore 6, PV 2. The
  build names them.
- **A two-letter match is reported at warning level.** That tier can hand a node
  the weather of a different zone of the same country.

An all-NaN column is excluded before the comparison rather than losing it. The
sum of an empty column is 0.0, which is indistinguishable from a zone that is
genuinely calm — onshore ships 24 such columns in every file, and ES07 sits
inside ES00's candidate pool.

## Which countries can be built

A country code with no matching column produces no `ts_cf` rows at all, and any
unit on that node can never generate for the whole run — downstream that is
indistinguishable from a unit nobody asked for. So the build names what it built
and what it did not.

On the shipped configuration, PV and onshore build all 22 codes. Offshore builds
20: **AT00 and CH00 have no offshore column**, which is correct — and neither has
an offshore unit, so neither is built or mentioned.

**A code is built only if a unit needs it.** `unitdata` carries a `flow` per unit
(from `unittypedata`), and Backbone reads a capacity factor only through such a
unit, so a series for a node with none is inert. The processor asks
`nodes_needing_flow` which nodes have a unit of its flow and builds those. What
that buys is the warning: a code that *does* have a unit and finds no PECD column
is a real problem — someone typed `offshore` where they meant `onshore` — and it
is no longer buried under two countries that are landlocked. Austria's `Offshore
Wind` row is zero capacity with `method: remove`, so the merged unitdata has no
such unit at all.

## Zeros and holes

A zero capacity factor is an ordinary statement — midnight for PV, a calm hour
for wind — so unlike a district heating node there is **no per-hour zero alarm**
here. It would fire on hundreds of correct hours every build and teach everyone
to skip warnings.

What is reported instead is a single empty hour with real generation on both
sides of it, which is what a dropped value looks like. The test is magnitude
aware, and it has to be: the source rounds to five decimals, so a genuine calm
spell produces long runs of values a hair above zero. A rule that ignored
magnitude fires 749 times on CH00 onshore alone, every one of them a zero wedged
between two values that are themselves almost nothing — median 0.0002, never
above 0.037.

Requiring both neighbours to sit at five times the written floor — 0.05 on the
shipped `cutoff_below` — left 6 flagged hours on onshore and 13 on offshore over
1982–2016, and every build reported them. Nobody ever acted on one: a hole
between two hours at five percent of nameplate is weather, not a dropped value.
A warning that fires on correct data every run is not a strict check, it is a
broken one, so the bar is now **half of nameplate on both sides**
(`ISOLATED_DROPOUT_NEIGHBOUR`), which the shipped data does not trip at all.

Half a capacity factor rather than a multiple of the floor, because a multiple
leaves the [0, 1] a capacity factor lives in as soon as the cutoff grows: at
fifty times, a `cutoff_below` of 0.05 would ask for neighbours at 2.5 and the
check would be silently dead.

Which hours count as *empty* does still follow **your** `cutoff_below`, and the
check runs against the values as they will be *written* — after
`rounding_precision` and after the cutoff. Both of those turn small numbers into
zeros, so a check on the unrounded values would answer a question nobody asked.

Nothing is repaired. What these hours actually are is not yet understood well
enough to write a repair rule, and a wrong repair here would be invisible
afterwards.

## What this does not model

Each of these is a limit of the current arrangement rather than a decision to
defend. Any of them would be worth revisiting given better data or a reason.

- **No capacity weighting inside a country.** One zone's profile serves the
  whole node; the other zones' output is discarded rather than blended.
- **No offshore for landlocked countries**, correctly, and none for any country
  PECD has no column for.
- **No technology mix within a node.** One capacity factor per flow per node, so
  old and new turbines in the same country share a profile.
- **No transmission-aware siting.** The best zone wins on output alone,
  regardless of where the demand or the grid is.
- **No correlation check between flows.** PV, onshore and offshore are read from
  three folders independently, and nothing verifies they came from the same
  climate years or the same reanalysis.

## Known open items

- **The move to PECD4.2.** 4.1 is deprecated. The shipped folders are still 4.1,
  and adopting 4.2 is a change of technology assumption rather than a refresh —
  see [above](#why-it-matters-more-than-a-version-bump). Both will exist in the
  wild for a while, which is why the build reports which one it read.
- **The selection belongs to the scenario, not to the installation.** Old-fleet
  profiles for a historical year, future turbines for 2040 — today both are
  chosen by editing `input_sub_folder`, so a multi-year run uses one fleet for
  every scenario year. Making the selection follow the scenario year is intended
  work.
- **Nothing compares the values against anything.** Whether a capacity factor is
  plausible for a country is a question this processor cannot answer; input data
  validation is where it belongs.

## Where the VRE timeseries is defined

- `src_files/timeseries/PECD-PV/`, `PECD-onshore/`, `PECD-offshore/` — the
  downloaded files, used unmodified.
- `src/timeseries/processors/VRE_PECD.py` — the processor, shared by all three
  specs, and the home of the filename field map.
- `src_files/config_*.ini` — the `PV`, `wind_onshore` and `wind_offshore`
  entries in `timeseries_specs`, plus `country_codes`, `rounding_precision` and
  `cutoff_below`.

## See also

- [Timeseries](timeseries.md) — the shared pipeline: climate years, windows,
  forecast branches, and what is checked before anything is written
- [Hydro data](hydro.md) — the other PECD-fed part of the model, and what its
  gaps are repaired with
- [District heating demand timeseries](dh-demand-timeseries.md) — where a zero
  hour *is* an alarm, and why the rule differs from this one
- [Electricity demand timeseries](elec-demand-timeseries.md) — the demand side
  of the same climate years
