# Running the model

How a built input folder becomes a solved model: the one command that starts a
run, where its output goes, how many runs can go at once, and how to tell a run
worth trusting from one that merely finished.

This model is installed inside a Backbone checkout, and that checkout — not this
repository — owns running. It knows which GAMS, which solver, and how to keep two
runs from spoiling each other. This page covers only what it cannot know: which
input folders this project builds, and which of its rules bite hardest in this
model.

## One minute summary

- **A run starts from the Backbone checkout, not from here.** `Backbone.gms`
  resolves its includes relatively, so the working directory cannot move.
  `run_model.py` handles that; you never need to `cd ..` yourself.
- **One run, one folder.** Every run writes into `results/<tag>/` — results,
  debug gdx, listing, log, scratch and `warnings.log` together. Runs that share a
  folder overwrite each other's `warnings.log`, which is the first thing you are
  supposed to read.
- **A run is pointed at a built input folder**, the one `build_input_data.py`
  wrote: `python run_model.py input_ObservedTrends_2030`.
- **The unit of parallelism is the input folder, not the run.** Two input folders
  can run at once; two runs of *the same* folder cannot, and the failure is silent.
- **Schedule and investment runs are run differently.** The command is the same
  and the init file differs. The rules on threads, parallel runs and pinning
  below are for schedule runs, the default.
- **A schedule run's speed is in running several at once.** Its solver uses one
  core, so runs side by side, unpinned, finish far more per hour than one at a
  time. Add runs until memory runs low. Holding runs on the performance cores is
  an optional safeguard on top.
- **Never write to `../input` or `../output`.** Those are shared with the Backbone
  repository and with whatever else is running. A run that fails to apply its
  `--output_dir` lands in `../output` without saying so — see *Traps* below.
- **A run that finishes is not a run to trust.** Exit code, then `warnings.log`,
  then the solve status, then the dummies.
- **The `run-*.cmd` files are the by-hand route.** Quick, and strictly one at a time.

## Schedule runs and investment runs

Both start with `run_model.py` on the same built input folder. The init file
makes the difference: `scheduleInit.gms` unless `-- --init_file=<file>` names
another. They differ in almost everything that decides how to run them:

| | Schedule run | Investment run |
|---|---|---|
| Init file | `scheduleInit.gms`, the default, shipped | your own, such as `investInit.gms`; none is shipped yet |
| What is solved | one day after another, each solve looking a horizon ahead | one large solve over the periods the init file picks |
| `--days`, `--forecasts`, horizon, forecast branches | read by `scheduleInit.gms` | not read, unless your init file reads them |
| Solver threads | one, from `1_options.gms` | more: `--threads -1` |
| Several at once | the way to go faster, until memory runs low | each wants several cores and holds its whole problem in memory |
| Performance-core pin | an optional safeguard for a long run | no: it would hold a many-thread solve on a few cores |

The rest of this page is about schedule runs where the two differ.
[Running an investment model](investment.md#running-an-investment-model) is the
investment side.

## Running one case

```
python run_model.py --list
python run_model.py input_ObservedTrends_2030 --year 1998 --days 1
python run_model.py input_ObservedTrends_2030_sens1 --year 2015 --days 7
```

The argument is the input folder a build wrote:
`<output_folder_prefix>_<scenario>_<year>_<build>`, from a config in `src_files/`.
One config can write several — one per scenario, year and build of its
[`variants`](variants.md) — and each is run the same way. A name is looked up in
this project; a full path works too, as long as it is inside the Backbone
checkout, because Backbone reads its input relative to that.

`--list` prints every built input folder here, whether a run currently holds it,
and which config and build wrote it. A folder no config names — a copy made by
hand, say — is listed too, and says so.

A folder that has not been built is refused before anything starts, and so is a
scenario name such as `OT2030`: the answer lists the folders that config builds.
[Building input files](../README.md#building-input-files) covers that half.

Useful flags:

| Flag | What it does |
|---|---|
| `--show` | print the command that would run, and exit. Nothing is launched. |
| `--days`, `--year` | the horizon and the climate year — the two every run sets |
| `--tag` | name the output folder something other than `<input folder>-<year>-<days>d` |
| `--solver`, `--gams` | override what the Backbone checkout would pick |
| `--force` | start even though another run holds this input folder |
| `--cpus` | keep a run on the fast cores — see [Performance cores](#performance-cores-an-optional-safeguard) |
| `--threads` | solver threads for this run — see [Solver threads](#solver-threads) |

Anything else goes to Backbone after a bare `--`:

```
python run_model.py input_ObservedTrends_2030 --year 1985 --days 7 -- --debug=2 --diag=yes
```

The switches this model declares live in `src_files/GAMS_files/changes.inc`
(`modelYear`, `climateYear`, `modelledDays`, `forecasts`) and
`src_files/GAMS_files/modelsInit.gms` (`init_file`). Read the defaults there
rather than from any prose, including this page. For Backbone's own parameters
see `../docs/running-backbone/command-line-parameters.md`.

## Horizon and horizon discount

Schedule runs only: this section and the next are what `scheduleInit.gms` sets.

Each daily solve looks `bb_horizon_weeks` ahead: 52 weeks by default, set in
`src_files/config_*.ini` and written into `scheduleInit.gms` when the input
folder is built, so a change needs a rebuild. Horizons shorter than a year have
worked poorly in year runs, and each extra week costs about half a percent more
solver iterations.

Inside the horizon the objective weighs every hour 1 on the realized day and the
three days after it, then less, falling linearly to 0.95 at the horizon end. That
is Backbone's horizon discount (`../docs/features/horizon-discount.md`), shaped
in `scheduleInit.gms`. It is a solver control, not a valuation:

- **A falling weight orders hours that would otherwise cost the same**, and an LP
  with fewer ties needs fewer iterations. In four-week OT2030 runs from 1 January
  and 1 September the ramp took 4.6–7.7% fewer CPLEX iterations, in every one of
  the 28 solves.
- **Five percent leaves the decisions nearly unchanged.** Over those four weeks
  storage stayed within 40 GWh of the undiscounted run, and prices within
  0.3 EUR/MWh.
- **The three days at 1** keep the daily cycling of pumped hydro, batteries and
  heat storage at full weight.

It needs a Backbone whose CHANGELOG lists the horizon discount; an older checkout
stops with a domain error on `horizonDiscountFlatLength`.

## Forecast branches

How many branches a run has, how long each lasts and how it ends come from the
config and are written into the input folder when it is built:
[Forecast branches](timeseries.md#forecast-branches) has the settings and what
they cost. `--forecasts=<n>` runs the first `n` of the branches built, so
`-- --forecasts=3` on a four-branch folder drops f04 and nothing else, and the
remaining probabilities are rescaled to sum to one.

Every built folder states its branch lengths as `forecastLength`, which needs a
Backbone whose CHANGELOG lists per-branch end methods and lengths; an older
checkout stops with a domain error on `forecastLength`.

## Where the output goes

`results/<tag>/`, holding `results.gdx`, `debug.gdx`, `info.txt`, `warnings.log`,
`Backbone.lst`, `backbone.log` and a `gams_scratch/` of its own. A one-day run
produces roughly 0.8 MB of results and 65 MB of debug gdx; `--debug=2` adds one
gdx per solve, which is how per-solve infeasibilities are inspected and which
grows fast.

Giving each run its own folder is not tidiness. Per-solve debug files are named
from the solve's own time step and cannot be renamed, so two runs sharing a
folder overwrite each other's — and `info.txt` and `warnings.log` are single
files per folder, so the checklist below would be read against the wrong run.

## Running more than one

**Two runs against the same input folder corrupt each other.** Backbone prepares
every run by writing into `<input_dir>/tempFiles/` under fixed names and by
converting `inputData.xlsx` in place. No flag makes those names unique. The
result is a wrong answer rather than an error: both runs complete and both report
numbers.

So:

- Different input folders in parallel — fine. Two builds of one config are two
  folders, so they can run side by side.
- One input folder in parallel — only with a copy of it per concurrent run. An
  input folder here is about 456 MB, which is the real price of going wider.

**For schedule runs, several at once is where the speed is.** A schedule run's
solver uses one core (see [Solver threads](#solver-threads)), so a machine with
several cores can carry several runs. Each one added slows the others a little, because the cores share a
power budget and memory, but together they finish far more per hour than one run
at a time. Leave them unpinned, so the operating system can use every core.

**Memory runs out before the cores do.** Each run holds its model in memory, and
its results so far: up to about a gigabyte for a short OT2030 run, and more for a
year run, whose results build up as it goes. Watch free memory as the runs start
(Task Manager, *Performance*). Once it is gone, Windows moves memory to disk and
every run slows to a crawl, so start no more runs than fit. On a machine with
8 GB, that is only a few.

**An investment run does not follow this.** Its one solve uses several cores,
and it holds the whole problem in memory at once, so a second one beside it
takes from the first. See
[Running an investment model](investment.md#running-an-investment-model).

`run_model.py` takes an advisory lock on the input folder and refuses to start a
second run against it, naming the one that holds it. The lock is advisory on
purpose: it stops the script colliding with itself, which is what a sweep does,
and it cannot see a run launched by a `.cmd` file or from another checkout. When
several people or sessions share this machine, ask before starting a run as well.

For an actual batch, do not write a loop around `run_model.py` — the Backbone
checkout already has the two rungs above it, with resume, a wall-clock budget and
one-toolchain-per-batch enforcement:
`../docs/automation/scripted-and-batch-runs.md`, and
`../.claude/skills/backbone-scenario-runner/` for the experiment-design half.

## Solver threads

`1_options.gms` sets `threads = 1`, and one thread wins in schedule runs only.
Given more, CPLEX races several LP algorithms at once on each daily solve, and
dual simplex, which uses one core, has finished every schedule solve logged here:
the other threads only take cores from runs beside it. One thread does the same
work, with the same iterations and the same objective, in less time.

An investment run is the other case: one large solve, where the solver can put
several threads to work. The setting is shared by both, because both read the
input folder's `1_options.gms`, so an investment run raises it for itself with
`--threads -1`
([Running an investment model](investment.md#running-an-investment-model)).

`--threads N` changes it for one run, over the input folder's own setting. With
`--cpus`, keep it no larger than the processors listed, or the threads that lose
the race crowd the one that finishes, and the run is slower than unpinned. It
writes `option threads = N;` into a mod folder inside the run's own output
folder, passed in the highest `--modN_dir` slot your own arguments leave free. A
later slot wins, so it overrides the input folder and any mod of yours below it.

## Performance cores: an optional safeguard

Many recent processors have two kinds of core: fast performance cores and slower
efficiency cores, where the same solver work takes noticeably longer. The
operating system usually gives a run a fast one, but not reliably: a run started
beside other work, on battery or during a virus scan can land on a slow one. This
section is about long schedule runs. An investment run is not pinned: held on a
few cores, its many-thread solve loses the rest.

`--cpus` holds GAMS and the solver on the logical processors listed, for as long
as the run lasts:

```
python run_model.py input_ObservedTrends_2030 --year 2015 --days 365 --cpus <list>
```

- **A safeguard, not a speed-up.** On a quiet machine an unpinned run already gets
  a fast core, and holding parallel runs on the performance cores adds little to
  what running them in parallel gives. What it buys is that a long run cannot
  slip onto a slow core when something else starts.
- **Fewer pinned runs than performance cores.** A pinned run cannot move. With
  every performance core held, other work lands on the same cores and slows all of
  them, where unpinned runs would move to free ones. Run the rest unpinned.
- **Only with the list `local-setup.txt` records.** Which processors are the fast
  ones is for the operating system to say, and the Backbone checkout's
  `backbone-quickstart` skill has a check that finds them and proves the pin. With
  no list recorded, do not guess one.
- **Check that it took**, once, while the run is going:
  `Get-Process gams, gamscmex | Select-Object Name, ProcessorAffinity` prints the
  processors as a mask, one bit each: 15 is processors 0-3.
- **Processor affinity** is how the pin works: Windows and Linux have it, macOS
  does not. On macOS the flag is refused, as is a processor the machine lacks, both
  before GAMS starts.

On such a machine an unpinned run's wall time includes whichever kind of core it
landed on, so compare runs by solver iterations, not by wall clock.
`../.claude/skills/backbone-scenario-runner/references/pinning-long-runs.md` has
the Backbone checkout's reasoning, and what to do for each thing a machine's
record can say.

## Is it still running?

**Nothing prints while a run is alive, by design.** The log goes to a file, and
one line appears when the run is over. Silence is not a symptom. The liveness
probe is the timestamp of `results/<tag>/backbone.log`, not its contents — a
solver can go minutes without a line.

```
Get-Content -Wait -Tail 20 results/<tag>/backbone.log
```

## Did it work?

A failed solve still writes `results.gdx`, covering fewer time steps than asked
for and marked in no way at all — it reads as a cheaper scenario. Check, in order:

1. **The exit code.** An execution error makes GAMS exit non-zero whatever
   `--debug` is set to.
2. **`warnings.log`, in full.** It is written before any model logic, so a missing
   one is a run that never got that far. `Note:` lines are informational;
   `!!! Warning:` lines are not. In this model a handful of conversion units with
   no efficiency or `conversionCoeff` are deactivated every run — that is this
   build's normal state, not a fault of the run.
3. **Every solve succeeded** — `r_info_solveStatus`, one row per solve, `modelStat`
   1 or 8 and `solveStat` 1.
4. **The cost is plausible** — `r_cost_realizedCost`, in MEUR.
5. **The dummies are zero** — `r_cost_penalty` and the `r_q*` tables. Non-zero
   means the solver bought its way out of an infeasibility. `info.txt` prints the
   five dummy totals at the top, which is the cheapest look.
6. **The horizon is complete** — `info.txt`'s *Last time step for results*.

```
python ../scripts/print_results.py results/<tag> -s "r_info_solveStatus,r_cost_realizedCost,r_cost_penalty"
python ../scripts/print_results.py results/*/ -p r_cost_        # several runs side by side
python ../scripts/compare_gdx.py results/base results/variant   # do two runs agree?
```

The printer shows numbers and never judges them; the comparer gives one verdict
per symbol and an exit code a script can gate on. The traps that give a plausible
wrong number — domain names that lie about direction, MEUR versus EUR, an absent
row meaning zero — are in `../.claude/skills/backbone-result-reader/`.

Dummies that appear only inside the forecast branches do not show up in the
realized `r_q*` tables at all; those live in the debug gdx, and `--debug=2` is
what splits them per solve.

## Traps that have already cost time here

- **A dry run that is not dry.** Confirm `--show` actually printed and exited
  before believing a check was free. A wrapper that mis-parses its own arguments
  can launch a solve while looking like it did nothing.
- **A valueless `--flag` eats the next argument.** GAMS accepts both `--key=value`
  and `--key value`, so `--diag` with no value consumes whatever follows it —
  including the `--output_dir` a runner appends last. Backbone then falls back to
  its default output folder, which is the shared `../output`, and nothing in the
  run's own output says so. Always write pass-through arguments as `--key=value`;
  `run_model.py` refuses them otherwise.
- **The listing file lands in the shared root** on a raw `gams` line that does not
  pass `o=`, where a concurrent run collides with it. `run_model.py` passes it.
- **A misspelled Backbone parameter is not an error.** Backbone falls back to the
  default and the run proceeds configured differently from how the command reads.
  `--show` is how you check the line before trusting a batch of it.

## The `run-*.cmd` files

Four scenario scripts are shared; any other `run-*.cmd` at the root is somebody's
own and stays on their machine. They call `gams` directly and are the quick route
for a run you are watching — **one at a time**, because they all share one input
folder per scenario and one `results/` folder. They end in a bare `cmd` so the log
can be read afterwards, which also means they hang forever under a script. Use
`run_model.py` for anything unattended.

## What the Backbone checkout above adds

Read these by path; they are plain markdown and nothing here duplicates them.

| Where | For |
|---|---|
| `../.claude/skills/backbone-scenario-runner/` | running and varying a model, sweeps, batches |
| `../.claude/skills/backbone-result-reader/` | reading, comparing and plotting results |
| `../.claude/skills/backbone-quickstart/` | install, environment, "why won't it run at all" |
| `../.claude/skills/backbone-model-builder/` | authoring a new input model from scratch |
| `../.claude/skills/backbone-mod-builder/` | adding a constraint, objective term or result table |
| `../.claude/skills/backbone-input-workbook/` | editing an input workbook without Excel |
| `../docs/automation/scripted-and-batch-runs.md` | the same ground as the scenario runner, for a person |
| `../docs/running-backbone/` | command line parameters, schedule and investment runs, solver option files |
| `../docs/dictionary.md`, `../docs/features.md` | the authoritative parameter and feature reference |

Editing anything under `../` needs asking first, and `../input` and `../output`
are never ours to write to.
