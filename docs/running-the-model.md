# Running the model

How a built input folder becomes a solved model: the one command that starts a
run, where its output goes, how many runs can go at once, and how to tell a run
worth trusting from one that merely finished.

This model is installed inside a Backbone checkout, and that checkout — not this
repository — owns running. It knows which GAMS, which solver, and how to keep two
runs from spoiling each other. This page covers only what it cannot know: what
the scenarios here are, and which of its rules bite hardest in this model.

## One minute summary

- **A run starts from the Backbone checkout, not from here.** `Backbone.gms`
  resolves its includes relatively, so the working directory cannot move.
  `run_model.py` handles that; you never need to `cd ..` yourself.
- **One run, one folder.** Every run writes into `results/<tag>/` — results,
  debug gdx, listing, log, scratch and `warnings.log` together. Runs that share a
  folder overwrite each other's `warnings.log`, which is the first thing you are
  supposed to read.
- **The unit of parallelism is the input folder, not the run.** Two scenarios can
  run at once; two runs of *the same* scenario cannot, and the failure is silent.
- **Never write to `../input` or `../output`.** Those are shared with the Backbone
  repository and with whatever else is running. A run that fails to apply its
  `--output_dir` lands in `../output` without saying so — see *Traps* below.
- **A run that finishes is not a run to trust.** Exit code, then `warnings.log`,
  then the solve status, then the dummies.
- **The `run-*.cmd` files are the by-hand route.** Quick, and strictly one at a time.

## Running one case

```
python run_model.py --list
python run_model.py OT2030 --year 1998 --days 1
```

`--list` prints every scenario, the input folder it builds to, and whether that
folder is built or currently busy. A scenario is the `<x>` in
`src_files/config_<x>.ini`, and its input folder name is derived from that config
by the same rule the builder uses — so a scenario you can build is a scenario you
can run, with no second list to keep in step.

If the folder has not been built yet, `run_model.py` says so and prints the
`build_input_data.py` command that creates it. [Building input
files](../README.md#building-input-files) covers that half.

Useful flags:

| Flag | What it does |
|---|---|
| `--show` | print the command that would run, and exit. Nothing is launched. |
| `--days`, `--year` | the horizon and the climate year — the two every run sets |
| `--tag` | name the output folder something other than `<scenario>-<year>-<days>d` |
| `--solver`, `--gams` | override what the Backbone checkout would pick |
| `--force` | start even though another run holds this input folder |

Anything else goes to Backbone after a bare `--`:

```
python run_model.py OT2030 --year 1985 --days 7 -- --debug=2 --diag=yes
```

The switches this model declares live in `src_files/GAMS_files/changes.inc`
(`modelYear`, `climateYear`, `modelledDays`, `forecasts`) and
`src_files/GAMS_files/modelsInit.gms` (`init_file`). Read the defaults there
rather than from any prose, including this page. For Backbone's own parameters
see `../docs/running-backbone/command-line-parameters.md`.

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

- Different scenarios in parallel — fine, they have different input folders.
- Same scenario in parallel — only with a copy of the input folder per concurrent
  run. An input folder here is about 456 MB, which is the real price of going
  wider.

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
