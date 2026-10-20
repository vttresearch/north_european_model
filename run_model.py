"""
Run one built input folder through the Backbone checkout this model is installed in.

    python run_model.py input_ObservedTrends_2030 --year 1998 --days 1
    python run_model.py input_ObservedTrends_2030_sens1 --year 2015 --days 7 --show
    python run_model.py --list

The argument is the folder `build_input_data.py` wrote, so every build of a config
-- every scenario, year and build of its `variants` -- is run the same way.

This is a worked example as much as a tool: it is the shortest correct way to
start a run here, and the place to copy from when writing a sweep. Everything it
does that a hand-written command line would not is below.


WHAT IT DELEGATES, AND WHY

Nothing here builds a `gams` command line. The input folder, the climate year and
the horizon are this project's business; which GAMS, which solver, and how a run is
isolated from every other run are the parent checkout's, and it answers them in
`scripts/run_backbone.py`. This file translates the first into arguments for the
second and hands over.

That matters because several things decide whether two runs are comparable, and a
hand-written wrapper usually carries none of them. `run_backbone.py` pins them per
run -- `scrdir`, `-optdir`, the listing file, the log, and the result and debug
gdx -- and creates nested output paths, which a raw `gams` line cannot: Backbone
makes the output folder with a single mkdir, so it can only add one level.
`../docs/automation/scripted-and-batch-runs.md` is the reasoning in full.

The dependency is real and one-directional: this model cannot run without the
Backbone checkout above it, and nothing above it knows this model exists. The root
is found by walking up for `Backbone.gms` rather than by a configured path, so a
checkout that has been moved or renamed still works. If the import fails, this
script says which part is missing instead of falling back to a command line of
its own.


THE LIMIT THAT IS NOT OBVIOUS

Backbone prepares every run by writing into `<input_dir>/tempFiles/` under fixed
names, and by converting `inputData.xlsx` in place. So the unit of parallelism
here is the input folder, not the run:

  - Two runs against *different* input folders are safe to overlap.
  - Two runs against the *same* input folder are not, and the failure is silent:
    each overwrites the other's prepared input, and both report numbers.

Backbone also reads `--input_dir` relative to its own checkout, so an input folder
has to sit inside it; one outside is refused here, before anything starts.

No flag makes those names unique; going wider needs a copy of the input folder,
which is ~450 MB. The lock this script takes covers exactly that -- it stops
`run_model.py` colliding with itself, which is what a sweep would otherwise do.
It is advisory, and cannot see a run started by `run-*.cmd` or from another
checkout. That is what `--force` is for, and why "is anyone else running?" is
still worth asking out loud.


WHAT VARIES A RUN

The switches this model declares are in `src_files/GAMS_files/changes.inc`
(modelYear, climateYear, modelledDays, forecasts) and
`src_files/GAMS_files/modelsInit.gms` (init_file). Those files state their own
defaults and this one deliberately does not repeat them, because a second copy
goes stale without anyone noticing. `--year` and `--days` are surfaced here
because every run sets them; anything else goes through after a bare `--`:

    python run_model.py input_ObservedTrends_2030 --year 1998 --days 7 -- --debug=2 --diag=yes

`../.claude/skills/backbone-scenario-runner/` covers the four ways a Backbone run
can be varied, and `../docs/running-backbone/command-line-parameters.md` the
parameters themselves.


NO SOLVER IS NAMED HERE

A solver is a fact about the machine, and machine facts stay out of tracked
files. Left alone, the parent resolves one and prints which it picked; `--solver`
and the BB_SOLVER environment variable override it. This is a deliberate
difference from the `run-*.cmd` files, which hardcode one.


THREADS, PARALLEL RUNS AND THE PERFORMANCE CORES

Schedule and investment runs are run differently; the init file decides which a
run is. The input folder's `1_options.gms` gives the solver one thread, which wins
in schedule runs only: dual simplex, which uses one core, finishes every daily
solve. So a schedule run's speed is in running several input folders at once,
unpinned, until memory runs low.

`--threads` sets the solver's threads for one run, over that setting. An
investment run, one large solve, can use more and should get them, unpinned
(`docs/investment.md`). Never set it above the processors `--cpus` lists: CPLEX
then races LP algorithms on more threads than cores, and the losing ones slow the
one that finishes. The setting is a mod
folder in this run's output folder whose `1_options.gms` sets `option threads`,
in the highest `--modN_dir` slot the caller left free -- the same layer the
parent's runner and performance-core probe use.

`--cpus` is the parent's own flag, passed through, and an optional safeguard for
schedule runs: it holds GAMS and the solver on the listed logical processors, so
a long run cannot slip onto an efficiency core. Fewer pinned runs than performance cores, and only
with the list `local-setup.txt` records.
`../.claude/skills/backbone-scenario-runner/references/pinning-long-runs.md` is
the parent's reasoning.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent

# This project first, so its own `src` and `tests` win over the parent's
# same-named folders; the Backbone root is appended later, where `scripts` is the
# only thing wanted from it.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

LOCK_NAME = ".run_model.lock"
STALE_HOURS = 24
THREADS_LAYER = "solver-threads"
SCHEDULE_INIT = "scheduleInit.gms"
# Backbone reads --mod1_dir to --mod5_dir, and a later slot's 1_options.gms wins.
MOD_DIR_SLOTS = 5


def find_backbone_root(start: Path) -> Path:
    """The Backbone checkout this model sits in: nearest parent holding Backbone.gms.

    Walking up beats a configured path because the run has to happen from that
    folder anyway -- Backbone.gms resolves its `inc/` includes relatively, so the
    working directory cannot move.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "Backbone.gms").is_file():
            return candidate
    raise SystemExit(
        f"No Backbone.gms found in {start} or any folder above it.\n"
        "This model runs inside a Backbone checkout; without one there is "
        "nothing to run it with."
    )


def import_runner(bb_root: Path):
    """The parent's run_backbone module, or a message naming what is missing."""
    if str(bb_root) not in sys.path:
        sys.path.append(str(bb_root))
    try:
        from scripts import run_backbone
    except ImportError as exc:
        raise SystemExit(
            f"Found the Backbone checkout at {bb_root}, but could not import "
            f"scripts/run_backbone.py from it: {exc}\n"
            "That file is the supported way to start a run. If it is genuinely "
            "absent, the checkout is older than this script expects."
        )
    return run_backbone


def is_built(folder: Path) -> bool:
    """Whether `folder` is a built input folder: one holding the inputData.xlsx a build writes."""
    return folder.is_dir() and (folder / "inputData.xlsx").is_file()


def config_folders(here: Path = HERE) -> dict:
    """{input folder name: (config file name, build name)} for every config in src_files/.

    By the builder's own naming rule, so a folder a config writes is recognised as
    that config's. A config that does not load names no folder; --list and the
    messages below are hints, and one broken config must not stop them.
    """
    import src.infrastructure.config_reader as config_reader

    known = {}
    for config_file in sorted((here / "src_files").glob("config_*.ini")):
        try:
            config = config_reader.load_config(config_file)
        except Exception:
            continue
        for scenario, year, build in product(config["scenarios"], config["scenario_years"],
                                             config_reader.selected_builds(config)):
            name = config_reader.output_folder_name(
                config["output_folder_prefix"], scenario, year, build)
            known.setdefault(name, (config_file.name, build))
    return known


def resolve_input_folder(name: str, here: Path = HERE) -> Path:
    """The built input folder `name` names: relative to this project, or a full path.

    A name that is no folder but a config's scenario (`OT2030`, the <x> of
    src_files/config_<x>.ini) is answered with the folders that config builds, since
    one config can build several.
    """
    folder = Path(name)
    if not folder.is_absolute():
        folder = here / folder
    if is_built(folder):
        return folder.resolve()
    if folder.is_dir():
        raise SystemExit(
            f"{folder} holds no inputData.xlsx, so it is not a built input folder. "
            f"Build it with build_input_data.py first."
        )

    config = here / "src_files" / f"config_{name}.ini"
    if config.is_file():
        builds = [n for n, (c, _) in config_folders(here).items() if c == config.name]
        listed = "\n  ".join(
            f"{n}{'' if is_built(here / n) else '  (not built)'}" for n in builds)
        raise SystemExit(
            f"run_model.py runs an input folder, not a scenario. {config.name} builds:\n"
            f"  {listed}\n"
            f"Run one of them, e.g.: python run_model.py {builds[0] if builds else '<folder>'}"
        )
    raise SystemExit(
        f"No input folder '{name}' in {here}. `python run_model.py --list` shows the "
        f"built ones."
    )


def check_inside(folder: Path, bb_root: Path) -> Path:
    """`folder`, refused when it is outside the Backbone checkout.

    Backbone takes --input_dir relative to its own root, so a folder outside it
    cannot be named at all.
    """
    try:
        folder.resolve().relative_to(bb_root.resolve())
    except ValueError:
        raise SystemExit(
            f"{folder} is outside the Backbone checkout at {bb_root}. Backbone reads "
            f"its input relative to that checkout, so build or copy the folder inside it."
        )
    return folder


def default_tag(folder: Path, year, days) -> str:
    """The results/ folder name a run gets unless --tag says otherwise."""
    return f"{folder.name}-{year}-{days}d"


def read_lock(folder: Path):
    lock = folder / LOCK_NAME
    if not lock.is_file():
        return None
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # An unreadable lock is still a lock: report it rather than step past it.
        return {"tag": "unreadable", "pid": "unknown", "started": "unknown"}


def take_lock(folder: Path, tag: str, force: bool) -> Path:
    """Claim this input folder, or explain who holds it.

    Advisory by construction -- see the module docstring. It catches the case this
    script can catch: a second run_model.py, which is what a sweep is.
    """
    held = read_lock(folder)
    if held and not force:
        age = ""
        try:
            started = datetime.fromisoformat(held["started"])
            hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
            age = f", started {hours:.1f} h ago"
            if hours > STALE_HOURS:
                age += (f" -- over {STALE_HOURS} h, so it may be left over from "
                        "a killed run")
        except (KeyError, ValueError, TypeError):
            pass
        raise SystemExit(
            f"{folder.name} is already being run by '{held.get('tag')}' "
            f"(pid {held.get('pid')}{age}).\n"
            "Two runs against one input folder overwrite each other's prepared "
            "input and both still report numbers, so this one has not started.\n"
            "Wait for it, run a different input folder, or pass --force if you know "
            "that run is gone."
        )
    lock = folder / LOCK_NAME
    lock.write_text(json.dumps({
        "tag": tag,
        "pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")
    return lock


def free_mod_slot(backbone_args) -> int:
    """The highest --modN_dir slot the caller's own arguments leave free.

    The highest, because Backbone includes the slots' 1_options.gms in order and
    the last one wins: below a caller's own mod, that mod could undo the cap.
    """
    taken = set()
    for a in backbone_args:
        m = re.match(r"--mod(\d)_dir=", a)
        if m:
            taken.add(int(m.group(1)))
    free = next((n for n in range(MOD_DIR_SLOTS, 0, -1) if n not in taken), None)
    if free is None:
        raise SystemExit(
            f"--threads needs a --modN_dir slot, and all {MOD_DIR_SLOTS} are taken by "
            "the arguments after --. Drop one of those mods, or set threads in one of them."
        )
    return free


def threads_note(args):
    """A line for a run on another init file that keeps the one solver thread, or None.

    Schedule and investment runs read the same 1_options.gms, and its one thread
    wins in schedule runs only, so an investment run left on it solves its one
    large problem on a single core. The init file is the only sign of which a run
    is, and a schedule init file of another name is possible too, hence "if".
    """
    init = next((a.split("=", 1)[1] for a in args.backbone_args
                 if a.startswith("--init_file=")), SCHEDULE_INIT)
    if args.threads is not None or Path(init.strip("'\"")).name.lower() == SCHEDULE_INIT.lower():
        return None
    return (f"[note] --init_file={init}: 1_options.gms gives every run one solver thread, "
            f"which wins in schedule runs only. If this is an investment run, give it more "
            f"with --threads -1 (docs/investment.md).")


def write_threads_layer(layer: Path, threads: int) -> Path:
    """The mod folder whose 1_options.gms caps the solver at `threads`."""
    layer.mkdir(parents=True, exist_ok=True)
    (layer / "1_options.gms").write_text(f"option threads = {threads};\n", encoding="utf-8")
    return layer


def build_argv(args, bb_root: Path, input_folder: Path, output_dir: Path,
               threads_layer: Path = None):
    """The run_backbone.py command line this run becomes."""
    def rel(path: Path) -> str:
        return path.relative_to(bb_root).as_posix()

    argv = ["--output-dir", rel(output_dir)]
    if args.gams:
        argv += ["--gams", args.gams]
    if args.solver:
        argv += ["--solver", args.solver]
    if args.cpus:
        argv += ["--cpus", args.cpus]
    if args.show:
        argv += ["--show"]
    # Anything the caller set explicitly after -- wins, so the same key is never
    # passed twice: GAMS takes the last occurrence, which reads as though the
    # first one was ignored.
    overridden = {a.split("=", 1)[0] for a in args.backbone_args if "=" in a}
    defaults = [
        f"--input_dir={rel(input_folder)}",
        "--input_file_excel=inputData.xlsx",
        f"--climateYear={args.year}",
        f"--modelledDays={args.days}",
        f"--debug={args.debug}",
    ]
    argv += ["--"] + [d for d in defaults if d.split("=", 1)[0] not in overridden]
    argv += list(args.backbone_args)
    if threads_layer is not None:
        argv.append(f"--mod{free_mod_slot(args.backbone_args)}_dir={rel(threads_layer)}")
    return argv


def split_passthrough(argv):
    """Split this script's own arguments from Backbone's, at the first bare --.

    argparse.REMAINDER cannot do this job. Placed after a positional it swallows
    every later token -- including this script's own options -- and forwards them
    to GAMS, where they are not inert: see check_passthrough below.
    """
    argv = list(argv)
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], [a for a in argv[cut + 1:] if a != "--"]
    return argv, []


def check_passthrough(passthrough):
    """Refuse a valueless --flag, which GAMS would resolve by eating the next one.

    GAMS accepts both `--key=value` and `--key value`, so a double-dash argument
    with no value takes whatever follows it as its value. Passed through to the
    end of the line, that is the flag the runner appends next -- `--output_dir`.
    The run then writes where Backbone defaults to instead, which is a shared
    folder, and nothing in its output says so.
    """
    bad = [a for a in passthrough if a.startswith("--") and "=" not in a]
    if bad:
        raise SystemExit(
            f"Pass-through arguments need a value: {', '.join(bad)}. "
            "GAMS reads a valueless --flag as taking the next argument as its "
            "value, which would swallow the output folder this run was given. "
            "Write them as --key=value."
        )
    return list(passthrough)


def do_list(here: Path = HERE) -> int:
    """Every built input folder here, whether a run holds it, and which config wrote it."""
    folders = sorted(p for p in here.iterdir() if is_built(p))
    if not folders:
        print(f"No built input folder in {here}. Build one with:\n"
              f"  python build_input_data.py src_files config_<x>.ini")
        return 0
    known = config_folders(here)
    for folder in folders:
        held = read_lock(folder)
        state = f"BUSY ({held.get('tag')})" if held else "free"
        config, build = known.get(folder.name, (None, ""))
        source = (f"{config}" + (f", build '{build}'" if build else "")
                  if config else "no config in src_files/ names it")
        print(f"{folder.name:44} {state:22} {source}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python run_model.py",
        description="Run this model through the Backbone checkout it sits in.",
        epilog="Backbone's own arguments go last, after a bare --, for example: "
               "-- --debug=2 --diag=yes",
    )
    parser.add_argument("input_folder", nargs="?",
                        help="The built input folder to run, e.g. input_ObservedTrends_2030: "
                             "a name in this project, or a path inside the Backbone checkout")
    parser.add_argument("--year", "--climate-year", dest="year", type=int, default=1998,
                        help="Climate year, passed as --climateYear (default: 1998)")
    parser.add_argument("--days", type=int, default=1,
                        help="Days to model, passed as --modelledDays (default: 1)")
    parser.add_argument("--tag", default=None,
                        help="Name for this run's folder under results/ "
                             "(default: <input folder>-<year>-<days>d)")
    parser.add_argument("--debug", default="1",
                        help="Backbone --debug level (default: 1, which writes the "
                             "debug gdx; 2 writes one gdx per solve)")
    parser.add_argument("--solver", default=None,
                        help="Solver to use. Default: whichever the Backbone checkout "
                             "resolves, which it prints.")
    parser.add_argument("--gams", default=None,
                        help="GAMS to use: a path, or an alias such as '47'")
    parser.add_argument("--cpus", default=None, metavar="LIST",
                        help="Optional safeguard for schedule runs: hold GAMS and the "
                             "solver on these logical processors, e.g. 0-3, so a long run "
                             "stays on the fast cores. Fewer pinned runs than performance "
                             "cores, with the list local-setup.txt records. Not for "
                             "investment runs.")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="Solver threads for this run, over the input folder's "
                             "1_options.gms, which sets 1 for schedule runs. An investment "
                             "run can take more, e.g. -1. With --cpus, no more than the "
                             "processors listed")
    parser.add_argument("--show", action="store_true",
                        help="Print the command that would run, and exit")
    parser.add_argument("--force", action="store_true",
                        help="Start even though another run holds this input folder")
    parser.add_argument("--list", action="store_true",
                        help="List the built input folders, whether a run holds each, "
                             "and which config wrote it")
    own, passthrough = split_passthrough(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(own)
    args.backbone_args = check_passthrough(passthrough)

    if args.list:
        return do_list()
    if not args.input_folder:
        parser.error("name an input folder, or pass --list to see the built ones")

    bb_root = find_backbone_root(HERE)
    input_folder = check_inside(resolve_input_folder(args.input_folder), bb_root)
    runner = import_runner(bb_root)

    tag = args.tag or default_tag(input_folder, args.year, args.days)
    output_dir = HERE / "results" / tag

    layer = output_dir / THREADS_LAYER if args.threads is not None else None
    runner_argv = build_argv(args, bb_root, input_folder, output_dir, layer)
    note = threads_note(args)
    if note:
        print(note, file=sys.stderr)

    if args.show:
        code = runner.main(runner_argv)
        if layer is not None:
            print(f"[note] --threads {args.threads}: {layer} is written when the run "
                  "starts, not by --show.", file=sys.stderr)
        return code

    lock = take_lock(input_folder, tag, args.force)
    started = time.time()
    try:
        if layer is not None:
            write_threads_layer(layer, args.threads)
        code = runner.main(runner_argv)
    finally:
        lock.unlink(missing_ok=True)

    print(f"\n{tag}: exit {code} after {time.time() - started:.0f}s -> {output_dir}")
    print("A run that finishes is not a run to trust. In order:")
    print(f"  1. read all of {output_dir / 'warnings.log'} "
          "-- absent means the run never got that far")
    print(f"  2. python {bb_root / 'scripts' / 'print_results.py'} {output_dir} "
          "-s r_info_solveStatus r_cost_realizedCost r_cost_penalty")
    print("  3. the rest of the checklist is in "
          f"{bb_root / '.claude' / 'skills' / 'backbone-result-reader'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
