"""
Run this model through the Backbone checkout it is installed in.

    python run_model.py OT2030 --year 1998 --days 1
    python run_model.py OT2030 --year 1998 --days 1 --show
    python run_model.py --list

This is a worked example as much as a tool: it is the shortest correct way to
start a run here, and the place to copy from when writing a sweep. Everything it
does that a hand-written command line would not is below.


WHAT IT DELEGATES, AND WHY

Nothing here builds a `gams` command line. The scenario, the climate year and the
horizon are this project's business; which GAMS, which solver, and how a run is
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

  - Two runs against *different* scenarios are safe to overlap.
  - Two runs against the *same* input folder are not, and the failure is silent:
    each overwrites the other's prepared input, and both report numbers.

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

    python run_model.py OT2030 --year 1998 --days 7 -- --debug=2 --diag=yes

`../.claude/skills/backbone-scenario-runner/` covers the four ways a Backbone run
can be varied, and `../docs/running-backbone/command-line-parameters.md` the
parameters themselves.


NO SOLVER IS NAMED HERE

A solver is a fact about the machine, and machine facts stay out of tracked
files. Left alone, the parent resolves one and prints which it picked; `--solver`
and the BB_SOLVER environment variable override it. This is a deliberate
difference from the `run-*.cmd` files, which hardcode one.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# This project first, so its own `src` and `tests` win over the parent's
# same-named folders; the Backbone root is appended later, where `scripts` is the
# only thing wanted from it.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

LOCK_NAME = ".run_model.lock"
STALE_HOURS = 24


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


def config_for(scenario: str) -> Path:
    """src_files/config_<scenario>.ini, or a message listing what does exist."""
    config = HERE / "src_files" / f"config_{scenario}.ini"
    if config.is_file():
        return config
    available = sorted(
        p.name[len("config_"):-len(".ini")]
        for p in (HERE / "src_files").glob("config_*.ini")
    )
    raise SystemExit(
        f"No config for scenario '{scenario}' ({config.name} does not exist).\n"
        f"Known scenarios: {', '.join(available)}"
    )


def input_folders_for(config_file: Path):
    """Every input folder a build of this config writes, by the build's own rule."""
    import src.infrastructure.config_reader as config_reader

    config = config_reader.load_config(config_file)
    return [HERE / name for name in config_reader.config_output_folder_names(config)]


def resolve_input_folder(scenario: str, override) -> Path:
    if override:
        folder = Path(override)
        if not folder.is_absolute():
            folder = HERE / folder
        if not folder.is_dir():
            raise SystemExit(f"--input-dir {folder} is not a folder.")
        return folder

    folders = input_folders_for(config_for(scenario))
    if len(folders) > 1:
        listed = "\n  ".join(f.name for f in folders)
        raise SystemExit(
            f"config_{scenario}.ini builds {len(folders)} input folders, so the "
            "scenario name alone does not say which to run. Name one with "
            f"--input-dir:\n  {listed}"
        )
    folder = folders[0]
    if not folder.is_dir():
        raise SystemExit(
            f"{folder.name} has not been built yet. Build it with:\n"
            f"  python build_input_data.py src_files src_files/config_{scenario}.ini"
        )
    return folder


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
            "Wait for it, run a different scenario, or pass --force if you know "
            "that run is gone."
        )
    lock = folder / LOCK_NAME
    lock.write_text(json.dumps({
        "tag": tag,
        "pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")
    return lock


def build_argv(args, bb_root: Path, input_folder: Path, output_dir: Path):
    """The run_backbone.py command line this run becomes."""
    def rel(path: Path) -> str:
        return path.relative_to(bb_root).as_posix()

    argv = ["--output-dir", rel(output_dir)]
    if args.gams:
        argv += ["--gams", args.gams]
    if args.solver:
        argv += ["--solver", args.solver]
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
    return argv + list(args.backbone_args)


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


def do_list() -> int:
    """Every scenario, where it builds to, and whether it is built or busy."""
    configs = sorted((HERE / "src_files").glob("config_*.ini"))
    if not configs:
        print("No config_*.ini in src_files/.")
        return 1
    for config_file in configs:
        scenario = config_file.name[len("config_"):-len(".ini")]
        try:
            folders = input_folders_for(config_file)
        except Exception as exc:  # report and keep listing; one bad config is not fatal
            print(f"{scenario:28} !! could not read: {exc}")
            continue
        for folder in folders:
            held = read_lock(folder) if folder.is_dir() else None
            if not folder.is_dir():
                state = "not built"
            elif held:
                state = f"BUSY ({held.get('tag')})"
            else:
                state = "built"
            print(f"{scenario:28} {folder.name:42} {state}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python run_model.py",
        description="Run this model through the Backbone checkout it sits in.",
        epilog="Backbone's own arguments go last, after a bare --, for example: "
               "-- --debug=2 --diag=yes",
    )
    parser.add_argument("scenario", nargs="?",
                        help="Scenario name, i.e. the <x> in src_files/config_<x>.ini")
    parser.add_argument("--year", "--climate-year", dest="year", type=int, default=1998,
                        help="Climate year, passed as --climateYear (default: 1998)")
    parser.add_argument("--days", type=int, default=1,
                        help="Days to model, passed as --modelledDays (default: 1)")
    parser.add_argument("--tag", default=None,
                        help="Name for this run's folder under results/ "
                             "(default: <scenario>-<year>-<days>d)")
    parser.add_argument("--debug", default="1",
                        help="Backbone --debug level (default: 1, which writes the "
                             "debug gdx; 2 writes one gdx per solve)")
    parser.add_argument("--input-dir", dest="input_dir", default=None,
                        help="Input folder to run, when a config builds more than one")
    parser.add_argument("--solver", default=None,
                        help="Solver to use. Default: whichever the Backbone checkout "
                             "resolves, which it prints.")
    parser.add_argument("--gams", default=None,
                        help="GAMS to use: a path, or an alias such as '47'")
    parser.add_argument("--show", action="store_true",
                        help="Print the command that would run, and exit")
    parser.add_argument("--force", action="store_true",
                        help="Start even though another run holds this input folder")
    parser.add_argument("--list", action="store_true",
                        help="List the scenarios, their input folders and their state")
    own, passthrough = split_passthrough(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(own)
    args.backbone_args = check_passthrough(passthrough)

    if args.list:
        return do_list()
    if not args.scenario:
        parser.error("name a scenario, or pass --list to see them")

    bb_root = find_backbone_root(HERE)
    runner = import_runner(bb_root)

    input_folder = resolve_input_folder(args.scenario, args.input_dir)
    tag = args.tag or f"{args.scenario}-{args.year}-{args.days}d"
    output_dir = HERE / "results" / tag

    runner_argv = build_argv(args, bb_root, input_folder, output_dir)

    if args.show:
        return runner.main(runner_argv)

    lock = take_lock(input_folder, tag, args.force)
    started = time.time()
    try:
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
