"""
forecast_branch_energy.py -- what each forecast branch carries, as energy.

Usage:
    python tools/forecast_branch_energy.py <built_folder> [--targets] [--by-node]
                                           [--windows DAYS [DAYS ...]]

Example:
    python tools/forecast_branch_energy.py input_ObservedTrends_2030 --targets

The question it answers
-----------------------
A forecast branch is a per-hour quantile across the climate windows
(`calculate_climatological_forecasts`), and a per-hour quantile is **not** a
quantile of the energy a branch carries over its length. Every hour of a p0.45
wind branch is a slightly-below-median hour, and a run of those adds up to far
less wind than any real period of that length has: the years trade good hours
for bad ones, and the branch never does. How far off it is depends on how skewed
the series is, so it differs between onshore and offshore wind, between series
and between countries. A quantile chosen by its name is therefore a guess; this
tool states what it amounts to.

What it reports
---------------
1. **Branches as built.** For every timeseries family and every forecast branch:
   the branch's energy over its own length, relative to the mean of the realized
   climate years over the same windows, next to the realized range (the lowest
   and the highest year) and the share of years that fall below the branch.
   A branch outside the realized range describes a period no year in the record
   has had.
2. **Targets**, with `--targets`. For every family and window length: the
   per-hour quantile whose window energy equals the realized p05, p10, mean, p90
   and p95 of that window. These are the numbers to put in `forecast_quantiles`
   for a branch meant as, say, a one-in-ten period of that length.

`--by-node` repeats both per series instead of per family.

How the numbers are made
------------------------
- A window is `length` consecutive hours starting at midnight of any day of the
  climate window, wrapping from its end to its start as Backbone's circulation
  does. Every figure is the mean over all those start days.
- Wind and solar are weighted by the capacity `inputData.xlsx` attaches to each
  (flow, node); everything else is summed as it is, in MWh.
- Demand is a negative `ts_influx`, so its ratios are ratios of two negative
  sums: above 1 is more demand, and a low quantile is the high-demand one.
- A branch's length is read from the folder's `scheduleInit.gms`. The central
  branch reaches the horizon, and is measured over the whole climate window.
- `ts_node` families are skipped: a storage limit is a level, not an energy.

It reads the folder and writes nothing. It needs an importable `gams.transfer`,
as the build does.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import src.GDX_exchange as GDX_exchange  # noqa: E402  (needs the path above)

#: The realized quantiles of a window's energy that --targets solves for.
TARGETS = (("p05", 0.05), ("p10", 0.10), ("mean", None), ("p90", 0.90), ("p95", 0.95))

#: The per-hour quantiles searched for a target.
_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

_CENTRAL = "f01"


# --- reading the folder ------------------------------------------------------

def read_schedule(folder: Path) -> dict:
    """Data length, and each branch's length in hours, from scheduleInit.gms."""
    text = (folder / "scheduleInit.gms").read_text(encoding="utf-8")

    def first_int(pattern, default=None):
        found = re.search(pattern, text)
        return int(found.group(1)) if found else default

    data_length = first_int(r"'dataLength'\)\s*=\s*(\d+)\s*;")
    if data_length is None:
        raise SystemExit(f"{folder / 'scheduleInit.gms'} has no dataLength line.")
    default_length = first_int(
        r"mSettings\('schedule', '(?:forecastLength|t_forecastLengthUnchanging)'\)\s*=\s*(?:min\()?(\d+)",
        data_length,
    )
    lengths = {
        label: int(value)
        for label, value in re.findall(
            r"p_forecast\('(f\d+)', 'forecastLength'\)\s*=\s*(\d+)\s*;", text
        )
    }
    return {"data_length": data_length, "default_length": default_length, "lengths": lengths}


def find_families(folder: Path) -> list:
    """(name, parameter, forecast file, {year: file}) for every family with branches."""
    families = []
    for forecast_file in sorted(folder.glob("ts_*_forecasts.gdx")):
        stem = forecast_file.name[:-len("_forecasts.gdx")]
        parameter = re.match(r"(ts_[A-Za-z]+)_", stem + "_").group(1)
        if parameter == "ts_node":
            continue
        years = {}
        for path in folder.glob(f"{stem}_*.gdx"):
            tail = path.name[len(stem) + 1:-len(".gdx")]
            if re.fullmatch(r"\d{4}", tail):
                years[int(tail)] = path
        if years:
            families.append((stem[len(parameter) + 1:], parameter, forecast_file, dict(sorted(years.items()))))
    return families


def flow_capacities(folder: Path) -> dict:
    """MW of each (flow, node), from the units inputData.xlsx ties to a flow."""
    book = pd.ExcelFile(folder / "inputData.xlsx")
    if not {"flowUnit", "p_gnu_io"} <= set(book.sheet_names):
        return {}
    flow_unit = book.parse("flowUnit")
    gnu = book.parse("p_gnu_io")
    merged = gnu.merge(flow_unit[["flow", "unit"]], on="unit")
    merged = merged[merged["input_output"].astype(str).str.lower() == "output"]
    return merged.groupby(["flow", "node"])["capacity"].sum().to_dict()


def _keys_and_hours(records: pd.DataFrame):
    """Series key, branch label and zero-based hour of every record."""
    dims = list(records.columns[:-1])
    key = records[dims[:-2]].astype(str).agg("|".join, axis=1)
    hour = records[dims[-1]].astype(str).str[1:].astype(int).to_numpy() - 1
    return key, records[dims[-2]].astype(str), hour


def load_family(parameter: str, forecast_file: Path, year_files: dict, hours: int):
    """realized[year, series, hour], branches {label: [series, hour]} and the series names.

    A key absent from a GDX is a zero there, as it is to GAMS.
    """
    container = GDX_exchange.new_container()
    frames = {}

    def keep(path, records):
        frames[path] = records
        return None

    GDX_exchange.read_gdx_parameter_over_files(
        [str(p) for p in year_files.values()], parameter, keep, container=container
    )
    forecast = GDX_exchange.read_gdx_parameter(str(forecast_file), parameter)

    names = {}
    parsed = {}
    for path, records in list(frames.items()) + [("forecast", forecast)]:
        if not len(records):
            continue
        key, label, hour = _keys_and_hours(records)
        for name in key.unique():
            names.setdefault(name, len(names))
        parsed[path] = (key.map(names).to_numpy(), label.to_numpy(), hour,
                        records["value"].to_numpy(dtype=float))

    realized = np.zeros((len(year_files), len(names), hours))
    for k, path in enumerate(str(p) for p in year_files.values()):
        if path in parsed:
            series, _, hour, value = parsed[path]
            ok = (hour >= 0) & (hour < hours)
            realized[k, series[ok], hour[ok]] = value[ok]

    branches = {}
    if "forecast" in parsed:
        series, label, hour, value = parsed["forecast"]
        ok = (hour >= 0) & (hour < hours)
        for name in sorted(set(label)):
            pick = ok & (label == name)
            branch = np.zeros((len(names), hours))
            branch[series[pick], hour[pick]] = value[pick]
            branches[name] = branch
    return realized, branches, list(names)


# --- the arithmetic ----------------------------------------------------------

def window_sums(x: np.ndarray, hours: int) -> np.ndarray:
    """Sum over `hours` from midnight of every day, wrapping at the end. x: (..., H)."""
    total = x.shape[-1]
    hours = min(hours, total)
    wrapped = np.concatenate([x, x[..., :hours]], axis=-1)
    cumulative = np.concatenate(
        [np.zeros(x.shape[:-1] + (1,)), np.cumsum(wrapped, axis=-1)], axis=-1
    )
    starts = np.arange(0, total, 24)
    return cumulative[..., starts + hours] - cumulative[..., starts]


def _safe_ratio(numerator, denominator):
    """numerator / denominator, NaN where the denominator is zero."""
    denominator = np.where(denominator == 0, np.nan, denominator)
    return numerator / denominator


def describe_branch(branch: np.ndarray, realized: np.ndarray, hours: int) -> dict:
    """One branch against the realized years. branch: (H,), realized: (year, H)."""
    r = window_sums(realized, hours)                      # year, start
    b = window_sums(branch, hours)                        # start
    ratios = _safe_ratio(r, r.mean(axis=0))               # year, start
    return {
        "ratio": float(np.nanmean(_safe_ratio(b, r.mean(axis=0)))),
        "low": float(np.nanmean(np.nanmin(ratios, axis=0))),
        "high": float(np.nanmean(np.nanmax(ratios, axis=0))),
        "below": float(np.mean((r < b).mean(axis=0))),
    }


def equivalent_quantiles(realized_series: np.ndarray, weights: np.ndarray, hours: int) -> dict:
    """The per-hour q whose window energy equals each realized target.

    realized_series: (year, series, H). Returns {target: (ratio to the mean, q)}.
    """
    by_q = np.quantile(realized_series, _GRID, axis=0)                  # q, series, H
    branch = (by_q * weights[None, :, None]).sum(axis=1)                # q, H
    realized = (realized_series * weights[None, :, None]).sum(axis=1)   # year, H
    r = window_sums(realized, hours)
    mean = r.mean(axis=0)
    by_q_ratio = np.nanmean(_safe_ratio(window_sums(branch, hours), mean), axis=1)
    order = np.argsort(by_q_ratio)
    out = {}
    for name, quantile in TARGETS:
        target = mean if quantile is None else np.quantile(r, quantile, axis=0)
        ratio = float(np.nanmean(_safe_ratio(target, mean)))
        if np.isnan(ratio) or np.isnan(by_q_ratio).all():
            out[name] = (float("nan"), float("nan"))
            continue
        out[name] = (ratio, float(np.interp(ratio, by_q_ratio[order], _GRID[order])))
    return out


# --- reporting ---------------------------------------------------------------

def _branch_hours(label: str, schedule: dict) -> int:
    if label == _CENTRAL:
        return schedule["data_length"]
    length = schedule["lengths"].get(label, schedule["default_length"])
    return min(length, schedule["data_length"])


def report(folder: Path, *, targets: bool, by_node: bool, windows) -> int:
    schedule = read_schedule(folder)
    hours = schedule["data_length"]
    families = find_families(folder)
    if not families:
        print(f"No ts_*_forecasts.gdx with climate-year files beside it in {folder}.")
        return 1
    capacities = flow_capacities(folder)

    print(f"{folder.name}: {hours} h of data per climate window")
    print("Energy of each forecast branch over its own length, relative to the mean of")
    print("the realized climate years over the same windows. 'years below' is the share")
    print("of climate years with less than the branch; demand is negative, so above 1")
    print("is more demand.\n")

    for name, parameter, forecast_file, year_files in families:
        realized, branches, series = load_family(parameter, forecast_file, year_files, hours)
        if parameter == "ts_cf":
            weights = np.array([capacities.get(tuple(s.split("|")), 0.0) for s in series])
            note = f"weighted by {weights.sum() / 1000:.1f} GW of capacity"
            if not weights.any():
                weights, note = np.ones(len(series)), "unweighted: no capacity found"
        else:
            weights, note = np.ones(len(series)), "summed"
        annual = (realized * weights[None, :, None]).sum(axis=(1, 2)) / 1e6

        print(f"=== {parameter} {name}: {len(series)} series, {len(year_files)} climate years, {note}")
        print(f"    realized, whole window: mean {annual.mean():.1f} TWh, "
              f"lowest {annual.min():.1f}, highest {annual.max():.1f}")

        groups = [("all", np.arange(len(series)))]
        if by_node:
            groups += [(s, np.array([i])) for i, s in enumerate(series)]

        print(f"    {'series':<22} {'branch':>6} {'days':>5} {'branch/mean':>12} "
              f"{'realized range':>16} {'years below':>12}")
        for group, members in groups:
            w = weights[members]
            total = (realized[:, members, :] * w[None, :, None]).sum(axis=1)
            if not total.any():
                continue
            for label, branch in branches.items():
                length = _branch_hours(label, schedule)
                row = describe_branch((branch[members] * w[:, None]).sum(axis=0), total, length)
                print(f"    {group:<22} {label:>6} {length / 24:>5.0f} {row['ratio']:>12.3f} "
                      f"{row['low']:>7.3f}-{row['high']:<8.3f} {row['below']:>12.2f}")

        if targets:
            days = windows or sorted({_branch_hours(label, schedule) // 24 for label in branches})
            print(f"    per-hour quantile that gives a realized window energy "
                  f"(energy relative to the mean -> q):")
            for group, members in groups:
                w = weights[members]
                if not (realized[:, members, :] * w[None, :, None]).any():
                    continue
                for length in days:
                    found = equivalent_quantiles(realized[:, members, :], w, length * 24)
                    cells = "  ".join(
                        f"{target} {ratio:.3f}->q{q:.2f}" for target, (ratio, q) in found.items()
                    )
                    print(f"    {group:<22} {length:>4} d  {cells}")
        print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/forecast_branch_energy.py",
        description="What each forecast branch of a built folder carries, as energy.",
    )
    parser.add_argument("folder", type=Path, help="A folder build_input_data.py built")
    parser.add_argument("--targets", action="store_true",
                        help="Also print the per-hour quantile that gives the realized "
                             "p05, p10, mean, p90 and p95 of each window's energy")
    parser.add_argument("--by-node", action="store_true",
                        help="Repeat every row per series, not only per family")
    parser.add_argument("--windows", type=int, nargs="+", metavar="DAYS",
                        help="Window lengths for --targets (default: the branch lengths)")
    args = parser.parse_args(argv)

    if not (args.folder / "scheduleInit.gms").is_file():
        parser.error(f"{args.folder} is not a built folder: it has no scheduleInit.gms")
    return report(args.folder, targets=args.targets, by_node=args.by_node, windows=args.windows)


if __name__ == "__main__":
    sys.exit(main())
