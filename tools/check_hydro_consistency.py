"""Does the hydro fleet match the water the build actually sends it?

The model's inflow comes from PECD and its capacities come from a compilation of
several sources that classify the same dams differently. Nothing in the build
compares the two, so a turbine fleet can quietly outgrow -- or fall short of -- the
river feeding it. This is the comparison, run against a folder a build produced.

Full load hours are the whole idea: annual inflow (MWh) over turbine capacity (MW)
gives the hours that capacity could run if it used every drop. It is the one number
that cannot be argued about between classifications, because both sides of it come
from the same build.

Four questions, in the order they matter:

1. does any node have to spill every year -- inflow above what its turbines pass in
   8760 hours -- regardless of how large its reservoir is;
2. does any zone's whole water-driven fleet sit outside a plausible band of full
   load hours, which is the gross-error net for a capacity refresh;
3. does any zone run its run-of-river *below* its reservoir, which is inverted: a
   river runs more hours than a store that is dispatched on price;
4. does the built model contain a hydro unit or node the hydro workbook did not
   write, which would mean another workbook leaked a row through.

The denominator for 2 and 3 is **water-driven** capacity, not every turbine with a
hydro label. Closed-loop pumped storage has no natural inflow by definition, and
several open-loop nodes have so little that they are closed-loop in all but name --
they run on pumped water and would drag a zone's ratio down for no reason. The
report names them rather than hiding the choice.

What it cannot see
------------------
It compares capacity against *annual* inflow. A fleet can match its water over a
year and still be unable to pass a wet week, which is a question about storage and
weekly shape that this does not ask.

It cannot tell a wrong capacity from a wrong classification. When a zone's split
looks off, the water may be booked to the wrong node rather than the turbine being
the wrong size; both show up here identically, and deciding between them needs a
person.

A low reservoir full-load-hours figure is **not** reported as a defect. A reservoir
that runs few hours is a store being saved for when it is worth using, which is what
a reservoir is for. Only the inversion in question 3 is a defect.

Two zones are exempt from questions 2 and 3 by decision, not by oversight -- see
``EXEMPT`` and ``docs/hydro.md``.

Usage:
    python tools/check_hydro_consistency.py <build_folder> [--workbook FILE]
                                            [--baseline FILE] [--reference FILE]

``--baseline`` is an earlier copy of the hydro workbook; given one, the report says
which zone totals moved. ``--reference`` is an external capacity vintage, shown as
context and never as a verdict. Both are optional and the checks do not need them.

Exit 0 when nothing is found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# The model's four hydro types, and the turbine unittype that draws from each.
TURBINE = {
    "reservoir": "reservoirTurbine",
    "ror": "rorTurbine",
    "psOpen": "psOpenTurbine",
    "psClosed": "psClosedTurbine",
}
HYDRO_UNITTYPES = set(TURBINE.values()) | {"psOpenPump", "psClosedPump"}

# Below this many hours of natural inflow a node is running on pumped water, not
# rainfall, and does not belong in a water-driven ratio. The gap is wide: the
# highest excluded node reaches 454 h and the lowest included one 771 h, so the
# threshold is not deciding any real case.
PUMP_DRIVEN_FLH = 600

# Above 8760 h the turbines cannot pass the year's water at full output for the
# whole year, so the node spills whatever its storage.
SPILL_FLH = 8760

# A gross-error net for a capacity refresh, not a tuned band. Observed zones run
# 1608-5846 h: continental 1900-3200, Nordic 3800-5800.
ZONE_FLH_BAND = (1200, 6500)

EXEMPT = {
    "FI00": "mostly run-of-river with large mid-river reservoirs; no clean type split exists",
    "SE04": "level data unreliable on four counts (docs/hydro.md); a ror fleet booked as reservoir",
}

WORKBOOK = REPO / "src_files" / "data_files" / "hydropower-compilation.xlsx"


def read_inflow(build: Path) -> tuple[dict[tuple[str, str], float], str | None]:
    """``(mean annual inflow in MWh per (zone, grid), reason it is empty)``.

    Read from the build's own ``ts_influx_hydro_<year>.gdx`` files, averaged over
    the climate years present. That is what the model receives -- after gap-filling,
    the week-53 rule and rounding -- so it is the right anchor rather than a
    re-derivation from the PECD CSVs, where the week/Day trap in the daily
    run-of-river file bites.

    Reading GDX needs ``gamsapi``. Without it the ratio checks cannot run, and the
    caller is told why rather than shown an empty report.
    """
    files = sorted(build.glob("ts_influx_hydro_*.gdx"))
    if not files:
        return {}, f"no ts_influx_hydro_<year>.gdx in {build}"

    sys.path.insert(0, str(REPO / "src"))
    try:
        import GDX_exchange as gx
    except Exception as exc:                       # gamsapi binds at import
        return {}, f"gamsapi is not importable ({type(exc).__name__})"

    def reduce_one(_gdx_file, records):
        if records is None or records.empty:
            return None
        return (records.assign(node=records["node"].astype(str))
                .groupby("node", observed=True)["value"].sum().reset_index())

    try:
        annual = gx.read_gdx_parameter_over_files([str(f) for f in files],
                                                  "ts_influx", reduce_one)
    except Exception as exc:
        return {}, f"could not read the hydro GDX files ({type(exc).__name__}: {exc})"
    if annual.empty:
        return {}, "the hydro GDX files carry no ts_influx records"

    # Each file contributed one row per node, so the mean over rows is the mean
    # over climate years.
    means = annual.groupby("node")["value"].mean()
    out = {}
    for node, value in means.items():
        zone, _, grid = str(node).partition("_")
        if grid in TURBINE:
            out[(zone, grid)] = float(value)
    return out, None


def read_workbook(path: Path) -> tuple[dict, dict, set, set]:
    """``(capacity, storage, removed, written_units, written_nodes)``.

    ``capacity`` is MW per (zone, grid) on the turbine; ``storage`` is MWh per
    (zone, grid) from ``upwardLimit``; ``removed`` holds the keys whose ``Method``
    is ``Remove``, which exist to suppress a row a base workbook would otherwise
    supply and so are deliberately absent from the build.

    The two ``written`` sets are every key the sheets name -- pumps included -- and
    they are what the coverage check needs rather than ``capacity`` and ``storage``:
    a row claims its key whether or not it carries a value, so a pump with no
    capacity or a node with no ``upwardLimit`` is owned, not leaked.
    """
    units = pd.read_excel(path, sheet_name="unitdata")
    nodes = pd.read_excel(path, sheet_name="nodedata")
    for frame in (units, nodes):
        frame.columns = [str(c).strip().lower() for c in frame.columns]

    removed = set()
    for _, row in units.iterrows():
        if str(row.get("method", "")).strip().lower() == "remove":
            removed.add((str(row.get("country")), str(row.get("unittype"))))
    for _, row in nodes.iterrows():
        if str(row.get("method", "")).strip().lower() == "remove":
            removed.add((str(row.get("country")), str(row.get("grid"))))

    live_units = units[units.get("method").isna()] if "method" in units else units
    capacity = {}
    for grid, unittype in TURBINE.items():
        rows = live_units[live_units.unittype == unittype]
        for _, row in rows.iterrows():
            value = row.get("capacity_output1")
            if pd.notna(value):
                capacity[(str(row.country), grid)] = float(value)

    written_units = {(str(row.country), str(row.unittype))
                     for _, row in live_units.iterrows()
                     if str(row.get("unittype")) in HYDRO_UNITTYPES}

    live_nodes = nodes[nodes.get("method").isna()] if "method" in nodes else nodes
    storage = {}
    written_nodes = set()
    for _, row in live_nodes.iterrows():
        grid, value = str(row.get("grid")), row.get("upwardlimit")
        if grid not in TURBINE:
            continue
        # Presence of the row is what claims the node; an upwardLimit is a separate
        # question, and a hydro node is allowed to have none.
        written_nodes.add((str(row.country), grid))
        if pd.notna(value):
            storage[(str(row.country), grid)] = float(value)

    return capacity, storage, removed, written_units, written_nodes


def read_built_hydro(build: Path) -> tuple[set, set]:
    """``(units, nodes)`` the build contains, as ``(zone, unittype)`` / ``(zone, grid)``."""
    path = build / "inputData.xlsx"
    if not path.exists():
        return set(), set()

    pairs = pd.read_excel(path, sheet_name="unitUnittype")
    units = {(str(u).split("_")[0], str(t))
             for u, t in zip(pairs.unit, pairs.unittype) if str(t) in HYDRO_UNITTYPES}

    # p_gn carries a GDXXRW label row above the data; a blank grid marks it.
    gn = pd.read_excel(path, sheet_name="p_gn")
    gn = gn[gn.grid.notna()]
    nodes = {(str(n).split("_")[0], str(g))
             for g, n in zip(gn.grid, gn.node) if str(g) in TURBINE}
    return units, nodes


def full_load_hours(inflow: float, capacity: float) -> float | None:
    if not capacity or capacity <= 0:
        return None
    return inflow / capacity


def classify_nodes(capacity: dict, inflow: dict) -> tuple[dict, dict]:
    """Split every turbine into water-driven and pump-driven, keyed by (zone, grid)."""
    water, pumped = {}, {}
    for key, megawatts in capacity.items():
        hours = full_load_hours(inflow.get(key, 0.0), megawatts)
        target = water if hours is not None and hours >= PUMP_DRIVEN_FLH else pumped
        target[key] = (megawatts, inflow.get(key, 0.0), hours)
    return water, pumped


def zone_totals(water: dict) -> dict[str, tuple[float, float]]:
    """``(MW, MWh)`` of water-driven capacity and its inflow, per zone."""
    totals: dict[str, list[float]] = {}
    for (zone, _grid), (megawatts, megawatthours, _hours) in water.items():
        entry = totals.setdefault(zone, [0.0, 0.0])
        entry[0] += megawatts
        entry[1] += megawatthours
    return {zone: (mw, mwh) for zone, (mw, mwh) in totals.items()}


def format_hours(hours: float | None) -> str:
    return "-" if hours is None else f"{hours:.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build", type=Path, help="a folder a build produced")
    parser.add_argument("--workbook", type=Path, default=None,
                        help="the hydro source workbook (default: found in src_files/data_files)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="an earlier copy of the workbook, to report zone totals that moved")
    parser.add_argument("--reference", type=Path, default=None,
                        help="an external capacity vintage, shown as context only")
    args = parser.parse_args()

    if not args.build.is_dir():
        print(f"Not a folder: {args.build}")
        return 1

    workbook = args.workbook or WORKBOOK
    if not workbook.exists():
        print(f"No hydro workbook at {workbook}; pass --workbook.")
        return 1

    inflow, why_no_inflow = read_inflow(args.build)

    capacity, storage, removed, written_units, written_nodes = read_workbook(workbook)
    built_units, built_nodes = read_built_hydro(args.build)
    water, pumped = classify_nodes(capacity, inflow)
    totals = zone_totals(water)

    problems = 0
    print(f"Hydro consistency: {args.build.name}, against {workbook.name}")
    print(f"  {len(capacity)} turbine fleet(s) in {len(totals)} zone(s); "
          f"{len(built_units)} hydro unit(s) and {len(built_nodes)} hydro node(s) built.\n")

    if why_no_inflow:
        # Everything below the coverage check is a ratio against inflow. Say so once,
        # run what can still run, and do not pretend the rest passed.
        print(f"Inflow unavailable -- {why_no_inflow}.")
        print("  The full-load-hour checks are skipped; coverage still runs.\n")

    print(f"Water-driven ({len(water)}), the denominator below:")
    for (zone, grid), (megawatts, _mwh, hours) in sorted(water.items()):
        print(f"  {zone}_{grid:<9} {megawatts:8.0f} MW  {format_hours(hours):>5} h")
    if pumped and not why_no_inflow:
        named = ", ".join(f"{z}_{g} ({format_hours(h)} h)"
                          for (z, g), (_mw, _i, h) in sorted(pumped.items()))
        total_mw = sum(mw for mw, _i, _h in pumped.values())
        print(f"\nPump-driven, excluded ({len(pumped)}, {total_mw:.0f} MW): {named}")
    print()

    print("Zone full load hours, water-driven capacity only:")
    for zone in sorted(totals, key=lambda z: totals[z][1] / totals[z][0]):
        megawatts, megawatthours = totals[zone]
        mark = "  (exempt)" if zone in EXEMPT else ""
        print(f"  {zone}  {megawatts:8.0f} MW  {megawatthours / 1e6:7.2f} TWh  "
              f"{megawatthours / megawatts:5.0f} h{mark}")
    print()

    spilling = [(key, hours) for key, (_mw, _i, hours) in sorted(water.items())
                if hours is not None and hours > SPILL_FLH]
    if spilling:
        problems += len(spilling)
        print(f"Must spill every year, whatever the storage ({len(spilling)}):")
        for (zone, grid), hours in spilling:
            print(f"  {zone}_{grid}: {hours:.0f} h of inflow against 8760 h of turbine")
        print()

    outside = []
    for zone, (megawatts, megawatthours) in sorted(totals.items()):
        if zone in EXEMPT:
            continue
        hours = megawatthours / megawatts
        if not ZONE_FLH_BAND[0] <= hours <= ZONE_FLH_BAND[1]:
            outside.append((zone, hours))
    if outside:
        problems += len(outside)
        low, high = ZONE_FLH_BAND
        print(f"Zone full load hours outside {low}-{high} h ({len(outside)}):")
        for zone, hours in outside:
            print(f"  {zone}: {hours:.0f} h -- the fleet and its water disagree")
        print()

    inverted = []
    for zone in sorted(totals):
        if zone in EXEMPT:
            continue
        ror = water.get((zone, "ror"))
        reservoir = water.get((zone, "reservoir"))
        if not ror or not reservoir or ror[2] is None or reservoir[2] is None:
            continue
        if ror[2] < reservoir[2]:
            inverted.append((zone, ror[2], reservoir[2]))
    if inverted:
        problems += len(inverted)
        print(f"Run-of-river running below its own reservoir ({len(inverted)}):")
        for zone, ror_hours, reservoir_hours in inverted:
            print(f"  {zone}: ror {ror_hours:.0f} h < reservoir {reservoir_hours:.0f} h "
                  "-- a river runs more hours than a store dispatched on price")
        print()

    missing_units = sorted(u for u in built_units
                           if u not in written_units and u not in removed)
    missing_nodes = sorted(n for n in built_nodes
                           if n not in written_nodes and n not in removed)
    if missing_units or missing_nodes:
        problems += len(missing_units) + len(missing_nodes)
        print("Built hydro the workbook does not write -- another workbook supplied it "
              f"({len(missing_units) + len(missing_nodes)}):")
        for zone, unittype in missing_units:
            print(f"  unit {zone}_{unittype}")
        for zone, grid in missing_nodes:
            print(f"  node {zone}_{grid}")
        print()

    print("Storage against inflow, weeks the reservoir holds:")
    for (zone, grid), megawatthours in sorted(storage.items()):
        water_in = inflow.get((zone, grid), 0.0)
        if water_in <= 0 or megawatthours <= 0:
            continue
        print(f"  {zone}_{grid:<9} {megawatthours / 1e6:7.3f} TWh  "
              f"{megawatthours / water_in * 52:6.2f} weeks")
    print()

    if args.baseline and args.baseline.exists():
        old_capacity, *_ = read_workbook(args.baseline)
        old_water = {k: v for k, v in old_capacity.items()
                     if (full_load_hours(inflow.get(k, 0.0), v) or 0.0) >= PUMP_DRIVEN_FLH}
        old_totals: dict[str, float] = {}
        for (zone, _grid), megawatts in old_water.items():
            old_totals[zone] = old_totals.get(zone, 0.0) + megawatts
        moved = [(z, old_totals.get(z, 0.0), totals[z][0]) for z in sorted(totals)
                 if abs(old_totals.get(z, 0.0) - totals[z][0]) > 1.0]
        if moved:
            print(f"Zone totals that moved against {args.baseline.name} ({len(moved)}) "
                  "-- each should be a decision:")
            for zone, before, after in moved:
                print(f"  {zone}: {before:.0f} -> {after:.0f} MW  ({after / before:.2f}x)"
                      if before else f"  {zone}: new, {after:.0f} MW")
        else:
            print(f"No zone total moved against {args.baseline.name}.")
        print()

    if args.reference and args.reference.exists():
        print(f"Context only, never a verdict -- {args.reference.name} is a different "
              "vintage and does not match PECD inflow.\n")

    if EXEMPT:
        print("Exempt from the zone and ordering checks, by decision:")
        for zone, why in sorted(EXEMPT.items()):
            print(f"  {zone} -- {why}")
        print()

    print("OK" if not problems else f"{problems} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
