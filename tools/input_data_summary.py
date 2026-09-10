"""
input_data_summary.py -- what is in one built input-data folder, as a report.

Reads <built_folder>/inputData.xlsx and, unless told not to, the per-climate-year
ts_influx_*, ts_cf_* and ts_node_hydro_storage_limits_* GDX files beside it.
Writes one self-contained report.md, with its figures, into a subfolder of the
built folder -- so a colleague can be handed a folder and read it.

Usage:
    python tools/input_data_summary.py <built_folder> [--zones]
        [--no-timeseries] [--out-subdir NAME]

Examples:
    python tools/input_data_summary.py input_ObservedTrends_2030
    python tools/input_data_summary.py input_ObservedTrends_2030 --zones
    python tools/input_data_summary.py input_tyndp2024_NationalTrends_2040

What it shows
-------------
Sixteen countries by default, the 22 bidding zones with --zones. One section per
energy carrier -- electricity, district heat, hydrogen -- each with production
capacity, consumption capacity and annual demand. Then storage, interconnection,
a net-load duration curve, how much 35 weather years move the numbers, and fuel,
CO2 and emission prices.

A carrier with no data still gets its section, saying so. That is deliberate:
during a data-adding phase an empty panel is the thing worth seeing, and a
section that disappears cannot tell anyone it is empty.

What it cannot see
------------------
Nothing below zone level -- there is no per-node figure at any flag setting. No
"Nordics" or "CWE" grouping: nothing in the tracked source data defines one, so
none is invented, and zone and country are the only two levels offered.

Whether a unit can actually run: availability and the efficiency curve are read
but never judged, because every unit in the shipped scenarios has availability
1.0 and one efficiency per unittype, so a check would fire on nothing forever.

How the model behaves. Every number here comes from the input data, never from a
solved run, so the net-load curve ignores storage, trade and dispatch, and the
VRE figures ignore curtailment and outages.

The technology grouping is this tool's own: the source data names 63 unittypes
and no grouping of them. Units are grouped by which grids they touch, and the
report counts how many fell outside the groups -- watch that count, not this
paragraph, for a unittype the grouping has not met yet.

Exit code is 0 when report.md was written -- including when the timeseries
sections were skipped by request or for want of a GAMS install, which is
documented degradation rather than failure -- 1 when inputData.xlsx is there but
unreadable, and 2 when the folder or the workbook is missing, so it can gate a
loop.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # no display on a build machine
import matplotlib.pyplot as plt

#: tools/ is not on sys.path as the repo root, and this tool imports src/.
#: profile_build.py does the same thing for the same reason.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ============================================================================
# Constants
# ============================================================================

#: The two columns adjust_excel adds for Excel's own table formatting. Neither
#: is data, and one of them is named like a sentence.
JUNK_COLUMN_EXACT = "The first row labels are for excel Table headers."
JUNK_COLUMN_PREFIX = "Unnamed:"

#: The sheets the builder gives a second header row, repeating the parameter name
#: in every parameter column for GDXXRW's benefit. pandas reads it as data, so it
#: is skipped -- but only here. Every other sheet's first data row is its first
#: row, and skipping one there loses a real record.
MARKER_ROW_SHEETS = frozenset({
    "p_gn", "p_gnBoundaryPropertiesForStates", "p_gnn", "p_gnu_io", "p_unit",
})

#: Display names only. Country *aggregation* needs no table -- zone[:2] is exact
#: for every zone the build writes -- but "NOS0" tells a reader nothing.
COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "DE": "Germany",
    "DK": "Denmark", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
    "FR": "France", "IT": "Italy", "LT": "Lithuania", "LV": "Latvia",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "SE": "Sweden", "UK": "United Kingdom",
}

#: Fuel grids to the label their capacity is reported under. The merges are
#: presentational: the per-grid number stays exact underneath.
FUEL_GRID_LABELS = {
    "nuclear": "nuclear",
    "gas": "gas",
    "coal": "coal & lignite",
    "lignite": "coal & lignite",
    "oilHeavy": "oil", "oilLight": "oil", "oilShale": "oil",
    "biomass": "bio & waste", "blackLiquor": "bio & waste", "waste": "bio & waste",
}

#: Grids whose nodes receive natural inflow. Derived from the data at run time;
#: this is the fallback when no inflow GDX can be read. psOpen belongs here and
#: psClosed does not -- Norway's psOpenTurbine fleet is inflow-driven generation,
#: and a rule keyed on the unittype name instead would file the whole Norwegian
#: power system under storage.
HYDRO_INFLOW_GRIDS_FALLBACK = frozenset({"psOpen", "reservoir", "ror"})

#: Carrier sections, in report order. A carrier absent from a build still gets
#: its section. 'steam' is deliberately not here: it is reported as one line,
#: because it carries no timeseries and has never been scenario-differentiated.
CARRIERS = [
    ("elec", "Electricity", "elec"),
    ("dheat", "District heat", "dheat"),
    ("H2", "Hydrogen", "h2"),
]

#: Which ts_influx family carries each carrier's demand. A carrier missing here
#: has no hourly demand in the build.
DEMAND_FAMILY = {"elec": "ts_influx_elec", "dheat": "ts_influx_dheat"}

#: flow -> (gdx family, capacity label). 'solar thermal' has units but no
#: ts_cf family, so it is not here.
VRE_FAMILIES = {
    "onshore": ("ts_cf_wind_onshore", "wind onshore"),
    "offshore": ("ts_cf_wind_offshore", "wind offshore"),
    "PV": ("ts_cf_PV", "solar"),
}

#: Explicit per-label colours, not a colormap: a colormap reassigns every colour
#: when one group is absent, so the same technology would change colour between
#: two scenarios. Ordered light-to-dark within a family so a stacked bar stays
#: readable in greyscale, with hatches as the second channel.
TECH_STYLE = {
    "wind onshore":   ("#4878a8", ""),
    "wind offshore":  ("#2a4d6e", "//"),
    "solar":          ("#e8b84b", ""),
    "hydro":          ("#5fa8d3", "\\\\"),
    "nuclear":        ("#8e6fb0", ""),
    "gas":            ("#d98050", ""),
    "coal & lignite": ("#5a5a5a", "xx"),
    "oil":            ("#8c6d4f", ".."),
    "bio & waste":    ("#6aa84f", "//"),
    "hydrogen":       ("#c45b8c", "\\\\"),
    "battery":        ("#b0b0b0", ""),
    "pumped storage": ("#7fb3a8", "xx"),
    "heat storage":   ("#c8a882", ""),
    "electric heat":  ("#9ec5e8", ".."),
    "electrolyser":   ("#d8a0c0", ".."),
    "demand response": ("#cfcfcf", "++"),
    "other":          ("#e0e0e0", ""),
}
OTHER_LABEL = "other"

#: Sharp at the width a markdown viewer renders an embedded PNG, without the
#: multi-megabyte files 300 dpi would produce for a 16:9 figure.
FIG_DPI = 150
FIG_WIDTH_IN = 9.0

#: matplotlib renamed boxplot's ``vert=False`` to ``orientation='horizontal'`` in
#: 3.11 and warns on the old spelling, but the new one does not exist before it.
#: Decided once here rather than at the call site, which would read as if the
#: figure were the complicated part.
_HORIZONTAL_BOXPLOT = (
    {"orientation": "horizontal"}
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) >= (3, 11)
    else {"vert": False}
)

#: Below this mean annual inflow a country's hydro percentage swings say more
#: about a small denominator than about the system, so the variability figure
#: leaves it out and says how many it left out.
MATERIAL_HYDRO_TWH = 10.0

#: Names before a count, per the project's reporting rule.
NAME_LIMIT = 3

MWH_TO_TWH = 1e-6
MW_TO_GW = 1e-3


def summarise(items: Sequence, limit: int = NAME_LIMIT) -> str:
    """The first few names, then how many are left."""
    items = [str(i) for i in items]
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} and {len(items) - limit} more"


# ============================================================================
# Workbook layer
# ============================================================================

def read_bb_sheet(xl: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """One sheet of a built inputData.xlsx, without the parts that are not data.

    The five parameter sheets carry a second header row that repeats the
    parameter name in every parameter column -- a marker for GDXXRW, which pandas
    reads as a data row -- plus the two formatting columns adjust_excel adds.
    Every other sheet starts its data on the row below the header, so skipping a
    row there would silently eat the first one: it cost a unit and a unittype off
    every count in this report before it was caught.
    """
    skip = [1] if sheet_name in MARKER_ROW_SHEETS else None
    try:
        df = xl.parse(sheet_name, header=0, skiprows=skip)
    except ValueError:
        return pd.DataFrame()
    keep = [
        c for c in df.columns
        if str(c) != JUNK_COLUMN_EXACT and not str(c).startswith(JUNK_COLUMN_PREFIX)
    ]
    return df[keep]


def col_or(df: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    """A column if the build wrote one, else a column of `default`.

    A parameter column exists only if some row set it: the builder drops
    all-zero columns because 0, NA and "not set" are the same thing to GAMS. So
    an absent column means nobody set it, not that the sheet is malformed, and
    `df[column]` on an optional parameter is a bug waiting for a scenario that
    happens not to use it.
    """
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def numeric_with_eps(series: pd.Series) -> pd.Series:
    """Numbers from a column that may carry the GAMS literal ``Eps``.

    ``errors='raise'``, not ``'coerce'``: in this workbook NaN already means
    "this row uses a timeseries instead of a constant", and coercing would make
    a genuinely new non-numeric value indistinguishable from that. Eps is a
    deliberate zero and is converted; anything else is louder than it is
    convenient.
    """
    # mask rather than replace: replace downcasts the column as a side effect,
    # which pandas 2.2 warns about and this project turns into a test failure.
    is_eps = series.map(lambda v: isinstance(v, str) and v.strip().lower() == "eps")
    return pd.to_numeric(series.mask(is_eps, 0.0), errors="raise")


@dataclass
class Workbook:
    """The sheets this report reads, already cleaned."""
    path: Path
    scenario: str
    year: str
    p_gnu_io: pd.DataFrame
    p_unit: pd.DataFrame
    p_gn: pd.DataFrame
    p_gnn: pd.DataFrame
    boundary: pd.DataFrame
    flow_unit: pd.DataFrame
    unit_unittype: pd.DataFrame
    n_emission: pd.DataFrame
    emission_price: pd.DataFrame
    grids: List[str]
    nodes: List[str]
    unittypes: List[str]


def load_workbook(xlsx_path: Path) -> Workbook:
    """Read every sheet the report needs, in one pass over the file."""
    xl = pd.ExcelFile(xlsx_path)

    tags = read_bb_sheet(xl, "add_scen_tags")
    scenario = str(tags["scenario"].iloc[0]) if "scenario" in tags and len(tags) else "unknown"
    year = str(tags["year"].iloc[0]) if "year" in tags and len(tags) else "unknown"

    p_gnu_io = read_bb_sheet(xl, "p_gnu_io")
    for c in ("capacity", "isActive", "conversionCoeff"):
        if c in p_gnu_io.columns:
            p_gnu_io[c] = pd.to_numeric(p_gnu_io[c], errors="coerce")

    p_gn = read_bb_sheet(xl, "p_gn")
    for c in ("price", "influx", "usePrice", "isActive"):
        if c in p_gn.columns:
            p_gn[c] = pd.to_numeric(p_gn[c], errors="coerce")

    p_gnn = read_bb_sheet(xl, "p_gnn")
    for c in ("transferCap", "availability", "transferLoss", "isActive"):
        if c in p_gnn.columns:
            p_gnn[c] = pd.to_numeric(p_gnn[c], errors="coerce")

    boundary = read_bb_sheet(xl, "p_gnBoundaryPropertiesForStates")
    if "constant" in boundary.columns:
        boundary["constant"] = numeric_with_eps(boundary["constant"])
    for c in ("useConstant", "useTimeseries"):
        if c in boundary.columns:
            boundary[c] = pd.to_numeric(boundary[c], errors="coerce")

    n_emission = read_bb_sheet(xl, "p_nEmission")
    if "value" in n_emission.columns:
        n_emission["value"] = pd.to_numeric(n_emission["value"], errors="coerce")

    emission_price = read_bb_sheet(xl, "ts_emissionPriceChange")
    if "value" in emission_price.columns:
        emission_price["value"] = pd.to_numeric(emission_price["value"], errors="coerce")

    def domain(sheet: str, column: str) -> List[str]:
        df = read_bb_sheet(xl, sheet)
        if column not in df.columns:
            return []
        return [str(v) for v in df[column].dropna().tolist()]

    return Workbook(
        path=xlsx_path,
        scenario=scenario,
        year=year,
        p_gnu_io=p_gnu_io,
        p_unit=read_bb_sheet(xl, "p_unit"),
        p_gn=p_gn,
        p_gnn=p_gnn,
        boundary=boundary,
        flow_unit=read_bb_sheet(xl, "flowUnit"),
        unit_unittype=read_bb_sheet(xl, "unitUnittype"),
        n_emission=n_emission,
        emission_price=emission_price,
        grids=domain("grid", "grid"),
        nodes=domain("node", "node"),
        unittypes=domain("unittype", "unittype"),
    )


# ============================================================================
# Zones, countries, names
# ============================================================================

def zone_of(node_or_unit: str) -> str:
    """'FI00_elec' -> 'FI00'; 'AT00_BatteryCharger' -> 'AT00'."""
    return str(node_or_unit).split("_", 1)[0]


def country_of(zone: str) -> str:
    """'NOS0' -> 'NO'. Exact for every zone code the builder writes."""
    return str(zone)[:2]


def area_of(node_or_unit: str, zones: bool) -> str:
    z = zone_of(node_or_unit)
    return z if zones else country_of(z)


def display_name(code: str, zones: bool) -> str:
    """'FI00 (Finland)' at zone level, 'FI (Finland)' at country level."""
    name = COUNTRY_NAMES.get(country_of(code))
    return f"{code} ({name})" if name else str(code)


# ============================================================================
# Classifying units by the grids they touch
# ============================================================================

def derive_storage_grids(workbook: Workbook, inflow_grids: Optional[set]) -> set:
    """Grids that hold energy but receive no natural inflow.

    Read from the data rather than listed by name. A literal set would have been
    written against one scenario's vocabulary, and the TYNDP scenarios rename
    'battery' to 'battery4h' -- which silently dropped a tenth of that build's
    capacity into "other" the one time it was tried.
    """
    if "grid" not in workbook.boundary.columns:
        return set()
    state_grids = set(workbook.boundary["grid"].dropna().astype(str))
    return state_grids - set(inflow_grids or HYDRO_INFLOW_GRIDS_FALLBACK)


def storage_label(grid: str) -> str:
    """Storage grids kept apart where their duration story differs."""
    g = str(grid).lower()
    if g.startswith("battery"):
        return "battery"
    if g.startswith("heatstor"):
        return "heat storage"
    return "pumped storage"


@dataclass
class Classification:
    """Output-side capacity per unit, labelled, plus what could not be labelled."""
    rows: pd.DataFrame               # node, unit, carrier, capacity, label
    unlabelled: pd.DataFrame         # rows that fell to 'other', with the reason
    inflow_grids: set
    storage_grids: set


def classify_capacity(
    workbook: Workbook,
    inflow_grids: Optional[set] = None,
    ) -> Classification:
    """Label every unit's output capacity by the grids that unit connects.

    The rule is which grids a unit touches, never the unittype's name. The two
    pumped-hydro families are the reason: psOpenTurbine receives real inflow and
    is generation, psClosedTurbine receives none and is storage, and their names
    differ by one word.

    Units are summed on their *output* side only. A CHP unit's fuel side and a
    battery's charging side are input-side rows of the same unit, and adding
    both would count one plant twice.
    """
    io = workbook.p_gnu_io
    if io.empty or "input_output" not in io.columns:
        empty = pd.DataFrame(columns=["node", "unit", "carrier", "capacity", "label"])
        return Classification(empty, empty, set(), set())

    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    io = io[active]

    inflow = set(inflow_grids) if inflow_grids else set(HYDRO_INFLOW_GRIDS_FALLBACK)
    storage = derive_storage_grids(workbook, inflow)

    outputs = io[io["input_output"].astype(str) == "output"].copy()
    inputs = io[io["input_output"].astype(str) == "input"]

    # Which grids feed each unit. A unit with none is either a flow-driven
    # generator (wind, solar) or something with no modelled fuel at all.
    feeds = (
        inputs.groupby("unit", observed=True)["grid"]
        .agg(lambda s: frozenset(str(v) for v in s))
        .to_dict()
    )
    flow_of = {}
    if {"flow", "unit"} <= set(workbook.flow_unit.columns):
        flow_of = dict(zip(workbook.flow_unit["unit"], workbook.flow_unit["flow"]))

    # A unit that produces without consuming any modelled commodity is flagged
    # in the data itself, by p_unit.isSource. In the shipped scenarios those are
    # exactly the demand-response tiers, and keying on the flag rather than on
    # the unittype name is what keeps a renamed tier in the group.
    sources = set()
    if not workbook.p_unit.empty and "isSource" in workbook.p_unit.columns:
        is_source = pd.to_numeric(workbook.p_unit["isSource"], errors="coerce").fillna(0) == 1
        sources = set(workbook.p_unit.loc[is_source, "unit"])

    labels, reasons = [], []
    for unit, out_grid in zip(outputs["unit"], outputs["grid"]):
        in_grids = feeds.get(unit, frozenset())
        flow = str(flow_of.get(unit, "")) if unit in flow_of else ""
        label, reason = _label_for(str(out_grid), in_grids, flow, inflow, storage)
        if label == OTHER_LABEL and unit in sources:
            label, reason = "demand response", ""
        labels.append(label)
        reasons.append(reason)

    outputs["label"] = labels
    outputs["reason"] = reasons
    outputs["carrier"] = outputs["grid"].astype(str)
    outputs["capacity"] = pd.to_numeric(outputs["capacity"], errors="coerce").fillna(0.0)

    rows = outputs[["node", "unit", "carrier", "capacity", "label"]].copy()
    unlabelled = outputs.loc[
        outputs["label"] == OTHER_LABEL, ["node", "unit", "carrier", "capacity", "reason"]
    ].copy()
    return Classification(rows, unlabelled, inflow, storage)


def _label_for(
    out_grid: str,
    in_grids: frozenset,
    flow: str,
    inflow_grids: set,
    storage_grids: set,
    ) -> Tuple[str, str]:
    """The technology label for one output row, and why, if it has none."""
    # A flow-driven generator has no input grid; the flow names the resource.
    if flow:
        for _, (_, label) in VRE_FAMILIES.items():
            pass
        mapping = {f: lbl for f, (_, lbl) in VRE_FAMILIES.items()}
        if flow in mapping:
            return mapping[flow], ""
        if "solar" in flow.lower():
            return "solar", ""
        if "wind" in flow.lower():
            return "wind onshore", ""

    if not in_grids:
        return OTHER_LABEL, "no input grid and no flow"

    if len(in_grids) > 1:
        fuels = {FUEL_GRID_LABELS[g] for g in in_grids if g in FUEL_GRID_LABELS}
        if len(fuels) == 1:
            return fuels.pop(), ""
        return OTHER_LABEL, f"several input grids: {', '.join(sorted(in_grids))}"

    in_grid = next(iter(in_grids))

    if in_grid in inflow_grids:
        return "hydro", ""
    if in_grid in storage_grids:
        return storage_label(in_grid), ""
    if in_grid in FUEL_GRID_LABELS:
        return FUEL_GRID_LABELS[in_grid], ""
    if in_grid == "H2":
        return "hydrogen", ""
    if in_grid == "elec":
        # Power-to-something: the output carrier says which.
        if out_grid == "H2":
            return "electrolyser", ""
        if out_grid in ("dheat", "steam"):
            return "electric heat", ""
        return "demand response", ""
    return OTHER_LABEL, f"input grid '{in_grid}' is in no technology group"


def consumption_capacity(workbook: Workbook, carrier: str, zones: bool) -> pd.DataFrame:
    """Input-side capacity on a carrier, labelled by what it produces.

    This is what makes the hydrogen section say anything: its electrolysers are
    elec-consuming and its turbines are H2-consuming, and neither appears in a
    generation total.
    """
    io = workbook.p_gnu_io
    if io.empty or "input_output" not in io.columns:
        return pd.DataFrame(columns=["area", "label", "capacity"])

    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    io = io[active]
    inputs = io[(io["input_output"].astype(str) == "input")
                & (io["grid"].astype(str) == carrier)].copy()
    if inputs.empty:
        return pd.DataFrame(columns=["area", "label", "capacity"])

    outs = io[io["input_output"].astype(str) == "output"]
    produces = (
        outs.groupby("unit", observed=True)["grid"]
        .agg(lambda s: sorted({str(v) for v in s})[0])
        .to_dict()
    )
    inputs["label"] = [
        f"to {produces.get(u, 'nothing')}" for u in inputs["unit"]
    ]
    inputs["capacity"] = pd.to_numeric(inputs["capacity"], errors="coerce").fillna(0.0)
    inputs["area"] = [area_of(n, zones) for n in inputs["node"]]
    return (
        inputs.groupby(["area", "label"], observed=True)["capacity"].sum().reset_index()
    )


# ============================================================================
# Aggregation from the workbook
# ============================================================================

def capacity_by_area(classification: Classification, carrier: str, zones: bool) -> pd.DataFrame:
    """area x label capacity in GW for one carrier."""
    rows = classification.rows
    rows = rows[rows["carrier"] == carrier]
    if rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    work["area"] = [area_of(n, zones) for n in work["node"]]
    table = (
        work.groupby(["area", "label"], observed=True)["capacity"].sum().unstack(fill_value=0.0)
        * MW_TO_GW
    )
    ordered = [l for l in TECH_STYLE if l in table.columns]
    extra = [c for c in table.columns if c not in ordered]
    return table[ordered + extra]


def steam_demand_twh(workbook: Workbook, carrier: str = "steam") -> float:
    """Constant influx carried in the workbook, as annual energy.

    Demand grids with no timeseries processor get a flat MWh/h instead of an
    hourly series, stored negative like every other demand.
    """
    gn = workbook.p_gn
    if gn.empty or "grid" not in gn.columns:
        return 0.0
    rows = gn[gn["grid"].astype(str) == carrier]
    influx = col_or(rows, "influx", 0.0).fillna(0.0)
    return float(-influx.sum() * 8760 * MWH_TO_TWH)


def transfer_by_area(workbook: Workbook, zones: bool) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Electricity transfer capacity per area, and the corridor matrix.

    Each corridor appears twice, once per direction, and 15 of the 44 in the
    shipped scenarios carry a different capacity each way. A corridor is
    summarised by the mean of its two directions: summing them double-counts,
    and taking one of them makes the answer depend on which row the builder
    happened to write first.

    Links inside a multi-zone country are kept apart from its cross-border
    total, so Sweden's number is not inflated by SE01-SE02.
    """
    gnn = workbook.p_gnn
    cols = {"grid", "from_node", "to_node"}
    if gnn.empty or not cols <= set(gnn.columns):
        return pd.DataFrame(), pd.DataFrame(), {}

    elec = gnn[gnn["grid"].astype(str) == "elec"].copy()
    if elec.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    elec["cap"] = pd.to_numeric(col_or(elec, "transferCap", 0.0), errors="coerce").fillna(0.0)
    elec["a"] = [zone_of(n) for n in elec["from_node"]]
    elec["b"] = [zone_of(n) for n in elec["to_node"]]
    elec["pair"] = [tuple(sorted((a, b))) for a, b in zip(elec["a"], elec["b"])]

    per_pair = elec.groupby("pair", observed=True).agg(
        mean_cap=("cap", "mean"),
        directions=("cap", "size"),
        spread=("cap", lambda s: float(s.max() - s.min())),
    ).reset_index()

    asymmetric = per_pair[per_pair["spread"] > 0]
    one_way = per_pair[
        (per_pair["directions"] == 2) & (per_pair["mean_cap"] > 0)
        & per_pair["pair"].map(
            lambda p: bool(((elec["pair"] == p) & (elec["cap"] == 0)).any())
        )
    ]

    rows = []
    for (a, b), cap in zip(per_pair["pair"], per_pair["mean_cap"]):
        internal = (not zones) and country_of(a) == country_of(b)
        # A cross-border corridor is capacity for the area at each of its ends,
        # so it is recorded twice, once per end. An internal one has both ends
        # in the same area and is recorded once: adding it per end would report
        # Sweden's own ring at twice its size.
        ends = ((a, b),) if internal else ((a, b), (b, a))
        for near, far in ends:
            rows.append({
                "area": near if zones else country_of(near),
                "other": far if zones else country_of(far),
                "cap": cap,
                "internal": internal,
            })
    long = pd.DataFrame(rows)

    per_area = long.groupby(["area", "internal"], observed=True)["cap"].sum().unstack(fill_value=0.0)
    per_area = per_area.rename(columns={False: "cross_border_MW", True: "inter_zonal_MW"})
    for c in ("cross_border_MW", "inter_zonal_MW"):
        if c not in per_area.columns:
            per_area[c] = 0.0

    matrix = (
        long[~long["internal"]]
        .groupby(["area", "other"], observed=True)["cap"].sum().unstack(fill_value=0.0)
    )

    facts = {
        "pairs": int(len(per_pair)),
        "asymmetric": int(len(asymmetric)),
        "one_way": int(len(one_way)),
        "total_GW": float(per_pair["mean_cap"].sum() * MW_TO_GW),
        "largest_asymmetries": [
            (f"{p[0]}-{p[1]}", float(s))
            for p, s in asymmetric.nlargest(3, "spread")[["pair", "spread"]].itertuples(index=False)
        ],
        "transfer_loss": _uniform_or_none(elec, "transferLoss"),
    }
    return per_area, matrix, facts


def _uniform_or_none(df: pd.DataFrame, column: str) -> Optional[float]:
    """The single value of a column, when the build wrote one and it is uniform."""
    if column not in df.columns:
        return None
    vals = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    return float(vals[0]) if len(vals) == 1 else None


def storage_energy(
    workbook: Workbook,
    storage_limit_ts: Optional[pd.DataFrame],
    zones: bool,
    ) -> Tuple[pd.DataFrame, Dict]:
    """Energy storage capacity, from the two places it is written.

    An upwardLimit row either carries a constant or says useTimeseries, never
    both. The twelve rows that say useTimeseries are every large Nordic and
    Alpine reservoir, and their constant cell is empty -- so a reader of the
    workbook alone sees about six per cent of the system's real reservoir
    energy.
    """
    boundary = workbook.boundary
    facts = {"constant_TWh": 0.0, "timeseries_TWh": 0.0, "timeseries_nodes": [],
             "by_grid_constant": {}, "grids_without_energy": []}
    if boundary.empty or "param_gnBoundaryTypes" not in boundary.columns:
        return pd.DataFrame(), facts

    upward = boundary[boundary["param_gnBoundaryTypes"].astype(str) == "upwardLimit"].copy()
    if upward.empty:
        return pd.DataFrame(), facts

    use_constant = col_or(upward, "useConstant", 0.0).fillna(0.0) == 1
    use_ts = col_or(upward, "useTimeseries", 0.0).fillna(0.0) == 1
    constant = col_or(upward, "constant", np.nan)

    fixed = upward[use_constant].assign(MWh=constant[use_constant].fillna(0.0))
    facts["constant_TWh"] = float(fixed["MWh"].sum() * MWH_TO_TWH)
    facts["by_grid_constant"] = {
        str(g): float(v * MWH_TO_TWH)
        for g, v in fixed.groupby("grid", observed=True)["MWh"].sum().items()
    }
    facts["timeseries_nodes"] = sorted(str(n) for n in upward.loc[use_ts, "node"])

    parts = [fixed[["grid", "node", "MWh"]].assign(source="workbook constant")]
    if storage_limit_ts is not None and not storage_limit_ts.empty:
        ts = storage_limit_ts.rename(columns={"value": "MWh"})
        facts["timeseries_TWh"] = float(ts["MWh"].sum() * MWH_TO_TWH)
        parts.append(ts[["grid", "node", "MWh"]].assign(source="timeseries"))

    table = pd.concat(parts, ignore_index=True)
    table["area"] = [area_of(n, zones) for n in table["node"]]

    # Storage grids that hold power but no stated energy -- a property of the
    # data worth naming, not a gap in this tool.
    io = workbook.p_gnu_io
    if not io.empty and "grid" in io.columns:
        power_grids = {
            str(g) for g in io["grid"].unique()
            if storage_label(str(g)) in ("battery", "heat storage")
            and str(g).lower().startswith(("battery", "heatstor"))
        }
        facts["grids_without_energy"] = sorted(power_grids - set(table["grid"].astype(str)))
    return table, facts


def storage_power_by_area(workbook: Workbook, zones: bool) -> pd.DataFrame:
    """Discharge power of the storages whose energy capacity is never stated."""
    io = workbook.p_gnu_io
    if io.empty or "input_output" not in io.columns:
        return pd.DataFrame()
    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    inputs = io[active & (io["input_output"].astype(str) == "input")]
    wanted = inputs[inputs["grid"].astype(str).str.lower().str.startswith(("battery", "heatstor"))]
    if wanted.empty:
        return pd.DataFrame()
    work = wanted.copy()
    work["label"] = [storage_label(str(g)) for g in work["grid"]]
    work["area"] = [area_of(n, zones) for n in work["node"]]
    work["capacity"] = pd.to_numeric(work["capacity"], errors="coerce").fillna(0.0)
    return (
        work.groupby(["area", "label"], observed=True)["capacity"].sum().unstack(fill_value=0.0)
        * MW_TO_GW
    )


def fuel_prices(workbook: Workbook) -> pd.DataFrame:
    """Per-fuel price, and whether it is the same in every zone."""
    gn = workbook.p_gn
    if gn.empty or "grid" not in gn.columns:
        return pd.DataFrame()
    priced = gn[col_or(gn, "usePrice", 0.0).fillna(0.0) == 1].copy()
    if priced.empty:
        return pd.DataFrame()
    priced["price"] = pd.to_numeric(col_or(priced, "price", np.nan), errors="coerce")
    out = priced.groupby("grid", observed=True)["price"].agg(
        price="median", distinct="nunique", nodes="size"
    ).reset_index()
    return out.sort_values("price")


def emission_factors(workbook: Workbook) -> pd.DataFrame:
    """Per-fuel emission intensity, collapsed over the zones that share it."""
    ne = workbook.n_emission
    if ne.empty or not {"node", "emission", "value"} <= set(ne.columns):
        return pd.DataFrame()
    work = ne.copy()
    work["fuel"] = [str(n).split("_", 1)[1] if "_" in str(n) else str(n) for n in work["node"]]
    return (
        work.groupby(["fuel", "emission"], observed=True)["value"]
        .agg(factor="median", distinct="nunique").reset_index().sort_values("factor", ascending=False)
    )


def co2_price(workbook: Workbook) -> Optional[float]:
    ep = workbook.emission_price
    if ep.empty or "value" not in ep.columns:
        return None
    vals = ep["value"].dropna()
    return float(vals.iloc[0]) if len(vals) else None


# ============================================================================
# Timeseries layer
# ============================================================================

@dataclass
class Timeseries:
    """Everything the GDX files contribute, already reduced to areas."""
    years: List[int] = field(default_factory=list)
    areas: List[str] = field(default_factory=list)
    hours_per_year: int = 0
    hourly: Dict[str, np.ndarray] = field(default_factory=dict)
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    node_cf_mean: pd.DataFrame = field(default_factory=pd.DataFrame)
    storage_limit: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.skipped is None and bool(self.years)


#: node -> area, filled once per run. Mapping is asked per GDX column, and there
#: are tens of thousands of columns across a sweep.
_AREA_CACHE: Dict[str, str] = {}


def open_gdx_backend():
    """(module, container) when GDX can be read here, else (None, reason).

    src/GDX_exchange.py imports gams.transfer at module scope, so a machine
    without gamsapi fails on the import rather than on a read -- which is why
    this import is here and not at the top of the file.
    """
    try:
        from src import GDX_exchange as gx
    except Exception as exc:
        return None, f"gamsapi is not importable ({type(exc).__name__})"
    try:
        container = gx.new_container()
    except Exception as exc:
        return None, f"no usable GAMS install ({type(exc).__name__}: {exc})"
    return (gx, container), None


def year_files(folder: Path, family: str) -> Dict[int, Path]:
    """{climate year: file} for one ts_* family, ignoring the forecast file."""
    found = {}
    for path in sorted(folder.glob(f"{family}_*.gdx")):
        stem = path.stem.rsplit("_", 1)[-1]
        if len(stem) == 4 and stem.isdigit():
            found[int(stem)] = path
    return found


def _records(container, path: Optional[Path], parameter: str) -> Optional[pd.DataFrame]:
    """One file's records, or None.

    Existence is checked rather than caught: gams.transfer raises a bare
    Exception for a missing file, which cannot be told apart from a real read
    failure by catching a narrower type. A deterministic or single-climate-year
    build writes no forecast file at all, so a missing file is an ordinary
    state, not an error.
    """
    if path is None or not Path(path).is_file():
        return None
    try:
        container.read(str(path), [parameter])
    except Exception:
        return None
    records = None
    if parameter in container.data:
        found = container[parameter].records
        if found is not None and len(found):
            records = found.reset_index(drop=True)
        container.removeSymbols([parameter])
    return records


def _area_hourly(
    records: pd.DataFrame,
    areas: List[str],
    weight_by_node: Optional[Dict[str, float]] = None,
    sign: float = 1.0,
    ) -> Optional[np.ndarray]:
    """One file's records as an (hours x areas) array, summed into areas.

    `weight_by_node` turns a per-unit capacity factor into MW. A node with no
    weight contributes nothing, which is how a capacity-factor series for a node
    carrying no units of that flow is ignored rather than counted as 1 MW.
    """
    if records is None or records.empty:
        return None
    wide = (
        records.assign(node=records["node"].astype(str))
        .pivot_table(index="t", columns="node", values="value", aggfunc="sum", observed=True)
        .sort_index()
    )
    out = pd.DataFrame(0.0, index=wide.index, columns=areas, dtype="float64")
    for col in wide.columns:
        area = _AREA_CACHE.get(col)
        if area is None or area not in out.columns:
            continue
        if weight_by_node is None:
            out[area] += wide[col].fillna(0.0)
        else:
            weight = float(weight_by_node.get(col, 0.0))
            if weight:
                out[area] += wide[col].fillna(0.0) * weight
    return out.to_numpy(dtype="float32") * sign


def _vre_capacity_by_node(workbook: Workbook) -> Dict[str, Dict[str, float]]:
    """{flow: {node: MW}} -- what a capacity factor has to be multiplied by."""
    io, fu = workbook.p_gnu_io, workbook.flow_unit
    if io.empty or fu.empty or not {"flow", "unit"} <= set(fu.columns):
        return {}
    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    outputs = io[active & (io["input_output"].astype(str) == "output")]
    joined = outputs.merge(fu, on="unit", how="inner")
    if joined.empty:
        return {}
    joined["capacity"] = pd.to_numeric(joined["capacity"], errors="coerce").fillna(0.0)
    grouped = joined.groupby(["flow", "node"], observed=True)["capacity"].sum()
    out: Dict[str, Dict[str, float]] = {}
    for (flow, node), mw in grouped.items():
        out.setdefault(str(flow), {})[str(node)] = float(mw)
    return out


def read_timeseries(folder: Path, workbook: Workbook, zones: bool) -> Timeseries:
    """Read every per-year GDX family, reducing each file as it is read.

    One family at a time, all of its climate years on one container, through
    ``GDX_exchange.read_gdx_parameter_over_files``. Building a container binds
    gams.transfer to a GAMS install, and that binding -- not the GDX parse -- is
    what a read costs, which is why the write path already builds one per write
    and why this must not build one per file.

    Per-node rows, about 200k per file, are collapsed to per-area hours inside
    the loop and never accumulate. What is kept is one hourly column per area
    per family: the input the net-load curve needs, and the thing the annual
    figures are derived from, so the files are read once and not once per
    question.
    """
    backend, reason = open_gdx_backend()
    if backend is None:
        return Timeseries(skipped=reason)
    gx, container = backend

    years = sorted(set().union(*(
        set(year_files(folder, family)) for family in DEMAND_FAMILY.values()
    )) if DEMAND_FAMILY else set())
    if not years:
        return Timeseries(skipped="no per-climate-year ts_influx GDX file in this folder")

    # Areas come from the nodes the workbook declares, so a node appearing only
    # in a GDX cannot silently invent a column.
    _AREA_CACHE.clear()
    _AREA_CACHE.update({str(n): area_of(str(n), zones) for n in workbook.nodes})
    areas = sorted(set(_AREA_CACHE.values()))
    vre_capacity = _vre_capacity_by_node(workbook)

    hourly_parts: Dict[str, List[Optional[np.ndarray]]] = {}
    cf_seen: Dict[Tuple[str, str], List[float]] = {}
    annual_parts: List[pd.DataFrame] = []
    hours = 0

    def sweep(key, family, parameter, sign=1.0, weights=None, flow=None):
        """One family, every climate year, reduced per file."""
        nonlocal hours
        available = year_files(folder, family)
        if not available:
            return
        # A year with no file of this family still takes a slot, so a family
        # that starts late cannot shift every later year out of step with the
        # others. A path that does not exist yields an empty frame.
        ordered = [available.get(y, folder / f"{family}_{y}.gdx") for y in years]
        frames: List[Optional[np.ndarray]] = []

        def reduce_one(gdx_file, records):
            array = _area_hourly(records, areas, weight_by_node=weights, sign=sign)
            frames.append(array)
            if flow is not None and records is not None and not records.empty:
                means = (
                    records.assign(node=records["node"].astype(str))
                    .groupby("node", observed=True)["value"].mean()
                )
                for node, mean_cf in means.items():
                    cf_seen.setdefault((flow, str(node)), []).append(float(mean_cf))
            if array is None:
                return None
            year = int(Path(gdx_file).stem.rsplit("_", 1)[-1])
            return pd.DataFrame({
                "key": key, "area": areas, "year": year,
                "TWh": array.sum(axis=0) * MWH_TO_TWH,
            })

        annual = gx.read_gdx_parameter_over_files(
            ordered, parameter, reduce_one,
            container=container,
            progress_desc=family if sys.stdout.isatty() else None,
        )
        if not annual.empty:
            annual_parts.append(annual)
        hourly_parts.setdefault(key, []).extend(frames)
        for array in frames:
            if array is not None:
                hours = max(hours, array.shape[0])

    for carrier, family in DEMAND_FAMILY.items():
        # Demand is stored negative, the way Backbone consumes it.
        sweep(f"demand_{carrier}", family, "ts_influx", sign=-1.0)
    sweep("inflow_hydro", "ts_influx_hydro", "ts_influx")
    for flow, (family, _) in VRE_FAMILIES.items():
        sweep(f"vre_{flow}", family, "ts_cf",
              weights=vre_capacity.get(flow, {}), flow=flow)

    hourly = _stack(hourly_parts, len(areas), hours)
    _combine_vre(hourly)

    annual = pd.concat(annual_parts, ignore_index=True) if annual_parts else pd.DataFrame()
    annual = _add_vre_annual(annual, hourly, areas, years, hours)

    cf_rows = [
        {"flow": flow, "node": node, "cf": float(np.mean(values))}
        for (flow, node), values in cf_seen.items()
    ]
    return Timeseries(
        years=years,
        areas=areas,
        hours_per_year=hours,
        hourly=hourly,
        annual=annual,
        node_cf_mean=pd.DataFrame(cf_rows),
        storage_limit=_read_storage_limit(container, folder),
    )


def _combine_vre(hourly: Dict[str, np.ndarray]) -> None:
    """Wind and solar summed into the one series the net-load curve subtracts."""
    parts = [hourly.pop(f"vre_{flow}") for flow in VRE_FAMILIES if f"vre_{flow}" in hourly]
    if parts:
        total = parts[0].copy()
        for part in parts[1:]:
            total += part
        hourly["vre"] = total


def _add_vre_annual(annual, hourly, areas, years, hours) -> pd.DataFrame:
    """The combined VRE series needs its own annual rows; its parts had theirs."""
    if "vre" not in hourly or not hours:
        return annual[annual["key"].isin(
            {"demand_elec", "demand_dheat", "inflow_hydro"}
        )] if not annual.empty else annual
    rows = []
    for index, year in enumerate(years):
        block = hourly["vre"][index * hours:(index + 1) * hours, :]
        if not block.size:
            continue
        for area, value in zip(areas, block.sum(axis=0) * MWH_TO_TWH):
            rows.append({"key": "vre", "area": area, "year": year, "TWh": float(value)})
    keep = annual[~annual["key"].astype(str).str.startswith("vre_")] if not annual.empty else annual
    return pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)


def _stack(stacks, n_areas: int, hours: int) -> Dict[str, np.ndarray]:
    """Years into one array per family, a missing year filled with zeros.

    A family present for some climate years and absent for others would
    otherwise shift every later year's hours out of step with the other
    families, and nothing downstream could detect it.
    """
    out = {}
    for key, per_year in stacks.items():
        if not any(a is not None for a in per_year):
            continue
        filled = []
        for arr in per_year:
            if arr is None:
                filled.append(np.zeros((hours, n_areas), dtype="float32"))
            elif arr.shape[0] < hours:
                filled.append(np.pad(arr, ((0, hours - arr.shape[0]), (0, 0))))
            else:
                filled.append(arr[:hours, :])
        out[key] = np.vstack(filled)
    return out


def _read_storage_limit(container, folder: Path) -> pd.DataFrame:
    """The reservoir energy ceiling, from exactly one file.

    upwardLimit in ts_node_hydro_storage_limits is identical in every climate
    year and every forecast branch: it is a reservoir's physical size wrapped in
    a useTimeseries flag, not a weather-dependent quantity. So one file answers
    it. The forecast file is tried first only because it is one file rather than
    35; a per-year file is the fallback for a deterministic or single-year build,
    which writes no forecast file at all.
    """
    candidates = [folder / "ts_node_hydro_storage_limits_forecasts.gdx"]
    per_year = year_files(folder, "ts_node_hydro_storage_limits")
    if per_year:
        candidates.append(per_year[min(per_year)])
    for path in candidates:
        records = _records(container, path, "ts_node")
        if records is None or records.empty:
            continue
        if "param_gnBoundaryTypes" not in records.columns:
            continue
        upward = records[records["param_gnBoundaryTypes"].astype(str) == "upwardLimit"]
        if upward.empty:
            continue
        return (
            upward.assign(
                node=upward["node"].astype(str), grid=upward["grid"].astype(str)
            )
            .groupby(["grid", "node"], observed=True)["value"].max().reset_index()
        )
    return pd.DataFrame()


def netload_by_area(timeseries: Timeseries) -> Dict[str, np.ndarray]:
    """Demand minus wind and solar, per area, every climate year pooled.

    Hydro is not subtracted: reservoir hydro is dispatchable, so subtracting it
    would answer a different question than what firm capacity has to cover.
    """
    demand = timeseries.hourly.get("demand_elec")
    if demand is None:
        return {}
    vre = timeseries.hourly.get("vre")
    net = demand if vre is None else demand - vre
    return {area: net[:, i] for i, area in enumerate(timeseries.areas)}


def capacity_weighted_cf(timeseries: Timeseries, workbook: Workbook, zones: bool) -> pd.DataFrame:
    """Per-area capacity factor, weighted by each node's own installed capacity.

    An unweighted mean over a country's zones would let a 200 MW zone move the
    answer as much as a 12 GW one.
    """
    if timeseries.node_cf_mean.empty:
        return pd.DataFrame()
    capacity = _vre_capacity_by_node(workbook)
    rows = []
    for flow, group in timeseries.node_cf_mean.groupby("flow", observed=True):
        weights = capacity.get(str(flow), {})
        work = group.assign(
            area=[area_of(n, zones) for n in group["node"]],
            mw=[weights.get(str(n), 0.0) for n in group["node"]],
        )
        work = work[work["mw"] > 0]
        if work.empty:
            continue
        for area, part in work.groupby("area", observed=True):
            rows.append({
                "flow": str(flow),
                "area": area,
                "cf": float(np.average(part["cf"], weights=part["mw"])),
                "capacity_GW": float(part["mw"].sum() * MW_TO_GW),
            })
    return pd.DataFrame(rows)


def annual_range(timeseries: Timeseries, key: str) -> pd.DataFrame:
    """Per-area mean, minimum and maximum over the climate years, with the years."""
    if timeseries.annual.empty:
        return pd.DataFrame()
    part = timeseries.annual[timeseries.annual["key"] == key]
    if part.empty:
        return pd.DataFrame()
    rows = []
    for area, group in part.groupby("area", observed=True):
        if group["TWh"].abs().sum() == 0:
            continue
        low = group.loc[group["TWh"].idxmin()]
        high = group.loc[group["TWh"].idxmax()]
        rows.append({
            "area": area,
            "mean": float(group["TWh"].mean()),
            "min": float(low["TWh"]), "min_year": int(low["year"]),
            "max": float(high["TWh"]), "max_year": int(high["year"]),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


# ============================================================================
# Figures
# ============================================================================

def _style_for(label: str) -> Tuple[str, str]:
    return TECH_STYLE.get(label, TECH_STYLE[OTHER_LABEL])


def _finish(fig, path: Path) -> str:
    """Every figure leaves through here, so they share one look and one size."""
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path.name


def _barh_height(n_rows: int, panels: int = 1) -> float:
    """Tall enough that labels never crowd, short enough to embed."""
    return max(2.6, 0.34 * max(n_rows, 1) + 1.3)


def _empty_panel(ax, message: str) -> None:
    """A panel that has nothing to draw still says what it looked for."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10,
            color="#666666", wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _stacked_barh(ax, table: pd.DataFrame, labels: List[str], unit: str) -> None:
    left = np.zeros(len(table))
    positions = np.arange(len(table))
    for label in labels:
        if label not in table.columns:
            continue
        values = table[label].to_numpy(dtype="float64")
        colour, hatch = _style_for(label)
        ax.barh(positions, values, left=left, label=label, color=colour,
                hatch=hatch, edgecolor="white", linewidth=0.4, height=0.74)
        left += values
    ax.set_yticks(positions)
    ax.set_yticklabels(table.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(unit, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)


def figure_carrier(
    out_dir: Path,
    slug: str,
    title: str,
    capacity: pd.DataFrame,
    demand: pd.DataFrame,
    consumption: pd.DataFrame,
    zones: bool,
    ) -> str:
    """Production capacity and annual demand for one carrier, side by side.

    The two panels keep the same country order so a reader can put a country's
    capacity against its own demand without moving between images.
    """
    order = None
    if capacity is not None and not capacity.empty:
        order = capacity.sum(axis=1).sort_values(ascending=False).index.tolist()
    elif demand is not None and not demand.empty:
        order = demand.set_index("area")["mean"].sort_values(ascending=False).index.tolist()

    n_rows = len(order) if order else 6
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, _barh_height(n_rows)),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    if order and capacity is not None and not capacity.empty:
        table = capacity.reindex(order).fillna(0.0)
        table.index = [display_name(a, zones) for a in table.index]
        labels = [l for l in TECH_STYLE if l in table.columns]
        _stacked_barh(axes[0], table, labels, "installed capacity, GW")
        axes[0].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
    else:
        _empty_panel(axes[0], f"No {title.lower()} production capacity\nin this scenario")

    if order and demand is not None and not demand.empty:
        series = demand.set_index("area")["mean"].reindex(order).fillna(0.0)
        positions = np.arange(len(series))
        axes[1].barh(positions, series.to_numpy(dtype="float64"),
                     color="#4a6fa5", height=0.74)
        axes[1].set_yticks(positions)
        axes[1].set_yticklabels([display_name(a, zones) for a in series.index], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("annual demand, TWh/yr", fontsize=9)
        axes[1].grid(axis="x", alpha=0.25)
        axes[1].tick_params(axis="x", labelsize=8)
    else:
        consumed = 0.0 if consumption is None or consumption.empty else float(
            consumption["capacity"].sum() * MW_TO_GW)
        note = f"No {title.lower()} demand built\nin this scenario"
        if consumed:
            note += f"\n\n({consumed:,.1f} GW of units consume it,\nbut nothing demands it directly)"
        _empty_panel(axes[1], note)

    fig.suptitle(f"{title}: capacity and demand by {'zone' if zones else 'country'}",
                 fontsize=11)
    return _finish(fig, out_dir / f"fig_{slug}.png")


def figure_storage(
    out_dir: Path,
    power: pd.DataFrame,
    energy: pd.DataFrame,
    zones: bool,
    ) -> str:
    """Power on the left, energy on the right, because they are different facts.

    Battery and heat storage have no stated energy capacity in this model, so the
    left panel is the whole of what is known about them.
    """
    n_rows = max(len(power) if power is not None else 0,
                 energy["area"].nunique() if energy is not None and not energy.empty else 0)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, _barh_height(n_rows)))

    if power is not None and not power.empty:
        table = power.loc[power.sum(axis=1).sort_values(ascending=False).index]
        table.index = [display_name(a, zones) for a in table.index]
        _stacked_barh(axes[0], table, [l for l in TECH_STYLE if l in table.columns],
                      "discharge power, GW")
        axes[0].legend(fontsize=7, loc="lower right")
        axes[0].set_title("Battery and heat storage\n(power only: no energy capacity is set)",
                          fontsize=9)
    else:
        _empty_panel(axes[0], "No battery or heat storage\nin this scenario")

    if energy is not None and not energy.empty:
        pivot = (
            energy.groupby(["area", "source"], observed=True)["MWh"].sum().unstack(fill_value=0.0)
            * MWH_TO_TWH
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        positions = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for source, colour in (("workbook constant", "#7fb3a8"), ("timeseries", "#2f6f62")):
            if source not in pivot.columns:
                continue
            values = pivot[source].to_numpy(dtype="float64")
            axes[1].barh(positions, values, left=left, label=source, color=colour,
                         edgecolor="white", linewidth=0.4, height=0.74)
            left += values
        axes[1].set_yticks(positions)
        axes[1].set_yticklabels([display_name(a, zones) for a in pivot.index], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("reservoir and pumped storage, TWh", fontsize=9)
        axes[1].grid(axis="x", alpha=0.25)
        axes[1].tick_params(axis="x", labelsize=8)
        axes[1].legend(fontsize=7, loc="lower right")
        axes[1].set_title("Hydro storage energy\n(where the number is written)", fontsize=9)
    else:
        _empty_panel(axes[1], "No hydro storage energy capacity\ncould be read")

    return _finish(fig, out_dir / "fig_storage_capacity.png")


def figure_interconnection(
    out_dir: Path,
    matrix: pd.DataFrame,
    per_area: pd.DataFrame,
    peak: Optional[pd.Series],
    zones: bool,
    ) -> str:
    """Who connects to whom, and who leans on it."""
    if matrix is None or matrix.empty:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
        _empty_panel(ax, "No electricity transfer capacity in this scenario")
        return _finish(fig, out_dir / "fig_interconnection.png")

    areas = sorted(set(matrix.index) | set(matrix.columns))
    grid = matrix.reindex(index=areas, columns=areas).fillna(0.0) * MW_TO_GW
    size = max(5.0, 0.36 * len(areas) + 2.4)
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, size),
        gridspec_kw={"width_ratios": [1.0, 0.42]},
    )

    data = grid.to_numpy(dtype="float64")
    shown = np.where(data > 0, data, np.nan)
    image = axes[0].imshow(shown, cmap="YlGnBu", aspect="equal")
    axes[0].set_xticks(range(len(areas)))
    axes[0].set_yticks(range(len(areas)))
    axes[0].set_xticklabels(areas, rotation=90, fontsize=7)
    axes[0].set_yticklabels(areas, fontsize=7)
    if len(areas) <= 22:
        for i in range(len(areas)):
            for j in range(len(areas)):
                if data[i, j] > 0:
                    axes[0].text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                                 fontsize=5.5, color="#222222")
    axes[0].set_title("Transfer capacity between areas, GW\n"
                      "(mean of the two directions)", fontsize=9)
    fig.colorbar(image, ax=axes[0], fraction=0.040, pad=0.03).ax.tick_params(labelsize=7)

    if peak is not None and not peak.empty and per_area is not None and not per_area.empty:
        total = (per_area["cross_border_MW"] + per_area["inter_zonal_MW"])
        share = (total / peak.reindex(total.index)).dropna().sort_values() * 100
        positions = np.arange(len(share))
        axes[1].barh(positions, share.to_numpy(dtype="float64"), color="#c46a4f", height=0.74)
        axes[1].axvline(100, color="#555555", linestyle=":", linewidth=1.0)
        axes[1].set_yticks(positions)
        axes[1].set_yticklabels(share.index, fontsize=7)
        axes[1].set_xlabel("transfer capacity\nas % of own peak demand", fontsize=8)
        axes[1].grid(axis="x", alpha=0.25)
        axes[1].tick_params(axis="x", labelsize=7)
        axes[1].set_title("Over 100% is not an error:\na small, well-connected area\n"
                          "can exceed its own peak", fontsize=8)
    else:
        _empty_panel(axes[1], "Peak demand needs the\ntimeseries files")

    return _finish(fig, out_dir / "fig_interconnection.png")


def figure_climate_variability(
    out_dir: Path,
    timeseries: Timeseries,
    material: List[str],
    zones: bool,
    ) -> str:
    """Each area's climate years as a percentage of its own mean.

    Demand and inflow share one axis only because both are normalised. In TWh
    they cannot: demand moves a couple of per cent across 35 years and inflow
    moves tens, so demand would draw as a flat line.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, _barh_height(len(material) * 2)))
    if not material:
        _empty_panel(ax, "No area has enough hydro inflow to compare")
        return _finish(fig, out_dir / "fig_climate_year_variability.png")

    annual = timeseries.annual
    spec = [("inflow_hydro", "hydro inflow", "#5fa8d3"), ("demand_elec", "electricity demand", "#c46a4f")]
    offset = {"inflow_hydro": -0.18, "demand_elec": 0.18}
    positions = np.arange(len(material))

    for key, label, colour in spec:
        boxes, at = [], []
        for index, area in enumerate(material):
            values = annual[(annual["key"] == key) & (annual["area"] == area)]["TWh"]
            if values.empty or values.mean() == 0:
                continue
            boxes.append((values / values.mean() * 100 - 100).to_numpy(dtype="float64"))
            at.append(index + offset[key])
        if not boxes:
            continue
        drawn = ax.boxplot(boxes, positions=at, widths=0.3,
                           patch_artist=True, manage_ticks=False,
                           flierprops={"markersize": 2.0, "markeredgecolor": colour},
                           **_HORIZONTAL_BOXPLOT)
        for patch in drawn["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        for part in ("medians", "whiskers", "caps"):
            for artist in drawn[part]:
                artist.set_color("#333333")
                artist.set_linewidth(0.8)
        ax.plot([], [], color=colour, linewidth=6, alpha=0.75, label=label)

    ax.axvline(0, color="#555555", linestyle=":", linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels([display_name(a, zones) for a in material], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"difference from the area's own {len(timeseries.years)}-year mean, %", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"What {len(timeseries.years)} weather years do to demand and to hydro inflow",
                 fontsize=10)
    return _finish(fig, out_dir / "fig_climate_year_variability.png")


def figure_netload(out_dir: Path, curves: Dict[str, np.ndarray], zones: bool) -> str:
    """A net-load duration curve per area, all climate years pooled.

    Each panel keeps its own y axis: the shape is the message, and one shared
    axis would flatten every small area against Germany.
    """
    areas = sorted(curves)
    if not areas:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
        _empty_panel(ax, "Net load needs the timeseries files")
        return _finish(fig, out_dir / "fig_netload_duration.png")

    ncols = 4
    nrows = int(np.ceil(len(areas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, 2.1 * nrows + 0.8))
    flat = np.atleast_1d(axes).ravel()

    for ax, area in zip(flat, areas):
        values = np.sort(curves[area].astype("float64"))[::-1] * MW_TO_GW
        share = np.linspace(0, 100, len(values))
        ax.plot(share, values, color="#2a4d6e", linewidth=1.2)
        ax.fill_between(share, values, 0, where=values > 0, color="#2a4d6e", alpha=0.18)
        negative = values < 0
        if negative.any():
            ax.fill_between(share, values, 0, where=negative, color="#d98050", alpha=0.30)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_title(
            f"{display_name(area, zones)}\npeak {values[0]:,.0f} GW, "
            f"surplus {100.0 * negative.mean():.0f}% of hours",
            fontsize=7.5,
        )
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.2)
    for ax in flat[len(areas):]:
        ax.set_visible(False)

    fig.supxlabel("share of all modelled hours, %", fontsize=9)
    fig.supylabel("net load (demand - wind - solar), GW", fontsize=9)
    fig.suptitle("Net-load duration curves, every climate year pooled", fontsize=11)
    return _finish(fig, out_dir / "fig_netload_duration.png")


# ============================================================================
# Checks
# ============================================================================

def run_checks(workbook: Workbook, classification: Classification, transfer: Dict) -> List[Tuple[str, str, bool]]:
    """Structurally impossible values only.

    Nothing here is a judgement about whether a number is plausible. A check
    that fires on correct data every run is not strict, it is broken, so a
    capacity-to-peak ratio or a capacity-factor band -- both of which have
    defensible extremes in the shipped scenarios -- is shown sorted in a table
    instead of being judged here.
    """
    checks = []

    negatives = []
    io = workbook.p_gnu_io
    if "capacity" in io.columns and (pd.to_numeric(io["capacity"], errors="coerce") < 0).any():
        negatives.append("capacity")
    gnn = workbook.p_gnn
    if "transferCap" in gnn.columns and (pd.to_numeric(gnn["transferCap"], errors="coerce") < 0).any():
        negatives.append("transferCap")
    ne = workbook.n_emission
    if "value" in ne.columns and (ne["value"] < 0).any():
        negatives.append("emission factor")
    checks.append((
        "Negative capacity, transferCap or emission factor",
        "none" if not negatives else f"found in {', '.join(negatives)}",
        not negatives,
    ))

    gn = workbook.p_gn
    priced = gn[col_or(gn, "usePrice", 0.0).fillna(0.0) == 1] if not gn.empty else pd.DataFrame()
    bad_price = priced[pd.to_numeric(col_or(priced, "price", np.nan), errors="coerce").fillna(0) <= 0] \
        if not priced.empty else pd.DataFrame()
    checks.append((
        "Price at or below zero on a priced node",
        f"none of {len(priced)} priced node(s)" if bad_price.empty
        else f"{len(bad_price)}: {summarise(bad_price['node'].tolist())}",
        bad_price.empty,
    ))

    checks.append((
        "One-way transfer capacity (the other direction is zero)",
        f"none of {transfer.get('pairs', 0)} corridor(s)" if not transfer.get("one_way")
        else f"{transfer['one_way']} corridor(s)",
        not transfer.get("one_way"),
    ))

    unlabelled = classification.unlabelled
    by_reason = {}
    if not unlabelled.empty:
        for reason, group in unlabelled.groupby("reason", observed=True):
            by_reason[str(reason)] = float(group["capacity"].sum() * MW_TO_GW)
    real = {r: gw for r, gw in by_reason.items() if gw > 0}
    checks.append((
        "Unit capacity the technology grouping has no home for",
        "none -- every unit matched a group" if not real
        else "; ".join(f"{gw:,.1f} GW, {r}" for r, gw in sorted(real.items(), key=lambda x: -x[1])),
        not real,
    ))

    boundary = workbook.boundary
    if boundary.empty or "param_gnBoundaryTypes" not in boundary.columns:
        checks.append(("Storage row claiming a constant but carrying none", "no storage rows", True))
    else:
        upward = boundary[boundary["param_gnBoundaryTypes"].astype(str) == "upwardLimit"]
        claims = upward[col_or(upward, "useConstant", 0.0).fillna(0.0) == 1]
        missing = claims[col_or(claims, "constant", np.nan).isna()]
        checks.append((
            "Storage row claiming a constant but carrying none",
            f"none of {len(claims)} row(s)" if missing.empty
            else f"{len(missing)}: {summarise(missing['node'].tolist())}",
            missing.empty,
        ))

    # A flow with units but no profile generator is a known modelling gap, not a
    # defect in the build: nobody running a build can act on it, so it is
    # reported and not counted against the run. It stays in the table because it
    # is the kind of thing that otherwise goes unnoticed for a year.
    flows = set(str(f) for f in workbook.flow_unit.get("flow", pd.Series(dtype=str)).unique())
    without = sorted(flows - set(VRE_FAMILIES))
    checks.append((
        "Flow with units but no capacity-factor timeseries",
        "none" if not without
        else f"{', '.join(without)} -- units exist, no profile is built for them. "
             f"A known gap, not a data error",
        True,
    ))
    return checks


# ============================================================================
# The report
# ============================================================================

def md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    """A markdown table. Returns '' for no rows, so a caller can skip a section."""
    if not rows:
        return ""
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def _num(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:,.{digits}f}"


@dataclass
class Report:
    lines: List[str] = field(default_factory=list)

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, headers, rows) -> None:
        rendered = md_table(headers, rows)
        if rendered:
            self.add(rendered)
            self.add()

    def figure(self, name: str, caption: str) -> None:
        self.add(f"![{caption.split('.')[0]}]({name})")
        self.add(f"*{caption}*")
        self.add()

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def build_report(
    workbook: Workbook,
    classification: Classification,
    timeseries: Timeseries,
    out_dir: Path,
    zones: bool,
    ) -> Tuple[str, List[str]]:
    """The whole report.md, and the figure names written beside it."""
    level = "bidding zone" if zones else "country"
    report = Report()
    figures: List[str] = []

    per_area_transfer, matrix, transfer_facts = transfer_by_area(workbook, zones)
    energy_table, storage_facts = storage_energy(workbook, timeseries.storage_limit, zones)
    power_table = storage_power_by_area(workbook, zones)
    checks = run_checks(workbook, classification, transfer_facts)
    peak = _peak_demand(timeseries)

    capacity_factors = capacity_weighted_cf(timeseries, workbook, zones)
    carrier_data = {}
    for carrier, title, slug in CARRIERS:
        carrier_data[carrier] = {
            "title": title,
            "slug": slug,
            "capacity": capacity_by_area(classification, carrier, zones),
            "consumption": consumption_capacity(workbook, carrier, zones),
            "demand": annual_range(timeseries, f"demand_{carrier}"),
            "capacity_factor": capacity_factors if carrier == "elec" else pd.DataFrame(),
        }

    elec_total = _total_gw(carrier_data["elec"]["capacity"])
    elec_demand = carrier_data["elec"]["demand"]
    demand_total = float(elec_demand["mean"].sum()) if not elec_demand.empty else 0.0
    failing = [name for name, _, ok in checks if not ok]

    # ---- lead -------------------------------------------------------------
    report.add(f"# Input data summary -- {workbook.scenario}, {workbook.year}")
    report.add()
    lead = (
        f"This scenario builds a {len(set(country_of(zone_of(n)) for n in workbook.nodes))}-country, "
        f"{len(set(zone_of(n) for n in workbook.nodes))}-bidding-zone system with "
        f"{len(workbook.unit_unittype):,} units across {len(workbook.unittypes)} unit types and "
        f"{_num(elec_total)} GW of electricity capacity"
    )
    if demand_total:
        lead += (f", against {_num(demand_total, 0)} TWh/yr of electricity demand averaged over "
                 f"{len(timeseries.years)} climate year(s)")
    lead += (". " + ("Every structural check passed."
                    if not failing else
                    f"{len(failing)} structural check(s) need a look: {summarise(failing)}."))
    report.add(lead)
    report.add()

    # ---- one minute summary ----------------------------------------------
    report.add("## One minute summary")
    report.add()
    report.add(f"- **Scenario**: {workbook.scenario}, {workbook.year}, read from "
               f"`{workbook.path.name}` in `{workbook.path.parent.name}/`.")
    report.add(f"- **Reported by {level}**: "
               f"{'all 22 bidding zones' if zones else '16 countries; pass `--zones` to split them'}.")
    for carrier, data in carrier_data.items():
        total = _total_gw(data["capacity"])
        consumed = float(data["consumption"]["capacity"].sum() * MW_TO_GW) if not data["consumption"].empty else 0.0
        demand = float(data["demand"]["mean"].sum()) if not data["demand"].empty else 0.0
        if not (total or consumed or demand):
            report.add(f"- **{data['title']}**: nothing built in this scenario.")
            continue
        parts = [f"{_num(total)} GW producing"]
        if consumed:
            parts.append(f"{_num(consumed)} GW consuming")
        parts.append(f"{_num(demand, 0)} TWh/yr demanded" if demand else "no demand built")
        report.add(f"- **{data['title']}**: " + ", ".join(parts) + ".")
    steam = steam_demand_twh(workbook)
    if steam:
        report.add(f"- **Industrial steam**: {_num(steam, 0)} TWh/yr, carried as a flat hourly "
                   f"rate in the workbook rather than a timeseries, so it has no weather variation.")
    if storage_facts["constant_TWh"] or storage_facts["timeseries_TWh"]:
        report.add(f"- **Hydro storage**: {_num(storage_facts['constant_TWh'] + storage_facts['timeseries_TWh'])} TWh, "
                   f"of which only {_num(storage_facts['constant_TWh'])} TWh is a number in the workbook -- "
                   f"the rest is in a timeseries, see Storage.")
    if transfer_facts:
        report.add(f"- **Interconnection**: {_num(transfer_facts['total_GW'])} GW over "
                   f"{transfer_facts['pairs']} zone pair(s).")
    report.add(f"- **Checks**: {len(checks) - len(failing)} of {len(checks)} passed.")
    report.add()

    summary_rows = [
        ["Electricity capacity", f"{_num(elec_total)} GW", "`p_gnu_io`, elec output side"],
    ]
    if not elec_demand.empty:
        worst = elec_demand.loc[elec_demand["mean"].idxmax()]
        summary_rows.append([
            "Electricity demand",
            f"{_num(demand_total, 0)} TWh/yr mean, largest {worst['area']} at {_num(worst['mean'], 0)}",
            f"`ts_influx_elec`, {len(timeseries.years)} climate year(s)",
        ])
    if storage_facts["timeseries_TWh"]:
        summary_rows.append([
            "Hydro storage",
            f"{_num(storage_facts['constant_TWh'] + storage_facts['timeseries_TWh'])} TWh",
            "`p_gnBoundaryPropertiesForStates` + `ts_node_hydro_storage_limits`",
        ])
    price = co2_price(workbook)
    if price is not None:
        summary_rows.append(["CO2 price", f"{_num(price, 0)} EUR/t, one value for the run",
                             "`ts_emissionPriceChange`"])
    report.table(["What", "Value", "Where it is read from"], summary_rows)

    # ---- contents ---------------------------------------------------------
    report.add("## Contents")
    report.add()
    for carrier, data in carrier_data.items():
        report.add(f"1. [{data['title']}](#{data['slug']})")
    report.add("1. [Storage](#storage)")
    report.add("1. [Interconnection](#interconnection)")
    report.add("1. [Net load](#net-load)")
    report.add("1. [What the weather years do](#what-the-weather-years-do)")
    report.add("1. [Fuel, CO2 and emissions](#fuel-co2-and-emissions)")
    report.add("1. [Checks](#checks)")
    report.add("1. [Not summarized](#not-summarized)")
    report.add("1. [What this cannot see](#what-this-cannot-see)")
    report.add()
    if not zones:
        report.add("Country codes roll up the bidding zones: "
                   + ", ".join(f"**{c}** {n}" for c, n in sorted(COUNTRY_NAMES.items())
                               if c in {country_of(zone_of(x)) for x in workbook.nodes})
                   + ".")
        report.add()

    # ---- carrier sections -------------------------------------------------
    for carrier, data in carrier_data.items():
        figures.append(_carrier_section(report, workbook, data, carrier, timeseries,
                                       out_dir, zones))

    # ---- storage ----------------------------------------------------------
    report.add('<a id="storage"></a>')
    report.add()
    report.add("## Storage")
    report.add()
    _storage_section(report, storage_facts, energy_table, power_table, timeseries, zones)
    figures.append(figure_storage(out_dir, power_table, energy_table, zones))
    report.figure(
        figures[-1],
        "Left: the discharge power of storages whose energy capacity this model never states. "
        "Right: hydro reservoir and pumped-storage energy, split by which of the two places it "
        "is written. Take-away: a handful of countries carry the system's seasonal energy, and "
        "everyone else has power without stated duration.",
    )

    # ---- interconnection --------------------------------------------------
    report.add('<a id="interconnection"></a>')
    report.add()
    report.add("## Interconnection")
    report.add()
    _transfer_section(report, per_area_transfer, transfer_facts, peak, zones)
    figures.append(figure_interconnection(out_dir, matrix, per_area_transfer, peak, zones))
    report.figure(
        figures[-1],
        "Transfer capacity between areas, and each area's total as a share of its own peak "
        "demand. Take-away: who is connected to whom, and who depends on those connections "
        "rather than on its own plant.",
    )

    # ---- net load ---------------------------------------------------------
    report.add('<a id="net-load"></a>')
    report.add()
    report.add("## Net load")
    report.add()
    curves = netload_by_area(timeseries) if timeseries.available else {}
    _netload_section(report, curves, timeseries, zones)
    figures.append(figure_netload(out_dir, curves, zones))
    report.figure(
        figures[-1],
        "Net load -- demand minus wind and solar -- for every hour of every climate year, "
        "sorted highest to lowest, one panel per area on its own scale. Orange is the part of "
        "the year when wind and solar alone exceed demand. Take-away: the left edge is what "
        "firm capacity has to cover; the orange area is what has to be stored, exported or "
        "curtailed. It ignores storage, trade and dispatch, and treats VRE as never curtailed "
        "and never out of service, so it is a statement about the input data and not a forecast.",
    )

    # ---- weather years ----------------------------------------------------
    report.add('<a id="what-the-weather-years-do"></a>')
    report.add()
    report.add("## What the weather years do")
    report.add()
    material = _material_hydro_areas(timeseries)
    _weather_section(report, timeseries, material, zones)
    figures.append(figure_climate_variability(out_dir, timeseries, material, zones))
    report.figure(
        figures[-1],
        f"Each area's {len(timeseries.years) or 35} climate year(s) as a percentage of that area's "
        "own mean, so a small country and a large one are comparable. Take-away: a bad weather "
        "year barely moves electricity demand but can change hydro inflow by tens of per cent.",
    )

    # ---- prices -----------------------------------------------------------
    report.add('<a id="fuel-co2-and-emissions"></a>')
    report.add()
    report.add("## Fuel, CO2 and emissions")
    report.add()
    _price_section(report, workbook)

    # ---- checks -----------------------------------------------------------
    report.add('<a id="checks"></a>')
    report.add()
    report.add("## Checks")
    report.add()
    report.add("Structurally impossible values only. None of these should fire on correctly "
               "built data, so one that does is worth following up; a number merely being "
               "surprising is not checked here, it is shown in the tables above.")
    report.add()
    report.table(["Check", "Result"],
                 [[name, result if ok else f"**{result}**"] for name, result, ok in checks])

    # ---- not summarized ---------------------------------------------------
    report.add('<a id="not-summarized"></a>')
    report.add()
    report.add("## Not summarized")
    report.add()
    _not_summarized(report, workbook)

    # ---- what this cannot see --------------------------------------------
    report.add('<a id="what-this-cannot-see"></a>')
    report.add()
    report.add("## What this cannot see")
    report.add()
    _limits_section(report, workbook, timeseries, storage_facts, zones)

    report.add("---")
    report.add()
    report.add(f"*Written by `tools/input_data_summary.py` from "
               f"`{workbook.path.parent.name}/`. Every run overwrites this folder rather than "
               f"keeping the previous one.*")
    return report.text(), figures


def _total_gw(capacity: pd.DataFrame) -> float:
    if capacity is None or capacity.empty:
        return 0.0
    return float(capacity.to_numpy(dtype="float64").sum())


def _peak_demand(timeseries: Timeseries) -> Optional[pd.Series]:
    """Each area's highest single hour of electricity demand, in MW.

    Taken on the area's own coincident hours, not as the sum of its zones' separate
    maxima: a country does not hit every zone's peak in the same hour.
    """
    demand = timeseries.hourly.get("demand_elec")
    if demand is None or not timeseries.areas:
        return None
    return pd.Series(demand.max(axis=0), index=timeseries.areas, dtype="float64")


def _material_hydro_areas(timeseries: Timeseries) -> List[str]:
    if timeseries.annual.empty:
        return []
    inflow = timeseries.annual[timeseries.annual["key"] == "inflow_hydro"]
    if inflow.empty:
        return []
    means = inflow.groupby("area", observed=True)["TWh"].mean()
    return means[means >= MATERIAL_HYDRO_TWH].sort_values(ascending=False).index.tolist()


def _carrier_section(report, workbook, data, carrier, timeseries, out_dir, zones) -> str:
    report.add(f'<a id="{data["slug"]}"></a>')
    report.add()
    report.add(f"## {data['title']}")
    report.add()

    capacity, demand, consumption = data["capacity"], data["demand"], data["consumption"]
    produced = _total_gw(capacity)
    consumed = float(consumption["capacity"].sum() * MW_TO_GW) if not consumption.empty else 0.0

    if not produced and not consumed and demand.empty:
        report.add(f"Nothing in this scenario produces, consumes or demands {data['title'].lower()}. "
                   f"The section stays so that a later build can be compared against this one.")
        report.add()
    else:
        opening = f"{_num(produced)} GW of capacity produces {data['title'].lower()}"
        if consumed:
            opening += f" and {_num(consumed)} GW consumes it"
        if not demand.empty:
            opening += f", against {_num(float(demand['mean'].sum()), 0)} TWh/yr of demand"
        elif carrier in DEMAND_FAMILY:
            # A demand series is written for this carrier in a normal build, so
            # its absence here is about this folder, not about the carrier.
            opening += ", and its demand series could not be read"
        elif consumed or produced:
            opening += ", and nothing demands it directly"
        report.add(opening + ".")
        report.add()

    if capacity is not None and not capacity.empty:
        labels = [l for l in capacity.columns]
        rows = []
        ordered = capacity.sum(axis=1).sort_values(ascending=False)
        for area in ordered.index:
            row = [display_name(area, zones)]
            row += [_num(capacity.loc[area, l]) if capacity.loc[area, l] else "-" for l in labels]
            row.append(f"**{_num(ordered[area])}**")
            rows.append(row)
        totals = ["**total**"] + [f"**{_num(capacity[l].sum())}**" for l in labels]
        totals.append(f"**{_num(produced)}**")
        rows.append(totals)
        report.add(f"Production capacity, GW, output side only -- a unit that also consumes "
                   f"another carrier is counted once, here.")
        report.add()
        report.table([level_header(zones)] + labels + ["total"], rows)

    if not demand.empty:
        rows = [[display_name(r["area"], zones), _num(r["mean"], 1)]
                for _, r in demand.iterrows()]
        rows.append(["**total**", f"**{_num(float(demand['mean'].sum()), 1)}**"])
        report.add(f"Annual demand, TWh/yr, the mean over {len(timeseries.years)} climate year(s). "
                   f"How far individual years move is in "
                   f"[What the weather years do](#what-the-weather-years-do).")
        report.add()
        report.table([level_header(zones), "TWh/yr"], rows)
    elif carrier in DEMAND_FAMILY and timeseries.skipped:
        report.add(f"Demand needs the timeseries files, which were not read: {timeseries.skipped}.")
        report.add()
    elif carrier in DEMAND_FAMILY:
        # The family exists for this carrier but no file of it was found, which
        # is different from a carrier that never has one.
        report.add(f"No `{DEMAND_FAMILY[carrier]}_<year>.gdx` file was found in this folder, so "
                   f"there is no demand to report. The units above still exist.")
        report.add()
    elif carrier not in DEMAND_FAMILY:
        report.add(f"No hourly demand series exists for {data['title'].lower()} in this build -- "
                   f"no `ts_influx` family is written for it.")
        report.add()

    if not consumption.empty:
        rows = []
        pivot = consumption.pivot_table(index="area", columns="label", values="capacity",
                                       aggfunc="sum", fill_value=0.0) * MW_TO_GW
        for area in pivot.sum(axis=1).sort_values(ascending=False).index:
            rows.append([display_name(area, zones)]
                        + [_num(pivot.loc[area, c]) if pivot.loc[area, c] else "-"
                           for c in pivot.columns])
        report.add(f"Capacity that consumes {data['title'].lower()}, GW, by what it makes from it. "
                   f"These are dispatchable units, not demand: the model decides when they run, "
                   f"so they are not part of the demand figures above.")
        report.add()
        report.table([level_header(zones)] + list(pivot.columns), rows)

    if carrier == "elec" and not data["capacity_factor"].empty:
        factors = data["capacity_factor"]
        flows = sorted(factors["flow"].unique())
        rows = []
        for area in sorted(factors["area"].unique()):
            part = factors[factors["area"] == area].set_index("flow")
            rows.append([display_name(area, zones)] + [
                f"{part.loc[f, 'cf']:.0%} ({_num(part.loc[f, 'capacity_GW'])} GW)"
                if f in part.index else "-" for f in flows
            ])
        report.add(f"What the wind and solar fleets actually yield, averaged over "
                   f"{len(timeseries.years)} climate year(s) and weighted by each node's own "
                   f"installed capacity -- an unweighted mean would let a 200 MW zone count as "
                   f"much as a 12 GW one. This is the number that turns the GW above into the "
                   f"TWh below, and the reason two areas with the same capacity do not have the "
                   f"same net load.")
        report.add()
        report.table([level_header(zones)] + [f"{f} CF" for f in flows], rows)

    name = figure_carrier(out_dir, data["slug"], data["title"], capacity, demand,
                          consumption, zones)
    report.figure(
        name,
        f"{data['title']} capacity and annual demand, same area order in both panels. "
        f"Take-away: what each area has built for this carrier, read directly against what it "
        f"uses.",
    )
    return name


def level_header(zones: bool) -> str:
    return "zone" if zones else "country"


def _storage_section(report, facts, energy_table, power_table, timeseries, zones) -> None:
    total = facts["constant_TWh"] + facts["timeseries_TWh"]
    if total:
        report.add(f"Hydro reservoir and pumped storage hold **{_num(total)} TWh**, and the number "
                   f"comes from two places that must not be confused.")
        report.add()
        report.add(f"- **{_num(facts['constant_TWh'])} TWh** is a constant in "
                   f"`p_gnBoundaryPropertiesForStates`"
                   + (f" ({', '.join(f'{g} {_num(v)}' for g, v in sorted(facts['by_grid_constant'].items(), key=lambda x: -x[1]))})"
                      if facts["by_grid_constant"] else "") + ".")
        if facts["timeseries_TWh"]:
            report.add(f"- **{_num(facts['timeseries_TWh'])} TWh** is in "
                       f"`ts_node_hydro_storage_limits`, because {len(facts['timeseries_nodes'])} "
                       f"node(s) set `useTimeseries` and leave their constant empty: "
                       f"{summarise(facts['timeseries_nodes'])}. That value is identical in every "
                       f"climate year -- it is the reservoir's physical size, not a weather-driven "
                       f"quantity -- so it is read from one file and reported as one number.")
            share = 100.0 * facts["timeseries_TWh"] / total
            report.add()
            report.add(f"Reading the workbook alone would report {_num(facts['constant_TWh'])} TWh "
                       f"and silently omit {_num(share, 0)}% of the system's reservoir energy.")
        report.add()
    elif timeseries.skipped:
        report.add(f"Storage energy could only be read from the workbook: {timeseries.skipped}. "
                   f"The number below is incomplete, not a total.")
        report.add()

    if facts["grids_without_energy"]:
        report.add(f"**{', '.join(facts['grids_without_energy'])}** carry no energy capacity "
                   f"anywhere in this workbook -- no `upwardLimit` row is written for them, and "
                   f"neither `p_gn` nor `p_unit` carries an alternative. What is defined is their "
                   f"charge and discharge power, below. That is a property of the data rather than "
                   f"something missing from this report.")
        report.add()

    if power_table is not None and not power_table.empty:
        rows = []
        for area in power_table.sum(axis=1).sort_values(ascending=False).index:
            rows.append([display_name(area, zones)]
                        + [_num(power_table.loc[area, c]) if power_table.loc[area, c] else "-"
                           for c in power_table.columns])
        rows.append(["**total**"] + [f"**{_num(power_table[c].sum())}**" for c in power_table.columns])
        report.table([level_header(zones)] + list(power_table.columns), rows)


def _transfer_section(report, per_area, facts, peak, zones) -> None:
    if per_area is None or per_area.empty:
        report.add("No electricity transfer capacity is written in this scenario.")
        report.add()
        return
    report.add(f"{facts['pairs']} corridor(s) carry {_num(facts['total_GW'])} GW in total. Each "
               f"corridor is written twice, once per direction, and a corridor's capacity here is "
               f"the mean of its two directions -- adding them would count it twice, and taking "
               f"one would depend on which row the builder wrote first.")
    report.add()
    if facts["asymmetric"]:
        largest = ", ".join(f"{name} differs by {_num(spread, 0)} MW"
                            for name, spread in facts["largest_asymmetries"])
        report.add(f"{facts['asymmetric']} of {facts['pairs']} corridor(s) carry a different "
                   f"capacity each way, which is normal for this kind of data and is not flagged: "
                   f"{largest}.")
        report.add()
    if facts.get("transfer_loss") is not None:
        report.add(f"Every corridor carries the same transfer loss, "
                   f"{_num(100 * facts['transfer_loss'], 1)}%.")
        report.add()

    rows = []
    total = (per_area["cross_border_MW"] + per_area["inter_zonal_MW"]).sort_values(ascending=False)
    for area in total.index:
        share = None
        if peak is not None and area in peak.index and peak[area] > 0:
            share = 100.0 * total[area] / peak[area]
        rows.append([
            display_name(area, zones),
            _num(per_area.loc[area, "cross_border_MW"] * MW_TO_GW),
            _num(per_area.loc[area, "inter_zonal_MW"] * MW_TO_GW) if per_area.loc[area, "inter_zonal_MW"] else "-",
            _num(peak[area], 0) if peak is not None and area in peak.index else "-",
            f"{_num(share, 0)}%" if share is not None else "-",
        ])
    report.table([level_header(zones), "cross-border GW", "internal GW",
                  "peak demand MW", "capacity / peak"], rows)
    if peak is not None:
        report.add("Peak demand is the highest single hour across every climate year, taken on the "
                   "area's own coincident hours rather than by adding its zones' separate maxima. "
                   "A share above 100% is not an error: a small, well-connected area can have more "
                   "transfer capacity than its own peak load.")
        report.add()


def _netload_section(report, curves, timeseries, zones) -> None:
    if not curves:
        report.add(f"Net load needs the hourly timeseries, which were not read"
                   + (f": {timeseries.skipped}" if timeseries.skipped else "") + ".")
        report.add()
        return
    report.add(f"Net load is electricity demand minus wind and solar output, hour by hour, over "
               f"all {len(timeseries.years)} climate year(s) pooled -- "
               f"{len(next(iter(curves.values()))):,} hours per area. Wind and solar output is "
               f"installed capacity times the capacity factor in the build, so it assumes nothing "
               f"is ever curtailed or out of service: that makes VRE an upper bound and net load "
               f"a lower one. Hydro is not subtracted, because reservoir hydro is dispatchable.")
    report.add()
    rows = []
    for area, values in sorted(curves.items(), key=lambda kv: -kv[1].max()):
        gw = values.astype("float64") * MW_TO_GW
        rows.append([
            display_name(area, zones),
            _num(gw.max()), _num(float(np.percentile(gw, 1))), _num(gw.min()),
            f"{_num(100.0 * (gw < 0).mean(), 1)}%",
        ])
    report.table([level_header(zones), "peak GW", "1st percentile GW", "minimum GW",
                  "hours in surplus"], rows)
    report.add("The peak is what firm capacity, imports and storage together have to cover. The "
               "hours in surplus are hours when wind and solar alone exceed demand, so something "
               "has to absorb, export or curtail them.")
    report.add()


def _weather_section(report, timeseries, material, zones) -> None:
    if not timeseries.available:
        report.add(f"The climate years were not read"
                   + (f": {timeseries.skipped}" if timeseries.skipped else "") + ".")
        report.add()
        return

    rows = []
    for key, label in (("demand_elec", "electricity demand"), ("demand_dheat", "district heat"),
                       ("inflow_hydro", "hydro inflow")):
        part = timeseries.annual[timeseries.annual["key"] == key]
        if part.empty:
            continue
        spreads = []
        for area, group in part.groupby("area", observed=True):
            mean = group["TWh"].mean()
            if mean <= 0:
                continue
            spreads.append((area, 100 * (group["TWh"].min() / mean - 1),
                            100 * (group["TWh"].max() / mean - 1)))
        if not spreads:
            continue
        widest = max(spreads, key=lambda s: s[2] - s[1])
        median_low = float(np.median([s[1] for s in spreads]))
        median_high = float(np.median([s[2] for s in spreads]))
        rows.append([label, f"{_num(median_low, 0)}% / +{_num(median_high, 0)}%",
                     f"{widest[0]} at {_num(widest[1], 0)}% / +{_num(widest[2], 0)}%"])
    report.add(f"Over {len(timeseries.years)} climate year(s) ({min(timeseries.years)}-"
               f"{max(timeseries.years)}), as a percentage of each area's own mean:")
    report.add()
    report.table(["quantity", "typical range", "widest area"], rows)

    elec = annual_range(timeseries, "demand_elec")
    if not elec.empty:
        system = timeseries.annual[timeseries.annual["key"] == "demand_elec"].groupby(
            "year", observed=True)["TWh"].sum()
        peak_year = int(system.idxmax())
        agreeing = sum(
            1 for _, group in timeseries.annual[timeseries.annual["key"] == "demand_elec"]
            .groupby("area", observed=True)
            if len(group) and int(group.loc[group["TWh"].idxmax(), "year"]) == peak_year
        )
        report.add(f"Electricity demand is the least weather-sensitive quantity in the build, and "
                   f"its extremes agree across areas: {peak_year} is the highest-demand year "
                   f"system-wide and in {agreeing} of {elec['area'].nunique()} area(s).")
        report.add()

    correlations = _correlations(timeseries)
    if not correlations.empty:
        report.add("Does a low-resource year fall in the same year as a high-demand year in the "
                   "same area? Pearson correlation of each area's annual anomalies, over the "
                   "climate years. This is a table and not a scatter plot on purpose: demand's "
                   "spread is so much narrower than inflow's that a shared axis would draw demand "
                   "as a flat line whatever the correlation.")
        report.add()
        rows = [[display_name(r["area"], zones),
                 _num(r["hydro"], 2) if np.isfinite(r["hydro"]) else "-",
                 _num(r["vre"], 2) if np.isfinite(r["vre"]) else "-"]
                for _, r in correlations.iterrows()]
        report.table([level_header(zones), "r(demand, hydro inflow)", "r(demand, wind and solar)"],
                     rows)
        strong_hydro = int((correlations["hydro"].abs() >= 0.3).sum())
        negative_vre = int((correlations["vre"] <= -0.3).sum())
        total_areas = len(correlations)
        report.add(f"In this build, hydro inflow and demand move together weakly and "
                   f"inconsistently -- only {strong_hydro} of {total_areas} area(s) reach "
                   f"|r| >= 0.3. Wind and solar are different: {negative_vre} of {total_areas} "
                   f"area(s) are at or below -0.3, meaning a low-VRE year tends to be a "
                   f"high-demand year. If this scenario is ever run as one hand-picked year "
                   f"rather than all of them, that is the relationship worth picking against.")
        report.add()
    if material:
        left_out = [a for a in timeseries.areas if a not in material]
        report.add(f"The figure below shows the {len(material)} area(s) with at least "
                   f"{_num(MATERIAL_HYDRO_TWH, 0)} TWh/yr of mean inflow. The other "
                   f"{len(left_out)} are left out because a large percentage swing on a very small "
                   f"inflow says more about the denominator than about the system.")
        report.add()


def _correlations(timeseries: Timeseries) -> pd.DataFrame:
    """Per-area correlation of demand anomaly against resource anomaly."""
    annual = timeseries.annual
    if annual.empty:
        return pd.DataFrame()
    wide = annual.pivot_table(index=["area", "year"], columns="key", values="TWh").reset_index()
    if "demand_elec" not in wide.columns:
        return pd.DataFrame()
    rows = []
    for area, group in wide.groupby("area", observed=True):
        if len(group) < 3 or group["demand_elec"].std() == 0:
            continue
        entry = {"area": area}
        for key, name in (("inflow_hydro", "hydro"), ("vre", "vre")):
            if key in group.columns and group[key].std() > 0 and group[key].abs().sum() > 0:
                entry[name] = float(group["demand_elec"].corr(group[key]))
            else:
                entry[name] = float("nan")
        rows.append(entry)
    out = pd.DataFrame(rows)
    return out.sort_values("vre") if not out.empty else out


def _price_section(report, workbook) -> None:
    prices = fuel_prices(workbook)
    if not prices.empty:
        uniform = prices[prices["distinct"] == 1]
        varying = prices[prices["distinct"] > 1]
        if len(uniform) == len(prices):
            report.add("Every fuel costs the same in every zone, so this is a line rather than a "
                       "table: "
                       + ", ".join(f"{r['grid']} {_num(r['price'])}"
                                   for _, r in prices.iterrows())
                       + " EUR/MWh.")
        else:
            report.add("Fuel prices, EUR/MWh.")
            report.add()
            report.table(["fuel", "EUR/MWh", "same in every zone?"],
                         [[r["grid"], _num(r["price"]),
                           "yes" if r["distinct"] == 1 else f"no, {int(r['distinct'])} values"]
                          for _, r in prices.iterrows()])
        report.add()
        if len(varying) and len(uniform) == len(prices):
            pass

    price = co2_price(workbook)
    factors = emission_factors(workbook)
    if price is not None:
        report.add(f"CO2 costs {_num(price, 0)} EUR/t, one value for the whole run.")
        report.add()
    if not factors.empty:
        rows = []
        for _, row in factors.iterrows():
            carbon = None
            if price is not None and str(row["emission"]).lower() == "co2":
                carbon = row["factor"] * price
            fuel_price = None
            if not prices.empty:
                match = prices[prices["grid"].astype(str) == str(row["fuel"])]
                if len(match):
                    fuel_price = float(match["price"].iloc[0])
            rows.append([
                row["fuel"], row["emission"], _num(row["factor"], 3),
                _num(carbon, 1) if carbon is not None else "-",
                _num(fuel_price + carbon, 1) if (carbon is not None and fuel_price is not None) else "-",
            ])
        report.add("Emission intensity, and what the CO2 price adds to each fuel. The last column "
                   "is the fuel price plus its carbon cost -- the number that actually orders the "
                   "merit stack.")
        report.add()
        report.table(["fuel", "emission", "t/MWh", "carbon cost EUR/MWh",
                      "fuel + carbon EUR/MWh"], rows)


def _not_summarized(report, workbook) -> None:
    items = []
    uc = read_bb_sheet(pd.ExcelFile(workbook.path), "p_userconstraint")
    if not uc.empty:
        groups = uc["group"].nunique() if "group" in uc.columns else 0
        items.append(f"{groups} user constraint group(s) over {len(uc)} row(s) "
                     f"(`p_userconstraint`) -- scenario-specific shapes with nothing in common "
                     f"across builds, so they are counted and not interpreted.")
    if not workbook.p_unit.empty:
        items.append("Unit efficiency curves and availability (`p_unit`) -- technology parameters "
                     "rather than facts about countries at annual level.")
    for item in items:
        report.add(f"- {item}")
    report.add()


def _limits_section(report, workbook, timeseries, storage_facts, zones) -> None:
    report.add(f"The technology grouping in the capacity tables is this tool's own. The source "
               f"data names {len(workbook.unittypes)} unit types and no grouping of them, so units "
               f"are grouped by which grids they connect -- which is what keeps inflow-driven "
               f"pumped hydro counted as generation and closed-loop pumped hydro counted as "
               f"storage, where a rule keyed on the unit type's name would merge them. The check "
               f"on capacity with no home is the thing to watch when a new unit type appears.")
    report.add()
    report.add("There is no figure or table below zone level, at any flag setting, and no "
               "\"Nordics\" or \"CWE\" grouping, because nothing in the tracked source data "
               "defines one.")
    report.add()
    if timeseries.available:
        report.add(f"The numbers over climate years read the {len(timeseries.years)} real per-year "
                   f"files, never the `f01`/`f02`/`f03` forecast branches. Those branches are a "
                   f"per-hour quantile taken independently across the same years, not three "
                   f"coherent alternative years, so a claim about \"the driest year\" cannot be "
                   f"built from them.")
    else:
        report.add(f"**The timeseries sections are missing from this report**: "
                   f"{timeseries.skipped}. Annual demand, hydro inflow, net load, peak demand, the "
                   f"climate-year comparison, and the larger half of the hydro storage number all "
                   f"need those files. Everything read from `inputData.xlsx` -- capacity, "
                   f"interconnection capacity, prices, emissions and the structural checks -- is "
                   f"unaffected.")
    report.add()
    report.add("Nothing here comes from a solved model. The net-load curve ignores storage, trade "
               "and dispatch; VRE output assumes no curtailment and no outages; and capacity times "
               "hours is never converted into energy for dispatchable plant, because the input "
               "data cannot support that claim.")
    report.add()


# ============================================================================
# Entry point
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument(
        "built_folder", type=Path,
        help="a folder build_input_data.py wrote, e.g. input_ObservedTrends_2030",
    )
    parser.add_argument(
        "--zones", action="store_true",
        help="report the bidding zones instead of rolling them up into countries",
    )
    parser.add_argument(
        "--no-timeseries", action="store_true",
        help="skip every section that reads a .gdx file. Applied automatically, with a "
             "printed reason, when GAMS or gamsapi cannot be opened",
    )
    parser.add_argument(
        "--out-subdir", default="summary", metavar="NAME",
        help="subfolder of built_folder for report.md and its figures "
             "(default: summary). Overwritten on every run",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    folder: Path = args.built_folder
    xlsx_path = folder / "inputData.xlsx"

    if not folder.is_dir():
        print(f"No such folder: {folder}")
        return 2
    if not xlsx_path.is_file():
        print(f"No such file: {xlsx_path}")
        return 2

    print(f"Reading {xlsx_path} ...")
    try:
        workbook = load_workbook(xlsx_path)
    except Exception as exc:
        print(f"Could not read {xlsx_path}: {type(exc).__name__}: {exc}")
        return 1

    timeseries = Timeseries(skipped="--no-timeseries was given")
    if not args.no_timeseries:
        try:
            timeseries = read_timeseries(folder, workbook, args.zones)
        except Exception as exc:
            # A mid-sweep failure degrades the same way a missing GAMS install
            # does. The workbook half of the report is still worth writing.
            timeseries = Timeseries(skipped=f"the timeseries read failed "
                                            f"({type(exc).__name__}: {exc})")
    if timeseries.skipped:
        print(f"Timeseries sections skipped: {timeseries.skipped}")

    inflow_grids = None
    if not timeseries.annual.empty or not timeseries.storage_limit.empty:
        inflow_grids = _inflow_grids_from_data(folder, workbook, timeseries)
    classification = classify_capacity(workbook, inflow_grids)

    out_dir = folder / args.out_subdir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        text, figures = build_report(workbook, classification, timeseries, out_dir, args.zones)
        (out_dir / "report.md").write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write the report into {out_dir}: {exc}")
        return 1

    print(f"{len(figures)} figure(s) written")
    print(f"Report written: {out_dir / 'report.md'}")
    return 0


def _inflow_grids_from_data(folder: Path, workbook: Workbook, timeseries: Timeseries) -> set:
    """Which storage grids actually receive inflow, read from the build.

    Taken from the data rather than from a list of grid names: the scenario
    families do not agree on those names, and a literal set written against one
    of them drops the other's capacity into "no technology group".
    """
    grids = set()
    backend, _ = open_gdx_backend()
    if backend is not None:
        _, container = backend
        files = year_files(folder, "ts_influx_hydro")
        if files:
            records = _records(container, files[min(files)], "ts_influx")
            if records is not None and "grid" in records.columns:
                grids = {str(g) for g in records["grid"].unique()}
    return grids or set(HYDRO_INFLOW_GRIDS_FALLBACK)


if __name__ == "__main__":
    sys.exit(main())
