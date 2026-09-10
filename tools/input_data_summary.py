"""
input_data_summary.py -- what is in one built input-data folder, as a report.

Reads <built_folder>/inputData.xlsx and, unless told not to, the per-climate-year
ts_influx_*, ts_cf_* and ts_node_hydro_storage_limits_* GDX files beside it.
Writes one self-contained report.md, with its figures, into a subfolder of the
built folder -- so a colleague can be handed a folder and read it.

Usage:
    python tools/input_data_summary.py <built_folder> [--zones]
        [--no-timeseries] [--no-neighbours] [--out-subdir NAME]

Examples:
    python tools/input_data_summary.py input_ObservedTrends_2030
    python tools/input_data_summary.py input_ObservedTrends_2030 --zones
    python tools/input_data_summary.py input_tyndp2024_NationalTrends_2040

What it shows
-------------
Sixteen countries by default, the 22 bidding zones with --zones. A map of which
carriers each area models and whether anything demands them; then one section
per energy carrier -- electricity, district heat, hydrogen -- each with
production capacity, consumption capacity and annual demand. Then storage,
interconnection as a corridor map, a net-load duration curve, how much 35
weather years move the numbers, and fuel, CO2 and emission prices.

The two maps need ``tools/data/zone_shapes.geojson``, which
``tools/prepare_zone_geometry.py`` writes. Without it the rest of the report is
unaffected and the map panels say what they wanted -- the same documented
degradation as a missing GAMS install. Nothing here imports a geometry library:
the asset is read with ``json`` and drawn with matplotlib patches.

A carrier with no data still gets its section, saying so. That is deliberate:
during a data-adding phase an empty panel is the thing worth seeing, and a
section that disappears cannot tell anyone it is empty.

What it cannot see
------------------
Nothing below zone level -- there is no per-node figure at any flag setting. No
"Nordics" or "CWE" grouping: nothing in the tracked source data defines one, so
none is invented, and zone and country are the only two levels offered.

Where anything is inside an area. The maps place a zone's whole capacity at one
point, because the build states no location below the zone, and a country map
is drawn from its zones' outlines rather than a dissolved border -- adjacent
zones in the source do not share vertices, so a seam would have to be guessed.

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
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # no display on a build machine
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

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

#: Weather-driven output, and capacity that moves energy in time rather than
#: adding any. Both are subtracted from the headline to leave firm capacity.
#: Names, not positions, so a scenario missing one of them still splits right.
VRE_LABELS = frozenset({"wind onshore", "wind offshore", "solar"})
SHIFTING_LABELS = frozenset({"battery", "pumped storage", "demand response"})

HOURS_PER_YEAR = 8760

#: Identifiers whose meaning a duration cannot supply. The pumped-storage pair
#: is the one that matters: psOpen has natural inflow and psClosed does not, and
#: filing Norway's psOpen fleet under storage would hide its whole power system.
GRID_GLOSS = {
    "psOpen": "pumped storage fed by natural inflow, so it generates as well as shifts",
    "psClosed": "pumped storage with no inflow, which only returns what was put in",
    "H2": "hydrogen",
    "dheat": "district heat",
    "steam": "industrial steam",
}

#: Storage durations the residual demand is decomposed over, shortest first.
#:
#: Every window is a whole multiple of the one before it, and that is not
#: decoration. The decomposition subtracts each window's leftover from the next
#: one's, so a longer window must always cancel at least as much as a shorter
#: one. That holds only when the coarser blocks are unions of the finer ones:
#: with a calendar month of 730 hours against a week of 168 the blocks cut
#: across each other, a pure seasonal swing came out *higher* at the quarter
#: than at the day, and the difference between them was a negative bar.
#:
#: So these are four weeks and sixteen weeks, not a month and a quarter, and
#: they are labelled as what they are rather than rounded to a calendar the
#: arithmetic cannot honour.
DURATION_WINDOWS = [
    ("within a day", 24),
    ("within a week", 168),
    ("within four weeks", 672),
    ("within sixteen weeks", 2688),
    ("within the year", None),          # None: the whole series as one block
]
DURATION_REMAINDER = "left after the year"

#: Short timescales light, long timescales dark: the eye reads the dark end as
#: the hard part, which is what it is. Fixed, so two scenarios compare.
DURATION_STYLE = {
    "within a day": "#cfe3f2",
    "within a week": "#9ec5e8",
    "within four weeks": "#6b9fd0",
    "within sixteen weeks": "#3d6fa5",
    "within the year": "#2a4d6e",
    DURATION_REMAINDER: "#c46a4f",
}

#: Sharp at the width a markdown viewer renders an embedded PNG, without the
#: multi-megabyte files 300 dpi would produce for a 16:9 figure.
FIG_DPI = 150
FIG_WIDTH_IN = 9.0

#: Europe is taller than it is wide once projected, so the maps are the one
#: figure that is not sized from its row count.
MAP_HEIGHT_IN = 9.5

#: Written by tools/prepare_zone_geometry.py. Absent is not an error: the maps
#: say what they wanted and the rest of the report is unaffected.
MAP_ASSET = Path(__file__).resolve().parent / "data" / "zone_shapes.geojson"

NEIGHBOUR_FILL = "#efefef"
NEIGHBOUR_EDGE = "#d8d8d8"

#: What a carrier is doing in an area. Kept as words rather than booleans
#: because "a node exists but nothing demands it" is a third thing, and it is
#: the one worth seeing.
PRESENCE_ABSENT = "absent"
PRESENCE_NODE = "node"
PRESENCE_DEMAND = "demand"
PRESENCE_UNKNOWN = "unknown"

PRESENCE_STYLE = {
    PRESENCE_DEMAND: ("#33333a", ""),
    PRESENCE_NODE: ("#ffffff", "///"),
    PRESENCE_UNKNOWN: ("#b9b9c0", ""),
    PRESENCE_ABSENT: ("#ffffff", ""),
}
PRESENCE_LABEL = {
    PRESENCE_DEMAND: "modelled, has demand",
    PRESENCE_NODE: "modelled, nothing demands it",
    PRESENCE_UNKNOWN: "modelled, demand not read",
    PRESENCE_ABSENT: "not modelled here",
}

#: One fill per carrier combination. See _blend_for for why this is a table and
#: not arithmetic. Ordered so that adding a carrier deepens the fill.
CARRIER_BLEND = {
    frozenset({"elec"}): "#a8cbe8",
    frozenset({"dheat"}): "#e8b48f",
    frozenset({"H2"}): "#dda8c8",
    frozenset({"elec", "dheat"}): "#7fb3a8",
    frozenset({"elec", "H2"}): "#a98fc4",
    frozenset({"dheat", "H2"}): "#cf9099",
    frozenset({"elec", "dheat", "H2"}): "#6b86b8",
}
CARRIER_BLEND_UNKNOWN = "#e4e4e8"

#: Corridor line widths, in points, for the thinnest and thickest drawn. Width
#: is linear in capacity between them: a square root would flatter the small
#: corridors, and the legend would then be measuring something else.
CORRIDOR_WIDTH_MIN = 0.6
CORRIDOR_WIDTH_MAX = 6.0
CORRIDOR_COLOUR = "#2a4d6e"
CORRIDOR_INTERNAL_COLOUR = "#c46a4f"

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

    # Which other carriers have corridors at all. A reader of the hydrogen
    # section cannot otherwise tell whether its zones are connected to each
    # other or are 22 islands, and in the shipped scenarios they are islands.
    other_grids = {}
    for grid, part in gnn.groupby(gnn["grid"].astype(str), observed=True):
        if grid == "elec":
            continue
        # Counted on node pairs, not zone pairs: the three Helsinki-area
        # district-heating links are one zone pair and three corridors, and
        # calling them one would be wrong in the direction of reassurance.
        node_pairs = {tuple(sorted((str(a), str(b))))
                      for a, b in zip(part["from_node"], part["to_node"])}
        other_grids[grid] = {
            "corridors": len(node_pairs),
            "between_areas": len({p for p in node_pairs if zone_of(p[0]) != zone_of(p[1])}),
        }

    facts = {
        "pairs": int(len(per_pair)),
        "asymmetric": int(len(asymmetric)),
        "one_way": int(len(one_way)),
        "other_grids": other_grids,
        "carrier_grids": sorted(set(gnn["grid"].astype(str))),
        # Which carriers exist at all. A carrier with no nodes has no corridors
        # either, and saying so would report an absence the source data states
        # on purpose -- which this project does not treat as a defect.
        "node_grids": (sorted(set(workbook.p_gn["grid"].astype(str)))
                       if not workbook.p_gn.empty and "grid" in workbook.p_gn.columns else []),
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


def model_permissions(workbook: Workbook) -> Dict:
    """What the build lets a unit do, as counts.

    Counts and not conclusions, per the project's reporting rule: whether an
    unconstrained fleet is right for the question being asked is the reader's
    call, not this tool's. Zero is a good answer and worth printing -- every one
    of these is zero or all in the shipped scenarios, and a reader who assumed
    otherwise would misread every peak in the report.
    """
    io = workbook.p_gnu_io
    unit = workbook.p_unit
    gn = workbook.p_gn
    facts: Dict = {}

    def rows_with_any(columns: Sequence[str]) -> int:
        """Rows where any of these columns is set -- not the sum per column.

        A row carrying both an up and a down ramp cost is one row. Adding the
        two column counts reports 106 of 903 where the answer is 53.
        """
        present = [c for c in columns if c in io.columns]
        if not present or io.empty:
            return 0
        hit = pd.Series(False, index=io.index)
        for column in present:
            hit |= pd.to_numeric(io[column], errors="coerce").fillna(0).ne(0)
        return int(hit.sum())

    ramp_columns = [c for c in ("maxRampUp", "maxRampDown") if c in io.columns]
    facts["ramp_limited_rows"] = rows_with_any(("maxRampUp", "maxRampDown"))
    facts["ramp_columns_present"] = bool(ramp_columns)
    facts["ramp_cost_rows"] = rows_with_any(("rampUpCost", "rampDownCost"))
    facts["rows"] = int(len(io))

    if not unit.empty and "availability" in unit.columns:
        availability = pd.to_numeric(unit["availability"], errors="coerce")
        facts["units"] = int(len(unit))
        facts["derated_units"] = int((availability.fillna(1.0) < 1.0).sum())
    else:
        facts["units"] = int(len(unit))
        facts["derated_units"] = 0

    if not gn.empty and "boundStart" in gn.columns:
        facts["bound_start_nodes"] = int(
            pd.to_numeric(gn["boundStart"], errors="coerce").fillna(0).eq(1).sum())
        facts["nodes"] = int(len(gn))
    else:
        facts["bound_start_nodes"] = 0
        facts["nodes"] = int(len(gn))

    # Not read by load_workbook: it is empty in every shipped scenario, and the
    # fact that it is empty is exactly what this section reports.
    try:
        commitment = read_bb_sheet(pd.ExcelFile(workbook.path), "effLevelGroupUnit")
        facts["commitment_rows"] = int(len(commitment.dropna(how="all")))
    except Exception:                              # noqa: BLE001 -- a count, not a crash
        facts["commitment_rows"] = None
    return facts


def firm_and_shifting(capacity: pd.DataFrame) -> pd.DataFrame:
    """Split producing capacity into what supplies energy and what moves it in time.

    A battery, a pumped store and a demand-response block are all counted in the
    headline capacity, and none of them adds a megawatt-hour to the year: they
    move one from an hour that had it to an hour that did not. Against a peak
    they still count, which is why they are a column of their own rather than
    dropped.
    """
    if capacity is None or capacity.empty:
        return pd.DataFrame()
    shifting_columns = [c for c in capacity.columns if c in SHIFTING_LABELS]
    vre_columns = [c for c in capacity.columns if c in VRE_LABELS]
    firm_columns = [c for c in capacity.columns
                    if c not in SHIFTING_LABELS and c not in VRE_LABELS]
    return pd.DataFrame({
        "firm_GW": capacity[firm_columns].sum(axis=1) if firm_columns else 0.0,
        "shifting_GW": capacity[shifting_columns].sum(axis=1) if shifting_columns else 0.0,
        "vre_GW": capacity[vre_columns].sum(axis=1) if vre_columns else 0.0,
    }, index=capacity.index)


def potential_vre_share(capacity_factors: pd.DataFrame, demand: pd.DataFrame) -> pd.Series:
    """Wind and solar energy at their own capacity factors, over annual demand.

    Every term is already in the report and the product is never taken, so a
    reader cannot tell an area short of energy from one whose problem is only
    timing. Above 100% the area could in principle make its whole year's
    electricity from wind and sun, and what it lacks is the hour-by-hour match.
    """
    if capacity_factors is None or capacity_factors.empty or demand is None or demand.empty:
        return pd.Series(dtype="float64")
    energy = capacity_factors.assign(
        TWh=capacity_factors["capacity_GW"] * capacity_factors["cf"] * HOURS_PER_YEAR / 1000.0
    ).groupby("area", observed=True)["TWh"].sum()
    annual = demand.set_index("area")["mean"]
    shared = energy.index.intersection(annual.index)
    share = 100.0 * energy.loc[shared] / annual.loc[shared]
    share = share.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    # The system's own share, from the totals rather than an average of the
    # parts: a mean of percentages would weight Latvia like Germany.
    if len(shared) and float(annual.loc[shared].sum()) > 0:
        share["system"] = 100.0 * float(energy.loc[shared].sum()) / float(annual.loc[shared].sum())
    return share


def storage_energy_from_ratio(workbook: Workbook, zones: bool) -> Tuple[pd.DataFrame, Dict]:
    """Energy capacity of the storages whose size is written as a ratio.

    ``p_gnu_io.upperLimitCapacityRatio`` is v_state units per MW of the unit's
    capacity, so where ``energyStoredPerUnitOfState`` is 1 -- which every
    storage node in the shipped scenarios sets -- v_state is MWh, the ratio is
    the storage's duration in hours, and energy is power times it.

    This is the alternative the report used to say did not exist. It looked at
    ``p_gn`` and ``p_unit`` for a second way to state a storage's size and,
    finding none, concluded there was none; the ratio was in ``p_gnu_io`` all
    along, on the input side of every discharge unit.

    The ratio is a per-grid constant rather than a per-area quantity -- one
    number for every battery in the build -- so the duration is reported once
    per grid and the energy per area is that duration times the area's power.
    """
    io = workbook.p_gnu_io
    facts: Dict = {"by_grid": {}, "total_TWh": 0.0, "unconvertible": []}
    needed = {"grid", "node", "capacity", "upperLimitCapacityRatio"}
    if io.empty or not needed <= set(io.columns):
        return pd.DataFrame(), facts

    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    ratio = pd.to_numeric(io["upperLimitCapacityRatio"], errors="coerce")
    rows = io[active & ratio.notna() & (ratio > 0)].copy()
    if rows.empty:
        return pd.DataFrame(), facts
    rows["hours"] = ratio[rows.index]
    rows["power_MW"] = pd.to_numeric(rows["capacity"], errors="coerce").fillna(0.0)

    # v_state is only MWh when energyStoredPerUnitOfState says so. Where it is
    # absent or zero -- which this project reads as "not set" -- the ratio
    # cannot be turned into energy, and saying so is better than assuming 1.
    gn = workbook.p_gn
    factor = pd.Series(1.0, index=rows.index)
    if not gn.empty and {"grid", "node", "energyStoredPerUnitOfState"} <= set(gn.columns):
        lookup = (
            gn.assign(key=list(zip(gn["grid"].astype(str), gn["node"].astype(str))))
            .set_index("key")["energyStoredPerUnitOfState"]
        )
        lookup = pd.to_numeric(lookup, errors="coerce")
        lookup = lookup[~lookup.index.duplicated()]
        keys = list(zip(rows["grid"].astype(str), rows["node"].astype(str)))
        factor = pd.Series([lookup.get(k, np.nan) for k in keys], index=rows.index)

    convertible = factor.notna() & (factor > 0)
    facts["unconvertible"] = sorted(str(n) for n in rows.loc[~convertible, "node"])
    rows = rows[convertible]
    if rows.empty:
        return pd.DataFrame(), facts
    rows["MWh"] = rows["power_MW"] * rows["hours"] * factor[rows.index]
    rows["label"] = [storage_label(str(g)) for g in rows["grid"]]
    rows["area"] = [area_of(n, zones) for n in rows["node"]]

    for grid, part in rows.groupby(rows["grid"].astype(str), observed=True):
        hours = sorted({float(h) for h in part["hours"]})
        facts["by_grid"][grid] = {
            "hours": hours[0] if len(hours) == 1 else None,
            "hours_seen": hours,
            "power_GW": float(part["power_MW"].sum() * MW_TO_GW),
            "TWh": float(part["MWh"].sum() * MWH_TO_TWH),
            "nodes": int(len(part)),
        }
    facts["total_TWh"] = float(rows["MWh"].sum() * MWH_TO_TWH)
    return rows[["grid", "node", "label", "area", "power_MW", "hours", "MWh"]], facts


def explain_grids(columns: Sequence[str], ratio_facts: Optional[Dict]) -> str:
    """Define the raw identifiers a table uses as headers.

    ``to psClosed`` and ``to heatStorXL`` are GAMS set elements, not English.
    A reader who has not opened the workbook cannot know that one is pumped
    storage without inflow and the other a seasonal pit, and the difference
    between psOpen and psClosed is the whole Norwegian hydro fleet.

    The duration is derived rather than written down, so a scenario that
    changes a battery from two hours to four says so by itself.
    """
    durations = {grid: entry["hours"]
                 for grid, entry in (ratio_facts or {}).get("by_grid", {}).items()
                 if entry.get("hours")}
    parts = []
    for column in columns:
        grid = str(column)[3:] if str(column).startswith("to ") else str(column)
        hours = durations.get(grid)
        if grid in GRID_GLOSS:
            parts.append(f"`{grid}` is {GRID_GLOSS[grid]}")
        elif hours:
            parts.append(f"`{grid}` is a {_num(hours, 0)}-hour {storage_label(grid)}")
    if not parts:
        return ""
    return parts[0][0].upper() + parts[0][1:] + ("; " + "; ".join(parts[1:]) if parts[1:] else "") + "."


def duration_by_label(facts: Dict) -> Dict[str, float]:
    """``{'battery': 4.0}`` for the labels whose grids all share one duration.

    A label covering two grids of different durations gets no entry: the legend
    would then be claiming a number that is not true of half the bar.
    """
    by_label: Dict[str, set] = {}
    for grid, entry in facts.get("by_grid", {}).items():
        by_label.setdefault(storage_label(grid), set()).update(entry["hours_seen"])
    return {label: hours.pop() for label, hours in by_label.items() if len(hours) == 1}


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


def _residue(values: np.ndarray, window: Optional[int]) -> float:
    """Net load left over when it may be cancelled freely within ``window`` hours.

    ``None`` means the whole series as a single block. Blocks are fixed rather
    than sliding: a sliding window would let the same surplus hour cancel two
    different deficits, and the answer would depend on where the year is cut in
    a way no reader could check.
    """
    if window is None:
        return float(max(values.sum(), 0.0))
    if window <= 1:
        return float(np.maximum(values, 0.0).sum())
    edges = np.arange(0, len(values), window)
    # reduceat handles the ragged final block: a week does not divide a year,
    # and 52 blocks plus a remainder is the honest split.
    return float(np.maximum(np.add.reduceat(values, edges), 0.0).sum())


def residual_by_timescale(curves: Dict[str, np.ndarray], timeseries: Timeseries) -> Dict:
    """How much of the residual demand storage of each duration could remove.

    Net load summed inside a window lets surplus hours pay for deficit hours
    within it, which is what a perfect, lossless, free store of that duration
    would do. The difference between two windows is the energy that only the
    longer one can reach; what survives a whole year is what no amount of
    storage can move, because the energy is not there to move.

    It is an upper bound on what storage can do and therefore a lower bound on
    what else is needed: nothing here is charged an efficiency, a power limit or
    a cost.
    """
    if not curves or not timeseries.available or not timeseries.years:
        return {}
    hours = timeseries.hours_per_year
    n_years = len(timeseries.years)
    if not hours or n_years < 1:
        return {}

    windows = [1] + [w for _, w in DURATION_WINDOWS]
    labels = [label for label, _ in DURATION_WINDOWS] + [DURATION_REMAINDER]

    per_area_year: Dict[str, np.ndarray] = {}
    usable = n_years * hours
    for area, values in curves.items():
        # A short series would raise on the reshape, and this tool does not
        # raise once it has started reporting -- it says less instead.
        if len(values) < usable:
            continue
        block = np.asarray(values[:usable], dtype="float64").reshape(n_years, hours)
        residues = np.array([[_residue(block[y], w) for w in windows] for y in range(n_years)])
        # Segment i is what window i+1 removes that window i did not. Nested
        # windows make this non-negative by construction; the clip is against
        # floating-point dust, not against a real negative.
        segments = np.clip(np.diff(residues, axis=1) * -1.0, 0.0, None)
        per_area_year[area] = np.column_stack([segments, residues[:, -1]]) * MWH_TO_TWH

    if not per_area_year:
        return {}
    system = np.sum(np.stack(list(per_area_year.values())), axis=0)
    by_area = pd.DataFrame(
        {area: values.sum(axis=0) / n_years for area, values in per_area_year.items()},
        index=labels,
    ).T
    by_year = pd.DataFrame(system, index=list(timeseries.years), columns=labels)
    return {"by_area": by_area, "by_year": by_year, "labels": labels}


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


def _polygon_path(polygon: List[List[List[float]]]):
    """One polygon, holes and all, as a matplotlib path.

    A Polygon patch cannot hold an interior ring, so an enclave would be filled
    over. A compound Path can, and fills correctly under the default rule.
    """
    vertices: List = []
    codes: List = []
    for ring in polygon:
        projected = _project([p[0] for p in ring], [p[1] for p in ring])
        points = list(zip(*projected))
        if len(points) < 3:
            continue
        vertices.extend(points)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    if not vertices:
        return None
    return MplPath(vertices, codes)


def _project(lon, lat):
    """Lambert conformal conic, because Europe is tall.

    Plotted as raw degrees, Norway and Finland are stretched sideways by more
    than a factor of two against Spain -- the map would be legible but wrong in
    a way a reader would read as data. Standard parallels at 40N and 65N put the
    distortion where the model has least to say.
    """
    rad = math.pi / 180.0
    phi1, phi2, phi0 = 40.0 * rad, 65.0 * rad, 52.0 * rad
    lambda0 = 15.0 * rad
    n = (math.log(math.cos(phi1) / math.cos(phi2))
         / math.log(math.tan(math.pi / 4 + phi2 / 2) / math.tan(math.pi / 4 + phi1 / 2)))
    f = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** n / n

    lon = np.asarray(lon, dtype="float64") * rad
    lat = np.clip(np.asarray(lat, dtype="float64") * rad, -math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)
    rho = f / np.tan(math.pi / 4 + lat / 2) ** n
    rho0 = f / math.tan(math.pi / 4 + phi0 / 2) ** n
    theta = n * (lon - lambda0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


@dataclass
class ZoneShapes:
    """The map asset, or the reason there is not one."""
    by_zone: Dict[str, List] = field(default_factory=dict)
    neighbours: List = field(default_factory=list)
    missing: str = ""

    @property
    def available(self) -> bool:
        return bool(self.by_zone)


def load_zone_shapes(path: Path = None) -> ZoneShapes:
    """The zone boundaries, or an empty result saying why there are none.

    A missing asset is documented degradation, not failure: the rest of the
    report is unaffected and the maps say what they wanted.
    """
    path = path or MAP_ASSET
    if not path.exists():
        return ZoneShapes(missing=f"no map asset at {path.name}; run tools/prepare_zone_geometry.py")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return ZoneShapes(missing=f"the map asset could not be read ({error})")

    by_zone: Dict[str, List] = {}
    neighbours: List = []
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        polygons = feature.get("geometry", {}).get("coordinates", [])
        if properties.get("role") == "model":
            by_zone.setdefault(str(properties.get("zone")), []).extend(polygons)
        else:
            neighbours.extend(polygons)
    if not by_zone:
        return ZoneShapes(missing="the map asset carries no model zones")
    return ZoneShapes(by_zone=by_zone, neighbours=neighbours)


def _ring_centroid(ring) -> Tuple[float, float, float]:
    """``(x, y, area)`` of one projected ring, by the shoelace formula.

    Area-weighted rather than a mean of vertices: a coastline carries far more
    vertices than an inland border, so a vertex mean drags every Nordic label
    out to sea.
    """
    x, y = _project([p[0] for p in ring], [p[1] for p in ring])
    if len(x) < 3:
        return 0.0, 0.0, 0.0
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = float(cross.sum()) / 2.0
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean()), 0.0
    cx = float(((x[:-1] + x[1:]) * cross).sum() / (6.0 * area))
    cy = float(((y[:-1] + y[1:]) * cross).sum() / (6.0 * area))
    return cx, cy, abs(area)


def _polygons_centroid(polygons: List) -> Optional[Tuple[float, float]]:
    """The centroid of a zone, or of a whole country, weighted by area."""
    total = 0.0
    sx = sy = 0.0
    for polygon in polygons:
        if not polygon:
            continue
        cx, cy, area = _ring_centroid(polygon[0])
        sx += cx * area
        sy += cy * area
        total += area
    if total <= 0:
        return None
    return sx / total, sy / total


def context_polygons(shapes: ZoneShapes, drawn: Sequence[str], zones: bool) -> List:
    """Everything drawn in grey: the neighbour ring, and zones this run does not report.

    The asset carries every zone the crosswalk names, which is more than any one
    scenario builds. A zone that is in the asset but not in this build would
    otherwise be drawn neither as data nor as context, leaving a hole where
    Portugal is.
    """
    out = list(shapes.neighbours)
    for zone, polygons in shapes.by_zone.items():
        if (zone if zones else country_of(zone)) not in drawn:
            out.extend(polygons)
    return out


def shapes_for_areas(shapes: ZoneShapes, areas: Sequence[str], zones: bool) -> Dict[str, List]:
    """Geometry per reported area -- one zone, or every zone of a country.

    Country-level maps are drawn from the same zone polygons rather than a
    dissolved outline. Adjacent zones in the source do not share vertices, so
    a union without a geometry library would have to invent the seam; drawing
    the parts in one colour shows the same shape and says what the model
    actually solves.
    """
    out: Dict[str, List] = {}
    for zone, polygons in shapes.by_zone.items():
        area = zone if zones else country_of(zone)
        if area in areas:
            out.setdefault(area, []).extend(polygons)
    return out


def carrier_presence(workbook: Workbook, timeseries: Timeseries, zones: bool) -> pd.DataFrame:
    """Per area and carrier: is there a node, and does anything demand it.

    Presence comes from ``p_gn``'s own ``grid`` column rather than from the node
    name: a Finnish district-heating node is ``FI00_dheat_HKI``, and splitting
    that on underscores gives ``dheat_HKI``, which is not a grid.
    """
    states = pd.DataFrame(
        PRESENCE_ABSENT,
        index=sorted({area_of(n, zones) for n in workbook.nodes}),
        columns=[grid for grid, _, _ in CARRIERS],
        dtype="object",
    )
    gn = workbook.p_gn
    if gn.empty or "grid" not in gn.columns or "node" not in gn.columns:
        return states

    demanded: Dict[str, set] = {}
    if timeseries.available and not timeseries.annual.empty:
        annual = timeseries.annual
        for carrier in states.columns:
            key = f"demand_{carrier}"
            rows = annual[(annual["key"] == key) & (annual["TWh"].abs() > 0)]
            demanded[carrier] = set(rows["area"].astype(str))

    for carrier in states.columns:
        nodes = gn.loc[gn["grid"].astype(str) == carrier, "node"].astype(str)
        for area in {area_of(n, zones) for n in nodes}:
            if area not in states.index:
                continue
            if not timeseries.available:
                states.loc[area, carrier] = PRESENCE_UNKNOWN
            elif area in demanded.get(carrier, set()):
                states.loc[area, carrier] = PRESENCE_DEMAND
            else:
                states.loc[area, carrier] = PRESENCE_NODE
    return states


def _blend_for(present: Sequence[str]) -> str:
    """The fill for one carrier set.

    Explicitly per combination rather than an arithmetic blend of the three
    carrier colours: mixing a blue and an orange gives grey whichever way it is
    done, and three of the seven combinations came out the same muddy tone. The
    palette below is ordered so that adding a carrier deepens the fill, which is
    what a blend was wanted for, and every combination keeps its own colour
    whether or not another scenario has it -- the rule ``TECH_STYLE`` already
    follows.
    """
    return CARRIER_BLEND.get(frozenset(present), CARRIER_BLEND_UNKNOWN)


def figure_carrier_map(
    out_dir: Path,
    shapes: ZoneShapes,
    presence: pd.DataFrame,
    zones: bool,
    neighbours: bool = True,
    ) -> str:
    """Which carriers each area models, and whether anything demands them.

    The fill answers the first question and a three-cell chip the second. The
    chip is what makes the figure worth drawing: in a build where every zone has
    a hydrogen node, the fills are nearly uniform and the chips are hollow
    everywhere, which is the whole finding.
    """
    path = out_dir / "fig_carrier_map.png"
    if not shapes.available or presence.empty:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
        _empty_panel(ax, shapes.missing or "No nodes to map")
        return _finish(fig, path)

    geometry = shapes_for_areas(shapes, list(presence.index), zones)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, MAP_HEIGHT_IN))

    if neighbours:
        context = context_polygons(shapes, list(presence.index), zones)
        patches = [p for p in (_polygon_path(poly) for poly in context) if p]
        ax.add_collection(PatchCollection(
            [PathPatch(p) for p in patches],
            facecolor=NEIGHBOUR_FILL, edgecolor=NEIGHBOUR_EDGE, linewidth=0.4, zorder=1))

    for area, polygons in geometry.items():
        present = [c for c in presence.columns if presence.loc[area, c] != PRESENCE_ABSENT]
        patches = [p for p in (_polygon_path(poly) for poly in polygons) if p]
        ax.add_collection(PatchCollection(
            [PathPatch(p) for p in patches],
            facecolor=_blend_for(present), edgecolor="#ffffff", linewidth=0.5, zorder=2))

    _draw_area_chips(ax, geometry, presence)
    _finish_map(ax)
    _carrier_map_legend(ax, presence)
    ax.set_title(_carrier_map_claim(presence), fontsize=10)
    return _finish(fig, path)


def _carrier_map_claim(presence: pd.DataFrame) -> str:
    """A title that states the finding when there is one, and labels when there is not.

    Computed from the data rather than written down, so it cannot go stale
    against a scenario that fixes what it points at: the moment something
    demands the hydrogen, this stops saying nothing does.
    """
    titles = {grid: title for grid, title, _ in CARRIERS}
    for carrier in presence.columns:
        modelled = int((presence[carrier] != PRESENCE_ABSENT).sum())
        hollow = int((presence[carrier] == PRESENCE_NODE).sum())
        if modelled and hollow == modelled:
            return (f"{titles.get(carrier, carrier)} is modelled in {modelled} areas "
                    f"and demanded in none")
    return "What each area models, and what demands it"


def _draw_area_chips(ax, geometry: Dict[str, List], presence: pd.DataFrame) -> None:
    """One small cell per carrier, at each area's centroid."""
    centroids = {area: _polygons_centroid(polygons) for area, polygons in geometry.items()}
    drawn = [c for c in centroids.values() if c]
    if not drawn:
        return
    span = max(max(x for x, _ in drawn) - min(x for x, _ in drawn), 1e-6)
    cell = span / 52.0
    carriers = list(presence.columns)

    for area, centre in centroids.items():
        if centre is None:
            continue
        cx, cy = centre
        width = cell * len(carriers)
        for index, carrier in enumerate(carriers):
            state = presence.loc[area, carrier]
            x = cx - width / 2 + index * cell
            face, hatch = PRESENCE_STYLE[state]
            ax.add_patch(Rectangle(
                (x, cy - cell / 2), cell, cell,
                facecolor=face, edgecolor="#33333a", linewidth=0.5,
                hatch=hatch, zorder=4))
        ax.text(cx, cy + cell * 0.85, area, fontsize=7, ha="center", va="bottom",
                zorder=5, color="#1a1a1f",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#ffffff")])


def _finish_map(ax) -> None:
    """Every map panel ends the same way: square, unframed, autoscaled.

    Anchored north because an equal-aspect map letterboxes inside whatever cell
    it is given, and a panel beside it sized by its row count is often taller.
    Left centred, the slack would sit between the title and the map.
    """
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")


def _carrier_map_legend(ax, presence: pd.DataFrame) -> None:
    """Two legends: what a fill means, and what a chip cell means."""
    seen = []
    for area in presence.index:
        present = tuple(c for c in presence.columns if presence.loc[area, c] != PRESENCE_ABSENT)
        if present and present not in seen:
            seen.append(present)
    titles = {grid: title for grid, title, _ in CARRIERS}
    fills = [mpatches.Patch(facecolor=_blend_for(combination), edgecolor="#ffffff",
                            label=" + ".join(titles.get(c, c) for c in combination))
             for combination in sorted(seen, key=len)]
    if fills:
        first = ax.legend(handles=fills, loc="upper left", fontsize=7,
                          title="carriers modelled", title_fontsize=7, framealpha=0.9)
        ax.add_artist(first)

    order = [PRESENCE_DEMAND, PRESENCE_NODE, PRESENCE_UNKNOWN, PRESENCE_ABSENT]
    states = [s for s in order if (presence.values == s).any()]
    cells = [mpatches.Patch(facecolor=PRESENCE_STYLE[s][0], edgecolor="#33333a",
                            hatch=PRESENCE_STYLE[s][1], label=PRESENCE_LABEL[s])
             for s in states]
    if cells:
        ax.legend(handles=cells, loc="lower left", fontsize=7,
                  title=f"chip cells, left to right: "
                        f"{', '.join(titles.get(c, c) for c in presence.columns)}",
                  title_fontsize=7, framealpha=0.9)


def figure_carrier(
    out_dir: Path,
    slug: str,
    title: str,
    capacity: pd.DataFrame,
    demand: pd.DataFrame,
    consumption: pd.DataFrame,
    zones: bool,
    ) -> str:
    """Production capacity and annual demand for one carrier, absolute then normalised.

    Every panel keeps the same country order so a reader can put a country's
    capacity against its own demand without moving between images.

    The bottom row exists because the top row has a readability floor: against
    Germany's 854 GW, Latvia's 5 GW is a hairline, and six of the sixteen
    countries -- the ones whose mix is most distinctive -- cannot be read at
    all. Normalised, every country is the same width and the mix is legible for
    all of them. The absolute panel is still the one that says who is large.
    """
    order = None
    if capacity is not None and not capacity.empty:
        order = capacity.sum(axis=1).sort_values(ascending=False).index.tolist()
    elif demand is not None and not demand.empty:
        order = demand.set_index("area")["mean"].sort_values(ascending=False).index.tolist()

    n_rows = len(order) if order else 6
    fig, axes_grid = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_IN, 2 * _barh_height(n_rows) - 0.6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    axes = axes_grid[0]
    lower = axes_grid[1]

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

    # ---- the same two quantities, normalised --------------------------------
    if order and capacity is not None and not capacity.empty:
        table = capacity.reindex(order).fillna(0.0)
        totals = table.sum(axis=1).replace(0.0, np.nan)
        share = (table.div(totals, axis=0) * 100.0).fillna(0.0)
        share.index = [display_name(a, zones) for a in table.index]
        labels = [l for l in TECH_STYLE if l in share.columns]
        _stacked_barh(lower[0], share, labels, "share of the area's own capacity, %")
        lower[0].set_xlim(0, 100)
    else:
        _empty_panel(lower[0], "")

    if order and demand is not None and not demand.empty:
        series = demand.set_index("area")["mean"].reindex(order).fillna(0.0)
        total = float(series.sum())
        if total > 0:
            values = 100.0 * series.to_numpy(dtype="float64") / total
            positions = np.arange(len(series))
            lower[1].barh(positions, values, color="#4a6fa5", height=0.74)
            lower[1].set_yticks(positions)
            lower[1].set_yticklabels([display_name(a, zones) for a in series.index], fontsize=8)
            lower[1].invert_yaxis()
            lower[1].set_xlabel("share of all demand in the build, %", fontsize=9)
            lower[1].grid(axis="x", alpha=0.25)
            lower[1].tick_params(axis="x", labelsize=8)
        else:
            _empty_panel(lower[1], "")
    else:
        _empty_panel(lower[1], "")

    fig.suptitle(f"{title}: capacity and demand by {'zone' if zones else 'country'}\n"
                 f"absolute above, each area's own share below",
                 fontsize=11)
    return _finish(fig, out_dir / f"fig_{slug}.png")


def figure_storage(
    out_dir: Path,
    power: pd.DataFrame,
    energy: pd.DataFrame,
    ratio_facts: Optional[Dict],
    zones: bool,
    ) -> str:
    """Power on the left, energy on the right, because they are different facts.

    Battery and heat storage state their size as a duration rather than an
    energy, and the duration is one number per grid. So the left panel stays a
    power chart and carries the duration in its legend: a reader multiplies the
    two rather than reading a second bar whose shape would be identical.
    """
    n_rows = max(len(power) if power is not None else 0,
                 energy["area"].nunique() if energy is not None and not energy.empty else 0)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, _barh_height(n_rows)))

    if power is not None and not power.empty:
        table = power.loc[power.sum(axis=1).sort_values(ascending=False).index]
        table.index = [display_name(a, zones) for a in table.index]
        _stacked_barh(axes[0], table, [l for l in TECH_STYLE if l in table.columns],
                      "discharge power, GW")
        # The duration rides in the legend, so the bar keeps meaning power while
        # the reader can still get to energy without leaving the figure.
        hours = duration_by_label(ratio_facts or {})
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles,
                       [f"{l} ({_num(hours[l], 0)} h)" if l in hours else l for l in labels],
                       fontsize=7, loc="lower right")
        total = (ratio_facts or {}).get("total_TWh") or 0.0
        axes[0].set_title(
            "Battery and heat storage\n"
            + (f"(size is a duration per grid: {_num(total * 1000, 0)} GWh in total)"
               if total else "(no energy capacity is set)"),
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


def _draw_corridors(ax, shapes: ZoneShapes, matrix: pd.DataFrame, zones: bool,
                    neighbours: bool = True) -> Dict:
    """The corridor graph on the map, width linear in capacity.

    Returns what the legend needs to state the width scale. An arrow or a line
    whose width encodes a quantity is decoration unless the reader is told what
    a given width is worth, so the caller must draw that key.
    """
    areas = sorted(set(matrix.index) | set(matrix.columns))
    geometry = shapes_for_areas(shapes, areas, zones)
    centres = {area: _polygons_centroid(polygons) for area, polygons in geometry.items()}

    if neighbours:
        context = context_polygons(shapes, areas, zones)
        paths = [p for p in (_polygon_path(poly) for poly in context) if p]
        ax.add_collection(PatchCollection(
            [PathPatch(p) for p in paths],
            facecolor=NEIGHBOUR_FILL, edgecolor=NEIGHBOUR_EDGE, linewidth=0.4, zorder=1))

    paths = [p for polygons in geometry.values()
             for p in (_polygon_path(poly) for poly in polygons) if p]
    ax.add_collection(PatchCollection(
        [PathPatch(p) for p in paths],
        facecolor="#dce7f0", edgecolor="#ffffff", linewidth=0.5, zorder=2))

    # Square and symmetric, so one triangle is the corridor list.
    grid = matrix.reindex(index=areas, columns=areas).fillna(0.0)
    largest = float(grid.to_numpy(dtype="float64").max()) if len(grid) else 0.0

    segments, widths, colours = [], [], []
    unplaced = set()
    for i, a in enumerate(areas):
        for b in areas[i + 1:]:
            capacity = max(float(grid.at[a, b]), float(grid.at[b, a]))
            if capacity <= 0:
                continue
            if centres.get(a) is None or centres.get(b) is None:
                unplaced.update(x for x in (a, b) if centres.get(x) is None)
                continue
            segments.append([centres[a], centres[b]])
            share = capacity / largest if largest else 0.0
            widths.append(CORRIDOR_WIDTH_MIN
                          + share * (CORRIDOR_WIDTH_MAX - CORRIDOR_WIDTH_MIN))
            colours.append(CORRIDOR_COLOUR if country_of(a) != country_of(b)
                           else CORRIDOR_INTERNAL_COLOUR)

    if segments:
        ax.add_collection(LineCollection(segments, linewidths=widths, colors=colours,
                                         alpha=0.85, zorder=3, capstyle="round"))
    for area, centre in centres.items():
        if centre is None:
            continue
        ax.plot(*centre, marker="o", markersize=3.2, color="#1a1a1f", zorder=4)
        ax.text(centre[0], centre[1], f"  {area}", fontsize=7, ha="left", va="center",
                zorder=5, color="#1a1a1f",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#ffffff")])
    return {"largest_GW": largest * MW_TO_GW, "corridors": len(segments),
            "internal": sum(1 for c in colours if c == CORRIDOR_INTERNAL_COLOUR),
            "unplaced": sorted(unplaced)}


def _corridor_width_legend(ax, info: Dict) -> None:
    """The width key, without which the widths mean nothing."""
    largest = info.get("largest_GW") or 0.0
    if largest <= 0:
        return
    handles = []
    for share, label in ((1.0, f"{_num(largest, 1)} GW"), (0.25, f"{_num(largest / 4, 1)} GW")):
        handles.append(mlines.Line2D(
            [], [], color=CORRIDOR_COLOUR,
            linewidth=CORRIDOR_WIDTH_MIN + share * (CORRIDOR_WIDTH_MAX - CORRIDOR_WIDTH_MIN),
            label=label))
    # Only when one is drawn: a key to a colour the picture does not use sends
    # the reader hunting for something that is not there.
    if info.get("internal"):
        handles.append(mlines.Line2D([], [], color=CORRIDOR_INTERNAL_COLOUR, linewidth=2.0,
                                     label="inside one country"))
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              title="corridor width\n(mean of both directions)", title_fontsize=7,
              framealpha=0.9)


def figure_interconnection(
    out_dir: Path,
    matrix: pd.DataFrame,
    per_area: pd.DataFrame,
    peak: Optional[pd.Series],
    zones: bool,
    shapes: Optional[ZoneShapes] = None,
    neighbours: bool = True,
    ) -> str:
    """Who connects to whom, and who leans on it.

    The left panel was a capacity matrix drawn as a heatmap. A map says the same
    thing better -- the matrix could not show that the Nordic zones form a ring
    while Germany is a hub -- and the exact numbers the heatmap carried are in
    the table above it, where they can be read to a decimal.
    """
    if matrix is None or matrix.empty:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
        _empty_panel(ax, "No electricity transfer capacity in this scenario")
        return _finish(fig, out_dir / "fig_interconnection.png")

    areas = sorted(set(matrix.index) | set(matrix.columns))
    size = max(5.0, 0.36 * len(areas) + 2.4)
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, size),
        gridspec_kw={"width_ratios": [1.0, 0.42]},
    )

    if shapes is not None and shapes.available:
        info = _draw_corridors(axes[0], shapes, matrix, zones, neighbours)
        _finish_map(axes[0])
        _corridor_width_legend(axes[0], info)
        axes[0].set_title("Electricity transfer capacity between areas", fontsize=9)
    else:
        _empty_panel(axes[0], (shapes.missing if shapes is not None
                               else "No map asset") + "\n-- the table above has the capacities")

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

    # The system as a panel of its own, drawn under the same rule as its
    # members: sixteen curves on sixteen scales cannot answer whether the
    # surplus hours coincide, which is the question the section is for.
    panels = dict(curves)
    if len(areas) > 1:
        panels["system"] = np.sum(
            np.vstack([curves[a].astype("float64") for a in areas]), axis=0)
    order = areas + (["system"] if "system" in panels else [])

    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, 2.1 * nrows + 0.8))
    flat = np.atleast_1d(axes).ravel()

    for ax, area in zip(flat, order):
        values = np.sort(panels[area].astype("float64"))[::-1] * MW_TO_GW
        share = np.linspace(0, 100, len(values))
        ax.plot(share, values, color="#2a4d6e", linewidth=1.2)
        ax.fill_between(share, values, 0, where=values > 0, color="#2a4d6e", alpha=0.18)
        negative = values < 0
        if negative.any():
            ax.fill_between(share, values, 0, where=negative, color="#d98050", alpha=0.30)
        ax.axhline(0, color="#555555", linewidth=0.7)
        name = "the system" if area == "system" else display_name(area, zones)
        ax.set_title(
            f"{name}\npeak {values[0]:,.0f} GW, "
            f"surplus {100.0 * negative.mean():.0f}% of hours",
            fontsize=7.5,
        )
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.2)
    for ax in flat[len(order):]:
        ax.set_visible(False)

    fig.supxlabel("share of all modelled hours, %", fontsize=9)
    fig.supylabel("net load (demand - wind - solar), GW", fontsize=9)
    fig.suptitle("Net-load duration curves, every climate year pooled", fontsize=11)
    return _finish(fig, out_dir / "fig_netload_duration.png")


def figure_duration(out_dir: Path, decomposition: Dict, zones: bool) -> str:
    """On what timescale each area's residual demand sits, and which years were hard.

    Left is each area as a share of its own residual demand, so a small country
    is comparable with a large one; right is the system in TWh, year by year,
    because the question there is which weather years were hard and on what
    timescale -- and that is a quantity, not a share.
    """
    path = out_dir / "fig_duration_decomposition.png"
    if not decomposition:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
        _empty_panel(ax, "The timescale decomposition needs the timeseries files")
        return _finish(fig, path)

    by_area = decomposition["by_area"]
    by_year = decomposition["by_year"]
    labels = decomposition["labels"]
    height = max(4.2, 0.30 * max(len(by_area), 8) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, height),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    totals = by_area.sum(axis=1).replace(0.0, np.nan)
    shares = (by_area.div(totals, axis=0) * 100.0).dropna(how="all")
    shares = shares.loc[shares[DURATION_REMAINDER].sort_values().index]
    positions = np.arange(len(shares))
    left = np.zeros(len(shares))
    for label in labels:
        values = shares[label].to_numpy(dtype="float64")
        axes[0].barh(positions, values, left=left, color=DURATION_STYLE[label],
                     edgecolor="white", linewidth=0.4, height=0.76, label=label)
        left += values
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([display_name(a, zones) for a in shares.index], fontsize=7)
    axes[0].set_xlabel("share of the area's own residual demand, %", fontsize=8)
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].set_title("On what timescale each area's\nresidual demand sits", fontsize=9)

    years = list(by_year.index)
    at = np.arange(len(years))
    bottom = np.zeros(len(years))
    for label in labels:
        values = by_year[label].to_numpy(dtype="float64")
        axes[1].bar(at, values, bottom=bottom, color=DURATION_STYLE[label],
                    edgecolor="white", linewidth=0.3, width=0.82, label=label)
        bottom += values
    axes[1].set_xticks(at)
    axes[1].set_xticklabels([str(y) for y in years], fontsize=5.5, rotation=90)
    axes[1].set_ylabel("system residual demand, TWh", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set_title("The system, year by year", fontsize=9)
    axes[1].legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.9)

    return _finish(fig, path)


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

    def figure(self, name: str, caption: str, caveat: Optional[str] = None) -> None:
        self.add(f"![{caption.split('.')[0]}]({name})")
        self.add(f"*{caption}*")
        if caveat:
            # Under the figure it qualifies, not in an end section nobody reads
            # while looking at the picture.
            self.add()
            self.add(f"<sub>{caveat}</sub>")
        self.add()

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def build_report(
    workbook: Workbook,
    classification: Classification,
    timeseries: Timeseries,
    out_dir: Path,
    zones: bool,
    shapes: Optional[ZoneShapes] = None,
    neighbours: bool = True,
    ) -> Tuple[str, List[str]]:
    """The whole report.md, and the figure names written beside it."""
    level = "bidding zone" if zones else "country"
    report = Report()
    figures: List[str] = []
    shapes = shapes if shapes is not None else load_zone_shapes()
    presence = carrier_presence(workbook, timeseries, zones)

    per_area_transfer, matrix, transfer_facts = transfer_by_area(workbook, zones)
    energy_table, storage_facts = storage_energy(workbook, timeseries.storage_limit, zones)
    power_table = storage_power_by_area(workbook, zones)
    ratio_table, ratio_facts = storage_energy_from_ratio(workbook, zones)
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
        hydro_total = storage_facts["constant_TWh"] + storage_facts["timeseries_TWh"]
        report.add(f"- **Hydro storage**: {_num(hydro_total)} TWh"
                   + (f", of which only {_num(storage_facts['constant_TWh'])} TWh is a number in "
                      f"the workbook -- the rest is in a timeseries, see Storage."
                      if storage_facts["timeseries_TWh"] else
                      ", every megawatt-hour of it a constant in the workbook."))
    if ratio_facts.get("total_TWh"):
        report.add(f"- **Battery and heat storage**: "
                   f"{_num(ratio_facts['total_TWh'] * 1000, 0)} GWh, written as a duration per "
                   f"grid rather than an energy, see Storage.")
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
    report.add("1. [Where the model is](#where-the-model-is)")
    for carrier, data in carrier_data.items():
        report.add(f"1. [{data['title']}](#{data['slug']})")
    report.add("1. [Storage](#storage)")
    report.add("1. [Interconnection](#interconnection)")
    report.add("1. [Net load](#net-load)")
    report.add("1. [On what timescale](#on-what-timescale)")
    report.add("1. [What the weather years do](#what-the-weather-years-do)")
    report.add("1. [Fuel, CO2 and emissions](#fuel-co2-and-emissions)")
    report.add("1. [What the model may do](#what-the-model-may-do)")
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

    # ---- where the model is -----------------------------------------------
    report.add('<a id="where-the-model-is"></a>')
    report.add()
    report.add("## Where the model is")
    report.add()
    _coverage_section(report, presence, shapes, zones)
    figures.append(figure_carrier_map(out_dir, shapes, presence, zones, neighbours))
    report.figure(
        figures[-1],
        "Each area filled by the carriers it models, with a chip showing whether anything "
        "demands each of them. Take-away: which carriers are modelled where, and where a "
        "carrier exists as a node that nothing draws on.",
    )

    # ---- carrier sections -------------------------------------------------
    for carrier, data in carrier_data.items():
        figures.append(_carrier_section(report, workbook, data, carrier, timeseries,
                                        out_dir, zones, ratio_facts))

    # ---- storage ----------------------------------------------------------
    report.add('<a id="storage"></a>')
    report.add()
    report.add("## Storage")
    report.add()
    _storage_section(report, storage_facts, energy_table, power_table, ratio_table,
                     ratio_facts, timeseries, zones)
    figures.append(figure_storage(out_dir, power_table, energy_table, ratio_facts, zones))
    report.figure(
        figures[-1],
        "Left: battery and heat storage discharge power, with each technology's duration in the "
        "legend. Right: hydro reservoir and pumped-storage energy, split by which of the two "
        "places it is written. Take-away: a handful of countries carry the system's seasonal "
        "energy in hydro, and the rest hold hours rather than months.",
    )

    # ---- interconnection --------------------------------------------------
    report.add('<a id="interconnection"></a>')
    report.add()
    report.add("## Interconnection")
    report.add()
    _transfer_section(report, per_area_transfer, transfer_facts, peak, zones)
    figures.append(figure_interconnection(out_dir, matrix, per_area_transfer, peak, zones,
                                          shapes, neighbours))
    report.figure(
        figures[-1],
        "Transfer capacity as a map, and each area's total as a share of its own peak demand. "
        "Take-away: who is connected to whom, and who depends on those connections rather than "
        "on its own plant.",
    )

    # ---- net load ---------------------------------------------------------
    report.add('<a id="net-load"></a>')
    report.add()
    report.add("## Net load")
    report.add()
    curves = netload_by_area(timeseries) if timeseries.available else {}
    split = firm_and_shifting(carrier_data["elec"]["capacity"])
    imports = (per_area_transfer["cross_border_MW"] if per_area_transfer is not None
               and not per_area_transfer.empty else pd.Series(dtype="float64"))
    vre_share = potential_vre_share(capacity_factors, carrier_data["elec"]["demand"])
    _netload_section(report, curves, timeseries, zones, split, imports, vre_share)
    figures.append(figure_netload(out_dir, curves, zones))
    report.figure(
        figures[-1],
        "Net load -- demand minus wind and solar -- for every hour of every climate year, "
        "sorted highest to lowest, one panel per area on its own scale, with the system added "
        "hour by hour in the last panel. Orange is the part of "
        "the year when wind and solar alone exceed demand. Take-away: the left edge is what "
        "firm capacity has to cover; the orange area is what has to be stored, exported or "
        "curtailed. It ignores storage, trade and dispatch, and treats VRE as never curtailed "
        "and never out of service, so it is a statement about the input data and not a forecast.",
    )

    # ---- weather years ----------------------------------------------------
    # ---- what storage of each duration could remove ------------------------
    report.add('<a id="on-what-timescale"></a>')
    report.add()
    report.add("## On what timescale")
    report.add()
    decomposition = residual_by_timescale(curves, timeseries)
    _duration_section(report, decomposition, timeseries, zones)
    figures.append(figure_duration(out_dir, decomposition, zones))
    report.figure(
        figures[-1],
        "Residual demand split by the storage duration that could remove it. Take-away: an area "
        "whose bar is mostly light needs hours of storage; one whose bar is mostly dark needs "
        "months, or needs energy it does not have.",
    )

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
        caveat=(f"Read from the {len(timeseries.years)} real per-year files, never the "
                f"`f01`/`f02`/`f03` forecast branches. Those are a per-hour quantile taken "
                f"independently across the same years, not three coherent alternative years, so "
                f"a claim about \"the driest year\" cannot be built from them."
                if timeseries.available else None),
    )

    # ---- prices -----------------------------------------------------------
    report.add('<a id="fuel-co2-and-emissions"></a>')
    report.add()
    report.add("## Fuel, CO2 and emissions")
    report.add()
    _price_section(report, workbook)

    # ---- what the model may do --------------------------------------------
    report.add('<a id="what-the-model-may-do"></a>')
    report.add()
    report.add("## What the model may do")
    report.add()
    _permissions_section(report, model_permissions(workbook))

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


def _carrier_section(report, workbook, data, carrier, timeseries, out_dir, zones,
                     ratio_facts=None) -> str:
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
        gloss = explain_grids(pivot.columns, ratio_facts)
        if gloss:
            report.add(gloss)
            report.add()

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
        report.add(f"What the wind and solar fleets actually yield, over "
                   f"{len(timeseries.years)} climate year(s):")
        report.add()
        report.add("`capacity factor = sum over nodes(capacity x its own factor) / "
                   "sum over nodes(capacity)`")
        report.add()
        report.table([level_header(zones)] + [f"{f} CF" for f in flows], rows)
        report.add("*Weighted, so a 200 MW zone cannot count for as much as a 12 GW one.*")
        report.add()

    name = figure_carrier(out_dir, data["slug"], data["title"], capacity, demand,
                          consumption, zones)
    report.figure(
        name,
        f"{data['title']} capacity and annual demand, same area order in every panel. "
        f"Take-away: what each area has built for this carrier, read directly against what it "
        f"uses.",
        caveat=(f"The technology grouping is this tool's own: the source data names "
                f"{len(workbook.unittypes)} unit types and no grouping of them, so units are "
                f"grouped by which grids they connect. That is what keeps inflow-driven pumped "
                f"hydro counted as generation and closed-loop pumped hydro as storage, where a "
                f"rule keyed on the unit type's name would merge them. Watch the check on "
                f"capacity with no home when a new unit type appears."
                if carrier == "elec" else None),
    )
    return name


def level_header(zones: bool) -> str:
    return "zone" if zones else "country"


def _permissions_section(report, facts: Dict) -> None:
    """What a unit may do in this build. Counts, no interpretation."""
    rows = [
        ["Units whose availability is below 1",
         f"{facts['derated_units']} of {facts['units']}",
         "`p_unit` `availability`"],
        ["Unit rows with a ramp limit",
         (f"{facts['ramp_limited_rows']} of {facts['rows']}" if facts["ramp_columns_present"]
          else f"0 of {facts['rows']} -- the columns are not written"),
         "`p_gnu_io` `maxRampUp`, `maxRampDown`"],
        ["Unit rows with a ramp cost", f"{facts['ramp_cost_rows']} of {facts['rows']}",
         "`p_gnu_io` `rampUpCost`, `rampDownCost`"],
        ["Unit-commitment rows",
         "-" if facts["commitment_rows"] is None else str(facts["commitment_rows"]),
         "`effLevelGroupUnit`"],
        ["Nodes starting from a bound state",
         f"{facts['bound_start_nodes']} of {facts['nodes']}", "`p_gn` `boundStart`"],
    ]
    report.table(["What", "Count", "Where it is read from"], rows)
    report.add("These are counts, not findings. A zero ramp-limit count means any unit in this "
               "build may go from nothing to full output in one hour; no unit-commitment rows "
               "means no minimum load and no start cost; availability 1 everywhere means the "
               "capacity totals in this report are plate ratings with no outage derating. "
               "Whether that suits the question being asked is a modelling decision this report "
               "does not make.")
    report.add()


def _coverage_section(report, presence: pd.DataFrame, shapes: ZoneShapes, zones: bool) -> None:
    """Which carriers each area models -- every area gets a row, including empty ones.

    Absence is not a defect here and nothing below calls it one: an area with no
    district heating is stating what the model contains, not a gap to fix. What
    is worth a reader's attention is the third state -- a carrier with a node
    that nothing demands -- which is a modelling choice they may not know they
    made.
    """
    if presence.empty:
        report.add("No nodes are written in this scenario.")
        report.add()
        return

    titles = {grid: title for grid, title, _ in CARRIERS}
    counts = {c: int((presence[c] != PRESENCE_ABSENT).sum()) for c in presence.columns}
    total = len(presence)
    report.add(f"{total} {level_header(zones)}(s), and what each one carries: "
               + ", ".join(f"{titles.get(c, c).lower()} in {counts[c]}" for c in presence.columns)
               + ".")
    report.add()

    hollow = {c: sorted(presence.index[presence[c] == PRESENCE_NODE]) for c in presence.columns}
    for carrier, areas in hollow.items():
        if areas and len(areas) == counts[carrier]:
            report.add(f"Every one of the {len(areas)} {level_header(zones)}(s) with a "
                       f"{titles.get(carrier, carrier).lower()} node has nothing demanding it: "
                       f"the units are built and the carrier has no sink in this build.")
            report.add()
        elif areas:
            report.add(f"{len(areas)} {level_header(zones)}(s) carry a "
                       f"{titles.get(carrier, carrier).lower()} node that nothing demands: "
                       f"{summarise(areas)}.")
            report.add()

    if not shapes.available:
        report.add(f"There is no map in this report: {shapes.missing}.")
        report.add()


def _storage_section(report, facts, energy_table, power_table, ratio_table, ratio_facts,
                     timeseries, zones) -> None:
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

    if ratio_facts.get("total_TWh"):
        by_grid = ratio_facts["by_grid"]
        report.add(f"Battery and heat storage hold a further **{_num(ratio_facts['total_TWh'])} "
                   f"TWh**, written a different way. They have no `upwardLimit` row; their size is "
                   f"`upperLimitCapacityRatio` in `p_gnu_io`, on the input side of the discharge "
                   f"unit. It is v_state units per MW, so where `energyStoredPerUnitOfState` is 1 "
                   f"it is the storage's duration in hours and the energy is power times it.")
        report.add()
        report.add("The duration is one number per grid, not per area, so the energy below is "
                   "each area's power at that grid's duration:")
        report.add()
        rows = []
        for grid in sorted(by_grid, key=lambda g: -by_grid[g]["TWh"]):
            entry = by_grid[grid]
            hours = (_num(entry["hours"], 0) if entry["hours"] is not None
                     else summarise([_num(h, 0) for h in entry["hours_seen"]]))
            rows.append([f"`{grid}`", hours, _num(entry["power_GW"]),
                         _num(entry["TWh"] * 1000, 0), str(entry["nodes"])])
        rows.append(["**total**", "", "", f"**{_num(ratio_facts['total_TWh'] * 1000, 0)}**", ""])
        report.table(["grid", "hours", "power GW", "energy GWh", "node(s)"], rows)
        if ratio_facts["unconvertible"]:
            report.add(f"{len(ratio_facts['unconvertible'])} node(s) state a ratio but no "
                       f"`energyStoredPerUnitOfState`, so their v_state is not MWh and their "
                       f"energy cannot be read from it: "
                       f"{summarise(ratio_facts['unconvertible'])}.")
            report.add()
    elif facts["grids_without_energy"]:
        report.add(f"**{', '.join(facts['grids_without_energy'])}** carry no energy capacity "
                   f"anywhere in this workbook -- no `upwardLimit` row is written for them, and "
                   f"no `upperLimitCapacityRatio` either. What is defined is their charge and "
                   f"discharge power, below. That is a property of the data rather than something "
                   f"missing from this report.")
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
    report.add(f"{facts['pairs']} corridor(s) carry {_num(facts['total_GW'])} GW in total.")
    report.add()
    report.add("`corridor capacity = (the A-to-B row + the B-to-A row) / 2`")
    report.add()
    report.add("*Every corridor is written once per direction. Adding the two would count it "
               "twice; taking one would depend on which row the builder wrote first.*")
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

    # Which carriers can move between areas at all, said out loud: a carrier
    # with units in every zone and no corridor between them is a set of
    # islands, and nothing else in the report would show it.
    titles = {grid: title for grid, title, _ in CARRIERS}
    carried = set(facts.get("carrier_grids", []))
    exists = set(facts.get("node_grids", []))
    silent = [grid for grid in titles
              if grid not in carried and grid != "elec" and grid in exists]
    for grid, counts in sorted(facts.get("other_grids", {}).items()):
        name = titles.get(grid, grid)
        if counts["between_areas"]:
            report.add(f"{name} also has corridors: {counts['between_areas']} of its "
                       f"{counts['corridors']} link one {level_header(zones)} to another.")
        else:
            report.add(f"{name} has {counts['corridors']} corridor(s), every one of them inside a "
                       f"single {level_header(zones)} and so below the level this report shows.")
        report.add()
    if silent:
        report.add(", ".join(titles[g] for g in silent)
                   + " has no corridor anywhere in this build: wherever it is modelled it is an "
                     "island, and nothing can move from one to the next.")
        report.add()

    rows = []
    total = (per_area["cross_border_MW"] + per_area["inter_zonal_MW"]).sort_values(ascending=False)
    for area in total.index:
        share = None
        if peak is not None and area in peak.index and peak[area] > 0:
            share = 100.0 * total[area] / peak[area]
        rows.append([
            display_name(area, zones),
            # The sort is on this column, so it is shown. Ordering a table by a
            # quantity that appears nowhere in it leaves the reader unable to
            # check the order, or to see how close two neighbouring rows are.
            _num(total[area] * MW_TO_GW),
            _num(per_area.loc[area, "cross_border_MW"] * MW_TO_GW),
            _num(per_area.loc[area, "inter_zonal_MW"] * MW_TO_GW) if per_area.loc[area, "inter_zonal_MW"] else "-",
            _num(peak[area], 0) if peak is not None and area in peak.index else "-",
            f"{_num(share, 0)}%" if share is not None else "-",
        ])
    report.table([level_header(zones), "total GW", "cross-border GW", "internal GW",
                  "peak demand MW", "capacity / peak"], rows)
    if peak is not None:
        report.add("`capacity / peak = (cross-border + internal) / the area's own highest hour`")
        report.add()
        report.add("*Peak demand is the highest single hour across every climate year, taken on "
                   "the area's own coincident hours rather than by adding its zones' separate "
                   "maxima. Above 100% is not an error: a small, well-connected area can have "
                   "more transfer capacity than its own peak load.*")
        report.add()


def _duration_section(report, decomposition: Dict, timeseries: Timeseries, zones: bool) -> None:
    """What storage of each duration could remove, and what nothing can."""
    if not decomposition:
        report.add("The timescale decomposition needs the hourly timeseries, which were not read"
                   + (f": {timeseries.skipped}" if timeseries.skipped else "") + ".")
        report.add()
        return

    by_area = decomposition["by_area"]
    by_year = decomposition["by_year"]
    report.add("Net load summed inside a window lets a surplus hour pay for a deficit hour within "
               "it -- what a perfect, lossless, free store of that duration would do. Each row "
               "below is the energy only a store of that length can reach, and the last is what "
               "survives a whole year, because the energy to move is simply not there. Blocks are "
               "fixed, not sliding, so no surplus hour is spent twice.")
    report.add()

    totals = by_area.sum(axis=1)
    rows = []
    for area in totals.sort_values(ascending=False).index:
        row = by_area.loc[area]
        rows.append([display_name(area, zones), _num(totals[area], 0)]
                    + [f"{_num(100.0 * row[l] / totals[area], 0)}%" if totals[area] else "-"
                       for l in decomposition["labels"]])
    system = by_year.sum(axis=0) / max(len(by_year), 1)
    system_total = float(system.sum())
    rows.append(["**system**", f"**{_num(system_total, 0)}**"]
                + [f"**{_num(100.0 * system[l] / system_total, 0)}%**" if system_total else "-"
                   for l in decomposition["labels"]])
    report.table([level_header(zones), "residual demand TWh/yr"] + decomposition["labels"], rows)

    hardest = by_year.sum(axis=1)
    report.add(f"Averaged over the climate years the system's residual demand is "
               f"{_num(system_total, 0)} TWh/yr, of which "
               f"{_num(100.0 * float(system[DURATION_REMAINDER]) / system_total, 0)}% is beyond "
               f"the reach of a year of storage. The hardest climate year is "
               f"{hardest.idxmax()} at {_num(float(hardest.max()), 0)} TWh and the easiest "
               f"{hardest.idxmin()} at {_num(float(hardest.min()), 0)} TWh, a spread of "
               f"{_num(100.0 * (float(hardest.max()) / float(hardest.min()) - 1), 0)}%.")
    report.add()
    report.add("*Every figure here is an upper bound on what storage could do, and so a lower "
               "bound on what else is needed: nothing is charged an efficiency, a power limit or "
               "a cost, and no energy is traded between areas.*")
    report.add()


def _netload_section(report, curves, timeseries, zones, split, imports, vre_share) -> None:
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
    report.add("`net load = electricity demand - wind output - solar output`, and "
               "`left over = firm + storage and DR + imports - peak`.")
    report.add()

    rows = []
    for area, values in sorted(curves.items(), key=lambda kv: -kv[1].max()):
        rows.append(_netload_row(area, values, split, imports, vre_share,
                                 display_name(area, zones)))

    # The system row is the areas added hour by hour, not their peaks added: the
    # question the section exists to answer is whether the worst hours coincide,
    # and a sum of separate maxima assumes they do.
    if len(curves) > 1:
        system = np.sum(np.vstack([v.astype("float64") for v in curves.values()]), axis=0)
        totals = pd.DataFrame({
            "firm_GW": [float(split["firm_GW"].sum()) if not split.empty else np.nan],
            "shifting_GW": [float(split["shifting_GW"].sum()) if not split.empty else np.nan],
        }, index=["system"])
        # Imports cancel inside the system, so the system covers its peak alone.
        rows.append(_netload_row("system", system, totals,
                                 pd.Series({"system": 0.0}), vre_share, "**system**",
                                 bold=True))

    report.table([level_header(zones), "peak GW", "1st pct GW", "minimum GW", "hours in surplus",
                  "VRE energy / demand", "firm GW", "storage & DR GW", "imports GW",
                  "left over GW"], rows)
    report.add("The last column is what is left once firm capacity, the storage and "
               "demand-response that can shift into that hour, and the whole import capacity are "
               "set against the peak. Negative does not mean the area fails: it means it cannot "
               "cover its own worst hour from those three alone, which is what interconnection "
               "and a wider system are for. The system row adds the areas hour by hour, so it "
               "already accounts for peaks that do not coincide, and its imports are zero because "
               "trade between members cancels inside it.")
    report.add()
    if not vre_share.empty:
        report.add("VRE energy over demand is installed wind and solar at their own "
                   "capacity-weighted factors across a year, against mean annual demand. Below "
                   "about 80% an area is short of energy; above 100% it could in principle make "
                   "its whole year from wind and sun and its problem is timing.")
        report.add()


def _netload_row(area, values, split, imports, vre_share, label, bold: bool = False):
    """One row of the net-load table, for an area or for the system."""
    gw = values.astype("float64") * MW_TO_GW
    peak = float(gw.max())
    firm = float(split.loc[area, "firm_GW"]) if area in split.index else None
    shifting = float(split.loc[area, "shifting_GW"]) if area in split.index else None
    imported = float(imports.get(area, np.nan)) * MW_TO_GW if area in imports.index else None
    left = None
    if firm is not None and shifting is not None and imported is not None:
        left = firm + shifting + imported - peak

    def mark(text: str) -> str:
        return f"**{text}**" if bold else text

    return [
        label,
        mark(_num(peak)), mark(_num(float(np.percentile(gw, 1)))), mark(_num(gw.min())),
        mark(f"{_num(100.0 * (gw < 0).mean(), 1)}%"),
        mark(f"{_num(vre_share[area], 0)}%" if area in vre_share.index else "-"),
        mark(_num(firm) if firm is not None else "-"),
        mark(_num(shifting) if shifting is not None else "-"),
        mark(_num(imported) if imported is not None else "-"),
        mark(_num(left) if left is not None else "-"),
    ]


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
        total_areas = len(correlations)
        floor = significance_floor(len(timeseries.years))
        if floor is None:
            report.add(f"With {len(timeseries.years)} climate year(s) there are too few points to "
                       f"say whether any of these correlations differs from noise.")
            report.add()
        else:
            strong_hydro = int((correlations["hydro"].abs() >= floor).sum())
            negative_vre = int((correlations["vre"] <= -floor).sum())
            report.add(
                f"With {len(timeseries.years)} climate years a correlation has to reach "
                f"|r| = {_num(floor, 2)} before it is distinguishable from noise at the usual 5% "
                f"level, so that -- and not a round number -- is what the counts below are "
                f"against. Hydro inflow and demand: {strong_hydro} of {total_areas} area(s) "
                f"clear it. Wind and solar: {negative_vre} of {total_areas} area(s) sit at or "
                f"below -{_num(floor, 2)}, so a low-VRE year does tend to be a high-demand year, "
                f"though with this many years the effect is barely resolved and its size should "
                f"not be read off these numbers. If this scenario is ever run as one hand-picked "
                f"year rather than all of them, that is still the relationship worth picking "
                f"against.")
            report.add()
    if material:
        left_out = [a for a in timeseries.areas if a not in material]
        report.add(f"The figure below shows the {len(material)} area(s) with at least "
                   f"{_num(MATERIAL_HYDRO_TWH, 0)} TWh/yr of mean inflow. The other "
                   f"{len(left_out)} are left out because a large percentage swing on a very small "
                   f"inflow says more about the denominator than about the system.")
        report.add()


def significance_floor(n_years: int) -> Optional[float]:
    """The smallest ``|r|`` that means anything at the usual 5% level, for ``n`` points.

    Thirty-five climate years is not many, and a threshold picked as a round
    number can sit below the level at which a correlation is distinguishable
    from noise -- 0.3 does, at n = 35, where the floor is 0.33.

    Fisher's z transform rather than the exact Student-t inverse: it needs only
    ``math``, and at n = 35 it gives 0.333 against the exact 0.334. scipy is
    installed on some machines here but is not in ``environment.yml``, so
    nothing in this tool may depend on it.
    """
    if n_years is None or n_years < 5:
        return None
    return float(np.tanh(1.959964 / math.sqrt(n_years - 3)))


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
    # The per-zone spread is a one-line fact; the prices themselves are a column
    # of the emissions table below, so they are not listed twice.
    varying = prices[prices["distinct"] > 1] if not prices.empty else prices
    if not prices.empty:
        if varying.empty:
            report.add("Every fuel costs the same in every zone.")
        else:
            report.add(f"{len(varying)} of {len(prices)} fuel(s) cost different amounts in "
                       f"different zones, and the table below shows one of them: "
                       + summarise([str(r["grid"]) for _, r in varying.iterrows()]) + ".")
        report.add()

    price = co2_price(workbook)
    factors = emission_factors(workbook)
    if price is not None:
        report.add(f"CO2 costs {_num(price, 0)} EUR/t, one value for the whole run.")
        report.add()
    if not factors.empty or not prices.empty:
        # Every priced fuel gets a row, not only the ones with an emission factor:
        # a table of seven under a sentence naming ten reads as if three were
        # missing data, when in fact they emit nothing this model counts.
        factor_by_fuel = {
            str(row["fuel"]): row
            for _, row in factors.iterrows()
            if str(row["emission"]).lower() == "co2"
        }
        price_by_fuel = {str(row["grid"]): float(row["price"]) for _, row in prices.iterrows()}

        rows = []
        for fuel in sorted(set(price_by_fuel) | set(factor_by_fuel)):
            factor_row = factor_by_fuel.get(fuel)
            fuel_price = price_by_fuel.get(fuel)
            carbon = (float(factor_row["factor"]) * price
                      if factor_row is not None and price is not None else None)
            total = None
            if fuel_price is not None:
                total = fuel_price + (carbon or 0.0)
            rows.append({
                "sort": total if total is not None else float("inf"),
                "cells": [
                    fuel,
                    _num(fuel_price, 1) if fuel_price is not None else "-",
                    _num(factor_row["factor"], 3) if factor_row is not None else "-",
                    _num(carbon, 1) if carbon is not None else "-",
                    _num(total, 1) if total is not None else "-",
                ],
            })
        rows.sort(key=lambda r: r["sort"])

        report.add("What each fuel costs once its carbon is paid for, cheapest first.")
        report.add()
        report.table(["fuel", "fuel EUR/MWh", "t CO2/MWh", "carbon EUR/MWh",
                      "fuel + carbon, EUR/MWh of fuel"], [r["cells"] for r in rows])
        report.add("A dash in the CO2 columns is a fuel this model charges no carbon for, not a "
                   "gap in the data. The last column orders *fuels*, not plants: it is per MWh "
                   "burned, and dividing by each unit's efficiency to get a cost per MWh "
                   "delivered can reorder it, because an efficient plant on a dear fuel can "
                   "undercut an inefficient one on a cheap fuel.")
        report.add()


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
    """What survives here rather than sitting under the figure it qualifies.

    Two caveats that used to live in this section now sit in small type under
    the figure each one is about -- the technology grouping under the capacity
    figures, the forecast branches under the climate-year figure -- because a
    caveat quarantined at the end of a document is read by nobody who is
    looking at the picture it applies to.
    """
    report.add("There is no figure or table below zone level, at any flag setting, and no "
               "\"Nordics\" or \"CWE\" grouping, because nothing in the tracked source data "
               "defines one. The maps inherit that: a zone's whole fleet sits at one point, "
               "and a country is drawn from its zones' outlines rather than a single border.")
    report.add()
    if not timeseries.available:
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
        "--no-neighbours", action="store_true",
        help="draw only the modelled areas on the maps, without the grey ring of "
             "countries around them",
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

    shapes = load_zone_shapes()
    if not shapes.available:
        print(f"Maps skipped: {shapes.missing}")

    out_dir = folder / args.out_subdir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        text, figures = build_report(workbook, classification, timeseries, out_dir, args.zones,
                                     shapes, not args.no_neighbours)
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
