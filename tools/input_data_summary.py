"""
input_data_summary.py -- what is in one built input-data folder, as a report.

Reads <built_folder>/inputData.xlsx and the per-climate-year ts_influx_*, ts_cf_*
and ts_node_hydro_storage_limits_* GDX files beside it, plus what the build
recorded about itself: cache/config_structural.json, cache/general_flags.json
and summary.log.
Writes one self-contained report.md, with its figures, into a subfolder of the
built folder -- so a colleague can be handed a folder and read it. Each run
rewrites report.md and the figures it draws; any other file in that subfolder
is left as it is.

Usage:
    python tools/input_data_summary.py <built_folder> [--zones]
        [--no-neighbours] [--out-subdir NAME]

Examples:
    python tools/input_data_summary.py input_ObservedTrends_2030
    python tools/input_data_summary.py input_ObservedTrends_2030 --zones
    python tools/input_data_summary.py input_tyndp2024_NationalTrends_2040

What it shows
-------------
Countries by default, the bidding zones with --zones. First the build settings
-- which config and variants built the folder, its climate windows and forecast
branches, and the features this report does not handle yet. Then how far each
carrier reaches, counted at both levels whatever the report is written at, as a
map of what each bidding zone models and a list of where each carrier is missing
or has nothing demanding it.

Then one chapter per carrier, each for a mean climate year. Electricity has
generation, consumption, net load, interconnection, storage -- in two nested
groups, what the power system fills itself and what the weather fills for it --
and demand management. Hydro follows, from the water's side: volume by type,
inflow, and how close each hydro storage comes to its floor. District heat, industrial
steam and hydrogen each have generation, consumption, interconnection and
storage. Every megawatt is counted once in a chapter: storage discharge is
under storage rather than generation, demand response under demand management,
and storage charging under storage rather than consumption. Consumption ends by
setting dispatchable generation against everything that could take its output.

Then what varies between the climate years, fuel, CO2 and emission prices, what
Backbone may do with the units, and the checks.

Climate windows: a window of a year or longer is reported; a shorter one is not.
The build starts a window in every climate year, so a window longer than 365
days overlaps the next. The annual and hourly figures therefore use the first
365 days of each window, which counts each climate year once, while the hydro
checks run every hour of every whole window, each window on its own, because
each is its own run. A window shorter than 365 days covers only part of
the seasons, so no annual figure can be made from it: the timeseries sections
are left out and the report says why.

Hydro is reported per type: storage, turbine draw, minimum generation, maximum
generation where availability is below one, pump delivery, full-load hours and
weeks of storage. Two checks run per bidding-zone storage node whatever the level:
whether inflow can carry the minimum generation in ``p_gnu_io`` (``minGen``)
without the storage falling below its floor, and whether a full storage can pass its
inflow with turbines and ``maxSpill`` at full. Every number is read from the
built folder, never from a source workbook.

A storage volume is reported three ways, because a reservoir bounded by two
seasonal series has no single size: usable is the mean of the gap between the
bounds, the seasonal envelope is the widest swing a year permits, and nameplate
is the ceiling alone. In the shipped scenarios the ceiling is about twice the
usable volume, and reporting it alone was wrong by that much.

The two maps need both ``tools/maps/zone_shapes.geojson`` and
``country_shapes.geojson``, which ``tools/prepare_zone_geometry.py`` writes: the
zones are filled and the country borders drawn over them. Without them the rest
of the report is unaffected and the map panels say
what they wanted -- the same documented degradation as a missing GAMS install.
Nothing here imports a geometry library: the asset is read with ``json`` and
drawn with matplotlib patches.

A carrier with no data still gets its chapter, saying so in one line: a chapter
that disappears cannot tell anyone it is empty. It gets no figure, which would
only repeat the line, and a carrier with one technology gets no capacity-mix
figure, which would draw every bar at 100%. A carrier other than electricity
with no corridors or no storage says so in its opening paragraph instead of
carrying an empty subsection.

What it cannot see
------------------
Nothing below zone level -- there is no per-node figure at any flag setting,
except the hydro checks, which are per storage node because a minimum binds one.
No "Nordics" or "CWE" grouping: nothing in the tracked source data defines one,
so none is invented, and zone and country are the only two levels offered.

Where anything is inside an area. The maps place a zone's whole capacity at one
point, because the build states no location below the zone.

Whether a unit can actually run, outside hydro: availability and efficiency are
read, and only the hydro checks use them -- to turn a minimum into water, and to
bound what turbines can give and release.

What hydro does in the forecast branches. The checks see the realised climate
years; Backbone also solves a dry inflow quantile with storage limits that
``changes.inc`` narrows at run time, and that is where minimum generation has
actually run short. They also start each window from the middle of its range,
where Backbone starts a psOpen or reservoir storage at its ``relative`` share of
the band on the starting date and any other storage at its ``reference``.

How Backbone behaves. Every number here comes from the input data, never from a
solved run, so the net-load curve ignores storage, trade and dispatch, and the
VRE figures ignore curtailment and outages.

Investment. When the build lets Backbone invest -- ``p_gn('maxInvest')``,
``p_unit('maxUnitCount')`` or ``p_gnn('transferCapInvLimit')`` -- or limits when
a unit is available through ``utAvailabilityLimits``, the capacities here are
the starting fleet only, and the build settings section says so.

Which carrier a storage really serves. Storage is reported in the chapter of the
carrier it gives its energy back to, so heat storage is under district heat and
reaches the electricity balance only through whatever produced that heat.
Putting a number on that help needs a solved run.

The technology grouping is this tool's own: the source data names 63 unittypes
and no grouping of them. Units are grouped by which grids they touch, and the
report counts how many fell outside the groups -- watch that count, not this
paragraph, for a unittype the grouping has not met yet.

This is the only plotting code in the repository, and the only tool with tests
of its own -- ``tests/unit/test_input_data_summary.py`` pins the arithmetic
conventions whose failures a reader of the report could not see.

Exit code is 0 when report.md was written -- including when the timeseries
sections were skipped for want of a GAMS install or a GDX file, which is
documented degradation rather than failure -- 1 when inputData.xlsx is there but
unreadable, and 2 when the folder or the workbook is missing, so it can gate a
loop.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # no display on a build machine
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
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

#: Carrier chapters, in report order; hydro sits between electricity and district
#: heat. Hydrogen is last because not every scenario builds it. A carrier absent
#: from a build still gets its chapter.
CARRIERS = [
    ("elec", "Electricity", "elec"),
    ("dheat", "District heat", "dheat"),
    ("steam", "Industrial steam", "steam"),
    ("H2", "Hydrogen", "h2"),
]
CARRIER_GRIDS = frozenset(grid for grid, _, _ in CARRIERS)
CARRIER_TITLES = {grid: title for grid, title, _ in CARRIERS}

#: The carriers the map tints, which is not all of them. Three carriers make
#: seven combinations and a fill colour can carry that; a fourth makes fifteen
#: and it cannot. Steam is the one left off -- it is site-local, it has no
#: weather variation, and its story is a table rather than a colour.
MAP_CARRIERS = ("elec", "dheat", "H2")

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

#: Output capacity that generates nothing: storage gives back what was put in,
#: and demand response is demand left unserved. Each has its own subsection, so
#: a generation table leaves them out and no megawatt is counted twice in one
#: chapter.
STORAGE_LABELS = frozenset({"battery", "pumped storage", "heat storage"})
DEMAND_RESPONSE = "demand response"

HOURS_PER_YEAR = 8760


def annualise(hours: int) -> float:
    """The factor that turns a sum over ``hours`` into a per-year figure.

    A climate window is whatever length the build was configured with, and a
    sum over it is a per-year number only when it happens to be 8760 hours.
    """
    return HOURS_PER_YEAR / hours if hours else 0.0

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

#: The two nested groups the electricity-balancing analysis reports, widest
#: last. Each contains the one before it, so the second row is the first plus
#: hydro rather than hydro alone.
#:
#: Fixed and ordered rather than discovered, so a build with no closed-loop
#: pumped hydro prints a zero instead of losing the row -- but *membership* is
#: derived from the data, in storage_group_of, because the scenario families
#: rename their grids.
STORAGE_GROUPS = [
    ("elec_literal", "electricity storage", "#1f5f52"),
    ("elec_practical", "+ inflow hydro", "#4f9d8c"),
]

#: What each group means, in one clause, for the reader who meets it in a table.
GROUP_GLOSS = {
    "elec_literal": "battery and closed-loop pumped hydro, which hold energy the "
                    "power system put there itself",
    "elec_practical": "plus reservoir, open-loop pumped hydro and run-of-river, which "
                      "hold energy the weather puts there",
}

#: Sharp at the width a markdown viewer renders an embedded PNG, without the
#: multi-megabyte files 300 dpi would produce for a 16:9 figure.
FIG_DPI = 150
FIG_WIDTH_IN = 9.0

#: Europe is taller than it is wide once projected, so the maps are the one
#: figure that is not sized from its row count.
MAP_HEIGHT_IN = 9.5

#: Written by tools/prepare_zone_geometry.py, one per level. Absent is not an
#: error: the maps say what they wanted and the rest of the report is
#: unaffected. Both are cut from the same land, so flipping between the two
#: levels does not move a coastline -- bar a fraction of a pixel on the
#: Norwegian and Swedish mainlands, which are the only ones cut on a grid.
ZONE_ASSET = Path(__file__).resolve().parent / "maps" / "zone_shapes.geojson"
COUNTRY_ASSET = Path(__file__).resolve().parent / "maps" / "country_shapes.geojson"

NEIGHBOUR_FILL = "#efefef"
NEIGHBOUR_EDGE = "#d8d8d8"

#: Country borders over the zone fills. Heavy enough to read as the senior
#: boundary against the hairline white seams between zones, dark enough to show
#: on both the grey ring and the carrier fills.
COUNTRY_EDGE = "#4a4a57"
COUNTRY_EDGE_WIDTH = 1.1

#: What a carrier is doing in an area. Kept as words rather than booleans
#: because "a node exists but nothing demands it" is a third thing, and it is
#: the one worth seeing.
PRESENCE_ABSENT = "absent"
PRESENCE_NODE = "node"
PRESENCE_DEMAND = "demand"
PRESENCE_UNKNOWN = "unknown"

PRESENCE_LABEL = {
    PRESENCE_DEMAND: "modelled, has demand",
    PRESENCE_NODE: "modelled, nothing demands it",
    PRESENCE_UNKNOWN: "modelled, demand not read",
    PRESENCE_ABSENT: "not modelled here",
}

#: One fill per carrier combination. See _blend_for for why this is a table and
#: not arithmetic.
#:
#: Every area in every shipped scenario has electricity, so the four fills that
#: are actually drawn are the four that contain it. Those four carry the design:
#: district heat turns the blue green, hydrogen turns it violet, and both
#: together go to an indigo dark enough that no reader mistakes it for the light
#: blue of electricity alone. The earlier pair for the hydrogen combinations sat
#: too close to that light blue to be told apart across a map this size.
#:
#: The three combinations without electricity cannot occur here and are kept
#: anyway: a lookup that returns grey for an unlisted set would be a silent
#: answer, and this one costs nothing.
CARRIER_BLEND = {
    frozenset({"elec"}): "#a8cbe8",
    frozenset({"dheat"}): "#e8b48f",
    frozenset({"H2"}): "#dda8c8",
    frozenset({"elec", "dheat"}): "#7fb3a8",
    frozenset({"elec", "H2"}): "#c49bd4",
    frozenset({"dheat", "H2"}): "#cf9099",
    frozenset({"elec", "dheat", "H2"}): "#6b6fa8",
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

#: Production capacity this many times everything that could take its output
#: -- peak demand, the capacity that consumes the carrier, and export capacity
#: -- is capacity that can never run, and the check asks for it to be fixed.
#: In the shipped scenarios electricity stays below 1.3 everywhere, district
#: heat runs 1.0-1.5 apart from two countries, and industrial steam 1.1-8.6.
OVERSIZE_RATIO = 2.0

#: What the capacity check leaves out of production, per carrier: output that
#: is weather-driven or moves energy in time, neither of which ever runs at
#: nameplate against a peak.
NOT_DISPATCHABLE = {
    "elec": frozenset({"wind onshore", "wind offshore", "solar",
                       "battery", "pumped storage", "demand response"}),
    "dheat": frozenset({"solar", "heat storage"}),
}

MWH_TO_TWH = 1e-6
MW_TO_GW = 1e-3


def summarise(items: Sequence, limit: int = NAME_LIMIT, sep: str = ", ") -> str:
    """The first few names, then how many are left.

    One name over the limit is named rather than counted: "and 1 more" costs
    the reader a lookup and saves no space. ``sep`` is for items that carry
    commas of their own.
    """
    items = [str(i) for i in items]
    if len(items) <= limit + 1:
        return sep.join(items)
    return f"{sep.join(items[:limit])} and {len(items) - limit} more"


#: Plurals that are not the singular plus "s".
_PLURALS = {"country": "countries"}


def plural(count: int, word: str) -> str:
    """'1 country', '16 countries' -- a count with its noun in the right number."""
    if int(count) == 1:
        return f"{count} {word}"
    return f"{count} {_PLURALS.get(word, word + 's')}"


def and_join(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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
    userconstraint: pd.DataFrame = field(default_factory=pd.DataFrame)
    availability_limits: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The build's name in the config's ``variants``; '' for the default build.
    build: str = ""

    def built_units(self) -> List[str]:
        """The units this folder carries parameters for.

        ``unitUnittype`` can list more: a unit the builder dropped for having no
        capacity keeps its row there, and a source workbook commonly names more
        unit types than one scenario uses. The report follows what was built.
        """
        if self.p_unit.empty or "unit" not in self.p_unit.columns:
            return []
        return sorted(set(self.p_unit["unit"].astype(str)))

    def built_unittypes(self) -> List[str]:
        """The unit types of the units this folder carries."""
        uu = self.unit_unittype
        if uu.empty or not {"unit", "unittype"} <= set(uu.columns):
            return []
        built = set(self.built_units())
        return sorted({str(t) for u, t in zip(uu["unit"].astype(str), uu["unittype"])
                       if u in built})


def load_workbook(xlsx_path: Path) -> Workbook:
    """Read every sheet the report needs, in one pass over the file."""
    xl = pd.ExcelFile(xlsx_path)

    tags = read_bb_sheet(xl, "add_scen_tags")
    scenario = str(tags["scenario"].iloc[0]) if "scenario" in tags and len(tags) else "unknown"
    year = str(tags["year"].iloc[0]) if "year" in tags and len(tags) else "unknown"
    build = tags["variant"].iloc[0] if "variant" in tags and len(tags) else ""
    build = "" if pd.isna(build) else str(build)

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

    userconstraint = read_bb_sheet(xl, "p_userconstraint")
    if "value" in userconstraint.columns:
        userconstraint["value"] = pd.to_numeric(userconstraint["value"], errors="coerce")

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
        userconstraint=userconstraint,
        availability_limits=read_bb_sheet(xl, "utAvailabilityLimits").dropna(how="all"),
        build=build,
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

def state_grids(workbook: Workbook) -> set:
    """Every grid whose nodes hold energy, by Backbone's own definition.

    A state is ``p_gn.energyStoredPerUnitOfState`` above zero. Reading it from
    the boundary sheet instead -- every grid with a row there -- looked
    equivalent and was not: dheat and steam carry balancePenalty and maxSpill
    rows without holding anything, so both came back as storage grids and a
    district-heat charger was labelled pumped storage.

    Derived rather than listed, because the TYNDP scenarios rename 'battery' to
    'battery4h' and a literal set silently dropped a tenth of that build's
    capacity into "other" the one time it was tried.
    """
    gn = workbook.p_gn
    if gn.empty or not {"grid", "energyStoredPerUnitOfState"} <= set(gn.columns):
        # The builder drops an all-zero column, so an absent one means no node
        # in this build holds a state. That is an answer, not a gap.
        return set()
    held = pd.to_numeric(gn["energyStoredPerUnitOfState"], errors="coerce").fillna(0.0) > 0
    return set(gn.loc[held, "grid"].dropna().astype(str))


def derive_storage_grids(workbook: Workbook, inflow_grids: Optional[set]) -> set:
    """Grids that hold energy but receive no natural inflow."""
    return state_grids(workbook) - set(inflow_grids or HYDRO_INFLOW_GRIDS_FALLBACK)


def technology_label(grid: str, inflow_grids: set) -> str:
    """What a storage grid is reported as, inflow taking precedence over form.

    The same ladder _label_for climbs: a grid with natural inflow is hydro
    whatever machinery sits on it. Without this, psOpen read as "pumped
    storage" in the storage tables and "hydro" in the capacity table, and the
    Norwegian fleet -- which pumps a few per cent of what it turbines -- was
    the whole difference between them.
    """
    return "hydro" if str(grid) in inflow_grids else storage_label(grid)


def storage_label(grid: str) -> str:
    """Storage grids kept apart where their duration story differs.

    An unrecognised grid falls to OTHER_LABEL, not to 'pumped storage'. The old
    fall-through made every future storage grid -- and, while dheat was wrongly
    counted as one, a heat charger -- read as a Norwegian reservoir.
    """
    g = str(grid).lower()
    if g.startswith("battery"):
        return "battery"
    if g.startswith("heatstor"):
        return "heat storage"
    if g.startswith("ps"):
        return "pumped storage"
    if g in ("reservoir", "ror"):
        return "hydro"
    return OTHER_LABEL


@dataclass
class Classification:
    """Output-side capacity per unit, labelled, plus what could not be labelled."""
    rows: pd.DataFrame               # node, unit, carrier, capacity, label
    unlabelled: pd.DataFrame         # rows that fell to 'other', with the reason
    inflow_grids: set
    storage_grids: set               # states with no inflow, for the technology labels
    states: set                      # every grid that holds energy, inflow or not


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
        return Classification(empty, empty, set(), set(), set())

    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    io = io[active]

    inflow = set(inflow_grids) if inflow_grids else set(HYDRO_INFLOW_GRIDS_FALLBACK)
    states = state_grids(workbook)
    storage = states - inflow

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
        label, reason = _label_for(str(out_grid), in_grids, flow, inflow, storage, states)
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
    return Classification(rows, unlabelled, inflow, storage, states)


def _label_for(
    out_grid: str,
    in_grids: frozenset,
    flow: str,
    inflow_grids: set,
    storage_grids: set,
    states: set,
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

    # A unit whose output is a state is charging it, and the storage it fills is
    # the technology. A heat-storage charger consumes district heat and a pump
    # consumes electricity; neither is a district-heat or an electricity plant,
    # and labelling them by their input grid put 8.7 GW of heat chargers under
    # whatever the input grid happened to resolve to.
    if out_grid in states:
        charging = storage_label(out_grid)
        if charging != OTHER_LABEL:
            return charging, ""
        return OTHER_LABEL, f"storage grid '{out_grid}' is in no technology group"

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
        label = storage_label(in_grid)
        if label != OTHER_LABEL:
            return label, ""
        return OTHER_LABEL, f"storage grid '{in_grid}' is in no technology group"
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


def split_consumption(consumption: pd.DataFrame,
                      storage_grids: set) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(consumers, charging)``: what a carrier is made into, and the storage it fills.

    A unit whose product is a storage grid is charging it, and the storage
    subsection reports it. Listing it among the consumers too would show the
    same megawatt twice in one chapter.
    """
    if consumption.empty:
        return consumption, consumption
    filling = consumption["label"].astype(str).str[3:].isin(set(storage_grids))
    return consumption[~filling], consumption[filling]


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


def generation_only(capacity: pd.DataFrame, carrier: str) -> pd.DataFrame:
    """``capacity_by_area`` less storage and, for electricity, demand response.

    Both have subsections of their own. Demand response is left in for any
    other carrier, which has no subsection to put it in: dropping it there
    would lose it from the report altogether.
    """
    if capacity is None or capacity.empty:
        return pd.DataFrame()
    moved = STORAGE_LABELS | ({DEMAND_RESPONSE} if carrier == "elec" else set())
    kept = capacity[[c for c in capacity.columns if c not in moved]]
    return kept[kept.sum(axis=1) > 0]


def constant_demand(workbook: Workbook, carrier: str, zones: bool) -> pd.DataFrame:
    """A carrier's constant demand per area, in the shape ``annual_range`` returns.

    Demand with no timeseries processor of its own is written as a flat MWh/h in
    ``p_gn``'s ``influx`` column instead of an hourly series, stored negative the
    way Backbone consumes it. Industrial steam is the whole of it in the shipped
    scenarios -- 495 TWh/yr, more than district heat -- and reading only the
    ``ts_influx`` families misses every watt of it.

    Only the outflux is summed, never the net. A grid with inflow at one node and
    demand at another would otherwise cancel down to a number that is neither.
    """
    gn = workbook.p_gn
    if gn.empty or "grid" not in gn.columns or "influx" not in gn.columns:
        return pd.DataFrame()
    rows = gn[gn["grid"].astype(str) == carrier].copy()
    if rows.empty:
        return pd.DataFrame()
    influx = col_or(rows, "influx", 0.0).fillna(0.0)
    rows["TWh"] = (-influx).clip(lower=0.0) * HOURS_PER_YEAR * MWH_TO_TWH
    rows["area"] = [area_of(n, zones) for n in rows["node"]]
    per_area = rows.groupby("area", observed=True)["TWh"].sum()
    per_area = per_area[per_area > 0]
    if per_area.empty:
        return pd.DataFrame()
    return (pd.DataFrame({"area": per_area.index, "mean": per_area.to_numpy()})
            .sort_values("mean", ascending=False))


def transfer_by_area(workbook: Workbook, zones: bool,
                     grid: str = "elec") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """One carrier's transfer capacity per area, and the corridor matrix.

    Each corridor appears twice, once per direction, and 15 of the 44 in the
    shipped scenarios carry a different capacity each way. A corridor is
    summarised by the mean of its two directions: summing them double-counts,
    and taking one of them makes the answer depend on which row the builder
    happened to write first.

    A corridor is a node pair. For electricity that is a zone pair; district
    heat can link networks inside one zone -- the Helsinki-area links are three
    corridors in FI00 -- and calling them one would be wrong in the direction
    of reassurance. Links inside one area are kept apart from its cross-border
    total, so Sweden's number is not inflated by SE01-SE02.
    """
    gnn = workbook.p_gnn
    cols = {"grid", "from_node", "to_node"}
    if gnn.empty or not cols <= set(gnn.columns):
        return pd.DataFrame(), pd.DataFrame(), {}

    links = gnn[gnn["grid"].astype(str) == grid].copy()
    if links.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    links["cap"] = pd.to_numeric(col_or(links, "transferCap", 0.0), errors="coerce").fillna(0.0)
    links["pair"] = [tuple(sorted((str(a), str(b))))
                     for a, b in zip(links["from_node"], links["to_node"])]

    per_pair = links.groupby("pair", observed=True).agg(
        mean_cap=("cap", "mean"),
        directions=("cap", "size"),
        spread=("cap", lambda s: float(s.max() - s.min())),
    ).reset_index()

    asymmetric = per_pair[per_pair["spread"] > 0]
    one_way = per_pair[
        (per_pair["directions"] == 2) & (per_pair["mean_cap"] > 0)
        & per_pair["pair"].map(
            lambda p: bool(((links["pair"] == p) & (links["cap"] == 0)).any())
        )
    ]

    rows, inside = [], 0
    for (a, b), cap in zip(per_pair["pair"], per_pair["mean_cap"]):
        near_zone, far_zone = zone_of(a), zone_of(b)
        internal = (near_zone == far_zone) if zones else country_of(near_zone) == country_of(far_zone)
        inside += int(internal)
        # A cross-border corridor is capacity for the area at each of its ends,
        # so it is recorded twice, once per end. An internal one has both ends
        # in the same area and is recorded once: adding it per end would report
        # Sweden's own ring at twice its size.
        ends = ((near_zone, far_zone),) if internal else ((near_zone, far_zone), (far_zone, near_zone))
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

    def name(pair: Tuple[str, str]) -> str:
        """A zone pair for electricity, the nodes themselves where both share a zone."""
        a, b = pair
        return f"{zone_of(a)}-{zone_of(b)}" if zone_of(a) != zone_of(b) else f"{a}-{b}"

    facts = {
        "pairs": int(len(per_pair)),
        "asymmetric": int(len(asymmetric)),
        "one_way": int(len(one_way)),
        "inside": inside,
        "total_GW": float(per_pair["mean_cap"].sum() * MW_TO_GW),
        "largest_asymmetries": [
            (name(p), float(s))
            for p, s in asymmetric.nlargest(3, "spread")[["pair", "spread"]].itertuples(index=False)
        ],
        "transfer_loss": _uniform_or_none(links, "transferLoss"),
    }
    return per_area, matrix, facts


def _uniform_or_none(df: pd.DataFrame, column: str) -> Optional[float]:
    """The single value of a column, when the build wrote one and it is uniform."""
    if column not in df.columns:
        return None
    vals = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    return float(vals[0]) if len(vals) == 1 else None


#: Per-node energy, in the three ways a storage's size can honestly be stated.
STORAGE_MEASURES = ["nameplate_MWh", "usable_mean_MWh", "usable_envelope_MWh"]


def storage_carriers(workbook: Workbook) -> Dict[str, Tuple[frozenset, frozenset]]:
    """``{grid: (charged from, discharged to)}`` for every grid units connect.

    Which carrier a storage serves is a fact about the units attached to it, not
    about its name, and the scenario families do not agree on the names.
    """
    io = workbook.p_gnu_io
    out: Dict[str, Tuple[set, set]] = {}
    if io.empty or not {"grid", "unit", "input_output"} <= set(io.columns):
        return {}
    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    io = io[active]
    side = io["input_output"].astype(str)
    ins = io[side == "input"].groupby("unit", observed=True)["grid"].agg(
        lambda s: {str(v) for v in s}).to_dict()
    outs = io[side == "output"].groupby("unit", observed=True)["grid"].agg(
        lambda s: {str(v) for v in s}).to_dict()
    for unit, produced in outs.items():
        for grid in produced:                       # this unit charges `grid`
            entry = out.setdefault(grid, (set(), set()))
            entry[0].update(ins.get(unit, set()) - {grid})
    for unit, consumed in ins.items():
        for grid in consumed:                       # this unit discharges `grid`
            entry = out.setdefault(grid, (set(), set()))
            entry[1].update(outs.get(unit, set()) - {grid})
    return {g: (frozenset(a), frozenset(b)) for g, (a, b) in out.items()}


def storage_group_of(
    grid: str,
    inflow_grids: set,
    states: set,
    carriers: Dict[str, Tuple[frozenset, frozenset]],
    ) -> Optional[str]:
    """Which nested group a storage grid belongs to, or None for neither.

    Inflow makes a storage practical rather than literal: an open-loop pumped
    hydro node is filled by weather far more than by its pump -- Norwegian
    psOpen pumping is a few per cent of its turbining -- so counting it as
    energy the power system put there would be counting the spring melt.

    A literal storage must take electricity *and* give it back. Testing only one
    end would file an electrolyser's product under electricity storage. A storage
    charged and discharged on some other carrier is in neither group, which is
    where heat storage sits: it is real and it helps, but not at full value and
    not on this balance.
    """
    grid = str(grid)
    if grid not in states:
        return None
    if grid in inflow_grids:
        return "elec_practical"
    charged, discharged = carriers.get(grid, (frozenset(), frozenset()))
    if "elec" in charged and "elec" in discharged:
        return "elec_literal"
    return None


def storage_groups_by_grid(workbook: Workbook, inflow_grids: set) -> Dict[str, Optional[str]]:
    """The group of every grid that holds energy, derived from the data."""
    states = state_grids(workbook)
    carriers = storage_carriers(workbook)
    return {g: storage_group_of(g, set(inflow_grids), states, carriers) for g in states}


@dataclass
class StorageInventory:
    """Every storage node in the build, sized three ways and grouped two ways."""
    by_node: pd.DataFrame            # grid, node, area, group, label, source, measures, power_MW
    by_group: pd.DataFrame           # cumulative group x measures, TWh, plus power_GW
    by_area: pd.DataFrame            # area x group, cumulative, TWh and GW
    power: pd.DataFrame              # area x label discharge power, GW
    ratio_facts: Dict
    facts: Dict


def _boundary_storage_energy(
    workbook: Workbook,
    storage_limit: Optional[pd.DataFrame],
    facts: Dict,
    ) -> pd.DataFrame:
    """Per-node energy for the storage nodes whose size is an upwardLimit.

    A row either carries a constant or says useTimeseries, never both. The
    constant rows are paired with their downwardLimit, which the builder writes
    as Eps for every storage node, so their usable volume is the ceiling. The
    twelve rows that say useTimeseries are every large Nordic and Alpine
    reservoir; their size -- and the seasonal floor underneath it -- is only in
    the GDX, which is why reading the workbook alone sees a fraction of it.
    """
    boundary = workbook.boundary
    if boundary.empty or "param_gnBoundaryTypes" not in boundary.columns:
        return pd.DataFrame()
    kinds = boundary["param_gnBoundaryTypes"].astype(str)
    rows = boundary[kinds.isin(["upwardLimit", "downwardLimit"])].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["kind"] = kinds[rows.index]
    rows["grid"] = rows["grid"].astype(str)
    rows["node"] = rows["node"].astype(str)
    rows["constant"] = col_or(rows, "constant", np.nan)
    rows["use_ts"] = col_or(rows, "useTimeseries", 0.0).fillna(0.0) == 1

    wide = rows.pivot_table(index=["grid", "node"], columns="kind",
                            values="constant", observed=True)
    tops = rows[rows["kind"] == "upwardLimit"].set_index(["grid", "node"])
    flags = tops["use_ts"]
    flags = flags[~flags.index.duplicated()]
    upward = tops.index.unique()
    if not len(upward):
        return pd.DataFrame()

    ceiling = wide["upwardLimit"] if "upwardLimit" in wide.columns else pd.Series(dtype="float64")
    floor = (wide["downwardLimit"] if "downwardLimit" in wide.columns
             else pd.Series(0.0, index=wide.index)).fillna(0.0)

    series = storage_limit if storage_limit is not None else pd.DataFrame()
    if not series.empty:
        series = series.set_index(
            pd.MultiIndex.from_arrays([series["grid"].astype(str), series["node"].astype(str)])
        )

    built, missing = [], []
    for key in upward:
        if bool(flags.get(key, False)):
            if not series.empty and key in series.index:
                row = series.loc[key]
                built.append({
                    "grid": key[0], "node": key[1], "source": "timeseries",
                    "nameplate_MWh": float(row["nameplate_MWh"]),
                    "usable_mean_MWh": float(row["usable_mean_MWh"]),
                    "usable_envelope_MWh": float(row["usable_envelope_MWh"]),
                })
            else:
                missing.append(key[1])
            continue
        top = float(ceiling.get(key, np.nan)) if len(ceiling) else float("nan")
        if not np.isfinite(top):
            continue
        usable = max(top - float(floor.get(key, 0.0)), 0.0)
        built.append({
            "grid": key[0], "node": key[1], "source": "workbook constant",
            "nameplate_MWh": top, "usable_mean_MWh": usable, "usable_envelope_MWh": usable,
        })
    facts["timeseries_nodes"] = sorted(str(k[1]) for k in upward if bool(flags.get(k, False)))
    facts["timeseries_missing"] = sorted(missing)
    return pd.DataFrame(built)


def storage_power_by_node(workbook: Workbook, states: set) -> pd.DataFrame:
    """Discharge power per storage node, read on the storage side of the unit.

    The input-side row of a discharging unit: a battery's charger has an output
    row on the same grid, and counting both would double every storage node. These are
    the same megawatts the electricity capacity table reports as hydro or
    battery -- read there per carrier, here per storage node.
    """
    return storage_side_power_by_node(workbook, states, "input")


def storage_side_power_by_node(workbook: Workbook, grids: set, side: str) -> pd.DataFrame:
    """Unit capacity per node, read on the storage's own row, from one side of the unit.

    ``side="input"`` is what units can draw out of a storage -- a turbine's water,
    a battery's discharge. ``side="output"`` is what units can put into it -- a
    pump's delivery. Both are the storage grid's row, so no efficiency sits
    between the number and the storage.
    """
    io = workbook.p_gnu_io
    empty = pd.DataFrame(columns=["grid", "node", "power_MW"])
    if io.empty or "input_output" not in io.columns:
        return empty
    active = col_or(io, "isActive", 1.0).fillna(1.0) == 1
    rows = io[active & (io["input_output"].astype(str) == side)].copy()
    rows["grid"] = rows["grid"].astype(str)
    wanted = rows[rows["grid"].isin(set(grids))]
    if wanted.empty:
        return empty
    work = wanted.copy()
    work["node"] = work["node"].astype(str)
    work["power_MW"] = pd.to_numeric(work["capacity"], errors="coerce").fillna(0.0)
    return work.groupby(["grid", "node"], observed=True)["power_MW"].sum().reset_index()


def storage_power_by_area(
    workbook: Workbook,
    zones: bool,
    states: set,
    inflow_grids: set,
    ) -> pd.DataFrame:
    """area x technology discharge power, GW, for every storage node in the build."""
    per_node = storage_power_by_node(workbook, states)
    if per_node.empty:
        return pd.DataFrame()
    work = per_node.copy()
    work["label"] = [technology_label(g, inflow_grids) for g in work["grid"]]
    work["area"] = [area_of(n, zones) for n in work["node"]]
    table = (
        work.groupby(["area", "label"], observed=True)["power_MW"].sum().unstack(fill_value=0.0)
        * MW_TO_GW
    )
    ordered = [l for l in TECH_STYLE if l in table.columns]
    return table[ordered + [c for c in table.columns if c not in ordered]]


def storage_inventory(
    workbook: Workbook,
    timeseries: "Timeseries",   # defined below, in the timeseries layer
    zones: bool,
    inflow_grids: Optional[set] = None,
    ) -> StorageInventory:
    """Every storage node, sized three ways, in two nested electricity groups.

    The two routes a size is written by stay separate until the last step: an
    upwardLimit in MWh for hydro, a duration per MW in p_gnu_io for battery and
    heat. Nothing in the shipped scenarios states both, and a node that did
    would be counted once, by its upwardLimit, and named.
    """
    inflow = set(inflow_grids) if inflow_grids else set(HYDRO_INFLOW_GRIDS_FALLBACK)
    states = state_grids(workbook)
    groups = storage_groups_by_grid(workbook, inflow)
    facts: Dict = {"timeseries_nodes": [], "timeseries_missing": [], "both_routes": [],
                   "grids_without_energy": [], "outside_groups": {}, "by_source": {},
                   "provenance": timeseries.storage_limit_from}

    bounded = _boundary_storage_energy(workbook, timeseries.storage_limit, facts)
    ratio_table, ratio_facts = storage_energy_from_ratio(workbook, zones)

    parts = [bounded] if not bounded.empty else []
    if not ratio_table.empty:
        seen = set(zip(bounded["grid"], bounded["node"])) if not bounded.empty else set()
        by_ratio = ratio_table.groupby(["grid", "node"], observed=True)["MWh"].sum().reset_index()
        by_ratio["grid"] = by_ratio["grid"].astype(str)
        by_ratio["node"] = by_ratio["node"].astype(str)
        facts["both_routes"] = sorted(
            n for g, n in zip(by_ratio["grid"], by_ratio["node"]) if (g, n) in seen)
        by_ratio = by_ratio[[(g, n) not in seen
                             for g, n in zip(by_ratio["grid"], by_ratio["node"])]]
        if not by_ratio.empty:
            # Storage sized as a duration has no seasonal bound, so its three measures agree.
            for measure in STORAGE_MEASURES:
                by_ratio[measure] = by_ratio["MWh"]
            parts.append(by_ratio.assign(source="duration in p_gnu_io").drop(columns=["MWh"]))

    power = storage_power_by_node(workbook, states)
    if not parts:
        empty = pd.DataFrame(columns=["grid", "node", "area", "group"] + STORAGE_MEASURES)
        facts["grids_without_energy"] = sorted(set(power["grid"])) if not power.empty else []
        return StorageInventory(empty, pd.DataFrame(), pd.DataFrame(),
                                storage_power_by_area(workbook, zones, states, inflow),
                                ratio_facts, facts)

    by_node = pd.concat(parts, ignore_index=True)
    by_node["area"] = [area_of(n, zones) for n in by_node["node"]]
    by_node["label"] = [technology_label(g, inflow) for g in by_node["grid"]]
    by_node["group"] = [groups.get(str(g)) for g in by_node["grid"]]
    by_node = by_node.merge(power, on=["grid", "node"], how="left")
    by_node["power_MW"] = by_node["power_MW"].fillna(0.0)

    # A state grid carrying discharge power but no stated energy anywhere is a
    # property of the data worth naming, not a gap in this tool.
    facts["grids_without_energy"] = (
        sorted(set(power["grid"]) - set(by_node["grid"].astype(str))) if not power.empty else [])
    facts["by_source"] = {
        str(s): float(v * MWH_TO_TWH)
        for s, v in by_node.groupby("source", observed=True)["usable_mean_MWh"].sum().items()
    }
    facts["outside_groups"] = {
        str(g): float(v * MWH_TO_TWH)
        for g, v in by_node[by_node["group"].isna()]
        .groupby("grid", observed=True)["usable_mean_MWh"].sum().items()
    }
    return StorageInventory(
        by_node=by_node,
        by_group=_group_totals(by_node, None),
        by_area=_group_totals(by_node, "area"),
        power=storage_power_by_area(workbook, zones, states, inflow),
        ratio_facts=ratio_facts,
        facts=facts,
    )


def _group_totals(by_node: pd.DataFrame, within: Optional[str]) -> pd.DataFrame:
    """Cumulative group totals in TWh and GW, for the build or per area.

    Cumulative because the groups are nested: the practical row is the literal
    row plus hydro, and a reader who has to add two rows to get the number the
    figure draws will eventually add them wrong.
    """
    if by_node.empty:
        return pd.DataFrame()
    keys = ([within] if within else []) + ["group"]
    columns = STORAGE_MEASURES + ["power_MW"]
    grouped = by_node.groupby(keys, observed=True)[columns].sum()
    areas = sorted(by_node[within].unique()) if within else [None]

    rows = []
    for area in areas:
        running = {c: 0.0 for c in columns}
        for key, _, _ in STORAGE_GROUPS:
            index = (area, key) if within else key
            if index in grouped.index:
                for column in columns:
                    running[column] += float(grouped.loc[index, column])
            row = {"group": key}
            if within:
                row[within] = area
            for measure in STORAGE_MEASURES:
                row[measure.replace("_MWh", "_TWh")] = running[measure] * MWH_TO_TWH
            row["power_GW"] = running["power_MW"] * MW_TO_GW
            rows.append(row)
    table = pd.DataFrame(rows)
    return table.set_index([within, "group"] if within else "group")


def storage_grids_serving(workbook: Workbook, carrier: str, inflow_grids: set) -> List[str]:
    """The storage grids that give their energy back to ``carrier``, inflow hydro aside.

    Storage is reported in the chapter of the carrier it discharges to: a
    battery under electricity, a heat pit under district heat. Inflow hydro
    discharges to electricity too, but the weather fills it, and the hydro
    chapter follows it from the water's side.
    """
    carriers = storage_carriers(workbook)
    candidates = state_grids(workbook) - CARRIER_GRIDS - set(inflow_grids)
    return sorted(g for g in candidates
                  if carrier in carriers.get(g, (frozenset(), frozenset()))[1])


def storage_by_area(workbook: Workbook, inventory: StorageInventory, grids: Sequence[str],
                    zones: bool) -> pd.DataFrame:
    """Per area: discharge GW per technology, then ``charge_GW`` and ``energy_GWh``.

    Discharge and charge are read on the storage's own rows, so no efficiency sits
    between them and the storage. Energy is the usable volume, which for storage
    sized as a duration is its power times those hours.
    """
    grids = set(grids)
    if not grids:
        return pd.DataFrame()
    out = storage_side_power_by_node(workbook, grids, "input")
    into = storage_side_power_by_node(workbook, grids, "output")
    if out.empty and into.empty:
        return pd.DataFrame()

    def with_area(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(area=[area_of(n, zones) for n in frame["node"]],
                            label=[storage_label(g) for g in frame["grid"]])

    table = pd.DataFrame()
    if not out.empty:
        table = (with_area(out).groupby(["area", "label"], observed=True)["power_MW"].sum()
                 .unstack(fill_value=0.0) * MW_TO_GW)
        ordered = [l for l in TECH_STYLE if l in table.columns]
        table = table[ordered + [c for c in table.columns if c not in ordered]]
    charge = (with_area(into).groupby("area", observed=True)["power_MW"].sum() * MW_TO_GW
              if not into.empty else pd.Series(dtype="float64"))
    nodes = inventory.by_node
    energy = (nodes[nodes["grid"].astype(str).isin(grids)]
              .groupby("area", observed=True)["usable_mean_MWh"].sum() / 1e3
              if len(nodes) else pd.Series(dtype="float64"))
    index = table.index.union(charge.index).union(energy.index)
    table = table.reindex(index).fillna(0.0)
    table["charge_GW"] = charge.reindex(index).fillna(0.0)
    table["energy_GWh"] = energy.reindex(index).fillna(0.0)
    return table


def storage_discharge(table: pd.DataFrame) -> pd.Series:
    """Per area discharge GW from a ``storage_by_area`` table, every technology together."""
    if table is None or table.empty:
        return pd.Series(dtype="float64")
    return table[[c for c in table.columns if c not in ("charge_GW", "energy_GWh")]].sum(axis=1)


def demand_response(workbook: Workbook, classification: Classification,
                    zones: bool) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Electricity demand response, area x tier in GW, and each tier's cost range.

    A tier is a unit type: the build writes one unit per zone and tier, each
    flagged ``isSource`` because it makes electricity from nothing -- which is
    how input data writes demand that may go unserved. Its cost is the
    ``vomCosts`` on its electricity row, EUR/MWh.
    """
    rows = classification.rows
    rows = rows[(rows["carrier"] == "elec") & (rows["label"] == DEMAND_RESPONSE)]
    if rows.empty:
        return pd.DataFrame(), {}
    uu = workbook.unit_unittype
    tier_of = (dict(zip(uu["unit"].astype(str), uu["unittype"].astype(str)))
               if not uu.empty and {"unit", "unittype"} <= set(uu.columns) else {})
    work = rows.assign(area=[area_of(n, zones) for n in rows["node"]],
                       tier=[tier_of.get(str(u), str(u)) for u in rows["unit"]])
    table = (work.groupby(["area", "tier"], observed=True)["capacity"].sum()
             .unstack(fill_value=0.0) * MW_TO_GW)
    table = table[sorted(table.columns)]

    costs: Dict[str, Tuple[float, float]] = {}
    io = workbook.p_gnu_io
    if "vomCosts" in io.columns:
        elec_out = io[(io["input_output"].astype(str) == "output")
                      & (io["grid"].astype(str) == "elec")]
        by_unit = dict(zip(elec_out["unit"].astype(str),
                           pd.to_numeric(elec_out["vomCosts"], errors="coerce")))
        for tier, part in work.groupby("tier", observed=True):
            values = [by_unit.get(str(u)) for u in part["unit"]]
            values = [float(v) for v in values if v is not None and np.isfinite(v)]
            if values:
                costs[str(tier)] = (min(values), max(values))
    return table, costs


def carrier_links(workbook: Workbook, carrier: str) -> Dict:
    """How one carrier's generation is tied to electricity, in GW, for the whole build.

    For a carrier other than electricity: ``co_GW`` is its capacity on units
    that also generate electricity -- combined heat and power -- with
    ``co_elec_GW`` their electricity and ``flexible_GW`` the part whose two
    outputs may move apart between limits, ``cv`` on top of ``cb``. The rest
    keeps a fixed ratio. ``from_elec_GW`` is its capacity made from electricity.

    For electricity: ``co_by_carrier`` is its capacity on units that also make
    each other carrier.
    """
    io = workbook.p_gnu_io
    if io.empty or not {"grid", "unit", "input_output", "capacity"} <= set(io.columns):
        return {}
    io = io[col_or(io, "isActive", 1.0).fillna(1.0) == 1]
    side = io["input_output"].astype(str)
    outputs = io[side == "output"].assign(
        _grid=lambda f: f["grid"].astype(str), _unit=lambda f: f["unit"].astype(str),
        _cap=lambda f: pd.to_numeric(f["capacity"], errors="coerce").fillna(0.0))
    makes = outputs.groupby("_unit", observed=True)["_grid"].agg(set).to_dict()
    takes = (io[side == "input"].assign(_unit=lambda f: f["unit"].astype(str))
             .groupby("_unit", observed=True)["grid"].agg(lambda s: {str(v) for v in s}).to_dict())

    def gw(units: set, grid: str) -> float:
        part = outputs[outputs["_unit"].isin(units) & (outputs["_grid"] == grid)]
        return float(part["_cap"].sum() * MW_TO_GW)

    if carrier == "elec":
        co = {}
        for other, _, _ in CARRIERS:
            if other == "elec":
                continue
            value = gw({u for u, g in makes.items() if {"elec", other} <= g}, "elec")
            if value:
                co[other] = value
        return {"co_by_carrier": co}

    chp = {u for u, g in makes.items() if {"elec", carrier} <= g}
    flexible = set()
    if "cv" in outputs.columns:
        cv = pd.to_numeric(outputs["cv"], errors="coerce").fillna(0.0)
        flexible = set(outputs.loc[cv != 0, "_unit"]) & chp
    from_elec = {u for u, g in takes.items() if "elec" in g and carrier in makes.get(u, set())}
    return {
        "co_GW": gw(chp, carrier),
        "co_elec_GW": gw(chp, "elec"),
        "flexible_GW": gw(flexible, carrier),
        "from_elec_GW": gw(from_elec, carrier),
    }


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

    # A storage node starts at a level (boundStart) or at a share of its band
    # (boundStartRelative), never both; either column is absent when no node set it.
    flags = [c for c in ("boundStart", "boundStartRelative") if c in gn.columns]
    if not gn.empty and flags:
        bound = gn[flags].apply(pd.to_numeric, errors="coerce").fillna(0).eq(1).any(axis=1)
        facts["bound_start_nodes"] = int(bound.sum())
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

    A battery, a pumped-storage plant and a demand-response block are all counted in the
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
# What the build recorded about itself
# ============================================================================

@dataclass
class BuildRecord:
    """The build's own record of its settings, read from beside the data.

    ``config`` is the config snapshot the build cache keeps
    (``cache/config_structural.json``), so it is what the build actually used
    even if the config file has changed since.
    """
    config: Dict = field(default_factory=dict)
    flags: Dict = field(default_factory=dict)
    config_file: Optional[str] = None
    built_at: Optional[str] = None
    variant_lines: List[str] = field(default_factory=list)
    warnings: int = 0
    errors: int = 0
    forecast_files: List[str] = field(default_factory=list)


#: Status marks the build log opens its lines with.
_LOG_MARKS = "✓⚠️❌⏩ "


def read_build_record(folder: Path) -> BuildRecord:
    """What ``build_input_data.py`` wrote about this folder: config, variants, status.

    Every part is optional. A folder copied without its cache or its log still
    gets a report, with the settings it cannot tell left out.
    """
    record = BuildRecord()
    for name, attr in (("config_structural.json", "config"), ("general_flags.json", "flags")):
        try:
            setattr(record, attr, json.loads((folder / "cache" / name).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    # The same lookup as `run_model.py --list`: which config in src_files/ names
    # this folder by the builder's own naming rule.
    try:
        from run_model import config_folders
        found = config_folders().get(folder.resolve().name)
        record.config_file = found[0] if found else None
    except Exception:                              # noqa: BLE001 -- a hint, not a requirement
        record.config_file = None

    try:
        lines = (folder / "summary.log").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    in_summary = False
    for line in lines:
        text = line.strip()
        body = text.lstrip(_LOG_MARKS)
        if body.startswith("Run timestamp:") and record.built_at is None:
            record.built_at = body.split(":", 1)[1].strip()
        elif body.startswith(("Variant '", "Dormant variants")):
            record.variant_lines.append(body)
        elif "Warnings and errors summary" in text:
            in_summary = True
        elif in_summary and text.startswith("⚠"):
            record.warnings += 1
        elif in_summary and text.startswith("❌"):
            record.errors += 1
    record.forecast_files = sorted(p.name for p in folder.glob("ts_*_forecasts.gdx"))
    return record


def investment_and_timing(workbook: Workbook) -> List[str]:
    """What lets the fleet change during a run, which this report does not follow."""
    def set_rows(frame: pd.DataFrame, column: str) -> int:
        if frame.empty or column not in frame.columns:
            return 0
        return int((pd.to_numeric(frame[column], errors="coerce").fillna(0) != 0).sum())

    found = []
    for frame, column, noun, what in (
            (workbook.p_gn, "maxInvest", "storage node", "may grow its storage"),
            (workbook.p_unit, "maxUnitCount", "unit", "may be invested in"),
            (workbook.p_gnn, "transferCapInvLimit", "corridor row", "may be invested in")):
        n = set_rows(frame, column)
        if n:
            found.append(f"{plural(n, noun)} {what} (`{column}`)")
    limits = workbook.availability_limits
    if not limits.empty:
        units = limits["unit"].nunique() if "unit" in limits.columns else len(limits)
        found.append(f"{plural(units, 'unit')} become available or unavailable during the run "
                     f"(`utAvailabilityLimits`)")
    return found


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
    storage_limit_from: Optional[str] = None
    skipped: Optional[str] = None
    #: t labels per climate window: the build's window length in days times 24.
    #: ``hours_per_year`` is what the annual and hourly figures use -- the first
    #: 365 days of a longer window, so overlapping windows count each climate
    #: year once. The hydro checks run on the whole window.
    window_hours: int = 0
    #: Per-node hydro series over each whole window, for the hydro checks.
    hydro: Optional["HydroSeries"] = None
    #: carrier -> the nodes its demand series is non-zero on.
    demand_nodes: Dict[str, set] = field(default_factory=dict)

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


def _node_hourly(records: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """One file's records as a t x node frame, or None for no records.

    Pivoted once per file: the area sums and the per-node hydro series are both
    read from it.
    """
    if records is None or records.empty:
        return None
    return (
        records.assign(node=records["node"].astype(str))
        .pivot_table(index="t", columns="node", values="value", aggfunc="sum", observed=True)
        .sort_index()
    )


def _collapse_to_areas(
    wide: Optional[pd.DataFrame],
    areas: List[str],
    weight_by_node: Optional[Dict[str, float]] = None,
    sign: float = 1.0,
    ) -> Optional[np.ndarray]:
    """A t x node frame as an (hours x areas) array, summed into areas.

    `weight_by_node` turns a per-unit capacity factor into MW. A node with no
    weight contributes nothing, which is how a capacity-factor series for a node
    carrying no units of that flow is ignored rather than counted as 1 MW.
    """
    if wide is None:
        return None
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


@dataclass
class HydroSeries:
    """Per-node hydro series over each whole climate window.

    The rest of the timeseries layer collapses nodes into areas as it reads. The
    hydro checks cannot: a minimum binds one storage node, and two nodes summed
    can hold water neither of them has. Every array is (windows, hours), in the
    order of ``years``, float32, NaN where a window has no data for that node --
    no file, or hours past the end of a shorter window.
    """
    years: List[int] = field(default_factory=list)
    hours: int = 0
    grid_of: Dict[str, str] = field(default_factory=dict)        # node -> grid, from ts_influx
    inflow: Dict[str, np.ndarray] = field(default_factory=dict)   # MWh/h
    upward: Dict[str, np.ndarray] = field(default_factory=dict)   # MWh, lowest forecast branch
    downward: Dict[str, np.ndarray] = field(default_factory=dict)  # MWh, 0 where none is written


def _t_positions(labels) -> Optional[np.ndarray]:
    """The 0-based hour of each t label -- 't000123' is 122 -- or None if any is not one."""
    text = pd.Index(labels).astype(str)
    digits = text.str.slice(1)
    if not len(text) or not (text.str.startswith("t").all() and digits.str.isdigit().all()):
        return None
    return digits.astype(int).to_numpy() - 1


def _without_initial_step(records: pd.DataFrame) -> pd.DataFrame:
    """ts_node records without t000000, which is not an hour of the window.

    The build writes t000000 into ts_node as a copy of t000001, because Backbone
    reads a storage's starting state at that step. ``_t_positions`` would place
    it at hour -1, which numpy reads as the last hour, and a mean over the hours
    would count t000001 twice.
    """
    if "t" not in records.columns:
        return records
    return records[records["t"] != "t000000"]


def _node_matrix(wide: Optional[pd.DataFrame]) -> Optional[Tuple[List[str], np.ndarray]]:
    """A t x node frame as ``(nodes, dense (hours, nodes) float32)``, placed by t label.

    GDX stores no zeros, so an hour that every node spends at zero has no row at
    all, and an array built from the rows present would pull every later hour
    one step early. Placed by label instead, the missing cells being the zero
    they stand for.
    """
    if wide is None or wide.empty:
        return None
    positions = _t_positions(wide.index)
    if positions is None:
        return None
    matrix = np.zeros((int(positions.max()) + 1, wide.shape[1]), dtype="float32")
    matrix[positions, :] = wide.fillna(0.0).to_numpy(dtype="float32")
    return [str(c) for c in wide.columns], matrix


def _stack_windows(
    per_window: Sequence[Optional[Tuple[List[str], np.ndarray]]],
    absent_node: float,
    ) -> Dict[str, np.ndarray]:
    """node -> (windows, hours), one row per window in order.

    A window with no file stays NaN. A node missing from a window that does
    have a file is ``absent_node``: zero for inflow, which GDX does not store,
    and NaN for a storage bound, which cannot be read as a ceiling of zero.
    """
    present = [part for part in per_window if part is not None]
    if not present:
        return {}
    hours = max(matrix.shape[0] for _, matrix in present)
    nodes = sorted({node for names, _ in present for node in names})
    out = {node: np.full((len(per_window), hours), np.nan, dtype="float32") for node in nodes}
    for window, part in enumerate(per_window):
        if part is None:
            continue
        names, matrix = part
        length = matrix.shape[0]
        if not np.isnan(absent_node):
            for node in nodes:
                out[node][window, :length] = absent_node
        for column, node in enumerate(names):
            out[node][window, :length] = matrix[:, column]
    return out


def _pad_hours(series: Dict[str, np.ndarray], hours: int) -> Dict[str, np.ndarray]:
    """Every array widened with NaN to ``hours``, so inflow and bounds line up."""
    out = {}
    for node, array in series.items():
        if array.shape[1] < hours:
            wider = np.full((array.shape[0], hours), np.nan, dtype="float32")
            wider[:, :array.shape[1]] = array
            array = wider
        out[node] = array
    return out


def _read_storage_limit_series(
    gx, container, folder: Path, years: Sequence[int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-node upward and downward limits for every climate window, hour by hour.

    ``_read_storage_limit`` reads one file for three aggregate measures. The
    hydro checks need the hours themselves, from the same window as the inflow
    they are set against, so this reads every per-year file on the lowest
    forecast branch, which is f00 there.
    """
    available = year_files(folder, "ts_node_hydro_storage_limits")
    if not available or not years:
        return {}, {}
    ordered = [available.get(y, folder / f"ts_node_hydro_storage_limits_{y}.gdx") for y in years]
    parts: Dict[str, List] = {"upwardLimit": [], "downwardLimit": []}

    def reduce_one(_gdx_file, records):
        usable = (records is not None and not records.empty
                  and "param_gnBoundaryTypes" in records.columns)
        if usable:
            records = _without_initial_step(records)
        if usable and "f" in records.columns:
            branch = min(str(f) for f in records["f"].unique())
            records = records[records["f"].astype(str) == branch]
        for kind, collected in parts.items():
            part = records[records["param_gnBoundaryTypes"].astype(str) == kind] if usable else None
            collected.append(_node_matrix(_node_hourly(part)) if usable and len(part) else None)
        return None

    gx.read_gdx_parameter_over_files(
        ordered, "ts_node", reduce_one, container=container,
        progress_desc="ts_node_hydro_storage_limits" if sys.stdout.isatty() else None,
    )
    upward = _stack_windows(parts["upwardLimit"], absent_node=np.nan)
    downward = _stack_windows(parts["downwardLimit"], absent_node=np.nan)
    # A node bounded only from above has a floor of zero wherever it has a ceiling.
    for node, top in upward.items():
        floor = downward.get(node)
        if floor is None or floor.shape != top.shape:
            filled = np.where(np.isnan(top), np.nan, 0.0).astype("float32")
            if floor is not None:
                width = min(floor.shape[1], top.shape[1])
                filled[:, :width] = np.where(np.isnan(floor[:, :width]),
                                             filled[:, :width], floor[:, :width])
            downward[node] = filled
        else:
            downward[node] = np.where(np.isnan(floor) & ~np.isnan(top), 0.0, floor).astype("float32")
    return upward, {n: a for n, a in downward.items() if n in upward}


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
    window = 0
    node_parts: List[Optional[Tuple[List[str], np.ndarray]]] = []
    grid_of: Dict[str, str] = {}
    demand_nodes: Dict[str, set] = {}
    last_label = f"t{HOURS_PER_YEAR:06d}"

    def sweep(key, family, parameter, sign=1.0, weights=None, flow=None, keep_nodes=False):
        """One family, every climate year, reduced per file.

        A window longer than a year is cut to its first 365 days for everything
        but the hydro checks: the build starts a window in every climate year,
        so the rest of a long window is the next windows' start, and keeping it
        would count those hours again. ``keep_nodes`` keeps the per-node hours,
        whole, for the one family the hydro checks need them from -- without a
        second read of it.
        """
        nonlocal hours, window
        available = year_files(folder, family)
        if not available:
            return
        # A year with no file of this family still takes a slot, so a family
        # that starts late cannot shift every later year out of step with the
        # others. A path that does not exist yields an empty frame.
        ordered = [available.get(y, folder / f"{family}_{y}.gdx") for y in years]
        frames: List[Optional[np.ndarray]] = []

        def reduce_one(gdx_file, records):
            nonlocal window
            wide = _node_hourly(records)
            array = _collapse_to_areas(wide, areas, weight_by_node=weights, sign=sign)
            if array is not None:
                window = max(window, array.shape[0])
                array = array[:HOURS_PER_YEAR]
            if key.startswith("demand_") and wide is not None:
                live = wide.columns[wide.abs().sum(axis=0).to_numpy() > 0]
                demand_nodes.setdefault(key[len("demand_"):], set()).update(map(str, live))
            frames.append(array)
            if keep_nodes:
                node_parts.append(_node_matrix(wide))
                if wide is not None and "grid" in records.columns:
                    pairs = records[["grid", "node"]].drop_duplicates()
                    grid_of.update(zip(pairs["node"].astype(str), pairs["grid"].astype(str)))
            if flow is not None and records is not None and not records.empty:
                first_year = records[records["t"].astype(str) <= last_label]
                means = (
                    first_year.assign(node=first_year["node"].astype(str))
                    .groupby("node", observed=True)["value"].mean()
                )
                for node, mean_cf in means.items():
                    cf_seen.setdefault((flow, str(node)), []).append(float(mean_cf))
            if array is None:
                return None
            year = int(Path(gdx_file).stem.rsplit("_", 1)[-1])
            # Annualised on the hours actually summed, which is a whole year
            # unless a series ends early.
            return pd.DataFrame({
                "key": key, "area": areas, "year": year,
                "TWh": array.sum(axis=0) * MWH_TO_TWH * annualise(array.shape[0]),
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
        if window and window < HOURS_PER_YEAR:
            # Read once, before anything else is: the length is the same in
            # every family, and the rest of the read would be thrown away.
            return Timeseries(
                years=years, window_hours=window,
                skipped=(f"its climate windows are {window / 24:,.4g} days, shorter than a "
                         f"year, so they cover only part of the seasons and no annual "
                         f"figure can be made from them"),
            )
    sweep("inflow_hydro", "ts_influx_hydro", "ts_influx", keep_nodes=True)
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
    storage_limit, storage_limit_from = _read_storage_limit(container, folder)

    hydro = None
    inflow = _stack_windows(node_parts, absent_node=0.0)
    if inflow:
        upward, downward = _read_storage_limit_series(gx, container, folder, years)
        window = max(array.shape[1] for array in [*inflow.values(), *upward.values()])
        hydro = HydroSeries(
            years=list(years), hours=window, grid_of=grid_of,
            inflow=_pad_hours(inflow, window),
            upward=_pad_hours(upward, window),
            downward=_pad_hours(downward, window),
        )

    return Timeseries(
        years=years,
        areas=areas,
        hours_per_year=hours,
        hourly=hourly,
        annual=annual,
        node_cf_mean=pd.DataFrame(cf_rows),
        storage_limit=storage_limit,
        storage_limit_from=storage_limit_from,
        window_hours=window,
        hydro=hydro,
        demand_nodes=demand_nodes,
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
        for area, value in zip(areas, block.sum(axis=0) * MWH_TO_TWH * annualise(hours)):
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


def _read_storage_limit(container, folder: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    """How much energy each seasonally-bounded node can actually move.

    Three measures per (grid, node), because a reservoir whose bounds are a
    season-long shape has no single size:

      nameplate_MWh        max(upwardLimit)        -- the physical reservoir
      usable_mean_MWh      mean(up - down)         -- the room on an average hour
      usable_envelope_MWh  max(up) - min(down)     -- the widest swing the year allows

    Reading only the ceiling, and only its annual maximum, made the twelve
    seasonal nodes hold 149 TWh when the mean usable volume is 70. Both are
    true statements about the same data and only one of them is a storage size.

    One file and one forecast branch. The aggregates above agree across climate
    years and across forecast branches to within 0.01%, which is the checkable
    version of the claim this docstring used to make; the hourly values do not,
    differing by up to 113 GWh, so the branches are never blended. The forecast
    file is tried first because it is one file rather than 35 -- it carries
    the forecast branches and no f00, while a per-year file carries f00 alone,
    so the lowest label present is the branch either way.
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
        table, branch = _storage_limit_measures(records)
        if table.empty:
            continue
        return table, f"{path.name}, branch {branch}" if branch else path.name
    return pd.DataFrame(), None


def _storage_limit_measures(records: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """The three measures, from one ts_node frame, on one forecast branch.

    Split out from the read so the arithmetic can be tested without a GDX file.
    Every dimension column is cast to str first: gams.transfer returns
    categoricals, and a groupby over those builds the full cartesian product of
    grid and node -- which manufactures a psOpen row for every reservoir node
    and fills it with NaN.
    """
    work = _without_initial_step(records).copy()
    for column in ("grid", "node", "param_gnBoundaryTypes", "f"):
        if column in work.columns:
            work[column] = work[column].astype(str)

    branch = None
    if "f" in work.columns and len(work):
        branch = min(work["f"].unique())
        work = work[work["f"] == branch]

    work = work[work["param_gnBoundaryTypes"].isin(["upwardLimit", "downwardLimit"])]
    if work.empty or "upwardLimit" not in set(work["param_gnBoundaryTypes"]):
        return pd.DataFrame(), branch

    wide = work.pivot_table(
        index=["grid", "node", "t"], columns="param_gnBoundaryTypes",
        values="value", observed=True,
    ).reset_index()
    # A node bounded only from above has no floor, which is a zero floor.
    if "downwardLimit" not in wide.columns:
        wide["downwardLimit"] = 0.0
    wide["downwardLimit"] = wide["downwardLimit"].fillna(0.0)
    wide = wide[wide["upwardLimit"].notna()]
    if wide.empty:
        return pd.DataFrame(), branch

    wide["band"] = wide["upwardLimit"] - wide["downwardLimit"]
    grouped = wide.groupby(["grid", "node"], observed=True)
    table = pd.DataFrame({
        "nameplate_MWh": grouped["upwardLimit"].max(),
        "usable_mean_MWh": grouped["band"].mean(),
        "usable_envelope_MWh": grouped["upwardLimit"].max() - grouped["downwardLimit"].min(),
    }).reset_index()
    return table, branch


def roll_up_to_countries(timeseries: Timeseries) -> Timeseries:
    """The same series summed from bidding zones to countries.

    The GDX files are always read at zone level, because that is the finest
    grain the build writes and a country is exactly the sum of its zones. A
    country report rolls them up here rather than re-reading 35 climate years
    at the coarser grain, which also lets the maps stay per zone while the
    tables are per country.

    ``node_cf_mean``, ``storage_limit`` and ``hydro`` are keyed by node, not
    area, so they pass through untouched.
    """
    if not timeseries.areas:
        return timeseries
    countries = sorted({country_of(area) for area in timeseries.areas})
    index = {country: position for position, country in enumerate(countries)}

    hourly: Dict[str, np.ndarray] = {}
    for key, array in timeseries.hourly.items():
        rolled = np.zeros((array.shape[0], len(countries)), dtype=array.dtype)
        for column, area in enumerate(timeseries.areas):
            rolled[:, index[country_of(area)]] += array[:, column]
        hourly[key] = rolled

    annual = timeseries.annual
    if not annual.empty:
        annual = (annual.assign(area=[country_of(a) for a in annual["area"]])
                  .groupby(["key", "area", "year"], observed=True)["TWh"]
                  .sum().reset_index())

    return replace(timeseries, areas=countries, hourly=hourly, annual=annual)


def _hourly_balance(timeseries: Timeseries, subtract: Sequence[str]) -> Dict[str, np.ndarray]:
    """Electricity demand less the named hourly series, per area."""
    demand = timeseries.hourly.get("demand_elec")
    if demand is None:
        return {}
    net = demand
    for key in subtract:
        part = timeseries.hourly.get(key)
        if part is not None:
            net = net - part
    return {area: net[:, i] for i, area in enumerate(timeseries.areas)}


def netload_by_area(timeseries: Timeseries) -> Dict[str, np.ndarray]:
    """Demand minus wind and solar, per area, every climate year pooled.

    Hydro is not subtracted: reservoir hydro is dispatchable, so subtracting it
    would answer a different question than what firm capacity has to cover.
    """
    return _hourly_balance(timeseries, ("vre",))


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
# Hydro
# ============================================================================

#: Row order for the hydro table, and nothing more: which grids are hydro is
#: read from the data, and a grid not named here follows these rather than
#: vanishing.
HYDRO_TYPE_ORDER = ("reservoir", "ror", "psOpen", "psClosed")

HOURS_PER_WEEK = 168

#: A storage node is close to its floor when it holds less than this many hours of its
#: own minimum-generation draw above it. A starting value to iterate on, with an
#: upper limit: the smallest run-of-river pond in the shipped scenarios holds
#: 18.8 hours of its draw in all, so a margin above that flags it in every
#: window whatever the water does.
MIN_GEN_MARGIN_HOURS = 12.0

#: Hours inside that margin, in any one window, before a storage node is worth watching.
MIN_GEN_WATCH_HOURS = 2


def hydro_grids(workbook: Workbook, inflow_grids: set) -> set:
    """The grids the hydro section reports: natural inflow, and pumped storage without it.

    Inflow is read from the data. Closed-loop pumped storage has none by
    definition, so it is found the way the storage tables find it, by the name
    of its grid.
    """
    pumped = {g for g in state_grids(workbook) if storage_label(g) == "pumped storage"}
    return set(inflow_grids) | pumped


def hydro_generation_by_node(workbook: Workbook, grids: set) -> pd.DataFrame:
    """Per hydro storage node: the minimum and the maximum its turbines generate.

    A turbine draws on the storage on its input row and generates on its output
    row. ``min_gen_MW`` is the ``minGen`` on those output rows and ``max_gen_MW``
    their capacity times the unit's ``availability``, both in MW of what the
    turbines make. ``timeseries_availability`` marks a storage node with a turbine that
    takes its availability from a timeseries, which this report does not read.
    A unit drawing on two storage nodes is counted on the first.
    """
    columns = ["node", "min_gen_MW", "max_gen_MW", "output_MW", "timeseries_availability"]
    io = workbook.p_gnu_io
    if io.empty or not {"grid", "node", "unit", "input_output"} <= set(io.columns):
        return pd.DataFrame(columns=columns)
    io = io[col_or(io, "isActive", 1.0).fillna(1.0) == 1].copy()
    for c in ("grid", "node", "unit", "input_output"):
        io[c] = io[c].astype(str)
    draws = io[(io["input_output"] == "input") & io["grid"].isin(set(grids))]
    if draws.empty:
        return pd.DataFrame(columns=columns)
    storage_of = draws.groupby("unit")["node"].agg(lambda s: sorted(set(s))[0]).to_dict()
    availability, _ = _unit_numbers(workbook)
    unit = workbook.p_unit
    from_series = set()
    if not unit.empty and "useTimeseriesAvailability" in unit.columns:
        flag = pd.to_numeric(unit["useTimeseriesAvailability"], errors="coerce").fillna(0) == 1
        from_series = set(unit.loc[flag, "unit"].astype(str))

    outputs = io[(io["input_output"] == "output") & io["unit"].isin(storage_of)
                 & ~io["grid"].isin(set(grids))]
    if outputs.empty:
        return pd.DataFrame(columns=columns)
    capacity = pd.to_numeric(outputs["capacity"], errors="coerce").fillna(0.0)
    min_gen = pd.to_numeric(col_or(outputs, "minGen", 0.0), errors="coerce").fillna(0.0)
    units = outputs["unit"]
    rows = pd.DataFrame({
        "node": [storage_of[u] for u in units],
        "min_gen_MW": min_gen.to_numpy(),
        "max_gen_MW": capacity.to_numpy() * np.array([availability.get(u, 0.0) for u in units]),
        "output_MW": capacity.to_numpy(),
        "timeseries_availability": [u in from_series for u in units],
    })
    return (rows.groupby("node", observed=True)
            .agg(min_gen_MW=("min_gen_MW", "sum"), max_gen_MW=("max_gen_MW", "sum"),
                 output_MW=("output_MW", "sum"),
                 timeseries_availability=("timeseries_availability", "any"))
            .reset_index())


def _sum_series(arrays: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Node series added hour by hour; an hour is NaN only where every node's is."""
    if not arrays:
        return None
    stack = np.stack([np.asarray(a, dtype="float64") for a in arrays])
    total = np.nansum(stack, axis=0)
    total[np.all(np.isnan(stack), axis=0)] = np.nan
    return total


def hydro_characteristics(
    workbook: Workbook,
    inventory: Optional["StorageInventory"],
    timeseries: Timeseries,
    zones: bool,
    grids: set,
    ) -> pd.DataFrame:
    """Storage, turbine, generation limits, pump and inflow per area and hydro type.

    One row per (area, type) that has anything, so an area without hydro is
    silent. A country's inflow is its zones' series added hour by hour.
    """
    columns = ["area", "grid", "storage_MWh", "turbine_MW", "min_gen_MW", "max_gen_MW",
               "output_MW", "timeseries_availability", "pump_MW", "inflow_MWh_yr", "flh",
               "storage_weeks"]
    grids = set(grids)
    turbine = storage_side_power_by_node(workbook, grids, "input")
    pump = storage_side_power_by_node(workbook, grids, "output")
    generation = hydro_generation_by_node(workbook, grids).set_index("node")
    by_node = inventory.by_node if inventory is not None else pd.DataFrame()
    storage = (by_node[by_node["grid"].astype(str).isin(grids)]
               if len(by_node) and "usable_mean_MWh" in by_node.columns else pd.DataFrame())
    hydro = timeseries.hydro

    def per_node(frame: pd.DataFrame, column: str) -> Dict[str, float]:
        if not len(frame):
            return {}
        return dict(zip(frame["node"].astype(str), pd.to_numeric(frame[column], errors="coerce")))

    node_grid: Dict[str, str] = {}
    for frame in (turbine, pump, storage):
        if len(frame):
            node_grid.update(zip(frame["node"].astype(str), frame["grid"].astype(str)))
    if hydro is not None:
        node_grid.update({n: g for n, g in hydro.grid_of.items() if g in grids})
    turbine_mw, pump_mw = per_node(turbine, "power_MW"), per_node(pump, "power_MW")
    storage_mwh = per_node(storage, "usable_mean_MWh")

    members: Dict[Tuple[str, str], List[str]] = {}
    for node, grid in node_grid.items():
        members.setdefault((area_of(node, zones), grid), []).append(node)

    rows = []
    for (area, grid), nodes in members.items():
        sizes = [storage_mwh[n] for n in nodes if np.isfinite(storage_mwh.get(n, np.nan))]
        stored = float(sum(sizes)) if sizes else np.nan
        turbines = float(sum(turbine_mw.get(n, 0.0) for n in nodes))
        pumps = float(sum(pump_mw.get(n, 0.0) for n in nodes))
        made = generation.loc[generation.index.intersection(nodes)]
        inflow_yr = flh = weeks = np.nan
        if hydro is not None:
            series = _sum_series([hydro.inflow[n] for n in nodes if n in hydro.inflow])
            finite = series is not None and bool(np.isfinite(series).any())
            mean_hour = float(np.nanmean(series)) if finite else 0.0
            inflow_yr = mean_hour * HOURS_PER_YEAR
            if mean_hour > 0:
                flh = inflow_yr / turbines if turbines > 0 else np.nan
                weeks = stored / (mean_hour * HOURS_PER_WEEK) if np.isfinite(stored) else np.nan
            else:
                flh = 0.0
        if not (turbines or pumps or (np.isfinite(stored) and stored)
                or (np.isfinite(inflow_yr) and inflow_yr)):
            continue
        rows.append(dict(area=area, grid=grid, storage_MWh=stored, turbine_MW=turbines,
                         min_gen_MW=float(made["min_gen_MW"].sum()),
                         max_gen_MW=float(made["max_gen_MW"].sum()),
                         output_MW=float(made["output_MW"].sum()),
                         timeseries_availability=bool(made["timeseries_availability"].any()),
                         pump_MW=pumps, inflow_MWh_yr=inflow_yr, flh=flh, storage_weeks=weeks))
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows, columns=columns)
    order = {g: i for i, g in enumerate(HYDRO_TYPE_ORDER)}
    table["_order"] = [order.get(g, len(order)) for g in table["grid"]]
    return (table.sort_values(["area", "_order", "grid"]).drop(columns="_order")
            .reset_index(drop=True))


def run_storage(inflow, upward, downward, outflow, margin=None) -> Dict[str, np.ndarray]:
    """A storage node run hour by hour, each climate window on its own.

    ``inflow``, ``upward`` and ``downward`` are (windows, hours, nodes) in MWh
    per hour and MWh, and a constant bound may be anything that broadcasts to
    that; ``outflow`` and ``margin`` are per node. Each window is its own run
    because each climate window is its own run.

    Every hour the storage takes its inflow and gives its outflow. What rises
    above the ceiling is counted and spilled; what falls below the floor is
    counted and made good, so one short hour is not charged again in every hour
    after it. Both sides are kept because the two hydro checks ask opposite
    questions of the same arithmetic: with the minimum as the outflow, a
    shortfall is water the minimum cannot have; with everything turbines and
    spill can pass as the outflow, an overflow is water that cannot leave.

    The start is the middle of the first hour's range. Backbone starts
    elsewhere: a psOpen or reservoir storage at its ``relative`` share of the band
    on the starting date, any other storage at its ``reference``.

    An hour with a NaN in any input is not run, which keeps a missing climate
    year, or the hours past the end of a shorter window, out of the counts.

    Returns (windows, nodes) arrays: ``hours_run``, ``below_hours``,
    ``below_MWh``, ``above_hours``, ``above_MWh``, ``near_floor_hours`` (hours
    at most ``margin`` above the floor and not below it), and the closest the
    storage came to each bound, ``floor_gap`` and ``ceiling_gap`` -- negative when
    it crossed, NaN when nothing ran.
    """
    inflow = np.asarray(inflow, dtype="float32")
    windows, hours, nodes = inflow.shape
    upward = np.broadcast_to(np.asarray(upward, dtype="float32"), inflow.shape)
    downward = np.broadcast_to(np.asarray(downward, dtype="float32"), inflow.shape)
    pairs = (windows, nodes)
    outflow = np.broadcast_to(np.asarray(outflow, dtype="float64"), pairs)
    margin = np.broadcast_to(np.asarray(0.0 if margin is None else margin, dtype="float64"), pairs)

    top0 = upward[:, 0, :].astype("float64")
    bottom0 = downward[:, 0, :].astype("float64")
    with np.errstate(invalid="ignore"):
        level = np.where(np.isfinite(top0) & np.isfinite(bottom0), (top0 + bottom0) / 2.0,
                         np.where(np.isfinite(bottom0), bottom0,
                                  np.where(np.isfinite(top0), top0, 0.0)))

    counts = {k: np.zeros(pairs, dtype="int32")
              for k in ("hours_run", "below_hours", "above_hours", "near_floor_hours")}
    energy = {k: np.zeros(pairs) for k in ("below_MWh", "above_MWh")}
    floor_gap = np.full(pairs, np.inf)
    ceiling_gap = np.full(pairs, np.inf)

    with np.errstate(invalid="ignore"):
        for t in range(hours):
            water = inflow[:, t, :].astype("float64")
            top = upward[:, t, :].astype("float64")
            bottom = downward[:, t, :].astype("float64")
            valid = ~(np.isnan(water) | np.isnan(top) | np.isnan(bottom))
            if not valid.any():
                continue
            moved = level + np.where(valid, water - outflow, 0.0)
            above = np.where(valid, moved - top, -np.inf)
            below = np.where(valid, bottom - moved, -np.inf)
            over, short = above > 0, below > 0
            counts["hours_run"] += valid
            counts["above_hours"] += over
            counts["below_hours"] += short
            energy["above_MWh"] += np.where(over, above, 0.0)
            energy["below_MWh"] += np.where(short, below, 0.0)
            # Headroom over the floor after the spill, which is what is left to use.
            floor = np.where(valid, np.minimum(moved, top) - bottom, np.inf)
            counts["near_floor_hours"] += valid & (floor >= 0) & (floor < margin)
            np.minimum(floor_gap, floor, out=floor_gap)
            np.minimum(ceiling_gap, np.where(valid, top - moved, np.inf), out=ceiling_gap)
            level = np.where(valid, np.minimum(np.maximum(moved, bottom), top), level)

    ran = counts["hours_run"] > 0
    return {**counts, **energy,
            "floor_gap": np.where(ran, floor_gap, np.nan),
            "ceiling_gap": np.where(ran, ceiling_gap, np.nan)}


def _unit_numbers(workbook: Workbook) -> Tuple[Dict[str, float], Dict[str, float]]:
    """``({unit: availability}, {unit: best efficiency})`` from ``p_unit``.

    An availability that is not written is zero, the way Backbone reads it:
    the unit is deactivated. The efficiency is the largest ``eff*`` column,
    which is the slope Backbone converts output to input with when no unit has
    an efficiency curve.
    """
    unit = workbook.p_unit
    if unit.empty or "unit" not in unit.columns:
        return {}, {}
    names = unit["unit"].astype(str)
    availability = pd.to_numeric(col_or(unit, "availability", 0.0), errors="coerce").fillna(0.0)
    eff_columns = [c for c in unit.columns if str(c).startswith("eff") and str(c)[3:].isdigit()]
    if eff_columns:
        effs = unit[eff_columns].apply(pd.to_numeric, errors="coerce")
        best = effs.where(effs > 0).max(axis=1)
    else:
        best = pd.Series(np.nan, index=unit.index)
    return dict(zip(names, availability.astype(float))), dict(zip(names, best.astype(float)))


def _boundary_sources(workbook: Workbook) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """``{node: {kind: ("constant", value) | ("timeseries", nan)}}`` from the boundary sheet."""
    boundary = workbook.boundary
    out: Dict[str, Dict[str, Tuple[str, float]]] = {}
    if boundary.empty or not {"node", "param_gnBoundaryTypes"} <= set(boundary.columns):
        return out
    use_ts = col_or(boundary, "useTimeseries", 0.0).fillna(0.0) == 1
    constants = pd.to_numeric(col_or(boundary, "constant", np.nan), errors="coerce")
    for node, kind, ts, value in zip(boundary["node"].astype(str),
                                     boundary["param_gnBoundaryTypes"].astype(str),
                                     use_ts, constants):
        if ts:
            out.setdefault(node, {})[kind] = ("timeseries", np.nan)
        elif pd.notna(value):
            out.setdefault(node, {})[kind] = ("constant", float(value))
    return out


def hydro_minimums(workbook: Workbook, grids: set) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """``(minimums the check can simulate, [(unit, why not)])`` for hydro ``minGen``.

    A ``minGen`` row in ``p_gnu_io`` is about hydro when its unit draws on a
    hydro storage node, or when the row itself sits on a hydro grid. A minimum on the
    output of a unit drawing on one storage node is simulated. Anything else is named
    with the reason, because a minimum the check misread would be a verdict
    about data it never understood.

    ``draw_MWh`` is the least water that meets the minimum: ``minGen`` through
    the unit's efficiency. ``deliverable_MW`` is the most the row can give,
    capacity times availability. ``minGen`` itself is not scaled by
    availability: Backbone holds the whole value in every step where
    availability is above zero.
    """
    columns = ["unit", "node", "grid", "minimum_MW", "draw_MWh", "deliverable_MW"]
    io = workbook.p_gnu_io
    if io.empty or "minGen" not in io.columns:
        return pd.DataFrame(columns=columns), []

    io = io[col_or(io, "isActive", 1.0).fillna(1.0) == 1].copy()
    for c in ("grid", "node", "unit", "input_output"):
        io[c] = io[c].astype(str)
    io["capacity"] = pd.to_numeric(io["capacity"], errors="coerce").fillna(0.0)
    io["minGen"] = pd.to_numeric(io["minGen"], errors="coerce").fillna(0.0)
    coeff = pd.to_numeric(col_or(io, "conversionCoeff", 1.0), errors="coerce").fillna(1.0)
    io["conversionCoeff"] = coeff.replace(0.0, 1.0)
    draws = io[(io["input_output"] == "input") & io["grid"].isin(grids)]
    draws_from = draws.groupby("unit")["node"].agg(lambda s: sorted(set(s))).to_dict()
    availability, efficiency = _unit_numbers(workbook)

    rows, skipped = [], []
    for row in io[io["minGen"] > 0].itertuples(index=False):
        unit = row.unit
        if unit not in draws_from and row.grid not in grids:
            continue
        if row.grid in grids:
            reason = "on its storage side"
        elif row.input_output != "output":
            reason = "on an input"
        elif len(draws_from[unit]) > 1:
            reason = (f"draws on {len(draws_from[unit])} different storage nodes "
                      f"({summarise(draws_from[unit])})")
        elif (io.loc[io["unit"] == unit, "conversionCoeff"] != 1.0).any():
            reason = "a conversionCoeff other than 1"
        elif not np.isfinite(efficiency.get(unit, np.nan)):
            reason = "no efficiency in `p_unit`"
        else:
            reason = None
        if reason:
            skipped.append((unit, reason))
            continue
        node = draws_from[unit][0]
        rows.append(dict(unit=unit, node=node,
                         grid=str(draws.loc[draws["node"] == node, "grid"].iloc[0]),
                         minimum_MW=row.minGen, draw_MWh=row.minGen / efficiency[unit],
                         deliverable_MW=row.capacity * availability.get(unit, 0.0)))
    return pd.DataFrame(rows, columns=columns), skipped


def _closest(gap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per node, the smallest gap over all windows and which window it was in.

    A node that ran in no window gets NaN, and window 0 as a placeholder.
    """
    filled = np.where(np.isnan(gap), np.inf, gap)
    window = filled.argmin(axis=0)
    smallest = filled.min(axis=0)
    return np.where(np.isinf(smallest) & (smallest > 0), np.nan, smallest), window


@dataclass
class HydroChecks:
    """The two hydro checks, per bidding-zone storage node."""
    minimum: pd.DataFrame = field(default_factory=pd.DataFrame)
    overflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    not_simulated: List[Tuple[str, str]] = field(default_factory=list)
    minimums: int = 0
    windows: int = 0
    window_hours: int = 0
    skipped: Optional[str] = None


def hydro_checks(workbook: Workbook, timeseries: Timeseries, grids: set) -> HydroChecks:
    """Minimum generation against inflow, and overflow of a full storage.

    Always per bidding-zone storage node, whatever level the report is written at: a
    minimum binds one storage node, and two added together can hold water
    neither of them has.
    """
    grids = set(grids)
    minimums, not_simulated = hydro_minimums(workbook, grids)
    result = HydroChecks(not_simulated=list(not_simulated),
                         minimums=len(minimums) + len(not_simulated))
    availability, _ = _unit_numbers(workbook)

    minimum = minimums.copy()
    if len(minimum):
        minimum["beyond_turbines"] = minimum["minimum_MW"] > minimum["deliverable_MW"] + 1e-6
        minimum = (minimum.groupby("node", sort=True)
                   .agg(units=("unit", lambda s: ", ".join(s)), grid=("grid", "first"),
                        minimum_MW=("minimum_MW", "sum"), draw_MWh=("draw_MWh", "sum"),
                        deliverable_MW=("deliverable_MW", "sum"),
                        beyond_turbines=("beyond_turbines", "any"))
                   .reset_index())

    hydro = timeseries.hydro
    if hydro is None or not hydro.inflow:
        result.skipped = timeseries.skipped or "no `ts_influx_hydro_<year>.gdx` file was read"
        result.minimum = minimum
        return result

    windows, hours = len(hydro.years), hydro.hours
    result.windows, result.window_hours = windows, hours
    shape = (windows, hours)
    bounds = _boundary_sources(workbook)
    no_file = np.isnan(next(iter(hydro.inflow.values())))

    def inflow_of(node: str) -> np.ndarray:
        series = hydro.inflow.get(node)
        return series if series is not None else np.where(no_file, np.nan, 0.0).astype("float32")

    def bound_of(node: str, kind: str) -> Optional[np.ndarray]:
        source = bounds.get(node, {}).get(kind)
        if source is None:
            return np.full(shape, np.inf if kind == "upwardLimit" else -np.inf, dtype="float32")
        how, value = source
        if how == "timeseries":
            return (hydro.upward if kind == "upwardLimit" else hydro.downward).get(node)
        return np.full(shape, value, dtype="float32")

    io = workbook.p_gnu_io
    io = io[col_or(io, "isActive", 1.0).fillna(1.0) == 1] if not io.empty else io

    def side_water(node: str, side: str) -> float:
        """Capacity times availability on the storage's own row, for one side."""
        if io.empty:
            return 0.0
        rows = io[(io["node"].astype(str) == node) & (io["input_output"].astype(str) == side)]
        capacity = pd.to_numeric(rows["capacity"], errors="coerce").fillna(0.0)
        return float(sum(c * availability.get(str(u), 0.0) for u, c in zip(rows["unit"], capacity)))

    def series_for(nodes: List[str]):
        usable, missing = [], []
        for node in nodes:
            top, bottom = bound_of(node, "upwardLimit"), bound_of(node, "downwardLimit")
            if top is None or bottom is None:
                missing.append(node)
            else:
                usable.append((node, top, bottom))
        return usable, missing

    years = np.asarray(hydro.years)

    # ---- minimum generation ------------------------------------------------
    if len(minimum):
        usable, missing = series_for(list(minimum["node"]))
        for node in missing:
            not_simulated.append((node, "claims a timeseries bound and no series was read"))
        result.not_simulated = list(not_simulated)
        minimum = minimum[minimum["node"].isin([n for n, _, _ in usable])].reset_index(drop=True)
    if len(minimum):
        names = list(minimum["node"])
        draw = minimum["draw_MWh"].to_numpy(dtype="float64")
        water = np.stack([inflow_of(n) for n in names], axis=2)
        top = np.stack([t for _, t, _ in usable], axis=2)
        bottom = np.stack([b for _, _, b in usable], axis=2)
        run = run_storage(water, top, bottom, draw, margin=draw * MIN_GEN_MARGIN_HOURS)
        pumps = np.array([side_water(n, "output") for n in names])
        with_pumps = run["below_hours"].copy()
        pumped = np.flatnonzero((run["below_hours"].sum(axis=0) > 0) & (pumps > 0))
        if len(pumped):
            again = run_storage(water[:, :, pumped] + pumps[pumped].astype("float32"),
                              top[:, :, pumped], bottom[:, :, pumped], draw[pumped])
            with_pumps[:, pumped] = again["below_hours"]
        closest, closest_window = _closest(run["floor_gap"])
        minimum["pump_MW"] = pumps
        minimum["windows_run"] = (run["hours_run"] > 0).sum(axis=0)
        minimum["windows_short"] = (run["below_hours"] > 0).sum(axis=0)
        minimum["short_hours"] = run["below_hours"].sum(axis=0)
        minimum["short_GWh"] = run["below_MWh"].sum(axis=0) / 1e3
        minimum["short_with_pumps"] = with_pumps.sum(axis=0)
        minimum["closest_MWh"] = closest
        minimum["closest_hours"] = closest / draw
        minimum["closest_year"] = years[closest_window]
        minimum["near_hours_worst"] = run["near_floor_hours"].max(axis=0)
        states = []
        for row in minimum.itertuples():
            if row.beyond_turbines or row.short_with_pumps > 0:
                states.append(CHECK_ACT)
            elif row.short_hours > 0 or row.near_hours_worst >= MIN_GEN_WATCH_HOURS:
                states.append(CHECK_WATCH)
            else:
                states.append(CHECK_OK)
        minimum["state"] = states
    result.minimum = minimum

    # ---- overflow ------------------------------------------------------------
    flowing = [n for n in sorted(hydro.inflow)
               if hydro.grid_of.get(n) in grids and np.nanmax(hydro.inflow[n], initial=0.0) > 0]
    usable, _ = series_for(flowing)
    usable = [(n, t, b) for n, t, b in usable if np.isfinite(t).any()]
    if usable:
        names = [n for n, _, _ in usable]

        def spill_of(node: str) -> float:
            how, value = bounds.get(node, {}).get("maxSpill", ("constant", 0.0))
            return max(value, 0.0) if how == "constant" else 0.0

        spill = np.array([spill_of(n) for n in names])
        turbines = np.array([side_water(n, "input") for n in names])
        water = np.stack([inflow_of(n) for n in names], axis=2)
        run = run_storage(water, np.stack([t for _, t, _ in usable], axis=2),
                        np.stack([b for _, _, b in usable], axis=2), turbines + spill)
        spare, tightest = _closest(run["ceiling_gap"])
        overflow = pd.DataFrame({
            "node": names,
            "grid": [hydro.grid_of.get(n) for n in names],
            "turbine_MW": turbines,
            "spill_MW": spill,
            "peak_inflow_MW": [float(np.nanmax(hydro.inflow[n], initial=0.0)) for n in names],
            "windows_run": (run["hours_run"] > 0).sum(axis=0),
            "windows_over": (run["above_hours"] > 0).sum(axis=0),
            "over_hours": run["above_hours"].sum(axis=0),
            "over_GWh": run["above_MWh"].sum(axis=0) / 1e3,
            "spare_MW": spare,
            "tightest_year": years[tightest],
        })
        overflow["state"] = np.where(overflow["over_hours"] > 0, CHECK_ACT, CHECK_OK)
        # Tightness relative to what the storage can release: in MW alone the
        # smallest storage would always look tightest.
        release = overflow["turbine_MW"] + overflow["spill_MW"]
        overflow["spare_share"] = np.where(release > 0, overflow["spare_MW"] / release, np.nan)
        result.overflow = overflow.sort_values("spare_share").reset_index(drop=True)
    return result


# ============================================================================
# Figures
# ============================================================================

def carrier_demand(workbook: Workbook, timeseries: Timeseries, carrier: str,
                   zones: bool) -> Tuple[pd.DataFrame, bool]:
    """``(per-area annual demand, whether it is a constant)`` for one carrier.

    The two sources are never added. ``p_gn``'s constant is *overridden* by a
    timeseries wherever one exists -- Backbone gates it on
    ``not gn_influxTs(grid, node)`` in six places -- so summing them would
    double-count every node carrying both, in every table at once.

    The constant is a workbook fact, so it survives a run that could not read a
    single GDX file. That is why the two meet here rather than inside
    ``read_timeseries``.
    """
    hourly = annual_range(timeseries, f"demand_{carrier}")
    if not hourly.empty:
        return hourly, False
    return constant_demand(workbook, carrier, zones), True


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
    path = path or ZONE_ASSET
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
    """Geometry per reported area.

    Each level has its own asset, and the key in it -- ``SE01`` or ``SE`` -- is
    already the area this reports on, so the same lookup serves both: a
    two-letter country code is its own ``country_of``.
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
    for carrier in states.columns:
        frame, _ = carrier_demand(workbook, timeseries, carrier, zones)
        if not frame.empty:
            demanded[carrier] = set(frame["area"].astype(str))

    for carrier in states.columns:
        # Unknown only where the answer was in a file that could not be read. A
        # carrier whose demand is a constant in the workbook is known either way,
        # and one with no demand anywhere is known to have none.
        unreadable = carrier in DEMAND_FAMILY and not timeseries.available
        nodes = gn.loc[gn["grid"].astype(str) == carrier, "node"].astype(str)
        for area in {area_of(n, zones) for n in nodes}:
            if area not in states.index:
                continue
            if area in demanded.get(carrier, set()):
                states.loc[area, carrier] = PRESENCE_DEMAND
            elif unreadable:
                states.loc[area, carrier] = PRESENCE_UNKNOWN
            else:
                states.loc[area, carrier] = PRESENCE_NODE
    return states


#: Strongest state first. A country is what its most-developed zone is: one
#: Finnish zone with district-heat demand makes Finland a district-heat country,
#: and the zones that do not have it are the --zones view's business.
PRESENCE_ORDER = [PRESENCE_DEMAND, PRESENCE_NODE, PRESENCE_UNKNOWN, PRESENCE_ABSENT]


def presence_by_country(zone_presence: pd.DataFrame) -> pd.DataFrame:
    """Zone-level presence rolled up, so both counts come from one frame.

    Counting countries from a separately-built country-level frame would be the
    same answer twice over, and the two could drift.
    """
    if zone_presence.empty:
        return zone_presence
    rows = {}
    for country, block in zone_presence.groupby(
            [country_of(z) for z in zone_presence.index], observed=True):
        rows[country] = {
            carrier: next(state for state in PRESENCE_ORDER
                          if (block[carrier] == state).any())
            for carrier in zone_presence.columns
        }
    return pd.DataFrame.from_dict(rows, orient="index")[list(zone_presence.columns)]


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


def _draw_context(ax, shapes: ZoneShapes, drawn: Sequence[str], neighbours: bool) -> None:
    """The grey ring behind everything: unmodelled countries, and zones this run skips."""
    if not neighbours:
        return
    context = context_polygons(shapes, list(drawn), True)
    patches = [p for p in (_polygon_path(poly) for poly in context) if p]
    ax.add_collection(PatchCollection(
        [PathPatch(p) for p in patches],
        facecolor=NEIGHBOUR_FILL, edgecolor=NEIGHBOUR_EDGE, linewidth=0.4, zorder=1))


def _draw_country_outlines(ax, countries: ZoneShapes, drawn: Sequence[str]) -> None:
    """Country borders over the zone fills, so both levels read at once.

    The maps are always drawn per bidding zone, which on a country report would
    otherwise leave the reader working out which zones belong together. A
    country's zones cover exactly its country outline, so this sits on the
    seams rather than near them.
    """
    if not countries.available:
        return
    wanted = sorted({country_of(area) for area in drawn})
    geometry = shapes_for_areas(countries, wanted, False)
    paths = [p for polygons in geometry.values()
             for p in (_polygon_path(poly) for poly in polygons) if p]
    if not paths:
        return
    ax.add_collection(PatchCollection(
        [PathPatch(p) for p in paths],
        facecolor="none", edgecolor=COUNTRY_EDGE, linewidth=COUNTRY_EDGE_WIDTH, zorder=3))


def _label_geometry(shapes: ZoneShapes, countries: ZoneShapes,
                    areas: Sequence[str], zones: bool) -> Dict[str, List]:
    """Where to write the names: one per zone with --zones, one per country without."""
    if zones:
        return shapes_for_areas(shapes, list(areas), True)
    return shapes_for_areas(countries, sorted({country_of(a) for a in areas}), False)


def _draw_area_labels(ax, geometry: Dict[str, List]) -> None:
    """Name each area at its own centroid, in white-stroked type over the fill."""
    for area, polygons in geometry.items():
        centre = _polygons_centroid(polygons)
        if centre is None:
            continue
        ax.text(centre[0], centre[1], area, fontsize=7.5, ha="center", va="center",
                zorder=5, color="#1a1a1f",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="#ffffff")])


def figure_carrier_map(
    out_dir: Path,
    shapes: ZoneShapes,
    countries: ZoneShapes,
    presence: pd.DataFrame,
    zones: bool,
    neighbours: bool = True,
    ) -> str:
    """Which carriers each area models and something actually demands.

    Always drawn per bidding zone, whatever level the tables use, with country
    borders over the top -- the zones are the thing Backbone solves and the
    borders are how a reader finds them.

    The fill counts a carrier where the area models it at all. It used to count
    only what something demanded, which sounds stricter and drew a worse map:
    hydrogen has nodes in 22 bidding zones of the TYNDP scenarios and demand in
    none, so the carrier the scenario was built to study tinted nothing, and the
    map came out identical to a scenario with no hydrogen in it. Which carriers
    have no sink is a fact about demand, and it is listed as one in the chapter
    around this figure.
    """
    path = out_dir / "fig_carrier_map.png"
    if not shapes.available or presence.empty:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
        _empty_panel(ax, shapes.missing or "No nodes to map")
        return _finish(fig, path)

    geometry = shapes_for_areas(shapes, list(presence.index), True)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, MAP_HEIGHT_IN))

    _draw_context(ax, shapes, list(presence.index), neighbours)
    for area, polygons in geometry.items():
        modelled = [c for c in MAP_CARRIERS
                    if c in presence.columns and presence.loc[area, c] != PRESENCE_ABSENT]
        patches = [p for p in (_polygon_path(poly) for poly in polygons) if p]
        ax.add_collection(PatchCollection(
            [PathPatch(p) for p in patches],
            facecolor=_blend_for(modelled), edgecolor="#ffffff", linewidth=0.5, zorder=2))

    _draw_country_outlines(ax, countries, list(presence.index))
    _draw_area_labels(ax, _label_geometry(shapes, countries, list(presence.index), zones))
    _finish_map(ax)
    _carrier_map_legend(ax, presence)
    ax.set_title("What each bidding zone models", fontsize=10)
    return _finish(fig, path)


def _carriers_without_demand(presence: pd.DataFrame) -> List[str]:
    """Carriers modelled somewhere and demanded nowhere, by their display names.

    Read from the data rather than written down, so it cannot go stale against a
    scenario that fixes what it points at: the moment something demands the
    hydrogen, this stops saying nothing does. It used to be the map's title,
    where a caveat was wearing the clothes of a finding.
    """
    titles = {grid: title for grid, title, _ in CARRIERS}
    named = []
    # Only the carriers this map can draw. Industrial steam is off it, and a key
    # that explained a fill the reader cannot find would be worse than silence;
    # the omissions table in the chapter carries steam.
    for carrier in [c for c in MAP_CARRIERS if c in presence.columns]:
        modelled = int((presence[carrier] != PRESENCE_ABSENT).sum())
        hollow = int((presence[carrier] == PRESENCE_NODE).sum())
        if modelled and hollow == modelled:
            named.append(titles.get(carrier, carrier).lower())
    return named


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
    """What a fill means, and the one thing the fill cannot say.

    A carrier modelled everywhere and demanded nowhere looks, on this map,
    exactly like a carrier in full use. That belongs beside the key rather than
    above the map as a title, which is where it used to sit.
    """
    seen = []
    for area in presence.index:
        modelled = tuple(c for c in MAP_CARRIERS
                         if c in presence.columns and presence.loc[area, c] != PRESENCE_ABSENT)
        if modelled and modelled not in seen:
            seen.append(modelled)
    titles = {grid: title for grid, title, _ in CARRIERS}
    fills = [mpatches.Patch(facecolor=_blend_for(combination), edgecolor="#ffffff",
                            label=" + ".join(titles.get(c, c) for c in combination))
             for combination in sorted(seen, key=len)]
    if not fills:
        return
    hollow = _carriers_without_demand(presence)
    title = "carriers modelled"
    if hollow:
        title += "\n(" + summarise(hollow) + ": nothing demands it)"
    ax.legend(handles=fills, loc="upper left", fontsize=7,
              title=title, title_fontsize=7, framealpha=0.9)


def _carrier_order(capacity: pd.DataFrame, demand: pd.DataFrame) -> Optional[List[str]]:
    """Largest producer first, or largest demand when nothing produces."""
    if capacity is not None and not capacity.empty:
        return capacity.sum(axis=1).sort_values(ascending=False).index.tolist()
    if demand is not None and not demand.empty:
        return demand.set_index("area")["mean"].sort_values(ascending=False).index.tolist()
    return None


def figure_carrier(
    out_dir: Path,
    slug: str,
    title: str,
    capacity: pd.DataFrame,
    demand: pd.DataFrame,
    consumption: pd.DataFrame,
    zones: bool,
    ) -> Optional[str]:
    """Production capacity and annual demand for one carrier, side by side.

    Both panels keep the same area order so a reader can put an area's capacity
    against its own demand without moving between images. None when the
    carrier has neither: the section says so in a line, and an empty figure
    would only repeat it.
    """
    order = _carrier_order(capacity, demand)
    if not order:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, _barh_height(len(order))),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    if capacity is not None and not capacity.empty:
        table = capacity.reindex(order).fillna(0.0)
        table.index = [display_name(a, zones) for a in table.index]
        labels = [l for l in TECH_STYLE if l in table.columns]
        _stacked_barh(axes[0], table, labels, "installed capacity, GW")
        axes[0].legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
    else:
        _empty_panel(axes[0], f"No {title.lower()} production capacity\nin this scenario")

    if demand is not None and not demand.empty:
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

    fig.suptitle(f"{title}: capacity and demand by {level_header(zones)}", fontsize=11)
    return _finish(fig, out_dir / f"fig_{slug}.png")


def figure_carrier_mix(out_dir: Path, slug: str, title: str, capacity: pd.DataFrame,
                       zones: bool) -> Optional[str]:
    """Each area's own capacity mix, every area the same width.

    The absolute panel has a readability floor: against Germany's 854 GW,
    Latvia's 5 GW is a hairline, and the countries whose mix is most
    distinctive cannot be read at all. None when there is one technology,
    which would draw every bar at 100%.
    """
    if capacity is None or capacity.empty:
        return None
    labels = [l for l in TECH_STYLE if l in capacity.columns]
    labels += [c for c in capacity.columns if c not in labels]
    if len(labels) < 2:
        return None
    order = _carrier_order(capacity, None)
    table = capacity.reindex(order).fillna(0.0)
    totals = table.sum(axis=1).replace(0.0, np.nan)
    share = (table.div(totals, axis=0) * 100.0).fillna(0.0)
    share.index = [display_name(a, zones) for a in table.index]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, _barh_height(len(order))))
    _stacked_barh(ax, share, labels, "share of the area's own capacity, %")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    fig.suptitle(f"{title}: each {level_header(zones)}'s capacity mix", fontsize=11)
    return _finish(fig, out_dir / f"fig_{slug}_mix.png")


def figure_storage(out_dir: Path, inventory: "StorageInventory", zones: bool,
                   power: Optional[pd.DataFrame] = None) -> str:
    """Power on the left, energy on the right, because they are different facts.

    The left panel is electricity storage only, area x technology GW: inflow
    hydro discharges through its turbines, which the chapter counts as
    generation. The right panel draws the two nested groups over each other on
    a log axis, the narrow dark bar being electricity storage and the wide light
    one the total with inflow hydro, and marks each area's nameplate with a
    tick. The gap between the bar and the tick is the reservoir space the
    seasonal floor never releases.
    """
    power = inventory.power if power is None else power
    by_area = inventory.by_area
    n_rows = max(len(power) if power is not None else 0,
                 by_area.index.get_level_values(0).nunique() if len(by_area) else 0)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, _barh_height(n_rows)))

    if power is not None and not power.empty:
        table = power.loc[power.sum(axis=1).sort_values(ascending=False).index]
        table.index = [display_name(a, zones) for a in table.index]
        _stacked_barh(axes[0], table, [l for l in TECH_STYLE if l in table.columns],
                      "discharge power, GW")
        # The duration rides in the legend, so the bar keeps meaning power while
        # the reader can still get to energy without leaving the figure.
        hours = duration_by_label(inventory.ratio_facts or {})
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles,
                       [f"{l} ({_num(hours[l], 0)} h)" if l in hours else l for l in labels],
                       fontsize=7, loc="lower right")
        axes[0].set_title(f"Discharge power of electricity storage\n"
                          f"({_num(float(table.to_numpy().sum()))} GW in total)", fontsize=9)
    else:
        _empty_panel(axes[0], "No electricity storage in this scenario")

    # Cumulative per group, in GWh, on a log axis: Norway's 37 TWh of hydro and
    # Latvia's battery of under 1 GWh are four orders apart, and on a linear
    # axis every battery in the build is a hairline. The groups are nested, so
    # each is drawn from the axis floor, the narrower group over the wider.
    cumulative = (by_area["usable_mean_TWh"].unstack("group") * 1000.0
                  if len(by_area) else pd.DataFrame())
    groups = [g for g in STORAGE_GROUPS if g[0] in cumulative.columns]
    if groups and float(cumulative.to_numpy().max()) > 0:
        cumulative = cumulative[[key for key, _, _ in groups]].fillna(0.0)
        widest = groups[-1][0]
        order = list(cumulative[widest].sort_values(ascending=False).index)
        cumulative = cumulative.loc[order]
        positive = cumulative.to_numpy()[cumulative.to_numpy() > 0]
        floor = 10.0 ** math.floor(math.log10(float(positive.min())))
        positions = np.arange(len(order))
        heights = np.linspace(0.74, 0.42, len(groups))      # widest group first, widest bar
        for (key, label, colour), height in zip(reversed(groups), heights):
            values = cumulative[key].to_numpy(dtype="float64")
            shown = values > 0
            axes[1].barh(positions[shown], values[shown] - floor, left=floor, color=colour,
                         label=label, edgecolor="white", linewidth=0.4, height=height)
        nameplate = np.array([
            float(by_area.loc[(area, widest), "nameplate_TWh"]) * 1000.0
            if (area, widest) in by_area.index else 0.0 for area in order])
        ticked = nameplate > 0
        axes[1].plot(nameplate[ticked], positions[ticked], marker="|", markersize=9,
                     linestyle="none", color="#333333", zorder=5, label="nameplate ceiling")
        axes[1].set_xscale("log")
        axes[1].set_xlim(left=floor)
        axes[1].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        axes[1].set_yticks(positions)
        axes[1].set_yticklabels([display_name(a, zones) for a in order], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("usable storage energy, GWh (log scale)", fontsize=9)
        axes[1].grid(axis="x", alpha=0.25)
        axes[1].tick_params(axis="x", labelsize=8)
        # Narrowest group first, as in the tables, then the tick. Lower right: the
        # smallest storage is at the bottom, so their bars end near the left
        # edge and leave that corner empty.
        by_label = dict(zip(*reversed(axes[1].get_legend_handles_labels())))
        wanted = [label for _, label, _ in groups] + ["nameplate ceiling"]
        shown = [label for label in wanted if label in by_label]
        axes[1].legend([by_label[label] for label in shown], shown,
                       fontsize=7, loc="lower right", framealpha=0.95)
        axes[1].set_title("Usable electricity storage\n(tick: the nameplate ceiling)",
                          fontsize=9)
    else:
        _empty_panel(axes[1], "No electricity storage energy\nis stated in this build")

    return _finish(fig, out_dir / "fig_storage_capacity.png")


def group_increments(by_area: pd.DataFrame, measure: str) -> pd.DataFrame:
    """area x group, each group holding only what the one before it did not.

    ``by_area`` is cumulative, which is what a table wants and a stacked bar
    cannot use: stacking cumulative totals would draw the literal storage twice.
    """
    if by_area is None or not len(by_area):
        return pd.DataFrame()
    wide = by_area[measure].unstack("group")
    order = [key for key, _, _ in STORAGE_GROUPS if key in wide.columns]
    if not order:
        return pd.DataFrame()
    wide = wide[order].fillna(0.0)
    out = wide.copy()
    for later, earlier in zip(order[1:], order[:-1]):
        out[later] = (wide[later] - wide[earlier]).clip(lower=0.0)
    return out


def _draw_corridors(ax, shapes: ZoneShapes, countries: ZoneShapes, matrix: pd.DataFrame,
                    neighbours: bool = True) -> Dict:
    """The corridor graph on the map, width linear in capacity.

    Always per bidding zone, whatever level the table beside it uses. Rolled up
    to countries the intra-country corridors disappear from the picture while
    the text above still counts them, and the Nordic ring -- which is most of
    what this figure has to say -- is drawn as four dots.

    Returns what the legend needs to state the width scale. A line whose width
    encodes a quantity is decoration unless the reader is told what a given
    width is worth, so the caller must draw that key.
    """
    areas = sorted(set(matrix.index) | set(matrix.columns))
    geometry = shapes_for_areas(shapes, areas, True)
    centres = {area: _polygons_centroid(polygons) for area, polygons in geometry.items()}

    _draw_context(ax, shapes, areas, neighbours)
    paths = [p for polygons in geometry.values()
             for p in (_polygon_path(poly) for poly in polygons) if p]
    ax.add_collection(PatchCollection(
        [PathPatch(p) for p in paths],
        facecolor="#dce7f0", edgecolor="#ffffff", linewidth=0.5, zorder=2))
    _draw_country_outlines(ax, countries, areas)

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
    zone_matrix: pd.DataFrame,
    shapes: Optional[ZoneShapes] = None,
    countries: Optional[ZoneShapes] = None,
    neighbours: bool = True,
    ) -> str:
    """Who connects to whom, and who leans on it.

    The left panel was a capacity matrix drawn as a heatmap. A map says the same
    thing better -- the matrix could not show that the Nordic zones form a ring
    while Germany is a hub -- and the exact numbers the heatmap carried are in
    the table above it, where they can be read to a decimal.

    The two panels are at different levels on purpose. The map is always per
    bidding zone, because that is what a corridor connects; the bars beside it
    are per reported area, because that is what the table above them lists.
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
        info = _draw_corridors(axes[0], shapes, countries or ZoneShapes(),
                               zone_matrix, neighbours)
        _finish_map(axes[0])
        _corridor_width_legend(axes[0], info)
        axes[0].set_title("Electricity transfer capacity between bidding zones", fontsize=9)
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


# ============================================================================
# Checks
# ============================================================================

#: A check's state. Only one of them asks for anything, and the report says so
#: before the table, so a row that is not "ok" does not read as a failure.
CHECK_OK = "ok"
CHECK_WATCH = "watch"
CHECK_ACT = "act"
CHECK_NOT_RUN = "not run"

CHECK_MEANING = {
    CHECK_OK: "nothing found",
    CHECK_WATCH: "nothing in the data is wrong, but it is close to a limit or a known gap, "
                 "and worth knowing when reading a run",
    CHECK_ACT: "the data as written cannot do what it asks: a run will lean on penalties or "
               "dummy variables, or carry capacity that can never run -- follow it up before "
               "trusting one",
    CHECK_NOT_RUN: "the files this check needs were not read",
}


class Check(NamedTuple):
    name: str
    result: str
    state: str


def production_without_sink(workbook: Workbook, timeseries: Timeseries) -> Dict[str, Dict]:
    """Per carrier, the nodes whose production has nowhere to go.

    A node's output can go to demand on it (a ``ts_influx`` series or a negative
    constant ``influx``), to a unit that consumes it, to a price node's market,
    or over a corridor to a node that has one of those. Corridors are followed
    to the end, so a chain of nodes with one sink between them is served. A
    storage on the node is not a sink: it fills once and then the output stops.

    Returns ``{carrier: {"nodes": [(node, MW)], "unknown": bool}}``, largest
    first. ``unknown`` marks a carrier whose demand is in timeseries files that
    were not read, so its answer is not known.
    """
    io = workbook.p_gnu_io
    gn = workbook.p_gn
    if io.empty or not {"grid", "node", "input_output"} <= set(io.columns):
        return {}
    io = io[col_or(io, "isActive", 1.0).fillna(1.0) == 1]
    capacity = pd.to_numeric(io["capacity"], errors="coerce").fillna(0.0)
    side, grid, node = (io["input_output"].astype(str), io["grid"].astype(str),
                        io["node"].astype(str))

    sinks: set = set()
    sinks.update(node[(side == "input") & (capacity > 0)])
    if not gn.empty and {"grid", "node"} <= set(gn.columns):
        influx = col_or(gn, "influx", 0.0).fillna(0.0)
        priced = col_or(gn, "usePrice", 0.0).fillna(0.0) == 1
        sinks.update(gn.loc[(influx < 0) | priced, "node"].astype(str))
    for nodes in timeseries.demand_nodes.values():
        sinks.update(nodes)

    # Nodes joined by a corridor share their sinks.
    parent: Dict[str, str] = {}

    def root(n: str) -> str:
        while parent.get(n, n) != n:
            n = parent[n]
        return n

    gnn = workbook.p_gnn
    if not gnn.empty and {"from_node", "to_node"} <= set(gnn.columns):
        open_link = pd.to_numeric(col_or(gnn, "transferCap", 0.0), errors="coerce").fillna(0.0) > 0
        for a, b in zip(gnn.loc[open_link, "from_node"].astype(str),
                        gnn.loc[open_link, "to_node"].astype(str)):
            parent[root(a)] = root(b)
    served = {root(n) for n in sinks}

    out: Dict[str, Dict] = {}
    produced = io[(side == "output") & (capacity > 0)].assign(
        _cap=capacity, _grid=grid, _node=node)
    for carrier, _, _ in CARRIERS:
        part = produced[produced["_grid"] == carrier]
        if part.empty:
            continue
        if carrier in DEMAND_FAMILY and not timeseries.available:
            out[carrier] = {"nodes": [], "unknown": True}
            continue
        per_node = part.groupby("_node", observed=True)["_cap"].sum()
        stranded = [(n, float(mw)) for n, mw in per_node.sort_values(ascending=False).items()
                    if root(n) not in served]
        out[carrier] = {"nodes": stranded, "unknown": False}
    return out


def _peak_by_area(timeseries: Timeseries, workbook: Workbook, carrier: str,
                  zones: bool) -> pd.Series:
    """Each area's highest hour of demand for one carrier, MW.

    The hourly series where there is one, on the area's own coincident hours;
    otherwise the constant rate, which is its own peak. Never both: the
    constant is ignored wherever a series exists.
    """
    hourly = timeseries.hourly.get(f"demand_{carrier}")
    if hourly is not None and timeseries.areas and float(np.abs(hourly).sum()) > 0:
        peak = pd.Series(hourly.max(axis=0), index=timeseries.areas, dtype="float64")
        return peak[peak > 0]
    constant = constant_demand(workbook, carrier, zones)
    if constant.empty:
        return pd.Series(dtype="float64")
    return (constant.set_index("area")["mean"] / MWH_TO_TWH / HOURS_PER_YEAR).astype("float64")


def _exports_by_area(workbook: Workbook, carrier: str, zones: bool) -> pd.Series:
    """Transfer capacity out of each area for one carrier, MW, each direction on its own."""
    gnn = workbook.p_gnn
    if gnn.empty or not {"grid", "from_node", "to_node"} <= set(gnn.columns):
        return pd.Series(dtype="float64")
    rows = gnn[gnn["grid"].astype(str) == carrier]
    if rows.empty:
        return pd.Series(dtype="float64")
    frm = [area_of(n, zones) for n in rows["from_node"]]
    to = [area_of(n, zones) for n in rows["to_node"]]
    cap = pd.to_numeric(col_or(rows, "transferCap", 0.0), errors="coerce").fillna(0.0)
    leaving = pd.Series(cap.to_numpy(), index=frm)[[a != b for a, b in zip(frm, to)]]
    return leaving.groupby(level=0).sum().astype("float64")


def capacity_against_uses(workbook: Workbook, classification: Classification,
                          timeseries: Timeseries, carrier: str, zones: bool) -> pd.DataFrame:
    """Per area: dispatchable production capacity against everything that could take it.

    What could take a carrier's output in its busiest hour is the area's peak
    demand, the capacity of every unit that consumes the carrier, and the
    capacity to export it. Production beyond all three together can never run,
    whatever Backbone decides. Weather-driven output and storage are left out
    of production, since neither runs at nameplate against a peak.

    Columns, GW: production, peak, consumers, exports, uses (the last three
    summed with the peak), and ``ratio`` = production / uses, infinite where
    production has no use at all. ``timeseries`` must be at the report's level.
    Empty for a carrier whose demand is in series that were not read: its peak
    is not known, and a zero in its place would flag every area.
    """
    columns = ["production", "peak", "consumers", "exports", "uses", "ratio"]
    if carrier in DEMAND_FAMILY and not timeseries.available:
        return pd.DataFrame(columns=columns, dtype="float64")
    rows = classification.rows
    rows = rows[(rows["carrier"] == carrier)
                & ~rows["label"].isin(NOT_DISPATCHABLE.get(carrier, frozenset()))]
    production = (rows.assign(area=[area_of(n, zones) for n in rows["node"]])
                  .groupby("area", observed=True)["capacity"].sum()) if len(rows) else \
        pd.Series(dtype="float64")
    consumption = consumption_capacity(workbook, carrier, zones)
    consumers = (consumption.groupby("area", observed=True)["capacity"].sum()
                 if not consumption.empty else pd.Series(dtype="float64"))
    table = pd.DataFrame({
        "production": production,
        "peak": _peak_by_area(timeseries, workbook, carrier, zones),
        "consumers": consumers,
        "exports": _exports_by_area(workbook, carrier, zones),
    }).fillna(0.0) * MW_TO_GW
    table = table[(table["production"] > 0) | (table["peak"] > 0)]
    table["uses"] = table["peak"] + table["consumers"] + table["exports"]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["ratio"] = np.where(table["uses"] > 0, table["production"] / table["uses"],
                                  np.where(table["production"] > 0, np.inf, np.nan))
    return table


def oversized(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``capacity_against_uses`` whose production can never all run.

    Largest stranded capacity first: 65 GW at 4.9x matters more than 0.7 GW at 8.6x.
    Production with no use at all is left to the nowhere-to-go check, which
    names its nodes, so the same capacity is not reported twice.
    """
    if table is None or table.empty:
        return pd.DataFrame()
    big = table[(table["ratio"] >= OVERSIZE_RATIO) & np.isfinite(table["ratio"])]
    stranded = (big["production"] - big["uses"]).clip(lower=0.0)
    return big.assign(stranded=stranded).sort_values("stranded", ascending=False)


def _ratio_text(ratio: float) -> str:
    return "no use" if np.isinf(ratio) else f"{ratio:.1f}x"


def run_checks(workbook: Workbook, classification: Classification, transfer: Dict,
               served_by_timeseries: Sequence[str] = ()) -> List[Check]:
    """Structurally impossible values, plus the known gaps worth watching.

    Nothing here is a judgement about whether a number is plausible. A check
    that fires on correct data every run is not strict, it is broken, so a
    capacity-to-peak ratio or a capacity-factor band -- both of which have
    defensible extremes in the shipped scenarios -- is shown sorted in a table
    instead of being judged here.
    """
    checks = []

    def add(name: str, result: str, ok: bool) -> None:
        checks.append(Check(name, result, CHECK_OK if ok else CHECK_ACT))


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
    add(
        "Negative capacity, transferCap or emission factor",
        "none" if not negatives else f"found in {', '.join(negatives)}",
        not negatives,
    )

    # A constant influx is overridden by a timeseries, never added to it -- the
    # Backbone gates it on `not gn_influxTs(grid, node)`. A node carrying both has a
    # number in the workbook that does nothing, and nothing would say so.
    gn = workbook.p_gn
    with_constant = set()
    if not gn.empty and "influx" in gn.columns and "grid" in gn.columns:
        live = col_or(gn, "influx", 0.0).fillna(0.0) != 0
        with_constant = {(str(g), str(n)) for g, n in zip(gn.loc[live, "grid"],
                                                          gn.loc[live, "node"])}
    overridden = sorted(f"{n}" for g, n in with_constant if g in served_by_timeseries)
    add(
        "Constant influx on a node that also has a timeseries",
        (f"none of {plural(len(with_constant), 'node')} with a constant influx" if not overridden
         else f"the constant is ignored on {summarise(overridden)}"),
        not overridden,
    )

    priced = gn[col_or(gn, "usePrice", 0.0).fillna(0.0) == 1] if not gn.empty else pd.DataFrame()
    bad_price = priced[pd.to_numeric(col_or(priced, "price", np.nan), errors="coerce").fillna(0) <= 0] \
        if not priced.empty else pd.DataFrame()
    add(
        "Price at or below zero on a priced node",
        f"none of {plural(len(priced), 'priced node')}" if bad_price.empty
        else f"{len(bad_price)}: {summarise(bad_price['node'].tolist())}",
        bad_price.empty,
    )

    add(
        "One-way transfer capacity (the other direction is zero)",
        f"none of {plural(transfer.get('pairs', 0), 'corridor')}" if not transfer.get("one_way")
        else plural(transfer["one_way"], "corridor"),
        not transfer.get("one_way"),
    )

    unlabelled = classification.unlabelled
    by_reason = {}
    if not unlabelled.empty:
        for reason, group in unlabelled.groupby("reason", observed=True):
            by_reason[str(reason)] = float(group["capacity"].sum() * MW_TO_GW)
    real = {r: gw for r, gw in by_reason.items() if gw > 0}
    add(
        "Unit capacity the technology grouping has no home for",
        "none -- every unit matched a group" if not real
        else "; ".join(f"{gw:,.1f} GW, {r}" for r, gw in sorted(real.items(), key=lambda x: -x[1])),
        not real,
    )

    boundary = workbook.boundary
    if boundary.empty or "param_gnBoundaryTypes" not in boundary.columns:
        add("Storage row claiming a constant but carrying none", "no storage rows", True)
    else:
        upward = boundary[boundary["param_gnBoundaryTypes"].astype(str) == "upwardLimit"]
        claims = upward[col_or(upward, "useConstant", 0.0).fillna(0.0) == 1]
        missing = claims[col_or(claims, "constant", np.nan).isna()]
        add(
            "Storage row claiming a constant but carrying none",
            f"none of {plural(len(claims), 'row')}" if missing.empty
            else f"{len(missing)}: {summarise(missing['node'].tolist())}",
            missing.empty,
        )

    # A flow unit generates its capacity times its ts_cf, so without a series it
    # never generates at all: capacity that can never run, like any other.
    flow_unit = workbook.flow_unit
    flows = set(str(f) for f in flow_unit.get("flow", pd.Series(dtype=str)).unique())
    without = sorted(flows - set(VRE_FAMILIES))
    parts = []
    for flow in without:
        units = set(flow_unit.loc[flow_unit["flow"].astype(str) == flow, "unit"].astype(str))
        rows = classification.rows[classification.rows["unit"].astype(str).isin(units)]
        parts.append(f"{flow}: {plural(len(units), 'unit')}, "
                     f"{_num(float(rows['capacity'].sum()) * MW_TO_GW)} GW")
    checks.append(Check(
        "Flow with units but no capacity-factor timeseries",
        "none" if not without
        else "; ".join(parts) + ". With no profile these units never generate",
        CHECK_ACT if without else CHECK_OK,
    ))
    return checks


def carrier_check_rows(stranded: Dict[str, Dict],
                       against_uses: Dict[str, pd.DataFrame]) -> List[Check]:
    """The two checks that set production against what could take it."""
    titles = {grid: title.lower() for grid, title, _ in CARRIERS}
    rows = []

    name = "Production with nowhere to go"
    found = [(c, v["nodes"]) for c, v in stranded.items() if v["nodes"]]
    unknown = [c for c, v in stranded.items() if v["unknown"]]
    if found:
        parts = []
        for carrier, nodes in found:
            gw = sum(mw for _, mw in nodes) * MW_TO_GW
            parts.append(f"{titles.get(carrier, carrier)}: {plural(len(nodes), 'node')}, "
                         f"{_num(gw)} GW -- {summarise([n for n, _ in nodes])}")
        text = "; ".join(parts) + ". No demand, consumer, price or corridor takes their output"
        if unknown:
            text += (f"; not known for {and_join([titles.get(c, c) for c in unknown])}, whose "
                     f"demand series were not read")
        rows.append(Check(name, text, CHECK_ACT))
    elif unknown:
        rows.append(Check(name, f"not known for {and_join([titles.get(c, c) for c in unknown])}, "
                                f"whose demand series were not read", CHECK_NOT_RUN))
    else:
        rows.append(Check(name, "none: every node that produces has something to take its "
                                "output", CHECK_OK))

    name = f"Production capacity at least {OVERSIZE_RATIO:g}x what can take it"
    parts = []
    for carrier, table in against_uses.items():
        big = oversized(table)
        if len(big):
            parts.append(f"{titles.get(carrier, carrier)} in "
                         + summarise([f"{a} ({_ratio_text(r)})"
                                      for a, r in zip(big.index, big["ratio"])]))
    if parts:
        rows.append(Check(name, "; ".join(parts) + ". See Consumption in each carrier's chapter",
                          CHECK_ACT))
    else:
        rows.append(Check(name, "none", CHECK_OK))
    return rows


def hydro_check_rows(result: "HydroChecks") -> List[Check]:
    """The hydro checks as rows of the Checks table.

    The minimum-generation check gets a row only when a storage node cannot carry its
    minimum. Storage nodes that come close are the ordinary state of run-of-river,
    and the Closest to floor table shows them.
    """
    rows = []
    minimum = result.minimum

    name = "Hydro minimum generation against inflow"
    act = (sorted(minimum.loc[minimum["state"] == CHECK_ACT, "node"])
           if len(minimum) and "state" in minimum.columns else [])
    beyond = (sorted(minimum.loc[minimum["beyond_turbines"], "node"])
              if len(minimum) and "beyond_turbines" in minimum.columns else [])
    if act:
        rows.append(Check(name, f"{plural(len(act), 'storage node')} cannot carry their minimum: "
                                f"{summarise(act)}", CHECK_ACT))
    elif result.skipped and beyond:
        rows.append(Check(name, f"{plural(len(beyond), 'storage node')} ask more than their turbines "
                                f"can give: {summarise(beyond)}", CHECK_ACT))

    name = "Hydro inflow a storage cannot pass with turbines and spill at full"
    overflow = result.overflow
    if result.skipped:
        rows.append(Check(name, f"inflow was not read: {result.skipped}", CHECK_NOT_RUN))
    elif not len(overflow):
        rows.append(Check(name, "no storage node with a ceiling receives inflow", CHECK_OK))
    else:
        over = list(overflow.loc[overflow["state"] == CHECK_ACT, "node"])
        if over:
            rows.append(Check(name, f"{plural(len(over), 'storage node')} overflow: {summarise(over)}",
                              CHECK_ACT))
        else:
            rows.append(Check(name, f"none of {plural(len(overflow), 'storage node')} overflows in "
                                    f"{plural(result.windows, 'climate window')}", CHECK_OK))
    return rows


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
    #: (level, title, anchor) of every heading, for the contents list.
    toc: List[Tuple[int, str, str]] = field(default_factory=list)

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def heading(self, anchor: str, title: str, level: int = 2) -> None:
        self.add(f'<a id="{anchor}"></a>')
        self.add()
        self.add(f"{'#' * level} {title}")
        self.add()
        self.toc.append((level, title, anchor))

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


@dataclass
class ReportContext:
    """What every chapter reads: the build, its series, and the per-carrier tables."""
    workbook: Workbook
    timeseries: Timeseries
    zones: bool
    out_dir: Path
    carriers: Dict[str, Dict]
    inventory: StorageInventory
    hydro_set: set
    hydro_table: pd.DataFrame
    hydro_result: "HydroChecks"
    peak: Optional[pd.Series]
    zone_matrix: pd.DataFrame
    shapes: ZoneShapes
    countries: ZoneShapes
    neighbours: bool
    demand_response: pd.DataFrame
    demand_response_costs: Dict[str, Tuple[float, float]]


def build_report(
    workbook: Workbook,
    classification: Classification,
    timeseries: Timeseries,
    out_dir: Path,
    zones: bool,
    shapes: Optional[ZoneShapes] = None,
    countries: Optional[ZoneShapes] = None,
    neighbours: bool = True,
    record: Optional[BuildRecord] = None,
    ) -> Tuple[str, List[str]]:
    """The whole report.md, and the figure names written beside it."""
    level = "bidding zone" if zones else "country"
    record = record if record is not None else BuildRecord()
    figures: List[str] = []
    # Both assets, whatever the report's level: the maps are always drawn per
    # bidding zone, with country borders over them.
    shapes = shapes if shapes is not None else load_zone_shapes(ZONE_ASSET)
    countries = countries if countries is not None else load_zone_shapes(COUNTRY_ASSET)
    # The files are read per zone; a country report is the roll-up of that. The
    # maps stay per zone either way, so they need the finer one.
    zone_timeseries = timeseries
    timeseries = timeseries if zones else roll_up_to_countries(timeseries)
    presence = carrier_presence(workbook, timeseries, zones)
    zone_presence = presence if zones else carrier_presence(workbook, zone_timeseries, True)

    inflow = classification.inflow_grids
    inventory = storage_inventory(workbook, timeseries, zones, inflow)
    hydro_set = hydro_grids(workbook, inflow)
    hydro_table = hydro_characteristics(workbook, inventory, timeseries, zones, hydro_set)
    # Per bidding-zone storage node whatever the level: the series are node-keyed and
    # pass through the country roll-up untouched.
    hydro_result = hydro_checks(workbook, zone_timeseries, hydro_set)
    capacity_factors = capacity_weighted_cf(timeseries, workbook, zones)
    # Any state that is not itself a carrier is storage, and what fills it is
    # charging it rather than consuming the carrier.
    storage = state_grids(workbook) - CARRIER_GRIDS

    carrier_data: Dict[str, Dict] = {}
    for carrier, title, slug in CARRIERS:
        capacity = capacity_by_area(classification, carrier, zones)
        consumption = consumption_capacity(workbook, carrier, zones)
        consumers, charging = split_consumption(consumption, storage)
        storage_grids = storage_grids_serving(workbook, carrier, inflow)
        carrier_data[carrier] = {
            "title": title,
            "slug": slug,
            "capacity": capacity,
            "generation": generation_only(capacity, carrier),
            "consumption": consumption,
            "consumers": consumers,
            "charging": charging,
            **dict(zip(("demand", "demand_is_constant"),
                       carrier_demand(workbook, timeseries, carrier, zones))),
            "peak": _peak_by_area(timeseries, workbook, carrier, zones),
            "capacity_factor": capacity_factors if carrier == "elec" else pd.DataFrame(),
            "against_uses": capacity_against_uses(workbook, classification, timeseries,
                                                  carrier, zones),
            "transfer": transfer_by_area(workbook, zones, carrier),
            "storage_grids": storage_grids,
            "storage": storage_by_area(workbook, inventory, storage_grids, zones),
            "links": carrier_links(workbook, carrier),
        }

    corridors = [data["transfer"][2] for data in carrier_data.values() if data["transfer"][2]]
    checks = run_checks(workbook, classification,
                        {"pairs": sum(f["pairs"] for f in corridors),
                         "one_way": sum(f["one_way"] for f in corridors)},
                        [c for c in DEMAND_FAMILY if not annual_range(
                            timeseries, f"demand_{c}").empty])
    checks += carrier_check_rows(
        production_without_sink(workbook, timeseries),
        {carrier: data["against_uses"] for carrier, data in carrier_data.items()})
    checks += hydro_check_rows(hydro_result)

    response, response_costs = demand_response(workbook, classification, zones)
    ctx = ReportContext(
        workbook=workbook, timeseries=timeseries, zones=zones, out_dir=out_dir,
        carriers=carrier_data, inventory=inventory, hydro_set=hydro_set,
        hydro_table=hydro_table, hydro_result=hydro_result, peak=_peak_demand(timeseries),
        zone_matrix=transfer_by_area(workbook, True)[1], shapes=shapes, countries=countries,
        neighbours=neighbours, demand_response=response, demand_response_costs=response_costs,
    )

    elec = carrier_data["elec"]
    elec_total = _total_gw(elec["generation"])
    elec_demand = elec["demand"]
    demand_total = float(elec_demand["mean"].sum()) if not elec_demand.empty else 0.0
    to_act = [c.name for c in checks if c.state == CHECK_ACT]
    to_watch = [c.name for c in checks if c.state == CHECK_WATCH]
    not_run = [c.name for c in checks if c.state == CHECK_NOT_RUN]

    # ---- lead -------------------------------------------------------------
    head = Report()
    head.add(f"# Input data summary -- {workbook.scenario}, {workbook.year}")
    head.add()
    lead = (
        f"This scenario builds a {len(set(country_of(zone_of(n)) for n in workbook.nodes))}-country, "
        f"{len(set(zone_of(n) for n in workbook.nodes))}-bidding-zone system with "
        f"{len(workbook.built_units()):,} units across {len(workbook.built_unittypes())} unit "
        f"types and {_num(elec_total)} GW of electricity generation"
    )
    if demand_total:
        lead += (f", against {_num(demand_total, 0)} TWh/yr of electricity demand averaged over "
                 f"{plural(len(timeseries.years), 'climate year')}")
    lead += (". " + ("No check asks for action."
                    if not to_act else
                    f"{plural(len(to_act), 'check')} {'asks' if len(to_act) == 1 else 'ask'} "
                    f"for action: {summarise(to_act)}."))
    if to_watch:
        lead += f" {len(to_watch)} worth watching: {summarise(to_watch)}."
    head.add(lead)
    head.add()

    # ---- one minute summary ----------------------------------------------
    head.add("## One minute summary")
    head.add()
    head.add(f"- **Scenario**: {workbook.scenario}, {workbook.year}, read from "
             f"`{workbook.path.name}` in `{workbook.path.parent.name}/`"
             + (f", built by `{record.config_file}`" if record.config_file else "")
             + (f" as `{workbook.build}`" if workbook.build else "") + ".")
    head.add(f"- **Reported by {level}**: {plural(len(presence), level)}"
             + ("." if zones else "; pass `--zones` to split them into bidding zones."))
    if timeseries.years:
        head.add(f"- **Climate windows**: {_window_text(record.config, timeseries)}. Every "
                 f"chapter reports a mean climate year; What varies between the years shows "
                 f"how far single years move.")
    if investment_and_timing(workbook):
        head.add("- **Not handled yet**: the build lets the fleet change during a run, which "
                 "this report does not follow; see Build settings.")
    for carrier, data in carrier_data.items():
        head.add(f"- **{data['title']}**: {_carrier_summary(ctx, carrier)}.")
        if carrier == "elec" and not hydro_table.empty:
            head.add(f"- **Hydro**: {_hydro_summary(ctx)}.")
    head.add(f"- **Checks**: {len(checks) - len(to_act) - len(to_watch) - len(not_run)} ok, "
             f"{len(to_watch)} to watch, {len(to_act)} to act on"
             + (f", {len(not_run)} not run" if not_run else "") + ", see Checks.")
    head.add()

    summary_rows = [
        ["Electricity generation", f"{_num(elec_total)} GW",
         "`p_gnu_io`, elec output side, less storage and demand response"],
    ]
    if not elec_demand.empty:
        worst = elec_demand.loc[elec_demand["mean"].idxmax()]
        summary_rows.append([
            "Electricity demand",
            f"{_num(demand_total, 0)} TWh/yr mean, largest {worst['area']} at {_num(worst['mean'], 0)}",
            f"`ts_influx_elec`, {plural(len(timeseries.years), 'climate year')}",
        ])
    if len(inventory.by_group) and float(inventory.by_group["usable_mean_TWh"].max()):
        widest = [key for key, _, _ in STORAGE_GROUPS][-1]
        summary_rows.append([
            "Electricity storage, with inflow hydro",
            f"{_num(inventory.by_group.loc[widest, 'usable_mean_TWh'])} TWh usable, "
            f"{_num(inventory.by_group.loc[widest, 'nameplate_TWh'])} TWh of nameplate",
            "`p_gnBoundaryPropertiesForStates` + `ts_node_hydro_storage_limits` + `p_gnu_io`",
        ])
    price = co2_price(workbook)
    if price is not None:
        summary_rows.append(["CO2 price", f"{_num(price, 0)} EUR/t, one value for the run",
                             "`ts_emissionPriceChange`"])
    head.table(["What", "Value", "Where it is read from"], summary_rows)

    # ---- the body, whose headings make the contents -----------------------
    body = Report()
    body.heading("build-settings", "Build settings")
    _build_section(body, record, workbook, timeseries)

    body.heading("where-the-model-is", "Where the model is")
    _coverage_section(body, presence, zone_presence, shapes, zones)
    figures.append(figure_carrier_map(out_dir, shapes, countries, zone_presence, zones,
                                      neighbours))
    body.figure(
        figures[-1],
        "Each bidding zone filled by the carriers it models, with country borders over the "
        "top. Industrial steam is not on the map: three carriers make seven fills and a "
        "colour can carry that, a fourth makes fifteen and it cannot. Take-away: how far "
        "each carrier reaches, and how much of the map is electricity alone.",
    )
    _omissions_section(body, presence, zones)

    figures.extend(_electricity_chapter(body, ctx))
    _hydro_chapter(body, ctx)
    for carrier, _, _ in CARRIERS:
        if carrier != "elec":
            figures.extend(_carrier_chapter(body, ctx, carrier))

    body.heading("what-varies-between-the-years", "What varies between the years")
    material = _material_hydro_areas(timeseries)
    _variation_section(body, timeseries, material, zones)
    figures.append(figure_climate_variability(out_dir, timeseries, material, zones))
    body.figure(
        figures[-1],
        f"Each area's climate years as a percentage of that area's "
        "own mean, so a small country and a large one are comparable. Take-away: a bad weather "
        "year barely moves electricity demand but can change hydro inflow by tens of per cent.",
        caveat=("Read from the realised climate years, not the forecast branches: those are "
                "per-hour quantiles taken across the years, not alternative years of their own."
                if timeseries.available else None),
    )

    body.heading("fuel-co2-and-emissions", "Fuel, CO2 and emissions")
    _price_section(body, workbook)

    body.heading("what-backbone-may-do", "What Backbone may do")
    _permissions_section(body, model_permissions(workbook))

    body.heading("checks", "Checks")
    body.add("Values that cannot be what was meant, capacity that can never run, known gaps, "
             "and hydro storage that cannot do what the data asks of them. A number that is "
             "merely surprising is not checked here; it is shown in the tables above. Each "
             "check is in one of four states, and only one of them asks for anything:")
    body.add()
    for state in (CHECK_OK, CHECK_WATCH, CHECK_ACT, CHECK_NOT_RUN):
        body.add(f"- **{state}**: {CHECK_MEANING[state]}.")
    body.add()
    body.table(["Check", "State", "Result"],
               [[c.name, f"**{c.state}**" if c.state == CHECK_ACT else c.state,
                 f"**{c.result}**" if c.state == CHECK_ACT else c.result] for c in checks])

    body.heading("not-summarized", "Not summarized")
    _not_summarized(body, workbook)

    body.heading("what-this-cannot-see", "What this cannot see")
    _limits_section(body, workbook, timeseries, inventory, zones)

    body.add("---")
    body.add()
    body.add(f"*Written by `tools/input_data_summary.py` from "
             f"`{workbook.path.parent.name}/`. Each run rewrites this report and the figures it "
             f"draws; any other file in this folder is left as it is.*")

    # ---- contents, from the headings the body wrote ------------------------
    head.add("## Contents")
    head.add()
    for depth, title, anchor in body.toc:
        head.add(("1. " if depth == 2 else "   - ") + f"[{title}](#{anchor})")
    head.add()
    if not zones:
        head.add("Country codes roll up the bidding zones: "
                 + ", ".join(f"**{c}** {n}" for c, n in sorted(COUNTRY_NAMES.items())
                             if c in {country_of(zone_of(x)) for x in workbook.nodes})
                 + ".")
        head.add()
    return Report(head.lines + body.lines).text(), figures


def _carrier_summary(ctx: ReportContext, carrier: str) -> str:
    """One carrier's line in the one-minute summary, each subsection's number once."""
    data = ctx.carriers[carrier]
    generated = _total_gw(data["generation"])
    consumed = _capacity_gw(data["consumers"])
    stored = storage_discharge(data["storage"])
    response = _total_gw(ctx.demand_response) if carrier == "elec" else 0.0
    corridors = data["transfer"][2]
    demand = float(data["demand"]["mean"].sum()) if not data["demand"].empty else 0.0
    if not (generated or consumed or demand or stored.sum()):
        return "nothing built in this scenario"
    parts = [f"{_num(generated)} GW generating"]
    if demand:
        parts.append(f"{_num(demand, 0)} TWh/yr demanded")
    elif carrier in DEMAND_FAMILY and not ctx.timeseries.available:
        parts.append("demand not read")
    else:
        parts.append("no demand built")
    if consumed:
        parts.append(f"{_num(consumed)} GW consuming")
    if stored.sum():
        energy = float(data["storage"]["energy_GWh"].sum())
        parts.append(f"{_num(float(stored.sum()))} GW of storage holding {_num(energy, 0)} GWh")
    if response:
        parts.append(f"{_num(response)} GW of demand response")
    if corridors:
        parts.append(f"{_num(corridors['total_GW'])} GW of interconnection")
    return ", ".join(parts)


def _hydro_summary(ctx: ReportContext) -> str:
    table = ctx.hydro_table
    text = f"{_num(float(table['storage_MWh'].sum()) * MWH_TO_TWH)} TWh of usable storage"
    if ctx.timeseries.hydro is not None:
        text += (f", {_num(float(table['inflow_MWh_yr'].sum()) * MWH_TO_TWH, 0)} TWh/yr of "
                 f"inflow in a mean year")
    return text


def _total_gw(capacity: pd.DataFrame) -> float:
    if capacity is None or capacity.empty:
        return 0.0
    return float(capacity.to_numpy(dtype="float64").sum())


def _capacity_gw(frame: pd.DataFrame) -> float:
    """The GW in a ``consumption_capacity``-shaped frame, which holds MW."""
    if frame is None or frame.empty:
        return 0.0
    return float(frame["capacity"].sum() * MW_TO_GW)


def _peak_demand(timeseries: Timeseries) -> Optional[pd.Series]:
    """Each area's highest single hour of electricity demand, in MW.

    Taken on the area's own coincident hours, not as the sum of its zones' separate
    maxima: a country does not hit every zone's peak in the same hour.
    """
    demand = timeseries.hourly.get("demand_elec")
    if demand is None or not timeseries.areas:
        return None
    return pd.Series(demand.max(axis=0), index=timeseries.areas, dtype="float64")


def window_caveat(timeseries: Timeseries) -> Optional[str]:
    """One sentence for a build whose climate windows are longer than a year.

    A shorter window never reaches here: the timeseries are not read for it.
    """
    hours = timeseries.window_hours
    if not timeseries.available or not hours or hours <= HOURS_PER_YEAR:
        return None
    return (f"Each climate window is {hours / 24:,.4g} days and one starts in every climate "
            f"year, so neighbouring windows overlap. The annual figures, the net load and the "
            f"climate-year comparison use the first 365 days of each window, which counts "
            f"each climate year once; the hydro checks run each window whole, as Backbone "
            f"runs it.")


def _material_hydro_areas(timeseries: Timeseries) -> List[str]:
    if timeseries.annual.empty:
        return []
    inflow = timeseries.annual[timeseries.annual["key"] == "inflow_hydro"]
    if inflow.empty:
        return []
    means = inflow.groupby("area", observed=True)["TWh"].mean()
    return means[means >= MATERIAL_HYDRO_TWH].sort_values(ascending=False).index.tolist()


def _electricity_chapter(report: Report, ctx: ReportContext) -> List[str]:
    """Electricity: every subsection, always, each saying so when it has nothing."""
    report.heading("elec", "Electricity")
    _chapter_opening(report, ctx, "elec")
    report.heading("elec-generation", "Generation", 3)
    figures = _generation_subsection(report, ctx, "elec")
    report.heading("elec-consumption", "Consumption", 3)
    _consumption_subsection(report, ctx, "elec")
    report.heading("elec-net-load", "Net load", 3)
    figures += _netload_subsection(report, ctx)
    report.heading("elec-interconnection", "Interconnection", 3)
    figures += _transfer_subsection(report, ctx)
    report.heading("elec-storage", "Storage", 3)
    figures += _elec_storage_subsection(report, ctx)
    report.heading("elec-demand-management", "Demand management", 3)
    _demand_management_subsection(report, ctx)
    return figures


def _carrier_chapter(report: Report, ctx: ReportContext, carrier: str) -> List[str]:
    """District heat, steam or hydrogen; corridors and storage only where the build has them."""
    data = ctx.carriers[carrier]
    slug = data["slug"]
    report.heading(slug, data["title"])
    if not _chapter_opening(report, ctx, carrier):
        return []
    report.heading(f"{slug}-generation", "Generation", 3)
    figures = _generation_subsection(report, ctx, carrier)
    report.heading(f"{slug}-consumption", "Consumption", 3)
    _consumption_subsection(report, ctx, carrier)
    if data["transfer"][2]:
        report.heading(f"{slug}-interconnection", "Interconnection", 3)
        _carrier_transfer_subsection(report, ctx, carrier)
    if _has_storage(data):
        report.heading(f"{slug}-storage", "Storage", 3)
        _carrier_storage_subsection(report, ctx, carrier)
    return figures


def _has_storage(data: Dict) -> bool:
    table = data["storage"]
    return bool(table is not None and not table.empty
                and (storage_discharge(table).sum() > 0 or table["energy_GWh"].sum() > 0))


def _chapter_opening(report: Report, ctx: ReportContext, carrier: str) -> bool:
    """The chapter's first paragraph, one number per subsection; False for an absent carrier."""
    data = ctx.carriers[carrier]
    title = data["title"].lower()
    generated = _total_gw(data["generation"])
    consumed = _capacity_gw(data["consumers"])
    stored = float(storage_discharge(data["storage"]).sum())
    demand = data["demand"]
    corridors = data["transfer"][2]
    response = _total_gw(ctx.demand_response) if carrier == "elec" else 0.0
    if not (generated or consumed or stored or not demand.empty):
        report.add(f"Nothing in this scenario produces, consumes or demands {title}.")
        report.add()
        return False

    text = f"{_num(generated)} GW of capacity generates {title}"
    if not demand.empty:
        text += f", against {_num(float(demand['mean'].sum()), 0)} TWh/yr of demand"
        if not data["demand_is_constant"]:
            text += " in a mean climate year"
    elif carrier in DEMAND_FAMILY:
        # A demand series is written for this carrier in a normal build, so its
        # absence here is about this folder, not about the carrier.
        text += ", and its demand series could not be read"
    else:
        text += ", and nothing demands it directly"
    text += "."
    beside = []
    if stored:
        beside.append(f"{_num(stored)} GW of storage")
    if response:
        beside.append(f"{_num(response)} GW of demand response")
    if corridors:
        beside.append(f"{_num(corridors['total_GW'])} GW of interconnection")
    if beside:
        text += f" Beside it {'is' if len(beside) == 1 else 'are'} {and_join(beside)}."
    if consumed:
        text += f" {_num(consumed)} GW of units consume {title} to make something else."
    if carrier != "elec":
        absent = [what for what, has in (("corridors", corridors), ("storage", stored))
                  if not has]
        if absent:
            text += f" It has no {' and no '.join(absent)} in this build."
    report.add(text)
    report.add()
    return True


def _generation_subsection(report: Report, ctx: ReportContext, carrier: str) -> List[str]:
    """Generation capacity, its links to electricity, its figures; for electricity, its yield."""
    data = ctx.carriers[carrier]
    zones = ctx.zones
    generation, title = data["generation"], data["title"]
    if generation.empty:
        report.add(f"Nothing generates {title.lower()} in this scenario.")
        report.add()
    else:
        produced = _total_gw(generation)
        labels = list(generation.columns)
        ordered = generation.sum(axis=1).sort_values(ascending=False)
        rows = []
        for area in ordered.index:
            row = [display_name(area, zones)]
            row += [_num(generation.loc[area, l]) if generation.loc[area, l] else "-"
                    for l in labels]
            row.append(f"**{_num(ordered[area])}**")
            rows.append(row)
        rows.append(["**total**"] + [f"**{_num(generation[l].sum())}**" for l in labels]
                    + [f"**{_num(produced)}**"])
        report.add("Generation capacity, GW, output side only -- a unit that also consumes "
                   "another carrier is counted once, here." + _moved_text(data))
        report.add()
        report.table([level_header(zones)] + labels + ["total"], rows)
        links = _links_text(data, carrier)
        if links:
            report.add(links)
            report.add()

    names = []
    name = figure_carrier(ctx.out_dir, data["slug"], title, generation, data["demand"],
                          data["consumers"], zones)
    if name:
        names.append(name)
        report.figure(
            name,
            f"{title} generation capacity and annual demand, in the same area order. Take-away: "
            f"what each area has built for this carrier, read directly against what it uses.",
            caveat=("Technologies are grouped by the grids each unit connects, not by its unit "
                    "type's name: pumped hydro with natural inflow counts as hydro, closed-loop "
                    "pumped hydro as storage. Capacity that fits no group is counted under Checks."
                    if carrier == "elec" else None),
        )
    name = figure_carrier_mix(ctx.out_dir, data["slug"], title, generation, zones)
    if name:
        names.append(name)
        report.figure(
            name,
            f"Each area's {title.lower()} generation capacity as shares of its own total, so a "
            f"small area reads as clearly as a large one. Take-away: which technologies each "
            f"area relies on.",
        )

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
                   f"{plural(len(ctx.timeseries.years), 'climate year')}:")
        report.add()
        report.add("`capacity factor = sum over nodes(capacity x its own factor) / "
                   "sum over nodes(capacity)`")
        report.add()
        report.table([level_header(zones)] + [f"{f} CF" for f in flows], rows)
        report.add("*Weighted, so a 200 MW zone cannot count for as much as a 12 GW one.*")
        report.add()
    if carrier == "elec" and not ctx.hydro_table.empty:
        report.add("Hydro here is turbine capacity. What the water allows it -- how much each "
                   "storage holds, what flows in, and what the turbines must keep generating -- is "
                   "in [Hydro](#hydro).")
        report.add()
    return names


def _moved_text(data: Dict) -> str:
    """Which output capacity the generation table leaves out, and where it went."""
    capacity, generation = data["capacity"], data["generation"]
    if capacity is None or capacity.empty:
        return ""
    moved = [c for c in capacity.columns if c not in generation.columns and capacity[c].sum() > 0]
    stored = [c for c in moved if c in STORAGE_LABELS]
    parts = []
    if stored:
        parts.append(f"{and_join(stored)} {'is' if len(stored) == 1 else 'are'} under Storage")
    if DEMAND_RESPONSE in moved:
        parts.append(f"{DEMAND_RESPONSE} under Demand management")
    if not parts:
        return ""
    text = parts[0] if len(parts) == 1 else f"{parts[0]}, and {parts[1]}"
    return f" {text[0].upper()}{text[1:]}, so no megawatt is counted twice."


def _links_text(data: Dict, carrier: str) -> str:
    """How the carrier's generation is tied to electricity, in one or two sentences."""
    links = data["links"] or {}
    if carrier == "elec":
        co = links.get("co_by_carrier") or {}
        if not co:
            return ""
        named = and_join([f"{_num(gw)} GW is on units that also make "
                          f"{CARRIER_TITLES[other].lower()}" for other, gw in co.items()])
        return (f"Of this, {named}: combined heat and power, which makes both from one fuel. "
                f"Their chapters say how tightly the two outputs are tied.")
    produced = _total_gw(data["generation"])

    def share(gw: float) -> str:
        return "All of it" if gw >= produced - 0.05 else f"{_num(gw)} GW of it"

    sentences = []
    co = links.get("co_GW", 0.0)
    if co:
        flexible = links.get("flexible_GW", 0.0)
        split = []
        if co - flexible > 0.05:
            split.append(f"{_num(co - flexible)} GW at a fixed ratio between the two outputs "
                         f"(`cb`)")
        if flexible > 0.05:
            split.append(f"{_num(flexible)} GW free to move between limits (`cb` with `cv`)")
        sentences.append(f"{share(co)} is combined heat and power that also generates "
                         f"{_num(links.get('co_elec_GW', 0.0))} GW of electricity"
                         + (f": {and_join(split)}" if split else "") + ".")
    made = links.get("from_elec_GW", 0.0)
    if made:
        sentences.append(f"{share(made)} is made from electricity.")
    return " ".join(sentences)


def _consumption_subsection(report: Report, ctx: ReportContext, carrier: str) -> None:
    """Demand and the units that consume the carrier, then capacity against every use."""
    data = ctx.carriers[carrier]
    zones, timeseries = ctx.zones, ctx.timeseries
    title = data["title"].lower()
    demand, consumers = data["demand"], data["consumers"]

    if demand.empty and carrier in DEMAND_FAMILY and timeseries.skipped:
        report.add(f"Demand needs the timeseries files, which were not read: {timeseries.skipped}.")
        report.add()
    elif demand.empty and carrier in DEMAND_FAMILY:
        # The family exists for this carrier but no file of it was found, which
        # is different from a carrier that never has one.
        report.add(f"No `{DEMAND_FAMILY[carrier]}_<year>.gdx` file was found in this folder, so "
                   f"there is no demand to report.")
        report.add()
    elif demand.empty:
        # Both routes checked: no ts_influx family is written for this carrier,
        # and p_gn carries no constant influx for it either.
        report.add(f"Nothing demands {title} in this build, by either route -- there is no "
                   f"`ts_influx` series for it and no constant `influx` in `p_gn`"
                   + (", so the units that consume it are its only outlet."
                      if not consumers.empty else "."))
        report.add()

    if not demand.empty or not consumers.empty:
        pivot = (consumers.pivot_table(index="area", columns="label", values="capacity",
                                       aggfunc="sum", fill_value=0.0) * MW_TO_GW
                 if not consumers.empty else pd.DataFrame())
        by_demand = demand.set_index("area")["mean"] if not demand.empty else pd.Series(dtype="float64")
        peak = data["peak"] * MW_TO_GW
        areas = list(by_demand.sort_values(ascending=False).index)
        if not pivot.empty:
            areas += [a for a in pivot.sum(axis=1).sort_values(ascending=False).index
                      if a not in areas]
        demanded = not demand.empty
        rows = []
        for area in areas:
            row = [display_name(area, zones)]
            if demanded:
                row += [_num(by_demand[area]) if area in by_demand.index else "-",
                        _num(peak[area]) if area in peak.index else "-"]
            row += [_num(pivot.loc[area, c]) if area in pivot.index and pivot.loc[area, c] else "-"
                    for c in pivot.columns]
            rows.append(row)
        rows.append(["**total**"] + ([f"**{_num(float(by_demand.sum()))}**", ""] if demanded else [])
                    + [f"**{_num(float(pivot[c].sum()))}**" for c in pivot.columns])

        if not demand.empty and data["demand_is_constant"]:
            source = (" Demand is a constant hourly rate in `p_gn`'s `influx` column, the same in "
                      "every climate year, so its peak is that rate.")
        elif not demand.empty:
            source = (f" Demand is the mean over {plural(len(timeseries.years), 'climate year')} "
                      f"and its peak the highest hour in any of them; [What varies between the "
                      f"years](#what-varies-between-the-years) shows how far single years move.")
        else:
            source = ""
        if pivot.empty:
            what = f"Demand for {title}."
        else:
            what = (f"{'Demand, and the units' if demanded else 'The units'} that consume "
                    f"{title}, GW, by what they make from it. A consuming unit is not demand: "
                    f"Backbone decides when it runs.")
        report.add(what + source)
        report.add()
        report.table([level_header(zones)] + (["demand TWh/yr", "peak GW"] if demanded else [])
                     + [str(c) for c in pivot.columns], rows)
        gloss = explain_grids(list(pivot.columns), ctx.inventory.ratio_facts)
        if gloss:
            report.add(gloss)
            report.add()
    if not data["charging"].empty:
        # No number: this is the carrier's side of the charger and Storage reads
        # the storage's side, and two totals for one unit would only disagree.
        report.add(f"Units that charge storage with {title} are under Storage, beside the storage "
                   f"they fill.")
        report.add()
    _uses_table(report, data, carrier, zones)


#: What "dispatchable" leaves out of generation, per carrier, said the way the
#: reader needs it. Storage and demand response are not generation to begin with.
_LEFT_OUT = {
    "elec": "wind and solar",
    "dheat": "solar heat",
}


def _uses_table(report: Report, data: Dict, carrier: str, zones: bool) -> None:
    """Dispatchable generation against everything that could take its output."""
    uses = data["against_uses"]
    if uses is None or uses.empty:
        return
    title = data["title"].lower()

    def cell(area, column):
        return _num(uses.loc[area, column]) if uses.loc[area, column] else "-"

    rows = []
    for area in uses.sort_values("production", ascending=False).index:
        ratio = uses.loc[area, "ratio"]
        shown = _ratio_text(ratio) if np.isfinite(ratio) or np.isinf(ratio) else "-"
        rows.append([display_name(area, zones), cell(area, "production"), cell(area, "peak"),
                     cell(area, "consumers"), cell(area, "exports"),
                     f"**{shown}**" if ratio >= OVERSIZE_RATIO else shown])
    rows.append(["**total**", f"**{_num(float(uses['production'].sum()))}**", "", "", "", ""])

    left_out = (f" Dispatchable generation leaves out {_LEFT_OUT[carrier]}, which never all run "
                f"at nameplate against a peak." if carrier in _LEFT_OUT else "")
    report.add(f"Dispatchable generation against everything that could take its output in the "
               f"busiest hour: the peak demand, the units that consume {title} or charge "
               f"storage with it, and the capacity to export it. The last column divides the "
               f"first by those three together.{left_out}")
    report.add()
    report.table([level_header(zones), "dispatchable GW", "peak GW", "consumers & charging GW",
                  "exports GW", "dispatchable / uses"], rows)

    big = oversized(uses)
    if len(big):
        named = summarise([f"{a} {_ratio_text(r)}, {_num(s)} GW"
                           for a, r, s in zip(big.index, big["ratio"], big["stranded"])],
                          sep="; ")
        report.add(f"**In {plural(len(big), level_header(zones))} dispatchable {title} capacity "
                   f"is at least {OVERSIZE_RATIO:g} times everything that could take it**, "
                   f"largest excess first, with the GW that can never run: {named}. Capacity this "
                   f"far above every use usually means a capacity in the source data is wrong.")
        report.add()
    idle = uses[np.isinf(uses["ratio"])]
    if len(idle):
        report.add(f"**{plural(len(idle), level_header(zones))} produce {title} with no use for "
                   f"it at all**: "
                   + summarise([f"{a} {_num(p)} GW" for a, p in
                                idle["production"].sort_values(ascending=False).items()])
                   + ". See the nowhere-to-go check.")
        report.add()


def _carrier_transfer_subsection(report: Report, ctx: ReportContext, carrier: str) -> None:
    """A carrier's corridors as a table; the map is electricity's."""
    data = ctx.carriers[carrier]
    per_area, _, facts = data["transfer"]
    zones = ctx.zones
    text = (f"{plural(facts['pairs'], 'corridor')} carry {_num(facts['total_GW'])} GW of "
            f"{data['title'].lower()}, each the mean of its two directions.")
    if facts["inside"]:
        text += (f" {facts['inside']} of them {'links' if facts['inside'] == 1 else 'link'} two "
                 f"nodes inside one {level_header(zones)} and {'counts' if facts['inside'] == 1 else 'count'} "
                 f"as internal.")
    report.add(text)
    report.add()
    total = (per_area["cross_border_MW"] + per_area["inter_zonal_MW"]).sort_values(ascending=False)
    rows = [[display_name(area, zones), _num(total[area] * MW_TO_GW),
             _num(per_area.loc[area, "cross_border_MW"] * MW_TO_GW)
             if per_area.loc[area, "cross_border_MW"] else "-",
             _num(per_area.loc[area, "inter_zonal_MW"] * MW_TO_GW)
             if per_area.loc[area, "inter_zonal_MW"] else "-"]
            for area in total.index]
    report.table([level_header(zones), "total GW", "cross-border GW", "internal GW"], rows)


def _duration_table(report: Report, ratio_facts: Dict, grids: Sequence[str], what: str) -> None:
    """The storage sized as hours of discharge power, one row per grid."""
    by_grid = {g: e for g, e in (ratio_facts or {}).get("by_grid", {}).items() if g in set(grids)}
    if not by_grid:
        return
    report.add(f"{what} sized as hours of discharge power (`upperLimitCapacityRatio` in "
               f"`p_gnu_io`), by grid:")
    report.add()
    rows = []
    for grid in sorted(by_grid, key=lambda g: -by_grid[g]["TWh"]):
        entry = by_grid[grid]
        hours = (_num(entry["hours"], 0) if entry["hours"] is not None
                 else summarise([_num(h, 0) for h in entry["hours_seen"]]))
        rows.append([f"`{grid}`", hours, _num(entry["power_GW"]),
                     _num(entry["TWh"] * 1000, 0), str(entry["nodes"])])
    if len(rows) > 1:
        rows.append(["**total**", "", "",
                     f"**{_num(sum(e['TWh'] for e in by_grid.values()) * 1000, 0)}**", ""])
    report.table(["grid", "hours", "power GW", "energy GWh", "nodes"], rows)


def _storage_gaps(report: Report, ctx: ReportContext, grids: Sequence[str]) -> None:
    """What the volumes above leave out, for these grids only."""
    facts, ratio_facts = ctx.inventory.facts, ctx.inventory.ratio_facts or {}
    grids = set(grids)
    io = ctx.workbook.p_gnu_io
    in_grids = (set(io.loc[io["grid"].astype(str).isin(grids), "node"].astype(str))
                if not io.empty and {"grid", "node"} <= set(io.columns) else set())
    unconvertible = [n for n in ratio_facts.get("unconvertible", []) if n in in_grids]
    if unconvertible:
        report.add(f"{plural(len(unconvertible), 'storage node')} state a duration but no "
                   f"`energyStoredPerUnitOfState`, so their energy is not known and is left out: "
                   f"{summarise(unconvertible)}.")
        report.add()
    empty = [g for g in facts.get("grids_without_energy", []) if g in grids]
    if empty:
        report.add(f"**{', '.join(empty)}** have discharge power but no stated energy -- no "
                   f"`upwardLimit` and no `upperLimitCapacityRatio` -- so their volume is not in "
                   f"the totals above.")
        report.add()
    both = [n for n in facts.get("both_routes", []) if n in in_grids]
    if both:
        report.add(f"{plural(len(both), 'storage node')} state a size both ways; each is counted once, "
                   f"by its `upwardLimit`: {summarise(both)}.")
        report.add()


def _carrier_storage_subsection(report: Report, ctx: ReportContext, carrier: str) -> None:
    """The storage a carrier other than electricity charges and discharges, by area."""
    data = ctx.carriers[carrier]
    zones = ctx.zones
    table, grids = data["storage"], data["storage_grids"]
    discharge = storage_discharge(table)
    report.add(f"{_num(float(table['energy_GWh'].sum()), 0)} GWh of storage in "
               f"{plural(int((discharge > 0).sum()), level_header(zones))}, with "
               f"{_num(float(discharge.sum()))} GW of discharge and "
               f"{_num(float(table['charge_GW'].sum()))} GW of charging, all of it on "
               f"{data['title'].lower()}.")
    report.add()
    _duration_table(report, ctx.inventory.ratio_facts, grids, "Storage")
    rows = [[display_name(area, zones), _num(discharge[area]),
             _num(table.loc[area, "charge_GW"]), _num(table.loc[area, "energy_GWh"], 0)]
            for area in table["energy_GWh"].sort_values(ascending=False).index]
    report.table([level_header(zones), "discharge GW", "charge GW", "energy GWh"], rows)
    _storage_gaps(report, ctx, grids)


def _elec_storage_subsection(report: Report, ctx: ReportContext) -> List[str]:
    """Electricity storage, the nested group with inflow hydro, and the pumps."""
    data = ctx.carriers["elec"]
    inventory, zones = ctx.inventory, ctx.zones
    table, grids = data["storage"], data["storage_grids"]
    groups = inventory.by_group
    widest = [key for key, _, _ in STORAGE_GROUPS][-1]
    has_groups = len(groups) and float(groups["usable_mean_TWh"].max()) > 0
    if (table is None or table.empty) and not has_groups:
        report.add("No electricity storage is built in this scenario.")
        report.add()
        return []

    labels = [c for c in (table.columns if table is not None else [])
              if c not in ("charge_GW", "energy_GWh")]
    discharge = storage_discharge(table)
    if table is not None and not table.empty:
        report.add(f"Electricity storage -- {GROUP_GLOSS['elec_literal']} -- holds "
                   f"{_num(float(table['energy_GWh'].sum()), 0)} GWh, with "
                   f"{_num(float(discharge.sum()))} GW of discharge and "
                   f"{_num(float(table['charge_GW'].sum()))} GW of charging."
                   + (f" Counting the storage the weather fills as well, "
                      f"{_num(float(groups.loc[widest, 'usable_mean_TWh']))} TWh is usable."
                      if has_groups else ""))
        report.add()
    if has_groups:
        rows = []
        for key, label, _ in STORAGE_GROUPS:
            if key not in groups.index:
                continue
            row = groups.loc[key]
            rows.append([label, _num(row["usable_mean_TWh"]), _num(row["usable_envelope_TWh"]),
                         _num(row["nameplate_TWh"])])
        report.table(["group", "usable TWh", "seasonal envelope TWh", "nameplate TWh"], rows)
        report.add("; ".join(f"**{label}** is {GROUP_GLOSS[key]}"
                             for key, label, _ in STORAGE_GROUPS if key in GROUP_GLOSS) + ". "
                   "The three measures differ only where the bounds follow the season; "
                   "[Storage volume](#hydro-volume) says how. Inflow hydro discharges "
                   "through its turbines, which are under Generation as hydro, so its power is "
                   "not repeated here.")
        report.add()
        if ctx.timeseries.skipped:
            report.add(f"Inflow hydro volumes need the timeseries, which were not read: "
                       f"{ctx.timeseries.skipped}. The totals with it are incomplete.")
            report.add()

    # Per area: the storage the power system fills, then the inflow hydro it
    # sits beside -- its volume and the pumps that put water back.
    hydro_pumps = storage_side_power_by_node(ctx.workbook, set(ctx.hydro_set) - set(grids),
                                             "output")
    pumps = (hydro_pumps.assign(area=[area_of(n, zones) for n in hydro_pumps["node"]])
             .groupby("area", observed=True)["power_MW"].sum() * MW_TO_GW
             if not hydro_pumps.empty else pd.Series(dtype="float64"))
    with_hydro = (inventory.by_area["usable_mean_TWh"].xs(widest, level="group") * 1000.0
                  if len(inventory.by_area) else pd.Series(dtype="float64"))
    base = table if table is not None and not table.empty else pd.DataFrame(
        columns=["charge_GW", "energy_GWh"], dtype="float64")
    areas = base.index.union(with_hydro[with_hydro > 0].index).union(pumps[pumps > 0].index)
    if len(areas):
        order = sorted(areas, key=lambda a: -(float(with_hydro.get(a, 0.0))
                                              + float(base["energy_GWh"].get(a, 0.0))))
        rows = []
        for area in order:
            row = [display_name(area, zones)]
            row += [_num(base.loc[area, l]) if area in base.index and base.loc[area, l] else "-"
                    for l in labels]
            row += [_num(base.loc[area, "charge_GW"]) if area in base.index
                    and base.loc[area, "charge_GW"] else "-",
                    _num(base.loc[area, "energy_GWh"], 0) if area in base.index
                    and base.loc[area, "energy_GWh"] else "-",
                    _num(float(with_hydro.get(area, 0.0)), 0) if with_hydro.get(area, 0.0) else "-",
                    _num(float(pumps.get(area, 0.0))) if pumps.get(area, 0.0) else "-"]
            rows.append(row)
        rows.append(["**total**"] + [f"**{_num(float(base[l].sum()))}**" for l in labels]
                    + [f"**{_num(float(base['charge_GW'].sum()))}**",
                       f"**{_num(float(base['energy_GWh'].sum()), 0)}**",
                       f"**{_num(float(with_hydro.sum()), 0)}**",
                       f"**{_num(float(pumps.sum()))}**"])
        report.add("Discharge and charging power, GW, read on each storage's own rows, and usable "
                   "energy, GWh. The last two columns are the inflow hydro beside it: the "
                   "usable volume with it added, and the pumps that put water back.")
        report.add()
        report.table([level_header(zones)] + [f"{l} GW" for l in labels]
                     + ["charging GW", "storage GWh", "with inflow hydro GWh",
                        "inflow hydro pumps GW"], rows)
    _duration_table(report, inventory.ratio_facts, grids, "Electricity storage")
    _storage_gaps(report, ctx, grids)

    power = table[labels] if labels else pd.DataFrame()
    name = figure_storage(ctx.out_dir, inventory, zones, power)
    report.figure(
        name,
        "Left: the discharge power of electricity storage, with each duration-sized "
        "technology's hours in the legend. Right: usable storage energy on a log scale, so "
        "batteries and reservoirs show on one axis -- the narrow dark bar is electricity "
        "storage, the wide light bar the total with inflow hydro, and the tick each area's "
        "nameplate ceiling. Take-away: a handful of countries carry the system's seasonal energy "
        "in hydro, the rest hold hours rather than months.",
    )
    return [name]


def _demand_management_subsection(report: Report, ctx: ReportContext) -> None:
    """Demand response by area and tier, against each area's own peak."""
    table, costs, zones = ctx.demand_response, ctx.demand_response_costs, ctx.zones
    if table is None or table.empty:
        report.add("No demand response is built in this scenario.")
        report.add()
        return

    def cost(tier: str) -> str:
        low, high = costs[tier]
        return (f"{_num(low, 0)} EUR/MWh" if round(high) - round(low) == 0
                else f"{_num(low, 0)}-{_num(high, 0)} EUR/MWh")

    total = table.sum(axis=1)
    tiers = list(table.columns)
    priced = [f"`{t}` at {cost(t)}" for t in tiers if t in costs]
    report.add(f"{_num(float(total.sum()))} GW of demand response in "
               f"{plural(int((total > 0).sum()), level_header(zones))}: demand Backbone may "
               f"leave unserved at a price, written as units that make electricity from nothing "
               f"(`isSource` in `p_unit`), one per zone and tier"
               + (f". The price is each tier's `vomCosts`: {and_join(priced)}." if priced else "."))
    report.add()
    peak = ctx.peak
    rows = []
    for area in total.sort_values(ascending=False).index:
        share = (f"{_num(100.0 * total[area] / (peak[area] * MW_TO_GW), 0)}%"
                 if peak is not None and area in peak.index and peak[area] > 0 else "-")
        rows.append([display_name(area, zones)]
                    + [_num(table.loc[area, t]) if table.loc[area, t] else "-" for t in tiers]
                    + [f"**{_num(total[area])}**", share])
    rows.append(["**total**"] + [f"**{_num(float(table[t].sum()))}**" for t in tiers]
                + [f"**{_num(float(total.sum()))}**", ""])
    report.table([level_header(zones)] + [f"`{t}` GW" for t in tiers]
                 + ["total GW", "share of peak demand"], rows)
    report.add("`share of peak demand = demand response / the area's own highest hour of "
               "electricity demand`")
    report.add()


def level_header(zones: bool) -> str:
    return "zone" if zones else "country"


def _forecast_text(config: Dict, files: Sequence[str]) -> str:
    """The forecast branches as the build set them up, one clause per branch."""
    if not files:
        return "none written: every solve sees only the realised weather"
    quantiles = config.get("energy_quantiles") or {}
    weights = config.get("forecast_weights") or {}
    shapes = config.get("forecast_branches") or {}
    parts = []
    for label in sorted(quantiles):
        head = f"{label} at the {quantiles[label]:g} quantile, weight {weights.get(label, 0):g}"
        shape = shapes.get(label)
        if not isinstance(shape, dict):
            parts.append(f"{head}, the central branch to the horizon")
            continue
        end = str(shape.get("end", "cut"))
        then = {"cut": "then ends",
                "bound": "then ends at the central branch's storage levels",
                "continue": "then continues on the central branch's data"}.get(end, f"then {end}")
        blend = shape.get("blend_days") or 0
        if end == "continue" and blend:
            then += f", blending over {blend:g} days"
        parts.append(f"{head}, {shape.get('length_days', '?')} days, {then}")
    text = "; ".join(parts) if parts else "written, with no branch recorded"
    overrides = [f"{name} {', '.join(f'{k} at {v:g}' for k, v in spec['energy_quantiles'].items())}"
                 for name, spec in sorted((config.get("timeseries_specs") or {}).items())
                 if isinstance(spec, dict) and spec.get("energy_quantiles")]
    if overrides:
        text += ". Series with their own quantiles: " + "; ".join(overrides)
    return text


def _window_text(config: Dict, timeseries: Timeseries) -> str:
    """How many climate windows, how long, and where each starts."""
    years = timeseries.years
    hours = timeseries.window_hours or int(config.get("bb_timeseries_length") or 0) * 24
    if not years:
        return "no per-year timeseries file was found"
    start = config.get("bb_timeseries_start") or "01-01"
    text = (f"{plural(len(years), 'window')} of {hours / 24:,.4g} days, one starting on "
            f"{start} of each year {min(years)}-{max(years)}")
    if hours and hours < HOURS_PER_YEAR:
        text += (". **Shorter than a year, so the annual figures, net load and climate-year "
                 "comparison are left out of this report**")
    elif hours > HOURS_PER_YEAR:
        text += (". Longer than a year, so neighbouring windows overlap: the annual figures "
                 "use the first 365 days of each, and the hydro checks the whole window")
    return text


def _build_section(report, record: BuildRecord, workbook: Workbook,
                   timeseries: Timeseries) -> None:
    """Which config and variants built the folder, and the settings that shape the data."""
    config = record.config
    rows = []
    if record.config_file:
        rows.append(["Config", f"`{record.config_file}`"])
    else:
        rows.append(["Config", "no config in `src_files/` names this folder"])
    rows.append(["Build", f"`{workbook.build}`" if workbook.build
                 else "the default build (no variant suffix)"])

    status = []
    if record.built_at:
        status.append(record.built_at)
    causes = record.flags.get("partial_causes") or []
    if causes:
        status.append(f"**PARTIAL**: {summarise(causes)}")
    elif record.flags.get("workflow_run_successfully") is False:
        status.append("**the build logged errors**")
    elif record.flags:
        status.append("complete")
    if record.warnings or record.errors:
        notes = [plural(record.warnings, "warning")] if record.warnings else []
        notes += [plural(record.errors, "error")] if record.errors else []
        status.append(f"{' and '.join(notes)} in `summary.log`")
    rows.append(["Built", ", ".join(status) if status else "not recorded"])

    if record.variant_lines:
        # "Variant 'dh-level': capitals in every country." -> "`dh-level`: capitals in every country"
        variants = "; ".join(re.sub(r"^Variant '([^']+)'", r"`\1`", line).rstrip(".")
                             for line in record.variant_lines)
        choice = config.get("variant_choice") or {}
        if not choice:
            variants += ". This build chooses none, so each variant takes its default"
        rows.append(["Variants", variants])
    else:
        rows.append(["Variants", "none declared in the source data"])

    zones = sorted({zone_of(n) for n in workbook.nodes})
    where = (f"{plural(len(zones), 'bidding zone')} in "
             f"{plural(len({country_of(z) for z in zones}), 'country')}")
    excluded = [*(config.get("exclude_grids") or []), *(config.get("exclude_nodes") or [])]
    if excluded:
        where += f"; left out by the config: {summarise(excluded)}"
    rows.append(["Areas", where])
    rows.append(["Climate windows", _window_text(config, timeseries)])
    rows.append(["Forecast branches", _forecast_text(config, record.forecast_files)])
    if config.get("bb_horizon_weeks"):
        rows.append(["Horizon", f"{config['bb_horizon_weeks']} weeks ahead of each solve"])
    report.table(["Setting", "Value"], rows)
    if not config:
        report.add("`cache/config_structural.json` is missing, so the settings the build used "
                   "are not known beyond what the data shows.")
        report.add()

    changing = investment_and_timing(workbook)
    if changing:
        report.add(f"**This report does not handle investment or changing fleets yet.** In this "
                   f"build, {'; '.join(changing)}. Every capacity here is the starting fleet; "
                   f"what Backbone may add, and when units come and go, is not in these "
                   f"numbers.")
        report.add()


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
         f"{facts['bound_start_nodes']} of {facts['nodes']}",
         "`p_gn` `boundStart`, `boundStartRelative`"],
    ]
    report.table(["What", "Count", "Where it is read from"], rows)
    meaning = []
    if not facts["ramp_limited_rows"]:
        meaning.append("with no ramp limit, any unit may go from nothing to full output in one "
                       "hour")
    else:
        meaning.append("a unit with no ramp limit may go from nothing to full output in one hour")
    if not facts["commitment_rows"]:
        meaning.append("with no unit-commitment rows, no unit has a minimum load or a start cost")
    if facts["derated_units"]:
        meaning.append(f"the capacity tables are plate ratings; {plural(facts['derated_units'], 'unit')} "
                       f"can deliver only part of theirs")
    else:
        meaning.append("with availability 1 everywhere, the capacity tables are plate ratings "
                       "with no outage derating")
    report.add("These are counts, not findings: " + "; ".join(meaning) + ".")
    report.add()


def _coverage_section(report, presence: pd.DataFrame, zone_presence: pd.DataFrame,
                     shapes: ZoneShapes, zones: bool) -> None:
    """What the build covers, in two counts and a list of what it does not.

    Absence is not a defect here and nothing below calls it one: an area with no
    district heating is stating what the model contains, not a gap to fix. The
    omissions are listed at the end all the same, because a reader who wants to
    know whether Spain has district heat should not have to read a map to find
    out that it does not.

    Both counts always, whatever level the report is written at. A country total
    with no zone total behind it hides that Finland's district heat is in four
    zones of five, and a zone total alone is not the number anyone quotes.
    """
    if presence.empty:
        report.add("No nodes are written in this scenario.")
        report.add()
        return

    titles = {grid: title for grid, title, _ in CARRIERS}
    by_country = presence_by_country(zone_presence) if not zone_presence.empty else presence
    countries, zone_areas = len(by_country), len(zone_presence)

    report.add(f"The input data covers {plural(countries, 'country')}, "
               f"{plural(zone_areas, 'bidding zone')}.")
    report.add()
    for carrier in presence.columns:
        in_countries = int((by_country[carrier] != PRESENCE_ABSENT).sum())
        in_zones = int((zone_presence[carrier] != PRESENCE_ABSENT).sum())
        report.add(f"- {titles.get(carrier, carrier).lower()} is modelled for "
                   f"{in_countries}/{countries} countries, {in_zones}/{zone_areas} bidding zones")
    report.add()

    if not shapes.available:
        report.add(f"There is no map in this report: {shapes.missing}.")
        report.add()


def _omissions_section(report, presence: pd.DataFrame, zones: bool) -> None:
    """Which areas a carrier skips, and where it is built with nothing to serve.

    One row per carrier that has something to say. A carrier modelled everywhere
    and demanded everywhere is not listed, because a line reading "- none, -"
    asks the reader to check that it is empty every time they read the report.
    """
    if presence.empty:
        return
    titles = {grid: title for grid, title, _ in CARRIERS}
    level = level_header(zones)

    def name(areas: list, out_of: int) -> str:
        """The areas, or the word for all of them. A list of sixteen is not a list."""
        if not areas:
            return "-"
        if len(areas) == out_of:
            return f"every {level}"
        return summarise(areas)

    rows = []
    for carrier in presence.columns:
        column = presence[carrier]
        absent = sorted(column.index[column == PRESENCE_ABSENT])
        hollow = sorted(column.index[column == PRESENCE_NODE])
        unread = sorted(column.index[column == PRESENCE_UNKNOWN])
        if not (absent or hollow or unread):
            continue
        cells = [name(absent, len(column)), name(hollow, len(column) - len(absent))]
        if unread:
            note = f"demand not read for {len(unread)}"
            cells[1] = note if cells[1] == "-" else f"{cells[1]}; {note}"
        rows.append([titles.get(carrier, carrier)] + cells)

    if not rows:
        report.add(f"Every carrier is modelled in every {level} and something demands each one.")
        report.add()
        return

    report.add("Energy carriers not fully modelled:")
    report.add()
    report.table(["carrier", f"not modelled ({level})", "modelled, nothing demands it"], rows)


#: How each route states a storage's size, said for a reader.
_SIZED_BY = {
    "timeseries": "seasonal bounds",
    "workbook constant": "a constant ceiling",
    "duration in p_gnu_io": "hours of discharge power",
}


def _hydro_chapter(report: Report, ctx: ReportContext) -> None:
    """Hydro from the water's side: what each storage holds, receives, and must give."""
    report.heading("hydro", "Hydro")
    table, checks, timeseries, zones = (ctx.hydro_table, ctx.hydro_result, ctx.timeseries,
                                        ctx.zones)
    if table.empty:
        report.add("No hydro is built in this scenario.")
        report.add()
        return
    read = timeseries.hydro is not None
    opening = (f"Hydro as water: {_num(float(table['storage_MWh'].sum()) * MWH_TO_TWH)} TWh of "
               f"usable storage in {plural(table['area'].nunique(), level_header(zones))}")
    if read:
        opening += (f", receiving {_num(float(table['inflow_MWh_yr'].sum()) * MWH_TO_TWH, 0)} TWh "
                    f"of inflow in a mean climate year")
    report.add(opening + ". The turbines are under Electricity as generation and the pumps "
               "under its Storage; this chapter is what the water allows them.")
    report.add()

    report.heading("hydro-storage", "Storage and inflow", 3)
    report.add(f"Every hydro type by {level_header(zones)}, as the build states it: how much it "
               f"stores, what its turbines can draw and must generate, what its pumps put back, "
               f"and the water it receives.")
    report.add()
    derated = bool(((table["output_MW"] - table["max_gen_MW"]).abs() > 0.5).any()
                   or table["timeseries_availability"].any())
    headers = [level_header(zones), "type", "storage GWh", "turbine MW", "min generation MW"]
    if derated:
        headers.append("max generation MW")
    headers += ["pump MW", "inflow TWh/yr", "full-load h", "storage weeks"]
    rows = []
    for r in table.itertuples():
        row = [display_name(r.area, zones), f"`{r.grid}`",
               _num(r.storage_MWh / 1e3, 0) if np.isfinite(r.storage_MWh) else "-",
               _num(r.turbine_MW, 0), _num(r.min_gen_MW, 0) if r.min_gen_MW else "-"]
        if derated:
            row.append("timeseries" if r.timeseries_availability else _num(r.max_gen_MW, 0))
        row += [_num(r.pump_MW, 0),
                _num(r.inflow_MWh_yr * MWH_TO_TWH, 2) if read else "-",
                _num(r.flh, 0) if np.isfinite(r.flh) else "-",
                (_num(r.storage_weeks, 1 if r.storage_weeks >= 1 else 2)
                 if np.isfinite(r.storage_weeks) else "-")]
        rows.append(row)
    report.table(headers, rows)
    report.add("- **storage GWh**: the usable volume, the mean of `upwardLimit` minus "
               "`downwardLimit` over the year; [Storage volume](#hydro-volume) has the other "
               "two measures.")
    report.add("- **turbine MW** and **pump MW**: what the turbines can draw out of the storage "
               "and what the pumps can put into it, in water, on the storage's own row.")
    report.add("- **min generation MW**: the `minGen` the turbines must keep generating, in MW "
               "of electricity.")
    if derated:
        report.add("- **max generation MW**: the turbines' output capacity times their "
                   "`availability`, shown because some turbine's availability is below 1; "
                   "'timeseries' where it comes from a series this report does not read.")
    report.add("- **full-load h**: mean annual inflow over turbine MW -- the hours the turbines "
               "would run at full draw to pass one average year's water. Zero means no inflow.")
    report.add("- **storage weeks**: the usable volume over a mean week of inflow -- how many "
               "average weeks of water the storage can hold. Below 1 it is a pond that "
               "passes its water on as it arrives; tens of weeks is a reservoir that can carry water "
               "from one season into the next. '-' when no water arrives.")
    report.add()
    if not read:
        report.add(f"The inflow columns need the hydro timeseries, which were not read"
                   + (f": {timeseries.skipped}" if timeseries.skipped else "") + ".")
        report.add()
    overflow = checks.overflow
    over = overflow[overflow["state"] == CHECK_ACT] if len(overflow) else overflow
    if len(over):
        report.add(f"**{plural(len(over), 'storage node')} overflow even with turbines and `maxSpill` "
                   f"at full**: "
                   + summarise([f"`{r.node}` {_num(r.over_GWh, 1)} GWh in "
                                f"{plural(r.windows_over, 'window')}"
                                for r in over.itertuples()])
                   + ". Water that cannot leave is met by a dummy variable or makes the problem "
                     "infeasible.")
        report.add()

    report.heading("hydro-volume", "Storage volume", 3)
    _hydro_volume(report, ctx)

    report.heading("hydro-floor", "Closest to floor", 3)
    _hydro_floor(report, checks)


def _hydro_volume(report: Report, ctx: ReportContext) -> None:
    """Each hydro type sized three ways, and which route states each size."""
    nodes = ctx.inventory.by_node
    facts = ctx.inventory.facts
    hydro = nodes[nodes["grid"].astype(str).isin(ctx.hydro_set)] if len(nodes) else nodes
    report.add("Storage whose bounds follow the season has no single size, so each is measured "
               "three ways. **Usable** is the mean of `upwardLimit` minus `downwardLimit` over "
               "the year: the room to move on an average hour. The **seasonal envelope** is "
               "`max(upwardLimit)` minus `min(downwardLimit)`, the widest swing those bounds "
               "permit across a year, and it is the one to hold against a seasonal demand. "
               "**Nameplate** is the ceiling on its own.")
    report.add()
    if len(hydro):
        rows = []
        for grid, part in sorted(hydro.groupby(hydro["grid"].astype(str), observed=True),
                                 key=lambda kv: -float(kv[1]["usable_mean_MWh"].sum())):
            sources = and_join([_SIZED_BY.get(str(s), str(s))
                                for s in sorted(part["source"].astype(str).unique())])
            rows.append([f"`{grid}`", str(len(part)),
                         _num(float(part["usable_mean_MWh"].sum()) * MWH_TO_TWH, 2),
                         _num(float(part["usable_envelope_MWh"].sum()) * MWH_TO_TWH, 2),
                         _num(float(part["nameplate_MWh"].sum()) * MWH_TO_TWH, 2), sources])
        rows.append(["**total**", f"**{len(hydro)}**",
                     f"**{_num(float(hydro['usable_mean_MWh'].sum()) * MWH_TO_TWH, 2)}**",
                     f"**{_num(float(hydro['usable_envelope_MWh'].sum()) * MWH_TO_TWH, 2)}**",
                     f"**{_num(float(hydro['nameplate_MWh'].sum()) * MWH_TO_TWH, 2)}**", ""])
        report.table(["type", "storage nodes", "usable TWh", "seasonal envelope TWh", "nameplate TWh",
                      "sized by"], rows)
        usable = float(hydro["usable_mean_MWh"].sum())
        ceiling = float(hydro["nameplate_MWh"].sum())
        if usable and ceiling > usable * 1.05:
            report.add(f"Here the nameplate is {_num(ceiling / usable)} times the usable volume: "
                       f"it counts reservoir space that the seasonal floor never lets go of.")
            report.add()
        seasonal = [n for n in facts.get("timeseries_nodes", [])
                    if n in set(hydro["node"].astype(str))]
        if seasonal:
            report.add(f"Seasonal bounds come from `ts_node_hydro_storage_limits`, for "
                       f"{plural(len(seasonal), 'storage node')}: {summarise(seasonal)}. A constant "
                       f"ceiling is `p_gnBoundaryPropertiesForStates`, with no floor.")
            report.add()
    if ctx.timeseries.skipped:
        report.add(f"Storage energy could only be read from the workbook: "
                   f"{ctx.timeseries.skipped}. The numbers above are incomplete, not totals.")
        report.add()
    if facts.get("timeseries_missing"):
        report.add(f"{plural(len(facts['timeseries_missing']), 'storage node')} take their bounds from "
                   f"a timeseries that was not read, so they are missing from the totals above: "
                   f"{summarise(facts['timeseries_missing'])}.")
        report.add()


def _hydro_floor(report: Report, checks: "HydroChecks") -> None:
    """How close each storage node with a minimum generation comes to its floor."""
    if checks.minimums == 0:
        report.add("No hydro storage has a minimum generation in this scenario.")
        report.add()
        return
    report.add(f"Each storage node with a minimum generation, followed hour by hour through every "
               f"climate window: it gives exactly its minimum, at its turbines' efficiency, and "
               f"spills whatever rises above its ceiling. **Closest to floor** is the least water "
               f"it ever has left above its floor, in hours of that minimum. Below the floor the "
               f"storage runs short, and Backbone meets those hours only by paying `minGenPenalty`. "
               f"A storage node is watched when it spends {MIN_GEN_WATCH_HOURS} or more hours of one "
               f"window within {MIN_GEN_MARGIN_HOURS:g} hours of its floor, or runs short only "
               f"without pumping; it is acted on when it runs short even with its pumps at full, "
               f"or when its minimum is more than its turbines can give.")
    report.add()
    if checks.skipped:
        report.add(f"The water could not be followed: {checks.skipped}.")
        report.add()

    minimum = checks.minimum
    if len(minimum) and "state" in minimum.columns:
        severity = {CHECK_ACT: 0, CHECK_WATCH: 1, CHECK_OK: 2}
        ordered = minimum.assign(_s=minimum["state"].map(severity)).sort_values(
            ["_s", "closest_hours"])
        rows = []
        for r in ordered.itertuples():
            short = r.short_hours > 0
            state = f"**{r.state}**" if r.state == CHECK_ACT else r.state
            pumps = _num(r.pump_MW, 0) if r.pump_MW else "-"
            if short and r.pump_MW:
                pumps += ", still short" if r.short_with_pumps else ", enough"
            rows.append([
                f"`{r.node}`", _num(r.minimum_MW, 0),
                f"**{_num(r.deliverable_MW, 0)}**" if r.beyond_turbines else _num(r.deliverable_MW, 0),
                "below the floor" if short else
                (f"{_num(r.closest_hours, 0)} h ({_num(r.closest_MWh / 1e3, 1)} GWh)"
                 if np.isfinite(r.closest_hours) else "-"),
                str(r.closest_year) if np.isfinite(r.closest_MWh) else "-",
                _num(r.near_hours_worst, 0),
                f"{r.windows_short} of {r.windows_run}", _num(r.short_GWh, 1),
                pumps, state,
            ])
        report.table(["storage node", "minimum MW", "turbines give MW", "closest to floor", "in",
                      f"hours within {MIN_GEN_MARGIN_HOURS:g} h of floor", "windows short",
                      "short GWh", "pumps MW", "state"], rows)
        report.add("**Turbines give** is their capacity times availability; **in** is the climate "
                   "window where the storage came closest.")
        report.add()
    if checks.not_simulated:
        report.add(f"{plural(len(checks.not_simulated), 'minimum')} cannot be followed this way "
                   f"and are left out: "
                   + summarise([f"`{name}` ({why})" for name, why in checks.not_simulated]) + ".")
        report.add()


def _transfer_subsection(report: Report, ctx: ReportContext) -> List[str]:
    """Electricity corridors, as a table and a map."""
    per_area, matrix, facts = ctx.carriers["elec"]["transfer"]
    peak, zones = ctx.peak, ctx.zones
    if per_area is None or per_area.empty:
        report.add("No electricity transfer capacity is written in this scenario.")
        report.add()
        return []
    report.add(f"{plural(facts['pairs'], 'corridor')} carry {_num(facts['total_GW'])} GW in "
               f"total. A corridor has a capacity in each direction, and the tables use the "
               f"mean of the two:")
    report.add()
    report.add("`corridor capacity = (the A-to-B capacity + the B-to-A capacity) / 2`")
    report.add()
    if facts["asymmetric"]:
        largest = ", ".join(f"{name} {_num(spread, 0)} MW"
                            for name, spread in facts["largest_asymmetries"])
        report.add(f"{facts['asymmetric']} of {plural(facts['pairs'], 'corridor')} have a "
                   f"different capacity each way; the largest differences are {largest}.")
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

    name = figure_interconnection(ctx.out_dir, matrix, per_area, peak, ctx.zone_matrix,
                                  ctx.shapes, ctx.countries, ctx.neighbours)
    report.figure(
        name,
        "Transfer capacity as a map, and each area's total as a share of its own peak demand. "
        "Take-away: who is connected to whom, and who depends on those connections rather than "
        "on its own plant.",
    )
    return [name]


def _netload_subsection(report: Report, ctx: ReportContext) -> List[str]:
    """Demand less wind and solar, hour by hour, and what is left to meet its peak."""
    timeseries, zones = ctx.timeseries, ctx.zones
    elec = ctx.carriers["elec"]
    curves = netload_by_area(timeseries) if timeseries.available else {}
    if not curves:
        report.add(f"Net load needs the hourly timeseries, which were not read"
                   + (f": {timeseries.skipped}" if timeseries.skipped else "") + ".")
        report.add()
        return []
    per_area_transfer = elec["transfer"][0]
    split = firm_and_shifting(elec["capacity"])
    imports = (per_area_transfer["cross_border_MW"] if per_area_transfer is not None
               and not per_area_transfer.empty else pd.Series(dtype="float64"))
    vre_share = potential_vre_share(elec["capacity_factor"], elec["demand"])

    report.add(f"Net load is electricity demand minus wind and solar output, hour by hour, over "
               f"all {plural(len(timeseries.years), 'climate year')} pooled -- "
               f"{len(next(iter(curves.values()))):,} hours per area. It is a screening view of "
               f"the input data, kept deliberately simple; what it leaves out is listed under "
               f"the table.")
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
    report.add("The last column sets firm generation, the storage and demand response that can "
               "shift into that hour, and the whole import capacity against the peak. Negative "
               "means the area cannot cover its own worst hour from those three alone, which is "
               "what interconnection and a wider system are for. The system row adds the areas "
               "hour by hour, so it already accounts for peaks that do not coincide, and its "
               "imports are zero because trade between members cancels inside it.")
    report.add()
    if not vre_share.empty:
        report.add("VRE energy over demand is installed wind and solar at their own "
                   "capacity-weighted factors across a year, against mean annual demand. Below "
                   "about 80% an area is short of energy; above 100% it could in principle make "
                   "its whole year from wind and sun and its problem is timing.")
        report.add()

    chp = sum((elec["links"] or {}).get("co_by_carrier", {}).values())
    derated = model_permissions(ctx.workbook)["derated_units"]
    report.add("**What this table leaves out.** Each of these is a reason the left-over column "
               "can be wrong, and none of them is small:")
    report.add()
    report.add("- **Wind and solar** are capacity times the capacity factor in the build: never "
               "curtailed, never out of service.")
    report.add("- **Firm capacity** is plate rating, with no outages"
               + (f" and no derating, though {plural(derated, 'unit')} have availability below 1"
                  if derated else "")
               + ". Hydro counts at full turbine capacity whatever water it has.")
    if chp:
        report.add(f"- **Combined heat and power**, {_num(chp)} GW of the firm capacity, counts "
                   f"at full electrical output. What it can give in an hour depends on the heat "
                   f"or steam it must make then, and on how freely its two outputs may move "
                   f"apart.")
    report.add("- **Storage and demand response** count at full power, however empty the storage "
               "or long the event.")
    report.add("- **Imports** count every corridor at full, as if every neighbour had power to "
               "spare in the same hour.")
    report.add("- **Nothing is dispatched**: no ramp limits, no reserves, no forecast error.")
    report.add()
    report.add("Whether an area covers its peak is something only a run of Backbone can say.")
    report.add()

    name = figure_netload(ctx.out_dir, curves, zones)
    report.figure(
        name,
        "Net load -- demand minus wind and solar -- for every hour of every climate year, "
        "sorted highest to lowest, one panel per area on its own scale, with the system added "
        "hour by hour in the last panel. Orange is the part of "
        "the year when wind and solar alone exceed demand. Take-away: the left edge is what "
        "firm capacity has to cover; the orange area is what has to be stored, exported or "
        "curtailed. It ignores storage, trade and dispatch, and treats VRE as never curtailed "
        "and never out of service, so it is a statement about the input data and not a forecast.",
    )
    return [name]


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


#: The quantities that move between climate years, in the order the chapter
#: lists them.
_VARYING = (("demand_elec", "electricity demand"), ("demand_dheat", "district heat demand"),
            ("vre", "wind and solar output"), ("inflow_hydro", "hydro inflow"))


def _variation_section(report: Report, timeseries: Timeseries, material, zones) -> None:
    """How far single climate years move from the mean the rest of the report uses."""
    if not timeseries.available:
        report.add(f"The climate years were not read"
                   + (f": {timeseries.skipped}" if timeseries.skipped else "") + ".")
        report.add()
        return

    report.add(f"Every chapter above reports a mean climate year. This one is about the years "
               f"themselves: {plural(len(timeseries.years), 'climate year')} "
               f"({min(timeseries.years)}-{max(timeseries.years)}), how far a single year moves "
               f"from that mean, and whether the bad ones arrive together.")
    report.add()
    rows = []
    for key, label in _VARYING:
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
    report.add("The lowest and highest year, as a percentage of each area's own mean; the "
               "typical range is the median across areas:")
    report.add()
    report.table(["quantity", "typical range", "widest area"], rows)
    caveat = window_caveat(timeseries)
    if caveat:
        report.add(f"*{caveat}*")
        report.add()

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
                   f"system-wide and in {agreeing} of {plural(elec['area'].nunique(), 'area')}.")
        report.add()

    correlations = _correlations(timeseries)
    if not correlations.empty:
        report.add("Does a low-resource year fall in the same year as a high-demand year in the "
                   "same area? Pearson correlation of each area's annual anomalies, over the "
                   "climate years:")
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
            report.add(f"With {plural(len(timeseries.years), 'climate year')} there are too few "
                       f"points to say whether any of these correlations differs from noise.")
            report.add()
        else:
            strong_hydro = int((correlations["hydro"].abs() >= floor).sum())
            negative_vre = int((correlations["vre"] <= -floor).sum())
            report.add(
                f"With {plural(len(timeseries.years), 'climate year')} a correlation has to reach "
                f"|r| = {_num(floor, 2)} before it is distinguishable from noise at the usual 5% "
                f"level. Hydro inflow and demand: {strong_hydro} of "
                f"{plural(total_areas, 'area')} clear it. Wind and solar: {negative_vre} of "
                f"{plural(total_areas, 'area')} sit at or below -{_num(floor, 2)}"
                + (", so a low-VRE year does tend to be a high-demand year, though the effect is "
                   "barely resolved and its size should not be read off these numbers. When one "
                   "climate year is picked to run, that is the relationship to pick against."
                   if negative_vre > total_areas / 2 else "."))
            report.add()
    if material:
        left_out = [a for a in timeseries.areas if a not in material]
        report.add(f"The figure below shows the {plural(len(material), 'area')} with at least "
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
            report.add(f"{len(varying)} of {plural(len(prices), 'fuel')} cost different amounts in "
                       f"different zones, and the table below shows each one's median: "
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
    uc = workbook.userconstraint
    if not uc.empty:
        groups = uc["group"].nunique() if "group" in uc.columns else 0
        items.append(f"{plural(groups, 'user constraint group')} over {plural(len(uc), 'row')} "
                     f"(`p_userconstraint`) -- scenario-specific shapes with nothing in common "
                     f"across builds, so they are counted and not interpreted.")
    if not workbook.p_unit.empty:
        items.append("Unit efficiency curves (`p_unit`) -- technology parameters rather than "
                     "facts about countries at annual level.")
    if not workbook.availability_limits.empty:
        items.append("When units become available or unavailable (`utAvailabilityLimits`) -- "
                     "every capacity here is counted as available for the whole run.")
    for item in items:
        report.add(f"- {item}")
    report.add()


def _limits_section(report, workbook, timeseries, inventory, zones) -> None:
    """The limits that apply to the whole report.

    A caveat about one figure sits in small type under that figure, where the
    reader looking at the picture sees it; this section holds the rest.
    """
    report.add("There is no figure or table below zone level, at any flag setting, other than "
               "the hydro checks, which are per storage node because a minimum binds one. There "
               "is no \"Nordics\" or \"CWE\" grouping, because nothing in the tracked source data "
               "defines one. The maps inherit that: a zone's whole fleet sits at one point. "
               "They are always drawn per bidding zone whatever level the tables use, with "
               "country borders over them.")
    report.add()
    if not timeseries.available:
        report.add(f"**The timeseries sections are missing from this report**: "
                   f"{timeseries.skipped}. Annual demand, hydro inflow, net load, peak demand, the "
                   f"climate-year comparison, and every hydro storage volume with a seasonal "
                   f"bound all need those files. Everything read from `inputData.xlsx` -- capacity, "
                   f"interconnection capacity, prices, emissions and the structural checks -- is "
                   f"unaffected.")
        report.add()
    report.add("Nothing here comes from a Backbone run. [Net load](#elec-net-load) lists what its "
               "simple balance leaves out, and capacity times hours is never converted into "
               "energy for dispatchable plant, because the input data cannot support that claim.")
    report.add()
    report.add("The hydro checks see the realised climate years only. Backbone also solves "
               "forecast branches -- per-hour quantiles of inflow, down to a dry one whose "
               "floor `changes.inc` raises by 10 % -- and that is where minimum generation runs "
               "short first, so a storage node this report finds sufficient can still pay a penalty "
               "there. The checks also start each window from the middle of its range, where "
               "Backbone starts a reservoir or psOpen storage at its `relative` share of the band "
               "and any other at its `reference`; they run hour by hour where Backbone may "
               "aggregate hours and take the tightest bound within each step; and they ignore "
               "ramp limits.")
    report.add()


# ============================================================================
# Entry point
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    # The docstring's first line is 'input_data_summary.py -- what it does'. The
    # filename half is already in the usage line above the description, and it is
    # the wrong filename when this is reached through build_input_summary.py.
    headline = __doc__.splitlines()[1].strip().split(' -- ', 1)[-1]
    parser = argparse.ArgumentParser(description=headline)
    parser.add_argument(
        "built_folder", type=Path,
        help="a folder build_input_data.py wrote, e.g. input_ObservedTrends_2030",
    )
    parser.add_argument(
        "--zones", action="store_true",
        help="report the bidding zones instead of rolling them up into countries",
    )
    parser.add_argument(
        "--no-neighbours", action="store_true",
        help="draw only the modelled areas on the maps, without the grey ring of "
             "countries around them",
    )
    parser.add_argument(
        "--out-subdir", default="summary", metavar="NAME",
        help="subfolder of built_folder for report.md and its figures "
             "(default: summary). Each run rewrites those files and leaves any other alone",
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

    try:
        # Always the finest grain the build writes; build_report rolls it up.
        timeseries = read_timeseries(folder, workbook, True)
    except Exception as exc:
        # A mid-sweep failure degrades the same way a missing GAMS install does.
        # The workbook half of the report is still worth writing.
        timeseries = Timeseries(skipped=f"the timeseries read failed "
                                        f"({type(exc).__name__}: {exc})")
    if timeseries.skipped:
        print(f"Timeseries sections skipped: {timeseries.skipped}")

    classification = classify_capacity(workbook, _inflow_grids_from_data(timeseries))

    shapes = load_zone_shapes(ZONE_ASSET)
    countries = load_zone_shapes(COUNTRY_ASSET)
    if not shapes.available:
        print(f"Maps skipped: {shapes.missing}")

    out_dir = folder / args.out_subdir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        text, figures = build_report(workbook, classification, timeseries, out_dir, args.zones,
                                     shapes, countries, not args.no_neighbours,
                                     read_build_record(folder))
        (out_dir / "report.md").write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write the report into {out_dir}: {exc}")
        return 1

    print(f"{plural(len(figures), 'figure')} written")
    print(f"Report written: {out_dir / 'report.md'}")
    return 0


def _inflow_grids_from_data(timeseries: Timeseries) -> set:
    """Which storage grids actually receive inflow, read from the build.

    Taken from the data rather than from a list of grid names: the scenario
    families do not agree on those names, and a literal set written against one
    of them drops the other's capacity into "no technology group". The inflow
    sweep already saw every grid it carries, so nothing is read twice.
    """
    grids = set(timeseries.hydro.grid_of.values()) if timeseries.hydro is not None else set()
    return grids or set(HYDRO_INFLOW_GRIDS_FALLBACK)


if __name__ == "__main__":
    sys.exit(main())
