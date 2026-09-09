"""
What a source workbook may say, and where it says it.

Two questions about the same sheets, and both are statements about workbook
shape that more than one stage needs and no stage owns.

Which table declares a dimension's *values* -- :func:`unknown_dimension_values`,
asked by the timeseries runner, and the subject of everything down to
:func:`tables_of`.

Which column *names* a table's sheets may carry at all --
:func:`unrecognised_columns`, asked by the source reader, and the subject of the
section at the bottom of the file.

The first question follows.

Backbone's dimensions -- ``grid``, ``node``, ``unit``, ``flow`` -- have to be
declared before they can be used, and a value that nothing declares is not a
failure: it is a name that looks real. A mistyped node quietly becomes a node
nothing balances; a series built for one is written and never read.

This module is the map that lets any stage ask whether a value is declared, and
:func:`unknown_dimension_values` is the question.

Where the declarations are
--------------------------
For ``grid`` and ``node``, in **four tables at once**. `nodedata` and `demanddata`
declare one per row; a `unitdata` row declares one per connection -- a battery
unit brings its own `batterystor` grid and `XX00_batterystor` node into being,
and every fuel grid arrives the same way -- and both ends of a `transferdata`
link count too. That union is exactly what ``BBExcelPipeline._collect_gn_pairs``
assembles, and this map mirrors it deliberately: if the two disagreed, this one
would be reporting values the workbook then writes anyway.

**So there is no reference-versus-declaration split inside the source workbooks.**
Checking one source table's node names against another's reports 110 correct rows
per build and nothing else. The question is worth asking at a boundary where the
producer sits *outside* that set of tables -- which is what a timeseries processor
is, and why ``ProcessorRunner`` asks it.

What is not here
----------------
``emission`` and ``group`` are left out on purpose. Neither is declared by a table
-- an emission is the suffix of a `nodedata` ``emission_XX`` column, and a group is
assembled by the Excel builder from emissions, user constraints and unit groups --
so there is nothing to check a value against. Adding them would mean inventing a
declaration rather than recording one. ``restype`` has no source at all here.

``p_userconstraint``'s four selector slots do refer to values declared elsewhere,
and could be checked. What each slot means depends on the row's own
``parameter``, though -- ``docs/dictionary.md`` gives a dimension contract per
parameter -- so that check needs that table before it can exist.

Cannot tell is not the same as empty
------------------------------------
A frame that is empty means the source excels were skipped this run, not that the
model has no nodes. :func:`known_dimension_values` returns ``None`` for that, and
callers treat it as "no question to ask" -- checking against it would report every
value in the model.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

import src.backbone_params as backbone_params

#: Dimension -> {source table: the columns of it that declare a value}.
#:
#: Tables are named without the ``df_`` prefix, the same vocabulary
#: ``BaseProcessor.requires_source_data`` and the contribution frames use. A
#: column may carry a connection suffix on top of the names given here:
#: ``grid_output1`` declares a grid exactly as ``grid`` does.
DIMENSION_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "grid": {
        "nodedata":     ("grid",),
        "demanddata":   ("grid",),
        "unitdata":     ("grid",),
        "transferdata": ("grid",),
    },
    "node": {
        "nodedata":     ("node",),
        "demanddata":   ("node",),
        "unitdata":     ("node",),
        "transferdata": ("from_node", "to_node"),
    },
    "unit": {"unitdata": ("unit",)},
    "flow": {"unitdata": ("flow",)},
}

#: The connection suffixes a unitdata column may carry. Mirrors what
#: build_unit_grid_and_node_columns produces; see
#: docs/source-workbook-conventions.md.
CONNECTION_SUFFIXES: tuple[str, ...] = tuple(
    f"_{put}{index}" for put in ("input", "output") for index in range(1, 6)
)


def base_column_name(column: str) -> str:
    """`column` without its connection suffix: ``grid_output1`` -> ``grid``."""
    name = str(column).lower()
    for suffix in CONNECTION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def known_dimension_values(
    dimension: str, tables: Mapping[str, pd.DataFrame]
) -> set[str] | None:
    """Every declared value of `dimension`, or ``None`` when nothing could answer.

    ``None`` means the tables that declare this dimension were not loaded, which
    is a different thing from the model having none of it -- see the module
    docstring.
    """
    known: set[str] = set()
    answered = False

    for table, columns in DIMENSION_SOURCES.get(dimension, {}).items():
        frame = tables.get(table)
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            if base_column_name(column) not in columns:
                continue
            known.update(frame[column].dropna().astype(str))
            answered = True

    return known if answered else None


def unknown_dimension_values(
    values,
    dimension: str,
    tables: Mapping[str, pd.DataFrame],
    *,
    also_known: Sequence[str] = (),
) -> list[str]:
    """The values of `dimension` that nothing declares, sorted.

    `also_known` covers a producer that declares the value itself in the same
    breath as using it -- a processor contributing the node it built a series for
    is stating both halves of one sentence, and warning about it would make them
    contradict each other.

    Empty when the question cannot be answered.
    """
    known = known_dimension_values(dimension, tables)
    if known is None:
        return []

    known = known | {str(v) for v in also_known}
    # Iterated rather than put through pd.Series: `values` is a categorical
    # column here and a plain set there, and Series() accepts only one of those.
    used = {str(v) for v in values if pd.notna(v)}
    return sorted(used - known)


def tables_of(source_data_pipeline) -> dict[str, pd.DataFrame]:
    """The ``df_*`` frames a dimension question needs, keyed without the prefix.

    So that a caller holding the pipeline can ask without repeating the
    ``getattr(..., f"df_{name}")`` dance for every table.
    """
    names = {table for sources in DIMENSION_SOURCES.values() for table in sources}

    tables = {}
    for name in sorted(names):
        # `or pd.DataFrame()` would raise here: a DataFrame has no truth value.
        frame = getattr(source_data_pipeline, f"df_{name}", None)
        tables[name] = pd.DataFrame() if frame is None else frame
    return tables


# --- What a column may be named -------------------------------------------
#
# The other half of the same question. Above: which table declares a
# dimension's *values*. Here: which column names a table's sheets may carry at
# all. A column nothing recognises is carried through the source stage and then
# ignored, so a mistyped header is indistinguishable from a deliberate one --
# see report_unused_columns in source_data_loader.py, which asks the question.
#
# The vocabulary comes from three consumers and only one of them is
# discoverable. backbone_params supplies the parameters, so that third cannot
# drift from what the builder writes. The dimension columns belong to this
# stage. The derivation inputs are read by name at a call site and nothing can
# find them automatically, so they are declared by hand below.

#: Columns any source sheet may carry, whatever it describes.
#:
#: ``method`` is recognised on every table even though several shipped sheets
#: omit it, because normalize_dataframe creates it when it is absent. ``note``
#: is the legacy marker ``##`` replaced; read_input_excels drops it before this
#: question is asked, and it is listed so that a caller which does not drop it
#: reaches the same answer.
COMMON_COLUMNS: tuple[str, ...] = ("scenario", "year", "method", "note")

#: table -> the columns saying *which thing* a row is about.
#:
#: Includes the topology columns this stage builds -- ``node``, ``unit``,
#: ``from_node`` -- as well as the components they are built from. A sheet does
#: not normally write a built one, but writing it is a statement about the
#: model rather than an unread column, so it is not reported.
#:
#: ``generator_id`` is absent because it no longer exists: a unitdata row names
#: its ``unittype`` directly. A sheet still carrying the old column is therefore
#: reported by report_unused_columns, naming the file and sheet, which is how an
#: unmigrated workbook says so.
#:
#: ``country`` is deliberately absent from four tables. Only nodedata,
#: demanddata, unitdata and userconstraintdata are filtered by it; a ``country``
#: column on unittypedata, emissiondata or transferdata reaches nothing, and an
#: author who wrote one believes they made something country-specific when they
#: did not. That is precisely what this check is for.
STRUCTURAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "unittypedata": ("unittype", "flow", "grid", "node"),
    "unitdata": ("country", "unit_name_prefix", "unittype",
                 "unit", "flow", "grid", "node", "node_suffix"),
    "nodedata": ("country", "grid", "node", "node_suffix"),
    "demanddata": ("country", "grid", "node", "node_suffix"),
    "transferdata": ("grid", "from_country", "to_country",
                     "from_suffix", "to_suffix", "from_node", "to_node"),
    "emissiondata": ("emission", "group"),
    "userconstraintdata": ("country", "group", "parameter",
                           "1st dimension", "2nd dimension",
                           "3rd dimension", "4th dimension"),
}

#: table -> the columns that identify a row and must therefore hold something.
#:
#: A subset of STRUCTURAL_COLUMNS: the ones without which the row describes
#: nothing. ``unit_name_prefix``, ``node_suffix`` and the transfer suffixes are
#: deliberately absent -- they are optional by design and blank on most rows.
#: ``scenario`` and ``year`` are absent too: apply_whitelist already counts a
#: blank in either, and a sheet may legitimately omit the column entirely.
#:
#: Checked only where the column is present. A sheet missing one of these
#: outright is a different failure, reported by whichever builder needs it.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "unittypedata": ("unittype",),
    "unitdata": ("country", "unittype"),
    "nodedata": ("country", "grid"),
    "demanddata": ("country", "grid"),
    "transferdata": ("grid", "from_country", "to_country"),
    "emissiondata": ("emission",),
    "userconstraintdata": ("group", "parameter"),
}

#: A year outside this range is not a year. ``1`` is the "every year" wildcard and
#: is allowed on its own; anything else below the range is a leftover or a typo,
#: and it matches no scenario year, so the row is dropped without a word.
PLAUSIBLE_YEARS = range(1900, 2201)

#: table -> the Backbone parameter blocks its sheets may carry.
#:
#: nodedata carries the boundary *types* as columns -- build_boundarydata melts
#: them into rows -- while the boundary *properties* are columns of the table
#: that melt produces, so they are not input here and are not listed.
#:
#: demanddata is given PARAM_GN deliberately generously: what it may carry
#: beyond twh/year has not been established, and a warning on correct data is
#: worse than a column this check lets past.
PARAMETER_COLUMNS: dict[str, tuple[str, ...]] = {
    "unittypedata": (*backbone_params.PARAM_GNU, *backbone_params.PARAM_UNIT),
    "unitdata": (*backbone_params.PARAM_GNU, *backbone_params.PARAM_UNIT),
    "nodedata": (*backbone_params.PARAM_GN,
                 *backbone_params.PARAM_GN_BOUNDARY_TYPES),
    "demanddata": (*backbone_params.PARAM_GN,),
    "transferdata": (*backbone_params.PARAM_GNN,),
    "emissiondata": (*backbone_params.PARAM_EMISSION,),
    "userconstraintdata": (*backbone_params.PARAM_USERCONSTRAINT,),
}

#: table -> columns a later stage reads by name rather than passing through as
#: a parameter. Each names its consumer, because nothing else records it.
#:
#: These are the only entries here that can rot: a consumer can stop reading
#: one and nothing would notice. test_unused_columns.py greps src/ for each,
#: which catches a declaration outliving its reader.
DERIVATION_INPUTS: dict[str, tuple[str, ...]] = {
    # BBExcelPipeline.create_effLevelGroupUnit reads unit_row['lp/mip'].
    # Written on unittypedata and reaching unitdata through the unittype merge.
    "unittypedata": ("lp/mip",),
    "unitdata": ("lp/mip",),
    # DH_demand_fromTemperature, elec_demand_TYNDP2024 and
    # TimeseriesPipeline._create_other_demands read both by name.
    "demanddata": ("twh/year", "constant_share"),
}

#: table -> header prefixes standing for a dimension value rather than naming a
#: parameter, so the set of them is open-ended by design.
#:
#: Per table, not global. ``emission_<name>`` on a node gives that node's factor
#: and create_p_nEmission takes the emission's name from the suffix;
#: ``emission_group<n>`` on a unit names a group it belongs to. Splitting them
#: is what catches ``emission_CO2`` written on a unit sheet -- an emission
#: factor on the one table that cannot carry one, which a single global
#: ``emission_`` rule would wave through.
#:
#: It does not catch the mirror case, ``emission_group1`` on nodedata, and that
#: is correct here: the column is read, as an emission named ``group1``. It
#: reaches something wrong rather than reaching nothing, so it is a question for
#: whatever eventually declares emission names -- see "What is not here" above.
COLUMN_FAMILIES: dict[str, tuple[str, ...]] = {
    "nodedata": (backbone_params.EMISSION_COLUMN_PREFIX,),
    "demanddata": (backbone_params.EMISSION_COLUMN_PREFIX,),
    "unitdata": ("emission_group",),
    "unittypedata": ("emission_group",),
}


def recognised_columns(table: str) -> frozenset[str]:
    """Every column name `table`'s sheets may carry, lower-cased.

    Excludes the open-ended families, which are prefixes rather than names --
    :func:`unrecognised_columns` applies those separately.
    """
    names = (
        *COMMON_COLUMNS,
        *STRUCTURAL_COLUMNS.get(table, ()),
        *PARAMETER_COLUMNS.get(table, ()),
        *DERIVATION_INPUTS.get(table, ()),
    )
    return frozenset(name.lower() for name in names)


def unrecognised_columns(columns, table: str) -> list[str]:
    """Those of `columns` that `table` has no meaning for, in the order given.

    Names are returned **as they were passed**, not lower-cased: the caller
    reports them to someone who will search for that spelling in a workbook.

    A table this module knows nothing about yields nothing rather than
    everything. That is the same "cannot tell is not the same as empty" rule
    :func:`known_dimension_values` follows -- a new data category must not make
    every one of its columns look wrong on the day it is added.
    """
    if table not in STRUCTURAL_COLUMNS:
        return []

    known = recognised_columns(table)
    families = COLUMN_FAMILIES.get(table, ())

    unrecognised = []
    for column in columns:
        name = base_column_name(column)
        if name in known:
            continue
        if any(name.startswith(family.lower()) for family in families):
            continue
        unrecognised.append(column)
    return unrecognised
