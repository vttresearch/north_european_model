"""
Which source table declares a dimension's values.

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
