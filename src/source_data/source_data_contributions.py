"""
What a stage outside the source workbooks may add to a ``df_*`` table.

A timeseries processor knows things the workbooks do not -- that a node's
upwardLimit comes from a series rather than a constant, that a demand grid with
no profile gets a flat influx. It says so by returning **contributions**: frames
named after the source-data tables, merged into them once the timeseries phase
is over, so that ``BBExcelPipeline`` reads one set of tables and needs to know
nothing about where a row came from.

The rules
---------
**One vocabulary.** A contribution is named without the ``df_`` prefix, exactly
as ``BaseProcessor.requires_source_data`` names the frames a processor reads. A
processor reads and writes ``nodedata`` by one name.

**One key per table** (:data:`CONTRIBUTION_KEYS`) -- the model-facing identity,
not the workbook merge key. ``merge_row_by_row`` keys nodedata on
``(country, grid, node)``, but a processor has no country to give and does not
need one: ``(grid, node)`` names the same row.

**The workbook wins.** A contribution fills only where the source frame says
nothing, the same precedence ``merge_unittypedata_into_unitdata`` uses for
type-level defaults. So a value written by hand overrides a processor: put an
``influx`` on a nodedata row and the flat one
``TimeseriesPipeline._influx_for_grids_without_a_processor`` derives for that grid
never reaches it. Boundaries are the one table this cannot be said of, because
they have no workbook to write in -- see docs/identified-gaps.md.

**First answer wins.** Where two producers describe the same row,
:func:`combine_contributions` keeps the first and drops the rest, which is the
rule :func:`validate_contribution` already applies inside a single frame. Neither
of them is arbitrating between two right answers: two producers writing the same
row is a mistake somewhere, and the merge below cannot say whose.

**Nothing derived is ever cached.** The cache holds what a processor returned
and nothing else; the melt below and the merge here are recomputed on every
build from ``workbooks + cached contributions``. A cache holding a half-merged
table cannot be reasoned about after a partial rerun.

Conventions
-----------
Source-side, on both sides of the merge: ``pd.NA`` and ``0`` are distinct, an
all-NA column is ``object`` rather than ``Float64``, and object columns hold
``pd.NA``. Only columns a contribution actually touches are re-typed, so a
column no contribution mentions keeps the exact dtype the source stage gave it.
"""

from __future__ import annotations

import pandas as pd

import src.backbone_params as backbone_params
import src.utils as utils


#: Contribution name -> the columns identifying one row of it.
#:
#: The names are the ``df_*`` attributes of :class:`SourceDataPipeline` without
#: the prefix. ``boundarydata`` is the one with no workbook of its own yet; it is
#: built by :func:`build_boundarydata` from nodedata's boundary columns.
CONTRIBUTION_KEYS: dict[str, tuple[str, ...]] = {
    "nodedata":           ("grid", "node"),
    "demanddata":         ("grid", "node"),
    "boundarydata":       ("grid", "node", "param_gnboundarytypes"),
    "transferdata":       ("grid", "from_node", "to_node"),
    "unitdata":           ("unit",),
    "emissiondata":       ("emission",),
    "userconstraintdata": ("group", *backbone_params.UC_DIMENSION_COLUMNS, "parameter"),
}

#: The columns of ``df_boundarydata``. The three keys, then the Backbone
#: properties of p_gnBoundaryPropertiesForStates lowercased to match the
#: source-side column convention.
BOUNDARY_COLUMNS: list[str] = [
    *CONTRIBUTION_KEYS["boundarydata"],
    *(p.lower() for p in backbone_params.PARAM_GN_BOUNDARY_PROPERTIES),
]


def _row_keys(df: pd.DataFrame, key_columns) -> list[tuple]:
    """Each row's key as a tuple of strings.

    Strings rather than a MultiIndex because the key columns can arrive as
    ``object``, ``string`` or -- for an all-NA column -- something with no values
    to type at all, and a comparison between two of those is not reliable.
    """
    return list(zip(*(df[c].astype(str) for c in key_columns)))


def validate_contribution(
    name: str,
    frame,
    *,
    processor: str,
    logger,
) -> pd.DataFrame | None:
    """Check one contributed frame, returning it cleaned or ``None``.

    Refusals are logged naming the processor and cost that contribution alone --
    the run carries on, and the processor's GDX output is unaffected. What is
    refused: a name no ``df_*`` table answers to, anything that is not a
    DataFrame, a frame missing one of its key columns, and a missing value inside
    a key column. That last one matters most: key values become GAMS set
    elements downstream, so a blank is a broken key rather than a gap.

    Column names are lower-cased on the way in, matching what
    ``normalize_dataframe`` does to a workbook sheet.
    """
    if name not in CONTRIBUTION_KEYS:
        logger.log_status(
            f"Processor '{processor}' contributed a frame named '{name}', which is not "
            f"one of the source data tables ({', '.join(sorted(CONTRIBUTION_KEYS))}). "
            f"Ignoring it.",
            level="warn",
        )
        return None

    if not isinstance(frame, pd.DataFrame):
        logger.log_status(
            f"Processor '{processor}' contributed '{name}' as "
            f"{type(frame).__name__}, expected pd.DataFrame. Ignoring it.",
            level="error",
        )
        return None

    if frame.empty:
        logger.log_status(
            f"Processor '{processor}' contributed an empty '{name}' frame, so nothing "
            f"reaches df_{name} from it.",
            level="warn",
        )
        return None

    frame = frame.copy()
    frame.columns = [str(c).lower() for c in frame.columns]

    key_columns = CONTRIBUTION_KEYS[name]
    missing = [c for c in key_columns if c not in frame.columns]
    if missing:
        logger.log_status(
            f"Processor '{processor}' contributed '{name}' without its key column(s) "
            f"{', '.join(missing)}. A contribution is matched to a source row by "
            f"({', '.join(key_columns)}), so without them there is nothing to match. "
            f"Ignoring it.",
            level="error",
        )
        return None

    for column in key_columns:
        if frame[column].isna().any():
            n_missing = int(frame[column].isna().sum())
            logger.log_status(
                f"Processor '{processor}' contributed '{name}' with {n_missing} row(s) "
                f"whose '{column}' is blank. Key values become GAMS set elements and "
                f"cannot be empty. Ignoring it.",
                level="error",
            )
            return None

    frame = utils.standardize_df_dtypes(frame)

    duplicated = pd.Series(_row_keys(frame, key_columns)).duplicated()
    if duplicated.any():
        logger.log_status(
            f"Processor '{processor}' contributed '{name}' with {int(duplicated.sum())} "
            f"row(s) repeating a ({', '.join(key_columns)}) already in the frame. "
            f"Keeping the first of each.",
            level="warn",
        )
        frame = frame[~duplicated.to_numpy()].reset_index(drop=True)

    return frame


def _keep_first_of_each_key(
    name: str, frame: pd.DataFrame, logger
) -> pd.DataFrame:
    """`frame` with the second and later row for any key dropped.

    Left alone when the table is not one this module knows, or the stacked frame
    is missing a key column: ``apply_contributions`` reports the first and
    ``merge_contribution`` the second, and neither ends up merged, so there is
    nothing a de-duplication here could protect.
    """
    key_columns = CONTRIBUTION_KEYS.get(name)
    if key_columns is None or not set(key_columns) <= set(frame.columns):
        return frame

    keys = pd.Series(_row_keys(frame, key_columns))
    duplicated = keys.duplicated()
    if not duplicated.any():
        return frame

    if logger is not None:
        repeated = [" ".join(key) for key in keys[duplicated.to_numpy()]]
        logger.log_status(
            f"{len(repeated)} contributed '{name}' row(s) repeat a "
            f"({', '.join(key_columns)}) another contribution already stated: "
            f"{utils.summarise(repeated)}. Keeping the first of each. Two producers "
            f"describing one row is a mistake somewhere, and nothing here can tell "
            f"which of them is right.",
            level="warn",
        )
    return frame[~duplicated.to_numpy()].reset_index(drop=True)


def combine_contributions(
    per_source: list[dict[str, pd.DataFrame]],
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Stack several producers' contributions into one frame per table name.

    Order follows the order they were produced in, and the **first answer wins**:
    where two producers describe the same row, the later one is dropped here
    rather than carried into the merge. The merge cannot express the choice --
    its matched-row fill would keep the last, and an unmatched duplicate would be
    appended twice, putting two rows with one key into a ``df_*`` table, which is
    the very state ``merge_contribution`` then refuses to merge into.

    So the rule is the one ``validate_contribution`` already applies inside a
    single frame, and a repeated key costs the same whether it arrives from one
    producer or from two.

    `logger` is optional only because a caller may have none; pass one wherever
    there is one, since a dropped row is worth saying out loud.
    """
    stacked: dict[str, list[pd.DataFrame]] = {}
    for frames in per_source:
        for name, frame in (frames or {}).items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                stacked.setdefault(name, []).append(frame)

    return {
        name: _keep_first_of_each_key(
            name,
            frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True),
            logger,
        )
        for name, frames in stacked.items()
    }


def merge_contribution(
    source: pd.DataFrame,
    contribution: pd.DataFrame,
    key_columns,
    *,
    name: str,
    logger,
) -> pd.DataFrame:
    """Fold one contribution into its source frame.

    Matched rows are filled only where the source is NA; unmatched rows are
    appended with ``pd.NA`` everywhere else. Existing rows keep their order and
    their values, and a column no contribution mentions keeps its dtype
    untouched -- the frame is not re-standardised as a whole, only the columns
    that were written to.
    """
    if contribution is None or contribution.empty:
        return source

    key_columns = list(key_columns)

    if source is None or source.empty:
        return contribution.reset_index(drop=True)

    # The same test validate_contribution applies to the contribution, applied to
    # the source. Not symmetry for its own sake: the key columns of
    # userconstraintdata are the four uc slots, which create_p_userconstraint
    # itself treats as optional and fills with pd.NA, so a source frame without
    # one is a shape this pipeline already produces -- and _row_keys would raise
    # a bare KeyError on it, in a phase that is supposed to carry on.
    missing = [c for c in key_columns if c not in source.columns]
    if missing:
        logger.log_status(
            f"df_{name} has no {', '.join(missing)} column, so a contribution cannot "
            f"be matched to it. Leaving the table as the source data left it.",
            level="warn",
        )
        return source

    source_keys = _row_keys(source, key_columns)
    if len(set(source_keys)) != len(source_keys):
        logger.log_status(
            f"df_{name} has more than one row for the same "
            f"({', '.join(key_columns)}), so a contribution cannot be matched to it "
            f"unambiguously. Leaving the table as the source data left it.",
            level="warn",
        )
        return source

    merged = source.reset_index(drop=True)
    written = [c for c in contribution.columns if c not in key_columns]

    # Columns the source has never heard of start as all-NA object: no
    # assumption has been made about them yet, and the fill below decides.
    for column in written:
        if column not in merged.columns:
            merged[column] = pd.Series(pd.NA, index=merged.index, dtype="object")

    position = {key: pos for pos, key in enumerate(source_keys)}
    contribution_keys = _row_keys(contribution, key_columns)

    matched_rows = [i for i, key in enumerate(contribution_keys) if key in position]
    new_rows = [i for i, key in enumerate(contribution_keys) if key not in position]

    # --- fill matched rows, source wins ---
    if matched_rows:
        for column in written:
            filler = pd.Series(pd.NA, index=merged.index, dtype="object")
            for i in matched_rows:
                filler.iloc[position[contribution_keys[i]]] = contribution[column].iloc[i]
            merged[column] = merged[column].where(merged[column].notna(), filler)

    # --- append unmatched rows ---
    #
    # By reindexing rather than concatenating a second frame: concat decides the
    # result dtype from both sides, so every column the new rows say nothing
    # about -- which is most of them -- would be re-decided from an all-NA
    # block. Reindexing extends each column in place and leaves its dtype alone.
    if new_rows:
        first_new = len(merged)
        merged = merged.reindex(range(first_new + len(new_rows)))
        for column in key_columns + written:
            merged.loc[first_new:, column] = contribution[column].iloc[new_rows].to_numpy()
        # reindex fills an object column with float NaN; object columns hold pd.NA.
        for column in merged.columns:
            if merged[column].dtype == "object":
                merged[column] = merged[column].where(merged[column].notna(), pd.NA)

    # Only what was written is re-typed. Doing the whole frame would re-decide
    # dtypes the source stage already settled, on columns nothing touched.
    retyped = key_columns + written if new_rows else written
    if retyped:
        merged[retyped] = utils.standardize_df_dtypes(merged[retyped])

    return merged


def apply_contributions(source_data_pipeline, frames: dict[str, pd.DataFrame], logger) -> None:
    """Merge every contribution into the pipeline's frames, in place.

    Called once, after the timeseries phase and before the workbook is built.
    Deliberately not inside ``SourceDataPipeline.run()``: the frames it produces
    are what the processors *read*, and folding their own output back in before
    they run would feed them their answers.
    """
    for name, frame in sorted(frames.items()):
        if name not in CONTRIBUTION_KEYS:
            logger.log_status(
                f"Ignoring a contribution named '{name}', which is not one of the "
                f"source data tables.",
                level="warn",
            )
            continue
        attribute = f"df_{name}"
        merged = merge_contribution(
            getattr(source_data_pipeline, attribute, pd.DataFrame()),
            frame,
            CONTRIBUTION_KEYS[name],
            name=name,
            logger=logger,
        )
        setattr(source_data_pipeline, attribute, merged)


def build_boundarydata(df_nodedata: pd.DataFrame, logger) -> pd.DataFrame:
    """Turn nodedata's boundary columns into ``df_boundarydata`` rows.

    ``nodedata`` is wide -- one column per boundary type, holding its constant --
    because one row per node is what a spreadsheet is good at.
    p_gnBoundaryPropertiesForStates is long, and the boundary type is one of its
    dimensions. This is the translation between the two, and it has the same
    status as the ``emission_XX`` columns create_p_nEmission reads: a documented
    input format, not a shim.

    Each non-blank cell becomes a row stating ``useconstant``. A ``0`` is kept
    rather than dropped -- ``pd.NA`` and ``0`` are still distinct here, and it is
    the builder that decides an all-zero boundary says nothing.
    """
    if df_nodedata is None or df_nodedata.empty:
        return pd.DataFrame(columns=BOUNDARY_COLUMNS)

    if "grid" not in df_nodedata.columns or "node" not in df_nodedata.columns:
        logger.log_status(
            "Node data has no 'grid' or 'node' column, so no state boundaries can be "
            "read from it. Nodes keep their state variables but no limits.",
            level="warn",
        )
        return pd.DataFrame(columns=BOUNDARY_COLUMNS)

    rows = []
    for boundary_type in backbone_params.PARAM_GN_BOUNDARY_TYPES:
        column = boundary_type.lower()
        if column not in df_nodedata.columns:
            continue
        present = df_nodedata[df_nodedata[column].notna()]
        if present.empty:
            continue
        rows.append(
            pd.DataFrame({
                "grid": present["grid"].to_numpy(),
                "node": present["node"].to_numpy(),
                "param_gnboundarytypes": boundary_type,
                "useconstant": 1,
                "constant": present[column].to_numpy(),
            })
        )

    if not rows:
        return pd.DataFrame(columns=BOUNDARY_COLUMNS)

    boundarydata = pd.concat(rows, ignore_index=True)
    for column in BOUNDARY_COLUMNS:
        if column not in boundarydata.columns:
            boundarydata[column] = pd.NA

    return utils.standardize_df_dtypes(boundarydata[BOUNDARY_COLUMNS])
