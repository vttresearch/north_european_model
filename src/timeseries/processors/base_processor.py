import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Union

import pandas as pd

import src.utils as utils
from src.timeseries.timeseries_results import ProcessorOutput


class SourceDataError(Exception):
    """An input file is not shaped the way the processor needs it.

    Raised by the reader helpers below, which have already logged the detail at
    ``error`` level. ProcessorRunner catches it and writes no GDX for that
    processor, which is the intended consequence.
    """


class BaseProcessor(ABC):
    """
    Abstract base class for all timeseries processors.

    Subclasses implement `process()`; the base class packages its return into a
    ProcessorOutput and provides the reader helpers below. Run by ProcessorRunner
    in src/pipeline/timeseries_processor.py.

    Declarations of intent
    ----------------------
    The class attributes below say what this processor's output must always look
    like, and ProcessorRunner checks the data actually being processed against
    them on every run. `requires_source_data` is the odd one out: it declares an
    *input* the processor needs, and ProcessorRunner acts on it instead of
    checking it.

    A declaration replaces a "this processor was checked" record, which can only
    claim that something passed once against data the reader does not have -- the
    VRE processor reads whatever CSVs a config-supplied folder holds, so its data
    can change entirely without a filename changing. Declarations are versioned
    for free: ProcessorRunner hashes the processor file, so editing one
    invalidates that processor's cache.

    Declaring nothing is fine. The defaults assert nothing and ask for nothing,
    and every check involving them is skipped.

    Attributes
    ----------
    value_range : tuple of (float or None, float or None)
        Inclusive (minimum, maximum) the `value` column should stay within, each
        independently optional. A capacity factor is `(0.0, 1.0)`. Violations are
        warnings, not errors -- an out-of-range value may be a real feature of
        the source data, where a broken time axis cannot be.

    value_sign : {"any", "non_negative", "non_positive"}
        Which side of zero values belong on. Per-class rather than per-parameter
        because the same Backbone parameter legitimately takes both signs: the
        demand processors emit negative `ts_influx`, `hydro_inflow_MAF2019`
        positive.

    expects_complete_datetime_axis : bool
        Whether the output is meant to be a complete, regular grid with one row
        per step per group. True for everything shipped.

    requires_source_data : tuple of str
        Which merged source-data frames this processor needs, named without the
        `df_` prefix -- `('nodedata',)` asks for `SourceDataPipeline.df_nodedata`
        and receives it as a `df_nodedata` kwarg. Use it instead of keeping a
        private copy of a value the source workbooks already carry: the copy and
        the original drift apart with nothing able to notice.

        Declared on the class rather than in the config spec so that the cache
        follows it, and with two consequences for the author. The frames are
        whitelisted per scenario, year and country, so declaring one makes the
        processor input-data-dependent and it will no longer be copied from a
        reference folder between scenarios. And the frames obey the source-side
        conventions, not the timeseries ones: `0` and `pd.NA` are distinct there,
        and an all-NA column arrives as `object` rather than `Float64`.

    main_result : pd.DataFrame or None
        Primary output, set by `run_processor()`. Do not modify directly.

    frames : dict of str to pd.DataFrame
        Contributions to the source data tables, filled in `process()` if the
        processor has something to say that is not a time series. Keys are table
        names without the `df_` prefix -- the same names `requires_source_data`
        uses, so a processor reads and writes `nodedata` by one name.

        Most processors contribute nothing: a node, a grid and a flow the model
        already knows about need no announcing, and the workbooks name them
        already. What does need saying is a fact the workbooks cannot hold --
        `hydro_storage_limits_MAF2019` contributes a `boundarydata` row stating
        that a node's upwardLimit comes from a series rather than a constant,
        because nothing downstream can work that out from the GDX.

        The rules the contribution is checked against, and what happens when two
        producers describe the same row, are in `source_data_contributions`.
    """

    #: See "Declarations of intent" above.
    value_range: tuple = (None, None)
    value_sign: str = "any"
    expects_complete_datetime_axis: bool = True
    requires_source_data: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        """
        Initialize the base processor.

        Subclasses should call `super().__init__(**kwargs)` before their own
        initialization, then extract and validate the parameters they need from
        the kwargs the configuration passed in.
        """
        self.logger = kwargs.get('logger')
        self.main_result: Optional[pd.DataFrame] = None
        self.frames: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Reading input files
    # ------------------------------------------------------------------
    #
    # Nothing forces a processor through these, but ProcessorRunner is no
    # substitute: it checks output *values*, while a malformed input file also
    # corrupts *dimensions*. An unquoted thousands separator in a CSV is read as
    # a field separator, so every column shifts and the node label becomes a
    # number -- pandas absorbs the extra field as an index and hands back a frame
    # that looks healthy. That label then reaches GDX as a set element, and these
    # helpers are the only place that can see it.

    def _reject(self, message: str) -> None:
        """Log at error level and stop the processor."""
        self.logger.log_status(message, level="error")
        raise SourceDataError(message)

    def _check_frame(self, df: pd.DataFrame, source: str) -> None:
        """Reject a frame containing failed numbers or Excel error values.

        Refuses rather than repairs, the opposite of what the source workbook
        gate does with the same finding. A hand-edited sheet makes isolated
        typos, so blanking one cell is proportionate. A generated file does not:
        one malformed number means the producer changed format, and blanking
        would turn a whole column into fabricated zeros that look like real data.
        """
        for report, what in (
            (utils.find_malformed_numeric_cells(df), "malformed number"),
            (utils.find_excel_error_values(df), "Excel error value"),
        ):
            if report.ok:
                continue
            detail = "; ".join(
                f"'{col}': {count} ({', '.join(repr(v) for v in report.examples[col])})"
                for col, count in report.counts.items()
            )
            self._reject(
                f"[{source}] {report.total} {what}(s) found -- {detail}. "
                f"A generated file does not make isolated typos, so this reads as a "
                f"format change at the source rather than a stray cell. "
                f"No GDX output will be written."
            )

    def read_input_csv(
        self,
        path: Union[str, Path],
        *,
        na_values=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Read a CSV, refusing anything whose numbers or field count are wrong.

        Parameters
        ----------
        path : str or Path
            The file to read.
        na_values : optional
            Extra strings to treat as missing, passed to ``pd.read_csv``. pandas
            already knows ``NA``, ``N/A``, ``n/a``, ``NULL``, ``NaN``, ``None``
            and ``#N/A``; ``-`` and ``n.a.`` are **not** among them. A source
            with a marker of its own declares it here, visible in the processor
            rather than buried in a gate.
        **kwargs
            Passed to ``pd.read_csv``. ``index_col`` is not accepted -- see below.

        Notes
        -----
        ``index_col=False`` is forced. Left to itself, pandas reads a row with
        one field too many as "this file has an index column" and silently shifts
        that row one place left; with ``index_col=False`` the columns stay
        aligned and a ``ParserWarning`` is the only signal that the file is
        malformed, so that warning is treated as an error here.

        There is deliberately no ``thousands`` default: ``1,000`` is a thousand
        or one-point-zero depending on the author's locale, and nothing in the
        file says which. A processor whose source genuinely uses grouped digits
        passes ``thousands=','`` and takes responsibility for that claim.

        Raises
        ------
        SourceDataError
            If the field count is inconsistent, or any column mixes real numbers
            with values that look like numbers but do not parse.
        """
        if "index_col" in kwargs:
            raise TypeError(
                "read_input_csv does not accept index_col: it forces index_col=False so "
                "that a row with too many fields is reported instead of silently shifting "
                "every column. Set an index on the returned frame if you need one."
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = pd.read_csv(path, index_col=False, na_values=na_values, **kwargs)

        for w in caught:
            if issubclass(w.category, pd.errors.ParserWarning):
                self._reject(
                    f"[{path}] Inconsistent number of fields per row: {w.message} "
                    f"Rows do not line up with the header, so column values are not "
                    f"what their names say. The usual cause is an unquoted comma inside "
                    f"a value, such as a thousands separator. "
                    f"No GDX output will be written."
                )

        self._check_frame(df, str(path))
        return df

    def read_input_excel(
        self,
        path: Union[str, Path],
        **kwargs,
    ) -> pd.DataFrame:
        """Read a sheet, refusing anything whose numbers are wrong.

        The spreadsheet counterpart of :meth:`read_input_csv`. There is no field
        alignment to check -- a sheet has cells, not delimiters -- so this is the
        numeric and Excel-error check alone.

        Raises
        ------
        SourceDataError
            If any column mixes real numbers with values that look like numbers
            but do not parse.
        """
        df = pd.read_excel(path, **kwargs)
        sheet = kwargs.get("sheet_name")
        source = f"{path}:{sheet}" if sheet is not None else str(path)
        self._check_frame(df, source)
        return df

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """
        Main processing logic - must be implemented by subclasses.

        May fill `self.frames`, and may use `self.logger.log_status()` to record
        progress. Exceptions should propagate; the caller logs them and recovers.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with exactly the columns:
                bb_parameter_dimensions (excluding 't' and 'f')  +  ['time', 'value']
            For example: ['grid', 'node', 'time', 'value']. Nothing more, nothing
            less. The 'time' column must contain datetime values covering the
            full range from start_year-01-01 to end_year-12-31 23:00.
            Climate-window slicing and f (forecast) column insertion are handled
            by the runner, not the processor.
        """
        pass

    def run_processor(self) -> ProcessorOutput:
        """
        Execute the processor and package its results into a ProcessorOutput.

        The main entry point, and not one to override. Exceptions from
        `process()` propagate to the caller, typically ProcessorRunner, which
        handles logging and recovery.
        """
        self.main_result = self.process()

        return ProcessorOutput(
            main_result=self.main_result,
            frames=self.frames,
        )

