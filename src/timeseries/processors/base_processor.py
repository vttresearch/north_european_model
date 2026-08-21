import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Union

import pandas as pd

import src.utils as utils
from src.timeseries.timeseries_results import ProcessorOutput


class SourceDataError(Exception):
    """An input file is not shaped the way the processor needs it.

    Raised by the reader helpers below so that ``process()`` stops where the bad
    data is, rather than carrying it forward. ``ProcessorRunner`` catches every
    exception out of ``run_processor()`` and writes no GDX for that processor,
    which is the intended consequence -- the reader has already logged the
    detail at ``error`` level by the time this propagates.
    """


class BaseProcessor(ABC):
    """
    Abstract base class for all timeseries processors.

    This class provides a standardized interface for processors that transform
    timeseries data. Subclasses must implement the `process()` method with their
    specific transformation logic.

    This is called by ProcessorRunner in src/pipeline/timeseries_processor.py

    The base class handles:
    - Result structuring via ProcessorOutput dataclass
    - Logging via the shared IterationLogger
    - Consistent execution pattern through `run_processor()`

    Declarations of intent
    ----------------------
    The first three class attributes below say what this processor's output is
    *supposed* to look like. ProcessorRunner reads them and checks the actual
    data against them on every run. ``requires_source_data`` is the odd one out:
    it declares an *input* the processor needs rather than an output property,
    and ProcessorRunner acts on it instead of checking it.

    They exist instead of a "this processor was checked" record committed
    alongside the code. Such a record can only claim that something passed once,
    somewhere, against data the reader does not have -- and the VRE processor
    reads whatever CSVs are in a config-supplied folder, so its data can change
    entirely without a single filename changing. A declaration makes no claim
    about the past. It states what must always be true, is checked against the
    data actually being processed, and cannot go stale.

    They are versioned for free: ProcessorRunner hashes the processor file, so
    editing one of these invalidates that processor's cache like any other edit.

    Declaring nothing is fine. The defaults assert nothing and every check
    involving them is skipped.

    Attributes
    ----------
    value_range : tuple of (float or None, float or None)
        Inclusive (minimum, maximum) the `value` column should stay within, each
        independently optional. A capacity factor is `(0.0, 1.0)`. Violations are
        warnings, not errors -- an out-of-range value may be a real feature of
        the source data, where a broken time axis cannot be.

    value_sign : {"any", "non_negative", "non_positive"}
        Which side of zero values belong on. Per-class rather than per-parameter
        because the same Backbone parameter legitimately takes both signs:
        `elec_demand_TYNDP2024` and `DH_demand_fromTemperature` emit negative
        `ts_influx`, `hydro_inflow_MAF2019` positive.

    expects_complete_datetime_axis : bool
        Whether the output is meant to be a complete, regular grid with one row
        per step per group. True for everything shipped. Named for the datetime
        axis rather than the hour because the checker takes the step as a
        parameter; the hourly assumption belongs to the labeller, whose window
        is `bb_ts_length * 24` labels.

    requires_source_data : tuple of str
        Which merged source-data frames this processor needs, named without the
        `df_` prefix -- `('nodedata',)` asks for `SourceDataPipeline.df_nodedata`
        and receives it as a `df_nodedata` kwarg.

        This is the alternative to a processor keeping its own copy of a value
        that source workbooks already carry. `hydro_storage_limits_MAF2019` used
        to read reservoir sizes from a CSV that duplicated
        `nodedata.upwardLimit` exactly, because the frame was unreachable from
        here; the two drifted apart with nothing able to notice.

        Declared on the class rather than in the config spec so that the cache
        follows it: ProcessorRunner hashes the concrete processor file, so
        changing what a processor needs invalidates that processor and nothing
        else, with no config edit anywhere.

        Two consequences a processor author should know. The frames are
        whitelisted per scenario, year and country, so declaring one makes the
        processor input-data-dependent and it will no longer be copied from a
        reference folder between scenarios. And the frames obey the source-side
        conventions, not the timeseries ones: `0` and `pd.NA` are distinct there,
        and an all-NA column arrives as `object` rather than `Float64`.

    main_result : pd.DataFrame or None
        The primary output DataFrame from the processor. This is automatically
        set when `run_processor()` is called and should not be modified directly.

    secondary_result : Any or None
        Optional secondary output (e.g., metadata, statistics, intermediate results).
        Set this attribute in your `process()` method if needed.

    Methods
    -------
    process()
        Abstract method - must be implemented by subclasses with the main
        transformation logic.

    run_processor()
        Template method that executes the processor and returns structured results.
        Do not override this unless you have a specific reason.

    See Also
    --------
    ProcessorOutput : Dataclass that structures the processor output
    ProcessorRunner : Orchestrates processor execution in the pipeline
    """

    #: See "Declarations of intent" above. Defaults assert nothing and ask for
    #: nothing.
    value_range: tuple = (None, None)
    value_sign: str = "any"
    expects_complete_datetime_axis: bool = True
    requires_source_data: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        """
        Initialize the base processor.

        Subclasses should call `super().__init__(**kwargs)` before performing
        their own initialization. The kwargs are preserved to allow flexible
        parameter passing from configuration.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments. Subclasses should extract and validate
            the parameters they need from this dict.

        Attributes Initialized
        ----------------------
        main_result : None
            Will be set when run_processor() is called
        secondary_result : None
            Can be set by subclass if additional outputs are needed
        """
        self.logger = kwargs.get('logger')
        self.main_result: Optional[pd.DataFrame] = None
        self.secondary_result: Optional[Any] = None

    # ------------------------------------------------------------------
    # Reading input files
    # ------------------------------------------------------------------
    #
    # Nothing forces a processor through these -- each processor is its own
    # package and reads whatever its source happens to be. They are here because
    # the alternative is worse than it looks.
    #
    # ProcessorRunner does check `main_result`, and its numeric check on the
    # `value` column is a real backstop. But it validates the *output*, and the
    # damage a malformed input file does is not confined to values: an unquoted
    # thousands separator in a CSV is read as a field separator, so every column
    # shifts and the node label becomes a number. pandas reports nothing -- it
    # absorbs the extra field as an index and hands back a frame that looks
    # healthy. Nothing downstream catches a corrupted *dimension* column, and it
    # reaches GDX as a set element. These helpers are the only place that can
    # see it.

    def _reject(self, message: str) -> None:
        """Log at error level and stop the processor."""
        self.logger.log_status(message, level="error")
        raise SourceDataError(message)

    def _check_frame(self, df: pd.DataFrame, source: str) -> None:
        """Reject a frame containing failed numbers or Excel error values.

        Refuses rather than repairs, which is the opposite of what the source
        workbook gate does with the same finding -- and deliberately so. A
        hand-edited sheet makes isolated typos, so blanking one cell and
        reporting it is proportionate. A generated file does not make typos: one
        malformed number means the producer changed format, and blanking would
        turn a whole column into a million fabricated zeros that look exactly
        like real data. Writing no GDX is the only honest outcome.
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
            that uses a marker of its own declares it here, which keeps the
            declaration visible in the processor instead of buried in a gate.
        **kwargs
            Passed to ``pd.read_csv``. ``index_col`` is not accepted -- see below.

        Notes
        -----
        ``index_col=False`` is forced. Left to itself, pandas treats a row with
        one field too many as "this file has an index column", silently shifting
        every value one place left for that row; with ``index_col=False`` it
        keeps the columns aligned and emits a ``ParserWarning`` instead, which is
        the only signal that the file is malformed. That warning is therefore
        treated as an error here rather than printed and forgotten.

        There is deliberately no ``thousands`` default. Reading ``1,000`` as a
        thousand would be a guess: the same cell is one-point-zero to an author
        writing in a locale where the comma is the decimal mark, and nothing in
        the file says which is meant. A processor whose source genuinely uses
        grouped digits can pass ``thousands=','`` explicitly and take
        responsibility for that claim.

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

        This method contains the core transformation logic of the processor.
        It should:
        1. Load or receive input data
        2. Transform the data according to processor requirements
        3. Optionally set self.secondary_result if needed
        4. Return the main result as a long-format pandas DataFrame

        The method can use `self.logger.log_status()` to record progress and diagnostic information.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with exactly the columns:
                bb_parameter_dimensions (excluding 't' and 'f')  +  ['time', 'value']
            For example: ['grid', 'node', 'time', 'value'].
            The 'time' column must contain datetime values covering the full
            range from start_year-01-01 to end_year-12-31 23:00.
            Climate-window slicing and f (forecast) column insertion are handled 
            by the runner, not the processor.
            Nothing more, nothing less than the required columns.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method
        Exception
            Any exception raised by the processing logic should be allowed to
            propagate. The caller will handle exception logging and recovery.
        """
        pass

    def run_processor(self) -> ProcessorOutput:
        """
        Execute the processor and return structured results.

        This is the main entry point for running a processor. It calls the
        abstract `process()` method implemented by the subclass, captures
        the results, and packages them into a ProcessorOutput dataclass.

        This method implements the Template Method pattern - it defines the
        execution skeleton while delegating the actual work to the subclass's
        `process()` method.

        Returns
        -------
        ProcessorOutput
            A dataclass containing:
            - main_result (pd.DataFrame): Primary output from process()
            - secondary_result (Any | None): Optional additional outputs

        Raises
        ------
        Exception
            Any exception raised by the `process()` method will propagate.
            The calling code (typically ProcessorRunner) is responsible for
            exception handling and logging.

        See Also
        --------
        process : The abstract method that contains the actual processing logic
        ProcessorOutput : The returned dataclass structure
        """
        # Run the main processing logic implemented by subclass
        self.main_result = self.process()

        # Package results into structured dataclass
        return ProcessorOutput(
            main_result=self.main_result,
            secondary_result=self.secondary_result,
        )

