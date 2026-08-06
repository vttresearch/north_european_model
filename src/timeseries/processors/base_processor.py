import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Any
from src.timeseries.timeseries_results import ProcessorOutput


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
    The three class attributes below say what this processor's output is
    *supposed* to look like. ProcessorRunner reads them and checks the actual
    data against them on every run.

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

    #: See "Declarations of intent" above. Defaults assert nothing.
    value_range: tuple = (None, None)
    value_sign: str = "any"
    expects_complete_datetime_axis: bool = True

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

