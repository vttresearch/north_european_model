# src/processors/base_processor.py

import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Any
from src.pipeline.timeseries_processor import ProcessorResult


class BaseProcessor(ABC):
    """
    Abstract base class for all timeseries processors.
    
    This class provides a standardized interface for processors that transform
    timeseries data. Subclasses must implement the `process()` method with their
    specific transformation logic.
    
    This is called by ProcessorRunner in src/pipeline/timeseries_processor.py

    The base class handles:
    - Result structuring via ProcessorResult dataclass
    - Logging infrastructure
    - Consistent execution pattern through `run_processor()`
    
    Attributes
    ----------
    processor_log : list[str]
        Accumulated log messages from processor execution. Each log entry
        should be added via the `log()` method rather than appending directly.
        
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
        
    log(message, level='info')
        Add a log message to the processor's log.    
    
    See Also
    --------
    ProcessorResult : Dataclass that structures the processor output
    ProcessorRunner : Orchestrates processor execution in the pipeline
    """
    
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
        processor_log : list[str]
            Empty list to accumulate log messages
        main_result : None
            Will be set when run_processor() is called
        secondary_result : None
            Can be set by subclass if additional outputs are needed
        """
        self.processor_log = []
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
        4. Return the main result as a pandas DataFrame
        
        The method can use `self.log()` to record progress and diagnostic information.
        
        Returns
        -------
        pd.DataFrame
            The primary result of the processor. This DataFrame will be:
            - Stored in self.main_result
            - Included in the ProcessorResult returned by run_processor()
            - Further processed by the pipeline (e.g., trimming, GDX conversion)
            
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method
        Exception
            Any exception raised by the processing logic should be allowed to
            propagate. The caller will handle exception logging and recovery.
        """
        pass
    
    def run_processor(self) -> ProcessorResult:
        """
        Execute the processor and return structured results.
        
        This is the main entry point for running a processor. It calls the
        abstract `process()` method implemented by the subclass, captures
        the results, and packages them into a ProcessorResult dataclass.
        
        This method implements the Template Method pattern - it defines the
        execution skeleton while delegating the actual work to the subclass's
        `process()` method.
        
        Returns
        -------
        ProcessorResult
            A dataclass containing:
            - main_result (pd.DataFrame): Primary output from process()
            - secondary_result (Any | None): Optional additional outputs
            - log_messages (list[str]): Accumulated log messages
            
        Raises
        ------
        Exception
            Any exception raised by the `process()` method will propagate.
            The calling code (typically ProcessorRunner) is responsible for
            exception handling and logging.
            
        See Also
        --------
        process : The abstract method that contains the actual processing logic
        ProcessorResult : The returned dataclass structure
        """
        # Run the main processing logic implemented by subclass
        self.main_result = self.process()
        
        # Package results into structured dataclass
        return ProcessorResult(
            main_result=self.main_result,
            secondary_result=self.secondary_result,
            log_messages=self.processor_log
        )
    
    def log(self, message: str, level: str = "info"):
        """
        Add a log message to the processor's log.
        
        This is the recommended way to record diagnostic information, progress
        updates, and warnings during processor execution. Log messages are
        accumulated and included in the ProcessorResult.
        
        Parameters
        ----------
        message : str
            The log message to record. Should be concise and informative.

        
        See Also
        --------
        src.utils.log_status : The underlying logging utility function
        """
        from src.utils import log_status
        log_status(message, self.processor_log, level=level)
