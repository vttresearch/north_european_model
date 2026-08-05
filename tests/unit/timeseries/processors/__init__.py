"""Tests for the individual timeseries processors.

Processor constructors do no I/O -- they validate kwargs and compute a date
range -- so the pure transforms in between can be exercised on hand-made frames
without any of the ~1 GB of real PECD/TYNDP input.
"""
