"""Shared test infrastructure.

The leading underscore keeps this folder out of pytest collection: it contains
no ``test_*.py``, so nothing here is mistaken for a test module.

House style, inherited from the parent Backbone repo's ``tests/_common/``:
helpers are plain **functions imported at the call site**, not pytest fixtures.
Arguments are then visible in the test that uses them, and functions compose
(``run_route`` -> ``build_input_folder`` -> ``write_workbook_text``) in a way
fixtures do not.
"""
