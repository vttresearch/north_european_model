"""Full-route tests: source workbooks -> ``inputData.xlsx``.

Not a mirror of ``src/`` -- a route test spans every module by definition, so it
is organised by the question it asks rather than by the code it touches.

Every route test starts with the same two assertions:

    r.logger.assert_no_errors()
    assert_workbook_consistent(r.sheets)

and then says whatever it is actually about, preferring the earliest assertion
family that can express it (rule R6 in tests/README.md).
"""
