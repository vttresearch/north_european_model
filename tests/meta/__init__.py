"""Tests for the test infrastructure itself.

Machinery that cannot fail is worthless: a delta helper that never reports a
difference, or a contract assertion that accepts anything, would make the whole
suite green and meaningless.  Everything in ``tests/_common/`` therefore has a
negative control here -- a case that must FAIL -- alongside its happy path.
"""
