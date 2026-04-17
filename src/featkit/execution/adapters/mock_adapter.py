"""In-memory adapter for tests and examples."""

from __future__ import annotations

import pandas as pd

from featkit.execution.adapters.base import DataSourceAdapter, EngineType


class MockAdapter(DataSourceAdapter):
    """In-memory adapter for tests and examples.

    Accepts a dict mapping SQL strings to pre-built DataFrames.
    Raises ``KeyError`` when an unregistered SQL string is executed.
    Has no external *engine* dependencies (no database connector required);
    ``pandas`` is still needed as it is the shared return type of all adapters.

    Each call to :meth:`execute` increments an internal counter so tests can
    verify how many times a query was executed.
    """

    def __init__(self, results: dict[str, pd.DataFrame]) -> None:
        self._results = results
        self._call_counts: dict[str, int] = {}

    def engine_execute(self, sql: str) -> pd.DataFrame:
        self._call_counts[sql] = self._call_counts.get(sql, 0) + 1
        return self._results[sql]

    def call_count(self, sql: str) -> int:
        """Return the number of times *sql* has been executed."""
        return self._call_counts.get(sql, 0)

    def engine_type(self) -> EngineType:
        return EngineType.MOCK
