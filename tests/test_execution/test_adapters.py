"""Tests for featkit.execution.adapters — MockAdapter and base contract."""

from __future__ import annotations

import pandas as pd
import pytest

from featkit.execution.adapters.base import DataSourceAdapter, EngineType
from featkit.execution.adapters.mock_adapter import MockAdapter

# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------


class TestMockAdapter:
    def _make(self, sql: str, values: list[str]) -> MockAdapter:
        df = pd.DataFrame({"col": values})
        return MockAdapter({sql: df})

    def test_returns_registered_dataframe(self) -> None:
        sql = "SELECT DISTINCT x FROM t WHERE x IS NOT NULL ORDER BY 1"
        adapter = self._make(sql, ["a", "b"])
        result = adapter.execute(sql)
        assert list(result["col"]) == ["a", "b"]

    def test_raises_key_error_for_unknown_sql(self) -> None:
        adapter = MockAdapter({})
        with pytest.raises(KeyError):
            adapter.execute("SELECT 1")

    def test_call_count_increments(self) -> None:
        sql = "SELECT 1"
        adapter = MockAdapter({sql: pd.DataFrame({"v": [1]})})
        assert adapter.call_count(sql) == 0
        adapter.execute(sql)
        assert adapter.call_count(sql) == 1
        adapter.execute(sql)
        assert adapter.call_count(sql) == 2

    def test_call_count_zero_for_unexecuted(self) -> None:
        adapter = MockAdapter({})
        assert adapter.call_count("never called") == 0

    def test_engine_type_is_mock(self) -> None:
        adapter = MockAdapter({})
        assert adapter.engine_type() == EngineType.MOCK

    def test_is_data_source_adapter(self) -> None:
        adapter = MockAdapter({})
        assert isinstance(adapter, DataSourceAdapter)

    def test_execute_logs_and_reraises_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        adapter = MockAdapter({})
        level_ctx = caplog.at_level(logging.ERROR, logger="featkit.execution.adapters.base")
        with level_ctx, pytest.raises(KeyError):
            adapter.execute("bad sql")
        assert "bad sql" in caplog.text


# ---------------------------------------------------------------------------
# EngineType
# ---------------------------------------------------------------------------


class TestEngineType:
    def test_all_values_exist(self) -> None:
        assert EngineType.SQLALCHEMY.value == "sqlalchemy"
        assert EngineType.DATABRICKS.value == "databricks"
        assert EngineType.SPARK.value == "spark"
        assert EngineType.MOCK.value == "mock"
