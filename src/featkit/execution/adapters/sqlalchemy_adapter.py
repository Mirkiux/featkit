"""Adapter backed by a SQLAlchemy-compatible connection string."""

from __future__ import annotations

from typing import Any

import pandas as pd

from featkit.execution.adapters.base import DataSourceAdapter, EngineType


class SQLAlchemyAdapter(DataSourceAdapter):
    """Adapter backed by a SQLAlchemy-compatible connection string.

    Supports any engine that SQLAlchemy supports: PostgreSQL, Oracle,
    Snowflake, MySQL, SQLite, and any JDBC-compatible backend via the
    appropriate dialect package.

    The SQLAlchemy engine is created lazily on the first :meth:`execute` call.
    """

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError as exc:
                raise ImportError(
                    "sqlalchemy is required for SQLAlchemyAdapter. "
                    "Install with: pip install sqlalchemy"
                ) from exc
            self._engine = create_engine(self._connection_string)
        return self._engine

    def engine_execute(self, sql: str) -> pd.DataFrame:
        engine = self._get_engine()
        result: pd.DataFrame = pd.read_sql_query(sql, engine)
        return result

    def engine_type(self) -> EngineType:
        return EngineType.SQLALCHEMY
