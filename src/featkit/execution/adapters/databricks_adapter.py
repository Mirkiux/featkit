"""Adapter for Databricks SQL warehouses."""

from __future__ import annotations

import pandas as pd

from featkit.execution.adapters.base import DataSourceAdapter, EngineType


class DatabricksAdapter(DataSourceAdapter):
    """Adapter for Databricks SQL warehouses.

    Requires the optional ``databricks-sql-connector`` package::

        pip install databricks-sql-connector

    The connector is imported lazily inside :meth:`engine_execute` so that the
    class can be referenced in code that runs on environments without the
    package installed — the ``ImportError`` is only raised when a query is
    actually attempted.
    """

    def __init__(
        self,
        host: str,
        token: str,
        http_path: str,
        catalog: str,
        schema: str,
    ) -> None:
        self._host = host
        self._token = token
        self._http_path = http_path
        self._catalog = catalog
        self._schema = schema

    def engine_execute(self, sql: str) -> pd.DataFrame:
        try:
            from databricks import sql as dbsql
        except ImportError as exc:
            raise ImportError(
                "databricks-sql-connector is required for DatabricksAdapter. "
                "Install with: pip install databricks-sql-connector"
            ) from exc

        with (
            dbsql.connect(
                server_hostname=self._host,
                http_path=self._http_path,
                access_token=self._token,
                catalog=self._catalog,
                schema=self._schema,
            ) as conn,
            conn.cursor() as cursor,
        ):
            cursor.execute(sql)
            result: pd.DataFrame = cursor.fetchall_arrow().to_pandas()
            return result

    def engine_type(self) -> EngineType:
        return EngineType.DATABRICKS
