"""Adapter that wraps a live PySpark SparkSession."""

from __future__ import annotations

from typing import Any

import pandas as pd

from featkit.execution.adapters.base import DataSourceAdapter, EngineType


class SparkAdapter(DataSourceAdapter):
    """Adapter that wraps a live PySpark ``SparkSession``.

    The ``spark_session`` parameter is typed as ``Any`` so that pyspark does
    not need to be installed in environments that use other adapters.  Pass a
    real ``SparkSession`` at runtime; pass a duck-typed fake in tests.
    """

    def __init__(self, spark_session: Any) -> None:
        self._spark = spark_session

    def engine_execute(self, sql: str) -> pd.DataFrame:
        result: pd.DataFrame = self._spark.sql(sql).toPandas()
        return result

    def engine_type(self) -> EngineType:
        return EngineType.SPARK
