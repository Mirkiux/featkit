"""Adapter for Databricks notebook environments.

In Databricks notebooks the Spark session is pre-instantiated and injected
into the notebook's global namespace as ``spark``.  This adapter discovers
that session automatically — no constructor arguments required.

Usage inside a Databricks notebook::

    from featkit.execution.adapters import DatabricksNotebookAdapter

    adapter = DatabricksNotebookAdapter()
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from featkit.execution.adapters.base import DataSourceAdapter, EngineType


def _resolve_spark() -> Any:
    """Locate the ``spark`` session injected by the Databricks notebook runtime.

    Databricks injects ``spark`` into the ``__main__`` module namespace before
    the first notebook cell executes.  This function retrieves it without
    requiring the caller to hold a reference or import PySpark explicitly.

    Raises
    ------
    RuntimeError
        When no ``spark`` object can be found.  This most commonly means the
        adapter is being used outside a Databricks notebook environment.
    """
    main = sys.modules.get("__main__", None)
    spark = getattr(main, "spark", None) if main is not None else None

    if spark is None:
        raise RuntimeError(
            "Could not locate 'spark' in the notebook runtime namespace. "
            "DatabricksNotebookAdapter is intended for use inside Databricks "
            "notebooks where 'spark' is pre-injected by the runtime. "
            "Outside that environment, use SparkAdapter(spark_session) instead."
        )
    return spark


class DatabricksNotebookAdapter(DataSourceAdapter):
    """Adapter for Databricks notebook environments.

    Wraps the ``spark`` session that the Databricks runtime pre-injects into
    every notebook's global namespace.  No constructor arguments are needed —
    the session is resolved lazily on the first :meth:`execute` call.

    Examples
    --------
    Inside a Databricks notebook::

        from featkit.execution.adapters import DatabricksNotebookAdapter
        from featkit.config import FeatureStoreConfig
        from featkit.dataset.base import SimpleDataset
        from featkit.enums import CategoricalTreatment, MeasurementType, TimeGranularity
        from featkit.fields.categorical_field import CategoricalField
        from featkit.fields.id_field import IDField
        from featkit.fields.measurement_field import MeasurementField
        from featkit.fields.time_field import TimeField
        from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator
        from featkit.pipeline import FeatureStorePipeline

        adapter = DatabricksNotebookAdapter()

        ds = SimpleDataset(
            "mydb.silver_transactions",
            [
                IDField("client_id"),
                TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("amount", MeasurementType.MONTO),
                CategoricalField("segment", CategoricalTreatment.PIVOT),  # no allowed_values
            ],
        )

        cfg = FeatureStoreConfig(
            dataset=ds,
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3, 6, 12],
            adapter=adapter,
        )

        pipeline = FeatureStorePipeline(config=cfg).build()
        result = DatabricksSQLCodeGenerator().generate(pipeline)
        result.save("/dbfs/mnt/output/features/")

    Outside a notebook (e.g. in a standalone script or test), use
    :class:`SparkAdapter` instead and pass the session explicitly::

        adapter = SparkAdapter(spark_session)
    """

    def __init__(self) -> None:
        self._spark: Any = None

    def engine_execute(self, sql: str) -> pd.DataFrame:
        """Execute *sql* via the notebook's pre-injected ``spark`` session.

        The session is resolved once and cached for the lifetime of this
        adapter instance.
        """
        if self._spark is None:
            self._spark = _resolve_spark()
        result: pd.DataFrame = self._spark.sql(sql).toPandas()
        return result

    def engine_type(self) -> EngineType:
        return EngineType.SPARK
