"""featkit.execution.adapters — data source adapters for domain resolution."""

from featkit.execution.adapters.base import DataSourceAdapter, EngineType
from featkit.execution.adapters.databricks_adapter import DatabricksAdapter
from featkit.execution.adapters.databricks_notebook_adapter import DatabricksNotebookAdapter
from featkit.execution.adapters.mock_adapter import MockAdapter
from featkit.execution.adapters.spark_adapter import SparkAdapter
from featkit.execution.adapters.sqlalchemy_adapter import SQLAlchemyAdapter

__all__ = [
    "DataSourceAdapter",
    "DatabricksAdapter",
    "DatabricksNotebookAdapter",
    "EngineType",
    "MockAdapter",
    "SparkAdapter",
    "SQLAlchemyAdapter",
]
