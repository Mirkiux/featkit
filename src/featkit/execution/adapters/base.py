"""Base adapter ABC and EngineType enum for featkit execution layer."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

_log = logging.getLogger(__name__)


class EngineType(Enum):
    """Identifies the execution engine behind a :class:`DataSourceAdapter`."""

    SQLALCHEMY = "sqlalchemy"
    DATABRICKS = "databricks"
    SPARK = "spark"
    MOCK = "mock"


class DataSourceAdapter(ABC):
    """Abstract base for all data source adapters.

    Subclasses implement :meth:`engine_execute` with the engine-specific query
    logic.  The public :meth:`execute` method wraps ``engine_execute`` with
    error handling: if the query fails the offending SQL is logged at ``ERROR``
    level before re-raising, making debugging significantly easier in
    production pipelines.

    This follows the *Template Method* pattern — the base class owns the
    algorithm skeleton; subclasses supply only the engine-specific step.
    """

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute *sql* and return a DataFrame.

        Delegates to :meth:`engine_execute`.  On failure, logs the SQL that
        caused the error and re-raises the original exception unchanged.
        """
        try:
            return self.engine_execute(sql)
        except Exception:
            _log.exception(
                "SQL execution failed on %s.\nFailed query:\n%s",
                self.__class__.__name__,
                sql,
            )
            raise

    @abstractmethod
    def engine_execute(self, sql: str) -> pd.DataFrame:
        """Execute *sql* against the underlying engine and return a DataFrame.

        Implement this method in each adapter subclass.  Do not call it
        directly — use :meth:`execute` instead so that error handling and any
        future cross-cutting behaviour (retries, metrics) are applied.
        """

    @abstractmethod
    def engine_type(self) -> EngineType:
        """Return the :class:`EngineType` for this adapter."""
