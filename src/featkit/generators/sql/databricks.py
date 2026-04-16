"""DatabricksSQLCodeGenerator — SQL generation targeting the Databricks dialect."""

from __future__ import annotations

from featkit.generators.sql.base import AbstractSQLCodeGenerator


class DatabricksSQLCodeGenerator(AbstractSQLCodeGenerator):
    """SQL code generator for the Databricks dialect.

    Inherits all generation logic from :class:`AbstractSQLCodeGenerator`; the
    sole responsibility of this subclass is to declare the SQLGlot dialect
    identifier so that all transpiled SQL is rendered with Databricks-specific
    syntax (backtick quoting, ``NULLS LAST`` ordering, etc.).
    """

    @property
    def dialect(self) -> str:
        return "databricks"
