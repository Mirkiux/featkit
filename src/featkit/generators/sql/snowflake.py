"""SnowflakeSQLCodeGenerator — SQL generation targeting the Snowflake dialect."""

from __future__ import annotations

from featkit.generators.sql.base import AbstractSQLCodeGenerator


class SnowflakeSQLCodeGenerator(AbstractSQLCodeGenerator):
    """SQL code generator for the Snowflake dialect.

    Inherits all generation logic from :class:`AbstractSQLCodeGenerator`; the
    sole responsibility of this subclass is to declare the SQLGlot dialect
    identifier so that all transpiled SQL is rendered with Snowflake-specific
    syntax (quoting, ``QUALIFY``, ``MAX_BY``, ``MEDIAN``, etc.).
    """

    @property
    def dialect(self) -> str:
        return "snowflake"
