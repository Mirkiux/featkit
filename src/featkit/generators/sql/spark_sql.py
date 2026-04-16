"""SparkSQLCodeGenerator — SQL generation targeting the Apache Spark SQL dialect."""

from __future__ import annotations

from featkit.generators.sql.base import AbstractSQLCodeGenerator


class SparkSQLCodeGenerator(AbstractSQLCodeGenerator):
    """SQL code generator for the Apache Spark SQL dialect.

    Inherits all generation logic from :class:`AbstractSQLCodeGenerator`; the
    sole responsibility of this subclass is to declare the SQLGlot dialect
    identifier so that all transpiled SQL is rendered with Spark-specific
    syntax (backtick quoting, ``NULLS LAST`` ordering, etc.).
    """

    @property
    def dialect(self) -> str:
        return "spark"
