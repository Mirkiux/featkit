"""AdapterDomainResolver — resolves CategoricalField domains via a live adapter."""

from __future__ import annotations

import re

from featkit.execution.adapters.base import DataSourceAdapter
from featkit.fields.categorical_field import CategoricalField

# Matches a simple SQL identifier: letters, digits, underscores; must start
# with a letter or underscore.  Dollar signs are excluded deliberately —
# they are technically valid in some dialects but uncommon and easy to abuse.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches a fully-qualified table reference: one to four dot-separated
# identifiers (e.g. "db", "db.schema", "db.schema.table",
# "catalog.db.schema.table").
_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,3}$")


def _require_safe_identifier(value: str, label: str) -> None:
    """Raise ``ValueError`` if *value* is not a safe SQL identifier."""
    if not _IDENT_RE.match(value):
        raise ValueError(
            f"{label} {value!r} is not a valid SQL identifier. "
            "Only letters, digits, and underscores are allowed; "
            "the value must start with a letter or underscore."
        )


def _require_safe_reference(value: str, label: str) -> None:
    """Raise ``ValueError`` if *value* is not a safe table reference."""
    if not _REF_RE.match(value):
        raise ValueError(
            f"{label} {value!r} is not a valid table reference. "
            "Expected one to four dot-separated SQL identifiers "
            "(e.g. 'mydb.myschema.my_table')."
        )


class AdapterDomainResolver:
    """Resolves the allowed-values domain of a CategoricalField by executing a
    ``SELECT DISTINCT`` query against the facts table.

    Both *source_reference* and the field name are validated against a strict
    identifier pattern before being interpolated into SQL.  This prevents SQL
    injection from user-supplied field or table names.

    The resolver is a callable compatible with the ``domain_resolver`` parameter
    of :class:`~featkit.builders.pivot_space.PivotSpaceBuilder` — pass an
    instance directly::

        resolver = AdapterDomainResolver(adapter, "mydb.silver_transactions")
        builder = PivotSpaceBuilder(dataset=ds, domain_resolver=resolver)

    Or configure it through :class:`~featkit.config.FeatureStoreConfig` so that
    :class:`~featkit.pipeline.FeatureStorePipeline` wires it automatically::

        cfg = FeatureStoreConfig(
            dataset=ds,
            ...,
            adapter=adapter,
        )
        pipeline = FeatureStorePipeline(config=cfg).build()

    Args:
        adapter: A :class:`~featkit.execution.adapters.base.DataSourceAdapter`
            instance used to execute the ``SELECT DISTINCT`` query.
        source_reference: Fully-qualified table name (e.g.
            ``"mydb.myschema.silver_transactions"``).  Validated against a
            safe identifier pattern at construction time.

    Raises:
        ValueError: At construction time if *source_reference* contains
            characters outside the safe identifier pattern, or at call time
            if the resolved field name does the same.
    """

    def __init__(self, adapter: DataSourceAdapter, source_reference: str) -> None:
        _require_safe_reference(source_reference, "source_reference")
        self._adapter = adapter
        self._source_reference = source_reference

    def __call__(self, field: CategoricalField) -> list[str]:
        """Return distinct non-null values for *field* from the facts table.

        Raises:
            ValueError: If ``field.name`` is not a safe SQL identifier.
        """
        _require_safe_identifier(field.name, "field.name")
        sql = (
            f"SELECT DISTINCT {field.name} "
            f"FROM {self._source_reference} "
            f"WHERE {field.name} IS NOT NULL "
            f"ORDER BY 1"
        )
        df = self._adapter.execute(sql)
        return list(df.iloc[:, 0].astype(str))
