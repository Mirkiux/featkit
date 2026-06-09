"""Domain resolvers — resolve CategoricalField domains via a live adapter."""

from __future__ import annotations

import logging
import re

from featkit.execution.adapters.base import DataSourceAdapter
from featkit.fields.categorical_field import CategoricalField

_log = logging.getLogger(__name__)

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
        verbose: When ``True``, emits a ``DEBUG``-level log message with the
            generated SQL before each query is executed.

    Raises:
        ValueError: At construction time if *source_reference* contains
            characters outside the safe identifier pattern, or at call time
            if the resolved field name does the same.
    """

    def __init__(
        self, adapter: DataSourceAdapter, source_reference: str, verbose: bool = False
    ) -> None:
        _require_safe_reference(source_reference, "source_reference")
        self._adapter = adapter
        self._source_reference = source_reference
        self._verbose = verbose

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
        if self._verbose:
            _log.debug("AdapterDomainResolver SQL: %s", sql)
        df = self._adapter.execute(sql)
        return list(df.iloc[:, 0].astype(str))


class AdapterCombinationResolver:
    """Resolves observed multi-field combinations by executing a single
    ``SELECT DISTINCT`` query against the facts table.

    Instead of generating the full Cartesian product of per-field domains,
    this resolver queries only the combinations that actually exist in the
    data — optionally filtered by each field's ``allowed_values``.  Fields
    without ``allowed_values`` are included in the SELECT but not filtered
    in the WHERE clause.

    Column names and the source reference are validated against a strict
    identifier pattern before being interpolated into SQL.  String values
    in ``allowed_values`` are escaped with standard SQL single-quote
    doubling to prevent injection.

    The resolver is a callable compatible with the ``combination_resolver``
    parameter of :class:`~featkit.builders.pivot_space.PivotSpaceBuilder`.

    Args:
        adapter: A :class:`~featkit.execution.adapters.base.DataSourceAdapter`
            instance used to execute the query.
        source_reference: Fully-qualified table name.  Validated at
            construction time.
        verbose: When ``True``, emits a ``DEBUG``-level log message with the
            generated SQL before each query is executed.

    Raises:
        ValueError: At construction time if *source_reference* is unsafe,
            or at call time if any field name is unsafe.
    """

    def __init__(
        self, adapter: DataSourceAdapter, source_reference: str, verbose: bool = False
    ) -> None:
        _require_safe_reference(source_reference, "source_reference")
        self._adapter = adapter
        self._source_reference = source_reference
        self._verbose = verbose

    def __call__(self, fields: list[CategoricalField]) -> list[dict[CategoricalField, str]]:
        """Return observed non-null combinations for *fields* from the facts table.

        Fields are sorted by name before building the query so the SQL is
        deterministic regardless of the order *fields* are supplied.

        Raises:
            ValueError: If any ``field.name`` is not a safe SQL identifier.
        """
        if not fields:
            return []
        for field in fields:
            _require_safe_identifier(field.name, "field.name")

        sorted_fields = sorted(fields, key=lambda f: f.name)
        col_list = ", ".join(f.name for f in sorted_fields)
        order_list = ", ".join(str(i + 1) for i in range(len(sorted_fields)))

        where_parts: list[str] = [f"{f.name} IS NOT NULL" for f in sorted_fields]
        for f in sorted_fields:
            if f.allowed_values is not None:
                if len(f.allowed_values) == 0:
                    return []
                if any(v is None for v in f.allowed_values):
                    raise ValueError(
                        f"CategoricalField {f.name!r}: allowed_values contains None; "
                        "None is reserved as the ∅ marginal sentinel"
                    )
                escaped = ", ".join("'" + str(v).replace("'", "''") + "'" for v in f.allowed_values)
                where_parts.append(f"{f.name} IN ({escaped})")

        sql = (
            f"SELECT DISTINCT {col_list} "
            f"FROM {self._source_reference} "
            f"WHERE {' AND '.join(where_parts)} "
            f"ORDER BY {order_list}"
        )
        if self._verbose:
            _log.debug("AdapterCombinationResolver SQL: %s", sql)
        df = self._adapter.execute(sql)
        if df.empty:
            return []
        df = df.astype(str)
        records = df.to_dict(orient="records")
        return [{f: rec[f.name] for f in fields} for rec in records]
