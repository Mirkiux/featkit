"""AdapterDomainResolver — resolves CategoricalField domains via a live adapter."""

from __future__ import annotations

from featkit.execution.adapters.base import DataSourceAdapter
from featkit.fields.categorical_field import CategoricalField


class AdapterDomainResolver:
    """Resolves the allowed-values domain of a CategoricalField by executing a
    ``SELECT DISTINCT`` query against the facts table.

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
            ``"mydb.myschema.silver_transactions"``).  Passed directly into
            the SQL — must be trusted input.
    """

    def __init__(self, adapter: DataSourceAdapter, source_reference: str) -> None:
        self._adapter = adapter
        self._source_reference = source_reference

    def __call__(self, field: CategoricalField) -> list[str]:
        """Return distinct non-null values for *field* from the facts table."""
        sql = (
            f"SELECT DISTINCT {field.name} "
            f"FROM {self._source_reference} "
            f"WHERE {field.name} IS NOT NULL "
            f"ORDER BY 1"
        )
        df = self._adapter.execute(sql)
        return list(df.iloc[:, 0].astype(str))
