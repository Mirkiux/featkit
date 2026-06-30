"""FeatureStoreConfig — top-level configuration object for a feature store pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from featkit.dataset.base import AbstractDataset
from featkit.enums import (
    Layer2Aggregator,
    Layer2OutputType,
    MeasurementType,
    RatioMode,
    TemporalOperator,
)

if TYPE_CHECKING:
    from featkit.execution.adapters.base import DataSourceAdapter


@dataclass
class FeatureStoreConfig:
    """Configuration for a FeatureStorePipeline run.

    Args:
        dataset: The source facts-table schema.
        output_schema: Destination schema (database/schema) for generated tables.
        output_table_prefix: Prefix applied to every output table name.
        time_windows: Window sizes (in granularity units) for standard windowed
            temporal operators.
        composed_windows: Window sizes for ``MEDIA_ABS`` and ``RATIO`` operators.
            When ``None`` those operators are omitted entirely.
        include_marginals: When ``True``, ``PivotSpaceBuilder`` includes the ∅
            marginal combination for each categorical.
        include_ratios: When ``True`` (and ``include_marginals`` is also
            ``True``), ``RatioSpaceBuilder`` derives a ratio column
            (``numerator / NULLIF(denominator, 0)``) for every pivot
            combination over each of its proper marginal projections.  Has no
            effect when ``include_marginals`` is ``False`` (no marginal
            denominators exist).
        ratio_mode: Controls which denominators are considered when
            ``include_ratios`` is ``True``.  ``RatioMode.ALL_PROJECTIONS``
            (default) pairs each numerator with every valid proper marginal
            projection.  ``RatioMode.GLOBAL_TOTAL`` restricts denominators to
            the single fully-marginalised column (all categorical fields set to
            ∅), producing one ratio per numerator representing its share of the
            grand total.
        aggregators_override: Per-measurement-type override for Layer 2
            aggregators. Only contract-valid aggregators are used.
        operators_override: Per-output-type override for temporal operators.
            Only contract-valid operators are used.
        adapter: Optional execution adapter.  When provided, categorical fields
            with no ``allowed_values`` have their domain resolved at
            ``FeatureStorePipeline.build()`` time via a ``SELECT DISTINCT``
            query against the facts table.
        verbose: When ``True``, the space builders emit ``DEBUG``-level log
            messages at key milestones: builder start/end, ``domain_resolver``
            invocations (PivotSpaceBuilder only), and each generated column name
            together with the combination that produced it.
    """

    dataset: AbstractDataset
    output_schema: str
    output_table_prefix: str
    time_windows: list[int]
    composed_windows: list[int] | None = None
    include_marginals: bool = True
    include_ratios: bool = True
    ratio_mode: RatioMode = RatioMode.ALL_PROJECTIONS
    aggregators_override: dict[MeasurementType, list[Layer2Aggregator]] | None = None
    operators_override: dict[Layer2OutputType, list[TemporalOperator]] | None = field(default=None)
    adapter: DataSourceAdapter | None = None
    verbose: bool = False
