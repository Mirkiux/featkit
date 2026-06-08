"""FeatureStorePipeline — orchestrates space builders and delegates to a code generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from featkit.builders.distributional_space import DistributionalSpaceBuilder
from featkit.builders.pivot_space import PivotSpaceBuilder
from featkit.builders.ratio_space import RatioSpaceBuilder
from featkit.builders.temporal_space import TemporalSpaceBuilder
from featkit.config import FeatureStoreConfig
from featkit.layer2.distributional import DistributionalColumn
from featkit.layer2.pivoted import PivotedColumn
from featkit.layer2.ratio import RatioPivotedColumn
from featkit.layer3.temporal_feature import TemporalFeature

if TYPE_CHECKING:
    from featkit.generators.base import AbstractCodeGenerator
    from featkit.generators.output import FeatureStoreOutput


class FeatureStorePipeline:
    """Orchestrates the three space builders and delegates code generation.

    The pipeline holds the results of the last ``build()`` call in three lists:
    ``layer2a`` (pivot columns), ``layer2b`` (distributional columns), and
    ``layer3`` (temporal features). All three are empty until ``build()`` is
    called.

    Args:
        config: The pipeline configuration.
    """

    def __init__(self, config: FeatureStoreConfig) -> None:
        self.config = config
        self.layer2a: list[PivotedColumn] = []
        self.layer2b: list[DistributionalColumn] = []
        self.layer2c: list[RatioPivotedColumn] = []
        self.layer3: list[TemporalFeature] = []

    def build(self) -> FeatureStorePipeline:
        """Run all three space builders and return ``self``.

        Idempotent: calling ``build()`` more than once replaces the previous
        results with an identical fresh computation.
        """
        cfg = self.config

        combination_resolver = None
        if cfg.adapter is not None:
            from featkit.execution.domain_resolver import AdapterCombinationResolver

            combination_resolver = AdapterCombinationResolver(
                cfg.adapter, cfg.dataset.source_reference
            )

        self.layer2a = PivotSpaceBuilder(
            dataset=cfg.dataset,
            include_marginals=cfg.include_marginals,
            aggregators_override=cfg.aggregators_override,
            combination_resolver=combination_resolver,
            verbose=cfg.verbose,
        ).build()
        self.layer2b = DistributionalSpaceBuilder(
            dataset=cfg.dataset,
            verbose=cfg.verbose,
        ).build()
        self.layer2c = (
            RatioSpaceBuilder(self.layer2a, verbose=cfg.verbose).build()
            if cfg.include_ratios and cfg.include_marginals
            else []
        )
        self.layer3 = TemporalSpaceBuilder(
            layer2_columns=[*self.layer2a, *self.layer2b, *self.layer2c],
            time_windows=cfg.time_windows,
            composed_windows=cfg.composed_windows,
            operators_override=cfg.operators_override,
            verbose=cfg.verbose,
        ).build()
        return self

    def run(self, generator: AbstractCodeGenerator) -> FeatureStoreOutput:
        """Delegate code generation entirely to the supplied generator."""
        return generator.generate(self)
