"""Tests for Plan 12 — FeatureStoreConfig and FeatureStorePipeline."""

from __future__ import annotations

from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    Layer2Aggregator,
    MeasurementType,
    TimeGranularity,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.layer2.distributional import DistributionalColumn
from featkit.layer2.pivoted import PivotedColumn
from featkit.layer3.temporal_feature import TemporalFeature
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _config() -> FeatureStoreConfig:
    ds = SimpleDataset(
        "db.schema.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            CategoricalField(
                "sector",
                CategoricalTreatment.PIVOT,
                allowed_values=["A", "B"],
            ),
            CategoricalField(
                "region",
                CategoricalTreatment.DISTRIBUTIONAL,
                distributional_metrics=[DistributionalMetric.ENTROPY],
            ),
        ],
    )
    return FeatureStoreConfig(
        dataset=ds,
        output_schema="analytics",
        output_table_prefix="feat_",
        time_windows=[3, 6],
    )


# ---------------------------------------------------------------------------
# FeatureStoreConfig
# ---------------------------------------------------------------------------


class TestFeatureStoreConfig:
    def test_required_fields_stored(self) -> None:
        cfg = _config()
        assert cfg.output_schema == "analytics"
        assert cfg.output_table_prefix == "feat_"
        assert cfg.time_windows == [3, 6]

    def test_optional_defaults(self) -> None:
        cfg = _config()
        assert cfg.composed_windows is None
        assert cfg.include_marginals is True
        assert cfg.aggregators_override is None
        assert cfg.operators_override is None

    def test_composed_windows_stored(self) -> None:
        cfg = _config()
        cfg.composed_windows = [12, 24]
        assert cfg.composed_windows == [12, 24]


# ---------------------------------------------------------------------------
# FeatureStorePipeline — initial state
# ---------------------------------------------------------------------------


class TestFeatureStorePipelineInitialState:
    def test_lists_empty_before_build(self) -> None:
        pipeline = FeatureStorePipeline(config=_config())
        assert pipeline.layer2a == []
        assert pipeline.layer2b == []
        assert pipeline.layer3 == []

    def test_config_stored(self) -> None:
        cfg = _config()
        pipeline = FeatureStorePipeline(config=cfg)
        assert pipeline.config is cfg


# ---------------------------------------------------------------------------
# FeatureStorePipeline.build()
# ---------------------------------------------------------------------------


class TestFeatureStorePipelineBuild:
    def test_build_returns_self(self) -> None:
        pipeline = FeatureStorePipeline(config=_config())
        assert pipeline.build() is pipeline

    def test_build_populates_layer2a(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert len(pipeline.layer2a) > 0

    def test_build_populates_layer2b(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert len(pipeline.layer2b) > 0

    def test_build_populates_layer3(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert len(pipeline.layer3) > 0

    def test_layer2a_type(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert all(isinstance(c, PivotedColumn) for c in pipeline.layer2a)

    def test_layer2b_type(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert all(isinstance(c, DistributionalColumn) for c in pipeline.layer2b)

    def test_layer3_type(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert all(isinstance(f, TemporalFeature) for f in pipeline.layer3)

    def test_build_is_idempotent(self) -> None:
        pipeline = FeatureStorePipeline(config=_config())
        pipeline.build()
        names2a = [c.column_name for c in pipeline.layer2a]
        names2b = [c.column_name for c in pipeline.layer2b]
        names3 = [f.column_name for f in pipeline.layer3]
        pipeline.build()
        assert [c.column_name for c in pipeline.layer2a] == names2a
        assert [c.column_name for c in pipeline.layer2b] == names2b
        assert [f.column_name for f in pipeline.layer3] == names3

    def test_layer2a_column_count(self) -> None:
        # sector: 2 values + 1 marginal (include_marginals=True) x MONTO 4 aggs = 12
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert len(pipeline.layer2a) == 3 * 4

    def test_layer2b_column_count(self) -> None:
        # region x MONTO (4 aggs) x ENTROPY = 4
        pipeline = FeatureStorePipeline(config=_config()).build()
        assert len(pipeline.layer2b) == 4

    def test_layer3_sources_cover_both_layer2s(self) -> None:
        pipeline = FeatureStorePipeline(config=_config()).build()
        layer3_source_names = {f.source.column_name for f in pipeline.layer3}
        layer2a_names = {c.column_name for c in pipeline.layer2a}
        layer2b_names = {c.column_name for c in pipeline.layer2b}
        assert layer3_source_names >= layer2a_names
        assert layer3_source_names >= layer2b_names


# ---------------------------------------------------------------------------
# FeatureStorePipeline — edge cases
# ---------------------------------------------------------------------------


class TestFeatureStorePipelineEdgeCases:
    def test_no_pivot_cats_produces_uncategorized_layer2a(self) -> None:
        # When no PIVOT-eligible categoricals exist, PivotSpaceBuilder still
        # produces the base measurement columns (no categorical slicing).
        ds = SimpleDataset(
            "tbl",
            [
                IDField("id"),
                TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("mto", MeasurementType.MONTO),
                CategoricalField(
                    "region",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
            ],
        )
        cfg = FeatureStoreConfig(
            dataset=ds, output_schema="s", output_table_prefix="p_", time_windows=[3]
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        # MONTO: SUM, MAX, MIN, AVG = 4 uncategorized pivot columns
        assert len(pipeline.layer2a) == 4
        assert all(isinstance(c, PivotedColumn) for c in pipeline.layer2a)

    def test_no_distributional_cats_gives_empty_layer2b(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                IDField("id"),
                TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("mto", MeasurementType.MONTO),
                CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A"]),
            ],
        )
        cfg = FeatureStoreConfig(
            dataset=ds, output_schema="s", output_table_prefix="p_", time_windows=[3]
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        assert pipeline.layer2b == []

    def test_aggregators_override_respected(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                IDField("id"),
                TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("mto", MeasurementType.MONTO),
                CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A"]),
            ],
        )
        cfg = FeatureStoreConfig(
            dataset=ds,
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=False,
            aggregators_override={MeasurementType.MONTO: [Layer2Aggregator.SUM]},
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        # 1 value x 1 agg (SUM only) = 1 pivot column
        assert len(pipeline.layer2a) == 1
        assert pipeline.layer2a[0].layer2_aggregator == Layer2Aggregator.SUM


# ---------------------------------------------------------------------------
# FeatureStorePipeline.run()
# ---------------------------------------------------------------------------


class TestFeatureStorePipelineRun:
    def test_run_delegates_to_generator(self) -> None:
        class _StubGenerator:
            received: FeatureStorePipeline | None = None
            _sentinel = object()

            def generate(self, pipeline: FeatureStorePipeline) -> object:
                self.received = pipeline
                return self._sentinel

        pipeline = FeatureStorePipeline(config=_config()).build()
        gen = _StubGenerator()
        result = pipeline.run(gen)  # type: ignore[arg-type]
        assert gen.received is pipeline
        assert result is gen._sentinel

    def test_run_passes_pipeline_with_built_data(self) -> None:
        received_counts: dict[str, int] = {}

        class _InspectingGenerator:
            def generate(self, pipeline: FeatureStorePipeline) -> None:
                received_counts["layer2a"] = len(pipeline.layer2a)
                received_counts["layer2b"] = len(pipeline.layer2b)
                received_counts["layer3"] = len(pipeline.layer3)

        pipeline = FeatureStorePipeline(config=_config()).build()
        pipeline.run(_InspectingGenerator())  # type: ignore[arg-type]
        assert received_counts["layer2a"] == len(pipeline.layer2a)
        assert received_counts["layer2b"] == len(pipeline.layer2b)
        assert received_counts["layer3"] == len(pipeline.layer3)
