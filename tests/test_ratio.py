"""Tests for RatioPivotedColumn, RatioSpaceBuilder, and SQL generation of ratio columns."""

from __future__ import annotations

import pytest

from featkit.builders.ratio_space import RatioSpaceBuilder
from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import (
    CategoricalTreatment,
    Layer2Aggregator,
    Layer2OutputType,
    MeasurementType,
    RatioMode,
    TimeGranularity,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.layer2.pivoted import PivotedColumn
from featkit.layer2.ratio import RatioPivotedColumn
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def amount() -> MeasurementField:
    return MeasurementField("amount", MeasurementType.MONTO)


@pytest.fixture()
def channel() -> CategoricalField:
    return CategoricalField("channel", CategoricalTreatment.PIVOT, allowed_values=["retail"])


@pytest.fixture()
def region() -> CategoricalField:
    return CategoricalField("region", CategoricalTreatment.PIVOT, allowed_values=["north"])


@pytest.fixture()
def full_combo(amount, channel, region) -> PivotedColumn:
    """channel=retail, region=north — fully specified."""
    return PivotedColumn(
        amount,
        Layer2Aggregator.SUM,
        {channel: "retail", region: "north"},
    )


@pytest.fixture()
def marginal_channel(amount, channel, region) -> PivotedColumn:
    """channel=∅, region=north — channel marginal."""
    return PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: "north"})


@pytest.fixture()
def marginal_region(amount, channel, region) -> PivotedColumn:
    """channel=retail, region=∅ — region marginal."""
    return PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: None})


@pytest.fixture()
def marginal_all(amount, channel, region) -> PivotedColumn:
    """channel=∅, region=∅ — global marginal."""
    return PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: None})


# ---------------------------------------------------------------------------
# RatioPivotedColumn — construction
# ---------------------------------------------------------------------------


class TestRatioPivotedColumnConstruction:
    def test_column_name_over_channel_marginal(self, full_combo, marginal_channel):
        ratio = RatioPivotedColumn(full_combo, marginal_channel)
        assert ratio.column_name == (
            "SUM__amount__channel_retail__region_north__over__SUM__amount__region_north"
        )

    def test_column_name_over_region_marginal(self, full_combo, marginal_region):
        ratio = RatioPivotedColumn(full_combo, marginal_region)
        assert ratio.column_name == (
            "SUM__amount__channel_retail__region_north__over__SUM__amount__channel_retail"
        )

    def test_column_name_over_global_marginal(self, full_combo, marginal_all):
        ratio = RatioPivotedColumn(full_combo, marginal_all)
        assert ratio.column_name == ("SUM__amount__channel_retail__region_north__over__SUM__amount")

    def test_output_type_is_numeric(self, full_combo, marginal_all):
        ratio = RatioPivotedColumn(full_combo, marginal_all)
        assert ratio.output_type == Layer2OutputType.NUMERIC

    def test_output_contract_is_numeric(self, full_combo, marginal_all):
        from featkit.contracts.output.defaults import NumericOutputContract

        ratio = RatioPivotedColumn(full_combo, marginal_all)
        assert isinstance(ratio.output_contract, NumericOutputContract)

    def test_numerator_denominator_attrs(self, full_combo, marginal_all):
        ratio = RatioPivotedColumn(full_combo, marginal_all)
        assert ratio.numerator is full_combo
        assert ratio.denominator is marginal_all

    def test_repr(self, full_combo, marginal_all):
        ratio = RatioPivotedColumn(full_combo, marginal_all)
        r = repr(ratio)
        assert "RatioPivotedColumn" in r
        assert "SUM__amount__channel_retail__region_north" in r
        assert "SUM__amount" in r


# ---------------------------------------------------------------------------
# RatioPivotedColumn — validation
# ---------------------------------------------------------------------------


class TestRatioPivotedColumnValidation:
    def test_aggregator_mismatch_raises(self, amount, channel, region):
        # Both SUM and AVG are valid for MONTO, so both PivotedColumns can be built
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(amount, Layer2Aggregator.AVG, {channel: None, region: "north"})
        with pytest.raises(ValueError, match="same aggregator"):
            RatioPivotedColumn(num, denom)

    def test_measurement_mismatch_raises(self, channel, region):
        mf1 = MeasurementField("amount", MeasurementType.MONTO)
        mf2 = MeasurementField("count", MeasurementType.CANTIDAD)
        num = PivotedColumn(mf1, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(mf2, Layer2Aggregator.SUM, {channel: None, region: "north"})
        with pytest.raises(ValueError, match="same source_measurement"):
            RatioPivotedColumn(num, denom)

    def test_different_instances_same_measurement_raises(self, channel, region):
        # RatioPivotedColumn uses identity (is not) for source_measurement:
        # equal-but-distinct MeasurementField objects are rejected by design.
        mf1 = MeasurementField("amount", MeasurementType.MONTO)
        mf2 = MeasurementField("amount", MeasurementType.MONTO)
        num = PivotedColumn(mf1, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(mf2, Layer2Aggregator.SUM, {channel: None, region: "north"})
        with pytest.raises(ValueError, match="same source_measurement"):
            RatioPivotedColumn(num, denom)

    def test_field_keys_mismatch_raises(self, amount, channel, region):
        other = CategoricalField("other", CategoricalTreatment.PIVOT, allowed_values=["x"])
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, other: None})
        with pytest.raises(ValueError, match="same categorical fields"):
            RatioPivotedColumn(num, denom)

    def test_partial_numerator_is_valid(self, amount, channel, region):
        # Numerator may have None fields; denominator just needs to be more marginal.
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: "north"})
        denom = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: None})
        ratio = RatioPivotedColumn(num, denom)
        assert ratio.column_name == "SUM__amount__region_north__over__SUM__amount"

    def test_denom_not_more_marginal_raises(self, amount, channel, region):
        # Denominator must marginalize at least one non-None field from numerator.
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        with pytest.raises(ValueError, match="proper marginal projection"):
            RatioPivotedColumn(num, denom)

    def test_all_none_numerator_raises(self, amount, channel, region):
        # A global marginal cannot be a numerator: no field to marginalize further.
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: None})
        denom = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: None})
        with pytest.raises(ValueError, match="proper marginal projection"):
            RatioPivotedColumn(num, denom)

    def test_denominator_value_mismatch_raises(self, amount, channel, region):
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        # denominator has region=south but numerator has region=north
        denom = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: "south"})
        with pytest.raises(ValueError, match="marginal projection"):
            RatioPivotedColumn(num, denom)


# ---------------------------------------------------------------------------
# RatioSpaceBuilder
# ---------------------------------------------------------------------------


class TestRatioSpaceBuilder:
    def test_produces_ratio_per_marginal(
        self, full_combo, marginal_channel, marginal_region, marginal_all
    ):
        # full → three denominators (all marginal projections)
        # marginal_channel ({ch:None, r:north}) → marginal_all  (marginalises region)
        # marginal_region  ({ch:retail, r:None}) → marginal_all  (marginalises channel)
        cols = [full_combo, marginal_channel, marginal_region, marginal_all]
        ratios = RatioSpaceBuilder(cols).build()
        names = {r.column_name for r in ratios}
        assert "SUM__amount__channel_retail__region_north__over__SUM__amount__region_north" in names
        assert (
            "SUM__amount__channel_retail__region_north__over__SUM__amount__channel_retail" in names
        )
        assert "SUM__amount__channel_retail__region_north__over__SUM__amount" in names
        assert "SUM__amount__region_north__over__SUM__amount" in names
        assert "SUM__amount__channel_retail__over__SUM__amount" in names
        assert len(ratios) == 5

    def test_partial_numerator_produces_ratios(self, amount, channel, region):
        # {channel=None, region=north} can be a numerator over {channel=None, region=None}
        partial = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: "north"})
        global_m = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: None, region: None})
        ratios = RatioSpaceBuilder([partial, global_m]).build()
        assert len(ratios) == 1
        assert ratios[0].column_name == "SUM__amount__region_north__over__SUM__amount"

    def test_only_global_marginals_returns_empty(self, marginal_all):
        ratios = RatioSpaceBuilder([marginal_all]).build()
        assert ratios == []

    def test_no_marginals_returns_empty(self, full_combo):
        ratios = RatioSpaceBuilder([full_combo]).build()
        assert ratios == []

    def test_empty_input_returns_empty(self):
        ratios = RatioSpaceBuilder([]).build()
        assert ratios == []

    def test_different_aggregators_not_paired(self, amount, channel, region):
        # SUM and AVG are both valid for MONTO — but cross-agg ratios should not be generated
        num = PivotedColumn(amount, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom_avg = PivotedColumn(amount, Layer2Aggregator.AVG, {channel: None, region: "north"})
        ratios = RatioSpaceBuilder([num, denom_avg]).build()
        assert ratios == []

    def test_different_measurements_not_paired(self, channel, region):
        mf1 = MeasurementField("amount", MeasurementType.MONTO)
        mf2 = MeasurementField("txn", MeasurementType.CANTIDAD)
        num = PivotedColumn(mf1, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(mf2, Layer2Aggregator.SUM, {channel: None, region: "north"})
        ratios = RatioSpaceBuilder([num, denom]).build()
        assert ratios == []

    def test_different_instances_same_measurement_not_paired(self, channel, region):
        # Builder uses identity (is not) for source_measurement — same as the constructor.
        # Equal-but-distinct MeasurementField objects are not paired by design.
        mf1 = MeasurementField("amount", MeasurementType.MONTO)
        mf2 = MeasurementField("amount", MeasurementType.MONTO)
        num = PivotedColumn(mf1, Layer2Aggregator.SUM, {channel: "retail", region: "north"})
        denom = PivotedColumn(mf2, Layer2Aggregator.SUM, {channel: None, region: "north"})
        ratios = RatioSpaceBuilder([num, denom]).build()
        assert ratios == []

    def test_no_duplicate_ratio_columns(self, full_combo, marginal_all):
        # Same pair twice in input should still produce one ratio
        ratios = RatioSpaceBuilder([full_combo, marginal_all, full_combo]).build()
        names = [r.column_name for r in ratios]
        assert len(names) == len(set(names))

    def test_no_categorical_columns_returns_empty(self, amount):
        col = PivotedColumn(amount, Layer2Aggregator.SUM, {})
        ratios = RatioSpaceBuilder([col]).build()
        assert ratios == []


# ---------------------------------------------------------------------------
# Pipeline — layer2c integration
# ---------------------------------------------------------------------------


def _two_cat_dataset() -> SimpleDataset:
    return SimpleDataset(
        "db.s.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("amount", MeasurementType.MONTO),
            CategoricalField("channel", CategoricalTreatment.PIVOT, allowed_values=["r", "o"]),
            CategoricalField("region", CategoricalTreatment.PIVOT, allowed_values=["n", "s"]),
        ],
    )


class TestPipelineRatioIntegration:
    def test_layer2c_populated_when_include_ratios_true(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
        )
        pipeline = FeatureStorePipeline(cfg).build()
        assert len(pipeline.layer2c) > 0
        assert all(isinstance(c, RatioPivotedColumn) for c in pipeline.layer2c)

    def test_layer2c_empty_when_include_ratios_false(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=False,
        )
        pipeline = FeatureStorePipeline(cfg).build()
        assert pipeline.layer2c == []

    def test_layer2c_empty_when_no_marginals(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=False,
            include_ratios=True,
        )
        pipeline = FeatureStorePipeline(cfg).build()
        assert pipeline.layer2c == []

    def test_layer3_includes_ratio_temporal_features(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
        )
        pipeline = FeatureStorePipeline(cfg).build()
        l3_names = {f.column_name for f in pipeline.layer3}
        # At least one layer3 feature should reference an __over__ ratio column
        assert any("__over__" in name for name in l3_names)

    def test_ratio_column_names_contain_over(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
        )
        pipeline = FeatureStorePipeline(cfg).build()
        for col in pipeline.layer2c:
            assert "__over__" in col.column_name


# ---------------------------------------------------------------------------
# SQL generation — ratio expressions in build_layer2a
# ---------------------------------------------------------------------------


class TestSQLRatioGeneration:
    def _pipeline(self) -> FeatureStorePipeline:
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
        )
        return FeatureStorePipeline(cfg).build()

    def test_layer2a_sql_contains_nullif_ratio(self):
        from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator

        pipeline = self._pipeline()
        sql = DatabricksSQLCodeGenerator().build_layer2a(pipeline).sql
        assert "NULLIF(" in sql
        assert "__over__" in sql

    def test_layer2a_sql_ratio_columns_present(self):
        from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator

        pipeline = self._pipeline()
        sql = DatabricksSQLCodeGenerator().build_layer2a(pipeline).sql
        for col in pipeline.layer2c:
            assert col.column_name in sql

    def test_final_join_sql_contains_ratio_columns(self):
        from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator

        pipeline = self._pipeline()
        sql = DatabricksSQLCodeGenerator().build_final_join(pipeline).sql
        for col in pipeline.layer2c:
            assert col.column_name in sql


# ---------------------------------------------------------------------------
# RatioMode.GLOBAL_TOTAL
# ---------------------------------------------------------------------------


class TestRatioModeGlobalTotal:
    """RatioMode.GLOBAL_TOTAL restricts denominators to the all-None column only."""

    def test_global_total_produces_fewer_ratios_than_all_projections(
        self, full_combo, marginal_channel, marginal_region, marginal_all
    ):
        cols = [full_combo, marginal_channel, marginal_region, marginal_all]
        all_proj = RatioSpaceBuilder(cols, ratio_mode=RatioMode.ALL_PROJECTIONS).build()
        global_only = RatioSpaceBuilder(cols, ratio_mode=RatioMode.GLOBAL_TOTAL).build()
        assert len(global_only) < len(all_proj)

    def test_global_total_denominators_are_all_none(
        self, full_combo, marginal_channel, marginal_region, marginal_all
    ):
        cols = [full_combo, marginal_channel, marginal_region, marginal_all]
        ratios = RatioSpaceBuilder(cols, ratio_mode=RatioMode.GLOBAL_TOTAL).build()
        for ratio in ratios:
            denom_vals = ratio.denominator.categorical_combination.values()
            assert all(v is None for v in denom_vals), (
                f"Expected all-None denominator, got {ratio.denominator.column_name}"
            )

    def test_global_total_full_combo_produces_one_ratio(
        self, full_combo, marginal_channel, marginal_region, marginal_all
    ):
        # full_combo (ch=retail, r=north) should only pair with marginal_all (ch=None, r=None)
        cols = [full_combo, marginal_channel, marginal_region, marginal_all]
        ratios = RatioSpaceBuilder(cols, ratio_mode=RatioMode.GLOBAL_TOTAL).build()
        full_combo_ratios = [r for r in ratios if r.numerator is full_combo]
        assert len(full_combo_ratios) == 1
        assert full_combo_ratios[0].denominator is marginal_all

    def test_global_total_partial_marginals_also_pair_with_global(
        self, full_combo, marginal_channel, marginal_region, marginal_all
    ):
        # marginal_channel (ch=None, r=north) should also pair with marginal_all
        cols = [full_combo, marginal_channel, marginal_region, marginal_all]
        ratios = RatioSpaceBuilder(cols, ratio_mode=RatioMode.GLOBAL_TOTAL).build()
        partial_ratios = [r for r in ratios if r.numerator is marginal_channel]
        assert len(partial_ratios) == 1
        assert partial_ratios[0].denominator is marginal_all

    def test_global_total_no_global_marginal_returns_empty(
        self, full_combo, marginal_channel, marginal_region
    ):
        # Without the all-None column there is no valid denominator
        cols = [full_combo, marginal_channel, marginal_region]
        ratios = RatioSpaceBuilder(cols, ratio_mode=RatioMode.GLOBAL_TOTAL).build()
        assert ratios == []

    def test_global_total_default_mode_is_all_projections(
        self, full_combo, marginal_channel, marginal_region, marginal_all
    ):
        cols = [full_combo, marginal_channel, marginal_region, marginal_all]
        default = RatioSpaceBuilder(cols).build()
        explicit = RatioSpaceBuilder(cols, ratio_mode=RatioMode.ALL_PROJECTIONS).build()
        assert [r.column_name for r in default] == [r.column_name for r in explicit]


class TestPipelineRatioModeIntegration:
    def test_global_total_via_config(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
            ratio_mode=RatioMode.GLOBAL_TOTAL,
        )
        pipeline = FeatureStorePipeline(cfg).build()
        assert len(pipeline.layer2c) > 0
        for col in pipeline.layer2c:
            denom_vals = col.denominator.categorical_combination.values()
            assert all(v is None for v in denom_vals)

    def test_global_total_fewer_ratios_than_all_projections(self):
        base = dict(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
        )
        all_proj = FeatureStorePipeline(FeatureStoreConfig(**base)).build()
        global_only = FeatureStorePipeline(
            FeatureStoreConfig(**base, ratio_mode=RatioMode.GLOBAL_TOTAL)
        ).build()
        assert len(global_only.layer2c) < len(all_proj.layer2c)

    def test_default_ratio_mode_is_all_projections(self):
        cfg = FeatureStoreConfig(
            dataset=_two_cat_dataset(),
            output_schema="s",
            output_table_prefix="p_",
            time_windows=[3],
            include_marginals=True,
            include_ratios=True,
        )
        assert cfg.ratio_mode == RatioMode.ALL_PROJECTIONS
