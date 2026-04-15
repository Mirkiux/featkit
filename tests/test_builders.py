"""Tests for Plans 09, 10, and 11 — space builders."""

from __future__ import annotations

import pytest

from featkit.builders.distributional_space import DistributionalSpaceBuilder
from featkit.builders.pivot_space import PivotSpaceBuilder
from featkit.builders.temporal_space import _COMPOSED_OPERATORS, TemporalSpaceBuilder
from featkit.contracts.measurement.defaults import get_default_contract
from featkit.dataset.base import SimpleDataset
from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    Layer2Aggregator,
    Layer2OutputType,
    MeasurementType,
    TemporalOperator,
    TimeGranularity,
    TimeWindowDirection,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.layer2.distributional import DistributionalColumn
from featkit.layer2.pivoted import PivotedColumn
from featkit.layer3.temporal_feature import _POINT_IN_TIME_OPERATORS, TemporalFeature

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _id() -> IDField:
    return IDField("id")


def _ts() -> TimeField:
    return TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY)


def _mto() -> MeasurementField:
    return MeasurementField("mto", MeasurementType.MONTO)


def _cnt() -> MeasurementField:
    return MeasurementField("cnt", MeasurementType.CANTIDAD)


_BWD = TimeWindowDirection.BACKWARD
_FWD = TimeWindowDirection.FORWARD


def _numeric_col() -> PivotedColumn:
    return PivotedColumn(MeasurementField("mto", MeasurementType.MONTO), Layer2Aggregator.SUM)


def _flag_col() -> PivotedColumn:
    return PivotedColumn(MeasurementField("flag", MeasurementType.FLAG), Layer2Aggregator.MAX)


# ---------------------------------------------------------------------------
# Plan 09 — PivotSpaceBuilder
# ---------------------------------------------------------------------------


class TestPivotSpaceBuilderNoCategoricals:
    def _ds(self) -> SimpleDataset:
        return SimpleDataset("tbl", [_id(), _ts(), _mto()])

    def test_returns_one_set_per_aggregator(self) -> None:
        cols = PivotSpaceBuilder(dataset=self._ds()).build()
        # MONTO: SUM, MAX, MIN, AVG → 4 columns
        assert len(cols) == 4

    def test_all_results_are_pivoted_columns(self) -> None:
        cols = PivotSpaceBuilder(dataset=self._ds()).build()
        assert all(isinstance(c, PivotedColumn) for c in cols)

    def test_clean_inputs_produce_no_duplicate_names(self) -> None:
        cols = PivotSpaceBuilder(dataset=self._ds()).build()
        names = [c.column_name for c in cols]
        assert len(names) == len(set(names))


class TestPivotSpaceBuilderMarginals:
    def _ds_one_cat(self) -> SimpleDataset:
        return SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector", CategoricalTreatment.PIVOT, allowed_values=["A", "B", "C"]
                ),
            ],
        )

    def test_include_marginals_true_adds_null_combination(self) -> None:
        cols = PivotSpaceBuilder(dataset=self._ds_one_cat(), include_marginals=True).build()
        # (3 values + 1 marginal) × 4 aggs = 16
        assert len(cols) == 4 * 4

    def test_include_marginals_false_excludes_null(self) -> None:
        cols = PivotSpaceBuilder(dataset=self._ds_one_cat(), include_marginals=False).build()
        # 3 values × 4 aggs = 12
        assert len(cols) == 3 * 4

    def test_two_categoricals_three_values_with_marginals(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector", CategoricalTreatment.PIVOT, allowed_values=["A", "B", "C"]
                ),
                CategoricalField(
                    "region", CategoricalTreatment.PIVOT, allowed_values=["X", "Y", "Z"]
                ),
            ],
        )
        cols = PivotSpaceBuilder(dataset=ds, include_marginals=True).build()
        # (3+1)² × 4 aggs = 64
        assert len(cols) == (3 + 1) ** 2 * 4

    def test_clean_domain_with_marginals_produces_no_duplicates(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A", "B"]),
            ],
        )
        cols = PivotSpaceBuilder(dataset=ds, include_marginals=True).build()
        names = [c.column_name for c in cols]
        assert len(names) == len(set(names))


class TestPivotSpaceBuilderTreatmentFiltering:
    def test_distributional_only_cats_excluded(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "region",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
            ],
        )
        cols = PivotSpaceBuilder(dataset=ds, include_marginals=False).build()
        # No pivot-eligible categoricals → same as no-cat case: 4 MONTO aggs
        assert len(cols) == 4

    def test_both_treatment_is_pivot_eligible(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.BOTH,
                    allowed_values=["A", "B"],
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
            ],
        )
        cols = PivotSpaceBuilder(dataset=ds, include_marginals=False).build()
        assert len(cols) == 2 * 4


class TestPivotSpaceBuilderDomainResolution:
    def _ds_no_values(self) -> SimpleDataset:
        return SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField("sector", CategoricalTreatment.PIVOT),
            ],
        )

    def test_raises_when_no_allowed_values_and_no_resolver(self) -> None:
        with pytest.raises(ValueError, match="domain_resolver"):
            PivotSpaceBuilder(dataset=self._ds_no_values()).build()

    def test_domain_resolver_is_invoked(self) -> None:
        calls: list[str] = []

        def resolver(f: CategoricalField) -> list[str]:
            calls.append(f.name)
            return ["X", "Y"]

        cols = PivotSpaceBuilder(
            dataset=self._ds_no_values(),
            domain_resolver=resolver,
            include_marginals=False,
        ).build()
        assert "sector" in calls
        assert len(cols) == 2 * 4

    def test_static_allowed_values_bypass_resolver(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A"]),
            ],
        )
        calls: list[str] = []

        def resolver(f: CategoricalField) -> list[str]:
            calls.append(f.name)
            return ["Z"]

        PivotSpaceBuilder(dataset=ds, domain_resolver=resolver, include_marginals=False).build()
        assert calls == []

    def test_domain_containing_none_raises(self) -> None:
        def bad_resolver(f: CategoricalField) -> list[str]:
            return ["A", None]  # type: ignore[list-item]

        with pytest.raises(ValueError, match="None"):
            PivotSpaceBuilder(
                dataset=self._ds_no_values(),
                domain_resolver=bad_resolver,
            ).build()


class TestPivotSpaceBuilderDuplicateDetection:
    def test_duplicate_domain_values_raise(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A", "A"]),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate pivot column name"):
            PivotSpaceBuilder(dataset=ds, include_marginals=False).build()


class TestPivotSpaceBuilderAggregatorsOverride:
    def test_override_restricts_to_specified_aggregators(self) -> None:
        ds = SimpleDataset("tbl", [_id(), _ts(), _mto()])
        cols = PivotSpaceBuilder(
            dataset=ds,
            aggregators_override={MeasurementType.MONTO: [Layer2Aggregator.SUM]},
        ).build()
        assert len(cols) == 1
        assert cols[0].layer2_aggregator == Layer2Aggregator.SUM

    def test_override_invalid_aggregators_are_skipped(self) -> None:
        # FLAG only allows MAX; COUNT is invalid → skipped
        ds = SimpleDataset(
            "tbl",
            [_id(), _ts(), MeasurementField("flag", MeasurementType.FLAG)],
        )
        cols = PivotSpaceBuilder(
            dataset=ds,
            aggregators_override={
                MeasurementType.FLAG: [Layer2Aggregator.COUNT, Layer2Aggregator.MAX]
            },
        ).build()
        assert len(cols) == 1
        assert cols[0].layer2_aggregator == Layer2Aggregator.MAX

    def test_override_only_applies_to_specified_type(self) -> None:
        ds = SimpleDataset("tbl", [_id(), _ts(), _mto(), _cnt()])
        cols = PivotSpaceBuilder(
            dataset=ds,
            aggregators_override={MeasurementType.MONTO: [Layer2Aggregator.SUM]},
        ).build()
        mto_cols = [
            c for c in cols if c.source_measurement.measurement_type == MeasurementType.MONTO
        ]
        cnt_cols = [
            c for c in cols if c.source_measurement.measurement_type == MeasurementType.CANTIDAD
        ]
        assert len(mto_cols) == 1  # overridden to SUM only
        assert len(cnt_cols) == 1  # CANTIDAD: SUM (contract default)


# ---------------------------------------------------------------------------
# Plan 10 — DistributionalSpaceBuilder
# ---------------------------------------------------------------------------


class TestDistributionalSpaceBuilderBasic:
    def _ds(self) -> SimpleDataset:
        return SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[
                        DistributionalMetric.ENTROPY,
                        DistributionalMetric.HHI,
                    ],
                ),
            ],
        )

    def test_returns_distributional_columns(self) -> None:
        cols = DistributionalSpaceBuilder(dataset=self._ds()).build()
        assert all(isinstance(c, DistributionalColumn) for c in cols)

    def test_column_count(self) -> None:
        # 1 cat x MONTO (4 aggs: SUM,MAX,MIN,AVG) x 2 metrics = 8
        cols = DistributionalSpaceBuilder(dataset=self._ds()).build()
        assert len(cols) == 4 * 2

    def test_no_duplicates(self) -> None:
        cols = DistributionalSpaceBuilder(dataset=self._ds()).build()
        names = [c.column_name for c in cols]
        assert len(names) == len(set(names))

    def test_all_metrics_represented(self) -> None:
        cols = DistributionalSpaceBuilder(dataset=self._ds()).build()
        metrics = {c.distributional_metric for c in cols}
        assert DistributionalMetric.ENTROPY in metrics
        assert DistributionalMetric.HHI in metrics


class TestDistributionalSpaceBuilderTreatmentFiltering:
    def test_pivot_only_cats_excluded(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField("region", CategoricalTreatment.PIVOT, allowed_values=["A"]),
            ],
        )
        cols = DistributionalSpaceBuilder(dataset=ds).build()
        assert len(cols) == 0

    def test_both_treatment_included(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.BOTH,
                    allowed_values=["A"],
                    distributional_metrics=[DistributionalMetric.MODE],
                ),
            ],
        )
        cols = DistributionalSpaceBuilder(dataset=ds).build()
        # 1 cat x MONTO (4 aggs) x 1 metric = 4
        assert len(cols) == 4

    def test_distributional_treatment_included(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
            ],
        )
        cols = DistributionalSpaceBuilder(dataset=ds).build()
        assert len(cols) > 0


class TestDistributionalSpaceBuilderValueMeasurements:
    def _ds_two_meas(self) -> SimpleDataset:
        return SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                _cnt(),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
            ],
        )

    def test_all_measurements_used_when_none(self) -> None:
        cols = DistributionalSpaceBuilder(dataset=self._ds_two_meas()).build()
        # MONTO: 4 aggs, CANTIDAD: 1 agg (SUM) -> 5 x 1 metric = 5
        assert len(cols) == 5

    def test_custom_value_measurements_restricts_sources(self) -> None:
        cols = DistributionalSpaceBuilder(
            dataset=self._ds_two_meas(),
            value_measurements=[_mto()],
        ).build()
        # Only MONTO: 4 aggs x 1 metric = 4
        assert len(cols) == 4
        assert all(c.source_measurement.measurement_type == MeasurementType.MONTO for c in cols)

    def test_empty_value_measurements_produces_no_columns(self) -> None:
        cols = DistributionalSpaceBuilder(
            dataset=self._ds_two_meas(),
            value_measurements=[],
        ).build()
        assert len(cols) == 0

    def test_value_measurement_not_in_dataset_raises(self) -> None:
        unknown = MeasurementField("unknown", MeasurementType.TICKET)
        with pytest.raises(ValueError, match="'unknown'"):
            DistributionalSpaceBuilder(
                dataset=self._ds_two_meas(),
                value_measurements=[unknown],
            ).build()

    def test_value_measurement_with_mismatched_contract_raises(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                MeasurementField(
                    "mto",
                    MeasurementType.MONTO,
                    contract=get_default_contract(MeasurementType.MONTO),
                ),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
            ],
        )
        with pytest.raises(ValueError, match="'mto'"):
            DistributionalSpaceBuilder(
                dataset=ds,
                value_measurements=[MeasurementField("mto", MeasurementType.MONTO)],
            ).build()


class TestDistributionalSpaceBuilderMultipleCategoricals:
    def test_two_cats_multiply_output(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                _id(),
                _ts(),
                _mto(),
                CategoricalField(
                    "sector",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.ENTROPY],
                ),
                CategoricalField(
                    "region",
                    CategoricalTreatment.DISTRIBUTIONAL,
                    distributional_metrics=[DistributionalMetric.HHI],
                ),
            ],
        )
        cols = DistributionalSpaceBuilder(dataset=ds).build()
        # (sector: 4 aggs x 1 metric) + (region: 4 aggs x 1 metric) = 8
        assert len(cols) == 8


# ---------------------------------------------------------------------------
# Plan 11 — TemporalSpaceBuilder
# ---------------------------------------------------------------------------


class TestTemporalSpaceBuilderPITOperators:
    def test_pit_operators_get_no_window_size(self) -> None:
        feats = TemporalSpaceBuilder(layer2_columns=[_numeric_col()], time_windows=[3, 6]).build()
        pit = [f for f in feats if f.operator in _POINT_IN_TIME_OPERATORS]
        assert pit, "expected at least one PIT feature"
        assert all(f.window_size is None for f in pit)

    def test_windowed_operators_use_time_windows(self) -> None:
        feats = TemporalSpaceBuilder(layer2_columns=[_numeric_col()], time_windows=[3, 6]).build()
        windowed = [
            f
            for f in feats
            if f.operator not in _POINT_IN_TIME_OPERATORS and f.operator not in _COMPOSED_OPERATORS
        ]
        sizes = {f.window_size for f in windowed}
        assert 3 in sizes
        assert 6 in sizes

    def test_all_features_are_temporal_feature_instances(self) -> None:
        feats = TemporalSpaceBuilder(layer2_columns=[_numeric_col()], time_windows=[3]).build()
        assert all(isinstance(f, TemporalFeature) for f in feats)


class TestTemporalSpaceBuilderComposedOperators:
    def test_composed_operators_skipped_when_no_composed_windows(self) -> None:
        feats = TemporalSpaceBuilder(
            layer2_columns=[_numeric_col()], time_windows=[3], composed_windows=None
        ).build()
        assert not any(f.operator in _COMPOSED_OPERATORS for f in feats)

    def test_composed_operators_present_when_composed_windows_provided(self) -> None:
        feats = TemporalSpaceBuilder(
            layer2_columns=[_numeric_col()], time_windows=[3], composed_windows=[6, 12]
        ).build()
        composed = [f for f in feats if f.operator in _COMPOSED_OPERATORS]
        assert composed, "expected at least one composed-window feature"

    def test_composed_operators_use_composed_windows(self) -> None:
        feats = TemporalSpaceBuilder(
            layer2_columns=[_numeric_col()], time_windows=[3], composed_windows=[6, 12]
        ).build()
        composed = [f for f in feats if f.operator in _COMPOSED_OPERATORS]
        sizes = {f.window_size for f in composed}
        assert 6 in sizes
        assert 12 in sizes

    def test_composed_operators_do_not_use_standard_windows(self) -> None:
        feats = TemporalSpaceBuilder(
            layer2_columns=[_numeric_col()], time_windows=[3], composed_windows=[6, 12]
        ).build()
        composed = [f for f in feats if f.operator in _COMPOSED_OPERATORS]
        assert not any(f.window_size == 3 for f in composed)


class TestTemporalSpaceBuilderOperatorsOverride:
    def test_override_restricts_to_specified_operators(self) -> None:
        feats = TemporalSpaceBuilder(
            layer2_columns=[_numeric_col()],
            time_windows=[3],
            operators_override={Layer2OutputType.NUMERIC: [TemporalOperator.PROM_U]},
        ).build()
        assert all(f.operator == TemporalOperator.PROM_U for f in feats)

    def test_override_contract_invalid_operators_filtered(self) -> None:
        # FLAG output allows: ULT_MES, PREV_MES, FREQ, XM, REC
        # PROM_U is not valid for FLAG -> filtered out
        feats = TemporalSpaceBuilder(
            layer2_columns=[_flag_col()],
            time_windows=[3],
            operators_override={
                Layer2OutputType.FLAG: [TemporalOperator.PROM_U, TemporalOperator.FREQ]
            },
        ).build()
        operators = {f.operator for f in feats}
        assert TemporalOperator.PROM_U not in operators
        assert TemporalOperator.FREQ in operators

    def test_override_not_applied_to_other_output_types(self) -> None:
        flag_col = _flag_col()
        feats = TemporalSpaceBuilder(
            layer2_columns=[flag_col],
            time_windows=[3],
            operators_override={Layer2OutputType.NUMERIC: [TemporalOperator.PROM_U]},
        ).build()
        # Override is for NUMERIC, flag_col is FLAG -> override does not apply
        operators = {f.operator for f in feats}
        assert TemporalOperator.PROM_U not in operators


class TestTemporalSpaceBuilderNoDuplicates:
    def test_duplicate_layer2_columns_deduplicated(self) -> None:
        col = _numeric_col()
        feats = TemporalSpaceBuilder(layer2_columns=[col, col], time_windows=[3]).build()
        names = [f.column_name for f in feats]
        assert len(names) == len(set(names))

    def test_no_duplicates_in_standard_build(self) -> None:
        feats = TemporalSpaceBuilder(layer2_columns=[_numeric_col()], time_windows=[3, 6]).build()
        names = [f.column_name for f in feats]
        assert len(names) == len(set(names))


class TestTemporalSpaceBuilderDirection:
    def test_direction_applied_to_all_features(self) -> None:
        feats = TemporalSpaceBuilder(
            layer2_columns=[_numeric_col()], time_windows=[3], direction=_FWD
        ).build()
        assert all(f.direction == _FWD for f in feats)

    def test_default_direction_is_backward(self) -> None:
        feats = TemporalSpaceBuilder(layer2_columns=[_numeric_col()], time_windows=[3]).build()
        assert all(f.direction == _BWD for f in feats)
