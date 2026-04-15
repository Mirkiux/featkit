"""Tests for Plans 09, 10, and 11 — space builders."""

from __future__ import annotations

from featkit.builders.distributional_space import DistributionalSpaceBuilder
from featkit.dataset.base import SimpleDataset
from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    MeasurementType,
    TimeGranularity,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.layer2.distributional import DistributionalColumn

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
