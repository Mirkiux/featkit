"""Tests for Plan 07 — TemporalFeature."""

from __future__ import annotations

import pytest

from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    Layer2Aggregator,
    MeasurementType,
    TemporalOperator,
    TimeWindowDirection,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.distributional import DistributionalColumn
from featkit.layer2.pivoted import PivotedColumn
from featkit.layer3.temporal_feature import _POINT_IN_TIME_OPERATORS, TemporalFeature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def numeric_col() -> PivotedColumn:
    """MONTO / SUM → NUMERIC output type."""
    return PivotedColumn(MeasurementField("amount", MeasurementType.MONTO), Layer2Aggregator.SUM)


@pytest.fixture()
def flag_col() -> PivotedColumn:
    """FLAG / MAX → FLAG output type."""
    return PivotedColumn(MeasurementField("is_active", MeasurementType.FLAG), Layer2Aggregator.MAX)


@pytest.fixture()
def categorical_col() -> DistributionalColumn:
    """MODE metric → CATEGORICAL output type."""
    return DistributionalColumn(
        MeasurementField("amount", MeasurementType.MONTO),
        Layer2Aggregator.SUM,
        CategoricalField("sector", CategoricalTreatment.BOTH, [DistributionalMetric.MODE]),
        DistributionalMetric.MODE,
    )


@pytest.fixture()
def temporal_col() -> PivotedColumn:
    """FECHA / MAX → TEMPORAL output type."""
    return PivotedColumn(MeasurementField("last_date", MeasurementType.FECHA), Layer2Aggregator.MAX)


_BWD = TimeWindowDirection.BACKWARD
_FWD = TimeWindowDirection.FORWARD


# ---------------------------------------------------------------------------
# Operator contract validation
# ---------------------------------------------------------------------------


class TestOperatorContractValidation:
    def test_valid_windowed_operator_accepted(self, numeric_col: PivotedColumn) -> None:
        TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)

    def test_valid_pit_operator_accepted(self, numeric_col: PivotedColumn) -> None:
        TemporalFeature(numeric_col, TemporalOperator.ULT_MES, _BWD)

    def test_invalid_operator_for_output_type_raises(self, flag_col: PivotedColumn) -> None:
        # FLAG output only allows ULT_MES, PREV_MES, FREQ, XM, REC
        with pytest.raises(ValueError, match="TemporalOperator.PROM_U"):
            TemporalFeature(flag_col, TemporalOperator.PROM_U, _BWD, window_size=3)

    def test_error_message_names_valid_operators(self, flag_col: PivotedColumn) -> None:
        with pytest.raises(ValueError, match="ULT_MES"):
            TemporalFeature(flag_col, TemporalOperator.PROM_U, _BWD, window_size=3)

    def test_categorical_output_accepts_rec(self, categorical_col: DistributionalColumn) -> None:
        TemporalFeature(categorical_col, TemporalOperator.REC, _BWD)

    def test_temporal_output_accepts_crec(self, temporal_col: PivotedColumn) -> None:
        TemporalFeature(temporal_col, TemporalOperator.CREC, _BWD, window_size=6)


# ---------------------------------------------------------------------------
# window_size validation
# ---------------------------------------------------------------------------


class TestWindowSizeValidation:
    def test_windowed_operator_with_window_size_accepted(self, numeric_col: PivotedColumn) -> None:
        TemporalFeature(numeric_col, TemporalOperator.SUM_U, _BWD, window_size=6)

    def test_windowed_operator_without_window_size_raises(self, numeric_col: PivotedColumn) -> None:
        with pytest.raises(ValueError, match="windowed operator.*window_size is required"):
            TemporalFeature(numeric_col, TemporalOperator.SUM_U, _BWD)

    def test_pit_operator_without_window_size_accepted(self, numeric_col: PivotedColumn) -> None:
        TemporalFeature(numeric_col, TemporalOperator.ULT_MES, _BWD)

    def test_pit_operator_with_window_size_raises(self, numeric_col: PivotedColumn) -> None:
        with pytest.raises(ValueError, match="point-in-time operator"):
            TemporalFeature(numeric_col, TemporalOperator.ULT_MES, _BWD, window_size=3)

    def test_rec_is_point_in_time(self, flag_col: PivotedColumn) -> None:
        # REC is valid for FLAG output and is point-in-time
        TemporalFeature(flag_col, TemporalOperator.REC, _BWD)
        with pytest.raises(ValueError, match="point-in-time"):
            TemporalFeature(flag_col, TemporalOperator.REC, _BWD, window_size=1)

    def test_prev_mes_is_point_in_time(self, numeric_col: PivotedColumn) -> None:
        TemporalFeature(numeric_col, TemporalOperator.PREV_MES, _BWD)

    def test_window_size_zero_raises(self, numeric_col: PivotedColumn) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=0)

    def test_window_size_negative_raises(self, numeric_col: PivotedColumn) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=-1)

    def test_window_size_one_accepted(self, numeric_col: PivotedColumn) -> None:
        TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=1)

    @pytest.mark.parametrize("op", list(_POINT_IN_TIME_OPERATORS))
    def test_all_pit_operators_reject_window_size(
        self, op: TemporalOperator, numeric_col: PivotedColumn
    ) -> None:
        # Only test operators that are valid for NUMERIC output
        from featkit.contracts.output.defaults import NumericOutputContract

        if NumericOutputContract().is_valid(op):
            with pytest.raises(ValueError, match="point-in-time"):
                TemporalFeature(numeric_col, op, _BWD, window_size=3)


# ---------------------------------------------------------------------------
# column_name
# ---------------------------------------------------------------------------


class TestTemporalFeatureColumnName:
    def test_windowed_column_name_format(self, numeric_col: PivotedColumn) -> None:
        feat = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)
        assert feat.column_name == "SUM__amount__PROM_U__BACKWARD__3"

    def test_pit_column_name_omits_window_size(self, numeric_col: PivotedColumn) -> None:
        feat = TemporalFeature(numeric_col, TemporalOperator.ULT_MES, _BWD)
        assert feat.column_name == "SUM__amount__ULT_MES__BACKWARD"

    def test_column_name_includes_source_column_name(self, numeric_col: PivotedColumn) -> None:
        feat = TemporalFeature(numeric_col, TemporalOperator.SUM_U, _FWD, window_size=6)
        assert feat.column_name.startswith("SUM__amount")

    def test_direction_reflected_in_name(self, numeric_col: PivotedColumn) -> None:
        fwd = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _FWD, window_size=3)
        bwd = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)
        assert "FORWARD" in fwd.column_name
        assert "BACKWARD" in bwd.column_name
        assert fwd.column_name != bwd.column_name

    def test_window_size_reflected_in_name(self, numeric_col: PivotedColumn) -> None:
        feat3 = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)
        feat6 = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=6)
        assert "3" in feat3.column_name
        assert "6" in feat6.column_name
        assert feat3.column_name != feat6.column_name

    def test_column_name_is_deterministic(self, numeric_col: PivotedColumn) -> None:
        feat_a = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)
        feat_b = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)
        assert feat_a.column_name == feat_b.column_name

    def test_column_name_with_categorical_source(
        self, categorical_col: DistributionalColumn
    ) -> None:
        feat = TemporalFeature(categorical_col, TemporalOperator.ULT_MES, _BWD)
        assert feat.column_name == "sector__amount__SUM__MODE__ULT_MES__BACKWARD"


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestTemporalFeatureRepr:
    def test_repr_contains_key_info(self, numeric_col: PivotedColumn) -> None:
        feat = TemporalFeature(numeric_col, TemporalOperator.PROM_U, _BWD, window_size=3)
        r = repr(feat)
        assert "TemporalFeature" in r
        assert "PROM_U" in r
        assert "BACKWARD" in r
        assert "3" in r
