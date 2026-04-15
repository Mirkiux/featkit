"""Tests for Plans 09, 10, and 11 — space builders."""

from __future__ import annotations

from featkit.builders.temporal_space import _COMPOSED_OPERATORS, TemporalSpaceBuilder
from featkit.enums import (
    Layer2Aggregator,
    Layer2OutputType,
    MeasurementType,
    TemporalOperator,
    TimeWindowDirection,
)
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.pivoted import PivotedColumn
from featkit.layer3.temporal_feature import _POINT_IN_TIME_OPERATORS, TemporalFeature

# ---------------------------------------------------------------------------
# Plan 11 — TemporalSpaceBuilder
# ---------------------------------------------------------------------------

_BWD = TimeWindowDirection.BACKWARD
_FWD = TimeWindowDirection.FORWARD


def _numeric_col() -> PivotedColumn:
    return PivotedColumn(MeasurementField("mto", MeasurementType.MONTO), Layer2Aggregator.SUM)


def _flag_col() -> PivotedColumn:
    return PivotedColumn(MeasurementField("flag", MeasurementType.FLAG), Layer2Aggregator.MAX)


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
