"""Tests for Plan 01 — all featkit enumerators."""

import pytest

from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    FieldRole,
    Layer2Aggregator,
    Layer2OutputType,
    MeasurementType,
    TemporalOperator,
    TimeGranularity,
    TimeWindowDirection,
)


class TestFieldRole:
    def test_all_values_present(self) -> None:
        assert {m.name for m in FieldRole} == {"ID", "TIME", "CATEGORICAL", "MEASUREMENT"}

    def test_value_equals_name(self) -> None:
        for member in FieldRole:
            assert member.value == member.name

    def test_individual_members(self) -> None:
        assert FieldRole.ID
        assert FieldRole.TIME
        assert FieldRole.CATEGORICAL
        assert FieldRole.MEASUREMENT


class TestMeasurementType:
    def test_all_values_present(self) -> None:
        expected = {
            "MONTO", "CANTIDAD", "TICKET", "FLAG",
            "FECHA", "BALANCE", "TIME_DIFF", "ESTADISTICO",
        }
        assert {m.name for m in MeasurementType} == expected

    def test_value_equals_name(self) -> None:
        for member in MeasurementType:
            assert member.value == member.name

    def test_cantidad_not_frecuencia(self) -> None:
        assert hasattr(MeasurementType, "CANTIDAD")
        assert not hasattr(MeasurementType, "FRECUENCIA")


class TestTimeGranularity:
    def test_all_values_present(self) -> None:
        expected = {"DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY"}
        assert {m.name for m in TimeGranularity} == expected

    def test_value_equals_name(self) -> None:
        for member in TimeGranularity:
            assert member.value == member.name


class TestCategoricalTreatment:
    def test_all_values_present(self) -> None:
        assert {m.name for m in CategoricalTreatment} == {"PIVOT", "DISTRIBUTIONAL", "BOTH"}

    def test_value_equals_name(self) -> None:
        for member in CategoricalTreatment:
            assert member.value == member.name


class TestLayer2Aggregator:
    def test_all_values_present(self) -> None:
        assert {m.name for m in Layer2Aggregator} == {"SUM", "COUNT", "MAX", "MIN", "AVG"}

    def test_value_equals_name(self) -> None:
        for member in Layer2Aggregator:
            assert member.value == member.name


class TestDistributionalMetric:
    def test_all_values_present(self) -> None:
        expected = {"ENTROPY", "HHI", "DOMINANT_PROPORTION", "MODE", "COUNT"}
        assert {m.name for m in DistributionalMetric} == expected

    def test_value_equals_name(self) -> None:
        for member in DistributionalMetric:
            assert member.value == member.name


class TestLayer2OutputType:
    def test_all_values_present(self) -> None:
        assert {m.name for m in Layer2OutputType} == {"NUMERIC", "FLAG", "CATEGORICAL", "TEMPORAL"}

    def test_value_equals_name(self) -> None:
        for member in Layer2OutputType:
            assert member.value == member.name

    def test_four_values_only(self) -> None:
        assert len(Layer2OutputType) == 4


class TestTemporalOperator:
    def test_all_values_present(self) -> None:
        expected = {
            "PROM_U", "PROM_P", "SUM_U", "SUM_P",
            "ULT_MES", "PREV_MES", "CREC", "FREQ",
            "MIN_U", "MAX_U", "REC", "XM", "MEDIA_ABS", "RATIO",
        }
        assert {m.name for m in TemporalOperator} == expected

    def test_value_equals_name(self) -> None:
        for member in TemporalOperator:
            assert member.value == member.name

    def test_fourteen_operators(self) -> None:
        assert len(TemporalOperator) == 14


class TestTimeWindowDirection:
    def test_all_values_present(self) -> None:
        assert {m.name for m in TimeWindowDirection} == {"BACKWARD", "FORWARD"}

    def test_value_equals_name(self) -> None:
        for member in TimeWindowDirection:
            assert member.value == member.name
