"""Tests for Plan 06 — AbstractLayer2Column, PivotedColumn, DistributionalColumn."""

from __future__ import annotations

import pytest

from featkit.contracts.output.base import AbstractLayer2OutputContract
from featkit.contracts.output.defaults import (
    CategoricalOutputContract,
    FlagOutputContract,
    NumericOutputContract,
    TemporalOutputContract,
)
from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    Layer2Aggregator,
    Layer2OutputType,
    MeasurementType,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.base import AbstractLayer2Column
from featkit.layer2.distributional import DistributionalColumn
from featkit.layer2.pivoted import PivotedColumn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def monto() -> MeasurementField:
    return MeasurementField("amount", MeasurementType.MONTO)


@pytest.fixture()
def cantidad() -> MeasurementField:
    return MeasurementField("trx_count", MeasurementType.CANTIDAD)


@pytest.fixture()
def flag_field() -> MeasurementField:
    return MeasurementField("is_active", MeasurementType.FLAG)


@pytest.fixture()
def fecha_field() -> MeasurementField:
    return MeasurementField("last_date", MeasurementType.FECHA)


@pytest.fixture()
def channel() -> CategoricalField:
    return CategoricalField(
        "channel",
        CategoricalTreatment.BOTH,
        distributional_metrics=[DistributionalMetric.ENTROPY],
    )


@pytest.fixture()
def region() -> CategoricalField:
    return CategoricalField(
        "region",
        CategoricalTreatment.BOTH,
        distributional_metrics=[DistributionalMetric.MODE],
    )


# ---------------------------------------------------------------------------
# AbstractLayer2Column
# ---------------------------------------------------------------------------


class TestAbstractLayer2Column:
    def test_cannot_instantiate_directly(self, monto: MeasurementField) -> None:
        with pytest.raises(TypeError):
            AbstractLayer2Column(monto, Layer2Aggregator.SUM)  # type: ignore[abstract]

    def test_output_contract_is_abstract_layer2_output_contract(
        self, monto: MeasurementField
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM)
        assert isinstance(col.output_contract, AbstractLayer2OutputContract)

    def test_output_contract_consistent_with_output_type(self, monto: MeasurementField) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM)
        assert col.output_contract.output_type == col.output_type

    def test_repr_contains_class_measurement_aggregator_output_type(
        self, monto: MeasurementField
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM)
        r = repr(col)
        assert "PivotedColumn" in r
        assert "amount" in r
        assert "SUM" in r
        assert "NUMERIC" in r


# ---------------------------------------------------------------------------
# Aggregator validation — enforced in AbstractLayer2Column for all subclasses
# ---------------------------------------------------------------------------


class TestAggregatorValidation:
    def test_valid_aggregator_does_not_raise(self, monto: MeasurementField) -> None:
        PivotedColumn(monto, Layer2Aggregator.SUM)

    def test_invalid_aggregator_raises_value_error_pivoted(self, monto: MeasurementField) -> None:
        with pytest.raises(ValueError, match="Layer2Aggregator.COUNT"):
            PivotedColumn(monto, Layer2Aggregator.COUNT)

    def test_invalid_aggregator_raises_value_error_distributional(
        self, channel: CategoricalField
    ) -> None:
        ticket_field = MeasurementField("ticket_avg", MeasurementType.TICKET)
        # Only AVG is valid for TICKET; SUM must raise
        with pytest.raises(ValueError, match="Layer2Aggregator.SUM"):
            DistributionalColumn(
                ticket_field, Layer2Aggregator.SUM, channel, DistributionalMetric.ENTROPY
            )

    def test_error_message_names_valid_aggregators(self, cantidad: MeasurementField) -> None:
        with pytest.raises(ValueError, match="SUM"):
            PivotedColumn(cantidad, Layer2Aggregator.MAX)

    def test_validation_uses_default_contract_when_field_has_none(self) -> None:
        field = MeasurementField("revenue", MeasurementType.TICKET)
        # Only AVG is valid for TICKET
        PivotedColumn(field, Layer2Aggregator.AVG)
        with pytest.raises(ValueError):
            PivotedColumn(field, Layer2Aggregator.SUM)

    def test_validation_uses_provided_contract(self) -> None:
        from featkit.contracts.measurement.base import AbstractMeasurementTypeContract

        class CountOnlyContract(AbstractMeasurementTypeContract):
            @property
            def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
                return frozenset({Layer2Aggregator.COUNT})

        field = MeasurementField(
            "x", MeasurementType.MONTO, contract=CountOnlyContract(MeasurementType.MONTO)
        )
        PivotedColumn(field, Layer2Aggregator.COUNT)
        with pytest.raises(ValueError):
            PivotedColumn(field, Layer2Aggregator.SUM)


# ---------------------------------------------------------------------------
# Column name separator validation
# ---------------------------------------------------------------------------


class TestColumnNameSeparatorValidation:
    def test_measurement_name_with_separator_raises(self) -> None:
        bad = MeasurementField("rev__enue", MeasurementType.MONTO)
        with pytest.raises(ValueError, match="__"):
            PivotedColumn(bad, Layer2Aggregator.SUM)

    def test_measurement_name_with_separator_raises_distributional(
        self, channel: CategoricalField
    ) -> None:
        bad = MeasurementField("rev__enue", MeasurementType.MONTO)
        with pytest.raises(ValueError, match="__"):
            DistributionalColumn(bad, Layer2Aggregator.SUM, channel, DistributionalMetric.ENTROPY)

    def test_categorical_field_name_with_separator_raises(self, monto: MeasurementField) -> None:
        bad_cat = CategoricalField(
            "chan__nel",
            CategoricalTreatment.PIVOT,
        )
        with pytest.raises(ValueError, match="__"):
            PivotedColumn(monto, Layer2Aggregator.SUM, {bad_cat: "online"})

    def test_categorical_value_with_separator_raises(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        with pytest.raises(ValueError, match="__"):
            PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "on__line"})

    def test_distributional_categorical_name_with_separator_raises(
        self, monto: MeasurementField
    ) -> None:
        bad_cat = CategoricalField(
            "chan__nel",
            CategoricalTreatment.BOTH,
            distributional_metrics=[DistributionalMetric.ENTROPY],
        )
        with pytest.raises(ValueError, match="__"):
            DistributionalColumn(monto, Layer2Aggregator.SUM, bad_cat, DistributionalMetric.ENTROPY)

    def test_valid_names_with_single_underscore_do_not_raise(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "north_east"})


# ---------------------------------------------------------------------------
# PivotedColumn — categorical_combination defensive copy
# ---------------------------------------------------------------------------


class TestPivotedColumnDefensiveCopy:
    def test_mutating_original_dict_does_not_affect_column(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        combo: dict[CategoricalField, str | None] = {channel: "online"}
        col = PivotedColumn(monto, Layer2Aggregator.SUM, combo)
        original_name = col.column_name
        combo[channel] = "offline"  # mutate original
        assert col.column_name == original_name


# ---------------------------------------------------------------------------
# PivotedColumn — output_type mapping
# ---------------------------------------------------------------------------


class TestPivotedColumnOutputType:
    @pytest.mark.parametrize(
        "mt,expected",
        [
            (MeasurementType.MONTO, Layer2OutputType.NUMERIC),
            (MeasurementType.CANTIDAD, Layer2OutputType.NUMERIC),
            (MeasurementType.TICKET, Layer2OutputType.NUMERIC),
            (MeasurementType.BALANCE, Layer2OutputType.NUMERIC),
            (MeasurementType.TIME_DIFF, Layer2OutputType.NUMERIC),
            (MeasurementType.ESTADISTICO, Layer2OutputType.NUMERIC),
            (MeasurementType.FLAG, Layer2OutputType.FLAG),
            (MeasurementType.FECHA, Layer2OutputType.TEMPORAL),
        ],
    )
    def test_output_type(self, mt: MeasurementType, expected: Layer2OutputType) -> None:
        _valid_agg: dict[MeasurementType, Layer2Aggregator] = {
            MeasurementType.MONTO: Layer2Aggregator.SUM,
            MeasurementType.CANTIDAD: Layer2Aggregator.SUM,
            MeasurementType.TICKET: Layer2Aggregator.AVG,
            MeasurementType.BALANCE: Layer2Aggregator.AVG,
            MeasurementType.TIME_DIFF: Layer2Aggregator.SUM,
            MeasurementType.ESTADISTICO: Layer2Aggregator.SUM,
            MeasurementType.FLAG: Layer2Aggregator.MAX,
            MeasurementType.FECHA: Layer2Aggregator.MAX,
        }
        col = PivotedColumn(MeasurementField(mt.value, mt), _valid_agg[mt])
        assert col.output_type == expected

    def test_mapping_is_exhaustive_for_all_measurement_types(self) -> None:
        from featkit.layer2.pivoted import _MT_TO_OUTPUT

        assert set(_MT_TO_OUTPUT.keys()) == set(MeasurementType)

    def test_flag_output_contract_is_flag_contract(self, flag_field: MeasurementField) -> None:
        col = PivotedColumn(flag_field, Layer2Aggregator.MAX)
        assert isinstance(col.output_contract, FlagOutputContract)

    def test_fecha_output_contract_is_temporal_contract(
        self, fecha_field: MeasurementField
    ) -> None:
        col = PivotedColumn(fecha_field, Layer2Aggregator.MAX)
        assert isinstance(col.output_contract, TemporalOutputContract)

    def test_monto_output_contract_is_numeric_contract(self, monto: MeasurementField) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM)
        assert isinstance(col.output_contract, NumericOutputContract)


# ---------------------------------------------------------------------------
# PivotedColumn — column_name
# ---------------------------------------------------------------------------


class TestPivotedColumnName:
    def test_no_categoricals(self, monto: MeasurementField) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM)
        assert col.column_name == "SUM__amount"

    def test_single_categorical_with_value(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "online"})
        assert col.column_name == "SUM__amount__channel_online"

    def test_single_categorical_marginal(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM, {channel: None})
        assert col.column_name == "SUM__amount"

    def test_multiple_categoricals_one_marginal(
        self,
        monto: MeasurementField,
        channel: CategoricalField,
        region: CategoricalField,
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM, {channel: None, region: "north"})
        assert col.column_name == "SUM__amount__region_north"

    def test_multiple_categoricals_all_marginals(
        self,
        monto: MeasurementField,
        channel: CategoricalField,
        region: CategoricalField,
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM, {channel: None, region: None})
        assert col.column_name == "SUM__amount"

    def test_multiple_categoricals_sorted_alphabetically(
        self,
        monto: MeasurementField,
        channel: CategoricalField,
        region: CategoricalField,
    ) -> None:
        col = PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "online", region: "north"})
        assert col.column_name == "SUM__amount__channel_online__region_north"

    def test_column_name_deterministic_regardless_of_insertion_order(
        self,
        monto: MeasurementField,
        channel: CategoricalField,
        region: CategoricalField,
    ) -> None:
        col_a = PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "online", region: "north"})
        col_b = PivotedColumn(monto, Layer2Aggregator.SUM, {region: "north", channel: "online"})
        assert col_a.column_name == col_b.column_name

    def test_distinct_combinations_produce_distinct_names(
        self,
        monto: MeasurementField,
        channel: CategoricalField,
        region: CategoricalField,
    ) -> None:
        names = {
            PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "online"}).column_name,
            PivotedColumn(monto, Layer2Aggregator.SUM, {channel: "offline"}).column_name,
            PivotedColumn(monto, Layer2Aggregator.SUM, {region: "north"}).column_name,
            PivotedColumn(
                monto, Layer2Aggregator.SUM, {channel: "online", region: "north"}
            ).column_name,
            PivotedColumn(monto, Layer2Aggregator.SUM).column_name,
        }
        assert len(names) == 5

    def test_aggregator_reflected_in_name(self, monto: MeasurementField) -> None:
        col_sum = PivotedColumn(monto, Layer2Aggregator.SUM)
        col_max = PivotedColumn(monto, Layer2Aggregator.MAX)
        assert col_sum.column_name != col_max.column_name
        assert "SUM" in col_sum.column_name
        assert "MAX" in col_max.column_name


# ---------------------------------------------------------------------------
# DistributionalColumn — output_type mapping
# ---------------------------------------------------------------------------


class TestDistributionalColumnOutputType:
    @pytest.mark.parametrize(
        "metric,expected",
        [
            (DistributionalMetric.MODE, Layer2OutputType.CATEGORICAL),
            (DistributionalMetric.ENTROPY, Layer2OutputType.NUMERIC),
            (DistributionalMetric.HHI, Layer2OutputType.NUMERIC),
            (DistributionalMetric.DOMINANT_PROPORTION, Layer2OutputType.NUMERIC),
            (DistributionalMetric.COUNT, Layer2OutputType.NUMERIC),
        ],
    )
    def test_output_type(
        self,
        metric: DistributionalMetric,
        expected: Layer2OutputType,
        monto: MeasurementField,
        channel: CategoricalField,
    ) -> None:
        col = DistributionalColumn(monto, Layer2Aggregator.SUM, channel, metric)
        assert col.output_type == expected

    def test_mode_output_contract_is_categorical_contract(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        col = DistributionalColumn(monto, Layer2Aggregator.SUM, channel, DistributionalMetric.MODE)
        assert isinstance(col.output_contract, CategoricalOutputContract)

    def test_entropy_output_contract_is_numeric_contract(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        col = DistributionalColumn(
            monto, Layer2Aggregator.SUM, channel, DistributionalMetric.ENTROPY
        )
        assert isinstance(col.output_contract, NumericOutputContract)


# ---------------------------------------------------------------------------
# DistributionalColumn — column_name
# ---------------------------------------------------------------------------


class TestDistributionalColumnName:
    def test_column_name_format(self, monto: MeasurementField, channel: CategoricalField) -> None:
        col = DistributionalColumn(
            monto, Layer2Aggregator.SUM, channel, DistributionalMetric.ENTROPY
        )
        assert col.column_name == "channel__amount__SUM__ENTROPY"

    def test_column_name_reflects_all_components(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        col = DistributionalColumn(monto, Layer2Aggregator.AVG, channel, DistributionalMetric.HHI)
        assert col.column_name == "channel__amount__AVG__HHI"

    def test_distinct_metrics_produce_distinct_names(
        self, monto: MeasurementField, channel: CategoricalField
    ) -> None:
        names = {
            DistributionalColumn(monto, Layer2Aggregator.SUM, channel, m).column_name
            for m in DistributionalMetric
        }
        assert len(names) == len(list(DistributionalMetric))
