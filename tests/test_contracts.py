"""Tests for Plan 04 — AbstractMeasurementTypeContract and concrete defaults."""

import pytest

from featkit.contracts.measurement.base import AbstractMeasurementTypeContract
from featkit.contracts.measurement.defaults import (
    BalanceContract,
    CantidadContract,
    EstadisticoContract,
    FechaContract,
    FlagContract,
    MontoContract,
    TicketContract,
    TimeDiffContract,
    get_default_contract,
)
from featkit.enums import Layer2Aggregator, MeasurementType


class TestAbstractMeasurementTypeContract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AbstractMeasurementTypeContract(MeasurementType.MONTO)  # type: ignore[abstract]

    def test_is_valid_true_for_valid_aggregator(self) -> None:
        assert MontoContract().is_valid(Layer2Aggregator.SUM) is True

    def test_is_valid_false_for_invalid_aggregator(self) -> None:
        assert TicketContract().is_valid(Layer2Aggregator.SUM) is False

    def test_repr_contains_class_and_measurement_type(self) -> None:
        r = repr(MontoContract())
        assert "MontoContract" in r
        assert "MONTO" in r


class TestConcreteContracts:
    @pytest.mark.parametrize(
        "contract_cls,mt,expected_aggs",
        [
            (
                MontoContract,
                MeasurementType.MONTO,
                {
                    Layer2Aggregator.SUM,
                    Layer2Aggregator.MAX,
                    Layer2Aggregator.MIN,
                    Layer2Aggregator.AVG,
                },
            ),
            (
                CantidadContract,
                MeasurementType.CANTIDAD,
                {Layer2Aggregator.SUM},
            ),
            (
                TicketContract,
                MeasurementType.TICKET,
                {Layer2Aggregator.AVG},
            ),
            (
                FlagContract,
                MeasurementType.FLAG,
                {Layer2Aggregator.MAX},
            ),
            (
                FechaContract,
                MeasurementType.FECHA,
                {Layer2Aggregator.MAX, Layer2Aggregator.MIN},
            ),
            (
                BalanceContract,
                MeasurementType.BALANCE,
                {Layer2Aggregator.MAX, Layer2Aggregator.MIN, Layer2Aggregator.AVG},
            ),
            (
                TimeDiffContract,
                MeasurementType.TIME_DIFF,
                {
                    Layer2Aggregator.SUM,
                    Layer2Aggregator.AVG,
                    Layer2Aggregator.MAX,
                    Layer2Aggregator.MIN,
                },
            ),
            (
                EstadisticoContract,
                MeasurementType.ESTADISTICO,
                {
                    Layer2Aggregator.SUM,
                    Layer2Aggregator.AVG,
                    Layer2Aggregator.MAX,
                    Layer2Aggregator.MIN,
                    Layer2Aggregator.COUNT,
                },
            ),
        ],
    )
    def test_valid_aggregators(self, contract_cls, mt, expected_aggs) -> None:  # type: ignore[no-untyped-def]
        contract = contract_cls()
        assert contract.measurement_type == mt
        assert contract.valid_layer2_aggregators == frozenset(expected_aggs)

    def test_returns_frozenset(self) -> None:
        instances: list[AbstractMeasurementTypeContract] = [
            MontoContract(),
            CantidadContract(),
            TicketContract(),
            FlagContract(),
            FechaContract(),
            BalanceContract(),
            TimeDiffContract(),
            EstadisticoContract(),
        ]
        for contract in instances:
            assert isinstance(contract.valid_layer2_aggregators, frozenset)

    def test_frozenset_is_non_empty(self) -> None:
        instances: list[AbstractMeasurementTypeContract] = [
            MontoContract(),
            CantidadContract(),
            TicketContract(),
            FlagContract(),
            FechaContract(),
            BalanceContract(),
            TimeDiffContract(),
            EstadisticoContract(),
        ]
        for contract in instances:
            assert len(contract.valid_layer2_aggregators) > 0


class TestGetDefaultContract:
    def test_covers_every_measurement_type(self) -> None:
        for mt in MeasurementType:
            contract = get_default_contract(mt)
            assert isinstance(contract, AbstractMeasurementTypeContract)
            assert contract.measurement_type == mt

    def test_returns_correct_contract_class(self) -> None:
        assert isinstance(get_default_contract(MeasurementType.MONTO), MontoContract)
        assert isinstance(get_default_contract(MeasurementType.CANTIDAD), CantidadContract)
        assert isinstance(get_default_contract(MeasurementType.TICKET), TicketContract)
        assert isinstance(get_default_contract(MeasurementType.FLAG), FlagContract)
        assert isinstance(get_default_contract(MeasurementType.FECHA), FechaContract)
        assert isinstance(get_default_contract(MeasurementType.BALANCE), BalanceContract)
        assert isinstance(get_default_contract(MeasurementType.TIME_DIFF), TimeDiffContract)
        assert isinstance(get_default_contract(MeasurementType.ESTADISTICO), EstadisticoContract)
