"""Default concrete contracts — one per MeasurementType."""

from __future__ import annotations

from featkit.contracts.measurement.base import AbstractMeasurementTypeContract
from featkit.enums import Layer2Aggregator, MeasurementType

_S = Layer2Aggregator.SUM
_C = Layer2Aggregator.COUNT
_MX = Layer2Aggregator.MAX
_MN = Layer2Aggregator.MIN
_AV = Layer2Aggregator.AVG


class MontoContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_S, _MX, _MN, _AV})

    def __init__(self) -> None:
        super().__init__(MeasurementType.MONTO)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class CantidadContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_S})

    def __init__(self) -> None:
        super().__init__(MeasurementType.CANTIDAD)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class TicketContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_AV})

    def __init__(self) -> None:
        super().__init__(MeasurementType.TICKET)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class FlagContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_MX})

    def __init__(self) -> None:
        super().__init__(MeasurementType.FLAG)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class FechaContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_MX, _MN})

    def __init__(self) -> None:
        super().__init__(MeasurementType.FECHA)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class BalanceContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_MX, _MN, _AV})

    def __init__(self) -> None:
        super().__init__(MeasurementType.BALANCE)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class TimeDiffContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_S, _AV, _MX, _MN})

    def __init__(self) -> None:
        super().__init__(MeasurementType.TIME_DIFF)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


class EstadisticoContract(AbstractMeasurementTypeContract):
    _AGGREGATORS: frozenset[Layer2Aggregator] = frozenset({_S, _AV, _MX, _MN, _C})

    def __init__(self) -> None:
        super().__init__(MeasurementType.ESTADISTICO)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return self._AGGREGATORS


_DEFAULTS: dict[MeasurementType, AbstractMeasurementTypeContract] = {
    MeasurementType.MONTO: MontoContract(),
    MeasurementType.CANTIDAD: CantidadContract(),
    MeasurementType.TICKET: TicketContract(),
    MeasurementType.FLAG: FlagContract(),
    MeasurementType.FECHA: FechaContract(),
    MeasurementType.BALANCE: BalanceContract(),
    MeasurementType.TIME_DIFF: TimeDiffContract(),
    MeasurementType.ESTADISTICO: EstadisticoContract(),
}


def get_default_contract(mt: MeasurementType) -> AbstractMeasurementTypeContract:
    """Return the default contract for the given :class:`~featkit.enums.MeasurementType`."""
    return _DEFAULTS[mt]
