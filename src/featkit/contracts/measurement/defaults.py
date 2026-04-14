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
    def __init__(self) -> None:
        super().__init__(MeasurementType.MONTO)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_S, _MX, _MN, _AV})


class CantidadContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.CANTIDAD)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_S, _C})


class TicketContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.TICKET)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_AV})


class FlagContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.FLAG)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_MX})


class FechaContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.FECHA)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_MX, _MN})


class BalanceContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.BALANCE)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_MX, _MN, _AV})


class TimeDiffContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.TIME_DIFF)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_S, _AV, _MX, _MN})


class EstadisticoContract(AbstractMeasurementTypeContract):
    def __init__(self) -> None:
        super().__init__(MeasurementType.ESTADISTICO)

    @property
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        return frozenset({_S, _AV, _MX, _MN, _C})


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
