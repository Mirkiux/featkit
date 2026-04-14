"""Default concrete output contracts — one per Layer2OutputType."""

from __future__ import annotations

from featkit.contracts.output.base import AbstractLayer2OutputContract
from featkit.enums import Layer2OutputType, TemporalOperator

_PU = TemporalOperator.PROM_U
_PP = TemporalOperator.PROM_P
_SU = TemporalOperator.SUM_U
_SP = TemporalOperator.SUM_P
_UM = TemporalOperator.ULT_MES
_PM = TemporalOperator.PREV_MES
_CR = TemporalOperator.CREC
_FR = TemporalOperator.FREQ
_MNU = TemporalOperator.MIN_U
_MXU = TemporalOperator.MAX_U
_RC = TemporalOperator.REC
_XM = TemporalOperator.XM
_MA = TemporalOperator.MEDIA_ABS
_RA = TemporalOperator.RATIO


class NumericOutputContract(AbstractLayer2OutputContract):
    def __init__(self) -> None:
        super().__init__(Layer2OutputType.NUMERIC)

    @property
    def valid_temporal_operators(self) -> frozenset[TemporalOperator]:
        return frozenset({_PU, _PP, _SU, _SP, _CR, _MNU, _MXU, _UM, _PM, _FR, _XM, _MA, _RA})


class FlagOutputContract(AbstractLayer2OutputContract):
    def __init__(self) -> None:
        super().__init__(Layer2OutputType.FLAG)

    @property
    def valid_temporal_operators(self) -> frozenset[TemporalOperator]:
        return frozenset({_UM, _PM, _FR, _XM, _RC})


class CategoricalOutputContract(AbstractLayer2OutputContract):
    def __init__(self) -> None:
        super().__init__(Layer2OutputType.CATEGORICAL)

    @property
    def valid_temporal_operators(self) -> frozenset[TemporalOperator]:
        return frozenset({_UM, _PM, _RC})


class TemporalOutputContract(AbstractLayer2OutputContract):
    def __init__(self) -> None:
        super().__init__(Layer2OutputType.TEMPORAL)

    @property
    def valid_temporal_operators(self) -> frozenset[TemporalOperator]:
        return frozenset({_UM, _PM, _RC, _MNU, _MXU, _CR})


_DEFAULTS: dict[Layer2OutputType, AbstractLayer2OutputContract] = {
    Layer2OutputType.NUMERIC: NumericOutputContract(),
    Layer2OutputType.FLAG: FlagOutputContract(),
    Layer2OutputType.CATEGORICAL: CategoricalOutputContract(),
    Layer2OutputType.TEMPORAL: TemporalOutputContract(),
}


def get_default_output_contract(ot: Layer2OutputType) -> AbstractLayer2OutputContract:
    """Return the default contract for the given :class:`~featkit.enums.Layer2OutputType`."""
    return _DEFAULTS[ot]
