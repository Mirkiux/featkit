"""Layer 2 → Layer 3 output type contracts."""

from featkit.contracts.output.base import AbstractLayer2OutputContract
from featkit.contracts.output.defaults import (
    CategoricalOutputContract,
    FlagOutputContract,
    NumericOutputContract,
    TemporalOutputContract,
    get_default_output_contract,
)

__all__ = [
    "AbstractLayer2OutputContract",
    "CategoricalOutputContract",
    "FlagOutputContract",
    "NumericOutputContract",
    "TemporalOutputContract",
    "get_default_output_contract",
]
