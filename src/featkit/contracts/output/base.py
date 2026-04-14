"""Abstract base for Layer 2 → Layer 3 type boundary contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from featkit.enums import Layer2OutputType, TemporalOperator


class AbstractLayer2OutputContract(ABC):
    """Governs which TemporalOperators are semantically valid for a Layer2OutputType.

    This is the Layer 2 → Layer 3 type boundary: the contract ensures that only
    operators that make semantic sense for the output column type are applied.
    """

    def __init__(self, output_type: Layer2OutputType) -> None:
        self.output_type = output_type

    @property
    @abstractmethod
    def valid_temporal_operators(self) -> frozenset[TemporalOperator]:
        """Frozen set of temporal operators valid for this output type."""
        ...

    def is_valid(self, operator: TemporalOperator) -> bool:
        """Return ``True`` if the operator is permitted by this contract."""
        return operator in self.valid_temporal_operators

    def __repr__(self) -> str:
        return f"{type(self).__name__}(output_type={self.output_type.name})"
