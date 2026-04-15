"""Abstract base for Layer 1 → Layer 2 type boundary contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from featkit.enums import Layer2Aggregator, MeasurementType


class AbstractMeasurementTypeContract(ABC):
    """Governs which Layer2Aggregators are semantically valid for a MeasurementType.

    This is the Layer 1 → Layer 2 type boundary: the contract ensures that only
    aggregators that make semantic sense for the measurement are applied.
    """

    def __init__(self, measurement_type: MeasurementType) -> None:
        self._measurement_type = measurement_type

    @property
    def measurement_type(self) -> MeasurementType:
        """The MeasurementType this contract governs (read-only)."""
        return self._measurement_type

    @property
    @abstractmethod
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        """Frozen set of aggregators that are semantically valid for this measurement type."""
        ...

    def is_valid(self, aggregator: Layer2Aggregator) -> bool:
        """Return ``True`` if the aggregator is permitted by this contract."""
        return aggregator in self.valid_layer2_aggregators

    def _key(self) -> tuple[object, ...]:
        return (self.measurement_type, self.valid_layer2_aggregators)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractMeasurementTypeContract):
            return NotImplemented
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self), self._key()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(measurement_type={self.measurement_type.name})"
