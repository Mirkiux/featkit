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
        self.measurement_type = measurement_type

    @property
    @abstractmethod
    def valid_layer2_aggregators(self) -> frozenset[Layer2Aggregator]:
        """Frozen set of aggregators that are semantically valid for this measurement type."""
        ...

    def is_valid(self, aggregator: Layer2Aggregator) -> bool:
        """Return ``True`` if the aggregator is permitted by this contract."""
        return aggregator in self.valid_layer2_aggregators

    def __repr__(self) -> str:
        return f"{type(self).__name__}(measurement_type={self.measurement_type.name})"
