"""Abstract base for Layer 2 output columns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from featkit.contracts.output.defaults import get_default_output_contract
from featkit.enums import Layer2Aggregator, Layer2OutputType
from featkit.fields.measurement_field import MeasurementField

if TYPE_CHECKING:
    from featkit.contracts.output.base import AbstractLayer2OutputContract


class AbstractLayer2Column(ABC):
    """Common base for every column in the Layer 2 horizontal concept table.

    Subclasses supply the concrete ``output_type`` and ``column_name``; this
    class derives ``output_contract`` from ``output_type`` automatically.
    """

    def __init__(
        self,
        source_measurement: MeasurementField,
        layer2_aggregator: Layer2Aggregator,
    ) -> None:
        self.source_measurement = source_measurement
        self.layer2_aggregator = layer2_aggregator

    @property
    @abstractmethod
    def output_type(self) -> Layer2OutputType:
        """Layer 2 output type that governs valid Layer 3 temporal operators."""
        ...

    @property
    def output_contract(self) -> AbstractLayer2OutputContract:
        """Contract for the Layer 2 → Layer 3 boundary, derived from ``output_type``."""
        return get_default_output_contract(self.output_type)

    @property
    @abstractmethod
    def column_name(self) -> str:
        """Deterministic name for this column in the Layer 2 output table."""
        ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"measurement={self.source_measurement.name!r}, "
            f"aggregator={self.layer2_aggregator.name!r}, "
            f"output_type={self.output_type.name!r})"
        )
