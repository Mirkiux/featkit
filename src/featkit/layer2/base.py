"""Abstract base for Layer 2 output columns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from featkit.contracts.measurement.defaults import get_default_contract
from featkit.contracts.output.defaults import get_default_output_contract
from featkit.enums import Layer2Aggregator, Layer2OutputType
from featkit.fields.measurement_field import MeasurementField

if TYPE_CHECKING:
    from featkit.contracts.output.base import AbstractLayer2OutputContract

#: Separator used between parts of every Layer 2 column name.
#: Field names and categorical values must not contain this string.
COLUMN_NAME_SEP = "__"


class AbstractL2Column(ABC):
    """Minimal interface for any Layer 2 column.

    All Layer 2 columns — whether pivot aggregations, distributional metrics,
    or derived ratio columns — satisfy this interface.  :class:`TemporalFeature`
    and :class:`TemporalSpaceBuilder` accept any subclass of this base.
    """

    @staticmethod
    def _check_name_part(value: str, description: str) -> None:
        """Raise ``ValueError`` if *value* contains :data:`COLUMN_NAME_SEP`."""
        if COLUMN_NAME_SEP in value:
            raise ValueError(
                f"{description} {value!r} must not contain the column name separator "
                f"{COLUMN_NAME_SEP!r}"
            )

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


class AbstractLayer2Column(AbstractL2Column):
    """Base for Layer 2 columns derived from a single measurement + aggregator.

    Subclasses supply the concrete ``output_type`` and ``column_name``; this
    class derives ``output_contract`` from ``output_type`` automatically and
    validates that the aggregator is permitted by the measurement's contract.

    Raises:
        ValueError: If ``layer2_aggregator`` is not permitted by the
            measurement's contract, or if ``source_measurement.name``
            contains the column name separator.
    """

    def __init__(
        self,
        source_measurement: MeasurementField,
        layer2_aggregator: Layer2Aggregator,
    ) -> None:
        AbstractLayer2Column._check_name_part(source_measurement.name, "source_measurement.name")
        contract = source_measurement.contract or get_default_contract(
            source_measurement.measurement_type
        )
        if not contract.is_valid(layer2_aggregator):
            valid = ", ".join(
                a.name for a in sorted(contract.valid_layer2_aggregators, key=lambda a: a.value)
            )
            raise ValueError(
                f"Layer2Aggregator.{layer2_aggregator.name} is not valid for "
                f"MeasurementType.{source_measurement.measurement_type.name}. "
                f"Valid aggregators: {valid}"
            )
        self.source_measurement = source_measurement
        self.layer2_aggregator = layer2_aggregator

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"measurement={self.source_measurement.name!r}, "
            f"aggregator={self.layer2_aggregator.name!r}, "
            f"output_type={self.output_type.name!r})"
        )
