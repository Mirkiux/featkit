"""MeasurementField — a numeric or typed measurement column."""

from __future__ import annotations

from typing import TYPE_CHECKING

from featkit.enums import FieldRole, MeasurementType
from featkit.fields.base import AbstractField

if TYPE_CHECKING:
    from featkit.contracts.measurement.base import AbstractMeasurementTypeContract


class MeasurementField(AbstractField):
    """Descriptor for a measurement column.

    Carries a :class:`~featkit.enums.MeasurementType` and an optional reference
    to the contract that governs valid Layer 2 aggregators. The contract is
    typically injected via :func:`~featkit.contracts.measurement.defaults.get_default_contract`
    after Plan 04 is available; ``None`` is valid at schema-definition time.
    """

    def __init__(
        self,
        name: str,
        measurement_type: MeasurementType,
        contract: AbstractMeasurementTypeContract | None = None,
    ) -> None:
        super().__init__(name)
        self.measurement_type = measurement_type
        self.contract = contract

    @property
    def role(self) -> FieldRole:
        return FieldRole.MEASUREMENT

    def _key(self) -> tuple[object, ...]:
        return (self.name, self.measurement_type)
