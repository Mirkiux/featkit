"""Dataset descriptors — pure schema metadata for the source facts table."""

from __future__ import annotations

from abc import ABC

from featkit.enums import FieldRole
from featkit.fields.base import AbstractField
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField


class AbstractDataset(ABC):  # noqa: B024
    """Schema descriptor for an input facts table.

    Holds pure metadata — no data access, no materialisation. Derived
    field-role properties filter :attr:`fields` by role; :meth:`validate`
    asserts structural consistency.

    Args:
        source_reference: Fully-qualified table name or SQL string that
            identifies the source of the facts.
        fields: All columns in the facts table.
    """

    def __init__(self, source_reference: str, fields: list[AbstractField]) -> None:
        self.source_reference = source_reference
        self.fields: list[AbstractField] = list(fields)

    # ------------------------------------------------------------------
    # Derived field-role properties
    # ------------------------------------------------------------------

    @property
    def id_fields(self) -> list[IDField]:
        """All fields whose role is :attr:`~featkit.enums.FieldRole.ID`."""
        return [f for f in self.fields if isinstance(f, IDField)]

    @property
    def time_field(self) -> TimeField:
        """The single :class:`~featkit.fields.TimeField` in this dataset.

        Raises:
            ValueError: If no TIME field exists or more than one does.
        """
        time_fields = [f for f in self.fields if isinstance(f, TimeField)]
        if len(time_fields) == 0:
            raise ValueError("Dataset has no TIME field; exactly one is required")
        if len(time_fields) > 1:
            raise ValueError(f"Dataset has {len(time_fields)} TIME fields; exactly one is required")
        return time_fields[0]

    @property
    def categorical_fields(self) -> list[CategoricalField]:
        """All fields whose role is :attr:`~featkit.enums.FieldRole.CATEGORICAL`."""
        return [f for f in self.fields if isinstance(f, CategoricalField)]

    @property
    def measurement_fields(self) -> list[MeasurementField]:
        """All fields whose role is :attr:`~featkit.enums.FieldRole.MEASUREMENT`."""
        return [f for f in self.fields if isinstance(f, MeasurementField)]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Assert structural consistency of the dataset schema.

        Checks:
        - Exactly one TIME field
        - At least one ID field
        - At least one MEASUREMENT field
        - No duplicate field names

        Raises:
            ValueError: With a description of every violation found.
        """
        violations: list[str] = []

        time_count = sum(1 for f in self.fields if f.role == FieldRole.TIME)
        if time_count == 0:
            violations.append("no TIME field found; exactly one is required")
        elif time_count > 1:
            violations.append(f"{time_count} TIME fields found; exactly one is required")

        if not any(f.role == FieldRole.ID for f in self.fields):
            violations.append("no ID field found; at least one is required")

        if not any(f.role == FieldRole.MEASUREMENT for f in self.fields):
            violations.append("no MEASUREMENT field found; at least one is required")

        seen: set[str] = set()
        duplicates: set[str] = set()
        for f in self.fields:
            if f.name in seen:
                duplicates.add(f.name)
            seen.add(f.name)
        if duplicates:
            violations.append(f"duplicate field name(s): {', '.join(sorted(duplicates))}")

        if violations:
            raise ValueError(
                "Dataset validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"source_reference={self.source_reference!r}, "
            f"fields={len(self.fields)})"
        )


class SimpleDataset(AbstractDataset):
    """Concrete dataset descriptor — the standard entry point for schema definition.

    Accepts ``source_reference`` and ``fields`` directly; no subclassing needed.
    """

    def __init__(self, source_reference: str, fields: list[AbstractField]) -> None:
        super().__init__(source_reference, fields)
