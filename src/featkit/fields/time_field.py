"""TimeField — the temporal axis of the facts table."""

from __future__ import annotations

from featkit.enums import FieldRole, TimeGranularity
from featkit.fields.base import AbstractField


class TimeField(AbstractField):
    """Descriptor for the time column.

    Tracks source and target granularity so downstream layers know whether
    date truncation is required before grouping (e.g. daily source → monthly target).
    """

    def __init__(
        self,
        name: str,
        source_granularity: TimeGranularity,
        target_granularity: TimeGranularity,
    ) -> None:
        super().__init__(name)
        self.source_granularity = source_granularity
        self.target_granularity = target_granularity

    @property
    def role(self) -> FieldRole:
        return FieldRole.TIME

    @property
    def truncation_required(self) -> bool:
        """``True`` when source and target granularities differ."""
        return self.source_granularity != self.target_granularity

    def _key(self) -> tuple[object, ...]:
        return (self.name, self.source_granularity, self.target_granularity)

    def __repr__(self) -> str:
        return (
            f"TimeField(name={self.name!r}, "
            f"source={self.source_granularity.name}, "
            f"target={self.target_granularity.name})"
        )
