"""IDField — identifies the entity being profiled."""

from __future__ import annotations

from featkit.enums import FieldRole
from featkit.fields.base import AbstractField


class IDField(AbstractField):
    """A column that serves as a GROUP BY key identifying the entity (e.g. customer ID)."""

    @property
    def role(self) -> FieldRole:
        return FieldRole.ID
