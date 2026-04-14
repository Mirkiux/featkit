"""Abstract base class for all column descriptors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from featkit.enums import FieldRole


class AbstractField(ABC):
    """Base descriptor for a column in the source facts table.

    Concrete subclasses fix :attr:`role` to one of the :class:`~featkit.enums.FieldRole`
    values.  The class carries only schema metadata — no data access.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Column name as it appears in the source table."""
        return self._name

    @property
    @abstractmethod
    def role(self) -> FieldRole:
        """The semantic role this column plays in the framework."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    def _key(self) -> tuple[object, ...]:
        """Return the identity components for equality and hashing.

        Subclasses that carry additional schema-defining metadata should
        override this method and extend the returned tuple.
        """
        return (self.name,)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractField):
            return NotImplemented
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self), self._key()))
