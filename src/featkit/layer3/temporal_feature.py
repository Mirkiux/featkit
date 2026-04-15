"""TemporalFeature — a single Layer 3 output column."""

from __future__ import annotations

from featkit.enums import TemporalOperator, TimeWindowDirection
from featkit.layer2.base import AbstractLayer2Column

#: Operators that operate on a single point in time and do not require a window.
_POINT_IN_TIME_OPERATORS: frozenset[TemporalOperator] = frozenset(
    {TemporalOperator.ULT_MES, TemporalOperator.PREV_MES, TemporalOperator.REC}
)


class TemporalFeature:
    """A single Layer 3 output column derived by applying a temporal operator to a Layer 2 column.

    Args:
        source: The Layer 2 column this feature is built from.
        operator: Temporal operator to apply.
        direction: Direction of the sliding window.
        window_size: Number of periods in the window. Required for windowed
            operators; must be ``None`` for point-in-time operators
            (e.g. ``ULT_MES``, ``REC``).

    Raises:
        ValueError: If ``operator`` is not permitted by ``source.output_contract``.
        ValueError: If a windowed operator is used with ``window_size=None``.
        ValueError: If a point-in-time operator is used with a non-``None`` ``window_size``.
        ValueError: If ``window_size`` is not a positive integer when provided.
    """

    def __init__(
        self,
        source: AbstractLayer2Column,
        operator: TemporalOperator,
        direction: TimeWindowDirection,
        window_size: int | None = None,
    ) -> None:
        if not source.output_contract.is_valid(operator):
            valid = ", ".join(
                op.name
                for op in sorted(
                    source.output_contract.valid_temporal_operators, key=lambda o: o.value
                )
            )
            raise ValueError(
                f"TemporalOperator.{operator.name} is not valid for "
                f"output type {source.output_type.name}. "
                f"Valid operators: {valid}"
            )

        is_pit = operator in _POINT_IN_TIME_OPERATORS
        if is_pit and window_size is not None:
            raise ValueError(
                f"TemporalOperator.{operator.name} is a point-in-time operator; "
                f"window_size must be None, got {window_size!r}"
            )
        if not is_pit and window_size is None:
            raise ValueError(
                f"TemporalOperator.{operator.name} is a windowed operator; window_size is required"
            )
        if window_size is not None and (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size < 1
        ):
            raise ValueError(f"window_size must be a positive integer, got {window_size!r}")

        self.source = source
        self.operator = operator
        self.direction = direction
        self.window_size = window_size

    @property
    def column_name(self) -> str:
        """Deterministic name for this Layer 3 feature column."""
        parts = [self.source.column_name, self.operator.value, self.direction.value]
        if self.window_size is not None:
            parts.append(str(self.window_size))
        return "__".join(parts)

    def __repr__(self) -> str:
        return (
            f"TemporalFeature("
            f"source={self.source.column_name!r}, "
            f"operator={self.operator.name!r}, "
            f"direction={self.direction.name!r}, "
            f"window_size={self.window_size!r})"
        )
