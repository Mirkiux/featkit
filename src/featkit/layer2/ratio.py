"""RatioPivotedColumn — a Layer 2 column that is the ratio of two PivotedColumns."""

from __future__ import annotations

from featkit.enums import Layer2OutputType
from featkit.layer2.base import AbstractL2Column
from featkit.layer2.pivoted import PivotedColumn


class RatioPivotedColumn(AbstractL2Column):
    """A Layer 2 column representing the ratio of a pivot cell over one of its
    marginal projections.

    The ratio is computed per entity-period in the Layer 2A table as::

        numerator_agg_expr / NULLIF(denominator_agg_expr, 0)

    Temporal operators are then applied to the pre-computed per-period ratio
    exactly as they are for any other numeric Layer 2 column.

    The denominator must be a *proper* marginal projection of the numerator:
    every non-``None`` denominator value must match the corresponding numerator
    value, and at least one field that is non-``None`` in the numerator must be
    ``None`` in the denominator (i.e. the denominator sums over that dimension).
    The numerator itself may contain ``None`` fields — those dimensions are
    already marginalised in both columns and are left unchanged.

    Args:
        numerator: A :class:`~featkit.layer2.pivoted.PivotedColumn` with at
            least one non-``None`` categorical value.
        denominator: A :class:`~featkit.layer2.pivoted.PivotedColumn` that is a
            proper marginal projection of *numerator* — same aggregator, same
            measurement instance, same categorical fields, every non-``None``
            denominator value equal to the corresponding numerator value, and
            at least one field that is ``None`` in the denominator but
            non-``None`` in the numerator.

    Raises:
        ValueError: If the numerator/denominator pair violates any of the
            constraints above.
    """

    def __init__(self, numerator: PivotedColumn, denominator: PivotedColumn) -> None:
        if numerator.layer2_aggregator != denominator.layer2_aggregator:
            raise ValueError(
                f"numerator and denominator must share the same aggregator; "
                f"got {numerator.layer2_aggregator.name!r} vs "
                f"{denominator.layer2_aggregator.name!r}"
            )
        if numerator.source_measurement is not denominator.source_measurement:
            raise ValueError(
                f"numerator and denominator must share the same source_measurement; "
                f"got {numerator.source_measurement.name!r} vs "
                f"{denominator.source_measurement.name!r}"
            )
        if numerator.categorical_combination.keys() != denominator.categorical_combination.keys():
            raise ValueError("numerator and denominator must have the same categorical fields")
        # Denominator must be a proper projection: it cannot contradict the numerator,
        # and must marginalize at least one field that numerator has a non-None value for.
        has_proper_marginal = False
        for field, dval in denominator.categorical_combination.items():
            nval = numerator.categorical_combination[field]
            if dval is not None and dval != nval:
                raise ValueError(
                    f"denominator value for field {field.name!r} is {dval!r} but "
                    f"numerator has {nval!r}; "
                    "denominator must be a proper marginal projection of the numerator"
                )
            if dval is None and nval is not None:
                has_proper_marginal = True
        if not has_proper_marginal:
            raise ValueError(
                "denominator must marginalize at least one field that has a non-None value "
                "in the numerator (denominator must be a proper marginal projection)"
            )

        self._numerator = numerator
        self._denominator = denominator

    @property
    def numerator(self) -> PivotedColumn:
        return self._numerator

    @property
    def denominator(self) -> PivotedColumn:
        return self._denominator

    @property
    def output_type(self) -> Layer2OutputType:
        return Layer2OutputType.NUMERIC

    @property
    def column_name(self) -> str:
        return f"{self._numerator.column_name}__over__{self._denominator.column_name}"

    def __repr__(self) -> str:
        return (
            f"RatioPivotedColumn("
            f"numerator={self._numerator.column_name!r}, "
            f"denominator={self._denominator.column_name!r})"
        )
