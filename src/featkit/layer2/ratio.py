"""RatioPivotedColumn — a Layer 2 column that is the ratio of two PivotedColumns."""

from __future__ import annotations

from featkit.enums import Layer2OutputType
from featkit.layer2.base import AbstractL2Column
from featkit.layer2.pivoted import PivotedColumn


class RatioPivotedColumn(AbstractL2Column):
    """A Layer 2 column representing the ratio of a fully-specified pivot cell
    over one of its marginal projections.

    The ratio is computed per entity-period in the Layer 2A table as::

        numerator_agg_expr / NULLIF(denominator_agg_expr, 0)

    Temporal operators are then applied to the pre-computed per-period ratio
    exactly as they are for any other numeric Layer 2 column.

    Args:
        numerator: A :class:`~featkit.layer2.pivoted.PivotedColumn` where every
            categorical value is non-``None`` (fully specified combination).
        denominator: A :class:`~featkit.layer2.pivoted.PivotedColumn` that is a
            marginal projection of *numerator* — same aggregator, same
            measurement, same categorical fields, at least one field set to
            ``None``, and all non-``None`` denominator values equal the
            corresponding numerator values.

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
        if numerator.source_measurement != denominator.source_measurement:
            raise ValueError(
                f"numerator and denominator must share the same source_measurement; "
                f"got {numerator.source_measurement.name!r} vs "
                f"{denominator.source_measurement.name!r}"
            )
        if numerator.categorical_combination.keys() != denominator.categorical_combination.keys():
            raise ValueError("numerator and denominator must have the same categorical fields")
        if any(v is None for v in numerator.categorical_combination.values()):
            raise ValueError(
                "numerator must have all categorical values non-None (fully-specified combination)"
            )
        if not any(v is None for v in denominator.categorical_combination.values()):
            raise ValueError(
                "denominator must have at least one categorical value set to None "
                "(marginal projection)"
            )
        for field, dval in denominator.categorical_combination.items():
            if dval is not None and dval != numerator.categorical_combination[field]:
                raise ValueError(
                    f"denominator value for field {field.name!r} is {dval!r} but "
                    f"numerator has {numerator.categorical_combination[field]!r}; "
                    "denominator must be a marginal projection of the numerator"
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
