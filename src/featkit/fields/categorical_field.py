"""CategoricalField — a column used for pivot or distributional grouping."""

from __future__ import annotations

from featkit.enums import CategoricalTreatment, DistributionalMetric, FieldRole
from featkit.fields.base import AbstractField


class CategoricalField(AbstractField):
    """Descriptor for a categorical column.

    Can participate in Layer 2A (``PIVOT``), Layer 2B (``DISTRIBUTIONAL``), or both.

    Args:
        name: Column name in the source table.
        treatment: How this column is used in Layer 2 construction.
        distributional_metrics: Metrics to compute in Layer 2B. Required when
            *treatment* includes ``DISTRIBUTIONAL``.
        allowed_values: Static list of distinct values. When set, the
            :class:`~featkit.builders.pivot_space.PivotSpaceBuilder` uses these
            directly without executing a query. When ``None``, a ``domain_resolver``
            must be supplied to the builder.
    """

    def __init__(
        self,
        name: str,
        treatment: CategoricalTreatment,
        distributional_metrics: list[DistributionalMetric] | None = None,
        allowed_values: list[str] | None = None,
    ) -> None:
        super().__init__(name)
        self.treatment = treatment
        self.distributional_metrics: list[DistributionalMetric] = distributional_metrics or []
        self.allowed_values = allowed_values

        if (
            treatment in (CategoricalTreatment.DISTRIBUTIONAL, CategoricalTreatment.BOTH)
            and not self.distributional_metrics
        ):
            raise ValueError(
                f"CategoricalField {name!r}: distributional_metrics must not be empty "
                f"when treatment is {treatment.name}"
            )

    @property
    def role(self) -> FieldRole:
        return FieldRole.CATEGORICAL

    def _key(self) -> tuple[object, ...]:
        return (self.name, self.treatment)
