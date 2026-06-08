"""RatioSpaceBuilder — generates RatioPivotedColumn objects from pivot columns."""

from __future__ import annotations

import logging

from featkit.layer2.pivoted import PivotedColumn
from featkit.layer2.ratio import RatioPivotedColumn

_log = logging.getLogger(__name__)


class RatioSpaceBuilder:
    """Generates all valid ratio columns from a list of pivot columns.

    For each fully-specified pivot column (no marginal fields), the builder
    finds every marginal projection of that combination in the same list and
    creates a :class:`~featkit.layer2.ratio.RatioPivotedColumn` for each
    (numerator, denominator) pair.

    A marginal projection is a pivot column that:

    * shares the same aggregator and source measurement,
    * has the same set of categorical fields,
    * has at least one field set to ``None`` (∅ marginal), and
    * for every non-``None`` field, the value equals the numerator's value.

    Args:
        pivot_columns: The full set of Layer 2A pivot columns, typically
            ``FeatureStorePipeline.layer2a``.
        verbose: When ``True``, emits ``DEBUG``-level log messages listing
            each generated ratio column name.
    """

    def __init__(
        self,
        pivot_columns: list[PivotedColumn],
        verbose: bool = False,
    ) -> None:
        self.pivot_columns = pivot_columns
        self.verbose = verbose

    def build(self) -> list[RatioPivotedColumn]:
        """Build and return all RatioPivotedColumn objects."""
        if self.verbose:
            _log.debug("RatioSpaceBuilder.build() started")

        full = [
            c
            for c in self.pivot_columns
            if c.categorical_combination
            and all(v is not None for v in c.categorical_combination.values())
        ]
        marginals = [
            c
            for c in self.pivot_columns
            if any(v is None for v in c.categorical_combination.values())
        ]

        results: list[RatioPivotedColumn] = []
        seen: set[str] = set()

        for num in full:
            num_fields = set(num.categorical_combination.keys())
            for denom in marginals:
                if (
                    denom.layer2_aggregator != num.layer2_aggregator
                    or denom.source_measurement is not num.source_measurement
                    or set(denom.categorical_combination.keys()) != num_fields
                ):
                    continue
                if not all(
                    dv is None or dv == num.categorical_combination[df]
                    for df, dv in denom.categorical_combination.items()
                ):
                    continue
                col = RatioPivotedColumn(num, denom)
                if col.column_name not in seen:
                    seen.add(col.column_name)
                    if self.verbose:
                        _log.debug("column_name: %r", col.column_name)
                    results.append(col)

        if self.verbose:
            _log.debug(
                "RatioSpaceBuilder.build() done — %d ratio column(s) generated", len(results)
            )
        return results
