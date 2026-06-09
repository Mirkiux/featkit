"""RatioSpaceBuilder — generates RatioPivotedColumn objects from pivot columns."""

from __future__ import annotations

import logging

from featkit.layer2.pivoted import PivotedColumn
from featkit.layer2.ratio import RatioPivotedColumn

_log = logging.getLogger(__name__)


class RatioSpaceBuilder:
    """Generates all valid ratio columns from a list of pivot columns.

    For each pivot column that has at least one non-``None`` categorical value
    (potential numerator), the builder finds every other column in the list
    that is a proper marginal projection of it and creates a
    :class:`~featkit.layer2.ratio.RatioPivotedColumn` for each valid
    (numerator, denominator) pair.

    A proper marginal projection (denominator) satisfies:

    * same aggregator and source measurement instance,
    * same set of categorical fields,
    * every non-``None`` denominator value equals the numerator's value for
      that field, and
    * at least one field that is ``None`` in the denominator but non-``None``
      in the numerator (the denominator sums over that dimension).

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

        # Potential numerators: any column with at least one non-None categorical value
        numerators = [
            c
            for c in self.pivot_columns
            if any(v is not None for v in c.categorical_combination.values())
        ]
        # Potential denominators: any column with at least one None categorical value
        denominators = [
            c
            for c in self.pivot_columns
            if any(v is None for v in c.categorical_combination.values())
        ]

        results: list[RatioPivotedColumn] = []
        seen: set[str] = set()

        for num in numerators:
            num_fields = set(num.categorical_combination.keys())
            for denom in denominators:
                if (
                    denom.layer2_aggregator != num.layer2_aggregator
                    or denom.source_measurement is not num.source_measurement
                    or set(denom.categorical_combination.keys()) != num_fields
                ):
                    continue
                # Denom must not contradict num, and must marginalize at least one
                # field that num has a non-None value for.
                valid = True
                is_proper = False
                for f, dv in denom.categorical_combination.items():
                    nv = num.categorical_combination[f]
                    if dv is not None and dv != nv:
                        valid = False
                        break
                    if dv is None and nv is not None:
                        is_proper = True
                if not valid or not is_proper:
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
