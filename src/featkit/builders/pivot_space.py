"""PivotSpaceBuilder — generates all PivotedColumn objects from a dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import combinations as _icombinations
from itertools import product
from typing import cast

from featkit.contracts.measurement.defaults import get_default_contract
from featkit.dataset.base import AbstractDataset
from featkit.enums import CategoricalTreatment, Layer2Aggregator, MeasurementType
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.pivoted import PivotedColumn

_log = logging.getLogger(__name__)


def _with_marginals(
    observed: list[dict[CategoricalField, str]],
    cats: list[CategoricalField],
) -> list[dict[CategoricalField, str | None]]:
    """Expand *observed* combinations with all ∅-substituted variants.

    For each observed combination and each subset of fields, a new
    combination is produced where those fields are replaced with ``None``
    (the ∅ marginal sentinel).  The all-None combination is always included
    even when *observed* is empty, since it represents an unconditional
    aggregate over all data.

    Duplicates are suppressed so overlapping projections of different
    observed combinations appear only once.
    """
    seen: set[tuple[tuple[str, str | None], ...]] = set()
    result: list[dict[CategoricalField, str | None]] = []

    def _append(combo: dict[CategoricalField, str | None]) -> None:
        key = tuple(sorted((f.name, combo[f]) for f in cats))
        if key not in seen:
            seen.add(key)
            result.append(combo)

    _append({f: None for f in cats})

    for combo in observed:
        for r in range(len(cats)):  # r == len(cats) (all-None) already added above
            for nulled in _icombinations(cats, r):
                c: dict[CategoricalField, str | None] = dict(combo)
                for f in nulled:
                    c[f] = None
                _append(c)

    return result


class PivotSpaceBuilder:
    """Generates the full set of PivotedColumn objects for a dataset.

    Two combination strategies are supported:

    * **Observed combinations** (preferred when an adapter is available):
      supply a ``combination_resolver`` callable.  It receives the list of
      pivot categorical fields and returns only the combinations that
      actually exist in the source table.  Marginals are then derived from
      those observed combinations rather than from the full Cartesian
      product.

    * **Cartesian product** (default, no adapter required): per-field
      domains are resolved from ``allowed_values`` or ``domain_resolver``
      and the full product is generated.

    Args:
        dataset: The source facts-table schema.
        include_marginals: When True, ∅-substituted combinations are added
            on top of the base combinations (observed or Cartesian).
        aggregators_override: Per-measurement-type override list. Only
            aggregators that are also contract-valid for the measurement
            type are used.
        combination_resolver: Callable that takes the list of pivot
            ``CategoricalField`` objects and returns the observed
            combinations as a list of ``{field: value}`` dicts.  When
            provided, ``domain_resolver`` is not used.
        domain_resolver: Callable invoked per-field to resolve the domain
            of a categorical whose ``allowed_values`` is None.  Used only
            in the Cartesian product path (i.e. when
            ``combination_resolver`` is not provided).  Raises
            ``ValueError`` at build time if a dynamic field is encountered
            and this is not provided.
        verbose: When ``True``, emits ``DEBUG``-level log messages at key
            milestones: builder start/end, each ``domain_resolver``
            invocation with its resolved values, each ``cat_combination``
            dict, and every generated column name.
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        include_marginals: bool = True,
        aggregators_override: dict[MeasurementType, list[Layer2Aggregator]] | None = None,
        combination_resolver: (
            Callable[[list[CategoricalField]], list[dict[CategoricalField, str]]] | None
        ) = None,
        domain_resolver: Callable[[CategoricalField], list[str]] | None = None,
        verbose: bool = False,
    ) -> None:
        self.dataset = dataset
        self.include_marginals = include_marginals
        self.aggregators_override = aggregators_override
        self.combination_resolver = combination_resolver
        self.domain_resolver = domain_resolver
        self.verbose = verbose

    def build(self) -> list[PivotedColumn]:
        """Build and return all PivotedColumn objects."""
        if self.verbose:
            _log.debug("PivotSpaceBuilder.build() started")

        all_cats = [cast(CategoricalField, f) for f in self.dataset.categorical_fields]
        pivot_cats = [
            c
            for c in all_cats
            if c.treatment in {CategoricalTreatment.PIVOT, CategoricalTreatment.BOTH}
        ]
        measurements = [cast(MeasurementField, f) for f in self.dataset.measurement_fields]

        all_combos: list[dict[CategoricalField, str | None]]

        if self.combination_resolver is not None and pivot_cats:
            observed = self.combination_resolver(pivot_cats)
            if self.include_marginals:
                all_combos = _with_marginals(observed, pivot_cats)
            else:
                all_combos = [dict(c) for c in observed]
        else:
            cat_domains: dict[CategoricalField, list[str | None]] = {}
            for cat in pivot_cats:
                if cat.allowed_values is not None:
                    raw: list[str] = list(cat.allowed_values)
                elif self.domain_resolver is not None:
                    if self.verbose:
                        _log.debug(
                            "domain_resolver: resolving domain for categorical %r", cat.name
                        )
                    raw = list(self.domain_resolver(cat))
                    if self.verbose:
                        _log.debug(
                            "domain_resolver: resolved %d value(s) for %r: %s",
                            len(raw),
                            cat.name,
                            raw,
                        )
                else:
                    raise ValueError(
                        f"CategoricalField {cat.name!r} has no allowed_values and no "
                        f"domain_resolver was provided"
                    )
                if any(v is None for v in raw):
                    raise ValueError(
                        f"CategoricalField {cat.name!r}: resolved domain contains None; "
                        f"None is reserved as the ∅ marginal sentinel"
                    )
                domain: list[str | None] = list(raw)
                if self.include_marginals:
                    domain = domain + [None]
                cat_domains[cat] = domain

            cats = list(cat_domains.keys())
            combos = product(*(cat_domains[c] for c in cats)) if cats else ((),)
            all_combos = [
                {cats[i]: combo[i] for i in range(len(cats))} if cats else {} for combo in combos
            ]

        results: list[PivotedColumn] = []
        seen: dict[str, PivotedColumn] = {}

        for cat_combination in all_combos:
            if self.verbose:
                _log.debug(
                    "cat_combination: %s",
                    {c.name: v for c, v in cat_combination.items()},
                )
            for mf in measurements:
                for agg in self._valid_aggregators(mf):
                    col = PivotedColumn(mf, agg, cat_combination)
                    if col.column_name in seen:
                        raise ValueError(
                            f"Duplicate pivot column name generated: {col.column_name!r}. "
                            f"Conflicting columns: {seen[col.column_name]!r} and {col!r}"
                        )
                    if self.verbose:
                        _log.debug("column_name: %r", col.column_name)
                    seen[col.column_name] = col
                    results.append(col)

        if self.verbose:
            _log.debug(
                "PivotSpaceBuilder.build() done — %d column(s) generated", len(results)
            )
        return results

    def _valid_aggregators(self, mf: MeasurementField) -> list[Layer2Aggregator]:
        contract = mf.contract or get_default_contract(mf.measurement_type)
        valid = contract.valid_layer2_aggregators
        if self.aggregators_override and mf.measurement_type in self.aggregators_override:
            return [a for a in self.aggregators_override[mf.measurement_type] if a in valid]
        return sorted(valid, key=lambda a: a.value)
