"""PivotSpaceBuilder — generates all PivotedColumn objects from a dataset."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import cast

from featkit.contracts.measurement.defaults import get_default_contract
from featkit.dataset.base import AbstractDataset
from featkit.enums import CategoricalTreatment, Layer2Aggregator, MeasurementType
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.pivoted import PivotedColumn


class PivotSpaceBuilder:
    """Generates the full set of PivotedColumn objects for a dataset.

    Args:
        dataset: The source facts-table schema.
        include_marginals: When True, each categorical domain is augmented with
            None (the ∅ marginal), producing one column per ∅-substituted
            combination in addition to the fully-specified ones.
        aggregators_override: Per-measurement-type override list. Only aggregators
            that are also contract-valid for the measurement type are used.
        domain_resolver: Callable invoked to resolve the domain of a categorical
            whose ``allowed_values`` is None. Must return a list of string values.
            Raises ``ValueError`` at build time if not provided for such a field.
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        include_marginals: bool = True,
        aggregators_override: dict[MeasurementType, list[Layer2Aggregator]] | None = None,
        domain_resolver: Callable[[CategoricalField], list[str]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.include_marginals = include_marginals
        self.aggregators_override = aggregators_override
        self.domain_resolver = domain_resolver

    def build(self) -> list[PivotedColumn]:
        """Build and return all PivotedColumn objects."""
        all_cats = [cast(CategoricalField, f) for f in self.dataset.categorical_fields]
        pivot_cats = [
            c
            for c in all_cats
            if c.treatment in {CategoricalTreatment.PIVOT, CategoricalTreatment.BOTH}
        ]

        cat_domains: dict[CategoricalField, list[str | None]] = {}
        for cat in pivot_cats:
            if cat.allowed_values is not None:
                raw: list[str] = list(cat.allowed_values)
            elif self.domain_resolver is not None:
                raw = list(self.domain_resolver(cat))
            else:
                raise ValueError(
                    f"CategoricalField {cat.name!r} has no allowed_values and no "
                    f"domain_resolver was provided"
                )
            if any(v is None for v in raw):  # type: ignore[misc]
                raise ValueError(
                    f"CategoricalField {cat.name!r}: resolved domain contains None; "
                    f"None is reserved as the \u2205 marginal sentinel"
                )
            domain: list[str | None] = list(raw)
            if self.include_marginals:
                domain = domain + [None]
            cat_domains[cat] = domain

        measurements = [cast(MeasurementField, f) for f in self.dataset.measurement_fields]

        cats = list(cat_domains.keys())
        combos = product(*(cat_domains[c] for c in cats)) if cats else ((),)

        results: list[PivotedColumn] = []
        seen: dict[str, PivotedColumn] = {}

        for combo in combos:
            cat_combination = {cats[i]: combo[i] for i in range(len(cats))} if cats else {}
            for mf in measurements:
                for agg in self._valid_aggregators(mf):
                    col = PivotedColumn(mf, agg, cat_combination)
                    if col.column_name in seen:
                        raise ValueError(
                            f"Duplicate pivot column name generated: {col.column_name!r}. "
                            f"Conflicting columns: {seen[col.column_name]!r} and {col!r}"
                        )
                    seen[col.column_name] = col
                    results.append(col)

        return results

    def _valid_aggregators(self, mf: MeasurementField) -> list[Layer2Aggregator]:
        contract = mf.contract or get_default_contract(mf.measurement_type)
        valid = contract.valid_layer2_aggregators
        if self.aggregators_override and mf.measurement_type in self.aggregators_override:
            return [a for a in self.aggregators_override[mf.measurement_type] if a in valid]
        return sorted(valid, key=lambda a: a.value)
