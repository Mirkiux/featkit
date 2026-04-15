"""DistributionalSpaceBuilder — generates all DistributionalColumn objects from a dataset."""

from __future__ import annotations

from featkit.contracts.measurement.defaults import get_default_contract
from featkit.dataset.base import AbstractDataset
from featkit.enums import CategoricalTreatment
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.distributional import DistributionalColumn


class DistributionalSpaceBuilder:
    """Generates the full set of DistributionalColumn objects for a dataset.

    For each DISTRIBUTIONAL- or BOTH-treatment categorical, each measurement
    field, each contract-valid aggregator, and each distributional metric
    declared on the categorical, one ``DistributionalColumn`` is produced.

    Args:
        dataset: The source facts-table schema.
        value_measurements: Restrict which measurement fields are used as the
            value source. If None, all measurement fields in the dataset are used.
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        value_measurements: list[MeasurementField] | None = None,
    ) -> None:
        self.dataset = dataset
        self.value_measurements = value_measurements

    def build(self) -> list[DistributionalColumn]:
        """Build and return all DistributionalColumn objects."""
        dist_cats = [
            f
            for f in self.dataset.categorical_fields
            if isinstance(f, CategoricalField)
            and f.treatment in {CategoricalTreatment.DISTRIBUTIONAL, CategoricalTreatment.BOTH}
        ]

        measurements: list[MeasurementField] = (
            self.value_measurements
            if self.value_measurements is not None
            else [f for f in self.dataset.measurement_fields if isinstance(f, MeasurementField)]
        )

        results: list[DistributionalColumn] = []
        seen: set[str] = set()

        for cat in dist_cats:
            for mf in measurements:
                contract = mf.contract or get_default_contract(mf.measurement_type)
                aggs = sorted(contract.valid_layer2_aggregators, key=lambda a: a.value)
                for agg in aggs:
                    for metric in cat.distributional_metrics:
                        col = DistributionalColumn(mf, agg, cat, metric)
                        if col.column_name not in seen:
                            seen.add(col.column_name)
                            results.append(col)

        return results
