"""DistributionalSpaceBuilder — generates all DistributionalColumn objects from a dataset."""

from __future__ import annotations

import logging
from typing import cast

from featkit.contracts.measurement.defaults import get_default_contract
from featkit.dataset.base import AbstractDataset
from featkit.enums import CategoricalTreatment
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.distributional import DistributionalColumn

_log = logging.getLogger(__name__)


class DistributionalSpaceBuilder:
    """Generates the full set of DistributionalColumn objects for a dataset.

    For each DISTRIBUTIONAL- or BOTH-treatment categorical, each measurement
    field, each contract-valid aggregator, and each distributional metric
    declared on the categorical, one ``DistributionalColumn`` is produced.

    Args:
        dataset: The source facts-table schema.
        value_measurements: Restrict which measurement fields are used as the
            value source. ``None`` means use all measurement fields in the dataset.
            An empty list produces no columns. Every entry must be present in the
            dataset (compared by name, type, and contract); a ``ValueError`` is
            raised for unknown fields.
        verbose: When ``True``, emits ``DEBUG``-level log messages at key
            milestones: builder start/end, and for each generated column the
            ``(categorical, measurement, aggregator, metric)`` combination dict
            and the resulting column name.
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        value_measurements: list[MeasurementField] | None = None,
        verbose: bool = False,
    ) -> None:
        self.dataset = dataset
        self.value_measurements = value_measurements
        self.verbose = verbose

    def build(self) -> list[DistributionalColumn]:
        """Build and return all DistributionalColumn objects."""
        if self.verbose:
            _log.debug("DistributionalSpaceBuilder.build() started")

        all_cats = [cast(CategoricalField, f) for f in self.dataset.categorical_fields]
        dist_cats = [
            c
            for c in all_cats
            if c.treatment in {CategoricalTreatment.DISTRIBUTIONAL, CategoricalTreatment.BOTH}
        ]

        dataset_measurements = [cast(MeasurementField, f) for f in self.dataset.measurement_fields]

        if self.value_measurements is not None:
            measurements: list[MeasurementField] = []
            for requested in self.value_measurements:
                matched = next((m for m in dataset_measurements if m == requested), None)
                if matched is None:
                    raise ValueError(
                        f"MeasurementField {requested.name!r} "
                        f"(type {requested.measurement_type.name}) is not present in the dataset"
                    )
                measurements.append(matched)
        else:
            measurements = dataset_measurements

        results: list[DistributionalColumn] = []
        seen: set[str] = set()

        for cat in dist_cats:
            for mf in measurements:
                contract = mf.contract or get_default_contract(mf.measurement_type)
                aggs = sorted(contract.valid_layer2_aggregators, key=lambda a: a.value)
                for agg in aggs:
                    for metric in cat.distributional_metrics:
                        col = DistributionalColumn(mf, agg, cat, metric)
                        if self.verbose:
                            _log.debug(
                                "combo: cat=%r, measurement=%r, aggregator=%s, metric=%s",
                                cat.name,
                                mf.name,
                                agg.value,
                                metric.value,
                            )
                            _log.debug(
                                "combination: %s",
                                {
                                    "categorical": cat.name,
                                    "measurement": mf.name,
                                    "aggregator": agg.value,
                                    "metric": metric.value,
                                },
                            )
                            _log.debug("column_name: %r", col.column_name)
                        if col.column_name not in seen:
                            seen.add(col.column_name)
                            results.append(col)

        if self.verbose:
            _log.debug(
                "DistributionalSpaceBuilder.build() done — %d column(s) generated", len(results)
            )
        return results
