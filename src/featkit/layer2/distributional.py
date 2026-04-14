"""DistributionalColumn — one Layer 2B distributional metric column."""

from __future__ import annotations

from featkit.enums import DistributionalMetric, Layer2Aggregator, Layer2OutputType
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.base import AbstractLayer2Column


class DistributionalColumn(AbstractLayer2Column):
    """A single distributional metric column in the Layer 2B output table.

    Args:
        source_measurement: Measurement field providing the numerical context.
        layer2_aggregator: SQL aggregation function applied to the measurement.
        categorical: Categorical field whose value distribution is measured.
        distributional_metric: The distributional statistic to compute.
    """

    def __init__(
        self,
        source_measurement: MeasurementField,
        layer2_aggregator: Layer2Aggregator,
        categorical: CategoricalField,
        distributional_metric: DistributionalMetric,
    ) -> None:
        super().__init__(source_measurement, layer2_aggregator)
        self.categorical = categorical
        self.distributional_metric = distributional_metric

    @property
    def output_type(self) -> Layer2OutputType:
        if self.distributional_metric == DistributionalMetric.MODE:
            return Layer2OutputType.CATEGORICAL
        return Layer2OutputType.NUMERIC

    @property
    def column_name(self) -> str:
        return (
            f"{self.categorical.name}"
            f"__{self.source_measurement.name}"
            f"__{self.layer2_aggregator.value}"
            f"__{self.distributional_metric.value}"
        )
