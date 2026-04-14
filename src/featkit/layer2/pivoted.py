"""PivotedColumn — one cell in the Layer 2A pivot table."""

from __future__ import annotations

from featkit.contracts.measurement.defaults import get_default_contract
from featkit.enums import Layer2Aggregator, Layer2OutputType, MeasurementType
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.base import AbstractLayer2Column

_MT_TO_OUTPUT: dict[MeasurementType, Layer2OutputType] = {
    MeasurementType.FLAG: Layer2OutputType.FLAG,
    MeasurementType.FECHA: Layer2OutputType.TEMPORAL,
}


class PivotedColumn(AbstractLayer2Column):
    """A single cell in the Layer 2A (pivot) output table.

    Args:
        source_measurement: Measurement field being aggregated.
        layer2_aggregator: SQL aggregation function applied to the measurement.
        categorical_combination: Mapping of categorical field → value that
            defines the filter for this cell. ``None`` as a value means the
            ∅ marginal (no filter on that dimension).

    Raises:
        ValueError: If ``layer2_aggregator`` is not permitted by the
            measurement's contract.
    """

    def __init__(
        self,
        source_measurement: MeasurementField,
        layer2_aggregator: Layer2Aggregator,
        categorical_combination: dict[CategoricalField, str | None] | None = None,
    ) -> None:
        super().__init__(source_measurement, layer2_aggregator)
        self.categorical_combination: dict[CategoricalField, str | None] = (
            categorical_combination or {}
        )
        contract = source_measurement.contract or get_default_contract(
            source_measurement.measurement_type
        )
        if not contract.is_valid(layer2_aggregator):
            valid = ", ".join(
                a.name for a in sorted(contract.valid_layer2_aggregators, key=lambda a: a.value)
            )
            raise ValueError(
                f"Layer2Aggregator.{layer2_aggregator.name} is not valid for "
                f"MeasurementType.{source_measurement.measurement_type.name}. "
                f"Valid aggregators: {valid}"
            )

    @property
    def output_type(self) -> Layer2OutputType:
        return _MT_TO_OUTPUT.get(self.source_measurement.measurement_type, Layer2OutputType.NUMERIC)

    @property
    def column_name(self) -> str:
        parts = [self.layer2_aggregator.value, self.source_measurement.name]
        for field, value in sorted(self.categorical_combination.items(), key=lambda kv: kv[0].name):
            parts.append(f"{field.name}_{value}" if value is not None else field.name)
        return "__".join(parts)
