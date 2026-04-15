"""PivotedColumn — one cell in the Layer 2A pivot table."""

from __future__ import annotations

from featkit.enums import Layer2Aggregator, Layer2OutputType, MeasurementType
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.measurement_field import MeasurementField
from featkit.layer2.base import AbstractLayer2Column, _check_name_part

_MT_TO_OUTPUT: dict[MeasurementType, Layer2OutputType] = {
    MeasurementType.MONTO: Layer2OutputType.NUMERIC,
    MeasurementType.CANTIDAD: Layer2OutputType.NUMERIC,
    MeasurementType.TICKET: Layer2OutputType.NUMERIC,
    MeasurementType.FLAG: Layer2OutputType.FLAG,
    MeasurementType.FECHA: Layer2OutputType.TEMPORAL,
    MeasurementType.BALANCE: Layer2OutputType.NUMERIC,
    MeasurementType.TIME_DIFF: Layer2OutputType.NUMERIC,
    MeasurementType.ESTADISTICO: Layer2OutputType.NUMERIC,
}


class PivotedColumn(AbstractLayer2Column):
    """A single cell in the Layer 2A (pivot) output table.

    Args:
        source_measurement: Measurement field being aggregated.
        layer2_aggregator: SQL aggregation function applied to the measurement.
        categorical_combination: Mapping of categorical field → value that
            defines the filter for this cell. ``None`` as a value means the
            ∅ marginal (no filter on that dimension). Defensively copied at
            construction time.

    Raises:
        ValueError: If ``layer2_aggregator`` is not permitted by the
            measurement's contract, or if any field name or categorical value
            contains the column name separator (``__``).
    """

    def __init__(
        self,
        source_measurement: MeasurementField,
        layer2_aggregator: Layer2Aggregator,
        categorical_combination: dict[CategoricalField, str | None] | None = None,
    ) -> None:
        super().__init__(source_measurement, layer2_aggregator)
        self.categorical_combination: dict[CategoricalField, str | None] = (
            dict(categorical_combination) if categorical_combination is not None else {}
        )
        for field, value in self.categorical_combination.items():
            _check_name_part(field.name, "categorical field name")
            if value is not None:
                _check_name_part(value, "categorical value")

    @property
    def output_type(self) -> Layer2OutputType:
        return _MT_TO_OUTPUT[self.source_measurement.measurement_type]

    @property
    def column_name(self) -> str:
        parts = [self.layer2_aggregator.value, self.source_measurement.name]
        for field, value in sorted(self.categorical_combination.items(), key=lambda kv: kv[0].name):
            parts.append(f"{field.name}_{value}" if value is not None else field.name)
        return "__".join(parts)
