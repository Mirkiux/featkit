# featkit

**featkit** is a Python framework for automated feature store generation from relational facts tables.

It implements a three-layer architecture:

- **Layer 1** — input facts table with typed columns (ID, time, categorical, measurement)
- **Layer 2** — horizontal concept table built via pivot (2A) and distributional aggregations (2B)
- **Layer 3** — temporal feature table produced by sliding operators over the Layer 2 columns

The framework is engine-agnostic: the same pipeline definition produces either a standalone SQL script (Snowflake, Databricks SQL, Spark SQL) or a lazy PySpark execution plan, with the choice abstracted behind a code generator interface.

## Key concepts

| Layer | What it does |
|---|---|
| Layer 2A — Pivot | `GROUP BY (ID, time)` + `CASE WHEN` per categorical combination × measurement × aggregator |
| Layer 2B — Distributional | Per-categorical CTEs computing entropy, HHI, dominant proportion, mode, count |
| Layer 3 — Temporal | Sliding window operators (PROM_U, SUM_U, CREC, FREQ, REC, …) over all Layer 2 columns |

## Installation

```bash
pip install featkit
```

## Quickstart

```python
from featkit import FeatureStorePipeline, FeatureStoreConfig
from featkit.dataset import SimpleDataset
from featkit.fields import IDField, TimeField, CategoricalField, MeasurementField
from featkit.enums import MeasurementType, TimeGranularity, CategoricalTreatment
from featkit.generators.sql import SnowflakeSQLCodeGenerator

# Define schema
fields = [
    IDField(name="ID_CLIENTE"),
    TimeField(name="PERIODO",
              source_granularity=TimeGranularity.MONTHLY,
              target_granularity=TimeGranularity.MONTHLY),
    CategoricalField(name="SECTOR", treatment=CategoricalTreatment.PIVOT,
                     allowed_values=["RETAIL", "CORP", "PYME"]),
    CategoricalField(name="CANAL",  treatment=CategoricalTreatment.PIVOT,
                     allowed_values=["DIGITAL", "PRESENCIAL", "TELEFONO"]),
    MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO),
    MeasurementField(name="TRX", measurement_type=MeasurementType.CANTIDAD),
]

dataset = SimpleDataset(
    source_reference="MY_DB.MY_SCHEMA.FACTS_TABLE",
    fields=fields,
)

config = FeatureStoreConfig(
    dataset=dataset,
    output_schema="MY_DB.MY_SCHEMA",
    output_table_prefix="FS",
    time_windows=[3, 6, 9, 12],
)

pipeline = FeatureStorePipeline(config).build()
output = pipeline.run(SnowflakeSQLCodeGenerator())

output.save("./output")
# Writes: output/script.sql, output/dag.json, output/diagram.md
```

## Architecture

See [docs/general_plan.md](docs/general_plan.md) for the full implementation plan.

## License

MIT
