# Quickstart

This guide walks through the minimum steps to generate a feature store script from a facts table.

## 1. Define the dataset

Describe the source table schema using field descriptors:

```python
from featkit.dataset.base import SimpleDataset
from featkit.enums import CategoricalTreatment, DistributionalMetric, MeasurementType, TimeGranularity
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField

ds = SimpleDataset(
    source_reference="mydb.myschema.silver_transactions",
    fields=[
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        CategoricalField(
            "product_category",
            CategoricalTreatment.PIVOT,
            allowed_values=["retail", "corporate", "sme"],
        ),
    ],
)
```

## 2. Configure the pipeline

```python
from featkit.config import FeatureStoreConfig

cfg = FeatureStoreConfig(
    dataset=ds,
    output_schema="analytics",
    output_table_prefix="feat_",
    time_windows=[3, 6, 12],   # look-back windows in months
)
```

## 3. Build and run

```python
from featkit.generators.sql.snowflake import SnowflakeSQLCodeGenerator
from featkit.pipeline import FeatureStorePipeline

pipeline = FeatureStorePipeline(config=cfg).build()
result = SnowflakeSQLCodeGenerator().generate(pipeline)

# Inspect the generated SQL
print(result.code.sql)

# Save to disk — writes script.sql, dag.json, diagram.md
result.save("output/")
```

## What gets generated

For the example above `result.save("output/")` writes:

| File | Contents |
|------|----------|
| `script.sql` | Five `CREATE OR REPLACE TABLE` statements in dependency order |
| `dag.json` | Machine-readable build graph (7 nodes) |
| `diagram.md` | Mermaid flowchart of the build graph |

The five SQL steps are:

1. **`feat_mob_ref`** — period cross-reference table: all `(period_analysis, period_relative, mob)` pairs
2. **`feat_layer2a`** — pivot aggregations: one column per `(category value × measurement × aggregator)` combination
3. **`feat_layer2b`** — distributional metrics (empty when no `DISTRIBUTIONAL` fields exist)
4. **`feat_layer3`** — temporal features: rolling aggregates over the look-back windows
5. **`feat_features`** — final join of all intermediate tables

## Switch generator

Swap the generator to target a different compute engine without changing any other code:

```python
from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator
from featkit.generators.sql.spark_sql import SparkSQLCodeGenerator
from featkit.generators.pyspark.databricks import PySparkCodeGenerator

databricks_result = DatabricksSQLCodeGenerator().generate(pipeline)
spark_result      = SparkSQLCodeGenerator().generate(pipeline)
pyspark_result    = PySparkCodeGenerator().generate(pipeline)

# PySpark output is a Python script string, not SQL
print(pyspark_result.code.code)
```
