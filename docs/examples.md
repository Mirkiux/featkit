# Examples

## Example 1 — Pivot categoricals with marginals

Generate features for every category value plus the unconditional total (∅ marginal).

```python
from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import CategoricalTreatment, MeasurementType, TimeGranularity
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.generators.sql.snowflake import SnowflakeSQLCodeGenerator
from featkit.pipeline import FeatureStorePipeline

ds = SimpleDataset(
    "mydb.silver_sales",
    [
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        MeasurementField("txn_count", MeasurementType.CANTIDAD),
        CategoricalField(
            "channel",
            CategoricalTreatment.PIVOT,
            allowed_values=["branch", "online", "mobile"],
        ),
        CategoricalField(
            "product",
            CategoricalTreatment.PIVOT,
            allowed_values=["current_account", "savings", "loan"],
        ),
    ],
)

cfg = FeatureStoreConfig(
    dataset=ds,
    output_schema="analytics",
    output_table_prefix="feat_",
    time_windows=[3, 6, 12],
    include_marginals=True,   # include ∅ (unconditional) totals
)

pipeline = FeatureStorePipeline(config=cfg).build()
print(f"Layer 2A columns : {len(pipeline.layer2a)}")
print(f"Layer 3  features: {len(pipeline.layer3)}")

result = SnowflakeSQLCodeGenerator().generate(pipeline)
result.save("output/snowflake/")
```


## Example 2 — Distributional metrics

Measure concentration, diversity, and dominant value of a categorical over time.

```python
from featkit.enums import CategoricalTreatment, DistributionalMetric, MeasurementType, TimeGranularity
from featkit.fields.categorical_field import CategoricalField

# All five distributional metrics
region_field = CategoricalField(
    "region",
    CategoricalTreatment.DISTRIBUTIONAL,
    distributional_metrics=[
        DistributionalMetric.ENTROPY,            # diversity index
        DistributionalMetric.HHI,                # Herfindahl–Hirschman index
        DistributionalMetric.DOMINANT_PROPORTION, # share of the top category
        DistributionalMetric.MODE,               # most frequent category
        DistributionalMetric.COUNT,              # number of active categories
    ],
)
```

The generated `feat_layer2b` table will have one column per
`(categorical × measurement × aggregator × metric)` combination.
For `ENTROPY` and `HHI` the output type is `NUMERIC`; for `MODE` it is
`CATEGORICAL`, which restricts the set of valid temporal operators
(`ULT_MES`, `PREV_MES`, `REC` only).


## Example 3 — Mixed pivot and distributional on the same field

Use `BOTH` treatment to get pivot columns *and* distributional metrics from a
single categorical field.

```python
from featkit.enums import CategoricalTreatment, DistributionalMetric

product_field = CategoricalField(
    "product_type",
    CategoricalTreatment.BOTH,
    allowed_values=["A", "B", "C"],
    distributional_metrics=[DistributionalMetric.ENTROPY, DistributionalMetric.HHI],
)
```


## Example 4 — Multiple entity keys

The pipeline supports composite primary keys. List every ID field; all generated
`GROUP BY` and join clauses will include all of them.

```python
from featkit.fields.id_field import IDField

ds = SimpleDataset(
    "mydb.silver_transactions",
    [
        IDField("country_code"),
        IDField("client_id"),
        # ... other fields
    ],
)
```


## Example 5 — Targeting Databricks SQL

```python
from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator

result = DatabricksSQLCodeGenerator().generate(pipeline)

# Databricks uses backtick quoting; syntax is otherwise identical to Snowflake
print(result.code.sql[:200])
result.save("output/databricks/")
```


## Example 6 — Generating a PySpark script

```python
from featkit.generators.pyspark.databricks import PySparkCodeGenerator

result = PySparkCodeGenerator().generate(pipeline)

# result.code is a PySparkOutput; .code contains the full Python script
script = result.code.code
print(script[:500])

# Save — writes script.py instead of script.sql
result.save("output/pyspark/")
```

The generated script is a self-contained Python file. Execute it by submitting
it to a Databricks job or a `spark-submit` invocation:

```bash
databricks jobs submit --existing-cluster-id <id> --python-file output/pyspark/script.py
```


## Example 7 — Operators override

Restrict which temporal operators are applied to each Layer 2 output type.
Useful when you only need a subset of features.

```python
from featkit.enums import Layer2OutputType, TemporalOperator
from featkit.config import FeatureStoreConfig

cfg = FeatureStoreConfig(
    dataset=ds,
    output_schema="analytics",
    output_table_prefix="feat_",
    time_windows=[3, 6],
    operators_override={
        # Only rolling averages and latest-month snapshots for numeric columns
        Layer2OutputType.NUMERIC: [
            TemporalOperator.PROM_U,
            TemporalOperator.SUM_U,
            TemporalOperator.ULT_MES,
        ],
    },
)
```


## Example 8 — Inspecting the DAG

```python
import json

result = SnowflakeSQLCodeGenerator().generate(pipeline)

# Programmatic access
for node in result.dag.nodes:
    print(f"{node.step_name:35s} depends on: {node.depends_on}")

# JSON (suitable for CI artefacts or a lineage tool)
print(json.dumps(json.loads(result.dag.to_json()), indent=2))

# Mermaid diagram (paste into any Mermaid renderer)
print(result.mermaid)
```

Expected output:

```
facts_table                         depends on: []
mob_table                           depends on: ['facts_table']
layer2a_pivot                       depends on: ['facts_table']
layer2b_distributional_ctes         depends on: ['facts_table']
layer2_join                         depends on: ['layer2a_pivot', 'layer2b_distributional_ctes']
layer3_temporal                     depends on: ['layer2_join', 'mob_table']
final_output                        depends on: ['layer2_join', 'layer3_temporal']
```


## Example 9 — Dynamic categorical domain resolution

When the categorical domain is not known at configuration time, supply a
`domain_resolver` callable. featkit will call it at `build()` time; no
database adapter or executor is built into the framework.

```python
from featkit.builders.pivot_space import PivotSpaceBuilder
from featkit.config import FeatureStoreConfig

# Provide your own query executor
def resolve_domain(field):
    return conn.execute(
        f"SELECT DISTINCT {field.name} FROM mydb.silver_transactions"
    ).fetchall()

ds = SimpleDataset(
    "mydb.silver_transactions",
    [
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        CategoricalField("segment", CategoricalTreatment.PIVOT),  # no allowed_values
    ],
)

# Build the pivot space manually with the resolver, then pass to the config
pivot_cols = PivotSpaceBuilder(
    dataset=ds,
    include_marginals=True,
    domain_resolver=resolve_domain,
).build()
```
