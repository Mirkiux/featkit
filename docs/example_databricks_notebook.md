# Example — Dynamic domain resolution in a Databricks notebook

This example shows how to let featkit resolve the `allowed_values` domain of a
`CategoricalField` at runtime by querying the facts table directly from a
Databricks notebook.

`DatabricksNotebookAdapter` discovers the pre-injected `spark` session
automatically — no constructor arguments are needed.

## Notebook cells

### Cell 1 — imports

```python
from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import CategoricalTreatment, MeasurementType, TimeGranularity
from featkit.execution.adapters import DatabricksNotebookAdapter
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator
from featkit.pipeline import FeatureStorePipeline
```

### Cell 2 — define the dataset (no `allowed_values` on the categorical)

```python
ds = SimpleDataset(
    "mydb.myschema.silver_transactions",
    [
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        MeasurementField("txn_count", MeasurementType.CANTIDAD),
        # No allowed_values — the adapter will resolve the domain at build() time
        CategoricalField("segment", CategoricalTreatment.PIVOT),
        CategoricalField("product_type", CategoricalTreatment.PIVOT),
    ],
)
```

### Cell 3 — configure with the notebook adapter

```python
adapter = DatabricksNotebookAdapter()

cfg = FeatureStoreConfig(
    dataset=ds,
    output_schema="analytics",
    output_table_prefix="feat_",
    time_windows=[3, 6, 12],
    include_marginals=True,
    adapter=adapter,           # triggers SELECT DISTINCT resolution at build()
)
```

### Cell 4 — build and generate

```python
# build() fires one SELECT DISTINCT per unresolved CategoricalField
pipeline = FeatureStorePipeline(config=cfg).build()

print(f"Layer 2A columns : {len(pipeline.layer2a)}")
print(f"Layer 3  features: {len(pipeline.layer3)}")

result = DatabricksSQLCodeGenerator().generate(pipeline)
print(result.code.sql[:500])
```

### Cell 5 — save the artefacts to DBFS

```python
result.save("/dbfs/mnt/output/features/")
# Writes:
#   /dbfs/mnt/output/features/script.sql
#   /dbfs/mnt/output/features/dag.json
#   /dbfs/mnt/output/features/diagram.md
```

## How it works

When `FeatureStorePipeline.build()` is called with an `adapter` set on the
config, it constructs an `AdapterDomainResolver` and passes it to
`PivotSpaceBuilder` as the `domain_resolver` callable.  For each
`CategoricalField` that has no `allowed_values`, the builder calls the resolver,
which executes:

```sql
SELECT DISTINCT segment
FROM mydb.myschema.silver_transactions
WHERE segment IS NOT NULL
ORDER BY 1
```

The returned values become the column domain exactly as if they had been listed
in `allowed_values` at configuration time.

## Mixing static and dynamic domains

Static and dynamic fields can coexist in the same dataset.  Fields that have
`allowed_values` set are used as-is; only fields without it trigger a query:

```python
ds = SimpleDataset(
    "mydb.myschema.silver_transactions",
    [
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        # Static domain — no query fired
        CategoricalField(
            "channel",
            CategoricalTreatment.PIVOT,
            allowed_values=["branch", "online", "mobile"],
        ),
        # Dynamic domain — one SELECT DISTINCT executed at build()
        CategoricalField("segment", CategoricalTreatment.PIVOT),
    ],
)
```

## Using a different adapter

Swap `DatabricksNotebookAdapter` for any other adapter without changing the
rest of the code:

```python
from featkit.execution.adapters import DatabricksAdapter

adapter = DatabricksAdapter(
    host="<workspace>.azuredatabricks.net",
    token="<pat>",
    http_path="/sql/1.0/warehouses/<warehouse-id>",
    catalog="mydb",
    schema="myschema",
)

cfg = FeatureStoreConfig(..., adapter=adapter)
```
