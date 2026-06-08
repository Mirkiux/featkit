# Example — Observed-combinations pivot in a Databricks notebook

This example shows how featkit resolves pivot combinations at runtime by
querying the facts table directly from a Databricks notebook.

When an adapter is configured, `FeatureStorePipeline` constructs an
`AdapterCombinationResolver` and passes it to `PivotSpaceBuilder`.  Instead of
generating the full Cartesian product of per-field domains, the builder issues a
**single `SELECT DISTINCT`** query for all pivot categoricals and builds only the
combinations that actually exist in the data.  Marginals are then derived from
those observed combinations via subset-projection.

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

### Cell 2 — define the dataset

```python
ds = SimpleDataset(
    "mydb.myschema.silver_transactions",
    [
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        MeasurementField("txn_count", MeasurementType.CANTIDAD),
        # allowed_values used as WHERE IN-filter; omit to query with no filter
        CategoricalField(
            "segment",
            CategoricalTreatment.PIVOT,
            allowed_values=["retail", "sme", "corporate"],
        ),
        CategoricalField(
            "product_type",
            CategoricalTreatment.PIVOT,
            allowed_values=["loan", "deposit", "card"],
        ),
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
    adapter=adapter,   # triggers SELECT DISTINCT combination query at build()
)
```

### Cell 4 — build and generate

```python
# build() issues ONE SELECT DISTINCT for all pivot categoricals:
#
#   SELECT DISTINCT product_type, segment
#   FROM mydb.myschema.silver_transactions
#   WHERE product_type IS NOT NULL
#     AND segment IS NOT NULL
#     AND product_type IN ('loan', 'deposit', 'card')
#     AND segment IN ('retail', 'sme', 'corporate')
#   ORDER BY 1, 2
#
# Only the returned combinations (plus their marginal projections) become
# pivot columns — unobserved cross-combinations are never generated.
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

`FeatureStorePipeline.build()` constructs an `AdapterCombinationResolver` and
passes it to `PivotSpaceBuilder` as the `combination_resolver` callable.  The
resolver executes a single multi-column `SELECT DISTINCT`:

```sql
SELECT DISTINCT product_type, segment
FROM mydb.myschema.silver_transactions
WHERE product_type IS NOT NULL
  AND segment IS NOT NULL
  AND product_type IN ('loan', 'deposit', 'card')
  AND segment IN ('retail', 'sme', 'corporate')
ORDER BY 1, 2
```

Suppose the query returns three rows:

| product_type | segment   |
|-------------|-----------|
| loan        | retail    |
| loan        | sme       |
| deposit     | corporate |

With `include_marginals=True`, the builder derives every subset-projection of
those rows:

| product_type | segment   | interpretation                          |
|-------------|-----------|------------------------------------------|
| loan        | retail    | observed combination                    |
| loan        | sme       | observed combination                    |
| deposit     | corporate | observed combination                    |
| loan        | `∅`       | all segments for loan                   |
| deposit     | `∅`       | all segments for deposit                |
| `∅`         | retail    | all products for retail                 |
| `∅`         | sme       | all products for sme                    |
| `∅`         | corporate | all products for corporate              |
| `∅`         | `∅`       | unconditional aggregate (always present)|

Unobserved combinations (e.g. `deposit × retail`) are **never generated**,
keeping the feature space lean.

## Fields without `allowed_values`

If a field has no `allowed_values`, it is still included in the `SELECT DISTINCT`
but its column is not filtered in the WHERE clause — all distinct values present
in the table are returned for that dimension:

```python
ds = SimpleDataset(
    "mydb.myschema.silver_transactions",
    [
        IDField("client_id"),
        TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
        MeasurementField("amount", MeasurementType.MONTO),
        # Static domain — used as IN-filter in the combined query
        CategoricalField(
            "channel",
            CategoricalTreatment.PIVOT,
            allowed_values=["branch", "online", "mobile"],
        ),
        # No allowed_values — column included without an IN-filter
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

## Using `AdapterCombinationResolver` directly

The resolver can also be wired manually to `PivotSpaceBuilder` without going
through the pipeline:

```python
from featkit.execution.domain_resolver import AdapterCombinationResolver
from featkit.builders.pivot_space import PivotSpaceBuilder

resolver = AdapterCombinationResolver(adapter, "mydb.myschema.silver_transactions")

columns = PivotSpaceBuilder(
    dataset=ds,
    include_marginals=True,
    combination_resolver=resolver,
).build()
```
