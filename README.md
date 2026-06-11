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

## Feature naming anatomy

Every feature produced by featkit has a deterministic, human-readable name built from fixed segments separated by `__` (double underscore). Understanding the segments lets you decode any feature name without looking at the code.

There are four families of features, each with its own naming pattern.

---

### Layer 2A — Pivot features

**Pattern:** `{AGG}__{MEASUREMENT}[__{FIELD}_{VALUE}…]`

| Segment | Source | Example |
|---|---|---|
| `AGG` | `Layer2Aggregator` enum | `SUM`, `COUNT`, `AVG`, `MIN`, `MAX` |
| `MEASUREMENT` | `MeasurementField.name` | `MTO`, `TRX` |
| `FIELD_VALUE` | `CategoricalField.name` + `_` + value, one per non-marginal field, sorted alphabetically by field name | `CANAL_DIGITAL`, `SECTOR_RETAIL` |

The valid aggregators for each `MEASUREMENT` depend on its `MeasurementType`. Only contract-permitted aggregator–measurement combinations are generated.

| Measurement type | Semantic meaning | Valid `AGG` values |
|---|---|---|
| `MONTO` | Monetary amount | `SUM`, `MAX`, `MIN`, `AVG` |
| `CANTIDAD` | Count / quantity | `SUM` |
| `TICKET` | Average ticket size | `AVG` |
| `FLAG` | Binary indicator | `MAX` |
| `FECHA` | Date / timestamp | `MAX`, `MIN` |
| `BALANCE` | Point-in-time balance | `MAX`, `MIN`, `AVG` |
| `TIME_DIFF` | Duration / elapsed time | `SUM`, `AVG`, `MAX`, `MIN` |
| `ESTADISTICO` | Generic statistic | `SUM`, `AVG`, `MAX`, `MIN`, `COUNT` |

Categorical fields set to the **∅ marginal** (no filter on that dimension) are omitted from the name entirely, so the name implicitly aggregates over all values of that dimension.

```
SUM__MTO                                  # global — all sectors, all channels
SUM__MTO__CANAL_DIGITAL                   # CANAL=DIGITAL, marginal over SECTOR
SUM__MTO__SECTOR_RETAIL                   # SECTOR=RETAIL, marginal over CANAL
SUM__MTO__CANAL_DIGITAL__SECTOR_RETAIL    # CANAL=DIGITAL and SECTOR=RETAIL (alphabetical order)
SUM__TRX__CANAL_PRESENCIAL                # sum of TRX (CANTIDAD → only SUM is valid) for PRESENCIAL channel
```

---

### Layer 2B — Distributional features

**Pattern:** `{CATEGORICAL}__{MEASUREMENT}__{AGG}__{METRIC}`

| Segment | Source | Example |
|---|---|---|
| `CATEGORICAL` | `CategoricalField.name` | `CANAL`, `SECTOR` |
| `MEASUREMENT` | `MeasurementField.name` | `MTO` |
| `AGG` | `Layer2Aggregator` enum | `SUM` |
| `METRIC` | `DistributionalMetric` enum | `ENTROPY`, `HHI`, `DOMINANT_PROPORTION`, `MODE`, `COUNT` |

These columns capture the shape of the value distribution of a categorical field, weighted by the aggregated measurement.

| Metric | What it measures |
|---|---|
| `ENTROPY` | Shannon entropy of the category distribution — higher means more uniform spread |
| `HHI` | Herfindahl-Hirschman Index — concentration; higher means more dominated by one value |
| `DOMINANT_PROPORTION` | Share of the most common category value |
| `MODE` | The most frequent category value (output type: categorical) |
| `COUNT` | Number of distinct observed values |

```
CANAL__MTO__SUM__ENTROPY            # entropy of channel distribution by amount
SECTOR__TRX__SUM__HHI               # HHI of sector distribution by transaction count (CANTIDAD → only SUM)
CANAL__MTO__SUM__MODE               # dominant channel by amount (categorical output)
```

---

### Layer 2C — Ratio features

**Pattern:** `{NUMERATOR}__over__{DENOMINATOR}`

where `NUMERATOR` and `DENOMINATOR` are full Layer 2A pivot feature names. The denominator is always a **proper marginal projection** of the numerator: it has at least one categorical dimension set to ∅ that is non-∅ in the numerator, and no contradicting values.

The underlying value is `numerator / NULLIF(denominator, 0)` computed per entity per period.

```
# Numerator: DIGITAL channel + RETAIL sector
# Denominator: RETAIL sector only (CANAL marginalized → share of DIGITAL within RETAIL)
SUM__MTO__CANAL_DIGITAL__SECTOR_RETAIL__over__SUM__MTO__SECTOR_RETAIL

# Denominator: DIGITAL channel only (SECTOR marginalized → share of RETAIL within DIGITAL)
SUM__MTO__CANAL_DIGITAL__SECTOR_RETAIL__over__SUM__MTO__CANAL_DIGITAL

# Denominator: global total (both marginalized → share of DIGITAL/RETAIL in total portfolio)
SUM__MTO__CANAL_DIGITAL__SECTOR_RETAIL__over__SUM__MTO
```

---

### Layer 3 — Temporal features

**Pattern:** `{L2_NAME}__{OPERATOR}__{DIRECTION}[__{WINDOW}]`

`L2_NAME` is the full name of any Layer 2A, 2B, or 2C feature. The temporal segments are appended at the end.

| Segment | Source | Notes |
|---|---|---|
| `OPERATOR` | `TemporalOperator` enum | See table below |
| `DIRECTION` | `TimeWindowDirection` enum | `BACKWARD` or `FORWARD` |
| `WINDOW` | `window_size` (integer, number of periods) | Omitted for point-in-time operators |

#### Temporal operators

| Operator | Type | Description |
|---|---|---|
| `PROM_U` | Windowed | Arithmetic mean of the monthly values over the window — each period contributes equally regardless of its volume |
| `PROM_P` | Windowed | Volume-proportional weighted mean — each period's contribution is weighted by its share of the total aggregated value across the window; weights are derived automatically from the data, no user configuration required |
| `SUM_U` | Windowed | Unweighted sum of the monthly values over the window |
| `SUM_P` | Windowed | Volume-weighted sum over the window (analogous weighting to `PROM_P`) |
| `MIN_U` | Windowed | Minimum value observed in the window |
| `MAX_U` | Windowed | Maximum value observed in the window |
| `CREC` | Windowed | Growth rate across the window |
| `FREQ` | Windowed | Fraction of periods in the window where the value was non-null / non-zero |
| `XM` | Windowed | Cross-period indicator |
| `MEDIA_ABS` | Windowed (composed) | Mean absolute deviation over the window |
| `RATIO` | Windowed (composed) | Ratio of two sub-windows |
| `ULT_MES` | Point-in-time | Value at the most recent period (no window suffix) |
| `PREV_MES` | Point-in-time | Value at the immediately preceding period (no window suffix) |
| `REC` | Point-in-time | Recency — periods elapsed since last non-null / non-zero observation (no window suffix) |

#### Valid operators per Layer 2 output type

| Output type | Valid operators |
|---|---|
| `NUMERIC` | `PROM_U`, `PROM_P`, `SUM_U`, `SUM_P`, `MIN_U`, `MAX_U`, `CREC`, `FREQ`, `XM`, `ULT_MES`, `PREV_MES`, `MEDIA_ABS`, `RATIO` |
| `FLAG` | `ULT_MES`, `PREV_MES`, `FREQ`, `XM`, `REC` |
| `CATEGORICAL` | `ULT_MES`, `PREV_MES`, `REC` |
| `TEMPORAL` | `ULT_MES`, `PREV_MES`, `REC`, `MIN_U`, `MAX_U`, `CREC` |

#### Examples

```
# Average amount (DIGITAL + RETAIL) over the last 6 months
SUM__MTO__CANAL_DIGITAL__SECTOR_RETAIL__PROM_U__BACKWARD__6

# Total transaction sum for RETAIL sector in the last 3 months (CANTIDAD → only SUM valid)
SUM__TRX__SECTOR_RETAIL__SUM_U__BACKWARD__3

# Most recent value of the CANAL entropy (by amount)
CANAL__MTO__SUM__ENTROPY__ULT_MES__BACKWARD

# Share of DIGITAL/RETAIL in total portfolio, averaged over last 12 months
SUM__MTO__CANAL_DIGITAL__SECTOR_RETAIL__over__SUM__MTO__PROM_U__BACKWARD__12

# Recency of the dominant channel (MODE is categorical → only REC/ULT_MES/PREV_MES valid)
CANAL__MTO__SUM__MODE__REC__BACKWARD
```

---

### Quick-reference: full name structure

```
┌─ Layer 2A pivot ──────────────────────────────────────────────────┐
│  AGG  __  MEASUREMENT  [__  FIELD_VALUE  …]                       │
└───────────────────────────────────────────────────────────────────┘

┌─ Layer 2B distributional ─────────────────────────────────────────┐
│  CATEGORICAL  __  MEASUREMENT  __  AGG  __  METRIC                │
└───────────────────────────────────────────────────────────────────┘

┌─ Layer 2C ratio ──────────────────────────────────────────────────┐
│  {Layer 2A name}  __over__  {Layer 2A name}                       │
└───────────────────────────────────────────────────────────────────┘

┌─ Layer 3 temporal (windowed) ─────────────────────────────────────┐
│  {Layer 2A/2B/2C name}  __  OPERATOR  __  DIRECTION  __  WINDOW   │
└───────────────────────────────────────────────────────────────────┘

┌─ Layer 3 temporal (point-in-time) ────────────────────────────────┐
│  {Layer 2A/2B/2C name}  __  OPERATOR  __  DIRECTION               │
└───────────────────────────────────────────────────────────────────┘
```

## Architecture

See [docs/general_plan.md](docs/general_plan.md) for the full implementation plan.

## License

MIT
