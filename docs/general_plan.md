# featkit — General Implementation Plan

This document sequences the implementation of the featkit framework into discrete, ordered plans. Each plan has a clear scope, inputs, outputs, and acceptance criteria. Plans must be executed in order unless explicitly noted as independent.

---

## Critical path

```
(01 ∥ 02) → (03 ∥ 04 ∥ 05) → 06 → (07 ∥ 08) → (09 ∥ 10 ∥ 11) → 12 → 13 → (14 ∥ 16) → 15 → 17
```

Plans connected by `∥` can be implemented in parallel. Arrows indicate that all predecessors must be complete before proceeding.

---

## Plan 01 — Enumerators

**Module:** `src/featkit/enums.py`

**Scope:** Define all enumerators that form the type system of the framework. No logic, no dependencies.

**Enums to implement:**

| Enum | Values |
|---|---|
| `FieldRole` | `ID, TIME, CATEGORICAL, MEASUREMENT` |
| `MeasurementType` | `MONTO, CANTIDAD, TICKET, FLAG, FECHA, BALANCE, TIME_DIFF, ESTADISTICO` |
| `TimeGranularity` | `DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY` |
| `CategoricalTreatment` | `PIVOT, DISTRIBUTIONAL, BOTH` |
| `Layer2Aggregator` | `SUM, COUNT, MAX, MIN, AVG` |
| `DistributionalMetric` | `ENTROPY, HHI, DOMINANT_PROPORTION, MODE, COUNT` |
| `Layer2OutputType` | `NUMERIC, FLAG, CATEGORICAL, TEMPORAL` |
| `TemporalOperator` | `PROM_U, PROM_P, SUM_U, SUM_P, ULT_MES, PREV_MES, CREC, FREQ, MIN_U, MAX_U, REC, XM, MEDIA_ABS, RATIO` |
| `TimeWindowDirection` | `BACKWARD, FORWARD` |

**Acceptance criteria:**
- All enums importable from `featkit.enums`
- Each value has a string representation matching its name
- Full test coverage in `tests/test_fields.py` (import + value checks)

---

## Plan 02 — AbstractField + IDField + TimeField

**Module:** `src/featkit/fields/base.py`, `id_field.py`, `time_field.py`

**Scope:** Define the abstract base for all column descriptors and the two structurally simplest concrete fields.

**Classes:**

- `AbstractField` (ABC): `name: str`, abstract property `role: FieldRole`
- `IDField(AbstractField)`: `role = FieldRole.ID`
- `TimeField(AbstractField)`: `source_granularity`, `target_granularity`, derived property `truncation_required: bool`

**Acceptance criteria:**
- `AbstractField` cannot be instantiated directly
- `IDField` and `TimeField` are concrete and instantiable
- `truncation_required` is `True` iff `source_granularity != target_granularity`
- Tests in `tests/test_fields.py`

---

## Plan 03 — CategoricalField + MeasurementField

**Module:** `src/featkit/fields/categorical_field.py`, `measurement_field.py`

**Depends on:** Plan 01, Plan 02

**Scope:** The two fields that carry domain semantics: categorical fields drive the pivot/distributional split; measurement fields carry the type contract reference.

**Classes:**

- `CategoricalField(AbstractField)`: `treatment: CategoricalTreatment`, `distributional_metrics: list[DistributionalMetric]` (required when treatment includes `DISTRIBUTIONAL`), `allowed_values: list[str] | None = None` (explicit domain override; if `None`, domain is resolved at build time via the `PivotSpaceBuilder`'s `domain_resolver`)
- `MeasurementField(AbstractField)`: `measurement_type: MeasurementType`, `contract: AbstractMeasurementTypeContract` (forward reference — filled in Plan 04)

**Acceptance criteria:**
- `CategoricalField` raises `ValueError` if `treatment` includes `DISTRIBUTIONAL` and `distributional_metrics` is empty
- `MeasurementField` stores the contract reference without executing it
- Tests in `tests/test_fields.py`

---

## Plan 04 — AbstractMeasurementTypeContract + concrete implementations (Layer 1 → Layer 2 boundary)

**Module:** `src/featkit/contracts/measurement/base.py`, `defaults.py`

**Depends on:** Plan 01

**Scope:** The contract that governs which `Layer2Aggregator`s are semantically valid for each `MeasurementType`. This is the Layer 1 → Layer 2 type boundary.

**Classes:**

- `AbstractMeasurementTypeContract` (ABC): `measurement_type: MeasurementType`, abstract property `valid_layer2_aggregators: frozenset[Layer2Aggregator]`
- Concrete contracts — one per `MeasurementType`:

| Contract | Valid aggregators |
|---|---|
| `MontoContract` | `SUM, MAX, MIN, AVG` |
| `CantidadContract` | `SUM, COUNT` |
| `TicketContract` | `AVG` |
| `FlagContract` | `MAX` |
| `FechaContract` | `MAX, MIN` |
| `BalanceContract` | `MAX, MIN, AVG` |
| `TimeDiffContract` | `SUM, AVG, MAX, MIN` |
| `EstadisticoContract` | `SUM, AVG, MAX, MIN, COUNT` |

**Note:** `CANTIDAD` represents a base count present in each source row (e.g., a `TRX_COUNT` column in the facts table). Event frequency across the observation window (how many periods contain non-zero activity, etc.) is a Layer 3 concept derived via temporal operators — it does not require a dedicated `MeasurementType`.

**Acceptance criteria:**
- Each contract returns a frozen, non-empty set of aggregators
- A convenience factory `get_default_contract(mt: MeasurementType) -> AbstractMeasurementTypeContract` is provided
- Tests in `tests/test_contracts.py`

---

## Plan 05 — AbstractLayer2OutputContract + concrete implementations (Layer 2 → Layer 3 boundary)

**Module:** `src/featkit/contracts/output/base.py`, `defaults.py`

**Depends on:** Plan 01

**Scope:** The contract that governs which `TemporalOperator`s are semantically valid for each `Layer2OutputType`. This is the Layer 2 → Layer 3 type boundary.

**Classes:**

- `AbstractLayer2OutputContract` (ABC): `output_type: Layer2OutputType`, abstract property `valid_temporal_operators: frozenset[TemporalOperator]`
- Concrete contracts:

| Contract | Valid temporal operators |
|---|---|
| `NumericOutputContract` | `PROM_U, PROM_P, SUM_U, SUM_P, CREC, MIN_U, MAX_U, ULT_MES, PREV_MES, FREQ, XM, MEDIA_ABS, RATIO` |
| `FlagOutputContract` | `ULT_MES, PREV_MES, FREQ, XM, REC` |
| `CategoricalOutputContract` | `ULT_MES, PREV_MES, REC` |
| `TemporalOutputContract` | `ULT_MES, PREV_MES, REC, MIN_U, MAX_U, CREC` |

**Acceptance criteria:**
- Each contract returns a frozen, non-empty set of operators
- A convenience factory `get_default_output_contract(ot: Layer2OutputType) -> AbstractLayer2OutputContract` is provided
- Tests in `tests/test_contracts.py`

---

## Plan 06 — AbstractLayer2Column + PivotedColumn + DistributionalColumn

**Module:** `src/featkit/layer2/base.py`, `pivoted.py`, `distributional.py`

**Depends on:** Plan 01, Plan 03, Plan 04, Plan 05

**Scope:** The data model for all columns in the horizontal concept table (Layer 2 output). Both subclasses carry an `output_type` that bridges to Layer 3.

**Classes:**

- `AbstractLayer2Column` (ABC):
  - `source_measurement: MeasurementField`
  - `layer2_aggregator: Layer2Aggregator`
  - abstract property `output_type: Layer2OutputType`
  - concrete property `output_contract: AbstractLayer2OutputContract` (derived from `output_type`)
  - abstract property `column_name: str`

- `PivotedColumn(AbstractLayer2Column)`:
  - `categorical_combination: dict[CategoricalField, str | None]` — `None` = ∅ marginal
  - `output_type` derived from `source_measurement.measurement_type`: `FLAG` → `FLAG`; `FECHA` → `TEMPORAL`; all others → `NUMERIC`
  - `column_name` derived as `{aggregator}__{measurement}__{catN_val}...` where categorical entries are sorted alphabetically by `field.name` — guarantees deterministic, collision-free names regardless of dict insertion order
  - Validates `layer2_aggregator ∈ source_measurement.contract.valid_layer2_aggregators` at construction

- `DistributionalColumn(AbstractLayer2Column)`:
  - `categorical: CategoricalField`
  - `distributional_metric: DistributionalMetric`
  - `output_type` derived from `distributional_metric`: MODE → `CATEGORICAL`; all others (COUNT, ENTROPY, HHI, DOMINANT_PROPORTION) → `NUMERIC`
  - `column_name` derived as `{categorical}__{measurement}__{aggregator}__{metric}`

**Acceptance criteria:**
- Invalid aggregator raises `ValueError` at construction (PivotedColumn)
- `output_contract` is always consistent with `output_type`
- Column name derivation is deterministic and collision-free for distinct inputs
- Tests in `tests/test_layer2.py`

---

## Plan 07 — TemporalFeature

**Module:** `src/featkit/layer3/temporal_feature.py`

**Depends on:** Plan 01, Plan 05, Plan 06

**Scope:** The data model for a single Layer 3 output column. Validates operator compatibility at construction time.

**Class:**

- `TemporalFeature`:
  - `source: AbstractLayer2Column`
  - `operator: TemporalOperator`
  - `window_size: int | None` (required for windowed operators; `None` for point-in-time operators like `ULT_MES`, `REC`)
  - `direction: TimeWindowDirection`
  - derived property `column_name: str` — `{source.column_name}__{operator}__{direction}__{window_size}`
  - Validates `operator ∈ source.output_contract.valid_temporal_operators` at construction
  - Validates `window_size` is provided for operators that require it

**Acceptance criteria:**
- Invalid operator raises `ValueError` at construction
- Missing `window_size` for windowed operator raises `ValueError`
- Column name is deterministic
- Tests in `tests/test_layer3.py`

---

## Plan 08 — AbstractDataset

**Module:** `src/featkit/dataset/base.py`

**Depends on:** Plan 01, Plan 02, Plan 03

**Scope:** The schema descriptor for the input facts table. Pure metadata — no data access, no materialisation.

**Classes:**

- `AbstractDataset` (ABC):
  - `source_reference: str` — fully-qualified table name or SQL string
  - `fields: list[AbstractField]`
  - Derived properties (filter by `FieldRole`):
    - `id_fields: list[IDField]`
    - `time_field: TimeField` — raises `ValueError` if missing or multiple
    - `categorical_fields: list[CategoricalField]`
    - `measurement_fields: list[MeasurementField]`
  - `validate()` — checks that exactly one `TIME` field and at least one `ID` and one `MEASUREMENT` field exist

- `SimpleDataset(AbstractDataset)`: concrete, no-subclass entry point. Constructor takes `source_reference: str` and `fields: list[AbstractField]` directly. Implements no abstract methods beyond what `AbstractDataset` already provides. Used in quickstart and tests.

**Acceptance criteria:**
- `validate()` raises descriptive `ValueError` for each violation
- Derived field properties always return correct subsets
- `SimpleDataset` is instantiable with only `source_reference` and `fields`
- Tests in `tests/test_fields.py`

---

## Plan 09 — PivotSpaceBuilder

**Module:** `src/featkit/builders/pivot_space.py`

**Depends on:** Plan 06, Plan 08

**Scope:** Generates the full set of `PivotedColumn` objects from a dataset, including marginals (∅ combinations).

**Class:**

- `PivotSpaceBuilder`:
  - `dataset: AbstractDataset`
  - `include_marginals: bool` — whether to include ∅ substitutions for each categorical
  - `aggregators_override: dict[MeasurementType, list[Layer2Aggregator]] | None`
  - `domain_resolver: Callable[[CategoricalField], list[str]] | None` — invoked when `field.allowed_values is None`; typically wraps a `SELECT DISTINCT {col} FROM {table}` query via the dataset's data source adapter
  - `build() → list[PivotedColumn]`:
    1. For each pivot-eligible categorical: resolve domain via `field.allowed_values` (static) when set, otherwise via `domain_resolver(field)` (dynamic); raise `ValueError` if both are `None`
    2. Compute cartesian product of `(resolved_domain ∪ {None})` across all pivot-eligible categoricals
    3. For each combination × measurement × valid aggregator: create a `PivotedColumn`
    4. Skip invalid aggregator/measurement combinations (contract-governed)

**Acceptance criteria:**
- With 2 categoricals having 3 values each + marginals: generates `(3+1)² × measurements × agg` columns
- No duplicates in output
- Raises `ValueError` when `field.allowed_values` is `None` and no `domain_resolver` is provided
- Tests in `tests/test_builders.py`

---

## Plan 10 — DistributionalSpaceBuilder

**Module:** `src/featkit/builders/distributional_space.py`

**Depends on:** Plan 06, Plan 08

**Scope:** Generates the full set of `DistributionalColumn` objects for all `DISTRIBUTIONAL`-treatment categorical fields.

**Class:**

- `DistributionalSpaceBuilder`:
  - `dataset: AbstractDataset`
  - `value_measurements: list[MeasurementField] | None` — if `None`, use all measurement fields
  - `build() → list[DistributionalColumn]`:
    For each (categorical with DISTRIBUTIONAL treatment) × (measurement) × (base_agg per contract) × (metric in categorical.distributional_metrics): create a `DistributionalColumn`

**Acceptance criteria:**
- Only processes categoricals where `treatment ∈ {DISTRIBUTIONAL, BOTH}`
- Tests in `tests/test_builders.py`

---

## Plan 11 — TemporalSpaceBuilder

**Module:** `src/featkit/builders/temporal_space.py`

**Depends on:** Plan 05, Plan 06, Plan 07

**Scope:** Generates the full set of `TemporalFeature` objects from a list of Layer 2 columns.

**Class:**

- `TemporalSpaceBuilder`:
  - `layer2_columns: list[AbstractLayer2Column]`
  - `time_windows: list[int]`
  - `composed_windows: list[int] | None` — for `MEDIA_ABS` and `RATIO`
  - `direction: TimeWindowDirection`
  - `operators_override: dict[Layer2OutputType, list[TemporalOperator]] | None`
  - `build() → list[TemporalFeature]`:
    For each layer2 column × valid operator (from output contract) × window size: create a `TemporalFeature`. Point-in-time operators (`ULT_MES`, `PREV_MES`, `REC`) get `window_size=None`.

**Acceptance criteria:**
- Output set only contains contract-valid operator/column combinations
- `MEDIA_ABS` and `RATIO` only generated when `composed_windows` is provided
- Tests in `tests/test_builders.py`

---

## Plan 12 — FeatureStoreConfig + FeatureStorePipeline

**Module:** `src/featkit/config.py`, `pipeline.py`

**Depends on:** Plans 01–11

**Scope:** Top-level configuration object and the pipeline orchestrator that ties builders together.

**Classes:**

- `FeatureStoreConfig`:
  - `dataset: AbstractDataset`
  - `output_schema: str`
  - `output_table_prefix: str`
  - `time_windows: list[int]`
  - `composed_windows: list[int] | None`
  - `include_marginals: bool`
  - `aggregators_override: dict[MeasurementType, list[Layer2Aggregator]] | None`
  - `operators_override: dict[Layer2OutputType, list[TemporalOperator]] | None`

- `FeatureStorePipeline`:
  - `config: FeatureStoreConfig`
  - `layer2a: list[PivotedColumn]` — populated by `build()`
  - `layer2b: list[DistributionalColumn]` — populated by `build()`
  - `layer3: list[TemporalFeature]` — populated by `build()`
  - `build() → self` — calls the three builders in sequence
  - `run(generator: AbstractCodeGenerator) → FeatureStoreOutput`

**Acceptance criteria:**
- `build()` is idempotent (calling twice gives same result)
- `run()` delegates entirely to the generator
- Tests in `tests/test_pipeline.py`

---

## Plan 13 — AbstractCodeGenerator + FeatureStoreOutput + DAG

**Module:** `src/featkit/generators/base.py`, `output.py`

**Depends on:** Plan 12

**Scope:** The abstract code generator interface, the output container, and the DAG/Mermaid generation logic (engine-agnostic).

**Classes:**

- `DAGNode`: `step_name: str`, `depends_on: list[str]`
- `DAG`: `nodes: list[DAGNode]`, `to_json() → str`
- `SQLOutput`: `sql: str`, `dialect: str`, `save(path: str) → None`
- `PySparkOutput`: placeholder for PySpark lazy chain
- `CodeOutput = SQLOutput | PySparkOutput`
- `FeatureStoreOutput`: `code: CodeOutput`, `dag: DAG`, `mermaid: str`, `save(directory: str) → None`

- `AbstractCodeGenerator` (ABC):
  - Abstract: `build_mob_table`, `build_layer2a`, `build_layer2b`, `build_layer3`, `build_final_join` — each takes `pipeline: FeatureStorePipeline` and returns `CodeOutput`
  - Concrete: `build_dag(pipeline) → DAG`
  - Concrete: `build_mermaid(dag: DAG) → str`
  - Concrete: `generate(pipeline) → FeatureStoreOutput` — orchestrates all steps

**DAG structure (fixed for all generators):**
```
facts_table → mob_table
facts_table → layer2a_pivot
facts_table → layer2b_distributional_ctes
layer2a_pivot + layer2b_distributional_ctes → layer2_join
layer2_join + mob_table → layer3_temporal
layer2_join + layer3_temporal → final_output
```

**Acceptance criteria:**
- `build_dag` produces correct node/edge structure
- `build_mermaid` produces valid Mermaid flowchart syntax
- `FeatureStoreOutput.save()` writes: `script.sql` (when `code` is `SQLOutput`), `script.py` (when `code` is `PySparkOutput`), plus `dag.json` and `diagram.md` in both cases
- Tests in `tests/test_generators/`

---

## Plan 14 — AbstractSQLCodeGenerator + SnowflakeSQLCodeGenerator

**Module:** `src/featkit/generators/sql/base.py`, `snowflake.py`

**Depends on:** Plan 13

**Scope:** SQL generation base class and first concrete SQL dialect. Uses SQLGlot for expression building and rendering.

**Classes:**

- `AbstractSQLCodeGenerator(AbstractCodeGenerator)` (ABC):
  - Implements `build_mob_table`, `build_layer2a`, `build_layer2b`, `build_layer3`, `build_final_join` using SQLGlot expression trees
  - Abstract: `dialect: str` — the SQLGlot dialect string
  - `render(expr) → str` — renders SQLGlot expression to `dialect`
  - Layer 2B builds one CTE per categorical field, joined in a single query

- `SnowflakeSQLCodeGenerator(AbstractSQLCodeGenerator)`:
  - `dialect = "snowflake"`
  - Handles Snowflake-specific syntax: `QUALIFY`, `ARRAY_CONSTRUCT`
  - All distributional metrics (ENTROPY, HHI, DOMINANT_PROPORTION, MODE, COUNT) are expressed as pure SQL aggregate expressions within CTEs — no custom UDAFs or stored procedures

**Acceptance criteria:**
- Generated SQL is parseable by SQLGlot in the target dialect
- MOB table uses `ROW_NUMBER() OVER (ORDER BY time_col)` cross-join pattern
- Layer 2B produces one CTE per categorical, consolidated via `LEFT JOIN USING`
- Tests in `tests/test_generators/test_sql_snowflake.py`

---

## Plan 15 — DatabricksSQLCodeGenerator + SparkSQLCodeGenerator

**Module:** `src/featkit/generators/sql/databricks.py`, `spark_sql.py`

**Depends on:** Plan 14

**Scope:** SQL generators for Databricks SQL and Spark SQL dialects. Inherits all logic from `AbstractSQLCodeGenerator`, overrides dialect and any syntax differences.

**Classes:**

- `DatabricksSQLCodeGenerator(AbstractSQLCodeGenerator)`: `dialect = "databricks"`
- `SparkSQLCodeGenerator(AbstractSQLCodeGenerator)`: `dialect = "sparksql"`

**Key dialect differences to handle:**
- Table creation syntax (`CREATE OR REPLACE` vs `CREATE TABLE IF NOT EXISTS`)
- Temporary table syntax

**Acceptance criteria:**
- Generated SQL is parseable by SQLGlot in each respective dialect
- Tests in `tests/test_generators/test_sql_databricks.py`

---

## Plan 16 — PySparkCodeGenerator

**Module:** `src/featkit/generators/pyspark/databricks.py`

**Depends on:** Plan 13

**Scope:** PySpark-native code generator that produces lazy DataFrame transformations instead of SQL strings. All steps use the PySpark DataFrame API; final output is a `PySparkOutput` containing a callable that, when invoked with a `SparkSession`, executes the full pipeline.

**Class:**

- `PySparkCodeGenerator(AbstractCodeGenerator)`:
  - `build_mob_table` → cross-join via `.crossJoin()` + `row_number()` window
  - `build_layer2a` → `.groupBy()` + `F.when()` CASE WHEN pivot
  - `build_layer2b` → per-categorical `.groupBy()` + pure aggregate expressions (entropy, HHI, dominant proportion, mode, count via standard PySpark functions), then join
  - `build_layer3` → MOB join + `.agg()` with windowed CASE WHEN
  - `build_final_join` → `left_join` of Layer 2 and Layer 3 DataFrames
  - All steps are lazy — no `.collect()` or `.show()` calls

**Acceptance criteria:**
- All transformations are lazy until the returned callable is invoked
- Tests in `tests/test_generators/test_pyspark.py` (may use mocked SparkSession)

---

## Plan 17 — Integration tests

**Module:** `tests/` (cross-cutting)

**Depends on:** Plans 01–16

**Scope:** End-to-end tests that run a complete pipeline from `AbstractDataset` definition through to generated SQL or PySpark plan, validating structural correctness of the output.

**Test scenarios:**

1. **Snowflake SQL — pivot only**: dataset with 2 categoricals (PIVOT treatment), 2 measurements, windows [3, 6], no marginals → validate column count and SQL syntax
2. **Snowflake SQL — distributional only**: dataset with 2 categoricals (DISTRIBUTIONAL), 2 measurements, ENTROPY + HHI UDAFs → validate CTE structure
3. **Snowflake SQL — full pipeline**: PIVOT + DISTRIBUTIONAL categoricals, composed windows, marginals enabled → validate complete output file
4. **Databricks SQL — full pipeline**: same as (3) with Databricks dialect
5. **PySpark — full pipeline**: same dataset with `PySparkCodeGenerator` → validate DataFrame lineage
6. **DAG correctness**: all scenarios → validate node count and edge structure in DAG
7. **Mermaid output**: validate generated Mermaid string is syntactically valid

**Acceptance criteria:**
- All scenarios pass without errors
- SQL outputs are parseable by SQLGlot
- Column counts match expected pivot space size for given inputs
