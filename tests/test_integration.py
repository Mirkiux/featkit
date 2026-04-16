"""Plan 17 — Integration tests.

End-to-end scenarios that drive the full pipeline from dataset definition
through to generated code, validating structural correctness, SQL
parseability, column counts, DAG topology, file output, and Mermaid syntax.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
from typing import cast

import pytest
import sqlglot

from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    MeasurementType,
    TimeGranularity,
)
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.generators.output import FeatureStoreOutput, PySparkOutput, SQLOutput
from featkit.generators.pyspark.databricks import PySparkCodeGenerator
from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator
from featkit.generators.sql.snowflake import SnowflakeSQLCodeGenerator
from featkit.generators.sql.spark_sql import SparkSQLCodeGenerator
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared pipeline factories
# ---------------------------------------------------------------------------


def _pivot_only_pipeline() -> FeatureStorePipeline:
    """Two pivot categoricals, one measurement, windows [3, 6]."""
    ds = SimpleDataset(
        "db.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A", "B"]),
            CategoricalField("region", CategoricalTreatment.PIVOT, allowed_values=["N", "S"]),
        ],
    )
    cfg = FeatureStoreConfig(
        dataset=ds,
        output_schema="out",
        output_table_prefix="p_",
        time_windows=[3, 6],
        include_marginals=True,
    )
    return FeatureStorePipeline(config=cfg).build()


def _distributional_only_pipeline() -> FeatureStorePipeline:
    """One distributional categorical, one measurement, window [3]."""
    ds = SimpleDataset(
        "db.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            CategoricalField(
                "region",
                CategoricalTreatment.DISTRIBUTIONAL,
                distributional_metrics=[DistributionalMetric.ENTROPY, DistributionalMetric.HHI],
            ),
        ],
    )
    cfg = FeatureStoreConfig(
        dataset=ds,
        output_schema="out",
        output_table_prefix="p_",
        time_windows=[3],
    )
    return FeatureStorePipeline(config=cfg).build()


def _full_pipeline() -> FeatureStorePipeline:
    """Two IDs, two measurements, pivot + distributional, windows [3, 6]."""
    ds = SimpleDataset(
        "db.facts",
        [
            IDField("country"),
            IDField("client_id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            MeasurementField("cnt", MeasurementType.CANTIDAD),
            CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A", "B", "C"]),
            CategoricalField(
                "channel",
                CategoricalTreatment.DISTRIBUTIONAL,
                distributional_metrics=[DistributionalMetric.ENTROPY, DistributionalMetric.MODE],
            ),
        ],
    )
    cfg = FeatureStoreConfig(
        dataset=ds,
        output_schema="analytics",
        output_table_prefix="feat_",
        time_windows=[3, 6],
        composed_windows=[12],
    )
    return FeatureStorePipeline(config=cfg).build()


def _parseable_sql(sql: str, dialect: str) -> bool:
    try:
        sqlglot.parse_one(sql, dialect=dialect)
        return True
    except Exception:
        return False


def _all_statements_parseable(sql: str, dialect: str) -> bool:
    for stmt in sql.split("\n\n"):
        stmt = stmt.strip()
        if stmt and not _parseable_sql(stmt, dialect):
            return False
    return True


# ---------------------------------------------------------------------------
# Scenario 1 — Snowflake SQL, pivot only
# ---------------------------------------------------------------------------


class TestSnowflakePivotOnly:
    """Full pivot-only pipeline through SnowflakeSQLCodeGenerator."""

    def setup_method(self) -> None:
        self.gen = SnowflakeSQLCodeGenerator()
        self.pipeline = _pivot_only_pipeline()
        self.result = self.gen.generate(self.pipeline)

    def test_returns_feature_store_output(self) -> None:
        assert isinstance(self.result, FeatureStoreOutput)

    def test_code_is_sql_output(self) -> None:
        assert isinstance(self.result.code, SQLOutput)

    def test_dialect_is_snowflake(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert self.result.code.dialect == "snowflake"

    def test_layer2a_column_count(self) -> None:
        # 2 cats × (2+1 values with marginals each) × 4 valid agg for MONTO = 36
        assert len(self.pipeline.layer2a) == 36

    def test_no_layer2b_columns(self) -> None:
        assert len(self.pipeline.layer2b) == 0

    def test_layer2b_sql_is_empty(self) -> None:
        out = self.gen.build_layer2b(self.pipeline)
        assert isinstance(out, SQLOutput)
        assert out.sql == ""

    def test_layer3_column_names_are_unique(self) -> None:
        names = [f.column_name for f in self.pipeline.layer3]
        assert len(names) == len(set(names))

    def test_all_statements_parseable(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert _all_statements_parseable(self.result.code.sql, "snowflake")

    def test_no_layer2b_reference_in_layer3(self) -> None:
        assert "layer2b" not in self.gen.build_layer3(self.pipeline).sql

    def test_no_layer2b_reference_in_final_join(self) -> None:
        assert "layer2b" not in self.gen.build_final_join(self.pipeline).sql


# ---------------------------------------------------------------------------
# Scenario 2 — Snowflake SQL, distributional only
# ---------------------------------------------------------------------------


class TestSnowflakeDistributionalOnly:
    """Full distributional-only pipeline through SnowflakeSQLCodeGenerator."""

    def setup_method(self) -> None:
        self.gen = SnowflakeSQLCodeGenerator()
        self.pipeline = _distributional_only_pipeline()
        self.result = self.gen.generate(self.pipeline)

    def test_layer2b_column_count(self) -> None:
        # 1 cat × 1 meas × 4 valid agg (MONTO) × 2 metrics (ENTROPY, HHI) = 8
        assert len(self.pipeline.layer2b) == 8

    def test_layer2b_sql_contains_cte_structure(self) -> None:
        out = self.gen.build_layer2b(self.pipeline)
        assert isinstance(out, SQLOutput)
        sql = out.sql
        assert "_raw" in sql
        assert "_shares" in sql
        assert "_metrics" in sql

    def test_layer2b_no_nested_aggregate(self) -> None:
        out = self.gen.build_layer2b(self.pipeline)
        assert isinstance(out, SQLOutput)
        assert "SUM(SUM(" not in out.sql.upper()

    def test_layer2b_entropy_uses_ln(self) -> None:
        out = self.gen.build_layer2b(self.pipeline)
        assert isinstance(out, SQLOutput)
        assert "LN(" in out.sql.upper()

    def test_layer2b_hhi_uses_power(self) -> None:
        out = self.gen.build_layer2b(self.pipeline)
        assert isinstance(out, SQLOutput)
        assert "POWER(" in out.sql.upper()

    def test_layer2b_sql_parseable(self) -> None:
        out = self.gen.build_layer2b(self.pipeline)
        assert isinstance(out, SQLOutput)
        assert _parseable_sql(out.sql, "snowflake")

    def test_layer3_references_layer2b(self) -> None:
        assert "layer2b" in self.gen.build_layer3(self.pipeline).sql

    def test_final_join_references_layer2b(self) -> None:
        assert "layer2b" in self.gen.build_final_join(self.pipeline).sql

    def test_all_statements_parseable(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert _all_statements_parseable(self.result.code.sql, "snowflake")


# ---------------------------------------------------------------------------
# Scenario 3 — Snowflake SQL, full pipeline + file output
# ---------------------------------------------------------------------------


class TestSnowflakeFullPipeline:
    """Full mixed pipeline through SnowflakeSQLCodeGenerator, including save()."""

    def setup_method(self) -> None:
        self.gen = SnowflakeSQLCodeGenerator()
        self.pipeline = _full_pipeline()
        self.result = self.gen.generate(self.pipeline)

    def test_layer2a_columns_present(self) -> None:
        assert len(self.pipeline.layer2a) > 0

    def test_layer2b_columns_present(self) -> None:
        assert len(self.pipeline.layer2b) > 0

    def test_layer3_columns_present(self) -> None:
        assert len(self.pipeline.layer3) > 0

    def test_layer2a_column_names_unique(self) -> None:
        names = [c.column_name for c in self.pipeline.layer2a]
        assert len(names) == len(set(names))

    def test_layer3_column_names_unique(self) -> None:
        names = [f.column_name for f in self.pipeline.layer3]
        assert len(names) == len(set(names))

    def test_all_statements_parseable(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert _all_statements_parseable(self.result.code.sql, "snowflake")

    def test_save_writes_script_sql(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            assert "script.sql" in os.listdir(d)

    def test_save_writes_dag_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            assert "dag.json" in os.listdir(d)

    def test_save_writes_diagram_md(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            assert "diagram.md" in os.listdir(d)

    def test_dag_json_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            with open(os.path.join(d, "dag.json")) as f:
                data = json.loads(f.read())
            assert isinstance(data, list)
            assert len(data) == 7

    def test_diagram_md_contains_mermaid(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            with open(os.path.join(d, "diagram.md")) as f:
                content = f.read()
            assert "flowchart TD" in content

    def test_sql_contains_multi_id_group_by(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        sql = self.result.code.sql
        assert "country" in sql
        assert "client_id" in sql


# ---------------------------------------------------------------------------
# Scenario 4 — Databricks SQL, full pipeline
# ---------------------------------------------------------------------------


class TestDatabricksFullPipeline:
    """Full mixed pipeline through DatabricksSQLCodeGenerator."""

    def setup_method(self) -> None:
        self.gen = DatabricksSQLCodeGenerator()
        self.pipeline = _full_pipeline()
        self.result = self.gen.generate(self.pipeline)

    def test_returns_feature_store_output(self) -> None:
        assert isinstance(self.result, FeatureStoreOutput)

    def test_dialect_is_databricks(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert self.result.code.dialect == "databricks"

    def test_uses_backtick_quoting(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert "`" in self.result.code.sql

    def test_all_statements_parseable(self) -> None:
        assert isinstance(self.result.code, SQLOutput)
        assert _all_statements_parseable(self.result.code.sql, "databricks")

    def test_same_layer_counts_as_snowflake(self) -> None:
        """Pipeline is generator-agnostic; layer counts must match."""
        sf_pipeline = _full_pipeline()
        assert len(self.pipeline.layer2a) == len(sf_pipeline.layer2a)
        assert len(self.pipeline.layer2b) == len(sf_pipeline.layer2b)
        assert len(self.pipeline.layer3) == len(sf_pipeline.layer3)

    def test_save_writes_script_sql(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            assert "script.sql" in os.listdir(d)


# ---------------------------------------------------------------------------
# Scenario 5 — PySpark, full pipeline
# ---------------------------------------------------------------------------


class TestPySparkFullPipeline:
    """Full mixed pipeline through PySparkCodeGenerator."""

    def setup_method(self) -> None:
        self.gen = PySparkCodeGenerator()
        self.pipeline = _full_pipeline()
        self.result = self.gen.generate(self.pipeline)

    def test_returns_feature_store_output(self) -> None:
        assert isinstance(self.result, FeatureStoreOutput)

    def test_code_is_pyspark_output(self) -> None:
        assert isinstance(self.result.code, PySparkOutput)

    def test_combined_script_is_valid_python(self) -> None:
        assert isinstance(self.result.code, PySparkOutput)
        ast.parse(self.result.code.code)

    def test_script_contains_pyspark_imports(self) -> None:
        assert isinstance(self.result.code, PySparkOutput)
        code = self.result.code.code
        assert "from pyspark.sql import SparkSession" in code
        assert "from pyspark.sql import functions as F" in code
        assert "from pyspark.sql.window import Window" in code

    def test_no_collect_or_show(self) -> None:
        assert isinstance(self.result.code, PySparkOutput)
        code = self.result.code.code
        assert ".collect()" not in code
        assert ".show()" not in code

    def test_script_contains_all_steps(self) -> None:
        assert isinstance(self.result.code, PySparkOutput)
        code = self.result.code.code
        assert "mob_ref" in code
        assert "layer2a" in code
        assert "layer2b" in code
        assert "layer3" in code
        assert "features" in code

    def test_mob_uses_period_self_join(self) -> None:
        assert isinstance(self.result.code, PySparkOutput)
        code = self.result.code.code
        assert "_periodos_ordenados" in code
        assert "ts_analysis" in code
        assert "ts_relative" in code

    def test_save_writes_script_py(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            assert "script.py" in os.listdir(d)

    def test_save_writes_dag_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.result.save(d)
            assert "dag.json" in os.listdir(d)

    def test_same_layer_counts_as_sql(self) -> None:
        sql_pipeline = _full_pipeline()
        assert len(self.pipeline.layer2a) == len(sql_pipeline.layer2a)
        assert len(self.pipeline.layer2b) == len(sql_pipeline.layer2b)
        assert len(self.pipeline.layer3) == len(sql_pipeline.layer3)


# ---------------------------------------------------------------------------
# Scenario 6 — DAG correctness (all generators)
# ---------------------------------------------------------------------------


_ALL_GENERATORS = [
    SnowflakeSQLCodeGenerator(),
    DatabricksSQLCodeGenerator(),
    SparkSQLCodeGenerator(),
    PySparkCodeGenerator(),
]

_GENERATOR_IDS = ["snowflake", "databricks", "spark", "pyspark"]


class TestDAGCorrectness:
    """DAG topology is generator-agnostic and must be consistent across all."""

    @pytest.fixture(params=_ALL_GENERATORS, ids=_GENERATOR_IDS)
    def result(self, request: pytest.FixtureRequest) -> FeatureStoreOutput:
        gen = request.param
        return cast(FeatureStoreOutput, gen.generate(_full_pipeline()))

    def test_seven_nodes(self, result: FeatureStoreOutput) -> None:
        assert len(result.dag.nodes) == 7

    def test_node_names(self, result: FeatureStoreOutput) -> None:
        names = {n.step_name for n in result.dag.nodes}
        assert names == {
            "facts_table",
            "mob_table",
            "layer2a_pivot",
            "layer2b_distributional_ctes",
            "layer2_join",
            "layer3_temporal",
            "final_output",
        }

    def test_facts_table_has_no_dependencies(self, result: FeatureStoreOutput) -> None:
        facts = next(n for n in result.dag.nodes if n.step_name == "facts_table")
        assert facts.depends_on == []

    def test_mob_table_depends_on_facts(self, result: FeatureStoreOutput) -> None:
        mob = next(n for n in result.dag.nodes if n.step_name == "mob_table")
        assert "facts_table" in mob.depends_on

    def test_layer2_join_depends_on_both_layer2(self, result: FeatureStoreOutput) -> None:
        join = next(n for n in result.dag.nodes if n.step_name == "layer2_join")
        assert "layer2a_pivot" in join.depends_on
        assert "layer2b_distributional_ctes" in join.depends_on

    def test_layer3_depends_on_layer2_join_and_mob(self, result: FeatureStoreOutput) -> None:
        l3 = next(n for n in result.dag.nodes if n.step_name == "layer3_temporal")
        assert "layer2_join" in l3.depends_on
        assert "mob_table" in l3.depends_on

    def test_final_output_depends_on_layer2_join_and_layer3(
        self, result: FeatureStoreOutput
    ) -> None:
        final = next(n for n in result.dag.nodes if n.step_name == "final_output")
        assert "layer2_join" in final.depends_on
        assert "layer3_temporal" in final.depends_on

    def test_dag_json_round_trips(self, result: FeatureStoreOutput) -> None:
        data = json.loads(result.dag.to_json())
        assert isinstance(data, list)
        assert len(data) == 7
        assert all("step_name" in node and "depends_on" in node for node in data)


# ---------------------------------------------------------------------------
# Scenario 7 — Mermaid output (all generators)
# ---------------------------------------------------------------------------


class TestMermaidOutput:
    @pytest.fixture(params=_ALL_GENERATORS, ids=_GENERATOR_IDS)
    def result(self, request: pytest.FixtureRequest) -> FeatureStoreOutput:
        return request.param.generate(_full_pipeline())

    def test_starts_with_flowchart_td(self, result: FeatureStoreOutput) -> None:
        assert result.mermaid.startswith("flowchart TD")

    def test_contains_all_node_names(self, result: FeatureStoreOutput) -> None:
        mermaid = result.mermaid
        assert "facts_table" in mermaid
        assert "mob_table" in mermaid
        assert "layer2a_pivot" in mermaid
        assert "layer3_temporal" in mermaid
        assert "final_output" in mermaid

    def test_contains_arrow_syntax(self, result: FeatureStoreOutput) -> None:
        assert "-->" in result.mermaid
