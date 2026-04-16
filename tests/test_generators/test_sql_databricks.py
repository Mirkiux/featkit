"""Tests for Plan 15 — DatabricksSQLCodeGenerator and SparkSQLCodeGenerator."""

from __future__ import annotations

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
from featkit.generators.output import FeatureStoreOutput, SQLOutput
from featkit.generators.sql.databricks import DatabricksSQLCodeGenerator
from featkit.generators.sql.spark_sql import SparkSQLCodeGenerator
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATABRICKS = DatabricksSQLCodeGenerator()
_SPARK = SparkSQLCodeGenerator()


def _pipeline_full() -> FeatureStorePipeline:
    """Pipeline with both pivot and distributional categoricals."""
    ds = SimpleDataset(
        "db.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A"]),
            CategoricalField(
                "region",
                CategoricalTreatment.DISTRIBUTIONAL,
                distributional_metrics=[DistributionalMetric.ENTROPY],
            ),
        ],
    )
    cfg = FeatureStoreConfig(
        dataset=ds,
        output_schema="myschema",
        output_table_prefix="feat_",
        time_windows=[3],
    )
    return FeatureStorePipeline(config=cfg).build()


def _pipeline_all_metrics() -> FeatureStorePipeline:
    """Pipeline exercising all distributional metrics."""
    ds = SimpleDataset(
        "db.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            CategoricalField(
                "region",
                CategoricalTreatment.DISTRIBUTIONAL,
                distributional_metrics=[
                    DistributionalMetric.ENTROPY,
                    DistributionalMetric.HHI,
                    DistributionalMetric.DOMINANT_PROPORTION,
                    DistributionalMetric.MODE,
                    DistributionalMetric.COUNT,
                ],
            ),
        ],
    )
    cfg = FeatureStoreConfig(
        dataset=ds,
        output_schema="myschema",
        output_table_prefix="p_",
        time_windows=[3],
    )
    return FeatureStorePipeline(config=cfg).build()


def _is_parseable(sql: str, dialect: str) -> bool:
    """Return True if *sql* is parseable by SQLGlot in *dialect*."""
    try:
        sqlglot.parse_one(sql, dialect=dialect)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dialect identity
# ---------------------------------------------------------------------------


class TestDialects:
    def test_databricks_dialect(self) -> None:
        assert _DATABRICKS.dialect == "databricks"

    def test_spark_dialect(self) -> None:
        assert _SPARK.dialect == "spark"

    def test_databricks_uses_backtick_quoting(self) -> None:
        sql = _DATABRICKS.build_mob_table(_pipeline_full()).sql
        assert "`" in sql

    def test_spark_uses_backtick_quoting(self) -> None:
        sql = _SPARK.build_mob_table(_pipeline_full()).sql
        assert "`" in sql

    def test_databricks_output_tagged_with_dialect(self) -> None:
        out = _DATABRICKS.build_mob_table(_pipeline_full())
        assert isinstance(out, SQLOutput)
        assert out.dialect == "databricks"

    def test_spark_output_tagged_with_dialect(self) -> None:
        out = _SPARK.build_mob_table(_pipeline_full())
        assert isinstance(out, SQLOutput)
        assert out.dialect == "spark"


# ---------------------------------------------------------------------------
# generate() end-to-end — Databricks
# ---------------------------------------------------------------------------


class TestDatabricksGenerateEndToEnd:
    def test_returns_feature_store_output(self) -> None:
        result = _DATABRICKS.generate(_pipeline_full())
        assert isinstance(result, FeatureStoreOutput)

    def test_code_is_sql_output(self) -> None:
        result = _DATABRICKS.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)

    def test_dialect_preserved(self) -> None:
        result = _DATABRICKS.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        assert result.code.dialect == "databricks"

    def test_combined_sql_contains_all_steps(self) -> None:
        result = _DATABRICKS.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        sql = result.code.sql
        assert "mob_ref" in sql
        assert "layer2a" in sql
        assert "layer2b" in sql
        assert "layer3" in sql
        assert "features" in sql

    def test_each_statement_is_parseable(self) -> None:
        result = _DATABRICKS.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        for stmt in result.code.sql.split("\n\n"):
            stmt = stmt.strip()
            if stmt:
                assert _is_parseable(stmt, "databricks"), f"Unparseable statement:\n{stmt}"

    def test_dag_has_seven_nodes(self) -> None:
        assert len(_DATABRICKS.generate(_pipeline_full()).dag.nodes) == 7

    def test_mermaid_is_valid(self) -> None:
        assert _DATABRICKS.generate(_pipeline_full()).mermaid.startswith("flowchart TD")


# ---------------------------------------------------------------------------
# generate() end-to-end — Spark
# ---------------------------------------------------------------------------


class TestSparkGenerateEndToEnd:
    def test_returns_feature_store_output(self) -> None:
        result = _SPARK.generate(_pipeline_full())
        assert isinstance(result, FeatureStoreOutput)

    def test_code_is_sql_output(self) -> None:
        result = _SPARK.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)

    def test_dialect_preserved(self) -> None:
        result = _SPARK.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        assert result.code.dialect == "spark"

    def test_combined_sql_contains_all_steps(self) -> None:
        result = _SPARK.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        sql = result.code.sql
        assert "mob_ref" in sql
        assert "layer2a" in sql
        assert "layer2b" in sql
        assert "layer3" in sql
        assert "features" in sql

    def test_each_statement_is_parseable(self) -> None:
        result = _SPARK.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        for stmt in result.code.sql.split("\n\n"):
            stmt = stmt.strip()
            if stmt:
                assert _is_parseable(stmt, "spark"), f"Unparseable statement:\n{stmt}"

    def test_dag_has_seven_nodes(self) -> None:
        assert len(_SPARK.generate(_pipeline_full()).dag.nodes) == 7

    def test_mermaid_is_valid(self) -> None:
        assert _SPARK.generate(_pipeline_full()).mermaid.startswith("flowchart TD")


# ---------------------------------------------------------------------------
# Per-step parseability — both dialects
# ---------------------------------------------------------------------------


class TestDatabricksStepParseability:
    def test_mob_table_parseable(self) -> None:
        assert _is_parseable(_DATABRICKS.build_mob_table(_pipeline_full()).sql, "databricks")

    def test_layer2a_parseable(self) -> None:
        assert _is_parseable(_DATABRICKS.build_layer2a(_pipeline_full()).sql, "databricks")

    def test_layer2b_parseable(self) -> None:
        assert _is_parseable(_DATABRICKS.build_layer2b(_pipeline_full()).sql, "databricks")

    def test_layer2b_all_metrics_parseable(self) -> None:
        assert _is_parseable(_DATABRICKS.build_layer2b(_pipeline_all_metrics()).sql, "databricks")

    def test_layer3_parseable(self) -> None:
        assert _is_parseable(_DATABRICKS.build_layer3(_pipeline_full()).sql, "databricks")

    def test_final_join_parseable(self) -> None:
        assert _is_parseable(_DATABRICKS.build_final_join(_pipeline_full()).sql, "databricks")


class TestSparkStepParseability:
    def test_mob_table_parseable(self) -> None:
        assert _is_parseable(_SPARK.build_mob_table(_pipeline_full()).sql, "spark")

    def test_layer2a_parseable(self) -> None:
        assert _is_parseable(_SPARK.build_layer2a(_pipeline_full()).sql, "spark")

    def test_layer2b_parseable(self) -> None:
        assert _is_parseable(_SPARK.build_layer2b(_pipeline_full()).sql, "spark")

    def test_layer2b_all_metrics_parseable(self) -> None:
        assert _is_parseable(_SPARK.build_layer2b(_pipeline_all_metrics()).sql, "spark")

    def test_layer3_parseable(self) -> None:
        assert _is_parseable(_SPARK.build_layer3(_pipeline_full()).sql, "spark")

    def test_final_join_parseable(self) -> None:
        assert _is_parseable(_SPARK.build_final_join(_pipeline_full()).sql, "spark")


# ---------------------------------------------------------------------------
# MOB table structure
# ---------------------------------------------------------------------------


class TestMobTable:
    def test_databricks_contains_periodos_unicos(self) -> None:
        assert "periodos_unicos" in _DATABRICKS.build_mob_table(_pipeline_full()).sql

    def test_databricks_contains_row_number(self) -> None:
        assert "ROW_NUMBER" in _DATABRICKS.build_mob_table(_pipeline_full()).sql.upper()

    def test_spark_contains_periodos_unicos(self) -> None:
        assert "periodos_unicos" in _SPARK.build_mob_table(_pipeline_full()).sql

    def test_spark_contains_row_number(self) -> None:
        assert "ROW_NUMBER" in _SPARK.build_mob_table(_pipeline_full()).sql.upper()


# ---------------------------------------------------------------------------
# Layer 2B: no nested aggregate
# ---------------------------------------------------------------------------


class TestLayer2bNoNestedAggregate:
    def test_databricks_no_nested_sum(self) -> None:
        out = _DATABRICKS.build_layer2b(_pipeline_all_metrics())
        assert isinstance(out, SQLOutput)
        assert "SUM(SUM(" not in out.sql.upper()

    def test_spark_no_nested_sum(self) -> None:
        out = _SPARK.build_layer2b(_pipeline_all_metrics())
        assert isinstance(out, SQLOutput)
        assert "SUM(SUM(" not in out.sql.upper()


# ---------------------------------------------------------------------------
# Table naming
# ---------------------------------------------------------------------------


class TestTableNaming:
    def test_databricks_schema_quoted(self) -> None:
        sql = _DATABRICKS.build_mob_table(_pipeline_full()).sql
        assert "`myschema`" in sql

    def test_spark_schema_quoted(self) -> None:
        sql = _SPARK.build_mob_table(_pipeline_full()).sql
        assert "`myschema`" in sql

    def test_databricks_prefix_applied(self) -> None:
        sql = _DATABRICKS.build_mob_table(_pipeline_full()).sql
        assert "feat_mob_ref" in sql

    def test_spark_prefix_applied(self) -> None:
        sql = _SPARK.build_mob_table(_pipeline_full()).sql
        assert "feat_mob_ref" in sql
