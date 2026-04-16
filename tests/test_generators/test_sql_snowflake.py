"""Tests for Plan 14 — AbstractSQLCodeGenerator and SnowflakeSQLCodeGenerator."""

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
from featkit.generators.sql.snowflake import SnowflakeSQLCodeGenerator
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GEN = SnowflakeSQLCodeGenerator()


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
        output_schema="out",
        output_table_prefix="feat_",
        time_windows=[3],
    )
    return FeatureStorePipeline(config=cfg).build()


def _pipeline_pivot_only() -> FeatureStorePipeline:
    """Pipeline with only pivot categoricals (no distributional)."""
    ds = SimpleDataset(
        "db.facts",
        [
            IDField("id"),
            TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("mto", MeasurementType.MONTO),
            CategoricalField("sector", CategoricalTreatment.PIVOT, allowed_values=["A", "B"]),
        ],
    )
    cfg = FeatureStoreConfig(
        dataset=ds,
        output_schema="analytics",
        output_table_prefix="feat_",
        time_windows=[3, 6],
    )
    return FeatureStorePipeline(config=cfg).build()


def _pipeline_all_metrics() -> FeatureStorePipeline:
    """Pipeline with all distributional metrics."""
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
        output_schema="out",
        output_table_prefix="p_",
        time_windows=[3],
    )
    return FeatureStorePipeline(config=cfg).build()


def _is_parseable(sql: str, dialect: str = "snowflake") -> bool:
    """Return True if *sql* is parseable by SQLGlot in the given dialect."""
    try:
        sqlglot.parse_one(sql, dialect=dialect)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SnowflakeSQLCodeGenerator dialect
# ---------------------------------------------------------------------------


class TestDialect:
    def test_dialect_is_snowflake(self) -> None:
        assert SnowflakeSQLCodeGenerator().dialect == "snowflake"

    def test_render_uses_snowflake_dialect(self) -> None:
        gen = SnowflakeSQLCodeGenerator()
        expr = sqlglot.parse_one("SELECT 1", dialect="snowflake")
        sql = gen.render(expr)
        assert "SELECT" in sql.upper()


# ---------------------------------------------------------------------------
# generate() end-to-end
# ---------------------------------------------------------------------------


class TestGenerateEndToEnd:
    def test_returns_feature_store_output(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result, FeatureStoreOutput)

    def test_code_is_sql_output(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)

    def test_dialect_preserved(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        assert result.code.dialect == "snowflake"

    def test_combined_sql_contains_all_steps(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        sql = result.code.sql
        assert "mob_ref" in sql
        assert "layer2a" in sql
        assert "layer2b" in sql
        assert "layer3" in sql
        assert "features" in sql

    def test_each_statement_is_parseable(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, SQLOutput)
        for stmt in result.code.sql.split("\n\n"):
            stmt = stmt.strip()
            if stmt:
                assert _is_parseable(stmt), f"Unparseable statement:\n{stmt}"

    def test_dag_has_seven_nodes(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert len(result.dag.nodes) == 7

    def test_mermaid_is_valid(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert result.mermaid.startswith("flowchart TD")


# ---------------------------------------------------------------------------
# build_mob_table
# ---------------------------------------------------------------------------


class TestBuildMobTable:
    def _sql(self, pipeline: FeatureStorePipeline | None = None) -> str:
        p = pipeline or _pipeline_full()
        out = _GEN.build_mob_table(p)
        assert isinstance(out, SQLOutput)
        return out.sql

    def test_contains_row_number(self) -> None:
        assert "ROW_NUMBER" in self._sql().upper()

    def test_uses_period_ctes(self) -> None:
        assert "periodos_ordenados" in self._sql()

    def test_mob_uses_periodos_unicos(self) -> None:
        assert "periodos_unicos" in self._sql()

    def test_references_source_table(self) -> None:
        assert "db.facts" in self._sql()

    def test_output_table_includes_prefix(self) -> None:
        assert "feat_mob_ref" in self._sql()

    def test_sql_is_parseable(self) -> None:
        assert _is_parseable(self._sql())

    def test_dialect_is_snowflake(self) -> None:
        out = _GEN.build_mob_table(_pipeline_full())
        assert out.dialect == "snowflake"


# ---------------------------------------------------------------------------
# build_layer2a
# ---------------------------------------------------------------------------


class TestBuildLayer2a:
    def _sql(self) -> str:
        out = _GEN.build_layer2a(_pipeline_full())
        assert isinstance(out, SQLOutput)
        return out.sql

    def test_contains_group_by(self) -> None:
        assert "GROUP BY" in self._sql().upper()

    def test_contains_case_when_for_pivot(self) -> None:
        assert "CASE WHEN" in self._sql().upper()

    def test_contains_output_table(self) -> None:
        assert "feat_layer2a" in self._sql()

    def test_references_source_table(self) -> None:
        assert "db.facts" in self._sql()

    def test_sql_is_parseable(self) -> None:
        assert _is_parseable(self._sql())

    def test_uncategorized_pipeline_has_no_case_when(self) -> None:
        ds = SimpleDataset(
            "tbl",
            [
                IDField("id"),
                TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("mto", MeasurementType.MONTO),
            ],
        )
        cfg = FeatureStoreConfig(
            dataset=ds, output_schema="s", output_table_prefix="p_", time_windows=[3]
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        out = _GEN.build_layer2a(pipeline)
        assert isinstance(out, SQLOutput)
        assert "CASE WHEN" not in out.sql.upper()
        assert "GROUP BY" in out.sql.upper()


# ---------------------------------------------------------------------------
# build_layer2b
# ---------------------------------------------------------------------------


class TestBuildLayer2b:
    def _sql(self) -> str:
        out = _GEN.build_layer2b(_pipeline_full())
        assert isinstance(out, SQLOutput)
        return out.sql

    def test_contains_with_cte(self) -> None:
        assert "WITH" in self._sql().upper()

    def test_contains_left_join(self) -> None:
        assert "LEFT JOIN" in self._sql().upper()

    def test_contains_shares_cte(self) -> None:
        assert "_shares" in self._sql()

    def test_contains_metrics_cte(self) -> None:
        assert "_metrics" in self._sql()

    def test_contains_output_table(self) -> None:
        assert "feat_layer2b" in self._sql()

    def test_sql_is_parseable(self) -> None:
        assert _is_parseable(self._sql())

    def test_empty_layer2b_returns_empty_sql(self) -> None:
        out = _GEN.build_layer2b(_pipeline_pivot_only())
        assert isinstance(out, SQLOutput)
        assert out.sql == ""

    def test_entropy_uses_ln(self) -> None:
        assert "LN(" in self._sql().upper()

    def test_all_metrics_parseable(self) -> None:
        out = _GEN.build_layer2b(_pipeline_all_metrics())
        assert isinstance(out, SQLOutput)
        assert _is_parseable(out.sql)

    def test_hhi_uses_power(self) -> None:
        sql = _GEN.build_layer2b(_pipeline_all_metrics()).sql.upper()
        assert "POWER(" in sql

    def test_mode_uses_max_by(self) -> None:
        sql = _GEN.build_layer2b(_pipeline_all_metrics()).sql.upper()
        assert "MAX_BY(" in sql

    def test_count_metric_present(self) -> None:
        sql = _GEN.build_layer2b(_pipeline_all_metrics()).sql.upper()
        assert "COUNT(" in sql

    def test_no_nested_aggregate(self) -> None:
        out = _GEN.build_layer2b(_pipeline_all_metrics())
        assert isinstance(out, SQLOutput)
        assert "SUM(SUM(" not in out.sql.upper()


# ---------------------------------------------------------------------------
# build_layer3
# ---------------------------------------------------------------------------


class TestBuildLayer3:
    def _sql(self) -> str:
        out = _GEN.build_layer3(_pipeline_full())
        assert isinstance(out, SQLOutput)
        return out.sql

    def test_contains_group_by(self) -> None:
        assert "GROUP BY" in self._sql().upper()

    def test_contains_case_when(self) -> None:
        assert "CASE WHEN" in self._sql().upper()

    def test_references_mob_ref(self) -> None:
        assert "mob_ref" in self._sql()

    def test_references_layer2a(self) -> None:
        assert "layer2a" in self._sql()

    def test_contains_output_table(self) -> None:
        assert "feat_layer3" in self._sql()

    def test_sql_is_parseable(self) -> None:
        assert _is_parseable(self._sql())

    def test_layer2b_join_present_when_distributional(self) -> None:
        assert "layer2b" in self._sql()

    def test_layer2b_join_absent_when_pivot_only(self) -> None:
        out = _GEN.build_layer3(_pipeline_pivot_only())
        assert "layer2b" not in out.sql


# ---------------------------------------------------------------------------
# build_final_join
# ---------------------------------------------------------------------------


class TestBuildFinalJoin:
    def _sql(self) -> str:
        out = _GEN.build_final_join(_pipeline_full())
        assert isinstance(out, SQLOutput)
        return out.sql

    def test_contains_output_table(self) -> None:
        assert "feat_features" in self._sql()

    def test_references_layer2a(self) -> None:
        assert "layer2a" in self._sql()

    def test_references_layer3(self) -> None:
        assert "layer3" in self._sql()

    def test_references_layer2b_when_distributional(self) -> None:
        assert "layer2b" in self._sql()

    def test_no_layer2b_when_pivot_only(self) -> None:
        out = _GEN.build_final_join(_pipeline_pivot_only())
        assert "layer2b" not in out.sql

    def test_sql_is_parseable(self) -> None:
        assert _is_parseable(self._sql())

    def test_no_select_star(self) -> None:
        sql = self._sql().upper()
        assert "SELECT *" not in sql
        assert "L2A.*" not in sql


# ---------------------------------------------------------------------------
# Table naming
# ---------------------------------------------------------------------------


class TestTableNaming:
    def test_schema_quoted_for_reserved_word(self) -> None:
        sql = _GEN.build_mob_table(_pipeline_full()).sql
        assert '"out"' in sql

    def test_prefix_applied(self) -> None:
        sql = _GEN.build_layer2a(_pipeline_pivot_only()).sql
        assert "feat_" in sql

    def test_custom_schema_and_prefix(self) -> None:
        ds = SimpleDataset(
            "src.tbl",
            [
                IDField("id"),
                TimeField("ts", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("mto", MeasurementType.MONTO),
            ],
        )
        cfg = FeatureStoreConfig(
            dataset=ds,
            output_schema="myschema",
            output_table_prefix="x_",
            time_windows=[3],
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        sql = _GEN.build_mob_table(pipeline).sql
        assert "myschema" in sql
        assert "x_mob_ref" in sql
