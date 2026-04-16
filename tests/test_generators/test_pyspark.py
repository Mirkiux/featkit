"""Tests for Plan 16 — PySparkCodeGenerator."""

from __future__ import annotations

import ast

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
from featkit.generators.output import FeatureStoreOutput, PySparkOutput
from featkit.generators.pyspark.databricks import PySparkCodeGenerator
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GEN = PySparkCodeGenerator()


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
    """Pipeline with all five distributional metrics."""
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


def _is_valid_python(code: str) -> bool:
    """Return True if *code* is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# generate() end-to-end
# ---------------------------------------------------------------------------


class TestGenerateEndToEnd:
    def test_returns_feature_store_output(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result, FeatureStoreOutput)

    def test_code_is_pyspark_output(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, PySparkOutput)

    def test_dag_has_seven_nodes(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert len(result.dag.nodes) == 7

    def test_mermaid_is_valid(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert result.mermaid.startswith("flowchart TD")

    def test_combined_code_is_valid_python(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, PySparkOutput)
        assert _is_valid_python(result.code.code)

    def test_header_contains_pyspark_imports(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, PySparkOutput)
        code = result.code.code
        assert "from pyspark.sql import SparkSession" in code
        assert "from pyspark.sql import functions as F" in code
        assert "from pyspark.sql.window import Window" in code

    def test_combined_code_contains_all_steps(self) -> None:
        result = _GEN.generate(_pipeline_full())
        assert isinstance(result.code, PySparkOutput)
        code = result.code.code
        assert "mob_ref" in code
        assert "layer2a" in code
        assert "layer2b" in code
        assert "layer3" in code
        assert "features" in code


# ---------------------------------------------------------------------------
# build_mob_table
# ---------------------------------------------------------------------------


class TestBuildMobTable:
    def _code(self, pipeline: FeatureStorePipeline | None = None) -> str:
        p = pipeline or _pipeline_full()
        out = _GEN.build_mob_table(p)
        assert isinstance(out, PySparkOutput)
        return out.code

    def test_contains_cross_join(self) -> None:
        assert "crossJoin" in self._code()

    def test_contains_row_number(self) -> None:
        assert "row_number" in self._code()

    def test_uses_period_self_join(self) -> None:
        code = self._code()
        assert "_periodos_ordenados" in code
        assert "ts_analysis" in code
        assert "ts_relative" in code

    def test_references_source_table(self) -> None:
        assert "db.facts" in self._code()

    def test_output_table_includes_prefix(self) -> None:
        assert "feat_mob_ref" in self._code()

    def test_code_is_valid_python(self) -> None:
        assert _is_valid_python(self._code())

    def test_no_collect_or_show(self) -> None:
        code = self._code()
        assert ".collect()" not in code
        assert ".show()" not in code


# ---------------------------------------------------------------------------
# build_layer2a
# ---------------------------------------------------------------------------


class TestBuildLayer2a:
    def _code(self) -> str:
        out = _GEN.build_layer2a(_pipeline_full())
        assert isinstance(out, PySparkOutput)
        return out.code

    def test_contains_group_by(self) -> None:
        assert "groupBy" in self._code()

    def test_contains_agg(self) -> None:
        assert ".agg(" in self._code()

    def test_contains_when_for_pivot(self) -> None:
        assert "F.when(" in self._code()

    def test_output_table_includes_prefix(self) -> None:
        assert "feat_layer2a" in self._code()

    def test_code_is_valid_python(self) -> None:
        assert _is_valid_python(self._code())

    def test_no_collect_or_show(self) -> None:
        code = self._code()
        assert ".collect()" not in code
        assert ".show()" not in code

    def test_uncategorized_has_no_when(self) -> None:
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
        assert isinstance(out, PySparkOutput)
        assert "F.when(" not in out.code
        assert "groupBy" in out.code


# ---------------------------------------------------------------------------
# build_layer2b
# ---------------------------------------------------------------------------


class TestBuildLayer2b:
    def _code(self) -> str:
        out = _GEN.build_layer2b(_pipeline_full())
        assert isinstance(out, PySparkOutput)
        return out.code

    def test_contains_join(self) -> None:
        assert ".join(" in self._code()

    def test_contains_group_by_shares(self) -> None:
        assert "groupBy" in self._code()

    def test_output_table_present(self) -> None:
        assert "feat_layer2b" in self._code()

    def test_code_is_valid_python(self) -> None:
        assert _is_valid_python(self._code())

    def test_no_collect_or_show(self) -> None:
        code = self._code()
        assert ".collect()" not in code
        assert ".show()" not in code

    def test_empty_layer2b_returns_comment(self) -> None:
        out = _GEN.build_layer2b(_pipeline_pivot_only())
        assert isinstance(out, PySparkOutput)
        # No join code for distributional columns
        assert "layer2b" not in out.code or "no distributional" in out.code

    def test_all_metrics_valid_python(self) -> None:
        out = _GEN.build_layer2b(_pipeline_all_metrics())
        assert isinstance(out, PySparkOutput)
        assert _is_valid_python(out.code)


# ---------------------------------------------------------------------------
# build_layer3
# ---------------------------------------------------------------------------


class TestBuildLayer3:
    def _code(self) -> str:
        out = _GEN.build_layer3(_pipeline_full())
        assert isinstance(out, PySparkOutput)
        return out.code

    def test_references_mob_ref(self) -> None:
        assert "mob_ref" in self._code()

    def test_references_layer2a(self) -> None:
        assert "layer2a" in self._code()

    def test_contains_group_by(self) -> None:
        assert "groupBy" in self._code()

    def test_contains_case_when(self) -> None:
        assert "F.when(" in self._code()

    def test_output_table_present(self) -> None:
        assert "feat_layer3" in self._code()

    def test_code_is_valid_python(self) -> None:
        assert _is_valid_python(self._code())

    def test_no_collect_or_show(self) -> None:
        code = self._code()
        assert ".collect()" not in code
        assert ".show()" not in code

    def test_layer2b_join_when_distributional(self) -> None:
        assert "layer2b" in self._code()

    def test_no_layer2b_join_when_pivot_only(self) -> None:
        out = _GEN.build_layer3(_pipeline_pivot_only())
        assert isinstance(out, PySparkOutput)
        assert "layer2b" not in out.code


# ---------------------------------------------------------------------------
# build_final_join
# ---------------------------------------------------------------------------


class TestBuildFinalJoin:
    def _code(self) -> str:
        out = _GEN.build_final_join(_pipeline_full())
        assert isinstance(out, PySparkOutput)
        return out.code

    def test_output_table_present(self) -> None:
        assert "feat_features" in self._code()

    def test_references_layer2a(self) -> None:
        assert "layer2a" in self._code()

    def test_references_layer3(self) -> None:
        assert "layer3" in self._code()

    def test_references_layer2b_when_distributional(self) -> None:
        assert "layer2b" in self._code()

    def test_no_layer2b_when_pivot_only(self) -> None:
        out = _GEN.build_final_join(_pipeline_pivot_only())
        assert isinstance(out, PySparkOutput)
        assert "layer2b" not in out.code

    def test_code_is_valid_python(self) -> None:
        assert _is_valid_python(self._code())

    def test_no_collect_or_show(self) -> None:
        code = self._code()
        assert ".collect()" not in code
        assert ".show()" not in code
