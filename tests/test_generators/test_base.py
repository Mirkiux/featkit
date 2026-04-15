"""Tests for Plan 13 — AbstractCodeGenerator, DAG, and FeatureStoreOutput."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

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
from featkit.generators.base import (
    _FACTS_TABLE,
    _FINAL_OUTPUT,
    _LAYER2_JOIN,
    _LAYER2A_PIVOT,
    _LAYER2B_DIST,
    _LAYER3_TEMPORAL,
    _MOB_TABLE,
    AbstractCodeGenerator,
)
from featkit.generators.output import (
    DAG,
    CodeOutput,
    DAGNode,
    FeatureStoreOutput,
    PySparkOutput,
    SQLOutput,
)
from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DIALECT = "test_dialect"


def _pipeline() -> FeatureStorePipeline:
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


class _StubSQLGenerator(AbstractCodeGenerator):
    """Minimal concrete generator that returns fixed SQLOutput from every step."""

    def build_mob_table(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        return SQLOutput(sql="-- mob", dialect=_DIALECT)

    def build_layer2a(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        return SQLOutput(sql="-- layer2a", dialect=_DIALECT)

    def build_layer2b(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        return SQLOutput(sql="-- layer2b", dialect=_DIALECT)

    def build_layer3(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        return SQLOutput(sql="-- layer3", dialect=_DIALECT)

    def build_final_join(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        return SQLOutput(sql="-- final", dialect=_DIALECT)


# ---------------------------------------------------------------------------
# DAGNode
# ---------------------------------------------------------------------------


class TestDAGNode:
    def test_stores_step_name(self) -> None:
        node = DAGNode("mob_table", ["facts_table"])
        assert node.step_name == "mob_table"

    def test_stores_depends_on(self) -> None:
        node = DAGNode("layer2_join", ["layer2a_pivot", "layer2b_distributional_ctes"])
        assert node.depends_on == ["layer2a_pivot", "layer2b_distributional_ctes"]

    def test_default_depends_on_is_empty(self) -> None:
        node = DAGNode("facts_table")
        assert node.depends_on == []


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


class TestDAG:
    def _dag(self) -> DAG:
        return DAG(
            nodes=[
                DAGNode("a", []),
                DAGNode("b", ["a"]),
                DAGNode("c", ["a", "b"]),
            ]
        )

    def test_to_json_is_valid_json(self) -> None:
        data = json.loads(self._dag().to_json())
        assert isinstance(data, list)

    def test_to_json_contains_all_nodes(self) -> None:
        data = json.loads(self._dag().to_json())
        names = [entry["step_name"] for entry in data]
        assert names == ["a", "b", "c"]

    def test_to_json_contains_depends_on(self) -> None:
        data = json.loads(self._dag().to_json())
        c_entry = next(e for e in data if e["step_name"] == "c")
        assert c_entry["depends_on"] == ["a", "b"]

    def test_to_json_root_has_empty_depends_on(self) -> None:
        data = json.loads(self._dag().to_json())
        a_entry = next(e for e in data if e["step_name"] == "a")
        assert a_entry["depends_on"] == []


# ---------------------------------------------------------------------------
# SQLOutput / PySparkOutput
# ---------------------------------------------------------------------------


class TestSQLOutput:
    def test_stores_sql_and_dialect(self) -> None:
        out = SQLOutput(sql="SELECT 1", dialect="snowflake")
        assert out.sql == "SELECT 1"
        assert out.dialect == "snowflake"

    def test_save_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "script.sql")
            SQLOutput(sql="SELECT 1", dialect="snowflake").save(path)
            assert Path(path).read_text() == "SELECT 1"

    def test_save_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sub" / "dir" / "script.sql")
            SQLOutput(sql="SELECT 1", dialect="snowflake").save(path)
            assert Path(path).exists()


class TestPySparkOutput:
    def test_default_code_is_empty_string(self) -> None:
        assert PySparkOutput().code == ""

    def test_save_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "script.py")
            PySparkOutput(code="df.show()").save(path)
            assert Path(path).read_text() == "df.show()"


# ---------------------------------------------------------------------------
# FeatureStoreOutput.save()
# ---------------------------------------------------------------------------


class TestFeatureStoreOutputSave:
    def _dag(self) -> DAG:
        return DAG(nodes=[DAGNode("a", []), DAGNode("b", ["a"])])

    def test_save_sql_writes_script_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = FeatureStoreOutput(
                code=SQLOutput(sql="SELECT 1", dialect="snowflake"),
                dag=self._dag(),
                mermaid="flowchart TD\n    a --> b",
            )
            out.save(tmp)
            assert (Path(tmp) / "script.sql").read_text() == "SELECT 1"

    def test_save_pyspark_writes_script_py(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = FeatureStoreOutput(
                code=PySparkOutput(code="df.show()"),
                dag=self._dag(),
                mermaid="flowchart TD\n    a --> b",
            )
            out.save(tmp)
            assert (Path(tmp) / "script.py").read_text() == "df.show()"

    def test_save_writes_dag_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = FeatureStoreOutput(
                code=SQLOutput(sql="SELECT 1", dialect="snowflake"),
                dag=self._dag(),
                mermaid="flowchart TD",
            )
            out.save(tmp)
            dag_file = Path(tmp) / "dag.json"
            assert dag_file.exists()
            data = json.loads(dag_file.read_text())
            names = [e["step_name"] for e in data]
            assert "a" in names and "b" in names

    def test_save_writes_diagram_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mermaid = "flowchart TD\n    a --> b"
            out = FeatureStoreOutput(
                code=SQLOutput(sql="SELECT 1", dialect="snowflake"),
                dag=self._dag(),
                mermaid=mermaid,
            )
            out.save(tmp)
            content = (Path(tmp) / "diagram.md").read_text()
            assert "```mermaid" in content
            assert mermaid in content

    def test_save_no_script_py_when_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            FeatureStoreOutput(
                code=SQLOutput(sql="SELECT 1", dialect="snowflake"),
                dag=self._dag(),
                mermaid="flowchart TD",
            ).save(tmp)
            assert not (Path(tmp) / "script.py").exists()

    def test_save_no_script_sql_when_pyspark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            FeatureStoreOutput(
                code=PySparkOutput(),
                dag=self._dag(),
                mermaid="flowchart TD",
            ).save(tmp)
            assert not (Path(tmp) / "script.sql").exists()

    def test_save_creates_directory_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "new_dir")
            FeatureStoreOutput(
                code=SQLOutput(sql="SELECT 1", dialect="snowflake"),
                dag=self._dag(),
                mermaid="flowchart TD",
            ).save(target)
            assert Path(target).is_dir()


# ---------------------------------------------------------------------------
# AbstractCodeGenerator.build_dag()
# ---------------------------------------------------------------------------


class TestBuildDAG:
    def test_dag_has_seven_nodes(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        assert len(dag.nodes) == 7

    def test_dag_contains_expected_step_names(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        names = {n.step_name for n in dag.nodes}
        assert names == {
            _FACTS_TABLE,
            _MOB_TABLE,
            _LAYER2A_PIVOT,
            _LAYER2B_DIST,
            _LAYER2_JOIN,
            _LAYER3_TEMPORAL,
            _FINAL_OUTPUT,
        }

    def test_facts_table_has_no_dependencies(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        node = next(n for n in dag.nodes if n.step_name == _FACTS_TABLE)
        assert node.depends_on == []

    def test_mob_table_depends_on_facts(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        node = next(n for n in dag.nodes if n.step_name == _MOB_TABLE)
        assert node.depends_on == [_FACTS_TABLE]

    def test_layer2_join_depends_on_both_layer2s(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        node = next(n for n in dag.nodes if n.step_name == _LAYER2_JOIN)
        assert set(node.depends_on) == {_LAYER2A_PIVOT, _LAYER2B_DIST}

    def test_layer3_depends_on_layer2_join_and_mob(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        node = next(n for n in dag.nodes if n.step_name == _LAYER3_TEMPORAL)
        assert set(node.depends_on) == {_LAYER2_JOIN, _MOB_TABLE}

    def test_final_output_depends_on_layer2_join_and_layer3(self) -> None:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        node = next(n for n in dag.nodes if n.step_name == _FINAL_OUTPUT)
        assert set(node.depends_on) == {_LAYER2_JOIN, _LAYER3_TEMPORAL}


# ---------------------------------------------------------------------------
# AbstractCodeGenerator.build_mermaid()
# ---------------------------------------------------------------------------


class TestBuildMermaid:
    def _mermaid(self) -> str:
        gen = _StubSQLGenerator()
        dag = gen.build_dag(_pipeline())
        return gen.build_mermaid(dag)

    def test_starts_with_flowchart_td(self) -> None:
        assert self._mermaid().startswith("flowchart TD")

    def test_contains_facts_to_mob_edge(self) -> None:
        assert f"{_FACTS_TABLE} --> {_MOB_TABLE}" in self._mermaid()

    def test_contains_facts_to_layer2a_edge(self) -> None:
        assert f"{_FACTS_TABLE} --> {_LAYER2A_PIVOT}" in self._mermaid()

    def test_contains_facts_to_layer2b_edge(self) -> None:
        assert f"{_FACTS_TABLE} --> {_LAYER2B_DIST}" in self._mermaid()

    def test_contains_layer2s_to_join_edges(self) -> None:
        m = self._mermaid()
        assert f"{_LAYER2A_PIVOT} --> {_LAYER2_JOIN}" in m
        assert f"{_LAYER2B_DIST} --> {_LAYER2_JOIN}" in m

    def test_contains_layer3_edges(self) -> None:
        m = self._mermaid()
        assert f"{_LAYER2_JOIN} --> {_LAYER3_TEMPORAL}" in m
        assert f"{_MOB_TABLE} --> {_LAYER3_TEMPORAL}" in m

    def test_contains_final_output_edges(self) -> None:
        m = self._mermaid()
        assert f"{_LAYER2_JOIN} --> {_FINAL_OUTPUT}" in m
        assert f"{_LAYER3_TEMPORAL} --> {_FINAL_OUTPUT}" in m

    def test_isolated_node_declared_explicitly(self) -> None:
        gen = _StubSQLGenerator()
        dag = DAG(nodes=[DAGNode("orphan", [])])
        mermaid = gen.build_mermaid(dag)
        assert "orphan" in mermaid


# ---------------------------------------------------------------------------
# AbstractCodeGenerator.generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_returns_feature_store_output(self) -> None:
        result = _StubSQLGenerator().generate(_pipeline())
        assert isinstance(result, FeatureStoreOutput)

    def test_generate_code_is_sql_output(self) -> None:
        result = _StubSQLGenerator().generate(_pipeline())
        assert isinstance(result.code, SQLOutput)

    def test_generate_combines_all_step_sql(self) -> None:
        result = _StubSQLGenerator().generate(_pipeline())
        assert isinstance(result.code, SQLOutput)
        sql = result.code.sql
        assert "-- mob" in sql
        assert "-- layer2a" in sql
        assert "-- layer2b" in sql
        assert "-- layer3" in sql
        assert "-- final" in sql

    def test_generate_dag_has_correct_structure(self) -> None:
        result = _StubSQLGenerator().generate(_pipeline())
        names = {n.step_name for n in result.dag.nodes}
        assert _FINAL_OUTPUT in names
        assert _FACTS_TABLE in names

    def test_generate_mermaid_is_valid(self) -> None:
        result = _StubSQLGenerator().generate(_pipeline())
        assert result.mermaid.startswith("flowchart TD")

    def test_generate_dialect_preserved(self) -> None:
        result = _StubSQLGenerator().generate(_pipeline())
        assert isinstance(result.code, SQLOutput)
        assert result.code.dialect == _DIALECT


# ---------------------------------------------------------------------------
# AbstractCodeGenerator._combine_code() — dialect mismatch
# ---------------------------------------------------------------------------


class TestCombineCodeDialectMismatch:
    def test_mixed_sql_dialects_raise_value_error(self) -> None:
        class _MixedDialectGenerator(AbstractCodeGenerator):
            def build_mob_table(self, pipeline: FeatureStorePipeline) -> CodeOutput:
                return SQLOutput(sql="-- mob", dialect="snowflake")

            def build_layer2a(self, pipeline: FeatureStorePipeline) -> CodeOutput:
                return SQLOutput(sql="-- layer2a", dialect="bigquery")

            def build_layer2b(self, pipeline: FeatureStorePipeline) -> CodeOutput:
                return SQLOutput(sql="-- layer2b", dialect="snowflake")

            def build_layer3(self, pipeline: FeatureStorePipeline) -> CodeOutput:
                return SQLOutput(sql="-- layer3", dialect="snowflake")

            def build_final_join(self, pipeline: FeatureStorePipeline) -> CodeOutput:
                return SQLOutput(sql="-- final", dialect="snowflake")

        with pytest.raises(ValueError, match="Mixed SQLOutput dialects"):
            _MixedDialectGenerator().generate(_pipeline())
