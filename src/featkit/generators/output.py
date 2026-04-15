"""Output containers and DAG model for generated feature-store code."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DAGNode:
    """A single node in the feature-store execution DAG.

    Args:
        step_name: Unique identifier for this pipeline step.
        depends_on: Names of steps that must complete before this one.
    """

    step_name: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class DAG:
    """Directed acyclic graph representing pipeline step dependencies.

    Args:
        nodes: All nodes in topological order (sources first).
    """

    nodes: list[DAGNode]

    def to_json(self) -> str:
        """Serialise the DAG to a JSON string."""
        return json.dumps(
            [{"step_name": n.step_name, "depends_on": n.depends_on} for n in self.nodes],
            indent=2,
        )


@dataclass
class SQLOutput:
    """Container for generated SQL code.

    Args:
        sql: The complete SQL script.
        dialect: The SQL dialect (e.g. ``"snowflake"``, ``"databricks"``).
    """

    sql: str
    dialect: str

    def save(self, path: str) -> None:
        """Write the SQL string to *path*, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.sql, encoding="utf-8")


@dataclass
class PySparkOutput:
    """Placeholder container for generated PySpark code.

    The ``code`` attribute holds the PySpark script as a plain string.
    Plan 16 will replace the string payload with a lazy DataFrame chain.

    Args:
        code: PySpark script string (may be empty for stubs).
    """

    code: str = ""

    def save(self, path: str) -> None:
        """Write the PySpark code string to *path*, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.code, encoding="utf-8")


#: Union type for any code output produced by a generator.
CodeOutput = SQLOutput | PySparkOutput


@dataclass
class FeatureStoreOutput:
    """Complete output of a code-generation run.

    Args:
        code: The generated script (SQL or PySpark).
        dag: Execution DAG for the pipeline steps.
        mermaid: Mermaid flowchart string for the DAG.
    """

    code: CodeOutput
    dag: DAG
    mermaid: str

    def save(self, directory: str) -> None:
        """Persist all artefacts under *directory*.

        Files written:

        * ``script.sql`` — when ``code`` is :class:`SQLOutput`
        * ``script.py``  — when ``code`` is :class:`PySparkOutput`
        * ``dag.json``   — JSON serialisation of the DAG
        * ``diagram.md`` — Mermaid diagram wrapped in a fenced code block
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        if isinstance(self.code, SQLOutput):
            self.code.save(str(d / "script.sql"))
        else:
            self.code.save(str(d / "script.py"))

        (d / "dag.json").write_text(self.dag.to_json(), encoding="utf-8")
        (d / "diagram.md").write_text(f"```mermaid\n{self.mermaid}\n```\n", encoding="utf-8")
