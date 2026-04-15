"""AbstractCodeGenerator — engine-agnostic base for all feature-store code generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from featkit.generators.output import (
    DAG,
    CodeOutput,
    DAGNode,
    FeatureStoreOutput,
    PySparkOutput,
    SQLOutput,
)

if TYPE_CHECKING:
    from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Fixed DAG node names
# ---------------------------------------------------------------------------

_FACTS_TABLE = "facts_table"
_MOB_TABLE = "mob_table"
_LAYER2A_PIVOT = "layer2a_pivot"
_LAYER2B_DIST = "layer2b_distributional_ctes"
_LAYER2_JOIN = "layer2_join"
_LAYER3_TEMPORAL = "layer3_temporal"
_FINAL_OUTPUT = "final_output"


class AbstractCodeGenerator(ABC):
    """Base class for all feature-store code generators.

    Subclasses implement the five abstract ``build_*`` methods; the concrete
    ``build_dag``, ``build_mermaid``, and ``generate`` methods are shared
    across all engines.

    The DAG structure is fixed for every generator:

    .. code-block:: text

        facts_table → mob_table
        facts_table → layer2a_pivot
        facts_table → layer2b_distributional_ctes
        layer2a_pivot + layer2b_distributional_ctes → layer2_join
        layer2_join + mob_table → layer3_temporal
        layer2_join + layer3_temporal → final_output
    """

    # ------------------------------------------------------------------
    # Abstract build steps
    # ------------------------------------------------------------------

    @abstractmethod
    def build_mob_table(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the Month-on-Books (MOB) reference table step."""
        ...

    @abstractmethod
    def build_layer2a(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the pivot (Layer 2a) aggregation step."""
        ...

    @abstractmethod
    def build_layer2b(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the distributional (Layer 2b) CTE step."""
        ...

    @abstractmethod
    def build_layer3(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the temporal features (Layer 3) step."""
        ...

    @abstractmethod
    def build_final_join(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the final join that assembles the output feature table."""
        ...

    # ------------------------------------------------------------------
    # Concrete DAG / Mermaid helpers
    # ------------------------------------------------------------------

    def build_dag(self, pipeline: FeatureStorePipeline) -> DAG:  # noqa: ARG002
        """Return the fixed execution DAG (pipeline argument reserved for future use)."""
        return DAG(
            nodes=[
                DAGNode(_FACTS_TABLE, []),
                DAGNode(_MOB_TABLE, [_FACTS_TABLE]),
                DAGNode(_LAYER2A_PIVOT, [_FACTS_TABLE]),
                DAGNode(_LAYER2B_DIST, [_FACTS_TABLE]),
                DAGNode(_LAYER2_JOIN, [_LAYER2A_PIVOT, _LAYER2B_DIST]),
                DAGNode(_LAYER3_TEMPORAL, [_LAYER2_JOIN, _MOB_TABLE]),
                DAGNode(_FINAL_OUTPUT, [_LAYER2_JOIN, _LAYER3_TEMPORAL]),
            ]
        )

    def build_mermaid(self, dag: DAG) -> str:
        """Render *dag* as a Mermaid ``flowchart TD`` string."""
        lines = ["flowchart TD"]
        has_edges: set[str] = set()

        for node in dag.nodes:
            for dep in node.depends_on:
                lines.append(f"    {dep} --> {node.step_name}")
                has_edges.add(dep)
                has_edges.add(node.step_name)

        # Isolated nodes (neither source nor target of any edge) need explicit declaration.
        for node in dag.nodes:
            if node.step_name not in has_edges:
                lines.append(f"    {node.step_name}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _combine_code(self, steps: list[CodeOutput]) -> CodeOutput:
        """Combine a sequence of per-step code outputs into one.

        Default implementation: concatenates ``SQLOutput`` payloads.
        Generators that produce :class:`~featkit.generators.output.PySparkOutput`
        should override :meth:`generate` instead.
        """
        sql_steps = [s for s in steps if isinstance(s, SQLOutput) and s.sql]
        pyspark_steps = [s for s in steps if isinstance(s, PySparkOutput) and s.code]

        if sql_steps and pyspark_steps:
            msg = (
                "Mixed non-empty code output types are not supported in "
                "_combine_code(); found both SQLOutput and PySparkOutput."
            )
            raise ValueError(msg)

        if sql_steps:
            dialects = {s.dialect for s in sql_steps}
            if len(dialects) > 1:
                msg = (
                    f"Mixed SQLOutput dialects are not supported in _combine_code(); "
                    f"found dialects: {dialects}."
                )
                raise ValueError(msg)
            return SQLOutput(
                sql="\n\n".join(s.sql for s in sql_steps),
                dialect=sql_steps[0].dialect,
            )
        if pyspark_steps:
            return PySparkOutput(code="\n\n".join(s.code for s in pyspark_steps))
        # All steps were empty SQLOutputs — return an empty one.
        if any(isinstance(s, SQLOutput) for s in steps):
            return SQLOutput(
                sql="", dialect=next(s.dialect for s in steps if isinstance(s, SQLOutput))
            )
        return PySparkOutput()

    def generate(self, pipeline: FeatureStorePipeline) -> FeatureStoreOutput:
        """Orchestrate all build steps and return the complete output artefact."""
        steps: list[CodeOutput] = [
            self.build_mob_table(pipeline),
            self.build_layer2a(pipeline),
            self.build_layer2b(pipeline),
            self.build_layer3(pipeline),
            self.build_final_join(pipeline),
        ]
        code = self._combine_code(steps)
        dag = self.build_dag(pipeline)
        mermaid = self.build_mermaid(dag)
        return FeatureStoreOutput(code=code, dag=dag, mermaid=mermaid)
