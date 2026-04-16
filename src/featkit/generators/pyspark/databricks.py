"""PySparkCodeGenerator — PySpark DataFrame code generator for Databricks."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from featkit.enums import DistributionalMetric, TemporalOperator, TimeWindowDirection
from featkit.generators.base import AbstractCodeGenerator
from featkit.generators.output import CodeOutput, FeatureStoreOutput, PySparkOutput

if TYPE_CHECKING:
    from featkit.layer2.distributional import DistributionalColumn
    from featkit.layer3.temporal_feature import TemporalFeature
    from featkit.pipeline import FeatureStorePipeline

# ---------------------------------------------------------------------------
# Code-generation header — injected at the top of each generated snippet
# ---------------------------------------------------------------------------

_HEADER = """\
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
"""


class PySparkCodeGenerator(AbstractCodeGenerator):
    """Code generator that emits a PySpark Python script.

    Each ``build_*`` method returns a :class:`~featkit.generators.output.PySparkOutput`
    whose ``code`` attribute contains one self-contained Python snippet.  The
    snippets are concatenated by the inherited ``generate()`` orchestrator into
    a single script that, when executed against a live ``SparkSession``, builds
    the full feature table using lazy DataFrame transformations.

    MOB reference table structure (period self-join, mirrors the SQL generator):

    * ``_periodos_ordenados`` — distinct periods with a sequential ``mob`` number
      assigned via ``Window.orderBy(time_col)``.
    * ``crossJoin`` of the ordered periods with itself produces all
      ``(ts_analysis, ts_relative)`` pairs.
    * ``mob = mob_b - mob_a`` gives the signed offset (0 = current, negative = past).

    Layer 3 features are derived by joining the MOB table with Layer 2A/B on
    ``ts_relative == ts``, then using ``groupBy(ids + [ts_analysis])`` with
    ``F.when(F.col("mob").between(lo, hi), col)`` aggregates — no window
    functions required.

    All generated transformations are lazy — no ``.collect()`` or ``.show()``
    calls are emitted.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tbl(pipeline: FeatureStorePipeline, suffix: str) -> str:
        """Return a fully-qualified intermediate table name string."""
        cfg = pipeline.config
        return f"{cfg.output_schema}.{cfg.output_table_prefix}{suffix}"

    @staticmethod
    def _spark_read(table: str) -> str:
        """Return a PySpark expression to read *table* as a DataFrame."""
        return f'spark.read.table("{table}")'

    # ------------------------------------------------------------------
    # build_mob_table
    # ------------------------------------------------------------------

    def build_mob_table(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the MOB reference DataFrame via period self-join.

        Assigns a sequential ``mob`` number to each distinct period using
        ``Window.orderBy(time_col)``, then cross-joins the ordered periods
        with themselves to produce all ``(ts_analysis, ts_relative)`` pairs.
        ``mob = mob_b - mob_a`` (0 = current period, negative = past).
        """
        ds = pipeline.config.dataset
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "mob_ref")

        time_analysis = f"{time_col}_analysis"
        time_relative = f"{time_col}_relative"

        code = (
            f"# --- MOB reference table ---\n"
            f"_facts = spark.read.table({src!r})\n"
            f'_periods = _facts.select("{time_col}").distinct()\n'
            f'_mob_win = Window.orderBy("{time_col}")\n'
            f'_periodos_ordenados = _periods.withColumn("mob", F.row_number().over(_mob_win))\n'
            f"_a = _periodos_ordenados.select(\n"
            f'    F.col("{time_col}").alias("{time_analysis}"),\n'
            f'    F.col("mob").alias("mob_a"),\n'
            f")\n"
            f"_b = _periodos_ordenados.select(\n"
            f'    F.col("{time_col}").alias("{time_relative}"),\n'
            f'    F.col("mob").alias("mob_b"),\n'
            f")\n"
            f"mob_ref = _a.crossJoin(_b).withColumn(\n"
            f'    "mob", F.col("mob_b") - F.col("mob_a")\n'
            f').drop("mob_a", "mob_b")\n'
            f"mob_ref.write.mode('overwrite').saveAsTable({tbl!r})\n"
        )
        return PySparkOutput(code=code)

    # ------------------------------------------------------------------
    # build_layer2a
    # ------------------------------------------------------------------

    def build_layer2a(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the Layer 2A pivot aggregation DataFrame."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "layer2a")

        group_cols = ", ".join(f'"{c}"' for c in id_cols + [time_col])

        agg_exprs: list[str] = []
        for col in pipeline.layer2a:
            meas = col.source_measurement.name
            agg = col.layer2_aggregator.value.lower()  # sum, max, min, avg, count
            alias = col.column_name

            conditions = [
                f'(F.col("{cat_field.name}") == "{cat_val}")'
                for cat_field, cat_val in sorted(
                    col.categorical_combination.items(), key=lambda kv: kv[0].name
                )
                if cat_val is not None
            ]

            if conditions:
                predicate = " & ".join(conditions)
                inner = f'F.when({predicate}, F.col("{meas}"))'
                agg_expr = f'F.{agg}({inner}).alias("{alias}")'
            else:
                agg_expr = f'F.{agg}(F.col("{meas}")).alias("{alias}")'

            agg_exprs.append(f"    {agg_expr}")

        agg_list = ",\n".join(agg_exprs)

        if agg_exprs:
            code = (
                f"# --- Layer 2A: pivot aggregations ---\n"
                f"_l2a_facts = spark.read.table({src!r})\n"
                f"layer2a = _l2a_facts.groupBy({group_cols}).agg(\n"
                f"{agg_list},\n"
                f")\n"
                f"layer2a.write.mode('overwrite').saveAsTable({tbl!r})\n"
            )
        else:
            code = (
                f"# --- Layer 2A: no pivot aggregations; preserve grouping grain ---\n"
                f"_l2a_facts = spark.read.table({src!r})\n"
                f"layer2a = _l2a_facts.select({group_cols}).distinct()\n"
                f"layer2a.write.mode('overwrite').saveAsTable({tbl!r})\n"
            )
        return PySparkOutput(code=code)

    # ------------------------------------------------------------------
    # build_layer2b
    # ------------------------------------------------------------------

    def build_layer2b(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the Layer 2B distributional metrics DataFrame.

        Builds one sub-DataFrame per (categorical, measurement, aggregator)
        group, computes the requested distributional metrics via PySpark
        aggregate functions, then joins all sub-DataFrames together.
        """
        if not pipeline.layer2b:
            return PySparkOutput(code="# --- Layer 2B: no distributional columns ---\n")

        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "layer2b")

        group_cols = ", ".join(f'"{c}"' for c in id_cols + [time_col])
        id_time_cols = ", ".join(f'"{c}"' for c in id_cols + [time_col])

        groups: dict[tuple[str, str, str], list[DistributionalColumn]] = defaultdict(list)
        for col in pipeline.layer2b:
            key = (col.categorical.name, col.source_measurement.name, col.layer2_aggregator.value)
            groups[key].append(col)

        snippets: list[str] = [
            f"# --- Layer 2B: distributional metrics ---\n"
            f"_l2b_facts = spark.read.table({src!r})\n"
            f"_base = _l2b_facts.select({id_time_cols}).distinct()\n"
        ]

        df_names: list[str] = []
        for i, ((cat_name, meas_name, agg_name), cols) in enumerate(groups.items()):
            agg_fn = agg_name.lower()
            df_var = f"_dist_{i}"
            df_names.append(df_var)

            cat_group = ", ".join(f'"{c}"' for c in id_cols + [time_col, cat_name])
            agg_col = f'"{cat_name}_{meas_name}_{agg_fn}_cat_val"'
            total_col = f'"{cat_name}_{meas_name}_{agg_fn}_total_val"'

            metric_agg_exprs: list[str] = []
            for col in cols:
                expr = self._distributional_agg_expr(
                    col.distributional_metric, cat_name, agg_col, total_col, col.column_name
                )
                metric_agg_exprs.append(f"    {expr}")

            metric_list = ",\n".join(metric_agg_exprs)
            shares_window = f"Window.partitionBy({group_cols})"

            snippets.append(
                f"_shares_{i} = (\n"
                f"    _l2b_facts\n"
                f"    .groupBy({cat_group})\n"
                f"    .agg(F.{agg_fn}({meas_name!r}).alias({agg_col}))\n"
                f"    .withColumn(\n"
                f"        {total_col},\n"
                f"        F.sum(F.col({agg_col})).over({shares_window}),\n"
                f"    )\n"
                f")\n"
                f"{df_var} = _shares_{i}.groupBy({group_cols}).agg(\n"
                f"{metric_list},\n"
                f")\n"
            )

        # Join all metric DataFrames onto base
        join_id_cols = ", ".join(f'"{c}"' for c in id_cols + [time_col])
        join_lines = "_layer2b = _base\n"
        for df_var in df_names:
            join_lines += f'_layer2b = _layer2b.join({df_var}, on=[{join_id_cols}], how="left")\n'

        snippets.append(join_lines)
        snippets.append(
            f"layer2b = _layer2b\nlayer2b.write.mode('overwrite').saveAsTable({tbl!r})\n"
        )

        return PySparkOutput(code="\n".join(snippets))

    def _distributional_agg_expr(
        self,
        metric: DistributionalMetric,
        cat_col: str,
        cat_val_col: str,
        total_val_col: str,
        alias: str,
    ) -> str:
        """Return a PySpark agg() expression string for one distributional metric."""
        cv = f"F.col({cat_val_col})"
        tv = f"F.col({total_val_col})"
        share = f"F.when({tv} != 0, {cv} / {tv}).otherwise(F.lit(0.0))"

        if metric == DistributionalMetric.ENTROPY:
            p = share
            return (
                f'F.sum(F.when({cv} > 0, -{p} * F.log({p})).otherwise(F.lit(0.0))).alias("{alias}")'
            )
        if metric == DistributionalMetric.HHI:
            return f'F.sum(F.pow({share}, F.lit(2))).alias("{alias}")'
        if metric == DistributionalMetric.DOMINANT_PROPORTION:
            return f'F.max({share}).alias("{alias}")'
        if metric == DistributionalMetric.MODE:
            return f'F.max_by(F.col("{cat_col}"), {cv}).alias("{alias}")'
        if metric == DistributionalMetric.COUNT:
            return f'F.count(F.when({cv} > 0, F.lit(1))).alias("{alias}")'
        raise ValueError(f"Unsupported distributional metric: {metric}")

    # ------------------------------------------------------------------
    # build_layer3
    # ------------------------------------------------------------------

    def build_layer3(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the Layer 3 temporal features DataFrame.

        Joins the MOB reference table (``ts_analysis``, ``ts_relative``, ``mob``)
        with Layer 2A on ``mob.ts_relative == l2a.ts``, then uses
        ``groupBy(ids + [ts_analysis])`` with ``F.when(mob.between(lo, hi), col)``
        aggregates to implement each temporal operator — no window functions.
        """
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        tbl = self._tbl(pipeline, "layer3")
        mob_tbl = self._tbl(pipeline, "mob_ref")
        l2a_tbl = self._tbl(pipeline, "layer2a")
        l2b_tbl = self._tbl(pipeline, "layer2b")

        time_analysis = f"{time_col}_analysis"
        time_relative = f"{time_col}_relative"

        id_time_group = ", ".join(f'"{c}"' for c in id_cols + [time_analysis])
        id_time_join = ", ".join(f'"{c}"' for c in id_cols + [time_col])

        feat_exprs: list[str] = []
        for feat in pipeline.layer3:
            expr = self._temporal_pyspark_expr(feat)
            feat_exprs.append(f"    {expr}")

        feat_list = ",\n".join(feat_exprs)

        has_l2b = bool(pipeline.layer2b)
        l2b_join = (
            f'\nl3_df = l3_df.join(spark.read.table({l2b_tbl!r}), on=[{id_time_join}], how="left")'
            if has_l2b
            else ""
        )

        if feat_exprs:
            agg_block = (
                f"layer3 = l3_df.groupBy({id_time_group}).agg(\n"
                f"{feat_list},\n"
                f').withColumnRenamed("{time_analysis}", "{time_col}")\n'
            )
        else:
            agg_block = (
                f"layer3 = l3_df.select({id_time_group}).distinct()"
                f'.withColumnRenamed("{time_analysis}", "{time_col}")\n'
            )

        code = (
            f"# --- Layer 3: temporal features ---\n"
            f"_mob = spark.read.table({mob_tbl!r})\n"
            f"_l2a = spark.read.table({l2a_tbl!r})\n"
            f'l3_df = _mob.join(_l2a, _mob["{time_relative}"] == _l2a["{time_col}"], "inner")'
            f"{l2b_join}\n" + agg_block + f"layer3.write.mode('overwrite').saveAsTable({tbl!r})\n"
        )
        return PySparkOutput(code=code)

    def _temporal_pyspark_expr(self, feat: TemporalFeature) -> str:
        """Return a PySpark groupBy-compatible agg expression for one temporal feature.

        Produces ``F.agg(F.when(F.col("mob").between(lo, hi), col))`` style
        expressions so that they can be passed directly to ``.agg()`` after
        a ``groupBy(ids + [ts_analysis])`` call.
        """
        src_col = feat.source.column_name
        col_ref = f'F.col("{src_col}")'
        alias = feat.column_name
        op = feat.operator
        w = feat.window_size
        bwd = feat.direction == TimeWindowDirection.BACKWARD
        mob = 'F.col("mob")'

        if w is not None:
            lo, hi = (-(w - 1), 0) if bwd else (0, w - 1)
            in_window = f"{mob}.between({lo}, {hi})"
            case_col = f"F.when({in_window}, {col_ref})"

        if op == TemporalOperator.PROM_U:
            return f'F.avg({case_col}).alias("{alias}")'
        if op == TemporalOperator.PROM_P:
            return f'F.avg({case_col}).alias("{alias}")'
        if op == TemporalOperator.SUM_U:
            return f'F.sum({case_col}).alias("{alias}")'
        if op == TemporalOperator.SUM_P:
            return f'F.sum({case_col}).alias("{alias}")'
        if op == TemporalOperator.MIN_U:
            return f'F.min({case_col}).alias("{alias}")'
        if op == TemporalOperator.MAX_U:
            return f'F.max({case_col}).alias("{alias}")'
        if op == TemporalOperator.ULT_MES:
            return f'F.max(F.when({mob} == 0, {col_ref})).alias("{alias}")'
        if op == TemporalOperator.PREV_MES:
            return f'F.max(F.when({mob} == -1, {col_ref})).alias("{alias}")'
        if op == TemporalOperator.CREC:
            curr = f"F.max(F.when({mob} == 0, {col_ref}))"
            prev = f"F.max(F.when({mob} == -1, {col_ref}))"
            return (
                f"({curr} / F.when({prev} != 0, {prev}).otherwise(F.lit(None))"
                f' - F.lit(1)).alias("{alias}")'
            )
        if op == TemporalOperator.FREQ:
            return (
                f'F.count(F.when({in_window} & {col_ref}.isNotNull(), F.lit(1))).alias("{alias}")'
            )
        if op == TemporalOperator.XM:
            return (
                f'F.count(F.when({in_window} & {col_ref}.isNotNull(), F.lit(1))).alias("{alias}")'
            )
        if op == TemporalOperator.REC:
            return f'(-F.max(F.when({col_ref}.isNotNull(), {mob}))).alias("{alias}")'
        if op == TemporalOperator.MEDIA_ABS:
            return f'F.percentile_approx({case_col}, F.lit(0.5)).alias("{alias}")'
        if op == TemporalOperator.RATIO:
            return f'F.sum({case_col}).alias("{alias}")'
        return f'F.max(F.when({mob} == 0, {col_ref})).alias("{alias}")'

    # ------------------------------------------------------------------
    # build_final_join
    # ------------------------------------------------------------------

    def build_final_join(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the final feature table by joining Layer 2 and Layer 3."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        tbl = self._tbl(pipeline, "features")
        l2a_tbl = self._tbl(pipeline, "layer2a")
        l2b_tbl = self._tbl(pipeline, "layer2b")
        l3_tbl = self._tbl(pipeline, "layer3")

        id_time_join = ", ".join(f'"{c}"' for c in id_cols + [ds.time_field.name])

        l2b_join = (
            f"\nfinal_df = final_df.join("
            f'spark.read.table({l2b_tbl!r}), on=[{id_time_join}], how="left")'
            if pipeline.layer2b
            else ""
        )
        l3_join = (
            f"\nfinal_df = final_df.join("
            f'spark.read.table({l3_tbl!r}), on=[{id_time_join}], how="left")'
            if pipeline.layer3
            else ""
        )

        code = (
            f"# --- Final join ---\n"
            f"final_df = spark.read.table({l2a_tbl!r})"
            f"{l2b_join}"
            f"{l3_join}\n"
            f"final_df.write.mode('overwrite').saveAsTable({tbl!r})\n"
        )
        return PySparkOutput(code=code)

    # ------------------------------------------------------------------
    # Override generate() to emit the header once
    # ------------------------------------------------------------------

    def generate(self, pipeline: FeatureStorePipeline) -> FeatureStoreOutput:
        """Orchestrate all build steps and prepend the PySpark import header."""
        result = super().generate(pipeline)
        assert isinstance(result, FeatureStoreOutput)
        assert isinstance(result.code, PySparkOutput)
        full_code = _HEADER + "\n" + result.code.code
        return FeatureStoreOutput(
            code=PySparkOutput(code=full_code),
            dag=result.dag,
            mermaid=result.mermaid,
        )
