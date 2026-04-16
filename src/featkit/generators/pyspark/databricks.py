"""PySparkCodeGenerator — PySpark DataFrame code generator for Databricks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from featkit.enums import DistributionalMetric, TemporalOperator, TimeWindowDirection
from featkit.generators.base import AbstractCodeGenerator
from featkit.generators.output import CodeOutput, PySparkOutput

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
        """Generate the MOB reference DataFrame via crossJoin + row_number."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "mob_ref")

        id_select = ", ".join(f'"{c}"' for c in id_cols)
        id_partition = ", ".join(f'"{c}"' for c in id_cols)

        code = (
            f"# --- MOB reference table ---\n"
            f"facts = spark.read.table({src!r})\n"
            f"entities = facts.select({id_select}).distinct()\n"
            f'periods = facts.select("{time_col}").distinct()\n'
            f"mob_cross = entities.crossJoin(periods)\n"
            f"_mob_window = Window.partitionBy({id_partition}).orderBy({time_col!r})\n"
            f'mob_ref = mob_cross.withColumn("mob", F.row_number().over(_mob_window))\n'
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

        code = (
            f"# --- Layer 2A: pivot aggregations ---\n"
            f"_l2a_facts = spark.read.table({src!r})\n"
            f"layer2a = _l2a_facts.groupBy({group_cols}).agg(\n"
            f"{agg_list},\n"
            f")\n"
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

        # Group by (cat, meas, agg) to share intermediate DataFrames
        from collections import defaultdict

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
                f'F.sum(F.when({cv} > 0, -{p} * F.log({p}))).otherwise(F.lit(0.0)).alias("{alias}")'
            )
        if metric == DistributionalMetric.HHI:
            return f'F.sum(F.pow({share}, F.lit(2))).alias("{alias}")'
        if metric == DistributionalMetric.DOMINANT_PROPORTION:
            return f'F.max({share}).alias("{alias}")'
        if metric == DistributionalMetric.MODE:
            return f'F.first(F.col("{cat_col}"), ignorenulls=True).alias("{alias}")'
        if metric == DistributionalMetric.COUNT:
            return f'F.count(F.when({cv} > 0, F.lit(1))).alias("{alias}")'
        raise ValueError(f"Unsupported distributional metric: {metric}")

    # ------------------------------------------------------------------
    # build_layer3
    # ------------------------------------------------------------------

    def build_layer3(self, pipeline: FeatureStorePipeline) -> CodeOutput:
        """Generate the Layer 3 temporal features DataFrame."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        tbl = self._tbl(pipeline, "layer3")
        mob_tbl = self._tbl(pipeline, "mob_ref")
        l2a_tbl = self._tbl(pipeline, "layer2a")
        l2b_tbl = self._tbl(pipeline, "layer2b")

        id_cols_str = ", ".join(f'"{c}"' for c in id_cols)
        id_time_join = ", ".join(f'"{c}"' for c in id_cols + [ds.time_field.name])

        feat_exprs: list[str] = []
        for feat in pipeline.layer3:
            expr = self._temporal_pyspark_expr(feat, id_cols)
            feat_exprs.append(f"    {expr}")

        feat_list = ",\n".join(feat_exprs)

        has_l2b = bool(pipeline.layer2b)
        l2b_join = (
            f'\nl3_df = l3_df.join(spark.read.table({l2b_tbl!r}), on=[{id_time_join}], how="left")'
            if has_l2b
            else ""
        )

        code = (
            f"# --- Layer 3: temporal features ---\n"
            f"_mob = spark.read.table({mob_tbl!r})\n"
            f"_l2a = spark.read.table({l2a_tbl!r})\n"
            f'l3_df = _mob.join(_l2a, on=[{id_time_join}], how="left")'
            f"{l2b_join}\n"
            f"layer3 = l3_df.select(\n"
            f"    *[{id_cols_str}],\n"
            f'    "{ds.time_field.name}",\n'
            f'    "mob",\n'
            f"{feat_list},\n"
            f")\n"
            f"layer3.write.mode('overwrite').saveAsTable({tbl!r})\n"
        )
        return PySparkOutput(code=code)

    def _temporal_pyspark_expr(self, feat: TemporalFeature, id_cols: list[str]) -> str:
        """Return a PySpark column expression string for one temporal feature."""
        from featkit.layer2.distributional import DistributionalColumn

        src_col = feat.source.column_name
        col_ref = (
            f'F.col("{src_col}")'
            if not isinstance(feat.source, DistributionalColumn)
            else f'F.col("{src_col}")'
        )
        alias = feat.column_name
        op = feat.operator
        w = feat.window_size
        bwd = feat.direction == TimeWindowDirection.BACKWARD

        id_partition = ", ".join(f'"{c}"' for c in id_cols)
        base_window = f'Window.partitionBy({id_partition}).orderBy("mob")'

        if w is not None:
            prec = w - 1
            if bwd:
                frame_w = f"{base_window}.rowsBetween(-{prec}, 0)"
            else:
                frame_w = f"{base_window}.rowsBetween(0, {prec})"

        if op == TemporalOperator.PROM_U:
            return f'F.avg({col_ref}).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.PROM_P:
            return f'F.avg({col_ref}).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.SUM_U:
            return f'F.sum({col_ref}).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.SUM_P:
            return f'F.sum({col_ref}).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.MIN_U:
            return f'F.min({col_ref}).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.MAX_U:
            return f'F.max({col_ref}).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.ULT_MES:
            return f'{col_ref}.alias("{alias}")'
        if op == TemporalOperator.PREV_MES:
            return f'F.lag({col_ref}, 1).over({base_window}).alias("{alias}")'
        if op == TemporalOperator.CREC:
            prev = f"F.lag({col_ref}, 1).over({base_window})"
            return f'(({col_ref} - {prev}) / F.when({prev} != 0, {prev})).alias("{alias}")'
        if op == TemporalOperator.FREQ:
            return (
                f"F.sum(F.when({col_ref}.isNotNull(), F.lit(1)).otherwise(F.lit(0)))"
                f'.over({frame_w}).alias("{alias}")'
            )
        if op == TemporalOperator.XM:
            return (
                f"F.sum(F.when({col_ref}.isNotNull(), F.lit(1)).otherwise(F.lit(0)))"
                f'.over({frame_w}).alias("{alias}")'
            )
        if op == TemporalOperator.REC:
            return (
                f'(F.col("mob") - F.max(F.when({col_ref}.isNotNull(), F.col("mob")))'
                f'.over({base_window})).alias("{alias}")'
            )
        if op == TemporalOperator.MEDIA_ABS:
            return f'F.percentile_approx({col_ref}, 0.5).over({frame_w}).alias("{alias}")'
        if op == TemporalOperator.RATIO:
            return f'F.sum({col_ref}).over({frame_w}).alias("{alias}")'
        return f'{col_ref}.alias("{alias}")'

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

    def generate(self, pipeline: FeatureStorePipeline) -> object:
        """Orchestrate all build steps and prepend the PySpark import header."""
        from featkit.generators.output import FeatureStoreOutput

        result = super().generate(pipeline)
        assert isinstance(result, FeatureStoreOutput)
        from featkit.generators.output import PySparkOutput as _PS

        assert isinstance(result.code, _PS)
        full_code = _HEADER + "\n" + result.code.code
        return FeatureStoreOutput(
            code=_PS(code=full_code),
            dag=result.dag,
            mermaid=result.mermaid,
        )
