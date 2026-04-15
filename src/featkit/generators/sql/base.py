"""AbstractSQLCodeGenerator — dialect-agnostic SQL generation via SQLGlot."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import sqlglot

from featkit.enums import DistributionalMetric, TemporalOperator, TimeWindowDirection
from featkit.generators.base import AbstractCodeGenerator
from featkit.generators.output import SQLOutput

if TYPE_CHECKING:
    from featkit.layer2.distributional import DistributionalColumn
    from featkit.layer3.temporal_feature import TemporalFeature
    from featkit.pipeline import FeatureStorePipeline


class AbstractSQLCodeGenerator(AbstractCodeGenerator):
    """Base for all SQL-emitting code generators.

    Subclasses must declare :attr:`dialect`. All five build steps are
    implemented here by composing SQL strings that are then validated and
    normalised through SQLGlot so that dialect-specific rendering (quoting,
    function names, formatting) is applied automatically.

    The schema name in all intermediate table references is double-quoted to
    guard against dialect-reserved words (e.g. ``out``, ``schema``).
    """

    @property
    @abstractmethod
    def dialect(self) -> str:
        """SQLGlot dialect identifier (e.g. ``"snowflake"``, ``"databricks"``)."""
        ...

    def render(self, expr: sqlglot.Expression) -> str:
        """Render a SQLGlot expression tree to the target dialect SQL string."""
        return expr.sql(dialect=self.dialect)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tbl(self, pipeline: FeatureStorePipeline, suffix: str) -> str:
        """Return a fully-qualified intermediate table name.

        The schema is double-quoted to prevent clashes with reserved words.
        """
        cfg = pipeline.config
        return f'"{cfg.output_schema}".{cfg.output_table_prefix}{suffix}'

    def _transpile(self, sql: str) -> str:
        """Parse *sql* in :attr:`dialect` and re-emit it (formatted)."""
        return sqlglot.transpile(sql, read=self.dialect, write=self.dialect, pretty=True)[0]

    # ------------------------------------------------------------------
    # build_mob_table
    # ------------------------------------------------------------------

    def build_mob_table(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the Month-on-Books cross-join + ROW_NUMBER reference table."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "mob_ref")

        id_sel_e = ", ".join(f"e.{c}" for c in id_cols)
        id_sel_src = ", ".join(id_cols)
        id_partition = ", ".join(f"e.{c}" for c in id_cols)

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"SELECT\n"
            f"  {id_sel_e},\n"
            f"  p.{time_col},\n"
            f"  ROW_NUMBER() OVER (PARTITION BY {id_partition} ORDER BY p.{time_col}) AS mob\n"
            f"FROM (SELECT DISTINCT {id_sel_src} FROM {src}) e\n"
            f"CROSS JOIN (SELECT DISTINCT {time_col} FROM {src}) p"
        )
        return SQLOutput(sql=self._transpile(sql), dialect=self.dialect)

    # ------------------------------------------------------------------
    # build_layer2a
    # ------------------------------------------------------------------

    def build_layer2a(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the Layer 2A pivot aggregation table."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "layer2a")

        select_parts: list[str] = list(id_cols) + [time_col]

        for col in pipeline.layer2a:
            meas = col.source_measurement.name
            agg = col.layer2_aggregator.value
            alias = col.column_name

            conditions = [
                f"{cat_field.name} = '{cat_val}'"
                for cat_field, cat_val in sorted(
                    col.categorical_combination.items(), key=lambda kv: kv[0].name
                )
                if cat_val is not None
            ]

            if conditions:
                predicate = " AND ".join(conditions)
                agg_expr = f"{agg}(CASE WHEN {predicate} THEN {meas} END)"
            else:
                agg_expr = f"{agg}({meas})"

            select_parts.append(f"{agg_expr} AS {alias}")

        group_cols = ", ".join(id_cols + [time_col])
        select_list = ",\n  ".join(select_parts)

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"SELECT\n  {select_list}\n"
            f"FROM {src}\n"
            f"GROUP BY {group_cols}"
        )
        return SQLOutput(sql=self._transpile(sql), dialect=self.dialect)

    # ------------------------------------------------------------------
    # build_layer2b
    # ------------------------------------------------------------------

    def build_layer2b(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the Layer 2B distributional CTEs table.

        Produces one pair of CTEs per (categorical, measurement, aggregator)
        group: a *shares* CTE that computes per-category sums and totals, and
        a *metrics* CTE that derives the requested distributional statistics.
        All metrics CTEs are joined back to a ``base`` (DISTINCT id × ts) CTE
        via ``LEFT JOIN``.
        """
        if not pipeline.layer2b:
            return SQLOutput(sql="", dialect=self.dialect)

        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "layer2b")

        id_list = ", ".join(id_cols)
        b_id_sel = ", ".join(f"b.{c}" for c in id_cols)

        # Group distributional columns by (cat_name, meas_name, agg_name)
        groups: dict[tuple[str, str, str], list[DistributionalColumn]] = defaultdict(list)
        for col in pipeline.layer2b:
            key = (col.categorical.name, col.source_measurement.name, col.layer2_aggregator.value)
            groups[key].append(col)

        cte_defs: list[str] = []
        metrics_cte_names: list[str] = []

        for (cat_name, meas_name, agg_name), cols in groups.items():
            safe = f"{cat_name}_{meas_name}_{agg_name.lower()}"
            shares_cte = safe + "_shares"
            metrics_cte = safe + "_metrics"
            metrics_cte_names.append(metrics_cte)

            cte_defs.append(
                f"{shares_cte} AS (\n"
                f"  SELECT\n"
                f"    {id_list},\n"
                f"    {time_col},\n"
                f"    {cat_name},\n"
                f"    {agg_name}({meas_name}) AS cat_val,\n"
                f"    SUM({agg_name}({meas_name})) OVER "
                f"(PARTITION BY {id_list}, {time_col}) AS total_val\n"
                f"  FROM {src}\n"
                f"  GROUP BY {id_list}, {time_col}, {cat_name}\n"
                f")"
            )

            metric_exprs = [
                "    "
                + self._distributional_expr(col.distributional_metric, cat_name, col.column_name)
                for col in cols
            ]

            cte_defs.append(
                f"{metrics_cte} AS (\n"
                f"  SELECT\n"
                f"    {id_list},\n"
                f"    {time_col},\n" + ",\n".join(metric_exprs) + f"\n  FROM {shares_cte}\n"
                f"  GROUP BY {id_list}, {time_col}\n"
                f")"
            )

        # Prepend the base CTE
        base_cte = f"base AS (\n  SELECT DISTINCT {id_list}, {time_col} FROM {src}\n)"
        all_cte_defs = [base_cte] + cte_defs

        all_metric_cols = ",\n  ".join(col.column_name for col in pipeline.layer2b)

        joins = "\n".join(
            f"LEFT JOIN {mc} USING ({id_list}, {time_col})" for mc in metrics_cte_names
        )

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"WITH\n" + ",\n".join(all_cte_defs) + f"\nSELECT\n"
            f"  {b_id_sel},\n"
            f"  b.{time_col},\n"
            f"  {all_metric_cols}\n"
            f"FROM base b\n"
            f"{joins}"
        )
        return SQLOutput(sql=self._transpile(sql), dialect=self.dialect)

    def _distributional_expr(self, metric: DistributionalMetric, cat_col: str, alias: str) -> str:
        """Return a SQL aggregate expression for one distributional metric.

        All expressions operate on columns produced by the corresponding
        *shares* CTE (``cat_val`` = per-category aggregate,
        ``total_val`` = entity×period total).
        """
        if metric == DistributionalMetric.ENTROPY:
            return (
                "-SUM(CASE WHEN cat_val > 0 "
                "THEN (cat_val / NULLIF(total_val, 0)) * LN(cat_val / NULLIF(total_val, 0)) "
                f"ELSE 0 END) AS {alias}"
            )
        if metric == DistributionalMetric.HHI:
            return f"SUM(POWER(cat_val / NULLIF(total_val, 0), 2)) AS {alias}"
        if metric == DistributionalMetric.DOMINANT_PROPORTION:
            return f"MAX(cat_val / NULLIF(total_val, 0)) AS {alias}"
        if metric == DistributionalMetric.MODE:
            return f"MAX_BY({cat_col}, cat_val) AS {alias}"
        if metric == DistributionalMetric.COUNT:
            return f"COUNT(CASE WHEN cat_val > 0 THEN 1 END) AS {alias}"
        raise ValueError(f"Unsupported distributional metric: {metric}")

    # ------------------------------------------------------------------
    # build_layer3
    # ------------------------------------------------------------------

    def build_layer3(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the Layer 3 temporal features table.

        Joins the MOB reference table with Layer 2A (and Layer 2B if present)
        and computes one window-function expression per :class:`TemporalFeature`.
        """
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        mob_tbl = self._tbl(pipeline, "mob_ref")
        l2a_tbl = self._tbl(pipeline, "layer2a")
        l2b_tbl = self._tbl(pipeline, "layer2b")
        tbl = self._tbl(pipeline, "layer3")

        id_m_sel = ", ".join(f"m.{c}" for c in id_cols)
        id_partition = ", ".join(f"m.{c}" for c in id_cols)
        id_using = ", ".join(id_cols)

        select_parts: list[str] = [id_m_sel, f"m.{time_col}", "m.mob"]
        for feat in pipeline.layer3:
            expr = self._temporal_expr(feat, id_partition)
            select_parts.append(f"{expr} AS {feat.column_name}")

        select_list = ",\n  ".join(select_parts)

        l2b_join = (
            f"\nLEFT JOIN {l2b_tbl} l2b USING ({id_using}, {time_col})" if pipeline.layer2b else ""
        )

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"SELECT\n  {select_list}\n"
            f"FROM {mob_tbl} m\n"
            f"LEFT JOIN {l2a_tbl} l2a USING ({id_using}, {time_col})"
            f"{l2b_join}"
        )
        return SQLOutput(sql=self._transpile(sql), dialect=self.dialect)

    def _temporal_expr(self, feat: TemporalFeature, id_partition: str) -> str:
        """Return a SQL window-function expression for one :class:`TemporalFeature`."""
        from featkit.layer2.distributional import DistributionalColumn

        # Qualify column with the correct Layer 2 alias
        src_col = feat.source.column_name
        col = (
            f"l2b.{src_col}" if isinstance(feat.source, DistributionalColumn) else f"l2a.{src_col}"
        )

        op = feat.operator
        w = feat.window_size
        bwd = feat.direction == TimeWindowDirection.BACKWARD

        over_pit = f"OVER (PARTITION BY {id_partition} ORDER BY m.mob)"

        if w is not None:
            prec = w - 1
            frame = (
                f"ROWS BETWEEN {prec} PRECEDING AND CURRENT ROW"
                if bwd
                else f"ROWS BETWEEN CURRENT ROW AND {prec} FOLLOWING"
            )
            over_w = f"OVER (PARTITION BY {id_partition} ORDER BY m.mob {frame})"

        if op == TemporalOperator.PROM_U:
            return f"AVG({col}) {over_w}"
        if op == TemporalOperator.PROM_P:
            return f"AVG({col}) {over_w}"
        if op == TemporalOperator.SUM_U:
            return f"SUM({col}) {over_w}"
        if op == TemporalOperator.SUM_P:
            return f"SUM({col}) {over_w}"
        if op == TemporalOperator.MIN_U:
            return f"MIN({col}) {over_w}"
        if op == TemporalOperator.MAX_U:
            return f"MAX({col}) {over_w}"
        if op == TemporalOperator.ULT_MES:
            return col
        if op == TemporalOperator.PREV_MES:
            return f"LAG({col}, 1) {over_pit}"
        if op == TemporalOperator.CREC:
            lag = f"LAG({col}, 1) {over_pit}"
            return f"({col} - {lag}) / NULLIF({lag}, 0)"
        if op == TemporalOperator.FREQ:
            return f"SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) {over_w}"
        if op == TemporalOperator.XM:
            return f"SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) {over_w}"
        if op == TemporalOperator.REC:
            return f"m.mob - MAX(CASE WHEN {col} IS NOT NULL THEN m.mob ELSE NULL END) {over_pit}"
        if op == TemporalOperator.MEDIA_ABS:
            return f"MEDIAN({col}) {over_w}"
        if op == TemporalOperator.RATIO:
            return f"SUM({col}) {over_w}"
        return col

    # ------------------------------------------------------------------
    # build_final_join
    # ------------------------------------------------------------------

    def build_final_join(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the final feature table joining Layer 2 and Layer 3."""
        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        l2a_tbl = self._tbl(pipeline, "layer2a")
        l2b_tbl = self._tbl(pipeline, "layer2b")
        l3_tbl = self._tbl(pipeline, "layer3")
        tbl = self._tbl(pipeline, "features")

        id_using = ", ".join(id_cols)

        # Explicit column list — avoids ambiguity from USING join semantics
        select_parts: list[str] = (
            [f"l2a.{c}" for c in id_cols]
            + [f"l2a.{time_col}"]
            + [f"l2a.{col.column_name}" for col in pipeline.layer2a]
        )
        if pipeline.layer2b:
            select_parts += [f"l2b.{col.column_name}" for col in pipeline.layer2b]
        if pipeline.layer3:
            select_parts += [f"l3.{feat.column_name}" for feat in pipeline.layer3]

        select_list = ",\n  ".join(select_parts)

        l2b_join = (
            f"\nLEFT JOIN {l2b_tbl} l2b USING ({id_using}, {time_col})" if pipeline.layer2b else ""
        )
        l3_join = (
            f"\nLEFT JOIN {l3_tbl} l3 USING ({id_using}, {time_col})" if pipeline.layer3 else ""
        )

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"SELECT\n  {select_list}\n"
            f"FROM {l2a_tbl} l2a"
            f"{l2b_join}"
            f"{l3_join}"
        )
        return SQLOutput(sql=self._transpile(sql), dialect=self.dialect)
