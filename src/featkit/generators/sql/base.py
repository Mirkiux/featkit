"""AbstractSQLCodeGenerator — dialect-agnostic SQL generation via SQLGlot."""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import sqlglot
import sqlglot.expressions as exp

from featkit.enums import DistributionalMetric, TemporalOperator, TimeWindowDirection
from featkit.generators.base import AbstractCodeGenerator
from featkit.generators.output import SQLOutput

if TYPE_CHECKING:
    from featkit.layer2.distributional import DistributionalColumn
    from featkit.layer2.pivoted import PivotedColumn
    from featkit.layer3.temporal_feature import TemporalFeature
    from featkit.pipeline import FeatureStorePipeline

_log = logging.getLogger(__name__)


class AbstractSQLCodeGenerator(AbstractCodeGenerator):
    """Base for all SQL-emitting code generators.

    Subclasses must declare :attr:`dialect`. All five build steps are
    implemented here by composing SQL strings that are validated and
    normalised through SQLGlot so that dialect-specific rendering is applied
    automatically.

    Both the schema and table portions of every generated table reference are
    double-quoted, guarding against reserved words and special characters in
    user-supplied names.  Categorical values used in CASE WHEN predicates are
    rendered as SQLGlot :class:`~sqlglot.expressions.Literal` objects so that
    single quotes and other special characters are escaped correctly.

    MOB reference table structure (period self-join):

    .. code-block:: text

        periodos_unicos → periodos_ordenados (ROW_NUMBER by period)
        periodos_ordenados A LEFT JOIN periodos_ordenados B ON 1=1
        → (ts_analysis, ts_relative, mob = B.mob - A.mob)

    Layer 3 features are then derived by joining with the MOB table and using
    GROUP BY + CASE WHEN aggregation to implement each temporal operator.
    """

    @property
    @abstractmethod
    def dialect(self) -> str:
        """SQLGlot dialect identifier (e.g. ``"snowflake"``, ``"databricks"``)."""
        ...

    def render(self, expr: exp.Expr) -> str:
        """Render a SQLGlot expression tree to the target dialect SQL string."""
        return str(expr.sql(dialect=self.dialect))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tbl(self, pipeline: FeatureStorePipeline, suffix: str) -> str:
        """Return a fully-qualified, double-quoted intermediate table reference.

        Both the schema and table identifiers are quoted through SQLGlot so
        reserved words and special characters are rendered safely per dialect.
        """
        cfg = pipeline.config
        return self.render(
            sqlglot.exp.Table(
                this=sqlglot.exp.to_identifier(f"{cfg.output_table_prefix}{suffix}", quoted=True),
                db=sqlglot.exp.to_identifier(cfg.output_schema, quoted=True),
            )
        )

    def _transpile(self, sql: str) -> str:
        """Parse *sql* in :attr:`dialect` and re-emit it (pretty-printed)."""
        return sqlglot.transpile(sql, read=self.dialect, write=self.dialect, pretty=True)[0]

    @staticmethod
    def _safe_cte_name(name: str) -> str:
        """Return a SQL-safe CTE identifier from *name*.

        Replaces any character outside ``[A-Za-z0-9_]`` with ``_`` so that
        field names containing hyphens, spaces, or other special characters
        produce valid unquoted identifiers.
        """
        return re.sub(r"[^A-Za-z0-9]", "_", name)

    def _str_literal(self, value: str) -> str:
        """Render *value* as a correctly escaped SQL string literal."""
        return exp.Literal.string(value).sql(dialect=self.dialect)

    def _quoted_id(self, name: str) -> str:
        """Render *name* as a quoted SQL identifier."""
        return exp.Identifier(this=name, quoted=True).sql(dialect=self.dialect)

    # ------------------------------------------------------------------
    # build_mob_table
    # ------------------------------------------------------------------

    def build_mob_table(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the MOB (Month-on-Books) period cross-reference table.

        The implementation follows the period self-join pattern:

        1. ``periodos_unicos`` — distinct periods from the source table.
        2. ``periodos_ordenados`` — assigns a sequential ``mob`` number to each
           period via ``ROW_NUMBER() OVER (ORDER BY {time_col})``.
        3. Self-join on ``1 = 1`` to produce all (analysis, relative) period
           combinations.
        4. ``mob = B.mob - A.mob`` gives the signed period offset: 0 = current,
           negative = past, positive = future.

        The resulting table has columns:
        ``{time_col}_analysis``, ``{time_col}_relative``, ``mob``.
        """
        verbose = pipeline.config.verbose
        if verbose:
            _log.debug("build_mob_table started")

        ds = pipeline.config.dataset
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "mob_ref")

        time_analysis = f"{time_col}_analysis"
        time_relative = f"{time_col}_relative"

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"WITH periodos_unicos AS (\n"
            f"  SELECT DISTINCT {time_col}\n"
            f"  FROM {src}\n"
            f"),\n"
            f"periodos_ordenados AS (\n"
            f"  SELECT {time_col},\n"
            f"         ROW_NUMBER() OVER (ORDER BY {time_col}) AS mob\n"
            f"  FROM periodos_unicos\n"
            f")\n"
            f"SELECT\n"
            f"  a.{time_col} AS {time_analysis},\n"
            f"  b.{time_col} AS {time_relative},\n"
            f"  b.mob - a.mob AS mob\n"
            f"FROM periodos_ordenados a\n"
            f"LEFT JOIN periodos_ordenados b ON 1 = 1"
        )
        if verbose:
            _log.debug("build_mob_table base SQL:\n%s", sql)
        result = SQLOutput(sql=self._transpile(sql), dialect=self.dialect)
        if verbose:
            _log.debug("build_mob_table done")
        return result

    # ------------------------------------------------------------------
    # build_layer2a
    # ------------------------------------------------------------------

    def _pivoted_agg_expr(self, col: PivotedColumn) -> str:
        """Return the bare aggregate SQL expression for *col* (without alias).

        Used both for regular pivot columns and as numerator/denominator
        sub-expressions when building ratio columns.
        """
        meas = col.source_measurement.name
        agg = col.layer2_aggregator.value
        conditions = [
            f"{self._quoted_id(cat_field.name)} = {self._str_literal(cat_val)}"
            for cat_field, cat_val in sorted(
                col.categorical_combination.items(), key=lambda kv: kv[0].name
            )
            if cat_val is not None
        ]
        if conditions:
            predicate = " AND ".join(conditions)
            return f"{agg}(CASE WHEN {predicate} THEN {meas} END)"
        return f"{agg}({meas})"

    def build_layer2a(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the Layer 2A pivot aggregation table.

        Categorical values in CASE WHEN predicates are rendered via
        :meth:`_str_literal` to escape special characters and prevent
        SQL injection in the generated script.  Column identifiers are
        double-quoted via :meth:`_quoted_id`.

        Ratio columns (``layer2c``) are computed in the same SELECT as the
        base pivot columns: ``numerator_expr / NULLIF(denominator_expr, 0)``.
        """
        verbose = pipeline.config.verbose
        if verbose:
            _log.debug("build_layer2a started")

        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        src = ds.source_reference
        tbl = self._tbl(pipeline, "layer2a")

        select_parts: list[str] = list(id_cols) + [time_col]

        for col in pipeline.layer2a:
            select_parts.append(f"{self._pivoted_agg_expr(col)} AS {col.column_name}")

        for ratio_col in pipeline.layer2c:
            num_expr = self._pivoted_agg_expr(ratio_col.numerator)
            denom_expr = self._pivoted_agg_expr(ratio_col.denominator)
            select_parts.append(f"{num_expr} / NULLIF({denom_expr}, 0) AS {ratio_col.column_name}")

        group_cols = ", ".join(id_cols + [time_col])
        select_list = ",\n  ".join(select_parts)

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"SELECT\n  {select_list}\n"
            f"FROM {src}\n"
            f"GROUP BY {group_cols}"
        )
        if verbose:
            _log.debug("build_layer2a base SQL:\n%s", sql)
        result = SQLOutput(sql=self._transpile(sql), dialect=self.dialect)
        if verbose:
            _log.debug("build_layer2a done")
        return result

    # ------------------------------------------------------------------
    # build_layer2b
    # ------------------------------------------------------------------

    def build_layer2b(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the Layer 2B distributional CTEs table.

        For each (categorical, measurement, aggregator) group three CTEs are
        produced:

        * ``{safe}_raw`` — ``GROUP BY`` that computes per-category aggregates
          (``cat_val``).
        * ``{safe}_shares`` — window function over ``{safe}_raw`` that adds
          ``total_val = SUM(cat_val) OVER (PARTITION BY ids, time_col)``.
          Splitting into two CTEs avoids the invalid Snowflake pattern of
          nesting an aggregate inside a window aggregate
          (``SUM(SUM(x)) OVER (...)``).
        * ``{safe}_metrics`` — computes the requested distributional statistics
          from ``{safe}_shares``.

        All metrics CTEs are joined back to a ``base`` (DISTINCT ids × ts) CTE
        via ``LEFT JOIN``.

        CTE names are sanitised via :meth:`_safe_cte_name` to handle field
        names that contain special characters.
        """
        verbose = pipeline.config.verbose
        if verbose:
            _log.debug("build_layer2b started")

        if not pipeline.layer2b:
            if verbose:
                _log.debug("build_layer2b done — no distributional columns, step skipped")
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
            safe = "_".join(self._safe_cte_name(p) for p in [cat_name, meas_name, agg_name.lower()])
            raw_cte = safe + "_raw"
            shares_cte = safe + "_shares"
            metrics_cte = safe + "_metrics"
            metrics_cte_names.append(metrics_cte)

            # Step 1: group-level aggregation (no nested aggregate)
            cte_defs.append(
                f"{raw_cte} AS (\n"
                f"  SELECT\n"
                f"    {id_list},\n"
                f"    {time_col},\n"
                f"    {cat_name},\n"
                f"    {agg_name}({meas_name}) AS cat_val\n"
                f"  FROM {src}\n"
                f"  GROUP BY {id_list}, {time_col}, {cat_name}\n"
                f")"
            )

            # Step 2: window function on already-aggregated cat_val
            cte_defs.append(
                f"{shares_cte} AS (\n"
                f"  SELECT\n"
                f"    *,\n"
                f"    SUM(cat_val) OVER (PARTITION BY {id_list}, {time_col}) AS total_val\n"
                f"  FROM {raw_cte}\n"
                f")"
            )

            # Step 3: distributional metrics
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
        if verbose:
            _log.debug("build_layer2b base SQL:\n%s", sql)
        result = SQLOutput(sql=self._transpile(sql), dialect=self.dialect)
        if verbose:
            _log.debug("build_layer2b done")
        return result

    def _distributional_expr(self, metric: DistributionalMetric, cat_col: str, alias: str) -> str:
        """Return a SQL aggregate expression for one distributional metric.

        Operates on columns from the *shares* CTE:
        ``cat_val`` = per-category aggregate, ``total_val`` = entity×period total.
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

        Joins the MOB reference table with Layer 2A (and Layer 2B if present),
        then derives every :class:`~featkit.layer3.temporal_feature.TemporalFeature`
        via a ``GROUP BY (id_cols, ts_analysis)`` aggregation.  Each operator is
        expressed as an aggregate over a ``CASE WHEN mob BETWEEN … THEN col END``
        filter, so the approach scales to any window size without window-function
        ``ROWS BETWEEN`` clauses.

        The join between Layer 2A/B and the MOB table is on
        ``l2a.{time_col} = mob.{time_col}_relative`` so that the correct
        relative-period values are aggregated for each analysis snapshot.
        """
        verbose = pipeline.config.verbose
        if verbose:
            _log.debug("build_layer3 started")

        ds = pipeline.config.dataset
        id_cols = [f.name for f in ds.id_fields]
        time_col = ds.time_field.name
        mob_tbl = self._tbl(pipeline, "mob_ref")
        l2a_tbl = self._tbl(pipeline, "layer2a")
        l2b_tbl = self._tbl(pipeline, "layer2b")
        tbl = self._tbl(pipeline, "layer3")

        time_relative = f"{time_col}_relative"
        time_analysis = f"{time_col}_analysis"

        l2a_id_sel = ", ".join(f"l2a.{c}" for c in id_cols)
        group_by = ", ".join([f"l2a.{c}" for c in id_cols] + [f"mob.{time_analysis}"])

        select_parts: list[str] = [l2a_id_sel, f"mob.{time_analysis} AS {time_col}"]
        for feat in pipeline.layer3:
            expr = self._temporal_expr(feat)
            select_parts.append(f"{expr} AS {feat.column_name}")

        select_list = ",\n  ".join(select_parts)

        # Layer 2B join: use l2a's (id, ts) since l2a.ts = mob.ts_relative already
        if pipeline.layer2b:
            l2b_join_conds = " AND ".join(
                [f"l2b.{c} = l2a.{c}" for c in id_cols] + [f"l2b.{time_col} = l2a.{time_col}"]
            )
            l2b_join = f"\nLEFT JOIN {l2b_tbl} l2b ON {l2b_join_conds}"
        else:
            l2b_join = ""

        sql = (
            f"CREATE OR REPLACE TABLE {tbl} AS\n"
            f"SELECT\n  {select_list}\n"
            f"FROM {mob_tbl} mob\n"
            f"JOIN {l2a_tbl} l2a ON l2a.{time_col} = mob.{time_relative}"
            f"{l2b_join}\n"
            f"GROUP BY {group_by}"
        )
        if verbose:
            _log.debug("build_layer3 base SQL:\n%s", sql)
        result = SQLOutput(sql=self._transpile(sql), dialect=self.dialect)
        if verbose:
            _log.debug("build_layer3 done")
        return result

    def _temporal_expr(self, feat: TemporalFeature, mob_col: str = "mob.mob") -> str:
        """Return a GROUP BY-compatible aggregate expression for one temporal feature.

        Uses ``CASE WHEN {mob_col} BETWEEN … THEN col END`` inside an aggregate
        function so that each operator maps to a standard SQL aggregate over the
        appropriate period range.  Backward windows use negative mob offsets
        (current = 0, one period back = -1, etc.); forward windows use positive
        offsets.
        """
        from featkit.layer2.distributional import DistributionalColumn

        src_col = feat.source.column_name
        prefix = "l2b" if isinstance(feat.source, DistributionalColumn) else "l2a"
        col = f"{prefix}.{src_col}"

        op = feat.operator
        w = feat.window_size
        bwd = feat.direction == TimeWindowDirection.BACKWARD

        if w is not None:
            if bwd:
                lo, hi = -(w - 1), 0
            else:
                lo, hi = 0, w - 1
            in_window = f"{mob_col} BETWEEN {lo} AND {hi}"
            case_col = f"CASE WHEN {in_window} THEN {col} END"
            case_notnull = f"CASE WHEN {in_window} AND {col} IS NOT NULL AND {col} > 0 THEN 1 END"

        if op == TemporalOperator.PROM_U:
            return f"AVG({case_col})"
        if op == TemporalOperator.PROM_P:
            return f"AVG({case_col})"
        if op == TemporalOperator.SUM_U:
            return f"SUM({case_col})"
        if op == TemporalOperator.SUM_P:
            return f"SUM({case_col})"
        if op == TemporalOperator.MIN_U:
            return f"MIN({case_col})"
        if op == TemporalOperator.MAX_U:
            return f"MAX({case_col})"
        if op == TemporalOperator.ULT_MES:
            return f"MAX(CASE WHEN {mob_col} = 0 THEN {col} END)"
        if op == TemporalOperator.PREV_MES:
            return f"MAX(CASE WHEN {mob_col} = -1 THEN {col} END)"
        if op == TemporalOperator.CREC:
            curr = f"MAX(CASE WHEN {mob_col} = 0 THEN {col} END)"
            prev = f"MAX(CASE WHEN {mob_col} = -1 THEN {col} END)"
            return f"({curr} / NULLIF({prev}, 0)) - 1"
        if op == TemporalOperator.FREQ:
            return f"COUNT({case_notnull})"
        if op == TemporalOperator.XM:
            return f"CASE WHEN COUNT({case_notnull}) = {w} THEN 1 ELSE 0 END"
        if op == TemporalOperator.REC:
            return f"-MAX(CASE WHEN {col} IS NOT NULL THEN {mob_col} END)"
        if op == TemporalOperator.MEDIA_ABS:
            return f"MEDIAN({case_col})"
        if op == TemporalOperator.RATIO:
            return f"SUM({case_col})"
        return f"MAX(CASE WHEN {mob_col} = 0 THEN {col} END)"

    # ------------------------------------------------------------------
    # build_final_join
    # ------------------------------------------------------------------

    def build_final_join(self, pipeline: FeatureStorePipeline) -> SQLOutput:
        """Generate the final feature table joining Layer 2 and Layer 3."""
        verbose = pipeline.config.verbose
        if verbose:
            _log.debug("build_final_join started")

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
            + [f"l2a.{col.column_name}" for col in pipeline.layer2c]
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
        if verbose:
            _log.debug("build_final_join base SQL:\n%s", sql)
        result = SQLOutput(sql=self._transpile(sql), dialect=self.dialect)
        if verbose:
            _log.debug("build_final_join done")
        return result
