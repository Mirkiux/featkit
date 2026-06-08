"""TemporalSpaceBuilder — generates all TemporalFeature objects from Layer 2 columns."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from featkit.enums import Layer2OutputType, TemporalOperator, TimeWindowDirection
from featkit.layer2.base import AbstractL2Column
from featkit.layer3.temporal_feature import _POINT_IN_TIME_OPERATORS, TemporalFeature

_log = logging.getLogger(__name__)

#: Operators that require composed (MEDIA_ABS / RATIO) window sizes.
_COMPOSED_OPERATORS: frozenset[TemporalOperator] = frozenset(
    {TemporalOperator.MEDIA_ABS, TemporalOperator.RATIO}
)


class TemporalSpaceBuilder:
    """Generates the full set of TemporalFeature objects from a list of Layer 2 columns.

    For each Layer 2 column, the operator set is taken from the column's output
    contract (or the per-output-type override). Each operator is then paired with
    the appropriate window sizes:

    - Point-in-time operators (``ULT_MES``, ``PREV_MES``, ``REC``): ``window_size=None``
    - ``MEDIA_ABS`` / ``RATIO``: ``composed_windows`` (skipped when ``None``)
    - All other windowed operators: ``time_windows``

    Args:
        layer2_columns: All Layer 2 columns to build features from.
        time_windows: Window sizes for standard windowed operators.
        composed_windows: Window sizes for ``MEDIA_ABS`` and ``RATIO``. When
            ``None``, those operators are omitted entirely.
        direction: Sliding-window direction applied to every feature.
        operators_override: Per-output-type override. Only operators that are
            also contract-valid for the column's output type are used.
        verbose: When ``True``, emits ``DEBUG``-level log messages at key
            milestones: builder start/end, and for each generated feature the
            ``(layer2_column, operator, window)`` combination dict and the
            resulting column name.
    """

    def __init__(
        self,
        layer2_columns: list[AbstractL2Column],
        time_windows: list[int],
        composed_windows: list[int] | None = None,
        direction: TimeWindowDirection = TimeWindowDirection.BACKWARD,
        operators_override: dict[Layer2OutputType, list[TemporalOperator]] | None = None,
        verbose: bool = False,
    ) -> None:
        self.layer2_columns: list[AbstractL2Column] = layer2_columns
        self.time_windows = time_windows
        self.composed_windows = composed_windows
        self.direction = direction
        self.operators_override = operators_override
        self.verbose = verbose

    def build(self) -> list[TemporalFeature]:
        """Build and return all TemporalFeature objects."""
        if self.verbose:
            _log.debug("TemporalSpaceBuilder.build() started")

        results: list[TemporalFeature] = []
        seen: set[str] = set()

        for col in self.layer2_columns:
            if self.operators_override and col.output_type in self.operators_override:
                operators = [
                    op
                    for op in self.operators_override[col.output_type]
                    if col.output_contract.is_valid(op)
                ]
            else:
                operators = sorted(
                    col.output_contract.valid_temporal_operators, key=lambda o: o.value
                )

            for op in operators:
                window_sizes: Sequence[int | None]
                if op in _COMPOSED_OPERATORS:
                    if self.composed_windows is None:
                        continue
                    window_sizes = list(self.composed_windows)
                elif op in _POINT_IN_TIME_OPERATORS:
                    window_sizes = [None]
                else:
                    window_sizes = list(self.time_windows)

                for ws in window_sizes:
                    feat = TemporalFeature(col, op, self.direction, window_size=ws)
                    if self.verbose:
                        _log.debug(
                            "combo: layer2_column=%r, operator=%s, window=%s",
                            col.column_name,
                            op.value,
                            ws,
                        )
                        _log.debug(
                            "combination: %s",
                            {
                                "layer2_column": col.column_name,
                                "operator": op.value,
                                "window": ws,
                            },
                        )
                        _log.debug("column_name: %r", feat.column_name)
                    if feat.column_name not in seen:
                        seen.add(feat.column_name)
                        results.append(feat)

        if self.verbose:
            _log.debug("TemporalSpaceBuilder.build() done — %d feature(s) generated", len(results))
        return results
