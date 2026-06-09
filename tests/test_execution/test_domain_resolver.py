"""Tests for AdapterDomainResolver and AdapterCombinationResolver."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import CategoricalTreatment, MeasurementType, TimeGranularity
from featkit.execution.adapters.mock_adapter import MockAdapter
from featkit.execution.domain_resolver import AdapterCombinationResolver, AdapterDomainResolver
from featkit.fields.categorical_field import CategoricalField
from featkit.fields.id_field import IDField
from featkit.fields.measurement_field import MeasurementField
from featkit.fields.time_field import TimeField
from featkit.pipeline import FeatureStorePipeline

_SOURCE = "mydb.facts"
_FIELD = CategoricalField("segment", CategoricalTreatment.PIVOT)


def _mock_adapter(field_name: str, values: list[str]) -> MockAdapter:
    sql = f"SELECT DISTINCT {field_name} FROM {_SOURCE} WHERE {field_name} IS NOT NULL ORDER BY 1"
    df = pd.DataFrame({field_name: values})
    return MockAdapter({sql: df})


# ---------------------------------------------------------------------------
# AdapterDomainResolver unit tests
# ---------------------------------------------------------------------------


class TestIdentifierValidation:
    def test_invalid_source_reference_raises(self) -> None:
        adapter = MockAdapter({})
        with pytest.raises(ValueError, match="source_reference"):
            AdapterDomainResolver(adapter, "mydb; DROP TABLE facts--")

    def test_source_reference_with_spaces_raises(self) -> None:
        adapter = MockAdapter({})
        with pytest.raises(ValueError, match="source_reference"):
            AdapterDomainResolver(adapter, "my db.facts")

    def test_valid_dotted_source_reference_accepted(self) -> None:
        adapter = _mock_adapter("segment", ["a"])
        # Should not raise
        AdapterDomainResolver(adapter, "catalog.db.schema.table")

    def test_invalid_field_name_raises(self) -> None:
        adapter = MockAdapter({})
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        bad_field = CategoricalField("bad; DROP TABLE--", CategoricalTreatment.PIVOT)
        with pytest.raises(ValueError, match="field.name"):
            resolver(bad_field)

    def test_field_name_with_spaces_raises(self) -> None:
        adapter = MockAdapter({})
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        bad_field = CategoricalField("bad field", CategoricalTreatment.PIVOT)
        with pytest.raises(ValueError, match="field.name"):
            resolver(bad_field)


class TestAdapterDomainResolver:
    def test_returns_list_of_strings(self) -> None:
        adapter = _mock_adapter("segment", ["retail", "sme", "corporate"])
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        result = resolver(_FIELD)
        assert result == ["retail", "sme", "corporate"]

    def test_values_cast_to_str(self) -> None:
        sql = f"SELECT DISTINCT segment FROM {_SOURCE} WHERE segment IS NOT NULL ORDER BY 1"
        df = pd.DataFrame({"segment": [1, 2, 3]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        result = resolver(_FIELD)
        assert result == ["1", "2", "3"]

    def test_executes_correct_sql(self) -> None:
        adapter = _mock_adapter("segment", ["a"])
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        resolver(_FIELD)
        expected_sql = (
            f"SELECT DISTINCT segment FROM {_SOURCE} WHERE segment IS NOT NULL ORDER BY 1"
        )
        assert adapter.call_count(expected_sql) == 1

    def test_empty_result_returns_empty_list(self) -> None:
        sql = f"SELECT DISTINCT segment FROM {_SOURCE} WHERE segment IS NOT NULL ORDER BY 1"
        df = pd.DataFrame({"segment": []})
        adapter = MockAdapter({sql: df})
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        assert resolver(_FIELD) == []

    def test_callable_interface(self) -> None:
        """Resolver is callable — compatible with PivotSpaceBuilder.domain_resolver."""
        adapter = _mock_adapter("segment", ["x"])
        resolver = AdapterDomainResolver(adapter, _SOURCE)
        assert callable(resolver)

    def test_different_fields_issue_different_queries(self) -> None:
        field_a = CategoricalField("region", CategoricalTreatment.PIVOT)
        field_b = CategoricalField("channel", CategoricalTreatment.PIVOT)

        sql_a = f"SELECT DISTINCT region FROM {_SOURCE} WHERE region IS NOT NULL ORDER BY 1"
        sql_b = f"SELECT DISTINCT channel FROM {_SOURCE} WHERE channel IS NOT NULL ORDER BY 1"

        adapter = MockAdapter(
            {
                sql_a: pd.DataFrame({"region": ["north", "south"]}),
                sql_b: pd.DataFrame({"channel": ["online", "branch"]}),
            }
        )
        resolver = AdapterDomainResolver(adapter, _SOURCE)

        assert resolver(field_a) == ["north", "south"]
        assert resolver(field_b) == ["online", "branch"]

    def test_verbose_true_logs_sql(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _mock_adapter("segment", ["a"])
        resolver = AdapterDomainResolver(adapter, _SOURCE, verbose=True)
        with caplog.at_level(logging.DEBUG, logger="featkit.execution.domain_resolver"):
            resolver(_FIELD)
        assert "AdapterDomainResolver SQL:" in caplog.text

    def test_verbose_false_emits_no_sql_log(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _mock_adapter("segment", ["a"])
        resolver = AdapterDomainResolver(adapter, _SOURCE, verbose=False)
        with caplog.at_level(logging.DEBUG, logger="featkit.execution.domain_resolver"):
            resolver(_FIELD)
        assert "AdapterDomainResolver SQL:" not in caplog.text


# ---------------------------------------------------------------------------
# Pipeline integration — adapter wired through FeatureStoreConfig
# ---------------------------------------------------------------------------


def _ds_no_allowed_values() -> SimpleDataset:
    return SimpleDataset(
        _SOURCE,
        [
            IDField("client_id"),
            TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("amount", MeasurementType.MONTO),
            CategoricalField("segment", CategoricalTreatment.PIVOT),
        ],
    )


def _ds_mixed() -> SimpleDataset:
    """One static domain, one dynamic domain."""
    return SimpleDataset(
        _SOURCE,
        [
            IDField("client_id"),
            TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
            MeasurementField("amount", MeasurementType.MONTO),
            CategoricalField(
                "channel",
                CategoricalTreatment.PIVOT,
                allowed_values=["branch", "online"],
            ),
            CategoricalField("segment", CategoricalTreatment.PIVOT),
        ],
    )


class TestPipelineWithAdapter:
    def test_build_resolves_domain_via_adapter(self) -> None:
        adapter = _mock_adapter("segment", ["retail", "sme"])
        cfg = FeatureStoreConfig(
            dataset=_ds_no_allowed_values(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
            adapter=adapter,
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        col_names = [c.column_name for c in pipeline.layer2a]
        assert any("retail" in n for n in col_names)
        assert any("sme" in n for n in col_names)

    def test_build_without_adapter_raises_for_missing_allowed_values(self) -> None:
        cfg = FeatureStoreConfig(
            dataset=_ds_no_allowed_values(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
        )
        with pytest.raises(ValueError, match="no allowed_values and no domain_resolver"):
            FeatureStorePipeline(config=cfg).build()

    def test_adapter_query_executed_once_per_field(self) -> None:
        adapter = _mock_adapter("segment", ["a", "b"])
        cfg = FeatureStoreConfig(
            dataset=_ds_no_allowed_values(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
            adapter=adapter,
        )
        FeatureStorePipeline(config=cfg).build()
        expected_sql = (
            f"SELECT DISTINCT segment FROM {_SOURCE} WHERE segment IS NOT NULL ORDER BY 1"
        )
        assert adapter.call_count(expected_sql) == 1

    def test_combined_query_replaces_per_field_queries(self) -> None:
        """One combined SELECT DISTINCT query is issued for all pivot fields.

        allowed_values become IN-filters in the WHERE clause rather than
        preventing the query from running altogether.
        """
        combined_sql = (
            f"SELECT DISTINCT channel, segment FROM {_SOURCE} "
            f"WHERE channel IS NOT NULL AND segment IS NOT NULL "
            f"AND channel IN ('branch', 'online') ORDER BY 1, 2"
        )
        df = pd.DataFrame({"channel": ["branch"], "segment": ["retail"]})
        adapter = MockAdapter({combined_sql: df})
        cfg = FeatureStoreConfig(
            dataset=_ds_mixed(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
            adapter=adapter,
        )
        FeatureStorePipeline(config=cfg).build()
        assert adapter.call_count(combined_sql) == 1
        per_field_sql = (
            f"SELECT DISTINCT segment FROM {_SOURCE} WHERE segment IS NOT NULL ORDER BY 1"
        )
        assert adapter.call_count(per_field_sql) == 0

    def test_verbose_true_logs_combination_resolver_sql(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        combined_sql = (
            f"SELECT DISTINCT channel, segment FROM {_SOURCE} "
            f"WHERE channel IS NOT NULL AND segment IS NOT NULL "
            f"AND channel IN ('branch', 'online') ORDER BY 1, 2"
        )
        df = pd.DataFrame({"channel": ["branch"], "segment": ["retail"]})
        adapter = MockAdapter({combined_sql: df})
        cfg = FeatureStoreConfig(
            dataset=_ds_mixed(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
            adapter=adapter,
            verbose=True,
        )
        with caplog.at_level(logging.DEBUG, logger="featkit.execution.domain_resolver"):
            FeatureStorePipeline(config=cfg).build()
        assert "AdapterCombinationResolver SQL:" in caplog.text

    def test_no_adapter_with_static_domains_builds_fine(self) -> None:
        ds = SimpleDataset(
            _SOURCE,
            [
                IDField("client_id"),
                TimeField("period", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY),
                MeasurementField("amount", MeasurementType.MONTO),
                CategoricalField(
                    "channel",
                    CategoricalTreatment.PIVOT,
                    allowed_values=["branch", "online"],
                ),
            ],
        )
        cfg = FeatureStoreConfig(
            dataset=ds,
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
        )
        pipeline = FeatureStorePipeline(config=cfg).build()
        assert len(pipeline.layer2a) > 0


# ---------------------------------------------------------------------------
# AdapterCombinationResolver unit tests
# ---------------------------------------------------------------------------

_CHANNEL = CategoricalField(
    "channel", CategoricalTreatment.PIVOT, allowed_values=["branch", "online"]
)
_SEGMENT = CategoricalField("segment", CategoricalTreatment.PIVOT)


def _combo_sql(*fields_and_values: tuple[str, list[str] | None], source: str = _SOURCE) -> str:
    """Build the expected combined SELECT DISTINCT SQL for the given fields.

    Each entry is (field_name, allowed_values_or_None).  Fields are sorted by
    name to match AdapterCombinationResolver's deterministic ordering.
    """
    sorted_fv = sorted(fields_and_values, key=lambda x: x[0])
    col_list = ", ".join(name for name, _ in sorted_fv)
    order_list = ", ".join(str(i + 1) for i in range(len(sorted_fv)))
    where_parts = [f"{name} IS NOT NULL" for name, _ in sorted_fv]
    for name, vals in sorted_fv:
        if vals:
            escaped = ", ".join("'" + v.replace("'", "''") + "'" for v in vals)
            where_parts.append(f"{name} IN ({escaped})")
    return (
        f"SELECT DISTINCT {col_list} FROM {source} "
        f"WHERE {' AND '.join(where_parts)} ORDER BY {order_list}"
    )


class TestAdapterCombinationResolverValidation:
    def test_invalid_source_reference_raises(self) -> None:
        adapter = MockAdapter({})
        with pytest.raises(ValueError, match="source_reference"):
            AdapterCombinationResolver(adapter, "mydb; DROP TABLE facts--")

    def test_invalid_field_name_raises(self) -> None:
        adapter = MockAdapter({})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        bad_field = CategoricalField("bad; DROP TABLE--", CategoricalTreatment.PIVOT)
        with pytest.raises(ValueError, match="field.name"):
            resolver([bad_field])


class TestAdapterCombinationResolver:
    def test_empty_fields_returns_empty_list(self) -> None:
        resolver = AdapterCombinationResolver(MockAdapter({}), _SOURCE)
        assert resolver([]) == []

    def test_single_field_no_allowed_values_generates_no_in_filter(self) -> None:
        sql = _combo_sql(("segment", None))
        df = pd.DataFrame({"segment": ["retail", "sme"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        result = resolver([_SEGMENT])
        assert adapter.call_count(sql) == 1
        assert len(result) == 2

    def test_single_field_with_allowed_values_adds_in_filter(self) -> None:
        sql = _combo_sql(("channel", ["branch", "online"]))
        df = pd.DataFrame({"channel": ["branch", "online"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        result = resolver([_CHANNEL])
        assert adapter.call_count(sql) == 1
        assert len(result) == 2

    def test_multiple_fields_sorted_by_name(self) -> None:
        sql = _combo_sql(("channel", ["branch", "online"]), ("segment", None))
        df = pd.DataFrame({"channel": ["branch", "online"], "segment": ["retail", "sme"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        result = resolver([_SEGMENT, _CHANNEL])  # intentionally reversed order
        assert adapter.call_count(sql) == 1
        assert len(result) == 2

    def test_returns_field_keyed_dicts(self) -> None:
        sql = _combo_sql(("channel", ["branch", "online"]), ("segment", None))
        df = pd.DataFrame({"channel": ["branch"], "segment": ["retail"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        result = resolver([_CHANNEL, _SEGMENT])
        assert result == [{_CHANNEL: "branch", _SEGMENT: "retail"}]

    def test_values_cast_to_str(self) -> None:
        sql = _combo_sql(("segment", None))
        df = pd.DataFrame({"segment": [1, 2]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        result = resolver([_SEGMENT])
        assert result == [{_SEGMENT: "1"}, {_SEGMENT: "2"}]

    def test_empty_result_returns_empty_list(self) -> None:
        sql = _combo_sql(("segment", None))
        df = pd.DataFrame({"segment": pd.Series([], dtype=str)})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        assert resolver([_SEGMENT]) == []

    def test_allowed_values_single_quotes_are_escaped(self) -> None:
        tricky = CategoricalField(
            "region", CategoricalTreatment.PIVOT, allowed_values=["it's here", "plain"]
        )
        sql = _combo_sql(("region", ["it's here", "plain"]))
        assert "it''s here" in sql
        df = pd.DataFrame({"region": ["it's here"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE)
        result = resolver([tricky])
        assert result == [{tricky: "it's here"}]

    def test_verbose_true_logs_sql(self, caplog: pytest.LogCaptureFixture) -> None:
        sql = _combo_sql(("segment", None))
        df = pd.DataFrame({"segment": ["retail"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE, verbose=True)
        with caplog.at_level(logging.DEBUG, logger="featkit.execution.domain_resolver"):
            resolver([_SEGMENT])
        assert "AdapterCombinationResolver SQL:" in caplog.text

    def test_verbose_false_emits_no_sql_log(self, caplog: pytest.LogCaptureFixture) -> None:
        sql = _combo_sql(("segment", None))
        df = pd.DataFrame({"segment": ["retail"]})
        adapter = MockAdapter({sql: df})
        resolver = AdapterCombinationResolver(adapter, _SOURCE, verbose=False)
        with caplog.at_level(logging.DEBUG, logger="featkit.execution.domain_resolver"):
            resolver([_SEGMENT])
        assert "AdapterCombinationResolver SQL:" not in caplog.text
