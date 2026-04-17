"""Tests for AdapterDomainResolver."""

from __future__ import annotations

import pandas as pd
import pytest

from featkit.config import FeatureStoreConfig
from featkit.dataset.base import SimpleDataset
from featkit.enums import CategoricalTreatment, MeasurementType, TimeGranularity
from featkit.execution.adapters.mock_adapter import MockAdapter
from featkit.execution.domain_resolver import AdapterDomainResolver
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

    def test_static_fields_do_not_trigger_adapter_query(self) -> None:
        sql_segment = f"SELECT DISTINCT segment FROM {_SOURCE} WHERE segment IS NOT NULL ORDER BY 1"
        adapter = MockAdapter({sql_segment: pd.DataFrame({"segment": ["retail"]})})
        cfg = FeatureStoreConfig(
            dataset=_ds_mixed(),
            output_schema="analytics",
            output_table_prefix="feat_",
            time_windows=[3],
            adapter=adapter,
        )
        FeatureStorePipeline(config=cfg).build()
        # Only the dynamic field triggered a query — channel had allowed_values
        sql_channel = f"SELECT DISTINCT channel FROM {_SOURCE} WHERE channel IS NOT NULL ORDER BY 1"
        assert adapter.call_count(sql_channel) == 0
        assert adapter.call_count(sql_segment) == 1

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
