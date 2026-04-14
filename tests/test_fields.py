"""Tests for Plans 02 and 03 — all field descriptors."""

import pytest

from featkit.enums import (
    CategoricalTreatment,
    DistributionalMetric,
    FieldRole,
    MeasurementType,
    TimeGranularity,
)
from featkit.fields import AbstractField, CategoricalField, IDField, MeasurementField, TimeField


class TestAbstractField:
    def test_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            AbstractField(name="x")  # type: ignore[abstract]

    def test_name_stored_correctly(self) -> None:
        field = IDField(name="ID_CLIENTE")
        assert field.name == "ID_CLIENTE"

    def test_repr_includes_class_and_name(self) -> None:
        field = IDField(name="ID_CLIENTE")
        assert "IDField" in repr(field)
        assert "ID_CLIENTE" in repr(field)

    def test_equality_same_type_same_name(self) -> None:
        assert IDField(name="X") == IDField(name="X")

    def test_equality_different_name(self) -> None:
        assert IDField(name="X") != IDField(name="Y")

    def test_equality_different_type(self) -> None:
        assert IDField(name="X") != TimeField(
            name="X",
            source_granularity=TimeGranularity.MONTHLY,
            target_granularity=TimeGranularity.MONTHLY,
        )

    def test_hashable(self) -> None:
        field = IDField(name="X")
        assert hash(field) is not None
        assert {field, IDField(name="X")} == {field}


class TestIDField:
    def test_instantiable(self) -> None:
        assert IDField(name="ID_CLIENTE") is not None

    def test_role_is_id(self) -> None:
        assert IDField(name="ID_CLIENTE").role == FieldRole.ID

    def test_name_property(self) -> None:
        assert IDField(name="MY_COL").name == "MY_COL"

    def test_repr(self) -> None:
        r = repr(IDField(name="MY_COL"))
        assert "IDField" in r
        assert "MY_COL" in r


class TestTimeField:
    def _make(
        self,
        name: str = "PERIODO",
        source: TimeGranularity = TimeGranularity.MONTHLY,
        target: TimeGranularity = TimeGranularity.MONTHLY,
    ) -> TimeField:
        return TimeField(name=name, source_granularity=source, target_granularity=target)

    def test_instantiable(self) -> None:
        assert self._make() is not None

    def test_role_is_time(self) -> None:
        assert self._make().role == FieldRole.TIME

    def test_truncation_required_false_when_same_granularity(self) -> None:
        assert self._make().truncation_required is False

    def test_truncation_required_true_when_granularities_differ(self) -> None:
        field = self._make(source=TimeGranularity.DAILY, target=TimeGranularity.MONTHLY)
        assert field.truncation_required is True

    def test_source_granularity_stored(self) -> None:
        field = self._make(source=TimeGranularity.DAILY, target=TimeGranularity.MONTHLY)
        assert field.source_granularity == TimeGranularity.DAILY

    def test_target_granularity_stored(self) -> None:
        field = self._make(source=TimeGranularity.DAILY, target=TimeGranularity.MONTHLY)
        assert field.target_granularity == TimeGranularity.MONTHLY

    def test_repr_includes_granularities(self) -> None:
        field = self._make(source=TimeGranularity.DAILY, target=TimeGranularity.MONTHLY)
        r = repr(field)
        assert "DAILY" in r
        assert "MONTHLY" in r

    @pytest.mark.parametrize(
        "src,tgt,expected",
        [
            (TimeGranularity.DAILY, TimeGranularity.DAILY, False),
            (TimeGranularity.DAILY, TimeGranularity.MONTHLY, True),
            (TimeGranularity.WEEKLY, TimeGranularity.MONTHLY, True),
            (TimeGranularity.MONTHLY, TimeGranularity.MONTHLY, False),
            (TimeGranularity.QUARTERLY, TimeGranularity.YEARLY, True),
        ],
    )
    def test_truncation_required_parametrized(
        self,
        src: TimeGranularity,
        tgt: TimeGranularity,
        expected: bool,
    ) -> None:
        assert self._make(source=src, target=tgt).truncation_required is expected

    def test_equal_when_all_components_match(self) -> None:
        a = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        assert a == b

    def test_not_equal_when_target_differs(self) -> None:
        a = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.DAILY)
        assert a != b

    def test_not_equal_when_source_differs(self) -> None:
        a = self._make(source=TimeGranularity.DAILY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        assert a != b

    def test_not_equal_when_name_differs(self) -> None:
        a = TimeField("PERIODO", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY)
        b = TimeField("FECHA", TimeGranularity.MONTHLY, TimeGranularity.MONTHLY)
        assert a != b

    def test_hash_equal_fields_have_same_hash(self) -> None:
        a = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        assert hash(a) == hash(b)

    def test_hash_different_granularity_produces_different_hash(self) -> None:
        a = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.DAILY)
        assert hash(a) != hash(b)

    def test_usable_as_dict_key(self) -> None:
        a = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.DAILY)
        d = {a: "monthly", b: "daily"}
        assert d[a] == "monthly"
        assert d[b] == "daily"

    def test_set_deduplication_respects_granularity(self) -> None:
        a = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        b = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.DAILY)
        c = self._make(source=TimeGranularity.MONTHLY, target=TimeGranularity.MONTHLY)
        assert len({a, b, c}) == 2


class TestCategoricalField:
    def test_role_is_categorical(self) -> None:
        field = CategoricalField(name="SECTOR", treatment=CategoricalTreatment.PIVOT)
        assert field.role == FieldRole.CATEGORICAL

    def test_pivot_needs_no_metrics(self) -> None:
        field = CategoricalField(name="SECTOR", treatment=CategoricalTreatment.PIVOT)
        assert field.distributional_metrics == []

    def test_distributional_raises_without_metrics(self) -> None:
        with pytest.raises(ValueError, match="distributional_metrics"):
            CategoricalField(name="SECTOR", treatment=CategoricalTreatment.DISTRIBUTIONAL)

    def test_both_raises_without_metrics(self) -> None:
        with pytest.raises(ValueError, match="distributional_metrics"):
            CategoricalField(name="SECTOR", treatment=CategoricalTreatment.BOTH)

    def test_distributional_stores_metrics(self) -> None:
        metrics = [DistributionalMetric.ENTROPY, DistributionalMetric.HHI]
        field = CategoricalField(
            name="SECTOR",
            treatment=CategoricalTreatment.DISTRIBUTIONAL,
            distributional_metrics=metrics,
        )
        assert field.distributional_metrics == metrics

    def test_allowed_values_default_none(self) -> None:
        field = CategoricalField(name="SECTOR", treatment=CategoricalTreatment.PIVOT)
        assert field.allowed_values is None

    def test_allowed_values_stored(self) -> None:
        vals = ["A", "B", "C"]
        field = CategoricalField(
            name="SECTOR", treatment=CategoricalTreatment.PIVOT, allowed_values=vals
        )
        assert field.allowed_values == vals

    def test_equal_same_name_and_treatment(self) -> None:
        a = CategoricalField(name="X", treatment=CategoricalTreatment.PIVOT)
        b = CategoricalField(name="X", treatment=CategoricalTreatment.PIVOT)
        assert a == b

    def test_not_equal_different_treatment(self) -> None:
        a = CategoricalField(name="X", treatment=CategoricalTreatment.PIVOT)
        b = CategoricalField(
            name="X",
            treatment=CategoricalTreatment.DISTRIBUTIONAL,
            distributional_metrics=[DistributionalMetric.ENTROPY],
        )
        assert a != b

    def test_not_equal_different_name(self) -> None:
        a = CategoricalField(name="A", treatment=CategoricalTreatment.PIVOT)
        b = CategoricalField(name="B", treatment=CategoricalTreatment.PIVOT)
        assert a != b

    def test_hashable_and_deduplicates_in_set(self) -> None:
        a = CategoricalField(name="X", treatment=CategoricalTreatment.PIVOT)
        b = CategoricalField(name="X", treatment=CategoricalTreatment.PIVOT)
        assert len({a, b}) == 1


class TestMeasurementField:
    def test_role_is_measurement(self) -> None:
        field = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        assert field.role == FieldRole.MEASUREMENT

    def test_measurement_type_stored(self) -> None:
        field = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        assert field.measurement_type == MeasurementType.MONTO

    def test_contract_defaults_to_none(self) -> None:
        field = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        assert field.contract is None

    def test_contract_stored_without_execution(self) -> None:
        from unittest.mock import MagicMock
        mock_contract = MagicMock()
        field = MeasurementField(
            name="MTO", measurement_type=MeasurementType.MONTO, contract=mock_contract
        )
        assert field.contract is mock_contract
        mock_contract.assert_not_called()

    def test_equal_same_name_and_type(self) -> None:
        a = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        b = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        assert a == b

    def test_not_equal_different_measurement_type(self) -> None:
        a = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        b = MeasurementField(name="MTO", measurement_type=MeasurementType.CANTIDAD)
        assert a != b

    def test_hashable_and_usable_as_dict_key(self) -> None:
        f = MeasurementField(name="MTO", measurement_type=MeasurementType.MONTO)
        d = {f: "value"}
        assert d[f] == "value"
