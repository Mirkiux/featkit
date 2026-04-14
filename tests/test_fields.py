"""Tests for Plan 02 — AbstractField, IDField, TimeField."""

import pytest

from featkit.enums import FieldRole, TimeGranularity
from featkit.fields import AbstractField, IDField, TimeField


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
