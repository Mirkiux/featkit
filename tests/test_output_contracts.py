"""Tests for Plan 05 — AbstractLayer2OutputContract and concrete defaults."""

import pytest

from featkit.contracts.output.base import AbstractLayer2OutputContract
from featkit.contracts.output.defaults import (
    CategoricalOutputContract,
    FlagOutputContract,
    NumericOutputContract,
    TemporalOutputContract,
    get_default_output_contract,
)
from featkit.enums import Layer2OutputType, TemporalOperator


class TestAbstractLayer2OutputContract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AbstractLayer2OutputContract(Layer2OutputType.NUMERIC)  # type: ignore[abstract]

    def test_is_valid_true_for_valid_operator(self) -> None:
        assert NumericOutputContract().is_valid(TemporalOperator.PROM_U) is True

    def test_is_valid_false_for_invalid_operator(self) -> None:
        assert CategoricalOutputContract().is_valid(TemporalOperator.PROM_U) is False

    def test_repr_contains_class_and_output_type(self) -> None:
        r = repr(NumericOutputContract())
        assert "NumericOutputContract" in r
        assert "NUMERIC" in r


class TestConcreteOutputContracts:
    @pytest.mark.parametrize(
        "contract_cls,ot,expected_ops",
        [
            (
                NumericOutputContract,
                Layer2OutputType.NUMERIC,
                {
                    TemporalOperator.PROM_U,
                    TemporalOperator.PROM_P,
                    TemporalOperator.SUM_U,
                    TemporalOperator.SUM_P,
                    TemporalOperator.CREC,
                    TemporalOperator.MIN_U,
                    TemporalOperator.MAX_U,
                    TemporalOperator.ULT_MES,
                    TemporalOperator.PREV_MES,
                    TemporalOperator.FREQ,
                    TemporalOperator.XM,
                    TemporalOperator.MEDIA_ABS,
                    TemporalOperator.RATIO,
                },
            ),
            (
                FlagOutputContract,
                Layer2OutputType.FLAG,
                {
                    TemporalOperator.ULT_MES,
                    TemporalOperator.PREV_MES,
                    TemporalOperator.FREQ,
                    TemporalOperator.XM,
                    TemporalOperator.REC,
                },
            ),
            (
                CategoricalOutputContract,
                Layer2OutputType.CATEGORICAL,
                {TemporalOperator.ULT_MES, TemporalOperator.PREV_MES, TemporalOperator.REC},
            ),
            (
                TemporalOutputContract,
                Layer2OutputType.TEMPORAL,
                {
                    TemporalOperator.ULT_MES,
                    TemporalOperator.PREV_MES,
                    TemporalOperator.REC,
                    TemporalOperator.MIN_U,
                    TemporalOperator.MAX_U,
                    TemporalOperator.CREC,
                },
            ),
        ],
    )
    def test_valid_operators(
        self,
        contract_cls: type[AbstractLayer2OutputContract],
        ot: Layer2OutputType,
        expected_ops: set[TemporalOperator],
    ) -> None:
        contract = contract_cls()  # type: ignore[call-arg]
        assert contract.output_type == ot
        assert contract.valid_temporal_operators == frozenset(expected_ops)

    def test_returns_frozenset(self) -> None:
        instances: list[AbstractLayer2OutputContract] = [
            NumericOutputContract(),
            FlagOutputContract(),
            CategoricalOutputContract(),
            TemporalOutputContract(),
        ]
        for contract in instances:
            assert isinstance(contract.valid_temporal_operators, frozenset)

    def test_frozenset_is_non_empty(self) -> None:
        instances: list[AbstractLayer2OutputContract] = [
            NumericOutputContract(),
            FlagOutputContract(),
            CategoricalOutputContract(),
            TemporalOutputContract(),
        ]
        for contract in instances:
            assert len(contract.valid_temporal_operators) > 0


class TestGetDefaultOutputContract:
    def test_covers_every_output_type(self) -> None:
        for ot in Layer2OutputType:
            contract = get_default_output_contract(ot)
            assert isinstance(contract, AbstractLayer2OutputContract)
            assert contract.output_type == ot

    def test_returns_correct_contract_class(self) -> None:
        assert isinstance(
            get_default_output_contract(Layer2OutputType.NUMERIC), NumericOutputContract
        )
        assert isinstance(get_default_output_contract(Layer2OutputType.FLAG), FlagOutputContract)
        assert isinstance(
            get_default_output_contract(Layer2OutputType.CATEGORICAL), CategoricalOutputContract
        )
        assert isinstance(
            get_default_output_contract(Layer2OutputType.TEMPORAL), TemporalOutputContract
        )

    def test_flag_excludes_sum_and_average_operators(self) -> None:
        contract = get_default_output_contract(Layer2OutputType.FLAG)
        assert TemporalOperator.SUM_U not in contract.valid_temporal_operators
        assert TemporalOperator.PROM_U not in contract.valid_temporal_operators

    def test_categorical_excludes_numeric_operators(self) -> None:
        contract = get_default_output_contract(Layer2OutputType.CATEGORICAL)
        assert TemporalOperator.PROM_U not in contract.valid_temporal_operators
        assert TemporalOperator.SUM_U not in contract.valid_temporal_operators
        assert TemporalOperator.CREC not in contract.valid_temporal_operators

    def test_numeric_has_most_operators(self) -> None:
        counts = {
            ot: len(get_default_output_contract(ot).valid_temporal_operators)
            for ot in Layer2OutputType
        }
        assert counts[Layer2OutputType.NUMERIC] == max(counts.values())
