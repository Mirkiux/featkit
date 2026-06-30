"""Enumerators that form the type system of the featkit framework."""

from enum import Enum


class FieldRole(Enum):
    """Role of a column in the source facts table."""

    ID = "ID"
    TIME = "TIME"
    CATEGORICAL = "CATEGORICAL"
    MEASUREMENT = "MEASUREMENT"


class MeasurementType(Enum):
    """Semantic type of a measurement column.

    Governs which :class:`~featkit.enums.Layer2Aggregator` values are valid
    for that column at the Layer 1 → Layer 2 boundary.
    """

    MONTO = "MONTO"
    CANTIDAD = "CANTIDAD"
    TICKET = "TICKET"
    FLAG = "FLAG"
    FECHA = "FECHA"
    BALANCE = "BALANCE"
    TIME_DIFF = "TIME_DIFF"
    ESTADISTICO = "ESTADISTICO"


class TimeGranularity(Enum):
    """Temporal granularity of a time column."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    YEARLY = "YEARLY"


class CategoricalTreatment(Enum):
    """How a categorical column is used when building Layer 2."""

    PIVOT = "PIVOT"
    DISTRIBUTIONAL = "DISTRIBUTIONAL"
    BOTH = "BOTH"


class Layer2Aggregator(Enum):
    """SQL aggregation function applied when collapsing facts into Layer 2 columns."""

    SUM = "SUM"
    COUNT = "COUNT"
    MAX = "MAX"
    MIN = "MIN"
    AVG = "AVG"


class DistributionalMetric(Enum):
    """Statistical metric computed per-categorical in Layer 2B.

    All metrics are implemented as pure SQL aggregate expressions within CTEs —
    no custom UDAFs or stored procedures are required.
    """

    ENTROPY = "ENTROPY"
    HHI = "HHI"
    DOMINANT_PROPORTION = "DOMINANT_PROPORTION"
    MODE = "MODE"
    COUNT = "COUNT"


class Layer2OutputType(Enum):
    """Output type of any Layer 2 column.

    Governs which :class:`~featkit.enums.TemporalOperator` values are valid
    at the Layer 2 → Layer 3 boundary.
    """

    NUMERIC = "NUMERIC"
    FLAG = "FLAG"
    CATEGORICAL = "CATEGORICAL"
    TEMPORAL = "TEMPORAL"


class TemporalOperator(Enum):
    """Sliding-window or point-in-time operator applied in Layer 3."""

    PROM_U = "PROM_U"
    PROM_P = "PROM_P"
    SUM_U = "SUM_U"
    SUM_P = "SUM_P"
    ULT_MES = "ULT_MES"
    PREV_MES = "PREV_MES"
    CREC = "CREC"
    FREQ = "FREQ"
    MIN_U = "MIN_U"
    MAX_U = "MAX_U"
    REC = "REC"
    XM = "XM"
    MEDIA_ABS = "MEDIA_ABS"
    RATIO = "RATIO"


class TimeWindowDirection(Enum):
    """Direction of the temporal sliding window."""

    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"


class RatioMode(Enum):
    """Controls which denominators are paired with each numerator in Layer 2C.

    ``ALL_PROJECTIONS`` (default): every proper marginal projection of the
    numerator is used as a denominator — partial marginals (some fields set to
    ∅) as well as the global total (all fields ∅).

    ``GLOBAL_TOTAL``: only the fully-marginalised column (all categorical fields
    set to ∅) is used as a denominator, producing a single ratio per numerator
    that represents its share of the grand total.
    """

    ALL_PROJECTIONS = "ALL_PROJECTIONS"
    GLOBAL_TOTAL = "GLOBAL_TOTAL"
