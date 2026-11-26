"""
Shared data schemas for the multi-asset allocation project.

Public interfaces use the security codes 510300, 511010 and 518880.
Asset names are presentation metadata only and are never public data keys.
"""

from datetime import date
from math import isfinite
from typing import Literal, NotRequired, TypedDict


ASSET_CODES = (
    "510300",
    "511010",
    "518880",
)

AssetCode = Literal["510300", "511010", "518880"]

AssetValues = TypedDict(
    "AssetValues",
    {
        "510300": float,
        "511010": float,
        "518880": float,
    },
)

MarketFeatureValues = TypedDict(
    "MarketFeatureValues",
    {
        "510300": float | None,
        "511010": float | None,
        "518880": float | None,
    },
)


INITIAL_CAPITAL = 1_000_000

TRANSACTION_COST_BUY = 0.0003
TRANSACTION_COST_SELL = 0.0003

FEEDBACK_WINDOW_SHORT = 20
FEEDBACK_WINDOW_MEDIUM = 60
RISK_WINDOW = 250

REBALANCE_FREQUENCY = "monthly"

WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0
WEIGHT_SUM_TOLERANCE = 1e-6


class MarketFeatures(TypedDict):
    asset_return_20d: MarketFeatureValues
    asset_return_60d: MarketFeatureValues
    asset_volatility_20d: MarketFeatureValues
    asset_volatility_60d: MarketFeatureValues
    asset_volatility_250d: MarketFeatureValues


class MarketDataInput(TypedDict):
    as_of_date: str
    market_features: MarketFeatures
    asset_correlations: dict
    data_quality: dict


class PortfolioReturn(TypedDict):
    d20: float
    d60: float


class PortfolioVolatility(TypedDict):
    d20: float
    d60: float


class TriggerFlags(TypedDict):
    drawdown: bool
    volatility: bool
    correlation: bool


class RiskChangeAttribution(TypedDict):
    weight: AssetValues
    vol: AssetValues
    corr: AssetValues
    residual: float


class RiskFeedback(TypedDict):
    portfolio_return: NotRequired[PortfolioReturn]
    portfolio_volatility: NotRequired[PortfolioVolatility]
    current_drawdown: NotRequired[float]
    max_drawdown: NotRequired[float]
    cvar_95: NotRequired[float]
    trigger_flags: NotRequired[TriggerFlags]
    risk_contribution: NotRequired[AssetValues]
    budget_deviation: NotRequired[AssetValues]
    drawdown_attribution: NotRequired[AssetValues]
    risk_change_attribution: NotRequired[RiskChangeAttribution]
    concentration_state: NotRequired[str]


class ModelInput(TypedDict):
    as_of_date: str
    market_features: MarketFeatures
    asset_correlations: dict
    data_quality: dict
    feedback: NotRequired[RiskFeedback]


class ModelDecision(TypedDict):
    decision_status: str
    as_of_date: str
    market_state: str
    confidence: float
    decision_summary: str
    key_evidence: list[str]
    failure_assumption: str
    risk_budget: dict[str, float] | None


class BacktestRecord(TypedDict):
    date: str
    group: str
    beginning_nav: float
    pre_trade_nav: float
    ending_nav: float
    gross_return: float
    net_return: float
    cost_amount: float
    cost_rate: float
    turnover: float
    asset_returns: AssetValues
    actual_weights: AssetValues
    effective_budget: AssetValues


MARKET_FEATURE_WINDOWS = {
    "asset_return_20d": ("return_20d", False),
    "asset_return_60d": ("return_60d", False),
    "asset_volatility_20d": ("volatility_20d", True),
    "asset_volatility_60d": ("volatility_60d", True),
    "asset_volatility_250d": ("volatility_250d", True),
}

CORRELATION_WINDOWS = {
    "corr_60d": "corr_60d",
    "corr_250d": "corr_250d",
}

WINDOW_STATUS_KEYS = {
    item[0]
    for item in MARKET_FEATURE_WINDOWS.values()
} | set(CORRELATION_WINDOWS.values())

MARKET_STATES = {
    "normal",
    "risk_elevated",
    "risk_transition",
    "uncertain",
}


def is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def validate_date(date_string: str) -> bool:
    try:
        return (
            isinstance(date_string, str)
            and date.fromisoformat(date_string).isoformat() == date_string
        )
    except (ValueError, TypeError):
        return False


def _has_asset_codes(values: dict) -> bool:
    return isinstance(values, dict) and set(values) == set(ASSET_CODES)


def validate_asset_values(values: dict) -> bool:
    return _has_asset_codes(values) and all(
        is_finite_number(value)
        for value in values.values()
    )


def _validate_market_feature_values(
    values: dict,
    non_negative: bool,
) -> bool:
    if not _has_asset_codes(values):
        return False

    for value in values.values():
        if value is None:
            continue
        if not is_finite_number(value):
            return False
        if non_negative and value < 0:
            return False

    return True


def validate_weights(weights: dict) -> bool:
    if not validate_asset_values(weights):
        return False

    values = list(weights.values())

    if not all(
        WEIGHT_MIN <= value <= WEIGHT_MAX
        for value in values
    ):
        return False

    return abs(sum(values) - 1.0) <= WEIGHT_SUM_TOLERANCE


def validate_risk_budget(budget: dict) -> bool:
    return (
        validate_weights(budget)
        and all(value > 0 for value in budget.values())
    )


def _validate_window_status(data_quality: dict) -> bool:
    if not isinstance(data_quality, dict):
        return False

    status = data_quality.get("window_status")
    insufficient = data_quality.get("insufficient_windows")

    if (
        not isinstance(status, dict)
        or set(status) != WINDOW_STATUS_KEYS
        or not isinstance(insufficient, list)
        or len(insufficient) != len(set(insufficient))
    ):
        return False

    for key, item in status.items():
        if (
            not isinstance(item, dict)
            or not {"required", "available", "sufficient", "reason"}
            .issubset(item)
            or not isinstance(item["required"], int)
            or isinstance(item["required"], bool)
            or item["required"] <= 0
            or not isinstance(item["available"], int)
            or isinstance(item["available"], bool)
            or item["available"] < 0
            or not isinstance(item["sufficient"], bool)
            or not isinstance(item["reason"], str)
            or not item["reason"]
        ):
            return False

        if item["sufficient"] != (item["reason"] == "ok"):
            return False

    expected_insufficient = {
        key
        for key, item in status.items()
        if not item["sufficient"]
    }

    return set(insufficient) == expected_insufficient


def _validate_correlation_matrix(matrix) -> tuple[bool, bool]:
    if matrix is None:
        return True, True

    if not _has_asset_codes(matrix):
        return False, False

    missing = False

    for row_code, row in matrix.items():
        if not _has_asset_codes(row):
            return False, False

        for column_code, value in row.items():
            if value is None:
                missing = True
                continue

            if (
                not is_finite_number(value)
                or not -1.0 <= value <= 1.0
                or (
                    row_code == column_code
                    and abs(value - 1.0) > 1e-12
                )
            ):
                return False, False

    for first in ASSET_CODES:
        for second in ASSET_CODES:
            left = matrix[first][second]
            right = matrix[second][first]

            if (left is None) != (right is None):
                return False, False

            if (
                left is not None
                and abs(left - right) > 1e-12
            ):
                return False, False

    return True, missing


def validate_market_input(market_data: dict) -> bool:
    required_fields = {
        "as_of_date",
        "market_features",
        "asset_correlations",
        "data_quality",
    }

    if (
        not isinstance(market_data, dict)
        or set(market_data) != required_fields
        or not validate_date(market_data["as_of_date"])
    ):
        return False

    features = market_data["market_features"]
    correlations = market_data["asset_correlations"]
    quality = market_data["data_quality"]

    if (
        not isinstance(features, dict)
        or set(features) != set(MARKET_FEATURE_WINDOWS)
        or not isinstance(correlations, dict)
        or set(correlations) != set(CORRELATION_WINDOWS)
        or not _validate_window_status(quality)
        or quality.get("no_lookahead") is not True
        or not isinstance(quality.get("sample_days_to_date"), int)
        or isinstance(quality.get("sample_days_to_date"), bool)
        or quality["sample_days_to_date"] < 1
    ):
        return False

    if (
        "as_of_date" in quality
        and quality["as_of_date"] != market_data["as_of_date"]
    ):
        return False

    status = quality["window_status"]

    for field, (window_key, non_negative) in (
        MARKET_FEATURE_WINDOWS.items()
    ):
        values = features[field]

        if not _validate_market_feature_values(
            values,
            non_negative,
        ):
            return False

        has_missing = any(
            value is None
            for value in values.values()
        )

        if status[window_key]["sufficient"] == has_missing:
            return False

    for field, window_key in CORRELATION_WINDOWS.items():
        valid, has_missing = _validate_correlation_matrix(
            correlations[field]
        )

        if (
            not valid
            or status[window_key]["sufficient"] == has_missing
        ):
            return False

    return True


def validate_model_decision(decision: dict) -> bool:
    required_fields = {
        "decision_status",
        "as_of_date",
        "market_state",
        "confidence",
        "decision_summary",
        "key_evidence",
        "failure_assumption",
        "risk_budget",
    }

    if (
        not isinstance(decision, dict)
        or set(decision) != required_fields
        or decision["decision_status"]
        not in {"ok", "insufficient_data"}
        or not validate_date(decision["as_of_date"])
        or decision["market_state"] not in MARKET_STATES
        or not is_finite_number(decision["confidence"])
        or not 0 <= decision["confidence"] <= 1
        or not isinstance(decision["decision_summary"], str)
        or not decision["decision_summary"].strip()
        or not isinstance(decision["failure_assumption"], str)
        or not decision["failure_assumption"].strip()
        or not isinstance(decision["key_evidence"], list)
        or not decision["key_evidence"]
        or not all(
            isinstance(item, str) and item.strip()
            for item in decision["key_evidence"]
        )
    ):
        return False

    if decision["decision_status"] == "ok":
        return validate_risk_budget(decision["risk_budget"])

    return (
        decision["market_state"] == "uncertain"
        and decision["confidence"] == 0
        and decision["risk_budget"] is None
    )


if __name__ == "__main__":
    test_weights = {
        "510300": 0.35,
        "511010": 0.50,
        "518880": 0.15,
    }

    test_decision = {
        "decision_status": "ok",
        "as_of_date": "2014-11-28",
        "market_state": "normal",
        "confidence": 0.85,
        "decision_summary": "Test decision",
        "key_evidence": ["Test evidence"],
        "failure_assumption": "Test assumption",
        "risk_budget": test_weights,
    }

    print("Asset codes:", ASSET_CODES)
    print("Weight validation:", validate_weights(test_weights))
    print("Date validation:", validate_date("2014-11-28"))
    print(
        "Model decision validation:",
        validate_model_decision(test_decision),
    )
