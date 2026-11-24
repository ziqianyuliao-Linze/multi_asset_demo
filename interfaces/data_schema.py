
"""
Shared data schemas for the multi-asset allocation project.

Project: Challenge Cup
Version: 1.0
Python: 3.11
"""

from datetime import date
from math import isfinite
from typing import TypedDict, Literal, NotRequired


# ==================================================
# 1. Unified asset definitions
# ==================================================

ASSET_CODES = (
    "510300",
    "511010",
    "518880",
)

ASSET_NAMES = {
    "510300": "stock",
    "511010": "bond",
    "518880": "gold",
}

AssetCode = Literal["510300", "511010", "518880"]


class AssetValues(TypedDict):
    """Values corresponding to the three assets."""
    stock: float
    bond: float
    gold: float


# ==================================================
# 2. Unified experiment parameters
# ==================================================

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


# ==================================================
# 3. Market features: Module 2 -> Modules 1 and 4
# ==================================================

class MarketFeatures(TypedDict):
    asset_return_20d: AssetValues
    asset_return_60d: AssetValues
    asset_volatility_20d: AssetValues
    asset_volatility_60d: AssetValues
    asset_volatility_250d: AssetValues


class MarketDataInput(TypedDict):
    as_of_date: str
    market_features: MarketFeatures
    asset_correlations: dict
    data_quality: dict


# ==================================================
# 4. Risk feedback: Module 4 -> Module 1
# ==================================================

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
    residual: AssetValues


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


# ==================================================
# 5. Model decision: Module 1 -> Module 5
# ==================================================

class ModelDecision(TypedDict):
    decision_status: str
    as_of_date: str
    market_state: str
    confidence: float
    decision_summary: str
    key_evidence: list[str]
    failure_assumption: str
    risk_budget: dict[str, float]


# ==================================================
# 6. Backtest daily record: Module 5
# ==================================================

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


# ==================================================
# 7. Validation functions
# ==================================================

def validate_date(date_string: str) -> bool:
    """Validate date format: YYYY-MM-DD."""
    try:
        return (
            isinstance(date_string, str)
            and date.fromisoformat(date_string).isoformat()
            == date_string
        )
    except (ValueError, TypeError):
        return False


def validate_asset_values(values: dict) -> bool:
    """Validate values for all three assets."""

    if not isinstance(values, dict):
        return False

    if set(values.keys()) != set(ASSET_NAMES.values()):
        return False

    return all(
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and isfinite(v)
        for v in values.values()
    )


def validate_weights(weights: dict) -> bool:
    """Validate portfolio weights."""

    if not isinstance(weights, dict):
        return False

    if set(weights.keys()) != set(ASSET_CODES):
        return False

    values = list(weights.values())

    if not all(
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and isfinite(v)
        for v in values
    ):
        return False

    if not all(
        WEIGHT_MIN <= v <= WEIGHT_MAX
        for v in values
    ):
        return False

    return abs(sum(values) - 1.0) <= WEIGHT_SUM_TOLERANCE


def validate_model_decision(decision: dict) -> bool:
    """Validate the eight-field model response."""

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

    if not isinstance(decision, dict):
        return False

    if set(decision.keys()) != required_fields:
        return False

    if not validate_date(decision["as_of_date"]):
        return False

    confidence = decision["confidence"]

    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        return False

    if not isinstance(decision["key_evidence"], list):
        return False

    return validate_weights(decision["risk_budget"])


# ==================================================
# 8. Basic interface tests
# ==================================================

if __name__ == "__main__":

    test_weights = {
        "510300": 0.35,
        "511010": 0.50,
        "518880": 0.15,
    }

    test_decision = {
        "decision_status": "success",
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
        validate_model_decision(test_decision)
    )
