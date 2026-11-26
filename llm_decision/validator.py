
"""
Input and output validation for the LLM decision module.

Project: Multi-Asset Allocation
Version: 1.0
Python: 3.11

Responsibilities:
1. Validate market input.
2. Validate group-specific risk feedback.
3. Build LLM input without exposing experiment group.
4. Validate model output.
"""

from copy import deepcopy
from typing import Any

from interfaces.data_schema import (
    ASSET_CODES,
    validate_market_input,
    validate_model_decision,
    validate_asset_values,
    is_finite_number,
)


# ==================================================
# 1. Experiment feedback rules
# ==================================================

GROUP_FEEDBACK_FIELDS = {
    "A": set(),

    "B": {
        "portfolio_return",
    },

    "C": {
        "portfolio_return",
        "portfolio_volatility",
        "current_drawdown",
        "max_drawdown",
        "cvar_95",
        "trigger_flags",
    },

    "D": {
        "portfolio_return",
        "portfolio_volatility",
        "current_drawdown",
        "max_drawdown",
        "cvar_95",
        "trigger_flags",
        "risk_contribution",
        "budget_deviation",
        "drawdown_attribution",
        "risk_change_attribution",
        "concentration_state",
    },
}


# ==================================================
# 2. Basic validation
# ==================================================

def validate_experiment_group(group: str) -> bool:
    """Check whether the experiment group is valid."""

    return group in GROUP_FEEDBACK_FIELDS


def validate_portfolio_return(data: dict) -> bool:

    if not isinstance(data, dict):
        return False

    if set(data.keys()) != {"d20", "d60"}:
        return False

    return all(
        is_finite_number(value)
        for value in data.values()
    )


def validate_portfolio_volatility(data: dict) -> bool:

    if not isinstance(data, dict):
        return False

    if set(data.keys()) != {"d20", "d60"}:
        return False

    return all(
        is_finite_number(value) and value >= 0
        for value in data.values()
    )


def validate_trigger_flags(data: dict) -> bool:

    if not isinstance(data, dict):
        return False

    required = {
        "drawdown",
        "volatility",
        "correlation",
    }

    if set(data.keys()) != required:
        return False

    return all(
        isinstance(value, bool)
        for value in data.values()
    )


def validate_risk_change_attribution(data: dict) -> bool:

    if not isinstance(data, dict):
        return False

    required = {
        "weight",
        "vol",
        "corr",
        "residual",
    }

    if set(data.keys()) != required:
        return False

    for field in ["weight", "vol", "corr"]:

        if not validate_asset_values(data[field]):
            return False

    return is_finite_number(data["residual"])


# ==================================================
# 3. Risk feedback validation
# ==================================================

def validate_risk_feedback(
    feedback: dict,
    group: str,
) -> bool:
    """
    Validate risk feedback according to A/B/C/D rules.

    A: No portfolio feedback.
    B: Portfolio return.
    C: Return + risk metrics + warnings.
    D: C group + risk attribution.
    """

    if not validate_experiment_group(group):
        return False

    if not isinstance(feedback, dict):
        return False

    allowed_fields = GROUP_FEEDBACK_FIELDS[group]

    # Reject fields not authorized for this group.
    if not set(feedback.keys()).issubset(allowed_fields):
        return False

    # All fields required by this group must be present.
    if set(feedback.keys()) != allowed_fields:
        return False

    if group == "A":
        return feedback == {}

    if not validate_portfolio_return(
        feedback["portfolio_return"]
    ):
        return False

    if group in ["C", "D"]:

        if not validate_portfolio_volatility(
            feedback["portfolio_volatility"]
        ):
            return False

        for field in [
            "current_drawdown",
            "max_drawdown",
        ]:
            if not is_finite_number(feedback[field]):
                return False

        if (
            not is_finite_number(feedback["cvar_95"])
            or feedback["cvar_95"] < 0
        ):
            return False

        if not validate_trigger_flags(
            feedback["trigger_flags"]
        ):
            return False

    if group == "D":

        for field in [
            "risk_contribution",
            "budget_deviation",
            "drawdown_attribution",
        ]:
            if not validate_asset_values(
                feedback[field]
            ):
                return False

        if not validate_risk_change_attribution(
            feedback["risk_change_attribution"]
        ):
            return False

        if not isinstance(
            feedback["concentration_state"],
            str,
        ):
            return False

    return True


# ==================================================
# 4. Build LLM input
# ==================================================

def build_model_input(
    market_data: dict,
    group: str,
    feedback: dict | None = None,
) -> dict:
    """
    Build the final input passed to the LLM.

    The experiment group is managed internally.
    It is never included in the returned model input.
    """

    if not validate_experiment_group(group):
        raise ValueError(
            f"Invalid experiment group: {group}"
        )

    if not validate_market_input(market_data):
        raise ValueError(
            "Market input validation failed."
        )

    model_input = deepcopy(market_data)
    quality = model_input["data_quality"]
    model_input["data_quality"] = {
        key: quality[key]
        for key in (
            "as_of_date",
            "sample_days_to_date",
            "no_lookahead",
            "window_status",
            "insufficient_windows",
        )
        if key in quality
    }

    if group == "A":

        if feedback not in (None, {}):
            raise ValueError(
                "Group A must not receive risk feedback."
            )

        return model_input

    if feedback is None:
        raise ValueError(
            f"Group {group} requires risk feedback."
        )

    if not validate_risk_feedback(feedback, group):
        raise ValueError(
            f"Group {group} risk feedback is invalid."
        )

    model_input["feedback"] = deepcopy(feedback)

    return model_input


# ==================================================
# 5. Model output validation
# ==================================================

def validate_decision_output(
    decision: dict,
    expected_date: str,
) -> bool:
    """Validate model response and check decision date."""

    if not validate_model_decision(decision):
        return False

    if decision["as_of_date"] != expected_date:
        return False

    return True


# ==================================================
# 6. Self-test
# ==================================================

if __name__ == "__main__":

    print("=" * 50)
    print("LLM Validator Self-Test")
    print("=" * 50)

    print("Valid groups:")

    for group in ["A", "B", "C", "D"]:
        print(
            group,
            validate_experiment_group(group)
        )

    print("\nA group feedback:")

    print(
        validate_risk_feedback({}, "A")
    )

    print("\nUnauthorized feedback test:")

    print(
        validate_risk_feedback(
            {"portfolio_return": {"d20": 0.01, "d60": 0.02}},
            "A",
        )
    )

    print("\nInvalid group test:")

    print(
        validate_experiment_group("E")
    )

    print("=" * 50)
