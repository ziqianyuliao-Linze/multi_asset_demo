
"""
Fallback mechanism for the LLM decision module.

Responsibilities:
1. Handle API failures.
2. Handle invalid model responses.
3. Handle insufficient market data.
4. Preserve previous effective risk budget.
5. Preserve actual portfolio holdings.
6. Handle first-decision warm start.

Important:
Risk budget and portfolio weights are different quantities.
Never substitute one for the other.
"""

from datetime import datetime, timezone

from interfaces.data_schema import (
    validate_model_decision,
    validate_risk_budget,
    validate_weights,
)


# ==================================================
# 1. Fallback action types
# ==================================================

MODEL_DECISION = "model_decision"
HOLD_PREVIOUS = "hold_previous"
HOLD_WARM_START = "hold_warm_start"


# ==================================================
# 2. Fallback resolver
# ==================================================

def resolve_fallback(
    decision: dict | None,
    expected_date: str,
    previous_effective_budget: dict | None,
    current_weights: dict | None,
    warm_start_weights: dict | None,
    failure_reason: str | None = None,
) -> dict:
    """
    Determine whether to use a new model decision or fallback.

    Parameters
    ----------
    decision:
        Model response.

    expected_date:
        Expected decision date.

    previous_effective_budget:
        Previous valid effective risk budget.

    current_weights:
        Actual current portfolio weights.

    warm_start_weights:
        Predefined initial portfolio weights.

    failure_reason:
        Optional error description.

    Returns
    -------
    dict:
        Structured fallback result.
    """

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------
    # 1. Check model decision
    # --------------------------------------------------

    valid_decision = (
        isinstance(decision, dict)
        and validate_model_decision(decision)
        and decision["as_of_date"] == expected_date
        and decision["decision_status"] == "ok"
    )

    if valid_decision:

        return {
            "action": MODEL_DECISION,
            "decision_date": expected_date,
            "risk_budget": decision["risk_budget"],
            "target_weights": None,
            "rebalance_required": True,
            "fallback_used": False,
            "failure_reason": None,
            "timestamp": timestamp,
        }

    # --------------------------------------------------
    # 2. Determine failure reason
    # --------------------------------------------------

    if failure_reason is None:

        if decision is None:
            failure_reason = "model_response_missing"

        elif not isinstance(decision, dict):
            failure_reason = "invalid_response_type"

        elif not validate_model_decision(decision):
            failure_reason = "invalid_model_decision"

        elif decision["as_of_date"] != expected_date:
            failure_reason = "decision_date_mismatch"

        else:
            failure_reason = "insufficient_data"

    # --------------------------------------------------
    # 3. Validate previous effective budget
    # --------------------------------------------------

    previous_budget_valid = (
        previous_effective_budget is not None
        and validate_risk_budget(
            previous_effective_budget
        )
    )

    # --------------------------------------------------
    # 4. Validate actual current holdings
    # --------------------------------------------------

    current_weights_valid = (
        current_weights is not None
        and validate_weights(current_weights)
    )

    if current_weights_valid:

        preserved_weights = dict(current_weights)

        action = HOLD_PREVIOUS

    else:

        # --------------------------------------------------
        # 5. First-decision warm start
        # --------------------------------------------------

        warm_start_valid = (
            warm_start_weights is not None
            and validate_weights(warm_start_weights)
        )

        if not warm_start_valid:
            raise ValueError(
                "Fallback failed: no valid current holdings "
                "or warm-start weights are available."
            )

        preserved_weights = dict(warm_start_weights)

        action = HOLD_WARM_START

    # --------------------------------------------------
    # 6. Return fallback result
    # --------------------------------------------------

    return {
        "action": action,
        "decision_date": expected_date,

        "risk_budget": (
            dict(previous_effective_budget)
            if previous_budget_valid
            else None
        ),

        "target_weights": preserved_weights,

        "rebalance_required": False,
        "fallback_used": True,

        "failure_reason": failure_reason,

        "timestamp": timestamp,
    }


# ==================================================
# 7. Self-test
# ==================================================

if __name__ == "__main__":

    print("=" * 50)
    print("LLM Fallback Self-Test")
    print("=" * 50)

    previous_budget = {
        "510300": 0.40,
        "511010": 0.40,
        "518880": 0.20,
    }

    current_weights = {
        "510300": 0.35,
        "511010": 0.50,
        "518880": 0.15,
    }

    warm_start_weights = {
        "510300": 1 / 3,
        "511010": 1 / 3,
        "518880": 1 / 3,
    }

    # Test 1: API failure with previous holdings.
    result_1 = resolve_fallback(
        decision=None,
        expected_date="2026-11-20",
        previous_effective_budget=previous_budget,
        current_weights=current_weights,
        warm_start_weights=warm_start_weights,
        failure_reason="api_timeout",
    )

    print(
        "Previous holdings preserved:",
        result_1["action"] == HOLD_PREVIOUS
        and result_1["target_weights"] == current_weights
    )

    # Test 2: First decision failure.
    result_2 = resolve_fallback(
        decision=None,
        expected_date="2026-11-20",
        previous_effective_budget=None,
        current_weights=None,
        warm_start_weights=warm_start_weights,
        failure_reason="first_decision_failed",
    )

    print(
        "Warm start preserved:",
        result_2["action"] == HOLD_WARM_START
        and result_2["target_weights"] == warm_start_weights
    )

    # Test 3: Previous budget preserved.
    print(
        "Previous risk budget preserved:",
        result_1["risk_budget"] == previous_budget
    )

    # Test 4: No new rebalance.
    print(
        "Rebalance disabled:",
        result_1["rebalance_required"] is False
    )

    print("=" * 50)
