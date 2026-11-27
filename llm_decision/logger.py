
"""
Decision logger for the multi-asset allocation project.

Responsibilities:
1. Record experiment group.
2. Record model information.
3. Record decision results.
4. Record API usage and latency.
5. Record fallback events.
6. Append records to JSONL.

Security:
- Never record API keys.
- Never record environment variables.
- Do not expose experiment group to the LLM.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ==================================================
# 1. Default log configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_LOG_FILE = BASE_DIR / "decision_log.jsonl"


# ==================================================
# 2. Build input summary
# ==================================================

def build_input_summary(market_data: dict) -> dict:
    """
    Extract a compact and auditable market input summary.
    """

    if not isinstance(market_data, dict):
        raise TypeError("Market data must be a dictionary.")

    return {
        "as_of_date": market_data.get("as_of_date"),
        "data_quality": market_data.get("data_quality"),
        "asset_codes": list(
            market_data.get("market_features", {})
            .get("asset_return_20d", {})
            .keys()
        ),
        "feedback_available": (
            "feedback" in market_data
        ),
    }


# ==================================================
# 3. Build decision log record
# ==================================================

def build_log_record(
    experiment_group: str,
    market_data: dict,
    decision: dict | None,
    metadata: dict | None = None,
    fallback_result: dict | None = None,
) -> dict:
    """
    Construct one decision log record.

    Experiment group is recorded internally.
    It is not part of the model input.
    """

    if experiment_group not in ["A", "B", "C", "D"]:
        raise ValueError("Invalid experiment group.")

    metadata = metadata or {}

    if decision is not None and not isinstance(decision, dict):
        raise TypeError("Decision must be a dictionary or None.")

    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary.")

    if fallback_result is not None and not isinstance(
        fallback_result, dict
    ):
        raise TypeError("Fallback result must be a dictionary.")

    fallback_used = (
        fallback_result is not None
        and fallback_result.get("fallback_used") is True
    )

    if decision is None:
        decision_status = "no_response"
        market_state = None
        risk_budget = None
        decision_summary = None
        key_evidence = []
    else:
        decision_status = decision.get(
            "decision_status", "unknown"
        )
        market_state = decision.get("market_state")
        risk_budget = decision.get("risk_budget")
        decision_summary = decision.get("decision_summary")
        key_evidence = decision.get("key_evidence", [])

    fallback_action = (
        fallback_result.get("action")
        if fallback_result else None
    )

    fallback_reason = (
        fallback_result.get("failure_reason")
        if fallback_result else None
    )

    target_weights = (
        fallback_result.get("target_weights")
        if fallback_result else None
    )

    input_summary = build_input_summary(market_data)

    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "as_of_date": market_data.get("as_of_date"),

        "experiment_group": experiment_group,

        "model": {
            "name": metadata.get("model"),
            "attempt": metadata.get("attempt"),
            "finish_reason": metadata.get("finish_reason"),
        },

        "input_summary": input_summary,

        "decision": {
            "status": decision_status,
            "market_state": market_state,
            "risk_budget": risk_budget,
            "summary": decision_summary,
            "key_evidence": key_evidence,
        },

        "api_usage": {
            "elapsed_seconds": metadata.get(
                "elapsed_seconds"
            ),
            "prompt_tokens": metadata.get(
                "prompt_tokens"
            ),
            "completion_tokens": metadata.get(
                "completion_tokens"
            ),
            "total_tokens": metadata.get(
                "total_tokens"
            ),
        },

        "fallback": {
            "used": fallback_used,
            "action": fallback_action,
            "failure_reason": fallback_reason,
            "target_weights": target_weights,
        },
    }


# ==================================================
# 4. Append JSONL log
# ==================================================

def write_decision_log(
    record: dict,
    log_file: str | Path | None = None,
) -> Path:
    """
    Append one JSON record to a JSONL file.

    Each line represents one decision event.
    """

    if not isinstance(record, dict):
        raise TypeError("Log record must be a dictionary.")

    output_file = Path(
        log_file if log_file is not None
        else DEFAULT_LOG_FILE
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
    )

    with open(
        output_file,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(serialized + "\n")

        f.flush()

    return output_file


# ==================================================
# 5. Complete logging interface
# ==================================================

def log_decision(
    experiment_group: str,
    market_data: dict,
    decision: dict | None,
    metadata: dict | None = None,
    fallback_result: dict | None = None,
    log_file: str | Path | None = None,
) -> Path:
    """
    Build and save a complete decision log.
    """

    record = build_log_record(
        experiment_group=experiment_group,
        market_data=market_data,
        decision=decision,
        metadata=metadata,
        fallback_result=fallback_result,
    )

    return write_decision_log(
        record=record,
        log_file=log_file,
    )


# ==================================================
# 6. Self-test
# ==================================================

if __name__ == "__main__":

    import tempfile

    print("=" * 50)
    print("Decision Logger Self-Test")
    print("=" * 50)

    test_market_data = {
        "as_of_date": "2026-11-20",
        "market_features": {
            "asset_return_20d": {
                "510300": -0.02,
                "511010": 0.005,
                "518880": 0.015,
            }
        },
        "data_quality": {
            "source": "synthetic_test",
        },
    }

    test_decision = {
        "decision_status": "ok",
        "as_of_date": "2026-11-20",
        "market_state": "normal",
        "confidence": 0.85,
        "decision_summary": "Test decision",
        "key_evidence": ["Test evidence"],
        "failure_assumption": "Test assumption",
        "risk_budget": {
            "510300": 0.40,
            "511010": 0.40,
            "518880": 0.20,
        },
    }

    test_metadata = {
        "model": "deepseek-flash",
        "attempt": 1,
        "finish_reason": "stop",
        "elapsed_seconds": 2.5,
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "total_tokens": 700,
    }

    with tempfile.TemporaryDirectory() as temp_dir:

        test_log = Path(temp_dir) / "test_log.jsonl"

        saved_path = log_decision(
            experiment_group="A",
            market_data=test_market_data,
            decision=test_decision,
            metadata=test_metadata,
            log_file=test_log,
        )

        with open(
            saved_path,
            "r",
            encoding="utf-8",
        ) as f:

            saved_record = json.loads(f.readline())

        print(
            "JSONL write:",
            saved_path.exists()
        )

        print(
            "Group recorded:",
            saved_record["experiment_group"] == "A"
        )

        print(
            "Risk budget recorded:",
            saved_record["decision"]["risk_budget"]
            == test_decision["risk_budget"]
        )

        print(
            "Token usage recorded:",
            saved_record["api_usage"]["total_tokens"] == 700
        )

        print(
            "Fallback default:",
            saved_record["fallback"]["used"] is False
        )

        print(
            "API key excluded:",
            "api_key" not in json.dumps(saved_record).lower()
        )

    print("=" * 50)
