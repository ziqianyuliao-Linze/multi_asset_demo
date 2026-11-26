
import copy
import json
import unittest
import tempfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai import APIConnectionError, APIStatusError

from llm_decision.client import DeepSeekTruncationError
from llm_decision.decision import generate_decision

from llm_decision.validator import (
    build_model_input,
    validate_risk_feedback,
    validate_decision_output,
)

from llm_decision.fallback import resolve_fallback

from llm_decision.logger import log_decision

from interfaces.data_schema import (
    validate_market_input,
    validate_model_decision,
)


BASE_DIR = Path(__file__).resolve().parent.parent


def load_market_data():
    with open(
        BASE_DIR / "decision_input.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


ASSET_VALUES = {
    "510300": 0.35,
    "511010": 0.50,
    "518880": 0.15,
}


RISK_BUDGET = {
    "510300": 0.40,
    "511010": 0.40,
    "518880": 0.20,
}


FEEDBACK_B = {
    "portfolio_return": {
        "d20": 0.012,
        "d60": 0.025,
    }
}


FEEDBACK_C = {
    **FEEDBACK_B,
    "portfolio_volatility": {
        "d20": 0.12,
        "d60": 0.14,
    },
    "current_drawdown": -0.03,
    "max_drawdown": -0.08,
    "cvar_95": 0.025,
    "trigger_flags": {
        "drawdown": False,
        "volatility": False,
        "correlation": False,
    },
}


FEEDBACK_D = {
    **FEEDBACK_C,
    "risk_contribution": ASSET_VALUES,
    "budget_deviation": {
        "510300": -0.05,
        "511010": 0.05,
        "518880": 0.00,
    },
    "drawdown_attribution": ASSET_VALUES,
    "risk_change_attribution": {
        "weight": ASSET_VALUES,
        "vol": ASSET_VALUES,
        "corr": ASSET_VALUES,
        "residual": 0.0,
    },
    "concentration_state": "moderate",
}


VALID_DECISION = {
    "decision_status": "ok",
    "as_of_date": "2026-11-20",
    "market_state": "normal",
    "confidence": 0.85,
    "decision_summary": "Offline test decision",
    "key_evidence": ["Test evidence"],
    "failure_assumption": "Test assumption",
    "risk_budget": RISK_BUDGET,
}


class TestLLMDecisionOffline(unittest.TestCase):

    def setUp(self):
        self.market_data = load_market_data()

    def test_market_input_uses_security_codes(self):
        self.assertTrue(validate_market_input(self.market_data))

        for values in self.market_data["market_features"].values():
            self.assertEqual(set(values), set(ASSET_VALUES))

        for matrix in self.market_data["asset_correlations"].values():
            self.assertEqual(set(matrix), set(ASSET_VALUES))
            for row in matrix.values():
                self.assertEqual(set(row), set(ASSET_VALUES))

    def test_market_input_rejects_asset_names(self):
        invalid = copy.deepcopy(self.market_data)
        values = invalid["market_features"]["asset_return_20d"]
        values["stock"] = values.pop("510300")
        self.assertFalse(validate_market_input(invalid))

    def test_market_input_rejects_non_strict_date(self):
        for invalid_date in (
            "2026-11-20T16:00:00",
            "2026-11-20garbage",
        ):
            with self.subTest(invalid_date=invalid_date):
                invalid = copy.deepcopy(self.market_data)
                invalid["as_of_date"] = invalid_date
                self.assertFalse(validate_market_input(invalid))

    def test_market_input_accepts_reported_missing_window(self):
        incomplete = copy.deepcopy(self.market_data)
        values = incomplete["market_features"][
            "asset_volatility_250d"
        ]
        for code in values:
            values[code] = None

        incomplete["asset_correlations"]["corr_250d"] = None
        quality = incomplete["data_quality"]
        quality["window_status"]["volatility_250d"].update(
            sufficient=False,
            reason="insufficient_window",
        )
        quality["window_status"]["corr_250d"].update(
            sufficient=False,
            reason="insufficient_window",
        )
        quality["insufficient_windows"] = [
            "volatility_250d",
            "corr_250d",
        ]

        self.assertTrue(validate_market_input(incomplete))

    def test_market_input_rejects_unreported_null(self):
        invalid = copy.deepcopy(self.market_data)
        invalid["asset_correlations"]["corr_60d"][
            "510300"
        ]["511010"] = None
        invalid["asset_correlations"]["corr_60d"][
            "511010"
        ]["510300"] = None

        self.assertFalse(validate_market_input(invalid))

    def test_model_input_removes_future_quality_metadata(self):
        result = build_model_input(self.market_data, "A")

        self.assertNotIn("rows", result["data_quality"])
        self.assertNotIn("end_date", result["data_quality"])
        self.assertNotIn("source", result["data_quality"])
        self.assertEqual(
            result["data_quality"]["sample_days_to_date"],
            3236,
        )

    def test_insufficient_data_decision(self):
        decision = {
            **VALID_DECISION,
            "decision_status": "insufficient_data",
            "market_state": "uncertain",
            "confidence": 0,
            "risk_budget": None,
        }

        self.assertTrue(validate_model_decision(decision))

        decision["risk_budget"] = RISK_BUDGET
        self.assertFalse(validate_model_decision(decision))

    def test_group_A(self):
        result = build_model_input(
            self.market_data, "A"
        )

        self.assertNotIn("feedback", result)
        self.assertNotIn("experiment_group", result)

    def test_group_B(self):
        result = build_model_input(
            self.market_data, "B", FEEDBACK_B
        )

        self.assertEqual(
            result["feedback"], FEEDBACK_B
        )

    def test_group_C(self):
        result = build_model_input(
            self.market_data, "C", FEEDBACK_C
        )

        self.assertEqual(
            result["feedback"], FEEDBACK_C
        )

    def test_group_D(self):
        result = build_model_input(
            self.market_data, "D", FEEDBACK_D
        )

        self.assertEqual(
            result["feedback"], FEEDBACK_D
        )

    def test_group_A_rejects_feedback(self):
        with self.assertRaises(ValueError):
            build_model_input(
                self.market_data,
                "A",
                FEEDBACK_B,
            )

    def test_group_B_rejects_C_feedback(self):
        self.assertFalse(
            validate_risk_feedback(
                FEEDBACK_C,
                "B",
            )
        )

    def test_group_C_rejects_D_feedback(self):
        self.assertFalse(
            validate_risk_feedback(
                FEEDBACK_D,
                "C",
            )
        )

    def test_valid_model_decision(self):
        self.assertTrue(
            validate_decision_output(
                VALID_DECISION,
                "2026-11-20",
            )
        )

    def test_invalid_model_date(self):
        decision = dict(VALID_DECISION)
        decision["as_of_date"] = "2026-11-19"

        self.assertFalse(
            validate_decision_output(
                decision,
                "2026-11-20",
            )
        )

    def test_fallback_preserves_holdings(self):
        result = resolve_fallback(
            decision=None,
            expected_date="2026-11-20",
            previous_effective_budget=RISK_BUDGET,
            current_weights=ASSET_VALUES,
            warm_start_weights=ASSET_VALUES,
            failure_reason="api_timeout",
        )

        self.assertTrue(result["fallback_used"])
        self.assertFalse(result["rebalance_required"])
        self.assertEqual(
            result["target_weights"],
            ASSET_VALUES,
        )
        self.assertEqual(
            result["risk_budget"],
            RISK_BUDGET,
        )

    def test_warm_start(self):
        result = resolve_fallback(
            decision=None,
            expected_date="2026-11-20",
            previous_effective_budget=None,
            current_weights=None,
            warm_start_weights={
                "510300": 1 / 3,
                "511010": 1 / 3,
                "518880": 1 / 3,
            },
            failure_reason="first_decision_failed",
        )

        self.assertEqual(
            result["action"],
            "hold_warm_start",
        )

    def test_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            log_path = Path(temp_dir) / "test.jsonl"

            log_decision(
                experiment_group="A",
                market_data=self.market_data,
                decision=VALID_DECISION,
                metadata={"model": "deepseek-flash"},
                log_file=log_path,
            )

            with open(
                log_path,
                "r",
                encoding="utf-8",
            ) as f:
                record = json.loads(f.readline())

            self.assertEqual(
                record["experiment_group"],
                "A",
            )

            self.assertEqual(
                record["decision"]["risk_budget"],
                RISK_BUDGET,
            )


class TestDeepSeekDecisionPipeline(unittest.TestCase):

    def setUp(self):
        self.market_data = load_market_data()
        self.decision = {
            **VALID_DECISION,
            "as_of_date": self.market_data["as_of_date"],
        }
        self.sdk = self.enterContext(patch("llm_decision.client.OpenAI"))
        self.enterContext(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test"}))
        self.sleep = self.enterContext(patch("llm_decision.client.time.sleep"))
        self.create = self.sdk.return_value.chat.completions.create
        self.create.return_value = self.response(json.dumps(self.decision))

    @staticmethod
    def response(content, finish_reason="stop"):
        return SimpleNamespace(
            model="deepseek-flash",
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(content=content),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=160,
                total_tokens=260,
            ),
        )

    def test_valid_response_through_project_pipeline(self):
        decision, metadata = generate_decision(self.market_data, group="A")

        self.assertEqual(decision, self.decision)
        self.sdk.assert_called_once_with(
            api_key="offline-test",
            base_url="https://api.deepseek.com",
            timeout=60,
            max_retries=0,
        )
        self.create.assert_called_once()
        request = self.create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 3072)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertFalse(request["stream"])
        self.assertEqual(
            json.loads(request["messages"][1]["content"]),
            build_model_input(self.market_data, "A"),
        )
        self.assertEqual(metadata["model"], "deepseek-flash")
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertEqual(metadata["attempt"], 1)
        self.assertEqual(metadata["prompt_tokens"], 100)
        self.assertEqual(metadata["completion_tokens"], 160)
        self.assertEqual(metadata["total_tokens"], 260)

    def test_length_fails_after_one_request_even_with_valid_json(self):
        self.create.return_value = self.response(
            json.dumps(self.decision), finish_reason="length"
        )

        with self.assertRaisesRegex(
            DeepSeekTruncationError,
            r"finish_reason=length, max_tokens=3072, .*completion_tokens=160, attempt=1",
        ):
            generate_decision(self.market_data, group="A")

        self.create.assert_called_once()
        self.sleep.assert_not_called()

    def test_invalid_json_fails_without_echoing_response(self):
        self.create.return_value = self.response('{"private-content":')

        with self.assertRaisesRegex(ValueError, "invalid JSON") as error:
            generate_decision(self.market_data, group="A")

        self.assertNotIn("private-content", str(error.exception))
        self.create.assert_called_once()
        self.sleep.assert_not_called()

    def test_blank_and_non_object_json_are_rejected(self):
        for content, error in [("   ", "empty JSON"), ("[]", "JSON object")]:
            with self.subTest(content=content):
                self.create.reset_mock()
                self.create.return_value = self.response(content)
                with self.assertRaisesRegex(ValueError, error):
                    generate_decision(self.market_data, group="A")
                self.create.assert_called_once()

    def test_schema_and_date_failures_do_not_repeat_request(self):
        for invalid_fields in (
            {"as_of_date": "2000-01-01"},
            {"risk_budget": {"510300": 0.4, "511010": 0.4, "518880": 0.3}},
            {"unexpected_field": True},
        ):
            with self.subTest(invalid_fields=invalid_fields):
                self.create.reset_mock()
                self.create.return_value = self.response(
                    json.dumps({**self.decision, **invalid_fields})
                )
                with self.assertRaisesRegex(ValueError, "invalid decision"):
                    generate_decision(self.market_data, group="A")
                self.create.assert_called_once()

    def test_insufficient_balance_is_terminal_and_sanitized(self):
        self.create.side_effect = APIStatusError(
            "private-error-detail",
            response=SimpleNamespace(
                status_code=402,
                request=SimpleNamespace(headers={}),
                headers={},
                history=[],
                next_request=None,
            ),
            body=None,
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP 402.*insufficient balance") as error:
            generate_decision(self.market_data, group="A")

        self.assertNotIn("private-error-detail", str(error.exception))
        self.create.assert_called_once()
        self.sleep.assert_not_called()

    def test_transient_connection_error_is_retried(self):
        self.create.side_effect = [
            APIConnectionError(
                request=SimpleNamespace(headers={})
            ),
            self.create.return_value,
        ]

        decision, metadata = generate_decision(self.market_data, group="A")

        self.assertEqual(decision, self.decision)
        self.assertEqual(metadata["attempt"], 2)
        self.assertEqual(self.create.call_count, 2)
        self.sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
