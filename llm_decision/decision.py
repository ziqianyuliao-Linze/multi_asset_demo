
"""
Main decision pipeline for multi-asset LLM allocation.

Responsibilities:
1. Validate market data.
2. Build experiment-specific model input.
3. Load unified system prompt.
4. Call DeepSeek API.
5. Validate model response.
6. Return decision and metadata.

Experiment group is managed internally.
It is never exposed to the LLM.
"""

from typing import Any

from llm_decision.client import DeepSeekClient
from llm_decision.prompt import get_system_prompt
from llm_decision.validator import (
    build_model_input,
    validate_decision_output,
)


# ==================================================
# 1. Main decision pipeline
# ==================================================

class LLMDecisionEngine:

    def __init__(self):

        self.client = DeepSeekClient()
        self.system_prompt = get_system_prompt()

    def decide(
        self,
        market_data: dict,
        group: str,
        feedback: dict | None = None,
    ) -> tuple[dict, dict]:

        # Step 1: Build authorized model input.
        model_input = build_model_input(
            market_data=market_data,
            group=group,
            feedback=feedback,
        )

        # Step 2: Call DeepSeek.
        result, metadata = self.client.generate(
            system_prompt=self.system_prompt,
            model_input=model_input,
        )

        # Step 3: Validate model response.
        expected_date = market_data["as_of_date"]

        if not validate_decision_output(
            decision=result,
            expected_date=expected_date,
        ):
            raise ValueError(
                "Model returned an invalid decision."
            )

        # Step 4: Return validated decision.
        return result, metadata


# ==================================================
# 2. Convenience function
# ==================================================

def generate_decision(
    market_data: dict,
    group: str,
    feedback: dict | None = None,
) -> tuple[dict, dict]:
    """
    Generate one validated LLM risk-budget decision.

    Returns:
        result: Eight-field model decision.
        metadata: API call information.
    """

    engine = LLMDecisionEngine()

    return engine.decide(
        market_data=market_data,
        group=group,
        feedback=feedback,
    )


# ==================================================
# 3. Import self-test
# ==================================================

if __name__ == "__main__":

    print("=" * 50)
    print("LLM Decision Engine")
    print("=" * 50)

    print("Decision engine imported successfully.")

    print("Available groups: A, B, C, D")

    print("Model output: Eight-field ModelDecision")

    print("Risk budget and portfolio weights are separated.")

    print("=" * 50)
