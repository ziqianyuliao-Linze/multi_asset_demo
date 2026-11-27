"""Import-safe compatibility entry point for the LLM decision module.

New code should import ``generate_decision`` from ``llm_decision.decision``
and ``build_model_input`` from ``llm_decision.validator`` directly.
"""

import json
from pathlib import Path

from llm_decision.decision import generate_decision
from llm_decision.validator import build_model_input


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "decision_input.json"
OUTPUT_FILE = BASE_DIR / "decision_output.json"


def build_user_payload(
    schema_input: dict,
    group: str,
    feedback: dict | None = None,
) -> dict:
    """Build the group-authorized JSON payload sent to the model."""

    return build_model_input(
        market_data=schema_input,
        group=group,
        feedback=feedback,
    )


def load_input(path: Path = INPUT_FILE) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_result(result: dict, path: Path = OUTPUT_FILE) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def main() -> None:
    market_data = load_input()
    result, metadata = generate_decision(
        market_data,
        group="A",
        feedback=None,
    )
    save_result(result)
    print(
        json.dumps(
            {
                "output_file": str(OUTPUT_FILE),
                "model": metadata["model"],
                "decision_status": result["decision_status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
