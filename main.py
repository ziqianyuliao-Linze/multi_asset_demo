import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path


MOCK_RESPONSE = '{"weights": {"stock": 0.30, "bond": 0.50, "gold": 0.20}}'
PROJECT_DIR = Path(__file__).resolve().parent


def validate_weights(response, assets):
    try:
        result = json.loads(response, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError:
        raise ValueError("模型响应不是有效的 JSON") from None

    weights = result.get("weights") if isinstance(result, dict) else None
    if not isinstance(weights, dict) or set(weights) != set(assets):
        raise ValueError("权重必须对应股票、债券和黄金")

    for weight in weights.values():
        if not isinstance(weight, Decimal) or not weight.is_finite():
            raise ValueError("权重必须是有限数字")
        if not 0 <= weight <= Decimal("0.60"):
            raise ValueError("单项权重必须在 0 到 0.60 之间")

    if sum(weights.values()) != Decimal("1"):
        raise ValueError("权重总和必须等于 1")
    return {asset: float(weights[asset]) for asset in assets}


def log_decision(outcome):
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "mock",
        **outcome,
    }
    log_path = PROJECT_DIR / "decision_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def main(mock_response=MOCK_RESPONSE):
    input_path = PROJECT_DIR / "input.json"
    assets = json.loads(input_path.read_text(encoding="utf-8"))["assets"]
    if not isinstance(assets, list) or sorted(assets, key=str) != ["bond", "gold", "stock"]:
        raise SystemExit("输入文件必须列出股票、债券和黄金各一次。")

    print("待配置资产：", ", ".join(assets))
    try:
        weights = validate_weights(mock_response, assets)
    except ValueError as error:
        log_decision({"status": "rejected", "reason": str(error)})
        raise SystemExit(f"权重校验失败：{error}") from None

    log_decision({"status": "accepted", "weights": weights})
    print("权重校验通过：", json.dumps(weights))
    print("决策已写入 decision_log.jsonl")


if __name__ == "__main__":
    main()
