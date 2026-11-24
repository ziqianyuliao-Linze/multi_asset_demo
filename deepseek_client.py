
import os
import json
import math
from pathlib import Path
from openai import OpenAI

# ==========================================
# 1. 项目路径
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "decision_input.json"
OUTPUT_FILE = BASE_DIR / "decision_output.json"

MODEL_NAME = "deepseek-flash"

ASSETS = ["510300", "511010", "518880"]


# ==========================================
# 2. 初始化DeepSeek客户端
# ==========================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有找到DEEPSEEK_API_KEY，请先配置API Key。"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ==========================================
# 3. 基础Prompt
# ==========================================

SYSTEM_PROMPT = """
你是一个多资产配置研究项目中的风险预算决策模块。

研究资产：
510300：沪深300ETF
511010：国债ETF
518880：黄金ETF

你的任务是根据输入的市场信息，
生成下一期三类资产的目标风险预算。

决策规则：

1. 只能使用输入提供的信息。
2. 不得使用未来数据。
3. 综合考虑收益、波动率、相关性和可用反馈。
4. 区分资金权重与风险预算。
5. 只输出风险预算，不输出资金权重。
6. 不得做空、加杠杆或改变资产池。
7. 不得虚构缺失的数据。
8. 三项风险预算必须大于0且小于1。
9. 三项风险预算之和必须等于1。
10. 不得仅根据近期收益机械地调整预算。

market_state只能为：
normal
risk_elevated
risk_transition
uncertain

输出合法JSON，字段如下：

{
  "schema_version": "1.0",
  "as_of_date": "日期",
  "decision_status": "ok",
  "market_state": "normal",
  "risk_budget": {
    "510300": 0.40,
    "511010": 0.40,
    "518880": 0.20
  },
  "confidence": 0.80,
  "decision_summary": "决策摘要",
  "key_evidence": ["依据1", "依据2"],
  "failure_assumption": "可能失效的假设"
}

如果输入信息不足：
decision_status设为insufficient_data。

不得输出JSON以外的文字。
"""


# ==========================================
# 4. 读取输入数据
# ==========================================

def load_input():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ==========================================
# 5. 调用DeepSeek
# ==========================================

def call_deepseek(data):

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    data,
                    ensure_ascii=False
                )
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0,
        max_tokens=1200,

        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },

        stream=False
    )

    choice = response.choices[0]

    if choice.finish_reason != "stop":
        raise RuntimeError(
            f"模型未正常完成输出：{choice.finish_reason}"
        )

    content = choice.message.content

    if not content:
        raise ValueError("模型返回内容为空")

    result = json.loads(content)

    return result, response


# ==========================================
# 6. 校验模型输出
# ==========================================

def validate_result(result, input_data):

    if result.get("decision_status") != "ok":
        raise ValueError(
            "模型未能完成有效决策，请检查输入数据。"
        )

    if result.get("as_of_date") != input_data.get("as_of_date"):
        raise ValueError("决策日期与输入日期不一致")

    if result.get("market_state") not in [
        "normal",
        "risk_elevated",
        "risk_transition",
        "uncertain"
    ]:
        raise ValueError("市场状态字段不合法")

    budget = result.get("risk_budget")

    if not isinstance(budget, dict):
        raise ValueError("risk_budget必须是JSON对象")

    if set(budget.keys()) != set(ASSETS):
        raise ValueError("资产代码不完整或存在多余资产")

    values = []

    for asset in ASSETS:

        value = budget[asset]

        if isinstance(value, bool):
            raise ValueError("风险预算不能是布尔值")

        if not isinstance(value, (int, float)):
            raise ValueError("风险预算必须是数值")

        if not math.isfinite(value):
            raise ValueError("风险预算必须是有限数值")

        if not 0 < value < 1:
            raise ValueError("风险预算必须大于0且小于1")

        values.append(value)

    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("风险预算总和不等于1")

    confidence = result.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("置信度字段不合法")

    if not isinstance(result.get("key_evidence"), list):
        raise ValueError("key_evidence必须是列表")

    if not isinstance(result.get("decision_summary"), str):
        raise ValueError("decision_summary格式错误")

    if not isinstance(result.get("failure_assumption"), str):
        raise ValueError("failure_assumption格式错误")

    return True


# ==========================================
# 7. 保存决策结果
# ==========================================

def save_result(result):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 8. 主程序
# ==========================================

def main():

    print("=" * 50)
    print("Multi-Asset LLM Decision System")
    print("=" * 50)

    print("\n[1] 正在读取市场数据...")

    data = load_input()

    print("[2] 正在调用DeepSeek...")

    result, response = call_deepseek(data)

    print("[3] 正在校验模型输出...")

    validate_result(result, data)

    print("[4] 正在保存决策结果...")

    save_result(result)

    print("\n决策成功！")

    print("\n模型版本：", response.model)

    print("\n风险预算：")

    for asset in ASSETS:

        print(
            f"{asset}: "
            f"{result['risk_budget'][asset]:.2%}"
        )

    print("\n市场状态：", result["market_state"])

    print("决策理由：", result["decision_summary"])

    print("\n结果已保存至：", OUTPUT_FILE)

    print("\n本次使用模拟数据，仅验证API和程序接口。")


if __name__ == "__main__":
    main()
