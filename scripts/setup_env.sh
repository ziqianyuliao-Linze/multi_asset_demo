#!/usr/bin/env bash
# 一键搭环境：自动挑 Python（3.11+），建虚拟环境，用国内镜像装依赖，最后自检。
# 用法：bash scripts/setup_env.sh [虚拟环境目录，默认 .venv]
set -euo pipefail

VENV="${1:-.venv}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> 项目目录：$ROOT"

# 1. 找可用的 Python（3.11 起才支持本项目的数据结构写法）
CANDIDATES=(
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
  /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11
  /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11
  python3.13 python3.12 python3.11
)
PY=""
for c in "${CANDIDATES[@]}"; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
    major="${v%%.*}"; minor="${v##*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then PY="$(command -v "$c")"; break; fi
  fi
done

if [ -z "$PY" ]; then
  cat <<'MSG'
没有找到 Python 3.11 或更高版本。二选一：
  A. 装官方包：https://www.python.org/downloads/macos/ 选 Python 3.12.x → macOS 64-bit universal2 installer
  B. 已经有 Homebrew 的话：brew install python@3.12
装完重新运行本脚本即可。
MSG
  exit 1
fi

echo "==> 使用 Python：$PY（$("$PY" -V 2>&1)）"

# 2. 建/复用虚拟环境
if [ -d "$VENV" ]; then
  cur="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
  major="${cur%%.*}"; minor="${cur##*.}"
  if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
    echo "==> 复用已有环境 $VENV（Python $cur）"
  else
    echo "==> 旧环境是 Python $cur，太旧，重建 $VENV"
    rm -rf "$VENV"
    "$PY" -m venv "$VENV"
  fi
else
  echo "==> 新建 $VENV"
  "$PY" -m venv "$VENV"
fi

# 3. 装依赖（清华源，失败自动退回官方源）
echo "==> 安装依赖"
if ! "$VENV/bin/pip" install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 5; then
  "$VENV/bin/pip" install -q --upgrade pip
fi
if ! "$VENV/bin/pip" install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 5; then
  echo "==> 镜像失败，改用官方源重试"
  "$VENV/bin/pip" install -r requirements.txt --timeout 60 --retries 5
fi

# 4. 自检
echo "==> 自检"
"$VENV/bin/python" - <<'PY'
import sys
print("Python:", sys.version.split()[0])
sys.path.insert(0, ".")
import pandas, numpy, openai          # noqa: F401
import llm_decision.validator          # 关键：1 号的模块必须能导入
print("依赖与接口模块加载成功")
PY

echo
echo "完成。常用命令："
echo "  跑集成测试：PYTHONPATH=. $VENV/bin/python tests/test_risk_integration.py data/unified_returns.xlsx"
echo "  跑计算器自测：cd risk_metrics/calculators && ../../$VENV/bin/python test_risk.py"
