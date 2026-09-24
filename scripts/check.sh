#!/usr/bin/env bash
# 发布前本地质量闸门 (与 CI 一致): 语法 → 导入 → 测试 → 配图冒烟。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4 语法编译 =="
python3 -m compileall -q autoquantum scripts

echo "== 2/4 导入冒烟 =="
python3 -c "import autoquantum; print('autoquantum', autoquantum.__version__)"

echo "== 3/4 单元测试 =="
python3 -m unittest discover -s tests "$@"

echo "== 4/4 配图冒烟 =="
python3 -m unittest tests.test_book_figures

echo "== 全部通过 ✓ =="
