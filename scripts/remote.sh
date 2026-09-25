#!/usr/bin/env bash
# AutoQuantum 远程计算工作流 (本地 Mac → c211 计算节点)
#
# 用法:
#   bash scripts/remote.sh sync              # 同步代码到服务器
#   bash scripts/remote.sh test              # 服务器上跑单元测试
#   bash scripts/remote.sh run "<命令>"      # 服务器上执行命令 (venv 内)
#   bash scripts/remote.sh fetch <远程目录>  # 取回结果到本地 results/
#
# 环境 (首次部署已完成): 代码 ~/AutoQuantum, venv ~/aqd-venv
# 节点: target-server (c211/xb002: 192核 EPYC 9654, 755G, A100-40G)
set -euo pipefail
HOST="${AQD_HOST:-target-server}"
REMOTE_DIR="${AQD_REMOTE_DIR:-\$HOME/AutoQuantum}"
R_PY="~/aqd-venv/bin/python"

case "${1:-}" in
  sync)
    tar czf /tmp/aqd.tar.gz --exclude .git --exclude "book/build" \
        --exclude "从势能面*" --exclude dist --exclude "__pycache__" \
        --exclude ".venv" --exclude results .
    scp -q /tmp/aqd.tar.gz "$HOST:~/"
    ssh "$HOST" "rm -rf ~/AutoQuantum/autoquantum ~/AutoQuantum/tests \
      ~/AutoQuantum/scripts && mkdir -p ~/AutoQuantum \
      && tar xzf ~/aqd.tar.gz -C ~/AutoQuantum && rm ~/aqd.tar.gz"
    rm /tmp/aqd.tar.gz
    echo "✓ 代码已同步到 $HOST:~/AutoQuantum"
    ;;
  test)
    ssh "$HOST" "cd ~/AutoQuantum && $R_PY -m unittest discover -s tests 2>&1 | tail -3"
    ;;
  run)
    shift
    ssh "$HOST" "cd ~/AutoQuantum && $R_PY $*"
    ;;
  fetch)
    shift
    REMOTE_PATH="${1:?用法: remote.sh fetch <远程目录>}"
    NAME=$(basename "$REMOTE_PATH")
    mkdir -p results
    scp -q -r "$HOST:$REMOTE_PATH" "results/$NAME"
    echo "✓ 已取回 results/$NAME"
    ;;
  *)
    grep -E "^#   " "$0" | sed 's/^#   //'
    exit 1
    ;;
esac
