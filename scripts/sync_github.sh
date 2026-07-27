#!/bin/bash
# AutoQuantum — 一键同步到 GitHub
# 使用方法: bash scripts/sync_github.sh

set -e

echo "=== AutoQuantum GitHub 同步 ==="

# 检查 gh CLI
if command -v gh &>/dev/null; then
    echo "[1] 使用 GitHub CLI (gh)"
    gh auth status &>/dev/null || gh auth login
    gh repo sync anfax/AutoQuantum
    echo "同步完成!"
    exit 0
fi

# 检查 SSH key
SSH_KEY="$HOME/.ssh/id_autoquantum"
if [ ! -f "$SSH_KEY" ]; then
    echo "[1] 生成 SSH key..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "autoquantum@local"
    echo ""
    echo "请将以下公钥添加到 GitHub → Settings → SSH and GPG keys:"
    echo "================================================================"
    cat "$SSH_KEY.pub"
    echo "================================================================"
    echo ""
    echo "完成后按 Enter 继续..."
    read -r
fi

echo "[2] 配置 SSH ..."
cat >> "$HOME/.ssh/config" << EOF

Host github.com
  HostName github.com
  IdentityFile $SSH_KEY
  IdentitiesOnly yes
EOF
chmod 600 "$HOME/.ssh/config"

echo "[3] 设置 remote ..."
cd "$(dirname "$0")/.."
git remote set-url origin git@github.com:anfax/AutoQuantum.git

echo "[4] 推送 ..."
git push -u origin main
echo ""
echo "同步完成!"
