#!/bin/bash
# TaskRouter — 一键推送到 GitHub
set -e

SKILL_DIR="/Users/huangweijie/Library/Application Support/CherryStudio/Data/Skills/task-router"
REPO_NAME="task-router"
GITHUB_USER="bobbyhuang666"

echo "=== TaskRouter — 发布到 GitHub ==="

# 1. 创建 GitHub 仓库（如果不存在）
echo "[1/5] 确保 GitHub 仓库存在..."
gh repo create "$REPO_NAME" --public --description "自进化 AI 成本优化引擎 — A3M 多信号路由 + 蒸馏学习闭环，简单任务走本地免费，复杂任务走云端付费" --yes 2>/dev/null || true

# 2. 准备 git 仓库
echo "[2/5] 准备 git 仓库..."
cd "$SKILL_DIR"

# 如果已有 .git，保留 remote 设置；否则新建
if [ ! -d ".git" ]; then
    git init
    echo "  → 新仓库初始化"
fi

# 确保 remote 存在
if ! git remote get-url origin 2>/dev/null; then
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "  → 添加 remote"
fi

# 3. 拉取最新远程内容（避免冲突）
echo "[3/5] 拉取远程..."
git fetch origin 2>/dev/null || true
if git rev-parse origin/main 2>/dev/null; then
    git merge origin/main --allow-unrelated-histories -m "merge remote" 2>/dev/null || true
fi

# 4. 提交
echo "[4/5] 提交代码..."
git add -A
git commit -m "v2.0: A3M多信号路由 + 语义缓存 + 蒸馏系统 + 输出验证降级" 2>/dev/null || echo "  → 无新变更"

# 5. 推送
echo "[5/5] 推送到 GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "=== 完成! ==="
echo "仓库地址: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "安装方式（其他机器）:"
echo "  git clone https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "  cd $REPO_NAME && bash install.sh"
