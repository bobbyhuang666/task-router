#!/bin/bash
# 安装 TaskRouter git hooks
# 用法: bash scripts/setup-hooks.sh

HOOKS_DIR="$(git rev-parse --git-dir)/hooks"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOOKS_DIR"
cp "$SCRIPT_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "✓ pre-commit hook 已安装到 $HOOKS_DIR/pre-commit"
echo "  现在每次 git commit 前会自动运行 ruff check"
