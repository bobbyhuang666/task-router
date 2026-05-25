#!/bin/bash
# TaskRouter 安装脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查依赖
command -v python3 >/dev/null 2>&1 || { echo "需要 Python 3"; exit 1; }

echo "安装 Python 依赖..."
pip3 install requests -q

echo "配置 sma 命令..."

# 添加到 PATH (使用 alias 方式)
ALIAS_CMD="alias sma='python3 $SCRIPT_DIR/scripts/task_router.py'"

if [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "alias sma=" "$HOME/.zshrc"; then
        echo "$ALIAS_CMD" >> "$HOME/.zshrc"
        echo "已添加到 ~/.zshrc"
    else
        echo "别名已存在 ~/.zshrc"
    fi
fi

if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q "alias sma=" "$HOME/.bashrc"; then
        echo "$ALIAS_CMD" >> "$HOME/.bashrc"
        echo "已添加到 ~/.bashrc"
    else
        echo "别名已存在 ~/.bashrc"
    fi
fi

echo ""
echo "安装完成！执行以下命令使别名生效："
echo "  source ~/.zshrc"
echo ""
echo "快速测试："
echo "  sma --estimate \"测试任务\""
