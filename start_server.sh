#!/bin/bash
# TaskRouter — 启动 API 服务器
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8930}"
HOST="${2:-127.0.0.1}"

echo "=== TaskRouter API 服务器 ==="
echo "  仪表盘: http://${HOST}:${PORT}/"
echo "  API: http://${HOST}:${PORT}/api/"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python3 "$SCRIPT_DIR/scripts/api_server.py" --host "$HOST" --port "$PORT"
