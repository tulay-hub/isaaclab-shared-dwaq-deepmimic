#!/usr/bin/env bash
# 启动 Lens110 DeepMimic 训练, 并自动跟随启动/守护 TensorBoard (端口 6006) + 打开浏览器。
#
# 用法 (在 legged_lab_lbot 目录下):
#   ./scripts/train_lens110_with_tb.sh --task LeggedLab-Isaac--Deepmimic-Lens110-v0 --headless \
#       --num_envs 1024 --max_iterations 60000

set -e

SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FRAMEWORK_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPOSITORY_ROOT=$(cd -- "$FRAMEWORK_ROOT/../../../../.." && pwd)
TB_LOGDIR="$REPOSITORY_ROOT/tb_logs"
TB_PORT=6006
PYTHON="${PYTHON:-python}"

# 1) 重启 TensorBoard (避免端口占用/残留)
pkill -f "tensorboard.main --logdir $TB_LOGDIR --port $TB_PORT" 2>/dev/null || true
sleep 1
nohup "$PYTHON" -m tensorboard.main --logdir "$TB_LOGDIR" --port "$TB_PORT" \
  --host 0.0.0.0 --reload_interval 5 > /tmp/tensorboard.log 2>&1 &
TB_PID=$!
echo "[tb] TensorBoard started PID=$TB_PID -> http://localhost:$TB_PORT"

# 2) 守护: 训练期间 TensorBoard 挂了自动重启
(
  while true; do
    if ! kill -0 "$TB_PID" 2>/dev/null; then
      nohup "$PYTHON" -m tensorboard.main --logdir "$TB_LOGDIR" --port "$TB_PORT" \
        --host 0.0.0.0 --reload_interval 5 >> /tmp/tensorboard.log 2>&1 &
      TB_PID=$!
      echo "[tb] TensorBoard restarted PID=$TB_PID"
      xdg-open "http://localhost:$TB_PORT/" 2>/dev/null || true
    fi
    sleep 30
  done
) &
WATCHER_PID=$!
trap 'kill "$WATCHER_PID" 2>/dev/null || true' INT TERM EXIT

# 3) 打开浏览器
sleep 2
xdg-open "http://localhost:$TB_PORT/" 2>/dev/null || true

# 4) 前台运行训练
cd "$FRAMEWORK_ROOT"
PYTHONPATH= "$PYTHON" scripts/rsl_rl/train.py "$@"
