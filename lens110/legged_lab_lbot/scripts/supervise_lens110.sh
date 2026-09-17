#!/bin/bash
# Lens110 DeepMimic 自动监控脚本:
#   - 每 10 分钟检查训练进程
#   - 进程死亡 -> 自动从最新 checkpoint 续训 (无 checkpoint 则从头)
#   - 日志出现 NaN/Inf -> 杀掉并从最新 checkpoint 重启
#   - 奖励崩溃 (跌到峰值 30% 以下持续 3 次检查) -> 从峰值附近 checkpoint 重启
LOG=/tmp/supervise_lens110.log
STATE=/tmp/supervise_lens110_state
TRAIN_LOG=/tmp/train_lens110_deepmimic.log
PY="${PYTHON:-python}"
SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FRAMEWORK_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_ROOT="$FRAMEWORK_ROOT/logs/rsl_rl/lens110_deepmimic"
cd "$FRAMEWORK_ROOT" || exit 1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

latest_run() { ls -dt "$RUN_ROOT"/*/ 2>/dev/null | head -1; }
latest_model() {
  local run
  run=$(latest_run) || return 1
  ls "$run"model_*.pt 2>/dev/null | sort -V | tail -1
}
model_iter() { basename "$1" | sed 's/model_\([0-9]*\)\.pt/\1/'; }

start_fresh() {
  log "START_FRESH"
  setsid nohup "$PY" scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-Lens110-v0 \
    --headless --num_envs 1024 --max_iterations 60000 >> "$TRAIN_LOG" 2>&1 &
}

resume_latest() {
  local model name run
  model=$(latest_model)
  if [ -n "$model" ]; then
    run=$(dirname "$model")
    name=$(basename "$model")
    log "RESUME $run $name"
    setsid nohup "$PY" scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-Lens110-v0 \
      --headless --num_envs 1024 --max_iterations 60000 \
      --resume --checkpoint "$name" >> "$TRAIN_LOG" 2>&1 &
  else
    log "NO_CHECKPOINT -> FRESH"
    start_fresh
  fi
}

kill_train() {
  pkill -f "[t]rain.py --task LeggedLab-Isaac--Deepmimic-Lens110-v0" 2>/dev/null
  sleep 10
}

while true; do
  running=0
  pgrep -f "[t]rain.py --task LeggedLab-Isaac--Deepmimic-Lens110-v0" >/dev/null && running=1

  it=$(rg -o "Learning iteration [0-9]+/60000" "$TRAIN_LOG" 2>/dev/null | tail -1)
  rew=$(rg -o "Mean reward: [0-9.+-]+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -o '[-0-9.]*$')

  if [ "$running" -eq 0 ]; then
    model=$(latest_model)
    if [ -n "$model" ]; then
      mi=$(model_iter "$model")
      if [ "${mi:-0}" -ge 59500 ]; then
        log "TRAINING_DONE at $mi"
        sleep 3600
        continue
      fi
    fi
    log "DEAD -> resume | $it | $rew"
    resume_latest
  else
    if rg -qiE '(^|[^a-z])(nan|inf)([^a-z]|$)' "$TRAIN_LOG" 2>/dev/null; then
      log "NAN_OR_INF -> restart | $it"
      kill_train
      resume_latest
    fi

    # 奖励崩溃检测
    if [ -n "$rew" ]; then
      rew_num=$(awk "BEGIN{print ($rew > 0 ? $rew : 0)}")
      peak=$(awk '{print $1}' "$STATE" 2>/dev/null || echo 0)
      peak_iter=$(awk '{print $2}' "$STATE" 2>/dev/null || echo 0)
      if awk "BEGIN{exit !($rew_num > $peak)}"; then
        echo "$rew_num $it" > "$STATE"
        peak=$rew_num
        peak_iter=$(echo "$it" | grep -o '[0-9]*' | tail -1)
      fi
      crash_count=$(awk '{print $3}' "$STATE" 2>/dev/null || echo 0)
      if [ -n "$peak" ] && [ "${peak_iter:-0}" -gt 0 ] && awk "BEGIN{exit !($rew_num < $peak * 0.3)}"; then
        crash_count=$((crash_count + 1))
        awk -v p="$peak" -v pi="$peak_iter" -v c="$crash_count" 'BEGIN{print p, pi, c}' > "$STATE"
        if [ "$crash_count" -ge 3 ]; then
          log "REWARD_COLLAPSE ($rew vs peak $peak) -> restart near peak_iter $peak_iter"
          target=$(( (peak_iter / 100) * 100 ))
          run=$(latest_run)
          if [ -n "$run" ] && [ -f "$run/model_${target}.pt" ]; then
            kill_train
            log "RESUME_PEAK model_${target}.pt"
            setsid nohup "$PY" scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-Lens110-v0 \
              --headless --num_envs 1024 --max_iterations 60000 \
              --resume --checkpoint "model_${target}.pt" >> "$TRAIN_LOG" 2>&1 &
          else
            log "PEAK_MODEL_MISSING -> keep running"
          fi
          echo "0 0 0" > "$STATE"
        fi
      else
        crash_count=0
        awk -v p="$peak" -v pi="$peak_iter" 'BEGIN{print p, pi, 0}' > "$STATE"
      fi
    fi
  fi

  log "CHECK running=$running $it $rew"
  sleep 600
done
