#!/usr/bin/env bash
set -uo pipefail

ROOT="${SWITCH_C2_ROOT:-/root/autodl-tmp/latentgrpo}"
OPS_ROOT="${SWITCH_C2_OPS_ROOT:-/root/autodl-tmp/switch-c2-ops}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-60}"
HEARTBEAT_MAX_AGE="${HEARTBEAT_MAX_AGE:-600}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-5}"

STATE_DIR="${OPS_ROOT}/state"
LOG_DIR="${OPS_ROOT}/logs"
EVIDENCE_DIR="${OPS_ROOT}/evidence"
PID_FILE="${STATE_DIR}/managed.pid"
HEARTBEAT_FILE="${STATE_DIR}/managed.heartbeat"
STATUS_FILE="${STATE_DIR}/managed.status"
WATCH_LOG="${LOG_DIR}/switch_c2_watchdog.log"
METRICS_FILE="${LOG_DIR}/switch_c2_gpu_metrics.csv"
RUNNER="${ROOT}/research/coordinate_invariance/run_switch_c2_autodl.sh"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${EVIDENCE_DIR}"
exec 8>"${STATE_DIR}/watchdog.lock"
if ! flock -n 8; then
  printf '%s watchdog already holds the lock\n' "$(date -u +%FT%TZ)" >&2
  exit 73
fi

utc_now() {
  date -u +%FT%TZ
}

log() {
  printf '%s %s\n' "$(utc_now)" "$*" | tee -a "${WATCH_LOG}"
}

status_value() {
  local key="$1"
  awk -F= -v key="${key}" '$1 == key {print $2}' "${STATUS_FILE}" 2>/dev/null | tail -n 1
}

managed_pid() {
  tr -dc '0-9' <"${PID_FILE}" 2>/dev/null
}

terminate_managed() {
  local pid="$1"
  local pgid=""
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
  if [[ -n "${pgid}" ]]; then
    kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
    sleep 60
    kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
  else
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  fi
}

emergency_shutdown() {
  local reason="$1"
  local pid="${2:-}"
  set +e
  reason="${reason//[^a-zA-Z0-9_.-]/_}"
  log "emergency stop reason=${reason} pid=${pid:-none}"
  printf '%s\n' "${reason}" >"${STATE_DIR}/watchdog.reason"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    terminate_managed "${pid}"
  fi
  AUTO_COLLECT=0 REQUIRE_STAGE=none PYTHON_BIN="${PYTHON_BIN}" \
    bash "${RUNNER}" collect >>"${WATCH_LOG}" 2>&1 || true
  cp -f "${ROOT}/artifacts/coordinate_invariance/switch_c2_return_bundle.tar.gz" \
    "${EVIDENCE_DIR}/" 2>/dev/null || true
  cp -f "${WATCH_LOG}" "${EVIDENCE_DIR}/" 2>/dev/null || true
  sync
  while true; do
    log "watchdog shutdown request"
    /usr/bin/shutdown >>"${WATCH_LOG}" 2>&1 || true
    sleep 60
  done
}

if [[ ! -f "${METRICS_FILE}" ]]; then
  printf 'epoch_utc,iso_utc,gpu_name,temp_c,util_pct,memory_used_mib,memory_total_mib,power_w,disk_free_bytes,managed_pid,state\n' \
    >"${METRICS_FILE}"
fi

log "watchdog starting"
for _ in $(seq 1 120); do
  pid="$(managed_pid)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
pid="$(managed_pid)"
if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
  emergency_shutdown managed_runner_never_started "${pid}"
fi

nvidia_failures=0
shutdown_pending_since=0
loop_count=0
while true; do
  now_epoch="$(date +%s)"
  now_iso="$(utc_now)"
  state="$(status_value state)"
  pid="$(managed_pid)"
  disk_free="$(df --output=avail -B1 "${ROOT}" 2>/dev/null | tail -n 1 | tr -d ' ')"
  gpu_metrics="$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null)"
  nvidia_rc=$?

  if [[ "${nvidia_rc}" == "0" && -n "${gpu_metrics}" ]]; then
    nvidia_failures=0
    printf '%s,%s,%s,%s,%s,%s\n' "${now_epoch}" "${now_iso}" "${gpu_metrics}" \
      "${disk_free:-0}" "${pid:-none}" "${state:-unknown}" >>"${METRICS_FILE}"
  else
    nvidia_failures=$((nvidia_failures + 1))
    log "nvidia-smi failure count=${nvidia_failures}"
  fi

  if [[ -n "${disk_free}" ]] && (( disk_free < MIN_FREE_DISK_GIB * 1024 * 1024 * 1024 )); then
    emergency_shutdown disk_below_${MIN_FREE_DISK_GIB}GiB "${pid}"
  fi
  if (( nvidia_failures >= 5 )); then
    emergency_shutdown nvidia_smi_failed_5_times "${pid}"
  fi

  if [[ -f "${HEARTBEAT_FILE}" ]]; then
    heartbeat_epoch="$(tr -dc '0-9' <"${HEARTBEAT_FILE}")"
    if [[ -n "${heartbeat_epoch}" ]] && (( now_epoch - heartbeat_epoch > HEARTBEAT_MAX_AGE )); then
      emergency_shutdown heartbeat_stale "${pid}"
    fi
  fi

  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
    if [[ "${state}" == "SHUTDOWN_PENDING" ]]; then
      if (( shutdown_pending_since == 0 )); then
        shutdown_pending_since="${now_epoch}"
      elif (( now_epoch - shutdown_pending_since > 180 )); then
        emergency_shutdown managed_shutdown_did_not_complete "${pid}"
      fi
    else
      emergency_shutdown managed_runner_disappeared_state_${state:-unknown} "${pid}"
    fi
  fi

  loop_count=$((loop_count + 1))
  if (( loop_count % 5 == 0 )); then
    {
      printf '\n[%s] pid=%s state=%s disk_free=%s\n' "${now_iso}" "${pid:-none}" "${state:-unknown}" "${disk_free:-unknown}"
      ps -eo pid,ppid,pgid,etime,%cpu,%mem,cmd --sort=-%cpu | head -n 20
    } >>"${WATCH_LOG}" 2>&1
  fi
  sleep "${INTERVAL_SECONDS}"
done
