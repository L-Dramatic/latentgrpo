#!/usr/bin/env bash
set -uo pipefail

ROOT="${SWITCH_C2_ROOT:-/root/autodl-tmp/latentgrpo}"
OPS_ROOT="${SWITCH_C2_OPS_ROOT:-/root/autodl-tmp/switch-c2-ops}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-180}"

STATE_DIR="${OPS_ROOT}/state"
LOG_DIR="${OPS_ROOT}/logs"
EVIDENCE_DIR="${OPS_ROOT}/evidence"
RUNNER="${ROOT}/research/coordinate_invariance/run_switch_c2_autodl.sh"
RUN_LOG="${LOG_DIR}/switch_c2_managed.log"
SYSTEM_LOG="${LOG_DIR}/switch_c2_system.log"
STATUS_FILE="${STATE_DIR}/managed.status"
PID_FILE="${STATE_DIR}/managed.pid"
HEARTBEAT_FILE="${STATE_DIR}/managed.heartbeat"
LOCK_FILE="${STATE_DIR}/managed.lock"
FINAL_BUNDLE="${ROOT}/artifacts/coordinate_invariance/switch_c2_return_bundle.tar.gz"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${EVIDENCE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '%s managed runner already holds the lock\n' "$(date -u +%FT%TZ)" >&2
  exit 73
fi

utc_now() {
  date -u +%FT%TZ
}

log() {
  printf '%s %s\n' "$(utc_now)" "$*" | tee -a "${RUN_LOG}"
}

write_status() {
  local state="$1"
  local stage="$2"
  local exit_code="$3"
  local reason="${4:-none}"
  local tmp="${STATUS_FILE}.tmp.$$"
  reason="${reason//[^a-zA-Z0-9_.-]/_}"
  {
    printf 'state=%s\n' "${state}"
    printf 'stage=%s\n' "${stage}"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'reason=%s\n' "${reason}"
    printf 'updated_at=%s\n' "$(utc_now)"
    printf 'pid=%s\n' "$$"
  } >"${tmp}"
  mv -f "${tmp}" "${STATUS_FILE}"
}

system_snapshot() {
  {
    printf '\n[%s]\n' "$(utc_now)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader
    df -h "${ROOT}"
    free -h
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  } >>"${SYSTEM_LOG}" 2>&1
}

prepare_network() {
  local probe_url="https://huggingface.co/api/datasets/HuggingFaceH4/MATH-500/revision/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
  local attempt=0

  if [[ -f /etc/network_turbo ]]; then
    # AutoDL provides this file to route outbound model and dataset downloads.
    # shellcheck disable=SC1091
    source /etc/network_turbo >/dev/null 2>&1 || true
  fi
  for attempt in $(seq 1 10); do
    if curl -fsS --connect-timeout 15 --max-time 45 "${probe_url}" -o /dev/null; then
      log "official Hugging Face connectivity passed attempt=${attempt}"
      return 0
    fi
    log "official Hugging Face connectivity failed attempt=${attempt}/10"
    sleep 30
  done
  log "official Hugging Face connectivity unavailable after retries"
  return 69
}

prepare_assets() {
  local offline_marker="${ROOT}/_models/hf_home/hub/SWITCH_C2_OFFLINE_READY"
  if [[ -f "${offline_marker}" ]]; then
    if ! "${PYTHON_BIN}" "${OPS_ROOT}/bin/verify_switch_c2_assets.py" \
      --hub-cache "${ROOT}/_models/hf_home/hub" \
      --output "${EVIDENCE_DIR}/switch_c2_asset_verification.json" \
      >>"${RUN_LOG}" 2>&1; then
      log "offline asset marker exists but integrity verification failed"
      return 65
    fi
    export HF_HUB_OFFLINE=1
    log "verified offline asset cache selected"
    return 0
  fi
  prepare_network
}

request_shutdown_forever() {
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    log "shutdown request attempt=${attempt}"
    sync
    /usr/bin/shutdown >>"${RUN_LOG}" 2>&1 || true
    sleep 60
  done
}

heartbeat_loop() {
  while true; do
    local tmp="${HEARTBEAT_FILE}.tmp.$$"
    date +%s >"${tmp}"
    mv -f "${tmp}" "${HEARTBEAT_FILE}"
    sleep 30
  done
}

heartbeat_pid=""
finalizing=0
finalize() {
  local run_rc="$1"
  local collect_rc=0
  local required_stage="none"
  local final_state="FAILED"

  if [[ "${finalizing}" == "1" ]]; then
    return
  fi
  finalizing=1
  trap - EXIT INT TERM
  set +e

  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" >/dev/null 2>&1 || true
    wait "${heartbeat_pid}" >/dev/null 2>&1 || true
  fi

  system_snapshot
  if [[ "${run_rc}" == "0" ]]; then
    required_stage="test"
  fi
  log "collecting evidence required_stage=${required_stage} run_rc=${run_rc}"
  AUTO_COLLECT=0 REQUIRE_STAGE="${required_stage}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${RUNNER}" collect >>"${RUN_LOG}" 2>&1
  collect_rc=$?

  if [[ "${run_rc}" == "0" && "${collect_rc}" == "0" ]]; then
    final_state="SUCCESS"
  elif [[ "${run_rc}" == "0" ]]; then
    run_rc=74
  fi

  if [[ -f "${FINAL_BUNDLE}" ]]; then
    cp -f "${FINAL_BUNDLE}" "${EVIDENCE_DIR}/"
    sha256sum "${FINAL_BUNDLE}" >"${EVIDENCE_DIR}/switch_c2_return_bundle.tar.gz.sha256"
    tar -tzf "${FINAL_BUNDLE}" >"${EVIDENCE_DIR}/switch_c2_return_bundle.contents.txt" 2>&1 || true
  fi
  cp -f "${RUN_LOG}" "${EVIDENCE_DIR}/" 2>/dev/null || true
  cp -f "${SYSTEM_LOG}" "${EVIDENCE_DIR}/" 2>/dev/null || true

  write_status "${final_state}" final "${run_rc}" "runner_exit_${run_rc}_collect_${collect_rc}"
  sync
  write_status SHUTDOWN_PENDING final "${run_rc}" "${final_state}"
  log "final_state=${final_state} run_rc=${run_rc} collect_rc=${collect_rc}; shutdown in ${SHUTDOWN_GRACE_SECONDS}s"
  sync
  sleep "${SHUTDOWN_GRACE_SECONDS}"
  request_shutdown_forever
}

trap 'finalize $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' "$$" >"${PID_FILE}"
rm -f "${STATE_DIR}/watchdog.reason" "${STATE_DIR}/shutdown.requested"
write_status RUNNING all -1 starting
system_snapshot
heartbeat_loop &
heartbeat_pid=$!

prepare_assets || exit $?
log "starting frozen SWITCH C2 pipeline"
set +e
PYTHONUNBUFFERED=1 PYTHON_BIN="${PYTHON_BIN}" AUTO_COLLECT=1 \
  stdbuf -oL -eL bash "${RUNNER}" test 2>&1 | tee -a "${RUN_LOG}"
pipeline_rc=("${PIPESTATUS[@]}")
set -u
runner_rc="${pipeline_rc[0]:-70}"
tee_rc="${pipeline_rc[1]:-70}"
if [[ "${runner_rc}" == "0" && "${tee_rc}" != "0" ]]; then
  runner_rc=74
fi
log "pipeline exited runner_rc=${runner_rc} tee_rc=${tee_rc}"
exit "${runner_rc}"
