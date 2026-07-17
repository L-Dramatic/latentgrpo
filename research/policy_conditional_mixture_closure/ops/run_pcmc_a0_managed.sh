#!/usr/bin/env bash
set -uo pipefail

ROOT="${PCMC_ROOT:-/root/autodl-tmp/latentgrpo}"
OPS_ROOT="${PCMC_OPS_ROOT:-/root/autodl-tmp/pcmc-a0-ops}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-180}"

STATE_DIR="${OPS_ROOT}/state"
LOG_DIR="${OPS_ROOT}/logs"
RESULT_DIR="${OPS_ROOT}/results"
EVIDENCE_DIR="${OPS_ROOT}/evidence"
PROTOCOL="${ROOT}/research/policy_conditional_mixture_closure/configs/pcmc_gate_ab_v1.json"
RUN_LOG="${LOG_DIR}/pcmc_a0_managed.log"
SYSTEM_LOG="${LOG_DIR}/pcmc_a0_system.log"
STATUS_FILE="${STATE_DIR}/managed.status"
PID_FILE="${STATE_DIR}/managed.pid"
HEARTBEAT_FILE="${STATE_DIR}/managed.heartbeat"
LOCK_FILE="${STATE_DIR}/managed.lock"
ASSET_REPORT="${RESULT_DIR}/asset_preflight.json"
LATENT_PREFLIGHT="${RESULT_DIR}/preflight_latent_grpo_llama_1b.json"
SOFT_PREFLIGHT="${RESULT_DIR}/preflight_soft_grpo_qwen_1_5b.json"
A0_RECORDS="${RESULT_DIR}/a0_records.jsonl"
A0_DECISION="${RESULT_DIR}/a0_decision.json"
FINAL_BUNDLE="${ROOT}/artifacts/pcmc_gate/pcmc_a0_return_bundle.tar.gz"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${RESULT_DIR}" "${EVIDENCE_DIR}" "$(dirname "${FINAL_BUNDLE}")"
cd "${ROOT}"
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
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    df -h "${ROOT}"
    free -h
  } >>"${SYSTEM_LOG}" 2>&1
}

heartbeat_loop() {
  while true; do
    local tmp="${HEARTBEAT_FILE}.tmp.$$"
    date +%s >"${tmp}"
    mv -f "${tmp}" "${HEARTBEAT_FILE}"
    sleep 30
  done
}

run_stage() {
  local stage="$1"
  shift
  local rc=0
  local pipeline_rc=()
  write_status RUNNING "${stage}" -1 starting
  log "starting stage=${stage}"
  set +e
  PYTHONUNBUFFERED=1 stdbuf -oL -eL "$@" 2>&1 | tee -a "${RUN_LOG}"
  pipeline_rc=("${PIPESTATUS[@]}")
  set -u
  rc="${pipeline_rc[0]:-70}"
  if [[ "${rc}" == "0" && "${pipeline_rc[1]:-70}" != "0" ]]; then
    rc=74
  fi
  log "finished stage=${stage} rc=${rc}"
  return "${rc}"
}

collect_evidence() {
  local tmp_bundle="${FINAL_BUNDLE}.tmp.$$"
  rm -f "${tmp_bundle}"
  cp -f "${RUN_LOG}" "${EVIDENCE_DIR}/" 2>/dev/null || true
  cp -f "${SYSTEM_LOG}" "${EVIDENCE_DIR}/" 2>/dev/null || true
  cp -f "${STATUS_FILE}" "${EVIDENCE_DIR}/" 2>/dev/null || true
  tar -czf "${tmp_bundle}" \
    -C "${ROOT}" \
    research/policy_conditional_mixture_closure/configs/pcmc_gate_ab_v1.json \
    research/policy_conditional_mixture_closure/CHECKPOINT_PREREGISTRATION.md \
    -C "${OPS_ROOT}" results logs state
  mv -f "${tmp_bundle}" "${FINAL_BUNDLE}"
  sha256sum "${FINAL_BUNDLE}" >"${EVIDENCE_DIR}/pcmc_a0_return_bundle.tar.gz.sha256"
  tar -tzf "${FINAL_BUNDLE}" >"${EVIDENCE_DIR}/pcmc_a0_return_bundle.contents.txt"
  cp -f "${FINAL_BUNDLE}" "${EVIDENCE_DIR}/"
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

heartbeat_pid=""
finalizing=0
finalize() {
  local run_rc="$1"
  local collect_rc=0
  local final_state=FAILED

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
  collect_evidence
  collect_rc=$?
  if [[ "${run_rc}" == "0" && "${collect_rc}" == "0" ]]; then
    final_state=SUCCESS
  elif [[ "${run_rc}" == "0" ]]; then
    run_rc=74
  fi
  write_status "${final_state}" final "${run_rc}" "runner_${run_rc}_collect_${collect_rc}"
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
rm -f "${STATE_DIR}/watchdog.reason"
write_status RUNNING bootstrap -1 starting
system_snapshot
heartbeat_loop &
heartbeat_pid=$!

run_stage protocol_lint \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.checkpoint_gate_runner \
  --mode lint --protocol "${PROTOCOL}" --workspace-root "${ROOT}" || exit $?

run_stage asset_preflight \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.checkpoint_gate_runner \
  --mode asset-preflight --protocol "${PROTOCOL}" --workspace-root "${ROOT}" \
  --output "${ASSET_REPORT}" || exit $?

run_stage latent_engineering_preflight \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.checkpoint_gate_runner \
  --mode engineering-preflight --protocol "${PROTOCOL}" --workspace-root "${ROOT}" \
  --checkpoint-id latent_grpo_llama_1b --asset-report "${ASSET_REPORT}" \
  --preflight-tasks 4 --output "${LATENT_PREFLIGHT}" || exit $?

run_stage latent_a0 \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.checkpoint_gate_runner \
  --mode run-a0 --protocol "${PROTOCOL}" --workspace-root "${ROOT}" \
  --checkpoint-id latent_grpo_llama_1b --asset-report "${ASSET_REPORT}" \
  --preflight-result "${LATENT_PREFLIGHT}" --output-root "${RESULT_DIR}" || exit $?

run_stage soft_engineering_preflight \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.checkpoint_gate_runner \
  --mode engineering-preflight --protocol "${PROTOCOL}" --workspace-root "${ROOT}" \
  --checkpoint-id soft_grpo_qwen_1_5b --asset-report "${ASSET_REPORT}" \
  --preflight-tasks 4 --output "${SOFT_PREFLIGHT}" || exit $?

run_stage soft_a0 \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.checkpoint_gate_runner \
  --mode run-a0 --protocol "${PROTOCOL}" --workspace-root "${ROOT}" \
  --checkpoint-id soft_grpo_qwen_1_5b --asset-report "${ASSET_REPORT}" \
  --preflight-result "${SOFT_PREFLIGHT}" --output-root "${RESULT_DIR}" || exit $?

run_stage a0_decision \
  "${PYTHON_BIN}" -m research.policy_conditional_mixture_closure.gate_a_analysis \
  --protocol "${PROTOCOL}" --records "${A0_RECORDS}" --completed-stage A0 \
  --output "${A0_DECISION}" || exit $?

write_status RUNNING complete 0 scientific_gate_complete
log "A0 pipeline complete; A1 remains unauthorized and will not run"
exit 0
