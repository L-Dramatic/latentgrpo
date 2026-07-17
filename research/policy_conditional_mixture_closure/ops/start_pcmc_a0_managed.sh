#!/usr/bin/env bash
set -euo pipefail

ROOT="${PCMC_ROOT:-/root/autodl-tmp/latentgrpo}"
OPS_ROOT="${PCMC_OPS_ROOT:-/root/autodl-tmp/pcmc-a0-ops}"
LOG_DIR="${OPS_ROOT}/logs"
RUNNER="${ROOT}/research/policy_conditional_mixture_closure/ops/run_pcmc_a0_managed.sh"
WATCHDOG="${ROOT}/research/policy_conditional_mixture_closure/ops/watch_pcmc_a0_managed.sh"

mkdir -p "${LOG_DIR}"
nohup setsid bash "${RUNNER}" >"${LOG_DIR}/runner_bootstrap.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 3
if ! kill -0 "${runner_pid}" >/dev/null 2>&1; then
  printf 'managed runner failed to remain alive; inspect %s\n' "${LOG_DIR}/runner_bootstrap.log" >&2
  exit 70
fi
nohup setsid bash "${WATCHDOG}" >"${LOG_DIR}/watchdog_bootstrap.log" 2>&1 < /dev/null &
watchdog_pid=$!
printf 'managed_pid=%s watchdog_pid=%s\n' "${runner_pid}" "${watchdog_pid}"
