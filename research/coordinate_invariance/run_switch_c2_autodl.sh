#!/usr/bin/env bash
set -euo pipefail

ROOT="${SWITCH_C2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAGE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${SWITCH_C2_VENV:-${ROOT}/.switch_c2_env}"
MIN_GPU_GIB="${MIN_GPU_GIB:-78}"
MIN_DISK_GIB="${MIN_DISK_GIB:-70}"
FORCE="${FORCE:-0}"
AUTO_COLLECT="${AUTO_COLLECT:-1}"
PROJECT_REF="${SWITCH_C2_PROJECT_REF:-switch-c2-frozen-v2}"

CONFIG="${ROOT}/research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json"
IDENTITY_CONFIG="${ROOT}/research/coordinate_invariance/configs/switch_checkpoint_identity_smoke_v1.json"
ARTIFACTS="${ROOT}/artifacts/coordinate_invariance"
JOURNALS="${ARTIFACTS}/journals"
IDENTITY_ARTIFACT="${ARTIFACTS}/switch_checkpoint_identity_smoke_v1.json"
ELIGIBILITY_ARTIFACT="${ARTIFACTS}/switch_c2_eligibility_v1.json"
CALIBRATION_ARTIFACT="${ARTIFACTS}/switch_c2_calibration_v1.json"
TEST_ARTIFACT="${ARTIFACTS}/switch_c2_test_v1.json"
COLLECTION_ARTIFACT="${ARTIFACTS}/switch_c2_return_bundle.tar.gz"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${ROOT}/_models/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

ensure_environment() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} is unavailable"
  if [[ ! -x "${VENV}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV}"
  fi
  "${VENV}/bin/python" -m pip install --upgrade pip
  "${VENV}/bin/python" -m pip install -r \
    "${ROOT}/research/coordinate_invariance/requirements-switch-c2.txt"
}

check_project_revision() {
  local head
  local expected
  git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "workspace is not a git checkout"
  head="$(git -C "${ROOT}" rev-parse HEAD)"
  expected="$(git -C "${ROOT}" rev-parse "${PROJECT_REF}^{commit}" 2>/dev/null)" || \
    die "frozen project ref ${PROJECT_REF} is unavailable"
  [[ "${head}" == "${expected}" ]] || \
    die "HEAD ${head} does not match frozen ${PROJECT_REF} (${expected})"
}

ensure_source() {
  local source_dir="${ROOT}/_external/switch"
  local commit="d8d97cdc6276fcfa6e48f6a6b19ce472c7b87fcd"
  mkdir -p "${ROOT}/_external"
  if [[ ! -e "${source_dir}" ]]; then
    git clone https://github.com/LARK-AI-Lab/SWITCH.git "${source_dir}"
  fi
  [[ -d "${source_dir}/.git" ]] || die "${source_dir} exists but is not a git checkout"
  if [[ "$(git -C "${source_dir}" rev-parse HEAD)" != "${commit}" ]]; then
    git -C "${source_dir}" fetch --depth 1 origin "${commit}"
    git -C "${source_dir}" checkout --detach "${commit}"
  fi
  [[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${commit}" ]] || \
    die "SWITCH source commit could not be pinned"
}

check_resources() {
  MIN_GPU_GIB="${MIN_GPU_GIB}" MIN_DISK_GIB="${MIN_DISK_GIB}" ROOT="${ROOT}" \
    "${VENV}/bin/python" - <<'PY'
import os
import shutil
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
gpu_gib = props.total_memory / 1024**3
minimum_gpu = float(os.environ["MIN_GPU_GIB"])
if gpu_gib < minimum_gpu:
    raise SystemExit(
        f"GPU has {gpu_gib:.1f} GiB, below the frozen {minimum_gpu:.1f} GiB floor"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("GPU does not report bfloat16 support")
disk_gib = shutil.disk_usage(os.environ["ROOT"]).free / 1024**3
minimum_disk = float(os.environ["MIN_DISK_GIB"])
if disk_gib < minimum_disk:
    raise SystemExit(
        f"workspace disk has {disk_gib:.1f} GiB free, below {minimum_disk:.1f} GiB"
    )
major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 5):
    raise SystemExit(f"PyTorch {torch.__version__} is older than the tested 2.5 line")
print(
    f"resource-check: gpu={props.name!r} vram={gpu_gib:.1f}GiB "
    f"disk_free={disk_gib:.1f}GiB torch={torch.__version__}"
)
PY
}

download_dataset() {
  "${VENV}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="HuggingFaceH4/MATH-500",
    repo_type="dataset",
    revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    cache_dir=os.environ["HF_HUB_CACHE"],
    allow_patterns=["test.jsonl"],
)
print(f"dataset-snapshot: {path}")
PY
}

check_prerequisites() {
  [[ -f "${ARTIFACTS}/switch_c2_source_preflight_v1.json" ]] || \
    die "source-preflight artifact is missing"
  [[ -f "${ARTIFACTS}/fctr_coconut_smoke_v1c.json" ]] || \
    die "Coconut C1c artifact is missing"
  [[ -f "${ROOT}/artifacts/coordinate_invariance/switch_c2_prompt_order_v1.json" ]] || \
    die "frozen prompt-order artifact is missing"
}

run_local_tests() {
  "${VENV}/bin/python" -m pytest \
    "${ROOT}/tests/test_switch_source_equivalence.py" \
    "${ROOT}/tests/test_switch_prompt_order.py" \
    "${ROOT}/tests/test_switch_c2_geometry.py" \
    "${ROOT}/tests/test_switch_c2_protocol.py" -q
}

collector_python() {
  if [[ -x "${VENV}/bin/python" ]]; then
    printf '%s\n' "${VENV}/bin/python"
  else
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

status_results() {
  local python_bin
  python_bin="$(collector_python)"
  "${python_bin}" "${ROOT}/research/coordinate_invariance/switch_c2_collect.py" \
    --workspace-root "${ROOT}"
}

collect_results() {
  local required_stage="${1:-none}"
  local python_bin
  python_bin="$(collector_python)"
  "${python_bin}" "${ROOT}/research/coordinate_invariance/switch_c2_collect.py" \
    --workspace-root "${ROOT}" \
    --require-stage "${required_stage}" \
    --output "${COLLECTION_ARTIFACT}"
}

auto_collect_on_exit() {
  local exit_code="$1"
  local collection_code=0
  trap - EXIT
  if [[ "${AUTO_COLLECT}" == "1" ]]; then
    set +e
    collect_results none
    collection_code=$?
    set -e
    if [[ "${collection_code}" != "0" ]]; then
      printf 'WARNING: automatic C2 evidence collection failed with status %s\n' \
        "${collection_code}" >&2
      if [[ "${exit_code}" == "0" ]]; then
        exit_code=2
      fi
    fi
  fi
  exit "${exit_code}"
}

artifact_matches() {
  local artifact="$1"
  local kind="$2"
  [[ "${FORCE}" == "0" && -f "${artifact}" ]] || return 1
  "${VENV}/bin/python" - "${artifact}" "${kind}" "${CONFIG}" "${IDENTITY_CONFIG}" <<'PY'
import hashlib
import json
import pathlib
import sys

artifact_path, kind, config_path, identity_config_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(artifact_path.read_text(encoding="utf-8"))
if report.get("status") != "pass":
    raise SystemExit(1)

def canonical(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

if kind == pathlib.Path("identity"):
    from research.coordinate_invariance.switch_checkpoint_identity_smoke import implementation_hashes
    valid = (
        report.get("config_sha256") == canonical(identity_config_path)
        and report.get("implementation_sha256") == implementation_hashes()
    )
elif kind == pathlib.Path("eligibility"):
    from research.coordinate_invariance.switch_c2_eligibility_scan import implementation_hashes
    valid = (
        report.get("config_sha256") == canonical(config_path)
        and report.get("implementation_sha256") == implementation_hashes()
    )
else:
    from research.coordinate_invariance.switch_c2_scientific_gate import implementation_hashes
    valid = (
        report.get("config_sha256") == canonical(config_path)
        and report.get("implementation_sha256") == implementation_hashes()
        and report.get("phase") == str(kind)
    )
raise SystemExit(0 if valid else 1)
PY
}

prepare() {
  cd "${ROOT}"
  mkdir -p "${ARTIFACTS}" "${JOURNALS}" "${HF_HUB_CACHE}"
  check_project_revision
  ensure_environment
  ensure_source
  check_resources
  download_dataset
  check_prerequisites
  run_local_tests
}

run_identity() {
  if artifact_matches "${IDENTITY_ARTIFACT}" identity; then
    printf 'identity: existing artifact is current and passed\n'
    return
  fi
  "${VENV}/bin/python" -m \
    research.coordinate_invariance.switch_checkpoint_identity_smoke \
    --config "${IDENTITY_CONFIG}" \
    --workspace-root "${ROOT}" \
    --output "${IDENTITY_ARTIFACT}"
}

run_eligibility() {
  if artifact_matches "${ELIGIBILITY_ARTIFACT}" eligibility; then
    printf 'eligibility: existing artifact is current and passed\n'
    return
  fi
  local key
  key="$("${VENV}/bin/python" - <<'PY'
import hashlib
import json
from pathlib import Path
from research.coordinate_invariance.switch_c2_eligibility_scan import canonical_config_hash, implementation_hashes
config = json.loads(Path("research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json").read_text())
payload = canonical_config_hash(config) + json.dumps(implementation_hashes(), sort_keys=True)
print(hashlib.sha256(payload.encode()).hexdigest()[:16])
PY
)"
  "${VENV}/bin/python" -m \
    research.coordinate_invariance.switch_c2_eligibility_scan \
    --config "${CONFIG}" \
    --workspace-root "${ROOT}" \
    --journal "${JOURNALS}/switch_c2_eligibility_${key}.jsonl" \
    --output "${ELIGIBILITY_ARTIFACT}"
}

scientific_key() {
  "${VENV}/bin/python" - <<'PY'
import hashlib
import json
from pathlib import Path
from research.coordinate_invariance.switch_c2_scientific_gate import canonical_config_hash, implementation_hashes
config = json.loads(Path("research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json").read_text())
payload = canonical_config_hash(config) + json.dumps(implementation_hashes(), sort_keys=True)
print(hashlib.sha256(payload.encode()).hexdigest()[:16])
PY
}

run_calibration() {
  if artifact_matches "${CALIBRATION_ARTIFACT}" calibration; then
    printf 'calibration: existing artifact is current and passed\n'
    return
  fi
  local key
  key="$(scientific_key)"
  "${VENV}/bin/python" -m \
    research.coordinate_invariance.switch_c2_scientific_gate \
    --phase calibration \
    --config "${CONFIG}" \
    --workspace-root "${ROOT}" \
    --eligibility-artifact "${ELIGIBILITY_ARTIFACT}" \
    --journal "${JOURNALS}/switch_c2_calibration_${key}.jsonl" \
    --output "${CALIBRATION_ARTIFACT}"
}

run_test() {
  if artifact_matches "${TEST_ARTIFACT}" test; then
    printf 'test: existing artifact is current and passed\n'
    return
  fi
  local key
  key="$(scientific_key)"
  "${VENV}/bin/python" -m \
    research.coordinate_invariance.switch_c2_scientific_gate \
    --phase test \
    --config "${CONFIG}" \
    --workspace-root "${ROOT}" \
    --eligibility-artifact "${ELIGIBILITY_ARTIFACT}" \
    --calibration-artifact "${CALIBRATION_ARTIFACT}" \
    --journal "${JOURNALS}/switch_c2_test_${key}.jsonl" \
    --output "${TEST_ARTIFACT}"
}

case "${STAGE}" in
  status)
    status_results
    ;;
  collect)
    collect_results "${REQUIRE_STAGE:-none}"
    ;;
  prepare)
    trap 'auto_collect_on_exit $?' EXIT
    prepare
    ;;
  identity)
    trap 'auto_collect_on_exit $?' EXIT
    prepare
    run_identity
    ;;
  eligibility)
    trap 'auto_collect_on_exit $?' EXIT
    prepare
    run_identity
    run_eligibility
    ;;
  calibration)
    trap 'auto_collect_on_exit $?' EXIT
    prepare
    run_identity
    run_eligibility
    run_calibration
    ;;
  test|all)
    trap 'auto_collect_on_exit $?' EXIT
    prepare
    run_identity
    run_eligibility
    run_calibration
    run_test
    ;;
  *)
    die "stage must be status, collect, prepare, identity, eligibility, calibration, test, or all"
    ;;
esac
