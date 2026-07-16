#!/usr/bin/env python3
"""Monitor a managed AutoDL C2 run, retrieve evidence, and verify shutdown."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import stat
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import paramiko


REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/latentgrpo")
REMOTE_OPS = PurePosixPath("/root/autodl-tmp/switch-c2-ops")
REMOTE_STATUS = REMOTE_OPS / "state/managed.status"
REMOTE_LOG = REMOTE_OPS / "logs/switch_c2_managed.log"
REMOTE_ARTIFACTS = REMOTE_ROOT / "artifacts/coordinate_invariance"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PinnedHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    def __init__(self, expected_sha256: str) -> None:
        self.expected_sha256 = expected_sha256.removeprefix("SHA256:").rstrip("=")

    def missing_host_key(self, client, hostname, key):  # type: ignore[no-untyped-def]
        actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if actual != self.expected_sha256:
            raise paramiko.SSHException(
                f"host key mismatch for {hostname}: expected SHA256:{self.expected_sha256}, "
                f"received SHA256:{actual}"
            )
        client.get_host_keys().add(hostname, key.get_name(), key)


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHostKeyPolicy(args.host_key_sha256))
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def remote_text(sftp: paramiko.SFTPClient, path: PurePosixPath) -> str:
    with sftp.open(str(path), "r") as handle:
        value = handle.read()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def parse_status(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def exec_text(client: paramiko.SSHClient, command: str, timeout: int = 30) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), output + error


def sftp_exists(sftp: paramiko.SFTPClient, path: PurePosixPath) -> bool:
    try:
        sftp.stat(str(path))
        return True
    except FileNotFoundError:
        return False


def pull_tree(sftp: paramiko.SFTPClient, remote: PurePosixPath, local: Path) -> None:
    attrs = sftp.listdir_attr(str(remote))
    local.mkdir(parents=True, exist_ok=True)
    for attr in attrs:
        remote_child = remote / attr.filename
        local_child = local / attr.filename
        if stat.S_ISDIR(attr.st_mode):
            pull_tree(sftp, remote_child, local_child)
        elif stat.S_ISREG(attr.st_mode):
            local_child.parent.mkdir(parents=True, exist_ok=True)
            temporary = local_child.with_name(f".{local_child.name}.part")
            sftp.get(str(remote_child), str(temporary))
            temporary.replace(local_child)


def pull_checkpoint(sftp: paramiko.SFTPClient, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    journals = REMOTE_ARTIFACTS / "journals"
    if sftp_exists(sftp, journals):
        pull_tree(sftp, journals, destination / "artifacts/coordinate_invariance/journals")
    for name in (
        "switch_checkpoint_identity_smoke_v1.json",
        "switch_c2_eligibility_v1.json",
        "switch_c2_calibration_v1.json",
        "switch_c2_test_v1.json",
        "switch_c2_return_bundle.tar.gz",
    ):
        remote = REMOTE_ARTIFACTS / name
        if sftp_exists(sftp, remote):
            local = destination / "artifacts/coordinate_invariance" / name
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_name(f".{local.name}.part")
            sftp.get(str(remote), str(temporary))
            temporary.replace(local)


def pull_final(sftp: paramiko.SFTPClient, destination: Path) -> dict[str, object]:
    pull_tree(sftp, REMOTE_ARTIFACTS, destination / "artifacts/coordinate_invariance")
    pull_tree(sftp, REMOTE_OPS, destination / "ops")
    bundle = destination / "artifacts/coordinate_invariance/switch_c2_return_bundle.tar.gz"
    if not bundle.is_file():
        raise RuntimeError("final evidence bundle is missing")

    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with tarfile.open(bundle, "r:gz") as archive:
        tar_members = archive.getmembers()
        members = [member.name for member in tar_members]
        for member in tar_members:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                while extracted.read(1024 * 1024):
                    pass

    result: dict[str, object] = {
        "retrieved_at": utc_now(),
        "bundle": str(bundle),
        "bundle_sha256": digest,
        "member_count": len(members),
        "members": members,
    }
    (destination / "retrieval_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def tcp_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password-env", default="AUTODL_PASSWORD")
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--shutdown-closed-checks", type=int, default=5)
    parser.add_argument("--shutdown-fallback-seconds", type=int, default=600)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.password = os.environ.get(args.password_env)
    if not args.password:
        raise SystemExit(f"missing password environment variable: {args.password_env}")
    args.destination = args.destination.resolve()

    last_status: tuple[str, str, str] | None = None
    last_checkpoint = 0.0
    final_seen_at: float | None = None
    final_pulled = False
    unavailable_checks = 0
    shutdown_observations: list[dict[str, object]] = []

    print(f"{utc_now()} local monitor started", flush=True)
    while True:
        try:
            client = connect(args)
        except (OSError, paramiko.SSHException) as exc:
            if final_seen_at is not None:
                unavailable_checks += 1
                tcp_proxy_open = tcp_is_open(args.host, args.port)
                shutdown_observations.append(
                    {
                        "checked_at": utc_now(),
                        "exception_type": type(exc).__name__,
                        "tcp_proxy_open": tcp_proxy_open,
                        "ssh_backend_available": False,
                    }
                )
                print(
                    f"{utc_now()} shutdown verification backend_unavailable="
                    f"{unavailable_checks}/{args.shutdown_closed_checks} "
                    f"tcp_proxy_open={tcp_proxy_open}",
                    flush=True,
                )
                if unavailable_checks >= args.shutdown_closed_checks:
                    summary = {
                        "verified_at": utc_now(),
                        "host": args.host,
                        "port": args.port,
                        "verification_method": "consecutive SSH backend unavailability",
                        "consecutive_unavailable_checks": unavailable_checks,
                        "observations": shutdown_observations,
                        "final_evidence_pulled": final_pulled,
                    }
                    args.destination.mkdir(parents=True, exist_ok=True)
                    (args.destination / "shutdown_verification.json").write_text(
                        json.dumps(summary, indent=2) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    print(f"{utc_now()} shutdown verified", flush=True)
                    return 0 if final_pulled else 2
            else:
                print(f"{utc_now()} connection unavailable before final state: {type(exc).__name__}", flush=True)
            time.sleep(args.interval)
            continue

        unavailable_checks = 0
        shutdown_observations = []
        try:
            sftp = client.open_sftp()
            status = parse_status(remote_text(sftp, REMOTE_STATUS))
            current = (
                status.get("state", "unknown"),
                status.get("stage", "unknown"),
                status.get("reason", "unknown"),
            )
            if current != last_status:
                print(
                    f"{utc_now()} remote state={current[0]} stage={current[1]} reason={current[2]}",
                    flush=True,
                )
                last_status = current

            if time.time() - last_checkpoint >= 300:
                pull_checkpoint(sftp, args.destination)
                last_checkpoint = time.time()
                rc, gpu = exec_text(
                    client,
                    "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw "
                    "--format=csv,noheader,nounits",
                )
                if rc == 0:
                    print(f"{utc_now()} gpu={gpu.strip()}", flush=True)

            if current[0] == "SHUTDOWN_PENDING":
                if final_seen_at is None:
                    final_seen_at = time.time()
                if not final_pulled:
                    result = pull_final(sftp, args.destination)
                    final_pulled = True
                    print(
                        f"{utc_now()} evidence verified sha256={result['bundle_sha256']} "
                        f"members={result['member_count']}",
                        flush=True,
                    )
                if time.time() - final_seen_at > args.shutdown_fallback_seconds:
                    print(f"{utc_now()} remote shutdown fallback requested", flush=True)
                    exec_text(client, "/usr/bin/shutdown", timeout=15)
            sftp.close()
        except (OSError, EOFError, paramiko.SSHException, RuntimeError) as exc:
            print(f"{utc_now()} monitor cycle error: {type(exc).__name__}: {exc}", flush=True)
        finally:
            client.close()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
