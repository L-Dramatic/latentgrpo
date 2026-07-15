#!/usr/bin/env python3
"""Resumably stage and verify the frozen SWITCH C2 assets on an SSH host."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import socket
import tarfile
import time
from pathlib import Path, PurePosixPath

import paramiko

try:
    from .monitor_switch_c2_remote import PinnedHostKeyPolicy, exec_text
except ImportError:  # Direct script execution places this directory on sys.path.
    from monitor_switch_c2_remote import PinnedHostKeyPolicy, exec_text


REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/latentgrpo")
REMOTE_OPS = PurePosixPath("/root/autodl-tmp/switch-c2-ops")
REMOTE_HUB = REMOTE_ROOT / "_models/hf_home/hub"
REMOTE_ARCHIVE = PurePosixPath("/root/autodl-tmp/switch_c2_offline_snapshots.tar")
REMOTE_VERIFIER = REMOTE_OPS / "bin/verify_switch_c2_assets.py"
REMOTE_VERIFICATION = REMOTE_OPS / "evidence/switch_c2_asset_verification.json"
REMOTE_MARKER = REMOTE_HUB / "SWITCH_C2_OFFLINE_READY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path) -> None:
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
    if not members:
        raise RuntimeError("offline archive is empty")
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"unsupported archive member type: {member.name}")


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
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    return client


def remote_sha256(client: paramiko.SSHClient, path: PurePosixPath) -> str | None:
    rc, output = exec_text(client, f"sha256sum {shlex.quote(str(path))}", timeout=3600)
    if rc != 0 or not output.strip():
        return None
    return output.split()[0].lower()


def upload_resumably(
    sftp: paramiko.SFTPClient,
    local: Path,
    remote_partial: PurePosixPath,
    *,
    chunk_size: int,
) -> None:
    local_size = local.stat().st_size
    try:
        remote_size = sftp.stat(str(remote_partial)).st_size
    except FileNotFoundError:
        remote_size = 0
    if remote_size > local_size:
        bad = f"{remote_partial}.oversize.{int(time.time())}"
        sftp.rename(str(remote_partial), bad)
        remote_size = 0

    print(f"UPLOAD_RESUME offset={remote_size} total={local_size}", flush=True)
    next_report = ((remote_size // (1024**3)) + 1) * 1024**3
    with local.open("rb") as source:
        source.seek(remote_size)
        mode = "ab" if remote_size else "wb"
        with sftp.open(str(remote_partial), mode) as destination:
            destination.set_pipelined(True)
            transferred = remote_size
            while transferred < local_size:
                block = source.read(min(chunk_size, local_size - transferred))
                if not block:
                    raise RuntimeError("local archive ended before its recorded size")
                destination.write(block)
                transferred += len(block)
                if transferred >= next_report or transferred == local_size:
                    print(
                        f"UPLOAD_PROGRESS bytes={transferred} total={local_size} "
                        f"percent={100.0 * transferred / local_size:.2f}",
                        flush=True,
                    )
                    next_report += 1024**3
            destination.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password-env", default="AUTODL_PASSWORD")
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--chunk-mib", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.password = os.environ.get(args.password_env)
    if not args.password:
        raise SystemExit(f"missing password environment variable: {args.password_env}")
    archive = args.archive.resolve()
    verifier = args.verifier.resolve()
    if not archive.is_file() or not verifier.is_file():
        raise SystemExit("archive or verifier is missing")
    validate_archive(archive)
    expected_sha256 = (args.archive_sha256 or sha256(archive)).lower()
    if len(expected_sha256) != 64:
        raise SystemExit("archive SHA-256 must contain 64 hexadecimal characters")
    print(
        f"LOCAL_ARCHIVE size={archive.stat().st_size} sha256={expected_sha256}",
        flush=True,
    )

    client = connect(args)
    try:
        rc, output = exec_text(
            client,
            f"mkdir -p {shlex.quote(str(REMOTE_OPS / 'bin'))} "
            f"{shlex.quote(str(REMOTE_OPS / 'evidence'))} {shlex.quote(str(REMOTE_HUB))}",
        )
        if rc != 0:
            raise RuntimeError(output)
        sftp = client.open_sftp()
        sftp.put(str(verifier), str(REMOTE_VERIFIER))
        sftp.chmod(str(REMOTE_VERIFIER), 0o700)

        existing_sha256 = remote_sha256(client, REMOTE_ARCHIVE)
        if existing_sha256 != expected_sha256:
            if existing_sha256 is not None:
                bad = f"{REMOTE_ARCHIVE}.bad.{int(time.time())}"
                sftp.rename(str(REMOTE_ARCHIVE), bad)
            partial = PurePosixPath(f"{REMOTE_ARCHIVE}.part")
            upload_resumably(
                sftp,
                archive,
                partial,
                chunk_size=args.chunk_mib * 1024 * 1024,
            )
            actual_sha256 = remote_sha256(client, partial)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"remote archive hash mismatch: {actual_sha256} != {expected_sha256}"
                )
            sftp.rename(str(partial), str(REMOTE_ARCHIVE))
        sftp.close()

        extract = (
            f"tar -xf {shlex.quote(str(REMOTE_ARCHIVE))} "
            f"-C {shlex.quote(str(REMOTE_HUB))}"
        )
        rc, output = exec_text(client, extract, timeout=3600)
        if rc != 0:
            raise RuntimeError(f"remote extraction failed: {output}")
        verify = (
            f"/root/miniconda3/bin/python {shlex.quote(str(REMOTE_VERIFIER))} "
            f"--hub-cache {shlex.quote(str(REMOTE_HUB))} "
            f"--output {shlex.quote(str(REMOTE_VERIFICATION))}"
        )
        rc, output = exec_text(client, verify, timeout=3600)
        print(output, end="", flush=True)
        if rc != 0:
            raise RuntimeError("remote asset verification failed")
        marker = (
            f"printf '%s\\n' {shlex.quote('archive_sha256=' + expected_sha256)} "
            f"> {shlex.quote(str(REMOTE_MARKER))} && sync"
        )
        rc, output = exec_text(client, marker, timeout=120)
        if rc != 0:
            raise RuntimeError(output)
        print("REMOTE_ASSETS_READY", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, socket.error, paramiko.SSHException) as error:
        raise SystemExit(f"SSH staging failed: {error}")
