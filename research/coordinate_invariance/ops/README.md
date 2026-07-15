# SWITCH C2 Managed Operations

These files operate the frozen `switch-c2-frozen-v5` scientific protocol. They
do not change its model, data order, estimator, thresholds, or decision rules.
The managed target is the C2 measurement gate, not GPU training.

## Components

- `run_switch_c2_managed.sh`: runs the frozen gate, records state and system
  snapshots, collects evidence on every exit, and requests instance shutdown.
- `watch_switch_c2_managed.sh`: independent GPU, disk, heartbeat, and process
  monitor with emergency evidence collection and shutdown.
- `monitor_switch_c2_remote.py`: local evidence synchronizer and independent
  shutdown verifier.
- `verify_switch_c2_assets.py`: offline verifier for every pinned model weight,
  the adapter, dataset, and required metadata.
- `stage_switch_c2_assets_remote.py`: resumable SSH uploader, archive verifier,
  remote extractor, and second integrity check for the offline cache.

## Persistence and restart behavior

The repository, virtual environment, Hugging Face cache, journals, logs, and
state files live under `/root/autodl-tmp`, the persistent data disk. Re-running
the managed entrypoint after a GPU restart reuses current config-bound passing
artifacts and resumes append-only prompt journals. File locks prevent duplicate
managed or watchdog processes.

The offline uploader writes to a `.part` file and resumes from its remote byte
count. It renames the archive and writes `SWITCH_C2_OFFLINE_READY` only after:

1. the uploaded archive SHA-256 matches the local archive;
2. extraction succeeds;
3. all six pinned model/adapter weight hashes and the pinned dataset hash pass;
4. all required model metadata is present.

The managed runner verifies those assets again on every offline-cache start.
When the official Hub is reachable, the frozen runner disables Xet and uses
bounded standard HTTP downloads because the AutoDL proxy reset Xet range
requests during attempt 4. Exact revisions and downstream file hashes remain
the identity boundary, independent of transport.

## Failure and shutdown contract

Any nonzero frozen-runner exit prevents later gates, collects the strongest
available evidence bundle, flushes logs, and enters `SHUTDOWN_PENDING`. The
primary controller and watchdog both retry AutoDL's custom `/usr/bin/shutdown`
without arguments. A local monitor retrieves and reads the final tar stream,
records its SHA-256, and requires five consecutive closed SSH checks before it
reports shutdown success.

The local monitor contains a final fallback shutdown request if the remote host
remains reachable beyond the grace window. Credentials are read only from the
`AUTODL_PASSWORD` process environment and must not be written to scripts, logs,
or repository files.

## Scientific stage order

The frozen runner performs the following fail-closed order in one invocation:

1. environment and resource preparation;
2. checkpoint identity;
3. natural-block eligibility;
4. finite-difference and numerical calibration;
5. held-out C2 test only if every prior gate passes.

A C2 pass authorizes estimator development. It does not by itself authorize
training or establish the full FCTR paper claim.
