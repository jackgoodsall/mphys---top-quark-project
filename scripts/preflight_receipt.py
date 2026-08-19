#!/usr/bin/env python
"""Issue/check a receipt proving preflight ran on this exact commit."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_DIR = ROOT / "slurm_outputs" / ".preflight"


def _run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _state(stage, config):
    tracked_changes = _run("git", "status", "--porcelain", "--untracked-files=no")
    untracked_code = _run(
        "git", "ls-files", "--others", "--exclude-standard", "--",
        "src", "tests", "scripts", "config", "submit.sh", "submit_true_jets.sh",
    )
    if tracked_changes or untracked_code:
        raise SystemExit("BLOCKED: preflight receipts require a clean, fully tracked worktree")
    config_path = Path(config).resolve() if config else None
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path else None
    contract = ROOT / "config" / "top_reconstruction_data_contract.yaml"
    return {
        "stage": stage,
        "commit": _run("git", "rev-parse", "HEAD"),
        "config": str(config_path) if config_path else None,
        "config_hash": config_hash,
        "contract_hash": hashlib.sha256(contract.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("issue", "check"))
    parser.add_argument("stage", choices=("audit", "training"))
    parser.add_argument("--config")
    args = parser.parse_args()
    state = _state(args.stage, args.config)
    path = RECEIPT_DIR / f"{args.stage}.json"
    if args.action == "issue":
        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        state["issued_at"] = int(time.time())
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
        return
    if not path.is_file():
        raise SystemExit(f"BLOCKED: missing preflight receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.pop("issued_at", None)
    if receipt != state:
        raise SystemExit("BLOCKED: preflight receipt does not match current code/config/contract")
    print("preflight receipt: OK")


if __name__ == "__main__":
    main()
