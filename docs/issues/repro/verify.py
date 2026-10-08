#!/usr/bin/env python3
"""Reproduce the documented defects, failing on build errors or unrelated failures."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--with-reference", action="store_true", help="Also run pinned PyTorch controls using uv")
    parser.add_argument("--target-dir", type=Path, default=Path(tempfile.gettempdir()) / "burn-issue-audit-target")
    parser.add_argument("--miri-target-dir", type=Path, default=Path(tempfile.gettempdir()) / "burn-nms-x86-target")
    args = parser.parse_args()
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    manifest = str(HERE / "Cargo.toml")
    offline = ["--offline"] if args.offline else []
    report = {
        "status": "running",
        "meaning": "Success means the documented defects reproduced; it does not mean the library is correct.",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "source_sha256": {},
        "checks": [],
    }
    for path in [
        "crates/burn-train/src/metric/wer.rs", "crates/burn-train/src/metric/bleu.rs",
        "crates/burn-train/src/metric/rouge.rs", "crates/burn-nn/src/loss/binary_cross_entropy.rs",
        "crates/burn-nn/src/loss/gaussian_nll.rs", "crates/burn-linalg/src/functions/vector_norm.rs",
        "crates/burn-nn/src/loss/triplet_margin.rs", "crates/burn-autodiff/src/ops/tensor.rs",
        "crates/burn-core/src/data/dataloader/batch.rs", "crates/burn-core/src/data/dataloader/multithread.rs",
        "crates/burn-signal/src/functions/stft.rs", "crates/burn-vision/src/backends/cpu/nms.rs",
        "crates/burn-nn/src/modules/norm/group.rs", "crates/burn-nn/src/modules/norm/rms.rs",
        "crates/burn-std/src/network.rs",
    ]:
        report["source_sha256"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

    def save():
        (results / "verification.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, *, env=None, error=None, timeout=600):
        print(f"Running {name}", flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{name}: timed out; not counted as a reproduction") from exc
        output = result.stdout
        (results / f"{name}.log").write_text(output)
        passed = result.returncode == 0 if error is None else (
            result.returncode != 0 and error in output and "crates/burn-vision/src/backends/cpu/nms.rs:" in output
        )
        report["checks"].append({
            "name": name, "command": command, "exit_code": result.returncode,
            "expected_error": error, "passed": passed,
            "environment_overrides": ({key: env[key] for key in ["MIRIFLAGS", "RUSTFLAGS"]} if env else {}),
        })
        save()
        if not passed:
            raise RuntimeError(f"{name}: unexpected result; see {results / (name + '.log')}")
        print(f"Confirmed {name}", flush=True)

    save()
    try:
        run("format", ["cargo", "fmt", "--manifest-path", manifest, "--check"])
        run("build", ["cargo", "build", *offline, "--manifest-path", manifest,
                      "--features", "runtime,network", "--bins", "--target-dir", str(args.target_dir)])
        for number in NATIVE:
            for attempt in [1, 2]:
                run(f"issue-{number:03}-run-{attempt}",
                    [str(args.target_dir.resolve() / "debug" / f"issue-{number:03}")], timeout=30)
        for number, flags, expected in [
            (9, "-Zmiri-symbolic-alignment-check", "alignment 1, but alignment 4 is required"),
            (8, "-Zmiri-disable-alignment-check", "constructing invalid value of type [bool; 16]"),
        ]:
            env = dict(os.environ, MIRIFLAGS=flags, RUSTFLAGS="-C target-cpu=x86-64-v2")
            for attempt in [1, 2]:
                run(f"issue-{number:03}-run-{attempt}",
                    ["cargo", "+nightly", "miri", "run", *offline, "--manifest-path", manifest,
                     "--features", "vision", "--bin", f"issue-{number:03}", "--target", "x86_64-unknown-linux-gnu",
                     "--target-dir", str(args.miri_target_dir)], env=env, error=expected)
        if args.with_reference:
            run("reference-torch", ["uv", "run", "--isolated", "--python", "3.12", "--with",
                                    "torch==2.14.1", "python", str(HERE / "reference_torch.py")])
        report["status"] = "all_13_reproduced_twice"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print("All 13 documented issues reproduced twice. See results/verification.json.", flush=True)


if __name__ == "__main__":
    main()
