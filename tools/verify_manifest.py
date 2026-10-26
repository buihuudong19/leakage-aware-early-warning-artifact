"""Check a checkout against outputs/manifest.json of the reported run.

usage: python tools/verify_manifest.py [OUTPUTS_DIR]

Two checks are made. The source files (src/**/*.py, run_pipeline.py, config.yaml, requirements-lock.txt) are hashed
as the pipeline hashes them, with Windows line endings normalised, and compared with `code_files` and `code_sha256`.
Every output file listed in the manifest that is present in OUTPUTS_DIR is hashed byte for byte and compared with its
entry. The repository ships outputs/tables and outputs/figures; the files under outputs/logs, outputs/data and
outputs/robustness are reported as absent. After a full run of your own, outputs/logs/pipeline.log differs from its
entry because the log is still being written when the manifest is built.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs"
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    src = sorted([*(ROOT / "src").rglob("*.py"), ROOT / "run_pipeline.py", ROOT / "config.yaml",
                  ROOT / "requirements-lock.txt"])
    code = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in src if p.is_file()}
    code_all = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    differ = sorted(k for k in set(code) | set(man["code_files"]) if code.get(k) != man["code_files"].get(k))
    for k in differ:
        print(f"code differs: {k}")
    print(f"code_sha256 {code_all[:12]} (manifest {man['code_sha256'][:12]}): "
          f"{'match' if code_all == man['code_sha256'] else 'MISMATCH'}")

    ok = bad = absent = 0
    for entry in man["outputs"]:
        p = out / entry["file"]
        if not p.is_file():
            absent += 1
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() == entry["sha256"]:
            ok += 1
        elif entry["file"] == "logs/pipeline.log":
            print("logs/pipeline.log differs (expected: the log is completed after the manifest is written)")
        else:
            bad += 1
            print(f"output differs: {entry['file']}")
    print(f"outputs: {ok} match, {bad} differ, {absent} not present")
    print(f"run recorded in the manifest: {man['time']}, Python {man['python']}, {man.get('platform', 'n/a')}")
    return 0 if code_all == man["code_sha256"] and bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
