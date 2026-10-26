"""Move the contents of outputs/ into _archive/outputs_<date>_<time>/ so that a full run starts from an empty folder.

Called by run_pipeline.bat and run_pipeline.sh before a full run (no arguments). Uses the standard library only.
If an item cannot be moved (on Windows usually because a file is open in another program), the items moved so far are
put back, outputs/ is left as it was, and the script exits with status 1 so that the run does not start on a mixed
folder.
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def archive(root=ROOT):
    out = root / "outputs"
    items = sorted(out.iterdir()) if out.is_dir() else []
    if not items:
        print("outputs/ is empty; nothing to archive.")
        return 0
    dest = root / "_archive" / f"outputs_{time.strftime('%Y-%m-%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=False)
    moved = []
    try:
        for p in items:
            shutil.move(str(p), str(dest / p.name))
            moved.append(p.name)
    except OSError as e:
        for name in reversed(moved):                       # roll back, so that outputs/ is unchanged
            shutil.move(str(dest / name), str(out / name))
        try:
            dest.rmdir()
        except OSError:
            pass
        print(f"Could not move {p.relative_to(root)} ({e}).\n"
              "Close any program that has a file of outputs/ open (PDF viewer, Excel, Explorer preview) and run again.",
              file=sys.stderr)
        return 1
    print(f"Previous outputs moved to {dest.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(archive())
