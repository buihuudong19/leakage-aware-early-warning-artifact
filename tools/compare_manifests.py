"""Compare two runs through their manifests, for example the reported run and a rerun from an empty output folder.

usage: python tools/compare_manifests.py OLD_MANIFEST NEW_MANIFEST
       python tools/compare_manifests.py --archive ARCHIVE_DIR NEW_MANIFEST

With --archive, OLD_MANIFEST is the newest manifest.json under ARCHIVE_DIR/outputs_* that records a full run (not
--quick); run_pipeline.bat and run_pipeline.sh move the outputs found before a full run there.

Each manifest lists the SHA-256 hash of every output file of its run. The script reports whether the code hashes agree
and which output files differ or occur in only one run. Some differences do not concern the content and are listed
separately:
  - run logs record clock times and durations;
  - matplotlib writes the time of creation into every PDF figure; such a PDF is compared again with the date removed;
  - gzip and zip archives store the time they were written; they are compared again after decompression.
The second comparison needs both files next to their manifests. An archive that is absent from one side (the
repository ships no outputs/data or outputs/robustness) is listed as not comparable. The exit status is 0 when the
code and all other output files agree.
"""
import gzip
import json
import re
import sys
import zipfile
from pathlib import Path

DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")


def same_content(a, b, name):
    """True or False when both files exist and their content can be compared; None otherwise."""
    if not (a.is_file() and b.is_file()):
        return None
    if name.endswith(".pdf"):
        return DATE.sub(b"", a.read_bytes()) == DATE.sub(b"", b.read_bytes())
    if name.endswith(".gz"):
        return gzip.decompress(a.read_bytes()) == gzip.decompress(b.read_bytes())
    if name.endswith(".zip"):
        with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
            na, nb = sorted(za.namelist()), sorted(zb.namelist())
            return na == nb and all(za.read(n) == zb.read(n) for n in na)
    return False


def newest_full_manifest(archive):
    for d in sorted(Path(archive).glob("outputs_*"), reverse=True):
        p = d / "manifest.json"
        if p.is_file() and not json.loads(p.read_text(encoding="utf-8")).get("quick", True):
            return p
    return None


def main():
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "--archive":
        p_old = newest_full_manifest(args[1])
        if p_old is None:
            print(f"no manifest of a full run under {args[1]}; nothing to compare")
            return 0
        p_new = Path(args[2])
    elif len(args) == 2:
        p_old, p_new = Path(args[0]), Path(args[1])
    else:
        sys.exit(__doc__)
    print(f"comparing {p_old} with {p_new}")
    old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (p_old, p_new))
    d_old, d_new = p_old.parent, p_new.parent
    print(f"old run: {old['time']}, Python {old['python']}, {old.get('platform', 'n/a')}")
    print(f"new run: {new['time']}, Python {new['python']}, {new.get('platform', 'n/a')}")
    if new.get("quick") or old.get("quick"):
        print("WARNING: one of the runs is a --quick run")
    for k in sorted(set(old["libs"]) | set(new["libs"])):
        if old["libs"].get(k) != new["libs"].get(k):
            print(f"library version differs: {k} {old['libs'].get(k)} -> {new['libs'].get(k)}")
    code_ok = old["code_sha256"] == new["code_sha256"]
    print(f"code_sha256: {'match' if code_ok else 'MISMATCH'} ({old['code_sha256'][:12]} / {new['code_sha256'][:12]})")

    h_old = {e["file"]: e["sha256"] for e in old["outputs"]}
    h_new = {e["file"]: e["sha256"] for e in new["outputs"]}
    only_old, only_new = sorted(set(h_old) - set(h_new)), sorted(set(h_new) - set(h_old))
    both = sorted(set(h_old) & set(h_new))
    differ = [f for f in both if h_old[f] != h_new[f]]
    logs, same, unknown, other = [], [], [], []
    for f in differ:
        if f.startswith("logs/"):
            logs.append(f)
            continue
        c = same_content(d_old / f, d_new / f, f) if f.endswith((".pdf", ".gz", ".zip")) else False
        (same if c else unknown if c is None and f.endswith((".gz", ".zip")) else other).append(f)
    for f in only_old:
        print(f"only in the old run: {f}")
    for f in only_new:
        print(f"only in the new run: {f}")
    for f in unknown:
        print(f"not comparable (archive with a time stamp, file absent from one run): {f}")
    for f in other:
        print(f"output differs: {f}")
    print(f"outputs: {len(both) - len(differ)} identical, {len(same)} identical apart from a time stamp "
          f"(PDF creation date, gzip or zip time), {len(logs)} run logs differ (expected), {len(unknown)} not "
          f"comparable, {len(other)} differ, {len(only_old) + len(only_new)} in one run only")
    return 0 if code_ok and not other and not only_old and not only_new else 1


if __name__ == "__main__":
    sys.exit(main())
