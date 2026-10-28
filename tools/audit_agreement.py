"""Agreement between the two coders of the literature audit (studies P01-P17 behind Table 1 of the article).

usage: python tools/audit_agreement.py SECOND_CODER_CSV [OUT_CSV]

The first coding is docs/literature_audit.csv; the second coder fills the blank form (same study IDs) without seeing
it. For each code that enters Table 1 the script reports the number of studies, the percent agreement and Cohen's
kappa, and lists every disagreement. Kappa is not defined when both coders give the same value to every study; the
percent agreement is then reported alone.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[1]
if len(sys.argv) not in (2, 3):
    sys.exit(__doc__)


def norm_split(x):
    x = str(x).strip().lower().replace(" ", "_")
    return {"random": "random_holdout", "random_holdout": "random_holdout", "temporal": "temporal",
            "time-ordered": "temporal", "time_ordered": "temporal", "not_stated": "not_stated",
            "unclear": "unclear"}.get(x, x)


def norm_pt(x):
    x = str(x).strip().lower()
    if x.startswith("stated") or x.startswith("after review"):
        return "stated"
    return "unclear" if x.startswith("unclear") else "not stated"


def norm_yn(x):
    x = str(x).strip().lower()
    return x if x in ("yes", "no") else "unclear"


def norm_purge(x):
    return "yes" if str(x).strip().lower() == "yes" else "no or not stated"


def data_first(note):
    d = re.search(r"\[dataset: ([^\]]+)\]", note).group(1)
    return "named" if d.startswith("Olist confirmed") else ("probable" if "probable" in d.lower() else "unconfirmed")


A = pd.read_csv(REPO / "docs" / "literature_audit.csv")
A = A[A.id.str.startswith("P")].set_index("id")
B = pd.read_csv(sys.argv[1], encoding="utf-8-sig").set_index("id")
B = B.loc[A.index]
missing = B.index[B[["split", "prediction_time", "post_outcome_features"]].isna().any(axis=1)].tolist()
if missing:
    sys.exit(f"the second coding is incomplete for {missing}")

fields = {
    "data_olist": (A.note.map(data_first), B.data_olist.astype(str).str.strip().str.lower()),
    "split": (A.split.map(norm_split), B.split.map(norm_split)),
    "prediction_time": (A.prediction_time.map(norm_pt), B.prediction_time.map(norm_pt)),
    "post_outcome_features": (A.post_outcome_features.map(norm_yn), B.post_outcome_features.map(norm_yn)),
    "label_availability_purge": (A.label_availability_purge.map(norm_purge), B.label_availability_purge.map(norm_purge)),
}
rows, dis = [], []
for f, (a, b) in fields.items():
    agree = float((a == b).mean())
    kappa = cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else float("nan")
    rows.append(dict(code=f, n=len(a), agreement=round(agree, 3), kappa=round(kappa, 3) if kappa == kappa else None))
    for i in a.index[a != b]:
        dis.append(dict(code=f, id=i, first=a[i], second=b[i]))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
if dis:
    print("\ndisagreements:\n" + pd.DataFrame(dis).to_string(index=False))
if len(sys.argv) == 3:
    R.to_csv(sys.argv[2], index=False)
    pd.DataFrame(dis).to_csv(Path(sys.argv[2]).with_name(Path(sys.argv[2]).stem + "_disagreements.csv"), index=False)
