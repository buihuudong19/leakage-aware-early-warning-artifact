"""Reading the nine raw Olist tables (with SHA-256 checks) and the feature artefacts written by the pipeline."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
TS_COLS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "items": ["shipping_limit_date"],
}


def sha256(path, decompress=False):
    """SHA-256 of a file; for a .gz file with decompress=True, the hash of the decompressed content."""
    h = hashlib.sha256()
    opener = gzip.open if decompress else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(data_dir, verify=True):
    data_dir = Path(data_dir)
    expected = json.loads((Path(__file__).parent / "raw_sha256.json").read_text())
    out, check = {}, {}
    for key, fn in TABLES.items():
        p = data_dir / fn
        if not p.is_file() and (data_dir / (fn + ".gz")).is_file():
            p = data_dir / (fn + ".gz")          # a gzip copy is accepted (e.g. the 61 MB geolocation table)
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}. Download the Olist data set from Kaggle and place the nine CSV files in {data_dir}")
        if verify:
            h = sha256(p, decompress=p.suffix == ".gz")
            check[fn] = h == expected[fn]
            if not check[fn]:
                raise ValueError(f"SHA-256 of {fn} does not match the verified release ({h})")
        df = pd.read_csv(p, encoding="utf-8-sig")
        for c in TS_COLS.get(key, []):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        out[key] = df
    return out, check


FEATURE_FILES = ("features_t0.pkl", "features_t0.pkl.xz", "features_t0.pkl.gz")


def load_features(out_dir):
    """Read the feature matrix at t0, preferring the .pkl file over the compressed .pkl.xz / .pkl.gz copies."""
    for name in FEATURE_FILES:
        p = Path(out_dir) / "data" / name
        if p.is_file():
            return pd.read_pickle(p)
    raise FileNotFoundError(f"No feature file in {Path(out_dir) / 'data'}. Run: python run_pipeline.py --stage sample")


def export_features_csv_zip(F, path):
    """Zipped CSV copy of the feature matrix for spreadsheet users; floats keep six significant digits."""
    import io as _io
    import zipfile
    buf = _io.StringIO()
    F.to_csv(buf, index=False, float_format="%.6g")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("features_t0.csv", "\ufeff" + buf.getvalue())
