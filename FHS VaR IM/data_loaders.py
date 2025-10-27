# src/irm2/data_loaders.py
from __future__ import annotations
import re, numpy as np, pandas as pd

REQ = ["trade_date","contract_code","contract_month","settle","volume"]

def _pick(df, names):
    cols = {c.lower().strip(): c for c in df.columns}
    for n in names:
        c = cols.get(n.lower())
        if c: return df[c]
    return None

def _kmb(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(",","").strip()
    if s in ("","-","—"): return np.nan
    m = re.match(r"^([+-]?\d*\.?\d+)([KkMmBb]?)$", s)
    if m:
        v = float(m.group(1)); mul = {"":1,"K":1e3,"M":1e6,"B":1e9}.get(m.group(2).upper(),1)
        return v*mul
    try: return float(s)
    except: return np.nan

def load_raw_csv_to_processed(in_path: str, contract_code: str) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    date   = _pick(df, ["Date"])
    settle = _pick(df, ["Price"])
    vol    = _pick(df, ["Vol."])
    # oi     = _pick(df, ["open interest","oi"])

    if date is None or settle is None:
        raise ValueError("Need Date and one of Price/Close/Settle.")

    out = pd.DataFrame({
        "trade_date": pd.to_datetime(date, errors="coerce"),
        "contract_code": contract_code,
        "settle": pd.to_numeric(settle, errors="coerce"),
        "volume": vol.map(_kmb) if vol is not None else 0,
        # "open_interest": pd.to_numeric(oi, errors="coerce") if oi is not None else np.nan,
    }).dropna(subset=["trade_date"]).sort_values("trade_date")

    out["volume"] = out["volume"].fillna(0).astype(int)
    out["contract_month"] = out["trade_date"].dt.to_period("M").astype(str)  # ok for continuous front
    return out[REQ]

def build_front_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cm_date"] = pd.to_datetime(df["contract_month"] + "-01", errors="coerce")
    front = (df.dropna(subset=["cm_date"])
               .sort_values(["contract_code","trade_date","cm_date"])
               .groupby(["contract_code","trade_date"]).nth(0).reset_index())
    return front[["contract_code","trade_date","contract_month","settle","volume"]].rename(
        columns={"settle":"settle_front","volume":"volume_front"}
    ).sort_values(["contract_code","trade_date"]).reset_index(drop=True)


pairs = [
    ("data/raw/ICE Dutch TTF Natural Gas Futures Historical Data.csv", "TTF",     "data/processed/ice_ttf_eod.csv"),
    ("data/raw/UK NBP Natural Gas Quaterly Futures Historical Data.csv","NBP",    "data/processed/ice_nbp_eod.csv"),
    ("data/raw/Brent Oil Futures Historical Data.csv",                  "BRENT",  "data/processed/ice_brent_eod.csv"),
    ("data/raw/Euribor Futures Historical Data.csv",                    "EURIBOR","data/processed/ice_euribor_eod.csv"),
    ("data/raw/Three Month SONIA Futures Historical Data.csv",          "SONIA",  "data/processed/ice_sonia_eod.csv"),
]
for src, code, dst in pairs:
    df = load_raw_csv_to_processed(src, code)
    df.to_csv(dst, index=False)
    print("Wrote", dst, len(df))
