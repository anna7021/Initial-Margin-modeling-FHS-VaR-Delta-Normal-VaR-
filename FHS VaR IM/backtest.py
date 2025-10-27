import argparse, os, math, glob
import numpy as np, pandas as pd
from scipy.stats import chi2
from math import log
from run_portfolio_irm2 import (
    DEFAULTS, build_front_month, ewma_vol, apc_blend,
    fhs_portfolio_var, parametric_var
)

def _safelog(x): return -1e12 if x <= 0 else log(x)

def kupiec_pof(n, x, alpha):
    """Proportion-of-failures test (LR_pof ~ chi2(1))."""
    p = 1 - alpha
    if n <= 0:
        return dict(LR=None, pvalue=None, rate=None, n=n, x=x)
    phat = x / n   # n: trading days, x: number of exceptions
    lnL0 = (n - x) * _safelog(1 - p) + x * _safelog(p)
    lnL1 = (n - x) * _safelog(1 - phat if phat < 1 else 1e-12) + x * _safelog(phat if phat > 0 else 1e-12)
    LR = max(0.0, -2.0 * (lnL0 - lnL1))
    try:
        pval = 1 - chi2.cdf(LR, 1)
    except Exception:
        pval = math.exp(-LR/2)
    return dict(LR=LR, pvalue=pval, rate=phat, n=n, x=x)

def christoffersen_ind(flags):
    """Independence test using 2-state Markov chain (LR_ind ~ chi2(1))."""
    f = np.asarray(flags, dtype=bool)
    if len(f) < 2:
        return dict(LR=None, pvalue=None, n00=0, n01=0, n10=0, n11=0)
    f0, f1 = f[:-1], f[1:]
    n00 = int((~f0 & ~f1).sum())
    n01 = int((~f0 &  f1).sum())
    n10 = int(( f0 & ~f1).sum())
    n11 = int(( f0 &  f1).sum())
    n0, n1 = n00 + n01, n10 + n11
    pi0 = n01 / n0 if n0 > 0 else 0.0    # P(exception tomorrow | no exception today)
    pi1 = n11 / n1 if n1 > 0 else 0.0    # P(exception tomorrow | exception today)
    pi  = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0
    lnL0 = (n0 + n1) * _safelog(1 - pi) + (n01 + n11) * _safelog(pi)  # simplified
    lnL1 = n00 * _safelog(1 - pi0) + n01 * _safelog(pi0) + n10 * _safelog(1 - pi1) + n11 * _safelog(pi1)
    LR = max(0.0, -2.0 * (lnL0 - lnL1))
    try:
        from scipy.stats import chi2
        pval = 1 - chi2.cdf(LR, 1)
    except Exception:
        pval = math.exp(-LR/2)
    return dict(LR=LR, pvalue=pval, n00=n00, n01=n01, n10=n10, n11=n11)

def christoffersen_cc(flags, alpha):
    """Conditional coverage: LR_cc = LR_pof + LR_ind ~ chi2(2)."""
    f = np.asarray(flags, dtype=bool)
    pof = kupiec_pof(len(f), int(f.sum()), alpha)
    ind = christoffersen_ind(f)
    if (pof["LR"] is None) or (ind["LR"] is None):
        return dict(LRcc=None, pvalue=None, pof=pof, ind=ind)
    LRcc = pof["LR"] + ind["LR"]
    try:
        from scipy.stats import chi2
        pval = 1 - chi2.cdf(LRcc, 2)
    except Exception:
        pval = math.exp(-LRcc/2)
    return dict(LRcc=LRcc, pvalue=pval, pof=pof, ind=ind)

def compute_prepped(front, inst, pos):
    inst2 = inst.copy()
    class_map = inst2.set_index("contract_code")["asset_class"].to_dict()
    codes = sorted(front["contract_code"].unique())
    energy = [c for c in codes if class_map.get(c, "Energy") == "Energy"]
    fins   = [c for c in codes if class_map.get(c, "Energy") != "Energy"]

    px  = front.pivot(index="trade_date", columns="contract_code", values="settle_front").sort_index()
    vol = front.pivot(index="trade_date", columns="contract_code", values="volume_front").sort_index()
    rlg = np.log(px).diff()

    meta = inst2.set_index("contract_code")[["contract_multiplier","bid_ask_bps","participation_rate"]]
    meta["contract_multiplier"] = meta["contract_multiplier"].fillna(1.0)
    pos2 = pos.merge(meta.reset_index(), on="contract_code", how="left")
    pos2["contract_multiplier"] = pos2["contract_multiplier"].fillna(1.0)
    pos2["value_units"] = pos2["position_contracts"] * pos2["contract_multiplier"]
    return dict(px=px, vol=vol, rlg=rlg, energy=energy, fins=fins, pos2=pos2, meta=meta)

def realized_pnl(px, pos2, t_idx, h):    # computes the portfolio’s h-day forward P&L from day t_idx, using a linearized approximation
    if t_idx + h >= len(px.index):
        return None
    # Linearized P&L with forward log-returns, to match the methodology of calculating VaR
    px_t = px.iloc[t_idx]
    r_seg = np.log(px.iloc[t_idx+1 : t_idx+1+h]).diff()
    if r_seg.isna().all(None):  # handle empty slice
        r_seg = np.log(px.iloc[[t_idx, t_idx+1]])
    r_seg = r_seg.fillna(np.log(px.iloc[t_idx+1]) - np.log(px.iloc[t_idx]))
    r_sum = r_seg.sum(axis=0)
    units = pos2.set_index("contract_code")["value_units"]
    prices= px_t.reindex(units.index).fillna(0.0)
    ret   = r_sum.reindex(units.index).fillna(0.0)
    return -float(np.sum(units.values * prices.values * ret.values))

def daily_im_at_t(t_idx, data, P, alpha, h, rng):
    px, vol, rlg = data["px"], data["vol"], data["rlg"]
    energy, fins, pos2, meta = data["energy"], data["fins"], data["pos2"], data["meta"]
    hist_idx = rlg.index[:t_idx+1]
    if len(hist_idx) < max(P["stress_window_days"]+10, P["min_lookback_days"]):
        return None

    latest_px = px.iloc[t_idx].to_dict()

    # Energy: residuals + APC
    Z = None; sig_vec=None; px_vec_e=None; e_codes=[]
    if energy:
        e_res = {}; sig_t = {}
        for c in energy:
            r = rlg[c].loc[hist_idx]
            if r.dropna().empty: continue
            sig = pd.Series(ewma_vol(r.values, P["ewma_lambda"]), index=r.index)
            sig_apc = apc_blend(sig, P["stress_window_days"], P["stress_weight"])
            z = r / sig_apc.replace(0.0, np.nan)
            z = z.dropna()
            if not z.empty:
                e_res[c] = z
                sig_t[c] = float(sig_apc.dropna().iloc[-1]) if not sig_apc.dropna().empty else np.nan
        if e_res:
            idx = None
            for _, z in e_res.items():
                idx = z.index if idx is None else idx.intersection(z.index)
            if idx is not None and len(idx) >= 60:
                e_codes = [c for c in energy if c in e_res]
                Z = np.column_stack([e_res[c].loc[idx].values for c in e_codes])
                sig_vec = np.array([sig_t[c] for c in e_codes])
                px_vec_e = np.array([latest_px.get(c, np.nan) for c in e_codes])

    # Financials: corr-preserving APC
    cov_fin=None; f_codes=[]; px_vec_f=None
    if fins:
        cols = []
        for c in fins:
            rc = rlg[c].loc[hist_idx].dropna()
            if len(rc) > 60: cols.append(c)
        if cols:
            R = rlg[cols].loc[hist_idx].dropna()
            if len(R) > 60:
                cov_curr = np.cov(R.values, rowvar=False)
                sig_curr = np.std(R.values, axis=0, ddof=1)
                # effective vol ~ 0.75-quantile of APC through history
                sig_eff = []
                for j,c in enumerate(cols):
                    rc = rlg[c].loc[hist_idx]
                    s  = pd.Series(ewma_vol(rc.values, P["ewma_lambda"]), index=rc.index)
                    sA = apc_blend(s, P["stress_window_days"], P["stress_weight"])
                    sig_eff.append(sA.quantile(0.75))
                sig_eff = np.maximum(sig_curr, np.array(sig_eff))
                denom = np.outer(sig_curr, sig_curr)
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = cov_curr / np.where(denom==0, np.nan, denom)
                corr = np.nan_to_num(corr, nan=0.0)
                D = np.diag(sig_eff)
                cov_fin = D @ corr @ D
                f_codes = cols
                px_vec_f = np.array([latest_px.get(c, np.nan) for c in f_codes])

    # Liquidity add-on (ADV over lookback window ending at t)
    liq = 0.0
    adv = {}
    if t_idx+1 >= P["adv_lookback_days"]:
        win = px.index[t_idx+1-P["adv_lookback_days"] : t_idx+1]
        for c in px.columns:
            v = vol[c].reindex(win).astype(float)
            adv[c] = float(v.mean()) if v.notna().any() else float("nan")

    pos_map = pos2.set_index("contract_code")["value_units"]

    VaR_e = IM_e = 0.0
    if Z is not None and len(e_codes)>0:
        w_e = pos_map.reindex(e_codes).fillna(0.0).values
        VaR_e = fhs_portfolio_var(Z, sig_vec, px_vec_e, w_e, alpha, h, N=P["fhs_samples"])
        IM_e  = max(P["floor_abs"], P["prudence_mult"] * VaR_e)

    VaR_f = IM_f = 0.0
    if (cov_fin is not None) and (len(f_codes)>0):
        w_f = (pos_map.reindex(f_codes).fillna(0.0).values * (px_vec_f if px_vec_f is not None else 0))
        VaR_f = parametric_var(w_f, cov_fin, alpha, h)
        IM_f  = max(P["floor_abs"], P["prudence_mult"] * VaR_f)

    if adv:
        for c, units in pos_map.items():
            px_c = latest_px.get(c, np.nan)
            a    = adv.get(c, np.nan)
            if not (px_c==px_c) or not (a==a) or a<=0: continue
            pr = float(meta.loc[c, "participation_rate"]) if (c in meta.index and pd.notna(meta.loc[c,"participation_rate"])) else P["participation_rate"]
            ba = float(meta.loc[c, "bid_ask_bps"])        if (c in meta.index and pd.notna(meta.loc[c,"bid_ask_bps"]))        else P["bid_ask_bps"]
            daily = max(1.0, pr * a)
            days  = max(1.0, abs(units)/daily)
            half_spread = px_c * (ba/10000.0) * 0.5
            liq += abs(units) * half_spread * days

    out = dict(VaR_energy=VaR_e, IM_energy=IM_e, VaR_financials=VaR_f, IM_financials=IM_f, AddOn_liquidity=liq)
    out["IM_total"] = IM_e + IM_f + liq
    return out

def basel_traffic_light_99(x):
    # canonical Basel thresholds for 1d 99% VaR with ~250 obs
    if x <= 4: return "GREEN"
    if x <= 9: return "YELLOW"
    return "RED"


def run(args):
    # Load inputs
    files = []
    for g in args.eod:
        files += (glob.glob(g) or glob.glob(os.path.join(os.getcwd(), g)))
    if not files:
        raise SystemExit("No EOD files found.")
    eods = [pd.read_csv(p) for p in files]
    raw = pd.concat(eods, ignore_index=True)
    raw.columns = [c.lower() for c in raw.columns]
    need = {"trade_date","contract_code","contract_month","settle","volume"}
    if not need.issubset(set(raw.columns)):
        missing = need - set(raw.columns)
        raise SystemExit(f"Missing columns in EOD: {missing}")

    front = build_front_month(raw)
    inst  = pd.read_csv(args.instruments)
    pos   = pd.read_csv(args.positions)

    P = DEFAULTS.copy()
    P["alpha"] = args.alpha
    P["fhs_samples"] = args.fhs_samples
    P["min_lookback_days"] = args.min_lookback
    if args.horizons: P["mpor_days"] = args.horizons

    data = compute_prepped(front, inst, pos)
    px = data["px"]
    rng = np.random.default_rng(42)

    rows = []
    for t_idx in range(len(px.index)):
        for h in P["mpor_days"]:
            im = daily_im_at_t(t_idx, data, P, P["alpha"], h, rng)
            if im is None: continue
            pnl = realized_pnl(px, data["pos2"], t_idx, h)
            if pnl is None: continue
            breach_var = pnl > (im["VaR_energy"] + im["VaR_financials"])    # True if the realized loss over the horizon is greater than the statistical VaR
            breach_im  = pnl > im["IM_total"]    # True if the realized loss exceeds the total initial margin, including prudence and liquidity add-on
            rows.append(dict(
                date=px.index[t_idx], horizon_days=h, pnl_realized=pnl,
                VaR_energy=im["VaR_energy"], IM_energy=im["IM_energy"],
                VaR_financials=im["VaR_financials"], IM_financials=im["IM_financials"],
                AddOn_liquidity=im["AddOn_liquidity"], IM_total=im["IM_total"],
                breach_var=breach_var, breach_im=breach_im
            ))

    if not rows:
        raise SystemExit("No backtest rows produced (insufficient history).")

    bt = pd.DataFrame(rows).sort_values(["horizon_days","date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_daily), exist_ok=True)
    bt.to_csv(args.out_daily, index=False)
    print(f"Wrote: {args.out_daily}  rows={len(bt)}")

    # Summaries per horizon
    sums = []
    for h, g in bt.groupby("horizon_days"):
        nV = len(g); xV = int(g["breach_var"].sum())  # nV: Number of observations (days) for this horizon; xV: Count of VaR breaches
        nI = len(g); xI = int(g["breach_im"].sum())
        kupV = kupiec_pof(nV, xV, P["alpha"])
        indV = christoffersen_ind(g["breach_var"].values)
        ccV  = christoffersen_cc(g["breach_var"].values, P["alpha"])
        kupI = kupiec_pof(nI, xI, P["alpha"])
        indI = christoffersen_ind(g["breach_im"].values)
        ccI  = christoffersen_cc(g["breach_im"].values, P["alpha"])
        traffic = basel_traffic_light_99(xV) if (P["alpha"]==0.99 and h==1) else None
        sums.append(dict(
            horizon_days=h, alpha=P["alpha"],
            n_obs_VaR=nV, exceptions_VaR=xV, hit_rate_VaR=(xV/max(nV,1)),
            kupiec_LR_VaR=kupV["LR"], kupiec_p_VaR=kupV["pvalue"],
            christ_ind_LR_VaR=indV["LR"], christ_ind_p_VaR=indV["pvalue"],
            christ_cc_LR_VaR=ccV["LRcc"], christ_cc_p_VaR=ccV["pvalue"],
            n_obs_IM=nI, exceptions_IM=xI, hit_rate_IM=(xI/max(nI,1)),
            kupiec_LR_IM=kupI["LR"], kupiec_p_IM=kupI["pvalue"],
            christ_ind_LR_IM=indI["LR"], christ_ind_p_IM=indI["pvalue"],
            christ_cc_LR_IM=ccI["LRcc"], christ_cc_p_IM=ccI["pvalue"],
            basel_traffic_light_99=traffic
        ))
    summ = pd.DataFrame(sums).sort_values("horizon_days")
    summ.to_csv(args.out_summary, index=False)
    print("Summary:\n", summ)
    print(f"Wrote: {args.out_summary}")

def main():
    ap = argparse.ArgumentParser(description="Backtest IRM2 using run_portfolio_irm2 helpers.")
    ap.add_argument('--eod', nargs='+', default=['data/processed/ice_*_eod.csv'])
    ap.add_argument('--instruments', default='data/instruments.csv')
    ap.add_argument('--positions', default='data/positions.csv')
    ap.add_argument('--alpha', type=float, default=0.99)
    ap.add_argument('--horizons', type=int, nargs='+', default=None, help='e.g. 1 2 5 (default from DEFAULTS)')
    ap.add_argument('--min_lookback', type=int, default=300)
    ap.add_argument('--fhs_samples', type=int, default=5000)
    ap.add_argument('--out_daily', default='out/backtest_daily.csv')
    ap.add_argument('--out_summary', default='out/backtest_summary.csv')
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
