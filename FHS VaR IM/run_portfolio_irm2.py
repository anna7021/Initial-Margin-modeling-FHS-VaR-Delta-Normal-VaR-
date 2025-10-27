import argparse, glob, os, math
import numpy as np, pandas as pd

DEFAULTS = dict(
    alpha=0.99,
    mpor_days=[2, 5],  # margin period of risk
    ewma_lambda=0.97,
    stress_window_days=300,
    stress_weight=0.3,
    floor_abs=0.0,
    prudence_mult=1.05,
    participation_rate=0.15,
    bid_ask_bps=5.0,
    adv_lookback_days=50,
    min_lookback = 300,
)

def ewma_vol(ret, lam):
    r = pd.Series(ret).fillna(0.0).values
    var = np.zeros_like(r); v = np.nanvar(r) if len(r)>1 else 0.0
    for i,x in enumerate(r):
        v = lam*v + (1-lam)*(x*x); var[i]=v
    return np.sqrt(var)

def garch_vol(ret, omega=1e-6, alpha=0.07, beta=0.92):
    r = pd.Series(ret).fillna(0.0).values
    n = len(r)
    var = np.zeros(n)
    v = np.nanvar(r) if n>1 else 0.0
    for i,x in enumerate(r):
        v = omega + alpha*(x*x) + beta*v
        var[i] = max(v, 0.0)
    return np.sqrt(var)

def apc_blend(sig, window=250, w_stress=0.25):
    cur_vol = pd.Series(sig).astype('float64')
    stress_leg = cur_vol.rolling(window, min_periods=window).max()
    # stress_leg = roll_max.expanding(min_periods=window).max()
    blend_vol = w_stress * stress_leg + (1 - w_stress) * cur_vol
    return blend_vol

def build_front_month(df):
    df = df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['cm_date'] = pd.to_datetime(df['contract_month']+'-01', errors='coerce')
    front = (df.dropna(subset=['cm_date'])
               .sort_values(['contract_code','trade_date','cm_date'])
               .groupby(['contract_code','trade_date']).nth(0).reset_index())    # picks the front contract for that day, so you get a daily time series per asset.
    front = front[['contract_code','trade_date','contract_month','settle','volume']].rename(
        columns={'settle':'settle_front','volume':'volume_front'})
    return front.sort_values(['contract_code','trade_date']).reset_index(drop=True)

def fhs_portfolio_var(Z, sig_today, px, pos_units, alpha, h, N=10000):
    T,A = Z.shape
    losses = np.empty(N)
    for i in range(N):
        agg = np.zeros(A)
        for _ in range(h):
            t = np.random.randint(0,T)
            agg += sig_today * Z[t,:]
        pnl = -np.sum(pos_units * px * agg)
        losses[i]=pnl
    return float(np.quantile(losses, alpha))

def parametric_var(weights, cov, alpha, h):
    z = {0.975:1.96, 0.99:2.326, 0.995:2.576, 0.999:3.090}.get(alpha, 2.326)
    var = float(weights @ cov @ weights.T)
    std = math.sqrt(max(var,0.0))
    return z * std * math.sqrt(h)

def main():
    ap = argparse.ArgumentParser()
    # Make these optional and provide defaults that point to data/processed
    ap.add_argument('--eod', nargs='+', default=['data/processed/ice_*_eod.csv'],
                    help='Glob(s) of processed CSVs (default: data/processed/ice_*_eod.csv)')
    ap.add_argument('--instruments', default='data/instruments.csv', help='data/instruments.csv (default)')
    ap.add_argument('--positions', default='data/positions.csv', help='data/positions.csv (default)')
    ap.add_argument('--out', default='out/irm2_summary.csv')
    args = ap.parse_args()

    # Expand globs for EOD files and print what we found
    files = []
    for g in args.eod:
        matched = glob.glob(g)
        if not matched:
            # also try joining with working dir if user passed relative fragment
            matched = glob.glob(os.path.join(os.getcwd(), g))
        files += matched

    print(f"Using EOD globs: {args.eod}")
    print(f"Matched EOD files ({len(files)}):")
    for p in files:
        print("  ", p)

    if not files: raise SystemExit('No EOD files found. Check the --eod glob or put processed files under data/processed.')

    eods = [pd.read_csv(p) for p in files]
    raw = pd.concat(eods, ignore_index=True)
    need = {'trade_date','contract_code','contract_month','settle','volume'}
    if not need.issubset(set(c.lower() for c in raw.columns)):
        raw.columns = [c.lower() for c in raw.columns]

    front = build_front_month(raw)
    inst = pd.read_csv(args.instruments)
    pos  = pd.read_csv(args.positions)
    P = DEFAULTS

    class_map = inst.set_index('contract_code')['asset_class'].to_dict()
    codes = sorted(front['contract_code'].unique())
    energy = [c for c in codes if class_map.get(c,'Energy')=='Energy']
    fins   = [c for c in codes if class_map.get(c,'Energy')!='Energy']

    latest_px = {}
    resids = {}
    sig_today = {}
    fin_ret = {}
    vol_series = {}

    for c in codes:
        sub = front[front['contract_code']==c].sort_values('trade_date')
        px  = pd.Series(sub['settle_front'].values, index=pd.to_datetime(sub['trade_date']))
        r1  = np.log(px).diff()
        sig = pd.Series(ewma_vol(r1, P['ewma_lambda']), index=r1.index)
        sig_eff = pd.Series(apc_blend(sig, P['stress_window_days'], P['stress_weight']), index=r1.index)
        vol_series[c] = sig_eff
        latest_px[c] = float(px.dropna().iloc[-1])
        if c in energy:
            z = r1 / sig_eff.replace(0.0, np.nan)
            resids[c] = z.dropna()
            sig_today[c] = float(sig_eff.dropna().iloc[-1])
        else:
            fin_ret[c] = r1.dropna()   # Keep raw returns to build a correlation matrix later

    # align Energy residuals
    Z = None; sig_vec=None; px_vec_e=None; e_codes=[]
    if energy:
        idx = None
        for c in energy:
            idx = resids[c].index if idx is None else idx.intersection(resids[c].index)
        if idx is not None and len(idx)>=60:
            e_codes = energy
            Z = np.column_stack([resids[c].loc[idx].values for c in e_codes])    # (T x n)
            sig_vec = np.array([sig_today[c] for c in e_codes])
            px_vec_e = np.array([latest_px[c] for c in e_codes])

    # Financials covariance with APC rescale (preserve corr)
    cov_fin=None; f_codes=[]; px_vec_f=None
    if fins:
        idxf=None
        for c in fins:
            idxf = fin_ret[c].index if idxf is None else idxf.intersection(fin_ret[c].index)
        if idxf is not None and len(idxf)>=60:
            f_codes = fins
            R = np.column_stack([fin_ret[c].loc[idxf].values for c in f_codes])
            cov_curr = np.cov(R, rowvar=False)
            sig_curr = np.std(R, axis=0, ddof=1)
            sig_stress = np.array([vol_series[c].loc[idxf].quantile(0.75) for c in f_codes])
            sig_eff = np.maximum(sig_curr, sig_stress)
            denom = np.outer(sig_curr, sig_curr)
            corr = cov_curr / np.where(denom==0, np.nan, denom)    # Pearson correlation
            corr = np.nan_to_num(corr, nan=0.0)
            D = np.diag(sig_eff)
            cov_fin = D @ corr @ D   # pair wise correlation
            px_vec_f = np.array([latest_px[c] for c in f_codes])

    # build position value units (contracts * multiplier)
    pos2 = pos.merge(inst[['contract_code','contract_multiplier','bid_ask_bps','participation_rate']],
                     on='contract_code', how='left')
    pos2['contract_multiplier'] = pos2['contract_multiplier'].fillna(1.0)
    pos2['value_units'] = pos2['position_contracts'] * pos2['contract_multiplier']

    rows=[]
    for h in P['mpor_days']:
        # Energy VaR (FHS)
        VaR_e=IM_e=0.0
        if Z is not None and len(e_codes)>0:
            w_e = pos2.set_index('contract_code').reindex(e_codes)['value_units'].fillna(0.0).values
            VaR_e = fhs_portfolio_var(Z, sig_vec, px_vec_e, w_e, P['alpha'], h, N=10000)
            IM_e  = max(P['floor_abs'], P['prudence_mult']*VaR_e)

        # Financials VaR (Parametric)
        VaR_f=IM_f=0.0
        if cov_fin is not None and len(f_codes)>0:
            w_f = (pos2.set_index('contract_code').reindex(f_codes)['value_units'].fillna(0.0).values * px_vec_f)
            VaR_f = parametric_var(w_f, cov_fin, P['alpha'], h)
            IM_f  = max(P['floor_abs'], P['prudence_mult']*VaR_f)

        # Liquidity add-on (average daily volume ADV x participation → days; cost = units * half-spread_per_unit)
        liq = 0.0
        adv = {}
        for c in codes:
            sub = (front[front['contract_code'] == c]['volume_front']
                   .rolling(P['adv_lookback_days']).mean().dropna())
            adv[c] = float(sub.iloc[-1]) if not sub.empty else float('nan')
        for _, r in pos2.iterrows():
            c = r['contract_code']
            px = latest_px.get(c, np.nan)
            a = adv.get(c, np.nan)
            if not (pd.notna(px) and pd.notna(a) and a > 0):
                continue
            pr = r['participation_rate'] if pd.notna(r['participation_rate']) else P['participation_rate']
            ba = r['bid_ask_bps'] if pd.notna(r['bid_ask_bps']) else P['bid_ask_bps']
            mult = r['contract_multiplier'] if pd.notna(r['contract_multiplier']) else 1.0
            n_contracts = float(r['position_contracts'])
            daily_cap = max(1.0, pr * a)  # contracts/day
            days_to_exit = max(1.0, abs(n_contracts) / daily_cap)
            half_spread_per_contract = px * (ba / 10000.0) * 0.5 * mult
            cost_currency = abs(n_contracts) * half_spread_per_contract
            liq += cost_currency

        rows.append(dict(
            horizon_days=h, alpha=P['alpha'],
            VaR_energy=VaR_e, IM_energy=IM_e,
            VaR_financials=VaR_f, IM_financials=IM_f,
            AddOn_liquidity=liq, IM_total=IM_e+IM_f+liq
        ))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out); print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()
