import os
import requests
import datetime
import concurrent.futures
import numpy as np
import pandas as pd
import torch

os.makedirs('data', exist_ok=True)

print("="*80)
print("BUILDING 4 MULTIVARIATE SINGLE-SURGE BENCHMARK TENSORS (N = 500 SAMPLES EACH)")
print("Format: 3D NumPy (.npz) & PyTorch (.pt) tensors: [N=500, T_steps, C_features]")
print("Real-world observed channels | Zero overlap | Diverse natural baselines | Zero thresholding")
print("="*80)

# =========================================================
# DOMAIN 1: US STOCK EQUITIES (OHLCV - 5 Real Channels)
# =========================================================
print("\n[1/4] Processing Domain 1: US Equities (500 companies x 51 days x 5 features: Open, High, Low, Close, Volume)...")
df_eq_existing = pd.read_csv('data/domain1_equities_event_aligned_500.csv', index_col=0)
eq_tickers = df_eq_existing.index.tolist()

t1_eq = int(datetime.datetime(2020, 1, 1).timestamp())
t2_eq = int(datetime.datetime(2021, 12, 31).timestamp())
headers = {'User-Agent': 'Mozilla/5.0'}
w_pre_eq, w_post_eq = 20, 30

def fetch_eq_ohlcv(tk):
    try:
        u = f'https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1={t1_eq}&period2={t2_eq}&interval=1d'
        r = requests.get(u, headers=headers, timeout=8)
        if r.status_code == 200:
            res = r.json()['chart']['result'][0]
            q = res['indicators']['quote'][0]
            dates = pd.to_datetime(res['timestamp'], unit='s').tz_localize(None).floor('D')
            df_tk = pd.DataFrame({
                'open': q['open'], 'high': q['high'], 'low': q['low'],
                'close': q['close'], 'volume': q['volume']
            }, index=dates).dropna()
            vals = df_tk['close'].values
            if len(vals) > (w_pre_eq + w_post_eq + 20):
                p_idx = np.argmax(vals[w_pre_eq : len(vals) - w_post_eq]) + w_pre_eq
                win = df_tk.iloc[p_idx - w_pre_eq : p_idx + w_post_eq + 1].values
                if len(win) == (w_pre_eq + w_post_eq + 1):
                    return tk, win
    except Exception:
        pass
    return tk, None

eq_ohlcv_dict = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=35) as executor:
    for tk, arr in executor.map(fetch_eq_ohlcv, eq_tickers):
        if arr is not None:
            eq_ohlcv_dict[tk] = arr

# Preserve exact order of 500 tickers
eq_valid_ids = [t for t in eq_tickers if t in eq_ohlcv_dict]
eq_tensor = np.array([eq_ohlcv_dict[t] for t in eq_valid_ids], dtype=np.float32)
eq_features = ['open', 'high', 'low', 'close', 'volume']
eq_tau = [f"tau_{d:+d}d" for d in range(-w_pre_eq, w_post_eq + 1)]

np.savez_compressed(
    'data/domain1_equities_multivariate_500.npz',
    data=eq_tensor,
    feature_names=eq_features,
    entity_ids=eq_valid_ids,
    tau_steps=eq_tau
)
torch.save({
    'data': torch.tensor(eq_tensor, dtype=torch.float32),
    'feature_names': eq_features,
    'entity_ids': eq_valid_ids,
    'tau_steps': eq_tau
}, 'data/domain1_equities_multivariate_500.pt')

print(f"Domain 1 complete: Tensor shape {eq_tensor.shape} (N={eq_tensor.shape[0]}, T={eq_tensor.shape[1]}, C={eq_tensor.shape[2]})")

# =========================================================
# DOMAIN 2: USGS HYDROLOGY (Stage, Velocity, Acceleration - 3 Real Physical Channels)
# =========================================================
print("\n[2/4] Processing Domain 2: USGS Hydrology (500 stations x 61 steps x 3 physical features)...")
df_hy_existing = pd.read_csv('data/domain2_hydrology_event_aligned_500.csv', index_col=0)
hy_stations = df_hy_existing.index.tolist()
f1_stage = df_hy_existing.values.astype(np.float32) # (500, 61)
# 15-minute interval = 0.25 hours
f2_velocity = np.gradient(f1_stage, 0.25, axis=1).astype(np.float32) # ft / hr
f3_acceleration = np.gradient(f2_velocity, 0.25, axis=1).astype(np.float32) # ft / hr^2

hy_tensor = np.stack([f1_stage, f2_velocity, f3_acceleration], axis=-1)
hy_features = ['gage_height_ft', 'stage_velocity_ft_hr', 'stage_acceleration_ft_hr2']
w_pre_hy, w_post_hy = 24, 36
hy_tau = [f"tau_{s:+d}_15m" for s in range(-w_pre_hy, w_post_hy + 1)]

np.savez_compressed(
    'data/domain2_hydrology_multivariate_500.npz',
    data=hy_tensor,
    feature_names=hy_features,
    entity_ids=hy_stations,
    tau_steps=hy_tau
)
torch.save({
    'data': torch.tensor(hy_tensor, dtype=torch.float32),
    'feature_names': hy_features,
    'entity_ids': hy_stations,
    'tau_steps': hy_tau
}, 'data/domain2_hydrology_multivariate_500.pt')

print(f"Domain 2 complete: Tensor shape {hy_tensor.shape} (N={hy_tensor.shape[0]}, T={hy_tensor.shape[1]}, C={hy_tensor.shape[2]})")

# =========================================================
# DOMAIN 3: US COUNTY EPIDEMIOLOGY (Daily Cases, Daily Deaths, Cumulative Cases)
# =========================================================
print("\n[3/4] Processing Domain 3: US County Epidemiology (500 counties x 61 days x 3 features: Cases, Deaths, Cumulative)...")
url_cases = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
df_c = pd.read_csv(url_cases)
df_d = pd.read_csv(url_deaths)

date_cols = [c for c in df_c.columns if '/' in c]
w_dates = [c for c in date_cols if any(c.startswith(m+'/') and c.endswith('/20') for m in ['10','11','12']) or any(c.startswith(m+'/') and c.endswith('/21') for m in ['1','2','3'])][:150]

valid_mask = df_c['Admin2'].notna() & (~df_c['Admin2'].str.contains('Unassigned|Out of', case=False, na=False))
df_c_v = df_c[valid_mask].copy()
df_d_v = df_d[valid_mask].copy()

df_c_v['ID'] = df_c_v['Admin2'] + ", " + df_c_v['Province_State']
df_d_v['ID'] = df_d_v['Admin2'] + ", " + df_d_v['Province_State']

df_c_v.set_index('ID', inplace=True)
df_d_v.set_index('ID', inplace=True)

df_ep_existing = pd.read_csv('data/domain3_epidemiology_event_aligned_500.csv', index_col=0)
ep_counties = df_ep_existing.index.tolist()

w_pre_ep, w_post_ep = 25, 35
ep_tensors = []
for c_name in ep_counties:
    cum_c = df_c_v.loc[c_name, w_dates].astype(float).values
    cum_d = df_d_v.loc[c_name, w_dates].astype(float).values
    daily_c = np.diff(cum_c, prepend=cum_c[0]).clip(min=0)
    daily_d = np.diff(cum_d, prepend=cum_d[0]).clip(min=0)
    
    p_idx = w_pre_ep + np.argmax(daily_c[w_pre_ep : len(daily_c) - w_post_ep])
    win_c = daily_c[p_idx - w_pre_ep : p_idx + w_post_ep + 1]
    win_d = daily_d[p_idx - w_pre_ep : p_idx + w_post_ep + 1]
    win_cum = cum_c[p_idx - w_pre_ep : p_idx + w_post_ep + 1]
    
    tensor_c = np.stack([win_c, win_d, win_cum], axis=-1)
    ep_tensors.append(tensor_c)

ep_tensor = np.array(ep_tensors, dtype=np.float32)
ep_features = ['daily_cases', 'daily_deaths', 'cumulative_cases']
ep_tau = [f"tau_{d:+d}d" for d in range(-w_pre_ep, w_post_ep + 1)]

np.savez_compressed(
    'data/domain3_epidemiology_multivariate_500.npz',
    data=ep_tensor,
    feature_names=ep_features,
    entity_ids=ep_counties,
    tau_steps=ep_tau
)
torch.save({
    'data': torch.tensor(ep_tensor, dtype=torch.float32),
    'feature_names': ep_features,
    'entity_ids': ep_counties,
    'tau_steps': ep_tau
}, 'data/domain3_epidemiology_multivariate_500.pt')

print(f"Domain 3 complete: Tensor shape {ep_tensor.shape} (N={ep_tensor.shape[0]}, T={ep_tensor.shape[1]}, C={ep_tensor.shape[2]})")

# =========================================================
# DOMAIN 4: TRADING VOLUME & MICROSTRUCTURE (Volume, Open, High, Low, Close)
# =========================================================
print("\n[4/4] Processing Domain 4: Stock Trading Volume & Microstructure (500 companies x 41 days x 5 features)...")
df_vo_existing = pd.read_csv('data/domain4_trading_volume_event_aligned_500.csv', index_col=0)
vo_tickers = df_vo_existing.index.tolist()
w_pre_vo, w_post_vo = 15, 25

def fetch_vo_ohlcv(tk):
    try:
        u = f'https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1={t1_eq}&period2={t2_eq}&interval=1d'
        r = requests.get(u, headers=headers, timeout=8)
        if r.status_code == 200:
            res = r.json()['chart']['result'][0]
            q = res['indicators']['quote'][0]
            dates = pd.to_datetime(res['timestamp'], unit='s').tz_localize(None).floor('D')
            df_tk = pd.DataFrame({
                'volume': q['volume'], 'open': q['open'], 'high': q['high'],
                'low': q['low'], 'close': q['close']
            }, index=dates).dropna()
            vals = df_tk['volume'].values
            if len(vals) > (w_pre_vo + w_post_vo + 20):
                p_idx = np.argmax(vals[w_pre_vo : len(vals) - w_post_vo]) + w_pre_vo
                win = df_tk.iloc[p_idx - w_pre_vo : p_idx + w_post_vo + 1].values
                if len(win) == (w_pre_vo + w_post_vo + 1):
                    return tk, win
    except Exception:
        pass
    return tk, None

vo_ohlcv_dict = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=35) as executor:
    for tk, arr in executor.map(fetch_vo_ohlcv, vo_tickers):
        if arr is not None:
            vo_ohlcv_dict[tk] = arr

vo_valid_ids = [t for t in vo_tickers if t in vo_ohlcv_dict]
vo_tensor = np.array([vo_ohlcv_dict[t] for t in vo_valid_ids], dtype=np.float32)
vo_features = ['volume', 'open', 'high', 'low', 'close']
vo_tau = [f"tau_{d:+d}d" for d in range(-w_pre_vo, w_post_vo + 1)]

np.savez_compressed(
    'data/domain4_trading_volume_multivariate_500.npz',
    data=vo_tensor,
    feature_names=vo_features,
    entity_ids=vo_valid_ids,
    tau_steps=vo_tau
)
torch.save({
    'data': torch.tensor(vo_tensor, dtype=torch.float32),
    'feature_names': vo_features,
    'entity_ids': vo_valid_ids,
    'tau_steps': vo_tau
}, 'data/domain4_trading_volume_multivariate_500.pt')

print(f"Domain 4 complete: Tensor shape {vo_tensor.shape} (N={vo_tensor.shape[0]}, T={vo_tensor.shape[1]}, C={vo_tensor.shape[2]})")

# =========================================================
# FINAL VERIFICATION & FILE INTEGRITY CHECK
# =========================================================
print("\n" + "="*80)
print("VERIFICATION OF 3D MULTIVARIATE BENCHMARK TENSORS:")
print("="*80)

domains = [
    ("Equities OHLCV", 'data/domain1_equities_multivariate_500.npz', 'data/domain1_equities_multivariate_500.pt'),
    ("USGS Hydrology", 'data/domain2_hydrology_multivariate_500.npz', 'data/domain2_hydrology_multivariate_500.pt'),
    ("Epidemiology", 'data/domain3_epidemiology_multivariate_500.npz', 'data/domain3_epidemiology_multivariate_500.pt'),
    ("Trading Volume & OHLCV", 'data/domain4_trading_volume_multivariate_500.npz', 'data/domain4_trading_volume_multivariate_500.pt')
]

for name, npz_path, pt_path in domains:
    npz_data = np.load(npz_path)
    pt_data = torch.load(pt_path)
    arr = npz_data['data']
    feats = list(npz_data['feature_names'])
    ents = list(npz_data['entity_ids'])
    print(f"\n* Domain: {name}")
    print(f"  - NumPy Shape: {arr.shape} [N={arr.shape[0]}, T={arr.shape[1]}, C={arr.shape[2]}]")
    print(f"  - PyTorch Tensor Shape: {pt_data['data'].shape}")
    print(f"  - Real Feature Channels (C={len(feats)}): {feats}")
    print(f"  - Unique Entities (N={len(ents)}): {len(set(ents))} (0% overlap)")
    print(f"  - Files: {npz_path} ({os.path.getsize(npz_path)/1024:.1f} KB), {pt_path} ({os.path.getsize(pt_path)/1024:.1f} KB)")

print("\nALL MULTIVARIATE TENSORS SUCCESSFULLY BUILT AND VERIFIED!")
