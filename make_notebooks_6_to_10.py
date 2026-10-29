import os
import nbformat as nbf

os.makedirs('data', exist_ok=True)

def create_nb():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    }
    return nb

# ==============================================================================
# NOTEBOOK 6: S&P 500 Multi-Decade Market Crashes & Flash Panics (1985–Present)
# ==============================================================================
nb6 = create_nb()
nb6.cells = [
    nbf.v4.new_markdown_cell(r"""# 📉 Multi-Decade Financial Crashes: S&P 500 Market Shocks (1985–Present)
## Long-Horizon Macro Equity Drawdowns, Flash Crashes & Liquidity Panics

### 1. Incident Background & Multi-Spike Characteristics
Unlike single-event case studies, modern financial markets have experienced recurring catastrophic crash episodes and volatility spikes across decades. When systemic shocks hit, equities exhibit **asymmetric downside velocity**: markets fall dramatically faster than they rise.

Key historical crash episodes in this 40-year benchmark include:
* **1987 "Black Monday" (Oct 19, 1987):** The largest single-day market crash in modern history (**-20.5% in a single session**).
* **1997 Asian Financial Crisis & 1998 Russian Default / LTCM:** Contagion shocks triggering emergency Fed rate cuts.
* **2000–2002 Dot-Com Bubble Collapse:** Extended multi-year crash with multiple steep cascade drawdowns totaling **-49.1%**.
* **2008 Global Financial Crisis (Lehman Collapse):** Cascading banking failures and systemic margin liquidation, producing consecutive daily drops of **-7% to -9%** and a **-56.8%** peak-to-trough collapse.
* **2010 Flash Crash (May 6, 2010):** High-frequency algorithmic liquidity vacuum driving an intraday **-9% plunge** in 36 minutes.
* **2011 US Sovereign Debt Downgrade:** S&P downgrades US credit rating, causing a **-6.7%** single-day plunge.
* **2020 COVID-19 Flash Crash (Feb–March 2020):** Fastest 30% drop in Wall Street history, triggering 4 NYSE circuit-breaker halts in 10 days, including a **-12.0%** crash on March 16.
* **2022 Inflation & Rate-Shock Bear Market:** Aggressive global monetary tightening driving an 11-month **-25.4%** drawdown.
* **August 5, 2024 Global Carry Trade Unwind:** Japanese Yen rate hike triggering a sudden global cross-asset liquidation shock.

### 2. Target Variables ($y$)
* **`SP500_close`**: S&P 500 daily closing price level.
* **`return_pct`**: Daily percentage return ($r_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100$).
* **`drawdown_pct`**: Realized peak-to-date drawdown ($DD_t = \frac{P_t - \max_{s \le t} P_s}{\max_{s \le t} P_s} \times 100$).

### 3. Data Source
* Direct Yahoo Finance REST API (`^GSPC`, 1985–Present).
* Fully reproducible with offline fallback caching to `./data/`.
"""),
    nbf.v4.new_code_cell("""import os
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Libraries loaded and environment initialized successfully!")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion & Local Caching"""),
    nbf.v4.new_code_cell("""def fetch_sp500_multi_decade(start_year=1985, cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'sp500_daily_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached S&P 500 daily telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    print(f"Fetching S&P 500 daily price history from {start_year} to present...")
    t1 = int(datetime.datetime(start_year, 1, 1).timestamp())
    t2 = int(datetime.datetime.now().timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={t1}&period2={t2}&interval=1d'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    res = response.json()['chart']['result'][0]
    
    dates = pd.to_datetime(res['timestamp'], unit='s').tz_localize(None).floor('D')
    quote = res['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'date': dates,
        'SP500_open': quote['open'],
        'SP500_high': quote['high'],
        'SP500_low': quote['low'],
        'SP500_close': quote['close'],
        'SP500_vol': quote['volume']
    }).dropna(subset=['SP500_close']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df)} trading days of S&P 500 data.")
    return df

df_sp = fetch_sp500_multi_decade()
df_sp.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Multi-Decade Crash Feature Engineering & Shock Quantification"""),
    nbf.v4.new_code_cell("""df_proc = df_sp.copy().sort_values('date').reset_index(drop=True)

# 1. Daily arithmetic return and log return
df_proc['return_pct'] = df_proc['SP500_close'].pct_change() * 100.0
df_proc['log_return'] = np.log(df_proc['SP500_close'] / df_proc['SP500_close'].shift(1))

# 2. Cumulative Rolling Peak (High-Water Mark) and Drawdown Profile
df_proc['peak_to_date'] = df_proc['SP500_close'].cummax()
df_proc['drawdown_pct'] = ((df_proc['SP500_close'] - df_proc['peak_to_date']) / df_proc['peak_to_date']) * 100.0

# 3. Rolling 20-day Annualized Realized Volatility
df_proc['volatility_20d_ann'] = df_proc['return_pct'].rolling(20).std() * np.sqrt(252)

# 4. Rolling Z-Score of Daily Returns (60-day window)
roll_mean = df_proc['return_pct'].rolling(60).mean()
roll_std = df_proc['return_pct'].rolling(60).std()
df_proc['return_zscore_60d'] = (df_proc['return_pct'] - roll_mean) / roll_std

# 5. Automated Multi-Crash Identification
# Flag days with extreme single-day plunge (<= -4.0%)
df_proc['is_crash_day'] = (df_proc['return_pct'] <= -4.0).astype(int)

# Flag severe bear market regime (Drawdown <= -20%)
df_proc['is_bear_regime'] = (df_proc['drawdown_pct'] <= -20.0).astype(int)

print(f"Total Trading Days Analyzed: {len(df_proc):,}")
print(f"Single-Day Flash Crash Days (Return <= -4%): {df_proc['is_crash_day'].sum()}")
print(f"Bear Market Days (Drawdown <= -20%):        {df_proc['is_bear_regime'].sum()}")
print(f"Maximum Historical Drawdown:                {df_proc['drawdown_pct'].min():.2f}%")
print(f"Worst Single-Day Return:                    {df_proc['return_pct'].min():.2f}% on {df_proc.loc[df_proc['return_pct'].idxmin(), 'date'].strftime('%Y-%m-%d')}")
df_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of 40 Years of Crashes"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Macro Price Trajectory (Log Scale) with Historic Shock Annotations
ax1 = axes[0]
ax1.plot(df_proc['date'], df_proc['SP500_close'], color='#1f77b4', lw=1.6, label='S&P 500 Daily Close')
ax1.set_yscale('log')
ax1.set_title("S&P 500 Index (1985–Present): 40-Year Macro Trajectory & Historical Crash Shocks", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Index Level (Log Scale)", fontsize=11, fontweight='bold')

# Historical Crash Milestones
crash_annotations = [
    ('1987-10-19', '1987 Black Monday\\n(-20.5%)', 224, 'bottom'),
    ('2002-10-09', 'Dot-Com Bottom\\n(-49.1%)', 776, 'bottom'),
    ('2008-11-20', '2008 GFC / Lehman\\n(-56.8%)', 752, 'bottom'),
    ('2011-08-08', '2011 US Downgrade\\n(-6.7%)', 1119, 'top'),
    ('2020-03-23', 'COVID Flash Crash\\n(-33.9%)', 2237, 'bottom'),
    ('2022-10-12', '2022 Inflation Shock\\n(-25.4%)', 3577, 'bottom'),
    ('2024-08-05', 'Yen Unwind Shock\\n(-3.0%)', 5186, 'top')
]

for date_str, label, y_pos, va in crash_annotations:
    dt = pd.to_datetime(date_str)
    if dt in df_proc['date'].values:
        actual_val = df_proc.loc[df_proc['date'] == dt, 'SP500_close'].values[0]
        ax1.scatter(dt, actual_val, color='red', s=45, zorder=5)
        ax1.annotate(label, xy=(dt, actual_val), xytext=(dt, actual_val * (0.6 if va=='bottom' else 1.35)),
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.2),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='red'))

ax1.legend(loc='upper left', frameon=True)

# 2. Historical Drawdown Profile (The "Underwater" Chart)
ax2 = axes[1]
ax2.plot(df_proc['date'], df_proc['drawdown_pct'], color='#d62728', lw=1.4, label='Peak-to-Trough Drawdown (%)')
ax2.fill_between(df_proc['date'], 0, df_proc['drawdown_pct'], color='#d62728', alpha=0.3)
ax2.axhline(-20.0, color='orange', ls='--', lw=1.2, label='Bear Market Threshold (-20%)')
ax2.axhline(-50.0, color='darkred', ls=':', lw=1.2, label='Mega-Crash Threshold (-50%)')
ax2.set_ylabel("Drawdown from High (%)", fontsize=11, fontweight='bold')
ax2.set_title("Historical Drawdown Profile & Bear Market Severity", fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', frameon=True)

# 3. Daily Percentage Return Needles (Impulse Shocks)
ax3 = axes[2]
crash_mask = df_proc['return_pct'] <= -4.0
surge_mask = df_proc['return_pct'] >= 4.0
normal_mask = ~crash_mask & ~surge_mask

ax3.plot(df_proc.loc[normal_mask, 'date'], df_proc.loc[normal_mask, 'return_pct'], 
         color='gray', alpha=0.5, lw=0.6, label='Daily Returns (|r| < 4%)')
ax3.scatter(df_proc.loc[crash_mask, 'date'], df_proc.loc[crash_mask, 'return_pct'], 
            color='red', s=25, label='Flash Crash Drops (<= -4%)', zorder=4)
ax3.scatter(df_proc.loc[surge_mask, 'date'], df_proc.loc[surge_mask, 'return_pct'], 
            color='green', s=25, label='Relief Rebound Surges (>= +4%)', zorder=4)
ax3.axhline(0, color='black', lw=0.8)
ax3.set_ylabel("Daily Return (%)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/sp500_crashes_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Statistical Crash Anatomy: Top 10 Single-Day Market Plunges"""),
    nbf.v4.new_code_cell("""# Top 10 worst single-day drops
top_drops = df_proc.sort_values('return_pct').head(10)[['date', 'SP500_close', 'return_pct', 'drawdown_pct', 'volatility_20d_ann']]
top_drops['date'] = top_drops['date'].dt.strftime('%Y-%m-%d')
print("=== Top 10 Steepest Single-Day S&P 500 Market Crashes (1985–Present) ===")
print(top_drops.to_string(index=False))

# Export clean processed dataset
out_path = 'data/sp500_crashes_processed.csv'
df_proc.to_csv(out_path, index=False)
print(f"\\nProcessed multi-crash dataset successfully saved to: {out_path}")""")
]

with open('06_sp500_multi_decade_market_crashes.ipynb', 'w') as f:
    nbf.write(nb6, f)
print("Saved 06_sp500_multi_decade_market_crashes.ipynb")


# ==============================================================================
# NOTEBOOK 7: CBOE Volatility Index (VIX) Panic Spikes (1990–Present)
# ==============================================================================
nb7 = create_nb()
nb7.cells = [
    nbf.v4.new_markdown_cell(r"""# ⚡ The "Fear Gauge" Volatility Spikes: CBOE VIX Telemetry (1990–Present)
## Multi-Decade Asymmetric Jump Processes, Panic Spikes & Mean-Reverting Decays

### 1. Incident Background & Multi-Spike Structure
The **CBOE Volatility Index (VIX)** is Wall Street's premier benchmark for equity market volatility and investor panic, derived from real-time S&P 500 index option prices. 

Unlike prices (which drift upward with economic growth), the VIX is a **stationary, mean-reverting jump-diffusion process**. It idles in quiet, tranquil regimes ($12 \le \text{VIX} \le 20$), but when market liquidity dries up or systemic fear explodes, it exhibits **vertical right-skewed upward spikes** ($40 \rightarrow 85+$), followed by power-law cooling decays.

Major historical panic spikes captured in this 35-year dataset:
* **1990 Gulf War Shock:** VIX spiked to **36.4**.
* **1997 Asian Financial Crisis:** Spike to **38.2**.
* **1998 Russian Debt Default & Long-Term Capital Management (LTCM) Implosion:** VIX reached **45.7**.
* **2001 Dot-Com Bubble Collapse & 9/11 Attacks:** VIX elevated to **43.7**.
* **2002 WorldCom & Enron Corporate Accounting Scandals:** Reached **45.1**.
* **2008 Lehman Brothers / Global Financial Crisis:** Record historic spike to **80.86** close (89.53 intraday).
* **2010 Flash Crash:** Intraday surge to **45.8**.
* **2011 US Credit Rating Downgrade & Eurozone Debt Crisis:** Spiked to **48.0**.
* **February 2018 "Volmageddon":** Single-day surge of **+115%** (largest single-day percentage explosion in history) as inverse volatility products collapsed.
* **March 2020 COVID-19 Panic:** All-time closing record spike to **82.69** on March 16, 2020.
* **August 5, 2024 Yen Carry Trade Unwind:** Intraday explosion to **65.73** (largest opening jump since 2008).

### 2. Target Variables ($y$)
* **`VIX_close`**: Daily CBOE Volatility Index close level.
* **`vix_daily_jump`**: Absolute daily point change ($\Delta y = VIX_t - VIX_{t-1}$).
* **`regime`**: Discrete categorical state (Tranquil, Elevated, Panic, Extreme Crisis).

### 3. Data Source
* Direct Yahoo Finance REST API (`^VIX`, 1990–Present).
* Fully reproducible with offline fallback caching to `./data/`.
"""),
    nbf.v4.new_code_cell("""import os
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Environment initialized successfully!")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion & Local Caching"""),
    nbf.v4.new_code_cell("""def fetch_vix_telemetry(start_year=1990, cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'cboe_vix_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached VIX telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    print(f"Fetching CBOE VIX historical series from {start_year} to present...")
    t1 = int(datetime.datetime(start_year, 1, 1).timestamp())
    t2 = int(datetime.datetime.now().timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1={t1}&period2={t2}&interval=1d'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    res = response.json()['chart']['result'][0]
    
    dates = pd.to_datetime(res['timestamp'], unit='s').tz_localize(None).floor('D')
    quote = res['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'date': dates,
        'VIX_open': quote['open'],
        'VIX_high': quote['high'],
        'VIX_low': quote['low'],
        'VIX_close': quote['close'],
        'VIX_vol': quote['volume']
    }).dropna(subset=['VIX_close']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df)} trading days of CBOE VIX telemetry.")
    return df

df_vix = fetch_vix_telemetry()
df_vix.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Multi-Spike Feature Engineering & Volatility Regime Segmentation"""),
    nbf.v4.new_code_cell("""df_vix_proc = df_vix.copy().sort_values('date').reset_index(drop=True)

# 1. First differences / Point Jump Velocity
df_vix_proc['vix_point_change'] = df_vix_proc['VIX_close'].diff()
df_vix_proc['vix_pct_change'] = df_vix_proc['VIX_close'].pct_change() * 100.0

# 2. Intraday Range / Spread (High - Low)
df_vix_proc['intraday_range'] = df_vix_proc['VIX_high'] - df_vix_proc['VIX_low']

# 3. Rolling 50-day Baseline & Z-Score
roll_mean_50 = df_vix_proc['VIX_close'].rolling(50).mean()
roll_std_50 = df_vix_proc['VIX_close'].rolling(50).std()
df_vix_proc['vix_zscore_50d'] = (df_vix_proc['VIX_close'] - roll_mean_50) / roll_std_50

# 4. Volatility Regime Categorization
# - Tranquil: VIX < 20
# - Elevated: 20 <= VIX < 30
# - Panic / Stress: 30 <= VIX < 50
# - Systemic Crisis: VIX >= 50
conditions = [
    (df_vix_proc['VIX_close'] < 20),
    (df_vix_proc['VIX_close'] >= 20) & (df_vix_proc['VIX_close'] < 30),
    (df_vix_proc['VIX_close'] >= 30) & (df_vix_proc['VIX_close'] < 50),
    (df_vix_proc['VIX_close'] >= 50)
]
choices = ['Tranquil (<20)', 'Elevated (20-30)', 'Panic (30-50)', 'Crisis (>=50)']
df_vix_proc['regime'] = np.select(conditions, choices, default='Tranquil (<20)')

# 5. Automated Peak Detection (Find prominent local spikes)
peak_indices, properties = find_peaks(df_vix_proc['VIX_close'].values, height=35, distance=60, prominence=10)
df_vix_proc['is_spike_peak'] = 0
df_vix_proc.loc[peak_indices, 'is_spike_peak'] = 1

print(f"Total Trading Days Analyzed: {len(df_vix_proc):,}")
print(f"Detected Prominent Panic Peaks (VIX >= 35): {len(peak_indices)}")
print(f"Days in Extreme Crisis Regime (VIX >= 50):   {(df_vix_proc['VIX_close'] >= 50).sum()}")
print(f"All-Time High VIX Close:                    {df_vix_proc['VIX_close'].max():.2f} on {df_vix_proc.loc[df_vix_proc['VIX_close'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
df_vix_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of 35 Years of Volatility Spikes"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Full 35-Year VIX Heartbeat with Regime Bands
ax1 = axes[0]
ax1.plot(df_vix_proc['date'], df_vix_proc['VIX_close'], color='#2b5c8f', lw=1.2, label='VIX Daily Close')

# Regime thresholds
ax1.axhline(20, color='gold', ls='--', lw=1.0, alpha=0.8, label='Elevated Threshold (20)')
ax1.axhline(30, color='darkorange', ls='--', lw=1.2, alpha=0.8, label='Panic Threshold (30)')
ax1.axhline(50, color='red', ls='--', lw=1.4, alpha=0.9, label='Systemic Crisis Threshold (50)')

# Highlight Crisis Zones (VIX >= 50)
ax1.fill_between(df_vix_proc['date'], 50, df_vix_proc['VIX_close'], 
                 where=(df_vix_proc['VIX_close'] >= 50), color='red', alpha=0.35, label='Extreme Panic Zone')

# Annotate Major Historic Spikes
vix_events = [
    ('1990-10-11', '1990 Gulf War (36.4)', 36.4),
    ('1998-10-08', '1998 LTCM Crisis (45.7)', 45.7),
    ('2002-08-05', '2002 Enron/WorldCom (45.1)', 45.1),
    ('2008-11-20', '2008 Lehman GFC (80.86)', 80.86),
    ('2011-08-08', '2011 US Downgrade (48.0)', 48.0),
    ('2018-02-05', '2018 Volmageddon (37.3)', 37.3),
    ('2020-03-16', '2020 COVID Peak (82.69)', 82.69),
    ('2024-08-05', '2024 Yen Shock (38.6)', 38.6)
]

for date_str, label, y_val in vix_events:
    dt = pd.to_datetime(date_str)
    if dt in df_vix_proc['date'].values:
        actual = df_vix_proc.loc[df_vix_proc['date'] == dt, 'VIX_close'].values[0]
        ax1.scatter(dt, actual, color='red', s=45, zorder=5)
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual + 6),
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.1),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax1.set_title("CBOE Volatility Index (VIX): 35-Year Multi-Spike Timeline (1990–Present)", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("VIX Level", fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, ncol=2)

# 2. Rolling 50-Day Z-Score (Departure from Calm Regimes)
ax2 = axes[1]
ax2.plot(df_vix_proc['date'], df_vix_proc['vix_zscore_50d'], color='#8c564b', lw=1.2, label='VIX Rolling 50d Z-Score')
ax2.axhline(0, color='black', lw=0.8)
ax2.axhline(3.0, color='red', ls='--', lw=1.2, label='+3-Sigma Outlier Threshold')
ax2.fill_between(df_vix_proc['date'], 0, df_vix_proc['vix_zscore_50d'], 
                 where=(df_vix_proc['vix_zscore_50d'] > 3.0), color='red', alpha=0.3)
ax2.set_ylabel("Z-Score (sigma)", fontsize=11, fontweight='bold')
ax2.set_title("Standardized Volatility Jump Impulses (50-Day Rolling Z-Score)", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Daily Velocity / Point Changes (Asymmetric Shock Waves)
ax3 = axes[2]
ax3.plot(df_vix_proc['date'], df_vix_proc['vix_point_change'], color='#7f7f7f', lw=0.8, alpha=0.6)
ax3.fill_between(df_vix_proc['date'], 0, df_vix_proc['vix_point_change'], 
                 where=(df_vix_proc['vix_point_change'] >= 5.0), color='darkred', alpha=0.7, label='Severe Jump Hike (>= +5 pts)')
ax3.fill_between(df_vix_proc['date'], 0, df_vix_proc['vix_point_change'], 
                 where=(df_vix_proc['vix_point_change'] <= -5.0), color='teal', alpha=0.7, label='Post-Crisis Cooldown (<= -5 pts)')
ax3.axhline(0, color='black', lw=0.8)
ax3.set_ylabel("Daily Delta (Points)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/cboe_vix_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Statistical Regime Distribution & Peak Ranking Table"""),
    nbf.v4.new_code_cell("""# Summary table of the top 12 historic VIX peaks
peak_df = df_vix_proc[df_vix_proc['is_spike_peak'] == 1].sort_values('VIX_close', ascending=False).head(12)
peak_summary = peak_df[['date', 'VIX_close', 'VIX_high', 'vix_point_change', 'vix_zscore_50d', 'regime']].copy()
peak_summary['date'] = peak_summary['date'].dt.strftime('%Y-%m-%d')
print("=== Top Historical CBOE VIX Panic Spike Peaks ===")
print(peak_summary.to_string(index=False))

# Regime breakdown
print("\\n=== Historical Regime Distribution ===")
regime_counts = df_vix_proc['regime'].value_counts()
regime_pcts = df_vix_proc['regime'].value_counts(normalize=True) * 100
for reg in ['Tranquil (<20)', 'Elevated (20-30)', 'Panic (30-50)', 'Crisis (>=50)']:
    cnt = regime_counts.get(reg, 0)
    pct = regime_pcts.get(reg, 0.0)
    print(f"  {reg:<20}: {cnt:>5} days ({pct:>5.1f}%)")

# Save processed dataset
out_path = 'data/cboe_vix_processed.csv'
df_vix_proc.to_csv(out_path, index=False)
print(f"\\nProcessed VIX dataset successfully saved to: {out_path}")""")
]

with open('07_cboe_vix_volatility_panic_spikes.ipynb', 'w') as f:
    nbf.write(nb7, f)
print("Saved 07_cboe_vix_volatility_panic_spikes.ipynb")


# ==============================================================================
# NOTEBOOK 8: Bitcoin Multi-Cycle Crashes and Surges (2015–Present)
# ==============================================================================
nb8 = create_nb()
nb8.cells = [
    nbf.v4.new_markdown_cell(r"""# 🪙 Crypto Hyper-Volatility: Bitcoin Multi-Cycle Booms & Collapses (2015–Present)
## Parabolic Speculative Surges, Multi-Day Flash Crashes & -80% Drawdowns

### 1. Incident Background & Multi-Cycle Anatomy
Cryptocurrency markets represent the highest-volatility liquid asset class in existence. Because crypto operates **24/7/365 without exchange circuit breakers, price limit collars, or government backstops**, it exhibits extreme recurring cycles: parabolic upside price explosions followed by catastrophic flash crashes and multi-year "crypto winters".

Major recurring spikes and crashes featured in this dataset:
* **2017 Mania Run-up & 2018 Burst:** Surged from **$1,000 $\rightarrow$ $19,800** (+1,880%), followed by an **-84.0%** collapse down to **$3,150** in late 2018.
* **March 12, 2020 "Black Thursday" Liquidity Liquidation:** Cascading automated margin calls drove a **-50.0% flash drop** in under 24 hours ($7,900 $\rightarrow$ $3,850).
* **Spring 2021 All-Time High Run:** Surged to **$64,800** in April 2021.
* **May 2021 China Mining Ban & ESG Shock:** Plunged from **$64,800 $\rightarrow$ $30,000** (**-53.7%**) in 10 days.
* **November 2021 Cycle Peak:** Reached **$68,789** peak before entering the 2022 Contagion Cascade.
* **2022 Crypto Contagion Triple-Crash:**
  1. *May 2022:* Terra/Luna algorithmic stablecoin death spiral ($40,000 $\rightarrow$ $28,000).
  2. *June 2022:* Celsius Network and Three Arrows Capital (3AC) bankruptcy ($28,000 $\rightarrow$ $19,000).
  3. *November 2022:* FTX & Alameda Research fraud collapse ($21,000 $\rightarrow$ **$15,500**, an overall **-77.5%** drawdown).
* **2024 Institutional Spot ETF Inflow Surge:** Vertical expansion to **$73,700+** followed by sharp $10k+ flash pullbacks.

### 2. Target Variables ($y$)
* **`BTC_close`**: Daily Bitcoin / US Dollar closing price.
* **`daily_return_pct`**: 24-hour percentage price return.
* **`drawdown_pct`**: Realized peak-to-date drawdown percentage.
* **`mayer_multiple`**: Ratio of price to its 200-day Simple Moving Average ($P_t / \text{SMA}_{200}$).

### 3. Data Source
* Direct Yahoo Finance REST API (`BTC-USD`, 2015–Present).
* Fully reproducible with offline fallback caching to `./data/`.
"""),
    nbf.v4.new_code_cell("""import os
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Environment initialized successfully!")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion & Local Caching"""),
    nbf.v4.new_code_cell("""def fetch_bitcoin_telemetry(start_year=2015, cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'btc_usd_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached Bitcoin telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    print(f"Fetching Bitcoin price history from {start_year} to present...")
    t1 = int(datetime.datetime(start_year, 1, 1).timestamp())
    t2 = int(datetime.datetime.now().timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?period1={t1}&period2={t2}&interval=1d'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    res = response.json()['chart']['result'][0]
    
    dates = pd.to_datetime(res['timestamp'], unit='s').tz_localize(None).floor('D')
    quote = res['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'date': dates,
        'BTC_open': quote['open'],
        'BTC_high': quote['high'],
        'BTC_low': quote['low'],
        'BTC_close': quote['close'],
        'BTC_vol': quote['volume']
    }).dropna(subset=['BTC_close']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df)} daily bars of Bitcoin telemetry.")
    return df

df_btc = fetch_bitcoin_telemetry()
df_btc.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Multi-Cycle Feature Engineering & Crash Detection"""),
    nbf.v4.new_code_cell("""df_btc_proc = df_btc.copy().sort_values('date').reset_index(drop=True)

# 1. Daily return metrics
df_btc_proc['daily_return_pct'] = df_btc_proc['BTC_close'].pct_change() * 100.0
df_btc_proc['log_return'] = np.log(df_btc_proc['BTC_close'] / df_btc_proc['BTC_close'].shift(1))

# 2. Cumulative Cycle Highs & Drawdown Profile
df_btc_proc['all_time_high'] = df_btc_proc['BTC_close'].cummax()
df_btc_proc['drawdown_pct'] = ((df_btc_proc['BTC_close'] - df_btc_proc['all_time_high']) / df_btc_proc['all_time_high']) * 100.0

# 3. 200-day Simple Moving Average & Mayer Multiple (Overheat / Undervalue Indicator)
df_btc_proc['sma_200'] = df_btc_proc['BTC_close'].rolling(200).mean()
df_btc_proc['mayer_multiple'] = df_btc_proc['BTC_close'] / df_btc_proc['sma_200']

# 4. Rolling 30-Day Realized Annualized Volatility
df_btc_proc['volatility_30d_ann'] = df_btc_proc['daily_return_pct'].rolling(30).std() * np.sqrt(365)

# 5. Flash Crash & Parabolic Surge Identifiers
df_btc_proc['is_flash_crash'] = (df_btc_proc['daily_return_pct'] <= -10.0).astype(int)
df_btc_proc['is_mega_surge'] = (df_btc_proc['daily_return_pct'] >= 10.0).astype(int)

print(f"Total Daily Observations:                     {len(df_btc_proc):,}")
print(f"Extreme Flash Crash Days (Drop <= -10% in 1 day): {df_btc_proc['is_flash_crash'].sum()}")
print(f"Parabolic Surge Days (Jump >= +10% in 1 day):     {df_btc_proc['is_mega_surge'].sum()}")
print(f"Deep Bear Market Days (Drawdown <= -50%):        {(df_btc_proc['drawdown_pct'] <= -50.0).sum()}")
print(f"All-Time Worst Single-Day Plunge:                 {df_btc_proc['daily_return_pct'].min():.2f}% on {df_btc_proc.loc[df_btc_proc['daily_return_pct'].idxmin(), 'date'].strftime('%Y-%m-%d')}")
print(f"All-Time Best Single-Day Surge:                  +{df_btc_proc['daily_return_pct'].max():.2f}% on {df_btc_proc.loc[df_btc_proc['daily_return_pct'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
df_btc_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of Bitcoin Multi-Cycle Booms and Busts"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Log-Scale Price Trajectory with Cycle Highs and Liquidation Crashes
ax1 = axes[0]
ax1.plot(df_btc_proc['date'], df_btc_proc['BTC_close'], color='#f7931a', lw=1.8, label='Bitcoin Price (USD)')
ax1.plot(df_btc_proc['date'], df_btc_proc['sma_200'], color='#555555', lw=1.2, ls='--', label='200-Day SMA')
ax1.set_yscale('log')
ax1.set_title("Bitcoin Multi-Cycle Dynamics (2015–Present): Parabolic Bubbles & Catastrophic Crashes", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Price (USD, Log Scale)", fontsize=11, fontweight='bold')

btc_milestones = [
    ('2017-12-16', 'Dec 2017 Peak\\n(USD 19.8k)', 19800, 'top'),
    ('2018-12-15', '2018 Winter Bottom\\n(USD 3.1k / -84%)', 3150, 'bottom'),
    ('2020-03-12', 'Black Thursday Crash\\n(USD 4.9k / -50%)', 4970, 'bottom'),
    ('2021-04-14', 'Spring 2021 Peak\\n(USD 64.8k)', 64800, 'top'),
    ('2021-05-19', 'China Ban Flash Drop\\n(USD 37.0k)', 37000, 'bottom'),
    ('2021-11-10', 'Nov 2021 Peak\\n(USD 68.8k)', 68800, 'top'),
    ('2022-11-21', 'FTX Collapse Bottom\\n(USD 15.5k / -77%)', 15500, 'bottom'),
    ('2024-03-14', 'ETF Inflow High\\n(USD 73.7k)', 73700, 'top')
]

for date_str, label, y_val, va in btc_milestones:
    dt = pd.to_datetime(date_str)
    if dt in df_btc_proc['date'].values:
        actual = df_btc_proc.loc[df_btc_proc['date'] == dt, 'BTC_close'].values[0]
        ax1.scatter(dt, actual, color='red' if 'Bottom' in label or 'Crash' in label or 'Drop' in label else 'green', s=45, zorder=5)
        offset_mult = 0.45 if va=='bottom' else 1.8
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual * offset_mult),
                     arrowprops=dict(arrowstyle="->", color='black', lw=1.1),
                     fontsize=8.5, fontweight='bold', color='#111111', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

ax1.legend(loc='upper left', frameon=True)

# 2. Cumulative Drawdown Profile (% Underwater from Highs)
ax2 = axes[1]
ax2.plot(df_btc_proc['date'], df_btc_proc['drawdown_pct'], color='#d62728', lw=1.4, label='Drawdown from All-Time High (%)')
ax2.fill_between(df_btc_proc['date'], 0, df_btc_proc['drawdown_pct'], color='#d62728', alpha=0.3)
ax2.axhline(-50.0, color='darkorange', ls='--', lw=1.2, label='-50% Deep Correction')
ax2.axhline(-80.0, color='darkred', ls=':', lw=1.5, label='-80% Macro Winter Threshold')
ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight='bold')
ax2.set_title("Bitcoin Bear Market Drawdowns: Repeated 50% to 85% Value Destruction", fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', frameon=True)

# 3. Mayer Multiple (Price / 200 SMA Overheat Indicator)
ax3 = axes[2]
ax3.plot(df_btc_proc['date'], df_btc_proc['mayer_multiple'], color='#9467bd', lw=1.3, label='Mayer Multiple (Price / 200 SMA)')
ax3.axhline(1.0, color='black', lw=0.9, ls='-')
ax3.axhline(2.4, color='red', ls='--', lw=1.2, label='Overheat / Bubble Exuberance (>= 2.4)')
ax3.axhline(0.6, color='green', ls='--', lw=1.2, label='Capitulation Bottom / Undervaluation (<= 0.6)')
ax3.fill_between(df_btc_proc['date'], 2.4, df_btc_proc['mayer_multiple'], 
                 where=(df_btc_proc['mayer_multiple'] >= 2.4), color='red', alpha=0.35)
ax3.fill_between(df_btc_proc['date'], df_btc_proc['mayer_multiple'], 0.6, 
                 where=(df_btc_proc['mayer_multiple'] <= 0.6), color='green', alpha=0.35)
ax3.set_ylabel("Multiple (Ratio)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/btc_multicycle_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Flash Crash Ranking & Processed Export"""),
    nbf.v4.new_code_cell("""# Top 10 worst single-day drops in Bitcoin history
top_btc_drops = df_btc_proc.sort_values('daily_return_pct').head(10)[['date', 'BTC_close', 'daily_return_pct', 'drawdown_pct', 'volatility_30d_ann']]
top_btc_drops['date'] = top_btc_drops['date'].dt.strftime('%Y-%m-%d')
print("=== Top 10 Steepest Single-Day Bitcoin Flash Drops (2015–Present) ===")
print(top_btc_drops.to_string(index=False))

# Export clean processed dataset
out_path = 'data/btc_multicycle_processed.csv'
df_btc_proc.to_csv(out_path, index=False)
print(f"\\nProcessed Bitcoin dataset successfully saved to: {out_path}")""")
]

with open('08_bitcoin_multicycle_crashes_and_surges.ipynb', 'w') as f:
    nbf.write(nb8, f)
print("Saved 08_bitcoin_multicycle_crashes_and_surges.ipynb")


# ==============================================================================
# NOTEBOOK 9: US Initial Jobless Claims Recession Spikes (FRED: 1967–Present)
# ==============================================================================
nb9 = create_nb()
nb9.cells = [
    nbf.v4.new_markdown_cell(r"""# 🏛️ Macroeconomic Labor Shocks: US Initial Jobless Claims (FRED: 1967–Present)
## Multi-Decade Recession Spikes & The Historic COVID-19 Lockdown Blast

### 1. Incident Background & Shock Characteristics
**Initial Jobless Claims (`ICSA`)** measures the number of individuals filing first-time applications for unemployment insurance benefits with state agencies across the United States. It is widely considered the earliest, highest-frequency leading indicator of macro employment conditions.

Unlike equities (which drift over time), initial claims is **stationary around a peaceful structural baseline ($200\text{k} - 350\text{k}$)** during economic expansions. However, when economic crises occur, claims experience **sharp, asymmetric vertical spikes**:
* **1974–1975 Stagflation / OPEC Oil Embargo Recession:** Spiked to **695,000**.
* **1980 & 1982 Volcker Fed Tightening Recessions:** Reached **695,000** as interest rates topped 20%.
* **1990–1991 Gulf War & Savings-and-Loan Crisis:** Spiked to **535,000**.
* **2001 Dot-Com Crash Recession & 9/11:** Reached **517,000**.
* **2008–2009 Great Financial Crisis:** Spiked to **665,000** in March 2009.
* **March 28, 2020 COVID-19 Lockdown Shock:** In a matter of two weeks, claims exploded from a 50-year low of 256,000 to an astronomical, unprecedented peak of **6,867,000 in a single week**. This represented a **+2,580% vertical explosion**—a staggering $20\sigma$ outlier that eclipses all prior labor shocks in modern economic history.

### 2. Target Variables ($y$)
* **`ICSA`**: Initial Claims (Weekly, Seasonally Adjusted, in thousands or units).
* **`claims_change_1w`**: Weekly impulse velocity ($\Delta y = ICSA_t - ICSA_{t-1}$).
* **`surge_multiplier`**: Ratio of current claims relative to the 52-week trailing baseline.

### 3. Data Source
* Federal Reserve Bank of St. Louis (**FRED**) REST CSV Service.
* Series IDs: `ICSA` (Weekly Initial Claims) and `USREC` (NBER Official Recession Dates).
* Zero authentication required; runs out-of-the-box on HiPerGator.
"""),
    nbf.v4.new_code_cell("""import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Environment initialized successfully!")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion & Local Caching"""),
    nbf.v4.new_code_cell("""def fetch_jobless_claims(cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'us_jobless_claims_raw.csv')
    rec_cache_path = os.path.join(cache_dir, 'us_recessions_raw.csv')
    
    if os.path.exists(cache_path) and os.path.exists(rec_cache_path):
        print(f"Loading cached claims and recession series from: {cache_dir}")
        df_claims = pd.read_csv(cache_path)
        df_rec = pd.read_csv(rec_cache_path)
    else:
        print("Fetching ICSA (Jobless Claims) and USREC (Recessions) directly from FRED...")
        url_claims = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=ICSA'
        url_rec = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC'
        
        df_claims = pd.read_csv(url_claims)
        df_rec = pd.read_csv(url_rec)
        
        df_claims.to_csv(cache_path, index=False)
        df_rec.to_csv(rec_cache_path, index=False)
        print(f"Downloaded and cached {len(df_claims)} weekly claims records.")
        
    date_col1 = 'observation_date' if 'observation_date' in df_claims.columns else 'DATE'
    df_claims = df_claims.rename(columns={date_col1: 'DATE'})
    df_claims['DATE'] = pd.to_datetime(df_claims['DATE'])
    df_claims['ICSA'] = pd.to_numeric(df_claims['ICSA'], errors='coerce')
    
    date_col2 = 'observation_date' if 'observation_date' in df_rec.columns else 'DATE'
    df_rec = df_rec.rename(columns={date_col2: 'DATE'})
    df_rec['DATE'] = pd.to_datetime(df_rec['DATE'])
    df_rec['USREC'] = pd.to_numeric(df_rec['USREC'], errors='coerce')
    
    # Merge on weekly dates
    df_merged = pd.merge_asof(df_claims.sort_values('DATE'), df_rec.sort_values('DATE'), on='DATE')
    return df_merged

df_claims_raw = fetch_jobless_claims()
df_claims_raw.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Macro Shock Feature Engineering & Multi-Recession Spikes"""),
    nbf.v4.new_code_cell("""df_claims_proc = df_claims_raw.copy().sort_values('DATE').reset_index(drop=True)

# Convert to thousands of claims for readability (e.g. 6,867k)
df_claims_proc['claims_thousands'] = df_claims_proc['ICSA'] / 1000.0

# 1. Weekly impulse velocity (delta y / delta t)
df_claims_proc['velocity_1w_k'] = df_claims_proc['claims_thousands'].diff()
df_claims_proc['velocity_pct'] = df_claims_proc['claims_thousands'].pct_change() * 100.0

# 2. 52-week trailing baseline median
df_claims_proc['baseline_52w_median'] = df_claims_proc['claims_thousands'].rolling(52).median()

# 3. Surge Multiplier relative to baseline
df_claims_proc['surge_multiplier'] = df_claims_proc['claims_thousands'] / df_claims_proc['baseline_52w_median']

# 4. Rolling 3-year Z-Score
roll_mean_3y = df_claims_proc['claims_thousands'].rolling(156).mean()
roll_std_3y = df_claims_proc['claims_thousands'].rolling(156).std()
df_claims_proc['claims_zscore_3y'] = (df_claims_proc['claims_thousands'] - roll_mean_3y) / roll_std_3y

# 5. Automated Peak Detection of Recession Tops
peaks, _ = find_peaks(df_claims_proc['claims_thousands'].values, height=500, distance=52, prominence=150)
df_claims_proc['is_recession_peak'] = 0
df_claims_proc.loc[peaks, 'is_recession_peak'] = 1

print(f"Total Weekly Periods Analyzed:          {len(df_claims_proc):,}")
print(f"Historical Median Claims:               {df_claims_proc['claims_thousands'].median():.1f}k")
print(f"Pre-COVID 2019 Average:                 {df_claims_proc[(df_claims_proc['DATE'] >= '2019-01-01') & (df_claims_proc['DATE'] < '2020-01-01')]['claims_thousands'].mean():.1f}k")
print(f"Historic COVID Spike Crest:             {df_claims_proc['claims_thousands'].max():.1f}k on {df_claims_proc.loc[df_claims_proc['claims_thousands'].idxmax(), 'DATE'].strftime('%Y-%m-%d')}")
print(f"COVID Surge Multiplier vs Baseline:     {df_claims_proc['surge_multiplier'].max():.2f}x")
print(f"COVID Peak 3-Year Z-Score:              +{df_claims_proc['claims_zscore_3y'].max():.2f} sigma")
df_claims_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of 55 Years of Recession Spikes"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Macro Claims Timeline with Recession Shading
ax1 = axes[0]
ax1.plot(df_claims_proc['DATE'], df_claims_proc['claims_thousands'], color='#1f77b4', lw=1.5, label='Weekly Initial Claims (Thousands)')
ax1.plot(df_claims_proc['DATE'], df_claims_proc['baseline_52w_median'], color='black', lw=1.1, ls='--', label='52-Week Trailing Median')

# Highlight official NBER Recessions
recession_mask = df_claims_proc['USREC'] == 1
ax1.fill_between(df_claims_proc['DATE'], 0, df_claims_proc['claims_thousands'].max() * 1.05, 
                 where=recession_mask, color='gray', alpha=0.25, label='NBER Recession')

recession_peaks = [
    ('1975-01-11', '1975 Stagflation\\n(695k)', 695),
    ('1982-10-02', '1982 Volcker Tightening\\n(695k)', 695),
    ('1991-03-23', '1991 Gulf War (535k)', 535),
    ('2001-09-29', '2001 Dot-Com / 9/11 (517k)', 517),
    ('2009-03-28', '2009 GFC Peak (665k)', 665),
    ('2020-03-28', 'COVID-19 Vertical Blast\\n(6,867k / +25x)', 6867)
]

for date_str, label, y_val in recession_peaks:
    dt = pd.to_datetime(date_str)
    if dt in df_claims_proc['DATE'].values:
        actual = df_claims_proc.loc[df_claims_proc['DATE'] == dt, 'claims_thousands'].values[0]
        ax1.scatter(dt, actual, color='red', s=45, zorder=5)
        offset_y = 1200 if y_val < 1000 else -1500
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual + offset_y),
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.1),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax1.set_title("US Initial Jobless Claims (1967–Present): Multi-Decade Recession Spikes & The COVID Outlier", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Claims (Thousands / Week)", fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)

# 2. Weekly Impulse Velocity (delta y / 1 week)
ax2 = axes[1]
ax2.plot(df_claims_proc['DATE'], df_claims_proc['velocity_1w_k'], color='#9467bd', lw=1.2, label='Weekly Change in Claims (Thousands)')
ax2.axhline(0, color='black', lw=0.8)
ax2.fill_between(df_claims_proc['DATE'], 0, df_claims_proc['velocity_1w_k'], 
                 where=(df_claims_proc['velocity_1w_k'] > 50), color='red', alpha=0.5, label='Acute Surge (+50k/week)')
ax2.fill_between(df_claims_proc['DATE'], 0, df_claims_proc['velocity_1w_k'], 
                 where=(df_claims_proc['velocity_1w_k'] < -50), color='green', alpha=0.5, label='Rapid Re-employment (-50k/week)')
ax2.set_ylabel("Weekly Delta (k Claims)", fontsize=11, fontweight='bold')
ax2.set_title("Labor Shock Impulse Velocity: Explosive Layoff Surges vs. Recovery Trajectories", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Surge Severity Multiplier (Claims / Trailing 52-Week Median)
ax3 = axes[2]
ax3.plot(df_claims_proc['DATE'], df_claims_proc['surge_multiplier'], color='#d62728', lw=1.3, label='Surge Multiplier (x Baseline)')
ax3.axhline(1.0, color='black', lw=0.9, ls='--', label='Normal Expansion Baseline (1.0x)')
ax3.axhline(2.0, color='orange', ls='--', lw=1.1, label='Severe Recession (2.0x)')
ax3.axhline(5.0, color='darkred', ls=':', lw=1.2, label='Extreme Structural Shock (5.0x)')
ax3.set_ylabel("Multiple (x Baseline)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('data/us_jobless_claims_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Cross-Recession Comparative Anatomy Table & Processed Export"""),
    nbf.v4.new_code_cell("""# Compare all detected recession peaks
peaks_df = df_claims_proc[df_claims_proc['is_recession_peak'] == 1].sort_values('claims_thousands', ascending=False)
peak_table = peaks_df[['DATE', 'claims_thousands', 'velocity_1w_k', 'surge_multiplier', 'claims_zscore_3y']].copy()
peak_table['DATE'] = peak_table['DATE'].dt.strftime('%Y-%m-%d')
print("=== Historical US Initial Jobless Claims Spike Crests ===")
print(peak_table.to_string(index=False))

# Export clean processed dataset
out_path = 'data/us_jobless_claims_processed.csv'
df_claims_proc.to_csv(out_path, index=False)
print(f"\\nProcessed jobless claims dataset successfully saved to: {out_path}")""")
]

with open('09_us_jobless_claims_recession_spikes.ipynb', 'w') as f:
    nbf.write(nb9, f)
print("Saved 09_us_jobless_claims_recession_spikes.ipynb")


# ==============================================================================
# NOTEBOOK 10: GameStop (GME) Multi-Spike Short Squeeze Manias (2020–Present)
# ==============================================================================
nb10 = create_nb()
nb10.cells = [
    nbf.v4.new_markdown_cell(r"""# 🚀 Retail Mania & Microstructure Shocks: GameStop Short Squeezes (2020–Present)
## Parabolic Squeeze Spikes, Intraday Halts, Liquidity Vacuum Crashes & Social Media Resurgence

### 1. Incident Background & Multi-Spike Structure
In January 2021, brick-and-mortar retailer **GameStop Corp. (`GME`)** became the epicenter of the most dramatic short squeeze in modern financial market history. Over 140% of the public share float was sold short by institutional hedge funds. A synchronized influx of retail options buying on Reddit's r/WallStreetBets forced market makers into an aggressive gamma squeeze and prime brokers into margin liquidation.

Crucially, **the squeeze was not a one-off event; it recurred repeatedly over four years**:
* **Episode 1: The January 2021 "Mother of All Short Squeezes" (MOASS):**
  * From January 11 to January 28, 2021, GME surged from **$4.98 $\rightarrow$ $120.75$ split-adjusted** (over **$483 pre-split**, a **+2,324% vertical explosion** in 14 trading days).
  * On January 28, retail brokerages (e.g. Robinhood) restricted buying due to clearinghouse collateral demands, triggering an instantaneous **-75% collapse** down to $13.25 in 5 days.
* **Episode 2: February 24, 2021 Secondary Resurgence:**
  * In the final 30 minutes of trading, GME exploded from **$11.19 $\rightarrow$ $45.00$** (+302%) on CFO resignation news and gamma ramp resurgence.
* **Episode 3: March 10, 2021 Flash Crash:**
  * Surged to an intraday high of **$87.00**, followed by an algorithmic flash dump to **$43.00** (-50.6%) within 25 minutes before trading halted.
* **Episode 4: June 2021 Pre-Offering Spike:**
  * Climbed back to **$86.00** before the company capitalized on volume with an ATM secondary share offering.
* **Episode 5: May 13–14, 2024 "Roaring Kitty" Resurgence Surge:**
  * After 3 years of silence, retail trader Keith Gill posted on X. GME surged from **$17.46 $\rightarrow$ $64.83$ intraday** (**+271% in 48 hours**) with multiple NYSE volatility halts.
* **Episode 6: June 6–7, 2024 Livestream Frenzy & Dilution Drop:**
  * Surged to **$46.55** on Gill's announced YouTube livestream, followed by a sudden **-40% drop** when GameStop unexpectedly filed to sell 75 million shares.

### 2. Target Variables ($y$)
* **`GME_close`**: Daily split-adjusted closing price.
* **`intraday_spread_pct`**: Intraday high-to-low percentage volatility ($\frac{\text{High} - \text{Low}}{\text{Low}} \times 100$).
* **`volume_surge_ratio`**: Daily trading volume divided by 30-day rolling median volume ($Vol_t / \text{Median}_{30}(Vol)$).

### 3. Data Source
* Direct Yahoo Finance REST API (`GME`, 2020–Present).
* Fully reproducible with offline fallback caching to `./data/`.
"""),
    nbf.v4.new_code_cell("""import os
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Environment initialized successfully!")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion & Local Caching"""),
    nbf.v4.new_code_cell("""def fetch_gamestop_telemetry(start_year=2020, cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'gamestop_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached GameStop telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    print(f"Fetching GameStop price and volume telemetry from {start_year} to present...")
    t1 = int(datetime.datetime(start_year, 1, 1).timestamp())
    t2 = int(datetime.datetime.now().timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/GME?period1={t1}&period2={t2}&interval=1d'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    res = response.json()['chart']['result'][0]
    
    dates = pd.to_datetime(res['timestamp'], unit='s').tz_localize(None).floor('D')
    quote = res['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'date': dates,
        'GME_open': quote['open'],
        'GME_high': quote['high'],
        'GME_low': quote['low'],
        'GME_close': quote['close'],
        'GME_vol': quote['volume']
    }).dropna(subset=['GME_close']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df)} trading sessions of GameStop telemetry.")
    return df

df_gme = fetch_gamestop_telemetry()
df_gme.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Short Squeeze & Volatility Feature Engineering"""),
    nbf.v4.new_code_cell("""df_gme_proc = df_gme.copy().sort_values('date').reset_index(drop=True)

# 1. Daily return and overnight gap
df_gme_proc['return_pct'] = df_gme_proc['GME_close'].pct_change() * 100.0
df_gme_proc['gap_pct'] = ((df_gme_proc['GME_open'] - df_gme_proc['GME_close'].shift(1)) / df_gme_proc['GME_close'].shift(1)) * 100.0

# 2. Intraday Spread / Volatility: (High - Low) / Low * 100
df_gme_proc['intraday_spread_pct'] = ((df_gme_proc['GME_high'] - df_gme_proc['GME_low']) / df_gme_proc['GME_low']) * 100.0

# 3. Volume Surge Ratio (Current volume vs. trailing 30-day median)
df_gme_proc['vol_median_30d'] = df_gme_proc['GME_vol'].rolling(30).median()
df_gme_proc['volume_surge_ratio'] = df_gme_proc['GME_vol'] / df_gme_proc['vol_median_30d']

# 4. Rolling 10-day Realized Volatility
df_gme_proc['volatility_10d_ann'] = df_gme_proc['return_pct'].rolling(10).std() * np.sqrt(252)

# 5. Peak Drawdown from Cumulative High
df_gme_proc['all_time_high'] = df_gme_proc['GME_close'].cummax()
df_gme_proc['drawdown_pct'] = ((df_gme_proc['GME_close'] - df_gme_proc['all_time_high']) / df_gme_proc['all_time_high']) * 100.0

# 6. Flag Super-Spike Sessions (Return >= +25% or Intraday Spread >= 50%)
df_gme_proc['is_super_spike'] = ((df_gme_proc['return_pct'] >= 25.0) | (df_gme_proc['intraday_spread_pct'] >= 50.0)).astype(int)

print(f"Total Trading Sessions Analyzed:                {len(df_gme_proc):,}")
print(f"Super-Spike Sessions Identified:               {df_gme_proc['is_super_spike'].sum()}")
print(f"Max Single-Day Gain:                           +{df_gme_proc['return_pct'].max():.1f}% on {df_gme_proc.loc[df_gme_proc['return_pct'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
print(f"Max Single-Day Loss:                            {df_gme_proc['return_pct'].min():.1f}% on {df_gme_proc.loc[df_gme_proc['return_pct'].idxmin(), 'date'].strftime('%Y-%m-%d')}")
print(f"Max Intraday Spread (High vs Low):             +{df_gme_proc['intraday_spread_pct'].max():.1f}%")
print(f"Peak Trading Volume:                           {df_gme_proc['GME_vol'].max():,}")
df_gme_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of Recurring Short Squeeze Manias"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Price Timeline with Multi-Episode Annotations
ax1 = axes[0]
ax1.plot(df_gme_proc['date'], df_gme_proc['GME_close'], color='#b22222', lw=1.6, label='GME Daily Close (Split-Adjusted USD)')
ax1.set_title("GameStop Corp. (GME): Multi-Spike Short Squeezes & Liquidity Collapses (2020–Present)", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Price (USD)", fontsize=11, fontweight='bold')

gme_episodes = [
    ('2021-01-27', 'Jan 2021 MOASS Peak\\n(USD 86.8 close / 120.7 high)', 86.8),
    ('2021-02-04', 'Robinhood Halt Drop\\n(USD 13.3 / -85%)', 13.3),
    ('2021-02-25', 'Feb 2021 Secondary Spike\\n(USD 27.1 / +18% to USD 45)', 27.1),
    ('2021-03-10', 'March 2021 Flash Dump\\n(USD 66.2 / -50% intraday)', 66.2),
    ('2021-06-08', 'June 2021 Offering Run\\n(USD 75.0)', 75.0),
    ('2024-05-14', 'May 2024 Roaring Kitty\\n(USD 48.7 close / 64.8 high)', 48.7),
    ('2024-06-06', 'June 2024 Livestream\\n(USD 46.5)', 46.5)
]

for date_str, label, y_val in gme_episodes:
    dt = pd.to_datetime(date_str)
    if dt in df_gme_proc['date'].values:
        actual = df_gme_proc.loc[df_gme_proc['date'] == dt, 'GME_close'].values[0]
        ax1.scatter(dt, actual, color='black', s=45, zorder=5)
        offset_y = 15 if actual < 50 else -25
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual + offset_y),
                     arrowprops=dict(arrowstyle="->", color='#8b0000', lw=1.2),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='#8b0000'))

ax1.legend(loc='upper left', frameon=True)

# 2. Volume Surge Ratio (Current Vol / 30-Day Median)
ax2 = axes[1]
ax2.plot(df_gme_proc['date'], df_gme_proc['volume_surge_ratio'], color='#2ca02c', lw=1.3, label='Volume Surge Factor (x 30d Median)')
ax2.axhline(1.0, color='black', lw=0.8, ls='--')
ax2.axhline(5.0, color='orange', ls='--', lw=1.1, label='Extreme Volume Inflow (5x)')
ax2.axhline(15.0, color='darkred', ls=':', lw=1.2, label='Short Squeeze Liquidity Wave (15x)')
ax2.fill_between(df_gme_proc['date'], 5.0, df_gme_proc['volume_surge_ratio'], 
                 where=(df_gme_proc['volume_surge_ratio'] >= 5.0), color='#2ca02c', alpha=0.3)
ax2.set_ylabel("Volume Multiplier", fontsize=11, fontweight='bold')
ax2.set_title("Trading Volume Explosions Accompanying Squeeze Spikes", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Intraday High-Low Volatility Spread (%)
ax3 = axes[2]
ax3.plot(df_gme_proc['date'], df_gme_proc['intraday_spread_pct'], color='#9467bd', lw=1.2, label='Intraday Spread %: (High - Low)/Low')
ax3.axhline(20.0, color='gold', ls='--', lw=1.0, label='High Volatility (20%)')
ax3.axhline(50.0, color='red', ls='--', lw=1.2, label='Circuit-Breaker Turbulence (50%)')
ax3.fill_between(df_gme_proc['date'], 50.0, df_gme_proc['intraday_spread_pct'], 
                 where=(df_gme_proc['intraday_spread_pct'] >= 50.0), color='purple', alpha=0.3)
ax3.set_ylabel("Intraday Spread (%)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/gamestop_spikes_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Squeeze Episode Breakdown Table & Processed Export"""),
    nbf.v4.new_code_cell("""# Summary table of the top 10 single-day percentage gains in GME
top_gme_gains = df_gme_proc.sort_values('return_pct', ascending=False).head(10)[['date', 'GME_close', 'return_pct', 'intraday_spread_pct', 'volume_surge_ratio']]
top_gme_gains['date'] = top_gme_gains['date'].dt.strftime('%Y-%m-%d')
print("=== Top 10 Single-Day GameStop Squeeze Jumps ===")
print(top_gme_gains.to_string(index=False))

# Export clean processed dataset
out_path = 'data/gamestop_spikes_processed.csv'
df_gme_proc.to_csv(out_path, index=False)
print(f"\\nProcessed GameStop dataset successfully saved to: {out_path}")""")
]

with open('10_gamestop_short_squeeze_manias.ipynb', 'w') as f:
    nbf.write(nb10, f)
print("Saved 10_gamestop_short_squeeze_manias.ipynb")

print("\\nNotebooks 6-10 recreated with syntax and compatibility fixes!")
