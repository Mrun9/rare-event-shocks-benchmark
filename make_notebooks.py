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
# NOTEBOOK 1: USGS Hurricane Helene Flash Flood Spike
# ==============================================================================
nb1 = create_nb()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# 🌊 Extreme Hydrological Shock: Hurricane Helene Flash Flood (Sept 2024)
## Real-Time Sensor Telemetry & Steep Spike Analysis

### 1. Incident Background
On **September 26–27, 2024**, Hurricane Helene dumped catastrophic precipitation (up to 30 inches in 48 hours) across Western North Carolina and the Blue Ridge Mountains. This caused historic flash flooding along the French Broad River basin.

### 2. Target Variables ($y$)
* **`discharge_cfs`** (USGS Parameter `00060`, Streamflow in cubic feet per second): Surged from a baseline of ~**727 cfs** to an all-time record crest of **112,000+ cfs** (**+15,400% surge**).
* **`gage_height_ft`** (USGS Parameter `00065`, Stage height in feet): Surged from **1.32 ft** to **24.82 ft** in under 36 hours, eclipsing the previous 1916 century flood record.

### 3. Data Source
* **USGS National Water Information System (NWIS)** Instantaneous Values (IV) REST API.
* Site ID: `03451500` (French Broad River at Asheville, NC).
* Sampling Frequency: 15-minute intervals.
* Zero authentication required; runs out-of-the-box on HiPerGator.
"""),
    nbf.v4.new_code_cell("""import os
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Libraries loaded and environment initialized successfully!")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion (USGS NWIS REST API with Local Caching)"""),
    nbf.v4.new_code_cell("""def fetch_usgs_telemetry(site_id='03451500', start_date='2024-09-24', end_date='2024-10-02', cache_dir='data'):
    cache_path = os.path.join(cache_dir, f'usgs_{site_id}_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached raw telemetry from: {cache_path}")
        return pd.read_csv(cache_path)
    
    print(f"Fetching live telemetry from USGS Water Services for site {site_id}...")
    url = 'https://waterservices.usgs.gov/nwis/iv/'
    params = {
        'format': 'json',
        'sites': site_id,
        'startDT': f'{start_date}T00:00',
        'endDT': f'{end_date}T23:59',
        'parameterCd': '00060,00065'  # 00060=Discharge, 00065=Gage Height
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    records = []
    for ts in data['value']['timeSeries']:
        p_code = ts['variable']['variableCode'][0]['value']
        p_name = ts['variable']['variableName']
        unit = ts['variable']['unit']['unitCode']
        for pt in ts['values'][0]['value']:
            records.append({
                'dateTime': pt['dateTime'],
                'parameter_code': str(p_code).zfill(5),
                'parameter_name': p_name,
                'unit': unit,
                'value': pt['value'],
                'qualifiers': pt['qualifiers'][0] if pt.get('qualifiers') else ''
            })
            
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df_raw)} sensor readings.")
    return df_raw

df_raw = fetch_usgs_telemetry()
df_raw.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Data Preprocessing & Feature Engineering"""),
    nbf.v4.new_code_cell("""# 1. Clean raw telemetry
df_clean = df_raw.copy()
df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
# Standardize timestamps to timezone-naive UTC
df_clean['dateTime'] = pd.to_datetime(df_clean['dateTime'], utc=True).dt.tz_localize(None)
df_clean['parameter_code'] = df_clean['parameter_code'].astype(str).str.zfill(5)

# 2. Pivot parameters into clean column names
df_pivot = df_clean.pivot_table(
    index='dateTime', 
    columns='parameter_code', 
    values='value'
).rename(columns={'00060': 'discharge_cfs', '00065': 'gage_height_ft'}).sort_index()

# 3. Regularize to uniform 15-min intervals
df_ts = df_pivot.resample('15min').mean().interpolate(method='time', limit=4)

# 4. Feature Engineering
# Rate of change per hour (* 4 for 15-min steps)
df_ts['discharge_rate_cfs_hr'] = df_ts['discharge_cfs'].diff() * 4
df_ts['height_rate_ft_hr'] = df_ts['gage_height_ft'].diff() * 4

# Pre-storm baseline (Sept 24)
baseline_discharge = df_ts.loc['2024-09-24', 'discharge_cfs'].mean()
baseline_height = df_ts.loc['2024-09-24', 'gage_height_ft'].mean()

# Relative Percentage Surge
df_ts['discharge_surge_pct'] = ((df_ts['discharge_cfs'] - baseline_discharge) / baseline_discharge) * 100
df_ts['height_surge_pct'] = ((df_ts['gage_height_ft'] - baseline_height) / baseline_height) * 100

# Rolling 6-hour volatility
df_ts['discharge_vol_6h'] = df_ts['discharge_cfs'].rolling('6h').std()

# Event-centered time relative to peak
peak_time = df_ts['discharge_cfs'].idxmax()
df_ts['tau_hours'] = (df_ts.index - peak_time).total_seconds() / 3600.0

print(f"Pre-storm baseline discharge: {baseline_discharge:.1f} cfs")
print(f"Pre-storm baseline height:    {baseline_height:.2f} ft")
print(f"Peak discharge reached:       {df_ts['discharge_cfs'].max():.1f} cfs at {peak_time}")
print(f"Peak gage height reached:     {df_ts['gage_height_ft'].max():.2f} ft")
print(f"Max Streamflow Surge:         +{df_ts['discharge_surge_pct'].max():.1f}%")
df_ts.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of the Shock"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# 1. Macro Timeline: Gage Height and Streamflow
ax1 = axes[0]
color1 = '#1f77b4'
color2 = '#d62728'

ax1.set_title("USGS Site 03451500 (French Broad River at Asheville, NC) - Hurricane Helene Hydrological Shock", 
              fontsize=14, fontweight='bold', pad=12)
ax1.plot(df_ts.index, df_ts['gage_height_ft'], color=color1, lw=2.2, label='Gage Height (ft)')
ax1.axhline(9.5, color='orange', ls='--', lw=1.2, label='Action Stage (9.5 ft)')
ax1.axhline(16.0, color='red', ls='--', lw=1.5, label='Major Flood Stage (16.0 ft)')
ax1.set_ylabel("Gage Height (ft)", color=color1, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)

# Secondary axis for discharge
ax1_twin = ax1.twinx()
ax1_twin.plot(df_ts.index, df_ts['discharge_cfs'] / 1000.0, color=color2, lw=2.0, ls=':', label='Discharge (k-cfs)')
ax1_twin.set_ylabel("Discharge (1,000 cfs)", color=color2, fontsize=11, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color2)
ax1.axvline(peak_time, color='purple', ls='-', lw=1.5, alpha=0.8)
ax1.text(peak_time, 22, f' Crest: 24.8 ft\\n {peak_time.strftime("%b %d %H:%M")}', 
         fontsize=10, fontweight='bold', color='purple', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
ax1.legend(loc='upper left', frameon=True)
ax1_twin.legend(loc='upper right', frameon=True)

# 2. Percentage Surge Relative to Normal
ax2 = axes[1]
ax2.plot(df_ts.index, df_ts['discharge_surge_pct'], color='#2ca02c', lw=2.0)
ax2.set_ylabel("Surge (% above baseline)", fontsize=11, fontweight='bold')
ax2.fill_between(df_ts.index, 0, df_ts['discharge_surge_pct'], color='#2ca02c', alpha=0.2)
ax2.axvline(peak_time, color='purple', ls='--', lw=1.2)
ax2.text(df_ts.index[10], 12000, f"Peak Surge: +{df_ts['discharge_surge_pct'].max():,.0f}%", 
         fontsize=11, fontweight='bold', color='#1b611b')

# 3. Rate of Rise / First Derivative (Impulse Shock)
ax3 = axes[2]
ax3.plot(df_ts.index, df_ts['discharge_rate_cfs_hr'] / 1000.0, color='#9467bd', lw=1.8)
ax3.axhline(0, color='black', lw=0.8, ls='-')
ax3.fill_between(df_ts.index, 0, df_ts['discharge_rate_cfs_hr'] / 1000.0, 
                 where=(df_ts['discharge_rate_cfs_hr'] > 0), color='#9467bd', alpha=0.3, label='Surge Acceleration')
ax3.fill_between(df_ts.index, 0, df_ts['discharge_rate_cfs_hr'] / 1000.0, 
                 where=(df_ts['discharge_rate_cfs_hr'] < 0), color='#8c564b', alpha=0.3, label='Recession')
ax3.set_ylabel("Rate of Change (k-cfs / hr)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (UTC)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/usgs_helene_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Zoomed Event Horizon: Centered Window ($\tau = t - T_{peak}$)"""),
    nbf.v4.new_code_cell("""horizon_df = df_ts[(df_ts['tau_hours'] >= -36) & (df_ts['tau_hours'] <= 48)].copy()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(horizon_df['tau_hours'], horizon_df['gage_height_ft'], lw=2.5, color='#d62728', label='Gage Height (ft)')
ax.axvline(0, color='black', ls='--', lw=1.5, label='Shock Peak (tau = 0)')
ax.axvspan(-24, 0, color='red', alpha=0.1, label='Rapid Escalation Window (24h)')
ax.set_title("Event-Horizon View: Gage Height Response Centered on Shock Peak", fontsize=13, fontweight='bold')
ax.set_xlabel("Hours Relative to Peak (tau)", fontsize=11, fontweight='bold')
ax.set_ylabel("Gage Height (ft)", fontsize=11, fontweight='bold')
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()

# Save processed dataset
out_path = 'data/usgs_helene_processed.csv'
df_ts.to_csv(out_path)
print(f"Processed dataset saved successfully to: {out_path}")""")
]

with open('01_usgs_hurricane_helene_flash_spike.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("Updated 01_usgs_hurricane_helene_flash_spike.ipynb")

# ==============================================================================
# NOTEBOOK 2: Industrial Pulp-and-Paper Mill Sheet Break
# ==============================================================================
nb2 = create_nb()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# 🏭 Industrial Rare-Event Breakdown: Pulp-and-Paper Manufacturing Process
## Multivariate Time Series Anomaly Detection & Abrupt Sensor Collapse

### 1. Incident Background
In paper manufacturing mills, large continuous rolls of paper pass through chemical baths, pressing rollers, and high-speed driers. Occasionally, a **sheet break** occurs: the moving paper web snaps. 
* A sheet break is a classic **rare event** (occurring in less than 1% of operational hours).
* Each break causes hours of downtime, machine shutdown, and expensive raw material waste.

### 2. Target Variables ($y$)
* **`y`** (Binary Label): `y = 1` during the rare break event, `y = 0` during normal operation (extreme class imbalance: ~0.67% positives).
* **`x21` / `x4`** (Continuous Process Sensors): Physical tension and flow sensors that experience an **instant cliff drop** down to zero during the breakdown.

### 3. Data Source
* Chitta Ranjan et al. (*arXiv:1809.10717: Dataset: Rare Event Classification in Multivariate Time Series*).
* Contains 18,398 time intervals sampled every 2 minutes across 61 synchronized sensors (`x1` to `x61`).
"""),
    nbf.v4.new_code_cell("""import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('data', exist_ok=True)
print("Environment initialized.")"""),
    nbf.v4.new_markdown_cell("""### 4. Data Ingestion & Local Caching"""),
    nbf.v4.new_code_cell("""def load_paper_mill_data(cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'pulp_paper_mill_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading paper mill data from cache: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        print("Downloading raw multivariate dataset from open GitHub repository...")
        url = "https://raw.githubusercontent.com/robertoamansur/rare_event_pred_maintanance/master/processminer-rare-event-mts%20-%20data.csv"
        df = pd.read_csv(url, sep=';')
        df.to_csv(cache_path, index=False)
        print(f"Downloaded and cached {df.shape} records.")
        
    return df

df_raw = load_paper_mill_data()
print(f"Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"Rare Break Events (y=1): {(df_raw['y'] == 1).sum()} ({df_raw['y'].mean()*100:.2f}%)")
df_raw.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Preprocessing & Sensor Behavior Analysis"""),
    nbf.v4.new_code_cell("""df_proc = df_raw.copy()
df_proc['time'] = pd.to_datetime(df_proc['time'], format='%m/%d/%y %H:%M')
df_proc = df_proc.sort_values('time').reset_index(drop=True)

# Drop static / constant sensors with near-zero variance
sensor_cols = [c for c in df_proc.columns if c.startswith('x')]
stds = df_proc[sensor_cols].std()
static_sensors = stds[stds < 1e-4].index.tolist()
print(f"Static / dead sensors removed: {static_sensors}")
df_proc = df_proc.drop(columns=static_sensors)
sensor_cols = [c for c in sensor_cols if c not in static_sensors]

# Calculate sensor drop delta across y=0 vs y=1
mean_normal = df_proc[df_proc['y'] == 0][sensor_cols].mean()
mean_break = df_proc[df_proc['y'] == 1][sensor_cols].mean()
drop_diff = mean_break - mean_normal
top_drop_sensor = drop_diff.idxmin()
top_spike_sensor = drop_diff.idxmax()

print(f"Sensor with sharpest drop during sheet break: {top_drop_sensor} (Mean shift: {drop_diff[top_drop_sensor]:.3f})")
print(f"Sensor with sharpest spike during sheet break: {top_spike_sensor} (Mean shift: {drop_diff[top_spike_sensor]:.3f})")

# Feature engineering: Rolling 10-step stats for the primary drop sensor
df_proc[f'{top_drop_sensor}_rolling_mean'] = df_proc[top_drop_sensor].rolling(10).mean()
df_proc[f'{top_drop_sensor}_rolling_std'] = df_proc[top_drop_sensor].rolling(10).std()
df_proc[f'{top_drop_sensor}_diff'] = df_proc[top_drop_sensor].diff()

df_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of the Sheet Break Drops"""),
    nbf.v4.new_code_cell("""# Pick a specific incident window with consecutive breakdown events
break_indices = df_proc.index[df_proc['y'] == 1].tolist()
sample_event_idx = break_indices[15] # Focus on an illustrative break event
window = 120 # +/- 120 steps (4 hours)
start_idx = max(0, sample_event_idx - window)
end_idx = min(len(df_proc), sample_event_idx + window)

sub_df = df_proc.iloc[start_idx:end_idx].copy()
event_time = df_proc.loc[sample_event_idx, 'time']

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# 1. Primary Drop Sensor Response
axes[0].set_title(f"Pulp-and-Paper Mill Rare Break Analysis — Sudden Drop in {top_drop_sensor}", 
                  fontsize=14, fontweight='bold', pad=10)
axes[0].plot(sub_df['time'], sub_df[top_drop_sensor], color='#1f77b4', lw=2.0, label=f'{top_drop_sensor} (Process Signal)')
axes[0].plot(sub_df['time'], sub_df[f'{top_drop_sensor}_rolling_mean'], color='black', lw=1.2, ls='--', label='10-step Rolling Mean')
axes[0].axvline(event_time, color='red', lw=2, ls='-', label='Sheet Break Timestamp (y=1)')
axes[0].set_ylabel(f"{top_drop_sensor} Value", fontsize=11, fontweight='bold')
axes[0].legend(loc='upper right', frameon=True)

# 2. Sudden Spike Sensor (Counter-Pressure or Valve Surge)
axes[1].plot(sub_df['time'], sub_df[top_spike_sensor], color='#ff7f0e', lw=2.0, label=f'{top_spike_sensor} (Spike Signal)')
axes[1].axvline(event_time, color='red', lw=2, ls='-')
axes[1].set_ylabel(f"{top_spike_sensor} Value", fontsize=11, fontweight='bold')
axes[1].legend(loc='upper right', frameon=True)

# 3. Rare Break Indicator (Target y) and Velocity
axes[2].fill_between(sub_df['time'], 0, sub_df['y'], color='red', alpha=0.4, step='mid', label='Breakdown Event (y=1)')
axes[2].plot(sub_df['time'], sub_df[f'{top_drop_sensor}_diff'], color='#2ca02c', lw=1.4, label='First Difference (delta y / delta t)')
axes[2].set_ylabel("State / Velocity", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Process Timestamp", fontsize=11, fontweight='bold')
axes[2].legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig('data/paper_mill_break_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Statistical Regime Shift: Normal Operation vs. Breakdown"""),
    nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Distribution of Top Drop Sensor
sns.boxplot(x='y', y=top_drop_sensor, data=df_proc, ax=ax1, palette=['#1f77b4', '#d62728'], hue='y', legend=False)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Normal (y=0)', 'Breakdown (y=1)'], fontsize=11, fontweight='bold')
ax1.set_title(f"Distribution Shift in {top_drop_sensor}", fontsize=12, fontweight='bold')

# Distribution of Top Spike Sensor
sns.boxplot(x='y', y=top_spike_sensor, data=df_proc, ax=ax2, palette=['#1f77b4', '#d62728'], hue='y', legend=False)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Normal (y=0)', 'Breakdown (y=1)'], fontsize=11, fontweight='bold')
ax2.set_title(f"Distribution Shift in {top_spike_sensor}", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Export clean dataset
df_proc.to_csv('data/pulp_paper_mill_processed.csv', index=False)
print("Processed paper mill dataset saved to data/pulp_paper_mill_processed.csv")""")
]

with open('02_industrial_paper_mill_break_drop.ipynb', 'w') as f:
    nbf.write(nb2, f)
print("Updated 02_industrial_paper_mill_break_drop.ipynb")

print("Notebooks 1 and 2 updated!")
