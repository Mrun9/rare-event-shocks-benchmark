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
# NOTEBOOK 11: USGS Decadal River Flood Spikes (1990–Present)
# ==============================================================================
nb11 = create_nb()
nb11.cells = [
    nbf.v4.new_markdown_cell(r"""# 🌊 Decadal Environmental Hydrology: USGS 35-Year River Flood Spikes (1990–Present)
## Long-Horizon Streamflow Dynamics, Extreme Flood Crests & Drought Baselines

### 1. Incident Background & Multi-Spike Hydrological Dynamics
In hydrological science, river flow is characterized by long quiescent periods punctuated by **severe meteorological shock events**: tropical storms, atmospheric rivers, and hurricanes dumping catastrophic precipitation over mountain drainage basins.

Unlike the short-horizon 2-week view of Hurricane Helene in Notebook 01, this notebook spans **35 continuous years (1990–Present, over 12,700 daily observations)** at **USGS Site `03451500` (French Broad River at Asheville, NC)**. 

Across this multi-decade record, river streamflow ($discharge\_cfs$) exhibits recurrent vertical surges where discharge explodes from a resting baseline of **~700 cfs** to extreme flood crests exceeding **10,000 to 50,000+ cfs** (**+1,500% to +7,000% surges**):
* **October 1995 (Hurricane Opal):** Spiked to **24,100 cfs**.
* **September 2004 (Hurricane Frances & Hurricane Ivan Double-Shock):** Historic consecutive tropical storm deluge driving peak daily streamflow to **40,500 cfs**.
* **January 2013 Flash Flooding:** Winter rain-on-snow event driving streamflow to **18,600 cfs**.
* **August 2021 (Tropical Storm Fred):** Catastrophic Blue Ridge flooding cresting at **25,100 cfs**.
* **September 2024 (Hurricane Helene):** Unprecedented multi-century record crest of **50,000+ cfs daily mean (112,000+ cfs peak instantaneous)**.

### 2. Target Variables ($y$)
* **`discharge_cfs`**: Mean daily streamflow in cubic feet per second.
* **`velocity_cfs_day`**: Daily discharge acceleration ($\Delta y = y_t - y_{t-1}$).
* **`surge_multiplier`**: Daily streamflow relative to the 365-day rolling median baseline ($y_t / \text{Median}_{365}(y)$).

### 3. Data Source
* **USGS National Water Information System (NWIS)** Daily Values REST API.
* Site: `03451500` (French Broad River at Asheville, NC).
* 100% public, free, zero API keys required.
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
    nbf.v4.new_code_cell("""def fetch_usgs_decadal_streamflow(site_id='03451500', start_date='1990-01-01', cache_dir='data'):
    cache_path = os.path.join(cache_dir, f'usgs_{site_id}_decadal_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached decadal USGS streamflow from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching 35-year daily streamflow for USGS Site {site_id} ({start_date} to {end_date})...")
    url = 'https://waterservices.usgs.gov/nwis/dv/'
    params = {
        'format': 'json',
        'sites': site_id,
        'startDT': start_date,
        'endDT': end_date,
        'statCd': '00003',      # 00003 = Daily Mean
        'parameterCd': '00060'  # 00060 = Streamflow (cfs)
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    ts_values = data['value']['timeSeries'][0]['values'][0]['value']
    records = []
    for item in ts_values:
        records.append({
            'date': item['dateTime'][:10],
            'discharge_cfs': item['value'],
            'qualifiers': item['qualifiers'][0] if item.get('qualifiers') else ''
        })
        
    df = pd.DataFrame(records)
    df['discharge_cfs'] = pd.to_numeric(df['discharge_cfs'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['discharge_cfs']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df):,} daily streamflow readings.")
    return df

df_usgs = fetch_usgs_decadal_streamflow()
df_usgs.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Multi-Decade Hydrological Feature Engineering"""),
    nbf.v4.new_code_cell("""df_hydro = df_usgs.copy().sort_values('date').reset_index(drop=True)

# 1. Rate of change (velocity) in cfs / day
df_hydro['velocity_cfs_day'] = df_hydro['discharge_cfs'].diff()

# 2. 365-day rolling median baseline (representing normal seasonal flow)
df_hydro['baseline_365d_median'] = df_hydro['discharge_cfs'].rolling(365, min_periods=90).median()

# 3. Surge severity multiplier (Discharge / Baseline)
df_hydro['surge_multiplier'] = df_hydro['discharge_cfs'] / df_hydro['baseline_365d_median']

# 4. Rolling 3-year Z-Score
roll_mean_3y = df_hydro['discharge_cfs'].rolling(1095, min_periods=180).mean()
roll_std_3y = df_hydro['discharge_cfs'].rolling(1095, min_periods=180).std()
df_hydro['discharge_zscore_3y'] = (df_hydro['discharge_cfs'] - roll_mean_3y) / roll_std_3y

# 5. Automated Peak Detection of Prominent Flood Crests
# Threshold: >= 10,000 cfs, separated by at least 60 days
peaks, _ = find_peaks(df_hydro['discharge_cfs'].values, height=10000, distance=60, prominence=5000)
df_hydro['is_flood_crest'] = 0
df_hydro.loc[peaks, 'is_flood_crest'] = 1

print(f"Total Observations:                       {len(df_hydro):,} days")
print(f"Long-Term Median Streamflow:              {df_hydro['discharge_cfs'].median():.1f} cfs")
print(f"Major Flood Crests (>= 10,000 cfs):        {len(peaks)} distinct episodes")
print(f"Extreme Century Flood Crests (>= 25,000):  {(df_hydro['discharge_cfs'] >= 25000).sum()} days")
print(f"All-Time Daily Maximum Streamflow:        {df_hydro['discharge_cfs'].max():,.0f} cfs on {df_hydro.loc[df_hydro['discharge_cfs'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
print(f"Peak Surge Multiplier vs Baseline:        {df_hydro['surge_multiplier'].max():.1f}x normal flow")
df_hydro.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of 35 Years of River Floods"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. 35-Year Streamflow Timeline with Major Hurricane Flood Crests
ax1 = axes[0]
ax1.plot(df_hydro['date'], df_hydro['discharge_cfs'], color='#1f77b4', lw=1.2, label='Daily Mean Streamflow (cfs)')
ax1.plot(df_hydro['date'], df_hydro['baseline_365d_median'], color='black', lw=1.0, ls='--', label='365-Day Rolling Baseline')
ax1.axhline(10000, color='orange', ls='--', lw=1.1, label='Moderate Flood Stage (10,000 cfs)')
ax1.axhline(25000, color='red', ls='--', lw=1.3, label='Major Severe Flood Stage (25,000 cfs)')

flood_events = [
    ('1995-10-05', '1995 Hurricane Opal\\n(24.1k cfs)', 24100),
    ('2004-09-08', '2004 Hurricane Frances\\n(30.8k cfs)', 30800),
    ('2004-09-17', '2004 Hurricane Ivan\\n(40.5k cfs)', 40500),
    ('2013-01-30', '2013 Winter Flood\\n(18.6k cfs)', 18600),
    ('2021-08-18', '2021 Tropical Storm Fred\\n(25.1k cfs)', 25100),
    ('2024-09-27', '2024 Hurricane Helene\\n(50.0k+ cfs)', 50000)
]

for date_str, label, y_val in flood_events:
    dt = pd.to_datetime(date_str)
    if dt in df_hydro['date'].values:
        actual = df_hydro.loc[df_hydro['date'] == dt, 'discharge_cfs'].values[0]
        ax1.scatter(dt, actual, color='red', s=45, zorder=5)
        offset_y = 6000 if actual < 40000 else -12000
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual + offset_y),
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.1),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax1.set_title("USGS Site 03451500 (French Broad River at Asheville, NC): 35-Year Multi-Flood Record (1990–Present)", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Discharge (cfs)", fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)

# 2. Daily Rate of Rise / Velocity (delta cfs / day)
ax2 = axes[1]
ax2.plot(df_hydro['date'], df_hydro['velocity_cfs_day'] / 1000.0, color='#9467bd', lw=1.0, label='Daily Rate of Rise (1,000 cfs / day)')
ax2.axhline(0, color='black', lw=0.8)
ax2.fill_between(df_hydro['date'], 0, df_hydro['velocity_cfs_day'] / 1000.0, 
                 where=(df_hydro['velocity_cfs_day'] > 5000), color='darkred', alpha=0.5, label='Extreme Flash Surge (>5k cfs/day)')
ax2.set_ylabel("Rate of Rise (k-cfs/day)", fontsize=11, fontweight='bold')
ax2.set_title("Hydrological Impulse Acceleration: Flash Flood Rise Velocity", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Surge Severity Multiplier (x Normal Baseline)
ax3 = axes[2]
ax3.plot(df_hydro['date'], df_hydro['surge_multiplier'], color='#d62728', lw=1.2, label='Surge Multiplier (x 365d Median)')
ax3.axhline(1.0, color='black', lw=0.8, ls='--')
ax3.axhline(10.0, color='orange', ls='--', lw=1.0, label='10x Baseline Surge')
ax3.axhline(30.0, color='darkred', ls=':', lw=1.2, label='30x Catastrophic Flood Shock')
ax3.fill_between(df_hydro['date'], 10.0, df_hydro['surge_multiplier'], 
                 where=(df_hydro['surge_multiplier'] >= 10.0), color='red', alpha=0.3)
ax3.set_ylabel("Surge Multiplier", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('data/usgs_decadal_flood_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Historical Flood Crest Ranking & Processed Export"""),
    nbf.v4.new_code_cell("""# Summary table of top 12 flood crests
crests_df = df_hydro[df_hydro['is_flood_crest'] == 1].sort_values('discharge_cfs', ascending=False).head(12)
crests_table = crests_df[['date', 'discharge_cfs', 'velocity_cfs_day', 'surge_multiplier', 'discharge_zscore_3y']].copy()
crests_table['date'] = crests_table['date'].dt.strftime('%Y-%m-%d')
print("=== Top Historical French Broad River Flood Crests (1990–Present) ===")
print(crests_table.to_string(index=False))

# Export clean processed dataset
out_path = 'data/usgs_decadal_flood_processed.csv'
df_hydro.to_csv(out_path, index=False)
print(f"\\nProcessed decadal flood dataset successfully saved to: {out_path}")""")
]

with open('11_usgs_decadal_river_flood_spikes.ipynb', 'w') as f:
    nbf.write(nb11, f)
print("Created 11_usgs_decadal_river_flood_spikes.ipynb")


# ==============================================================================
# NOTEBOOK 12: AWS CloudWatch Server Network Traffic Blasts (NAB Benchmark)
# ==============================================================================
nb12 = create_nb()
nb12.cells = [
    nbf.v4.new_markdown_cell(r"""# 🖥️ Cloud Infrastructure Telemetry: AWS CloudWatch Traffic Blasts (NAB Benchmark)
## Distributed Server Shocks, Volumetric Traffic Surges & Service Outages

### 1. Incident Background & Multi-Spike Structure
In cloud computing and site reliability engineering (SRE), production infrastructure operates within bounded capacity envelopes. When anomalous events occur—such as **distributed denial-of-service (DDoS) attacks, microservice cascading retry storms, or runaway batch data pipelines**—server metrics experience extreme vertical impulses.

This notebook analyzes high-frequency production telemetry from the **Numenta Anomaly Benchmark (NAB)**, captured from production Amazon Web Services (AWS) EC2 instances at 5-minute sampling intervals.

Key telemetry dynamics:
* **Baseline Operating Regime:** Normal network ingress rate hovers quietly between **50 and 5,000 bytes / 5-min**.
* **Volumetric Surge Shocks:** The instance experiences sudden, vertical packet bursts where incoming traffic explodes to **8,285,420 bytes / 5-min**—a **+16,000,000% vertical surge** that overwhelms ingress network queues and triggers CPU saturation.
* **Cascading Failures:** Multiple recurring surges occur across days, followed by cliff drops as load-balancers drop unresponsive nodes.

### 2. Target Variables ($y$)
* **`network_in_bytes`**: Bytes received per 5-minute window on the network interface.
* **`traffic_velocity`**: First difference rate of change ($\Delta y = y_t - y_{t-1}$).
* **`surge_ratio`**: Current network ingress relative to the 24-hour rolling median.

### 3. Data Source
* **Numenta Anomaly Benchmark (NAB)** open GitHub repository (`realAWSCloudwatch/ec2_network_in_5abac7.csv`).
* Real anonymized AWS production telemetry.
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
    nbf.v4.new_code_cell("""def fetch_aws_network_telemetry(cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'aws_network_traffic_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached AWS CloudWatch telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    print("Downloading raw AWS CloudWatch production metrics from NAB repository...")
    url = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_network_in_5abac7.csv'
    df = pd.read_csv(url)
    df = df.rename(columns={'value': 'network_in_bytes'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df):,} 5-minute telemetry intervals.")
    return df

df_aws = fetch_aws_network_telemetry()
df_aws.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Server Stress Feature Engineering & Spike Identification"""),
    nbf.v4.new_code_cell("""df_aws_proc = df_aws.copy().sort_values('timestamp').reset_index(drop=True)

# Convert to Kilobytes (KB) and Megabytes (MB)
df_aws_proc['network_in_kb'] = df_aws_proc['network_in_bytes'] / 1024.0
df_aws_proc['network_in_mb'] = df_aws_proc['network_in_bytes'] / (1024.0 * 1024.0)

# 1. Traffic velocity (Rate of change in KB per 5-min step)
df_aws_proc['velocity_kb_step'] = df_aws_proc['network_in_kb'].diff()

# 2. 24-hour trailing median baseline (288 steps of 5-min)
df_aws_proc['baseline_24h_median_kb'] = df_aws_proc['network_in_kb'].rolling(288, min_periods=36).median()

# 3. Ingress Surge Ratio
df_aws_proc['surge_ratio'] = df_aws_proc['network_in_kb'] / df_aws_proc['baseline_24h_median_kb'].replace(0, np.nan)

# 4. Rolling 12-hour Z-Score
roll_mean_12h = df_aws_proc['network_in_kb'].rolling(144, min_periods=24).mean()
roll_std_12h = df_aws_proc['network_in_kb'].rolling(144, min_periods=24).std()
df_aws_proc['traffic_zscore_12h'] = (df_aws_proc['network_in_kb'] - roll_mean_12h) / roll_std_12h

# 5. Automated Peak Detection of Volumetric Network Blasts
peaks, _ = find_peaks(df_aws_proc['network_in_mb'].values, height=2.0, distance=36, prominence=1.5)
df_aws_proc['is_traffic_blast'] = 0
df_aws_proc.loc[peaks, 'is_traffic_blast'] = 1

print(f"Total 5-Minute Telemetry Windows:        {len(df_aws_proc):,}")
print(f"Baseline Median Network Ingress:          {df_aws_proc['network_in_kb'].median():.2f} KB / 5-min")
print(f"Peak Network Ingress Reached:             {df_aws_proc['network_in_mb'].max():.2f} MB / 5-min ({df_aws_proc['network_in_bytes'].max():,} bytes)")
print(f"Peak Blast Surge vs Median:               +{df_aws_proc['network_in_bytes'].max() / df_aws_proc['network_in_bytes'].median():,.0f}x baseline")
print(f"Detected Severe Volumetric Blasts (>=2MB): {len(peaks)}")
df_aws_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of Cloud Infrastructure Traffic Blasts"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Network Ingress Traffic Timeline (MB / 5-min) with Blast Annotations
ax1 = axes[0]
ax1.plot(df_aws_proc['timestamp'], df_aws_proc['network_in_mb'], color='#1f77b4', lw=1.2, label='Network Ingress (MB / 5-min)')
ax1.axhline(2.0, color='orange', ls='--', lw=1.1, label='Volumetric Blast Threshold (2.0 MB)')
ax1.axhline(5.0, color='red', ls='--', lw=1.3, label='Critical Queue Overflow Threshold (5.0 MB)')

for p_idx in peaks:
    ts = df_aws_proc.loc[p_idx, 'timestamp']
    val = df_aws_proc.loc[p_idx, 'network_in_mb']
    ax1.scatter(ts, val, color='red', s=45, zorder=5)
    ax1.annotate(f"{val:.1f} MB\\n({ts.strftime('%b %d %H:%M')})", 
                 xy=(ts, val), xytext=(ts, val * 0.75),
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.1),
                 fontsize=8.0, fontweight='bold', color='#8b0000', ha='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax1.set_title("AWS Production EC2 Instance: CloudWatch Ingress Traffic Blasts & Overload Shocks", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Ingress (MB / 5-min)", fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True)

# 2. Traffic Velocity / Rate of Rise (delta KB / 5-min)
ax2 = axes[1]
ax2.plot(df_aws_proc['timestamp'], df_aws_proc['velocity_kb_step'] / 1024.0, color='#9467bd', lw=1.0, label='Traffic Acceleration (MB / step)')
ax2.axhline(0, color='black', lw=0.8)
ax2.fill_between(df_aws_proc['timestamp'], 0, df_aws_proc['velocity_kb_step'] / 1024.0, 
                 where=(df_aws_proc['velocity_kb_step'] > 1024), color='red', alpha=0.5, label='Acute Inflow Burst (>1 MB/step)')
ax2.set_ylabel("Acceleration (MB/step)", fontsize=11, fontweight='bold')
ax2.set_title("Network Surge Velocity: Sudden Inflow Impulses", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Rolling 12-Hour Standardized Z-Score
ax3 = axes[2]
ax3.plot(df_aws_proc['timestamp'], df_aws_proc['traffic_zscore_12h'], color='#d62728', lw=1.1, label='12h Rolling Z-Score (sigma)')
ax3.axhline(0, color='black', lw=0.8)
ax3.axhline(3.0, color='gold', ls='--', lw=1.0, label='+3-Sigma Alert')
ax3.axhline(6.0, color='darkred', ls=':', lw=1.2, label='+6-Sigma Outlier')
ax3.fill_between(df_aws_proc['timestamp'], 3.0, df_aws_proc['traffic_zscore_12h'], 
                 where=(df_aws_proc['traffic_zscore_12h'] >= 3.0), color='red', alpha=0.3)
ax3.set_ylabel("Z-Score (sigma)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Operational Timeline", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/aws_traffic_spikes_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Ingress Blast Ranking Table & Processed Export"""),
    nbf.v4.new_code_cell("""# Summary table of top traffic blasts
blast_df = df_aws_proc[df_aws_proc['is_traffic_blast'] == 1].sort_values('network_in_mb', ascending=False)
blast_table = blast_df[['timestamp', 'network_in_mb', 'velocity_kb_step', 'surge_ratio', 'traffic_zscore_12h']].copy()
blast_table['timestamp'] = blast_table['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
print("=== Top AWS CloudWatch Volumetric Traffic Blasts ===")
print(blast_table.to_string(index=False))

# Export clean processed dataset
out_path = 'data/aws_traffic_spikes_processed.csv'
df_aws_proc.to_csv(out_path, index=False)
print(f"\\nProcessed AWS traffic dataset successfully saved to: {out_path}")""")
]

with open('12_aws_cloudwatch_server_traffic_spikes.ipynb', 'w') as f:
    nbf.write(nb12, f)
print("Created 12_aws_cloudwatch_server_traffic_spikes.ipynb")


# ==============================================================================
# NOTEBOOK 13: NOAA Space Weather Solar Flare X-Ray Flux Spikes
# ==============================================================================
nb13 = create_nb()
nb13.cells = [
    nbf.v4.new_markdown_cell(r"""# ☀️ Space Weather & Solar Physics: NOAA GOES Solar Flare Spikes
## Extreme Coronal Mass Ejections, Multi-Magnitude X-Ray Radiation Explosions & Geomagnetic Shocks

### 1. Incident Background & Multi-Spike Radiation Dynamics
In solar astrophysics and space weather forecasting, solar flares are explosive releases of magnetic energy in the Sun's corona. When magnetic reconnection occurs in active sunspot regions, the Sun emits intense bursts of electromagnetic radiation across the X-ray and EUV spectrum.

The **NOAA Space Weather Prediction Center (SWPC)** monitors real-time solar X-ray flux using primary **GOES (Geostationary Operational Environmental Satellite)** instruments at 1-minute cadence in two wavelength bands:
* **Shortwave (`0.05 - 0.4 nm`)**
* **Longwave (`0.1 - 0.8 nm`)**

### 2. Multi-Magnitude Logarithmic Shock Classification
Solar flares are classified on a logarithmic scale based on peak longwave X-ray flux ($W/m^2$):
* **A-Class ($< 10^{-7} \text{W/m}^2$):** Background quiet solar state.
* **B-Class ($10^{-7} \text{ to } 10^{-6} \text{W/m}^2$):** Low background activity.
* **C-Class ($10^{-6} \text{ to } 10^{-5} \text{W/m}^2$):** Small flares with minor terrestrial impact.
* **M-Class ($10^{-5} \text{ to } 10^{-4} \text{W/m}^2$):** Medium-strength flares causing brief high-frequency radio blackouts at Earth's polar caps.
* **X-Class ($\ge 10^{-4} \text{W/m}^2$):** Major coronal explosions that trigger planet-wide radio blackouts, satellite telemetry interference, and extreme geomagnetic storms (e.g. the historic May 2024 G5 solar storm).

When a flare erupts, radiation **explodes by 2 to 4 orders of magnitude within 5 to 15 minutes**, followed by hours of gradual thermal plasma cooling.

### 3. Data Source
* NOAA Space Weather Prediction Center (SWPC) REST API (`services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json`).
* 100% public, real-time satellite telemetry, zero authentication.
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
    nbf.v4.new_code_cell("""def fetch_noaa_solar_xrays(cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'noaa_solar_xrays_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached NOAA solar X-ray telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        return df
    
    print("Fetching high-frequency solar X-ray telemetry from NOAA SWPC...")
    url = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    df_raw = pd.DataFrame(data)
    df_raw.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df_raw):,} raw X-ray readings.")
    df_raw['time_tag'] = pd.to_datetime(df_raw['time_tag'])
    return df_raw

df_xray_raw = fetch_noaa_solar_xrays()
df_xray_raw.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Solar Flare Feature Engineering & Log-Scale Shock Segmentation"""),
    nbf.v4.new_code_cell("""# Filter to standard longwave band (0.1 - 0.8 nm) used for official NOAA flare classification
df_solar = df_xray_raw[df_xray_raw['energy'] == '0.1-0.8nm'].copy()
df_solar['flux'] = pd.to_numeric(df_solar['flux'], errors='coerce')
df_solar = df_solar.dropna(subset=['flux']).sort_values('time_tag').reset_index(drop=True)

# 1. Log10 Flux (orders of magnitude)
df_solar['log10_flux'] = np.log10(df_solar['flux'])

# 2. Flare Class Categorization
# A: < 1e-7, B: 1e-7 to 1e-6, C: 1e-6 to 1e-5, M: 1e-5 to 1e-4, X: >= 1e-4
def classify_flare(flux_val):
    if flux_val >= 1e-4:
        return f"X{flux_val/1e-4:.1f}"
    elif flux_val >= 1e-5:
        return f"M{flux_val/1e-5:.1f}"
    elif flux_val >= 1e-6:
        return f"C{flux_val/1e-6:.1f}"
    elif flux_val >= 1e-7:
        return f"B{flux_val/1e-7:.1f}"
    else:
        return "A"

df_solar['flare_class'] = df_solar['flux'].apply(classify_flare)

# 3. Flare Velocity (First Difference of log-flux / 1-min)
df_solar['flux_velocity_per_min'] = df_solar['flux'].diff()
df_solar['log_velocity_per_min'] = df_solar['log10_flux'].diff()

# 4. Trailing 6-hour baseline flux median
df_solar['baseline_6h_median'] = df_solar['flux'].rolling(360, min_periods=30).median()
df_solar['surge_multiplier'] = df_solar['flux'] / df_solar['baseline_6h_median']

# 5. Automated Peak Detection of Solar Flare Spikes
# Detect prominent peaks on log-scale
peaks, _ = find_peaks(df_solar['log10_flux'].values, distance=45, prominence=0.35)
df_solar['is_flare_peak'] = 0
df_solar.loc[peaks, 'is_flare_peak'] = 1

print(f"Total 1-Minute Telemetry Records:      {len(df_solar):,}")
print(f"Quiet Baseline Flux (Median):           {df_solar['flux'].median():.2e} W/m^2")
print(f"Maximum Solar Flare Peak:               {df_solar['flux'].max():.2e} W/m^2 ({classify_flare(df_solar['flux'].max())})")
print(f"Detected Prominent Flare Peaks:         {len(peaks)} spikes")
print(f"Peak Surge Multiplier vs Baseline:      +{df_solar['surge_multiplier'].max():.1f}x")
df_solar.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of Satellite Solar Flare Spikes"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Log-Scale Solar X-Ray Flux with NOAA Flare Class Bands
ax1 = axes[0]
ax1.plot(df_solar['time_tag'], df_solar['flux'], color='#d62728', lw=1.2, label='GOES 0.1-0.8nm X-ray Flux (W/m^2)')
ax1.set_yscale('log')

# Class boundary horizontal lines
ax1.axhline(1e-6, color='goldenrod', ls='--', lw=1.0, alpha=0.8, label='C-Class Threshold (10^-6 W/m^2)')
ax1.axhline(1e-5, color='darkorange', ls='--', lw=1.2, alpha=0.9, label='M-Class Threshold (10^-5 W/m^2)')
ax1.axhline(1e-4, color='red', ls='--', lw=1.4, alpha=0.9, label='X-Class Threshold (10^-4 W/m^2)')

# Annotate prominent detected flare peaks
top_peaks = df_solar[df_solar['is_flare_peak'] == 1].sort_values('flux', ascending=False).head(6)
for _, row in top_peaks.iterrows():
    ts = row['time_tag']
    val = row['flux']
    f_class = row['flare_class']
    ax1.scatter(ts, val, color='purple', s=45, zorder=5)
    ax1.annotate(f"{f_class}\\n({ts.strftime('%b %d %H:%M')})", 
                 xy=(ts, val), xytext=(ts, val * 1.8),
                 arrowprops=dict(arrowstyle="->", color='purple', lw=1.1),
                 fontsize=8.0, fontweight='bold', color='purple', ha='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='purple'))

ax1.set_title("NOAA Primary GOES Satellite: Real-Time Solar X-Ray Flare Spikes (0.1–0.8 nm)", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Flux (W/m^2, Log Scale)", fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True)

# 2. Surge Multiplier Relative to Quiet Background
ax2 = axes[1]
ax2.plot(df_solar['time_tag'], df_solar['surge_multiplier'], color='#ff7f0e', lw=1.2, label='Surge Factor (x 6h Baseline)')
ax2.axhline(1.0, color='black', lw=0.8, ls='--')
ax2.axhline(5.0, color='orange', ls='--', lw=1.0, label='5x Quiet Background')
ax2.axhline(10.0, color='darkred', ls=':', lw=1.2, label='10x Severe Radiation Spike')
ax2.fill_between(df_solar['time_tag'], 5.0, df_solar['surge_multiplier'], 
                 where=(df_solar['surge_multiplier'] >= 5.0), color='#ff7f0e', alpha=0.3)
ax2.set_ylabel("Surge Multiplier", fontsize=11, fontweight='bold')
ax2.set_title("Relative Radiation Impulse: Energy Release Factor", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Log-Flux Velocity (Instantaneous Explosion Rate / min)
ax3 = axes[2]
ax3.plot(df_solar['time_tag'], df_solar['log_velocity_per_min'], color='#2ca02c', lw=0.8, alpha=0.7)
ax3.fill_between(df_solar['time_tag'], 0, df_solar['log_velocity_per_min'], 
                 where=(df_solar['log_velocity_per_min'] > 0.05), color='red', alpha=0.6, label='Explosive Reconnection Phase')
ax3.fill_between(df_solar['time_tag'], 0, df_solar['log_velocity_per_min'], 
                 where=(df_solar['log_velocity_per_min'] < -0.05), color='blue', alpha=0.4, label='Thermal Plasma Cooling')
ax3.axhline(0, color='black', lw=0.8)
ax3.set_ylabel("d(Log Flux) / dt", fontsize=11, fontweight='bold')
ax3.set_xlabel("Observation Timeline (UTC)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('data/noaa_solar_flares_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Solar Flare Spike Ranking Table & Processed Export"""),
    nbf.v4.new_code_cell("""# Summary table of detected flare peaks
flares_df = df_solar[df_solar['is_flare_peak'] == 1].sort_values('flux', ascending=False).head(12)
flare_table = flares_df[['time_tag', 'flux', 'flare_class', 'surge_multiplier', 'log_velocity_per_min']].copy()
flare_table['time_tag'] = flare_table['time_tag'].dt.strftime('%Y-%m-%d %H:%M')
print("=== Top Detected NOAA Solar Flare Radiation Spikes ===")
print(flare_table.to_string(index=False))

# Export clean processed dataset
out_path = 'data/noaa_solar_flares_processed.csv'
df_solar.to_csv(out_path, index=False)
print(f"\\nProcessed solar flare dataset successfully saved to: {out_path}")""")
]

with open('13_noaa_space_weather_solar_flare_spikes.ipynb', 'w') as f:
    nbf.write(nb13, f)
print("Created 13_noaa_space_weather_solar_flare_spikes.ipynb")


# ==============================================================================
# NOTEBOOK 14: Henry Hub Natural Gas Extreme Weather Freeze Spikes (1997–Present)
# ==============================================================================
nb14 = create_nb()
nb14.cells = [
    nbf.v4.new_markdown_cell(r"""# ❄️ Energy Thermodynamics: Natural Gas Weather Freeze Spikes (FRED: 1997–Present)
## Polar Vortex Shocks, Pipeline Wellhead Freezes & Hurricane Supply Disruptions

### 1. Incident Background & Multi-Spike Commodity Physics
Natural gas is the primary fuel used for both **winter space heating** and **peaking electricity generation** across North America. Unlike oil (which can be stockpiled in tanks), natural gas is transported instantaneously via pressurized interstate pipeline networks with limited linepack storage.

When severe **polar vortex freezes** descend upon North America, two physical shocks collide simultaneously:
1. **Demand Explosion:** Residential heating furnaces and gas-fired power turbines fire at maximum capacity.
2. **Supply Collapse ("Freeze-Offs"):** Water vapor produced alongside raw methane freezes inside wellheads and gathering lines, physically choking off supply.

Across this 28-year benchmark from the Federal Reserve Bank of St. Louis (FRED `DHHNGSP`, covering 1997 to Present), the **Henry Hub Spot Price ($/MMBtu)** idles at a peaceful baseline of **$2.00 to $3.50 / MMBtu**, but experiences violent **+500% to +1,000% vertical price spikes** during extreme weather shocks:
* **February 2003 Polar Freeze:** Spiked to **$18.48 / MMBtu**.
* **September–December 2005 (Hurricanes Katrina & Rita):** Offshore Gulf production knocked offline; prices spiked to **$14.84 / MMBtu**.
* **January–February 2014 "Polar Vortex":** Arctic air paralyzed the Midwest; regional prices topped $100/MMBtu.
* **February 2021 Winter Storm Uri:** Historic deep freeze crippled Texas and Midcontinent grids; Henry Hub surged to **$23.86 / MMBtu** (and regional cash hubs traded over **$1,000 / MMBtu**).
* **January 2026 Arctic Blast:** Historic Arctic front pushing spot Henry Hub to **$30.72 / MMBtu**.

### 2. Target Variables ($y$)
* **`DHHNGSP`**: Henry Hub Natural Gas Spot Price ($/Million Btu).
* **`price_change_1d`**: Daily velocity ($\Delta y = y_t - y_{t-1}$).
* **`surge_multiplier`**: Daily price relative to the 365-day rolling median baseline.

### 3. Data Source
* Federal Reserve Bank of St. Louis (**FRED**) REST CSV Service (`DHHNGSP`).
* 100% public, free, zero API keys required.
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
    nbf.v4.new_code_cell("""def fetch_henry_hub_prices(cache_dir='data'):
    cache_path = os.path.join(cache_dir, 'henry_hub_natural_gas_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached Henry Hub telemetry from: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        print("Fetching Henry Hub daily spot prices directly from FRED...")
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP'
        df = pd.read_csv(url)
        df.to_csv(cache_path, index=False)
        print(f"Downloaded and cached {len(df):,} daily spot price records.")
        
    date_col = 'observation_date' if 'observation_date' in df.columns else 'DATE'
    df = df.rename(columns={date_col: 'DATE'})
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['price'] = pd.to_numeric(df['DHHNGSP'], errors='coerce')
    df = df.dropna(subset=['price']).sort_values('DATE').reset_index(drop=True)
    return df

df_gas = fetch_henry_hub_prices()
df_gas.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Freeze Shock Feature Engineering & Spike Segmentation"""),
    nbf.v4.new_code_cell("""df_gas_proc = df_gas.copy().sort_values('DATE').reset_index(drop=True)

# 1. Daily dollar change and percentage return
df_gas_proc['price_change_1d'] = df_gas_proc['price'].diff()
df_gas_proc['return_pct'] = df_gas_proc['price'].pct_change() * 100.0

# 2. 365-day rolling median baseline
df_gas_proc['baseline_365d_median'] = df_gas_proc['price'].rolling(365, min_periods=60).median()

# 3. Surge Multiplier relative to baseline
df_gas_proc['surge_multiplier'] = df_gas_proc['price'] / df_gas_proc['baseline_365d_median']

# 4. Rolling 1-year Z-Score
roll_mean_1y = df_gas_proc['price'].rolling(252, min_periods=40).mean()
roll_std_1y = df_gas_proc['price'].rolling(252, min_periods=40).std()
df_gas_proc['price_zscore_1y'] = (df_gas_proc['price'] - roll_mean_1y) / roll_std_1y

# 5. Automated Peak Detection of Weather Shock Spikes (price >= $12, distance >= 60 days)
peaks, _ = find_peaks(df_gas_proc['price'].values, height=12.0, distance=60, prominence=5.0)
df_gas_proc['is_freeze_spike'] = 0
df_gas_proc.loc[peaks, 'is_freeze_spike'] = 1

print(f"Total Trading Sessions Analyzed:         {len(df_gas_proc):,}")
print(f"Historical Median Spot Price:             USD {df_gas_proc['price'].median():.2f} / MMBtu")
print(f"Detected Severe Freeze/Storm Spikes:      {len(peaks)} major episodes")
print(f"All-Time Record Spot Price:               USD {df_gas_proc['price'].max():.2f} / MMBtu on {df_gas_proc.loc[df_gas_proc['price'].idxmax(), 'DATE'].strftime('%Y-%m-%d')}")
print(f"Max Single-Day Surge:                     +USD {df_gas_proc['price_change_1d'].max():.2f} (+{df_gas_proc['return_pct'].max():.1f}%)")
print(f"Peak Surge Multiplier vs Baseline:        {df_gas_proc['surge_multiplier'].max():.1f}x normal price")
df_gas_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of 28 Years of Energy Freeze Spikes"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. 28-Year Spot Price Timeline with Extreme Weather Annotations
ax1 = axes[0]
ax1.plot(df_gas_proc['DATE'], df_gas_proc['price'], color='#1f77b4', lw=1.2, label='Henry Hub Spot Price (USD / MMBtu)')
ax1.plot(df_gas_proc['DATE'], df_gas_proc['baseline_365d_median'], color='black', lw=1.0, ls='--', label='365-Day Rolling Baseline')
ax1.axhline(10.0, color='orange', ls='--', lw=1.1, label='Extreme Stress Threshold (USD 10)')
ax1.axhline(20.0, color='red', ls='--', lw=1.3, label='Crisis Freeze Threshold (USD 20)')

gas_events = [
    ('2003-02-25', '2003 Polar Freeze\\n(USD 18.48)', 18.48),
    ('2005-09-22', 'Hurricanes Katrina/Rita\\n(USD 14.84)', 14.84),
    ('2008-07-03', '2008 Commodity Boom\\n(USD 13.31)', 13.31),
    ('2014-02-19', '2014 Polar Vortex\\n(USD 8.15)', 8.15),
    ('2021-02-17', 'Winter Storm Uri\\n(USD 23.86)', 23.86),
    ('2026-01-23', '2026 Arctic Blast\\n(USD 30.72)', 30.72)
]

for date_str, label, y_val in gas_events:
    dt = pd.to_datetime(date_str)
    if dt in df_gas_proc['DATE'].values:
        actual = df_gas_proc.loc[df_gas_proc['DATE'] == dt, 'price'].values[0]
        ax1.scatter(dt, actual, color='red', s=45, zorder=5)
        offset_y = 5.0 if actual < 25 else -6.0
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual + offset_y),
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.1),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax1.set_title("Henry Hub Natural Gas (1997–Present): Multi-Decade Weather Freeze & Supply Shocks", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Spot Price (USD / MMBtu)", fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)

# 2. Daily Price Velocity (delta $/MMBtu / day)
ax2 = axes[1]
ax2.plot(df_gas_proc['DATE'], df_gas_proc['price_change_1d'], color='#9467bd', lw=1.0, label='Daily Point Change (USD / day)')
ax2.axhline(0, color='black', lw=0.8)
ax2.fill_between(df_gas_proc['DATE'], 0, df_gas_proc['price_change_1d'], 
                 where=(df_gas_proc['price_change_1d'] >= 2.0), color='darkred', alpha=0.6, label='Severe Jump Hike (>= +USD 2/day)')
ax2.fill_between(df_gas_proc['DATE'], 0, df_gas_proc['price_change_1d'], 
                 where=(df_gas_proc['price_change_1d'] <= -2.0), color='teal', alpha=0.6, label='Post-Freeze Thaw Collapse (<= -USD 2/day)')
ax2.set_ylabel("Daily Delta (USD)", fontsize=11, fontweight='bold')
ax2.set_title("Commodity Jump Velocity: Freeze-Off Price Accelerations vs. Thaw Collapses", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Surge Multiplier (Price / 365d Median)
ax3 = axes[2]
ax3.plot(df_gas_proc['DATE'], df_gas_proc['surge_multiplier'], color='#d62728', lw=1.2, label='Surge Multiplier (x Baseline)')
ax3.axhline(1.0, color='black', lw=0.8, ls='--')
ax3.axhline(3.0, color='orange', ls='--', lw=1.0, label='3x Baseline')
ax3.axhline(6.0, color='darkred', ls=':', lw=1.2, label='6x Severe Weather Outlier')
ax3.fill_between(df_gas_proc['DATE'], 3.0, df_gas_proc['surge_multiplier'], 
                 where=(df_gas_proc['surge_multiplier'] >= 3.0), color='red', alpha=0.3)
ax3.set_ylabel("Surge Multiplier", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('data/henry_hub_freeze_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Historical Freeze Shock Ranking & Processed Export"""),
    nbf.v4.new_code_cell("""# Summary table of top energy freeze spikes
freeze_df = df_gas_proc[df_gas_proc['is_freeze_spike'] == 1].sort_values('price', ascending=False).head(12)
freeze_table = freeze_df[['DATE', 'price', 'price_change_1d', 'surge_multiplier', 'price_zscore_1y']].copy()
freeze_table['DATE'] = freeze_table['DATE'].dt.strftime('%Y-%m-%d')
print("=== Top Historical Henry Hub Weather Freeze Spikes ===")
print(freeze_table.to_string(index=False))

# Export clean processed dataset
out_path = 'data/henry_hub_freeze_processed.csv'
df_gas_proc.to_csv(out_path, index=False)
print(f"\\nProcessed natural gas dataset successfully saved to: {out_path}")""")
]

with open('14_henry_hub_natural_gas_freeze_spikes.ipynb', 'w') as f:
    nbf.write(nb14, f)
print("Created 14_henry_hub_natural_gas_freeze_spikes.ipynb")


# ==============================================================================
# NOTEBOOK 15: Wikipedia Breaking Disaster Attention Spikes (2016–Present)
# ==============================================================================
nb15 = create_nb()
nb15.cells = [
    nbf.v4.new_markdown_cell(r"""# 🌐 Collective Attention Shocks: Wikipedia Disaster Pageview Spikes (2016–Present)
## Human Information Inflow, Breaking Crisis Attention Surges & Power-Law Decay Dynamics

### 1. Incident Background & Information Shock Dynamics
When sudden catastrophic natural disasters or geopolitical emergencies strike, hundreds of thousands of people simultaneously search for real-time situational awareness. In information theory and socio-technical telemetry, **Wikipedia pageview traffic serves as an instantaneous sensor for global collective attention**.

Under normal circumstances, an article such as English Wikipedia's **"Earthquake"** idles in a stationary, quiet baseline of **~2,000 to 3,000 daily pageviews**. 

However, when major seismic disasters rupture fault lines worldwide, pageviews experience **instantaneous vertical explosions of 100x to 250x (+10,000% to +25,000% within 24 hours)**, reaching **80,000 to 200,000+ views in a single day**, followed by power-law memory decay:
* **November 2016 (Kaikōura, New Zealand M7.8):** Spiked to **78,000 views**.
* **September 2017 (Puebla, Mexico M7.1):** Spiked to **75,000 views**.
* **February 6, 2023 (Kahramanmaraş, Turkey–Syria M7.8):** Catastrophic earthquake killing 50,000+; pageviews surged to **87,865 views**.
* **March 21, 2023 (Badakhshan, Afghanistan M6.5):** Spiked to all-time record crest of **200,477 views**.
* **November 3, 2023 (Nepal M5.7):** Surged to **126,797 views**.
* **January 1, 2024 (Noto Peninsula, Japan M7.5):** New Year's Day seismic disaster driving immediate spike to **72,000 views**.

### 2. Target Variables ($y$)
* **`views`**: Total daily human and automated pageviews on English Wikipedia.
* **`velocity_views_day`**: First difference daily attention acceleration ($\Delta y = y_t - y_{t-1}$).
* **`attention_multiplier`**: Daily views relative to the trailing 30-day median baseline ($y_t / \text{Median}_{30}(y)$).

### 3. Data Source
* **Wikimedia Foundation REST API** (`wikimedia.org/api/rest_v1/metrics/pageviews/`).
* 100% public, free, zero API keys required.
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
    nbf.v4.new_code_cell("""def fetch_wikipedia_disaster_pageviews(article='Earthquake', start_date='20160101', cache_dir='data'):
    clean_name = article.lower().replace(' ', '_')
    cache_path = os.path.join(cache_dir, f'wikipedia_{clean_name}_views_raw.csv')
    
    if os.path.exists(cache_path):
        print(f"Loading cached Wikipedia pageviews from: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    print(f"Fetching daily Wikipedia pageviews for '{article}' ({start_date} to {end_date})...")
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start_date}/{end_date}'
    headers = {'User-Agent': 'SpikesBenchmarkSuite/1.0 (academic research test; mrunal@ufl.edu)'}
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data['items'])
    df['date'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H')
    df['views'] = pd.to_numeric(df['views'], errors='coerce')
    df = df[['date', 'views', 'article']].dropna(subset=['views']).sort_values('date').reset_index(drop=True)
    
    df.to_csv(cache_path, index=False)
    print(f"Downloaded and cached {len(df):,} daily pageview records.")
    return df

df_wiki = fetch_wikipedia_disaster_pageviews()
df_wiki.head()"""),
    nbf.v4.new_markdown_cell("""### 5. Collective Attention Feature Engineering & Shock Segmentation"""),
    nbf.v4.new_code_cell("""df_wiki_proc = df_wiki.copy().sort_values('date').reset_index(drop=True)

# 1. Daily rate of change (acceleration of public awareness)
df_wiki_proc['velocity_views_day'] = df_wiki_proc['views'].diff()

# 2. 30-day trailing baseline median
df_wiki_proc['baseline_30d_median'] = df_wiki_proc['views'].rolling(30, min_periods=7).median()

# 3. Attention Multiplier relative to baseline
df_wiki_proc['attention_multiplier'] = df_wiki_proc['views'] / df_wiki_proc['baseline_30d_median']

# 4. Rolling 90-day Z-Score
roll_mean_90 = df_wiki_proc['views'].rolling(90, min_periods=14).mean()
roll_std_90 = df_wiki_proc['views'].rolling(90, min_periods=14).std()
df_wiki_proc['views_zscore_90d'] = (df_wiki_proc['views'] - roll_mean_90) / roll_std_90

# 5. Automated Peak Detection of Disaster Attention Surges (views >= 30,000, distance >= 21 days)
peaks, _ = find_peaks(df_wiki_proc['views'].values, height=30000, distance=21, prominence=15000)
df_wiki_proc['is_attention_spike'] = 0
df_wiki_proc.loc[peaks, 'is_attention_spike'] = 1

print(f"Total Daily Observations:                {len(df_wiki_proc):,}")
print(f"Quiescent Baseline Traffic (Median):      {df_wiki_proc['views'].median():,.0f} views / day")
print(f"Detected Major Emergency Attention Surges:{len(peaks)} episodes")
print(f"All-Time Daily Attention Crest:           {df_wiki_proc['views'].max():,.0f} views on {df_wiki_proc.loc[df_wiki_proc['views'].idxmax(), 'date'].strftime('%Y-%m-%d')}")
print(f"Max Attention Surge Multiplier:           {df_wiki_proc['attention_multiplier'].max():.1f}x baseline traffic")
df_wiki_proc.head()"""),
    nbf.v4.new_markdown_cell("""### 6. Timeline Visualizations of Public Attention Shocks"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# 1. Daily Pageviews Timeline (Views / Day in Thousands) with Disaster Annotations
ax1 = axes[0]
ax1.plot(df_wiki_proc['date'], df_wiki_proc['views'] / 1000.0, color='#1f77b4', lw=1.2, label='Daily Pageviews (Thousands)')
ax1.plot(df_wiki_proc['date'], df_wiki_proc['baseline_30d_median'] / 1000.0, color='black', lw=1.0, ls='--', label='30-Day Rolling Baseline')
ax1.axhline(30.0, color='orange', ls='--', lw=1.1, label='Attention Surge Threshold (30k views)')
ax1.axhline(75.0, color='red', ls='--', lw=1.3, label='Global Disaster Shock Threshold (75k views)')

disaster_events = [
    ('2017-09-19', '2017 Mexico M7.1\\n(75k views)', 75),
    ('2023-02-06', 'Turkey-Syria M7.8\\n(88k views)', 88),
    ('2023-03-21', 'Afghanistan M6.5\\n(200k views)', 200),
    ('2023-11-03', '2023 Nepal M5.7\\n(127k views)', 127),
    ('2024-01-01', 'Japan Noto M7.5\\n(72k views)', 72)
]

for date_str, label, y_val in disaster_events:
    dt = pd.to_datetime(date_str)
    if dt in df_wiki_proc['date'].values:
        actual = df_wiki_proc.loc[df_wiki_proc['date'] == dt, 'views'].values[0] / 1000.0
        ax1.scatter(dt, actual, color='red', s=45, zorder=5)
        offset_y = 35 if actual < 150 else -45
        ax1.annotate(label, xy=(dt, actual), xytext=(dt, actual + offset_y),
                     arrowprops=dict(arrowstyle="->", color='red', lw=1.1),
                     fontsize=8.5, fontweight='bold', color='#8b0000', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))

ax1.set_title("Wikipedia 'Earthquake' Pageviews (2016–Present): Global Breaking Disaster Attention Shocks", 
              fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel("Views (Thousands / Day)", fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)

# 2. Daily Rate of Rise / Attention Acceleration (delta views / day)
ax2 = axes[1]
ax2.plot(df_wiki_proc['date'], df_wiki_proc['velocity_views_day'] / 1000.0, color='#9467bd', lw=1.0, label='Attention Velocity (k-views / day)')
ax2.axhline(0, color='black', lw=0.8)
ax2.fill_between(df_wiki_proc['date'], 0, df_wiki_proc['velocity_views_day'] / 1000.0, 
                 where=(df_wiki_proc['velocity_views_day'] > 20000), color='darkred', alpha=0.5, label='Acute Viral Inflow (>20k views/day)')
ax2.set_ylabel("Velocity (k-views/day)", fontsize=11, fontweight='bold')
ax2.set_title("Information Acceleration: Rapid Collective Inflow Dynamics", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True)

# 3. Attention Multiplier (Views / 30-Day Median)
ax3 = axes[2]
ax3.plot(df_wiki_proc['date'], df_wiki_proc['attention_multiplier'], color='#d62728', lw=1.2, label='Attention Multiplier (x Baseline)')
ax3.axhline(1.0, color='black', lw=0.8, ls='--')
ax3.axhline(10.0, color='orange', ls='--', lw=1.0, label='10x Baseline Attention')
ax3.axhline(50.0, color='darkred', ls=':', lw=1.2, label='50x Catastrophic Public Inflow')
ax3.fill_between(df_wiki_proc['date'], 10.0, df_wiki_proc['attention_multiplier'], 
                 where=(df_wiki_proc['attention_multiplier'] >= 10.0), color='red', alpha=0.3)
ax3.set_ylabel("Attention Multiplier", fontsize=11, fontweight='bold')
ax3.set_xlabel("Timeline (Years)", fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('data/wikipedia_attention_timeline.png', dpi=300)
plt.show()"""),
    nbf.v4.new_markdown_cell("""### 7. Disaster Attention Ranking Table & Processed Export"""),
    nbf.v4.new_code_cell("""# Summary table of top attention surges
surges_df = df_wiki_proc[df_wiki_proc['is_attention_spike'] == 1].sort_values('views', ascending=False).head(12)
surge_table = surges_df[['date', 'views', 'velocity_views_day', 'attention_multiplier', 'views_zscore_90d']].copy()
surge_table['date'] = surge_table['date'].dt.strftime('%Y-%m-%d')
print("=== Top Historical Disaster Attention Surges on Wikipedia ===")
print(surge_table.to_string(index=False))

# Export clean processed dataset
out_path = 'data/wikipedia_attention_processed.csv'
df_wiki_proc.to_csv(out_path, index=False)
print(f"\\nProcessed Wikipedia attention dataset successfully saved to: {out_path}")""")
]

with open('15_wikipedia_disaster_attention_spikes.ipynb', 'w') as f:
    nbf.write(nb15, f)
print("Created 15_wikipedia_disaster_attention_spikes.ipynb")

print("\\nNotebooks 11-15 generated successfully!")
