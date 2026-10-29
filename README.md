# Rare-Event Shocks & Extreme Time Series Benchmark Suite
### Ingestion, Preprocessing, and Timeline Visualizations for HiPerGator (UF HPC)

This repository contains a comprehensive, verified suite of **17 Jupyter Notebooks** designed to investigate **rare-event shocks, extreme time series anomalies, and recurring crisis dynamics** across financial markets, industrial processes, environmental hydrology, cloud infrastructure, solar astrophysics, energy thermodynamics, and human crisis information flows.

Each notebook is **fully self-contained**, downloads open-source data without requiring API keys, caches raw files locally to `./data/` for offline/HPC execution, implements feature engineering (event-centering, velocity, relative surge, rolling volatility, peak detection), contains **formal academic citations & BibTeX blocks**, and generates publication-grade multi-panel timeline visualizations.

---

## 📂 Comprehensive Notebook Catalog (01–16)

### Part I: Single-Event Flash Shocks (Focused Incident Event Studies)

| Notebook | Domain | Real-World Incident | Target Variable ($y$) | Rare Event Pattern | Academic Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`01_usgs_hurricane_helene_flash_spike.ipynb`** | Environmental IoT Telemetry | **Hurricane Helene** (Late Sept 2024) | `discharge_cfs` (Streamflow) & `gage_height_ft` | **+15,400% Vertical Surge** (727 cfs $\rightarrow$ 112,000 cfs; 1.3 ft $\rightarrow$ 24.8 ft) | *USGS NWIS (2024)* & *Chen et al. (ICLR 2024)* |
| **`02_industrial_paper_mill_break_drop.ipynb`** | Industrial IoT & Predictive Maint. | **Paper Machine Sheet Break** | Binary indicator `y = 1` and sensor `x21` / `x4` | **Instant Drop to 0** (Cliff breakdown occurring in <0.7% of timestamps) | *Ranjan et al. (arXiv:1809.10717)* |
| **`03_crowdstrike_it_outage_market_shock.ipynb`** | Cybersecurity & Market Contagion | **CrowdStrike Global IT Outage** (July 19, 2024) | `CRWD_close` & `VIX_close` | **-28.5% Equity Cliff-Drop** & Volatility Explosion | *Aldasoro et al. (J. Financial Stability 2022)* |
| **`04_wti_crude_negative_price_crash.ipynb`** | Macro Commodities & Storage Pinch | **WTI Crude Negative Price Collapse** (April 20, 2020) | `DCOILWTICO` (Spot Price $/bbl) | **Drop Below Zero** (Plunged from +$18.31 to -$36.98 in 1 day) | *Platt & EIA / FRED (2020)* |
| **`05_fed_emergency_lending_svb_spike.ipynb`** | Banking Contagion & Liquidity | **Silicon Valley Bank (SVB) Collapse** (March 2023) | `WLCFLL` (Fed Loans to Banks, $M) | **+2,228% Vertical Surge** ($15.2B $\rightarrow$ $354.2B in 14 days) | *Jiang, Matvos, Piskorski, & Seru (NBER 2023)* |

---

### Part II: Multi-Spike & Multi-Crash Financial Benchmarks (Long-Horizon Recurring Cycles)

| Notebook | Domain | Time Horizon | Target Variable ($y$) | Recurring Shock / Spike Pattern | Academic Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`06_sp500_multi_decade_market_crashes.ipynb`** | Equity Macro & Flash Crashes | **1985–Present** (40 Years) | `SP500_close`, `return_pct`, `drawdown_pct` | **Multi-Crash Benchmark:** 1987 Black Monday (-20.5%), 2000 Dot-Com (-49%), 2008 Lehman GFC (-56.8%), 2020 COVID Flash Crash (-33.9%), 2022 Inflation Shock, 2024 Yen Unwind | *Ding et al. (KDD '19)* & *Kirilenko et al. (J. Finance 2017)* |
| **`07_cboe_vix_volatility_panic_spikes.ipynb`** | Financial Volatility & Panic Dynamics | **1990–Present** (35 Years) | `VIX_close`, `vix_point_change`, `regime` | **Multi-Spike Jump Process:** The classic "heartbeat" time series with recurrent explosive right-skewed upward spikes from baseline 12–18 to peaks of 40–82+ (1998, 2008, 2011, 2018 Volmageddon, 2020 COVID) | *Whaley (J. Portfolio Mgmt 2000)* & *Paparrizos et al. (PVLDB 2022)* |
| **`08_bitcoin_multicycle_crashes_and_surges.ipynb`** | Crypto & 24/7 Liquidation Shocks | **2015–Present** (10+ Years) | `BTC_close`, `drawdown_pct`, `mayer_multiple` | **Recurrent $\pm50\%$ to $\pm85\%$ Booms & Busts:** 2017 Mania ($19.8k $\rightarrow$ $3.1k), March 2020 Black Thursday (-50% in 24h), May 2021 China Ban (-53%), 2022 Terra/FTX collapse ($15.5k), 2024 ATH | *Liu & Tsyvinski (RFS 2021)* & *Griffin & Shams (J. Finance 2020)* |
| **`09_us_jobless_claims_recession_spikes.ipynb`** | Macro Labor & Systemic Crises | **1967–Present** (55+ Years) | `ICSA` (Weekly Claims), `surge_multiplier` | **Multi-Decade Recession Spikes & COVID Blast:** Stationary 200k–350k baseline punctuated by recession spikes (1974, 1982, 1991, 2001, 2008 at 665k–695k) and the historic **+2,580% vertical blast to 6.87M in March 2020** | *Coibion, Gorodnichenko, & Weber (NBER 2020)* |
| **`10_gamestop_short_squeeze_manias.ipynb`** | Market Microstructure & Retail Frenzy | **2020–Present** (4+ Years) | `GME_close`, `intraday_spread_pct`, `volume_surge_ratio` | **Recurring Squeeze Spikes & Halts:** Jan 2021 MOASS (+2,324%), Feb 2021 secondary surge (+302%), March 2021 flash dump (-50% in 25m), and May/June 2024 "Roaring Kitty" resurgence (+271%) | *U.S. SEC Staff Report (2021)* & *Eaton et al. (JFQA 2022)* |

---

### Part III: Multi-Spike Non-Financial & Physical System Benchmarks

| Notebook | Domain | Time Horizon | Target Variable ($y$) | Recurring Shock / Spike Pattern | Academic Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`11_usgs_decadal_river_flood_spikes.ipynb`** | Decadal Environmental Hydrology | **1990–Present** (35 Years) | `discharge_cfs` (Daily Streamflow), `surge_multiplier` | **Recurrent Major Hurricane Crests:** Stationary ~700 cfs baseline punctuated by vertical flood crests: 1995 Opal (24k cfs), 2004 Ivan (40.5k cfs), 2013 flash floods, 2021 Fred (25k cfs), and 2024 Helene (50k+ cfs) | *Villarini & Smith (Water Resources Res. 2010)* & *Liu & Paparrizos (NeurIPS 2024)* |
| **`12_aws_cloudwatch_server_traffic_spikes.ipynb`** | Cloud Infrastructure & Reliability | **High-Frequency Telemetry** (5-min intervals) | `network_in_mb`, `velocity_kb_step`, `surge_ratio` | **Volumetric Traffic Blasts:** Baseline 50 bytes/step exploding by **+16,000,000%** to 8.28 MB in single steps, triggering queue overflow, latency spikes, and server dropouts | *Ahmad, Lavin, Purdy, & Agha (JMLR 2017)* |
| **`13_noaa_space_weather_solar_flare_spikes.ipynb`** | Space Weather & Solar Physics | **Real-Time Satellite Telemetry** (1-min intervals) | `flux` (0.1–0.8 nm W/m²), `flare_class`, `log_velocity` | **Multi-Order Radiation Explosions:** Solar X-ray flux jumping 2 to 4 orders of magnitude in minutes (B-class background $\rightarrow$ C, M, and catastrophic X-class flares) with exponential cooling tails | *Aschwanden (Astrophysical Journal 2020)* |
| **`14_henry_hub_natural_gas_freeze_spikes.ipynb`** | Energy Commodities & Polar Weather | **1997–Present** (28 Years) | `DHHNGSP` (Spot $/MMBtu), `price_change_1d`, `surge_multiplier` | **Polar Vortex Freezes & Hurricane Shut-ins:** Peaceful $2–$3 baseline exploding into sudden vertical spikes: 2003 Freeze ($18.5), 2005 Katrina ($14.8), 2021 Winter Storm Uri ($23.9), and 2026 Polar Blast ($30.7) | *FERC & NERC Cold Weather Outages Report (2021)* |
| **`15_wikipedia_disaster_attention_spikes.ipynb`** | Collective Human Attention Shocks | **2016–Present** (9 Years) | `views` (Daily Pageviews), `velocity_views_day`, `attention_multiplier` | **100x–250x Breaking Emergency Surges:** "Earthquake" pageviews exploding from 2,500/day baseline to 80,000–200,000+ views within 24h during global seismic disasters (2017 Mexico, 2023 Turkey, 2024 Japan) | *Alanyali, Moat, & Preis (Scientific Reports 2013)* |
| **`16_smd_server_machine_anomalies.ipynb`** | Enterprise Cluster Telemetry (KDD '19) | **5 Weeks Continuous** (1-min intervals) | `label` (Binary Anomaly), 38 continuous telemetry metrics (`dim_0`..`dim_37`) | **High-Dimensional Resource Contention & Breakdown Spikes:** Ground-truth labeled multivariate benchmark from an internet company with sudden CPU, memory, and network queue spikes | *Su, Zhao, Niu, Liu, Sun, & Pei (KDD '19)* |

---

### Part IV: Cross-Sectional & Event-Fragmented Benchmark Ensembles (For Model Evaluation & Panel Training)

| Notebook | Domain | Real-World Incident / System | Target Variable & Ensembles | Rare Event Pattern & Sample Counts ($N$) | Academic Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`17_multi_domain_single_surge_fragment_benchmarks.ipynb`** | Multi-Domain Event Epoching & Panel Benchmarking | **Hurricane Helene, COVID Crash, Paper Machine Breaks, Cloud Contention** | Standardized `[N, T]` matrices ($\tilde{y} = y / y_{\text{base}}$) | **Multi-Domain Single-Surge Ensembles:**<br>• **USGS Hydrology:** $N=332$ stream gages across NC & GA during Helene (15-min intervals, $\tau \in [-12\text{h}, +24\text{h}]$)<br>• **S&P 500 Equities:** $N=480$ stocks during 2020 COVID Crash (Daily closes, $\tau \in [-20\text{d}, +40\text{d}]$)<br>• **Industrial IoT:** $N=53$ paper mill sheet break drops (2-min intervals, $\tau \in [-60\text{m}, +30\text{m}]$)<br>• **Cloud Telemetry (SMD):** $N=76$ resource saturation spikes (1-min intervals, $\tau \in [-30\text{m}, +30\text{m}]$) | *MacKinlay (J. Econ. Lit. 1997)*, *Su et al. (KDD '19)*, *Chen et al. (ICLR '24)*, *Ranjan et al. (2018)* |

---

## 🚀 Running on HiPerGator (UF HPC)

### Option A: Via Open OnDemand (OOD) Web Interface (Recommended)
1. Log into HiPerGator Open OnDemand: [https://ood.rc.ufl.edu](https://ood.rc.ufl.edu)
2. In the top navigation bar, select **Interactive Apps** $\rightarrow$ **JupyterLab** (or **Jupyter Notebook**).
3. Select your SLURM resources:
   * **CPUs:** 2 to 4 cores
   * **Memory:** 8 GB to 16 GB
   * **Time:** 2 hours
   * **Environment:** Default Conda or standard Python module.
4. Launch the session, navigate to this directory (`Spikes data`), and click on any notebook.

### Option B: Via Command Line / SLURM Batch Session
If running in a shell or submitting a SLURM batch job:
```bash
# Load Python environment on HiPerGator
module load python/3.11

# Verify required standard libraries
python -c "import pandas, numpy, matplotlib, seaborn, requests, scipy; print('HiPerGator Python Ready!')"

# Execute any notebook in-place
jupyter nbconvert --to notebook --execute --inplace 16_smd_server_machine_anomalies.ipynb
```

---

## 🛠️ Data Pipeline & Preprocessing Methodology

Across all 16 notebooks, data preprocessing adheres to a standardized time series and event-study protocol:

1. **Self-Healing Local Ingestion:**
   * Every notebook checks if the raw data file already exists in `./data/`. If present, it loads locally (enabling operation behind HPC compute nodes without internet access). If missing, it automatically queries the public REST API and caches the raw data.
2. **Temporal Regularization & Standardization:**
   * High-frequency telemetry (e.g. USGS 15-minute readings, Paper mill 2-minute steps, AWS 5-minute steps, NOAA 1-minute steps, SMD 1-minute steps) is regularized onto uniform grids.
   * Daily and weekly financial, macro, environmental, and attention series are calendar-aligned with missing value handling.
3. **Multi-Shock & Event Detection Algorithms:**
   * Single-event studies mark $T_0$ ($\tau = t - T_0$) to compute centered before-and-after responses.
   * Multi-spike studies implement automated peak finding (`scipy.signal.find_peaks`), rolling Z-score thresholds ($Z > 3\sigma$), logarithmic jump differentiation, drawdown profiles ($DD_t \le -20\%$), and regime clustering.
4. **Impulse & Volatility Feature Engineering:**
   * First difference / Velocity ($\Delta y / \Delta t$) captures sudden acceleration and directionality.
   * Relative surge / drawdown percentage ($\frac{y_t - y_{\text{base}}}{y_{\text{base}}} \times 100$).
   * Rolling volatility and standardized anomaly Z-scores.
   * Volume and severity multipliers ($y_t / \text{Median}_{\text{rolling}}(y)$).
5. **Output Exports:**
   * Each notebook exports its clean, feature-engineered dataset to `./data/{dataset_name}_processed.csv` and high-resolution figures to `./data/{dataset_name}_timeline.png`.
