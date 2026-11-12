# ⚡ Reworked Power Grid Demand Surge Benchmark (Hourly, Weather-Onset Aligned)

Multi-region U.S. electricity demand benchmark built for **surge / disruption time-series forecasting** around extreme thermal weather onsets ($\tau = 0$: heatwaves and Arctic freezes), with normal diurnal pre-onset baselines, sustained multi-day load surges, strict chronological separation (`2021–2023` Train vs. `2024` Test), raw physical MW units, and explicit `past_only` vs. `past_future` vs. `static` covariate roles.

![Power Grid Benchmark Verification](data/power_grid_verification.png)

---

## Notebooks & Datasets

| File | Description | Size / Shape |
| :--- | :--- | :--- |
| [`01_power_grid_dataset_and_correlation.ipynb`](01_power_grid_dataset_and_correlation.ipynb) | Loads full dataset, engineers `thermal_discomfort_c = \|temp_c - 18.0\|`, uses the 20-feature correlation matrix (`\|r\| > 0.90`) to justify pruning 6 redundant features, saves `power_grid_simple.csv`, documents all covariate roles, verifies chronological split, and plots before/after load surge. | 15 cells (executed) |
| [`02_power_grid_visualize_features_around_onset.ipynb`](02_power_grid_visualize_features_around_onset.ipynb) | Commented, readable visualization walkthrough built strictly on `power_grid_simple.csv` (4-panel overview, 14-feature grid around $\tau = 0$, heatwave vs. freeze split, PJM June 2024 heatwave trace, and before/after summary table). | 11 cells (executed) |
| `data/power_grid_simple.csv` | **Recommended modeling & EDA table:** 569 extreme weather events across 50 US Balancing Authorities $\times$ 145 hours ($\tau \in [-72, +72]$), zero redundant features ($\max \|r\| = 0.612$, max $\text{VIF} = 3.35$), 0 NaNs. | `82,505 rows x 25 cols` (`14.59 MB`) |
| `data/reworked_power_grid_hourly.npz` / `.pt` | Pre-split multivariate tensors containing all 19 full candidate dynamic channels (`past_covariates`, `past_future_covariates`), `static_features`, `tau_steps`, `train_mask`, and `test_mask`. | `(569, 145, F)` |
| `data/reworked_power_grid_metadata.csv` | One row per event window with Balancing Authority, interconnection region, `event_type` (`heatwave`/`freeze`), `onset_utc`, baseline MW, peak MW, and peak surge ratio. | `569 rows x 16 cols` |

---

## Data Sources (Public Archives)

1. **U.S. EIA-930 Electric Grid Monitor (`EIA930_BALANCE` 2021–2024)** — Hourly system demand, demand forecast, net generation, total interchange, and generation by fuel source (natural gas, coal, nuclear, solar, wind, hydro).
2. **NASA POWER (`MERRA-2` Reanalysis)** — Hourly 2m air temperature, relative humidity, downward shortwave solar irradiance, 10m wind speed, and surface pressure at each Balancing Authority's centroid.
3. **Electricity Maps / EIA Capacity Metadata** — Balancing Authority centroids (`latitude`, `longitude`) and installed nameplate generation capacity (`installed_cap_mw`).

---

## Feature Roles (Simple Clean CSV: 14 Dynamic + 3 Static + 8 Metadata)

Correlation pruning on the 20 candidate dynamic features removed **6 redundant scale/temperature duplicates** where $|r| > 0.90$: `demand_forecast_mw` ($r = +0.998$), `net_gen_mw` ($r = +0.996$), `load_rolling_mean_24h` ($r = +0.990$), `gas_gen_mw` ($r = +0.956$), `coal_gen_mw` ($r = +0.905$), and `apparent_temp_c` ($r = +0.988$ vs. `temp_c`). Scale-invariant fuel shares (`gas_share`, `solar_share`, `wind_share`, `import_share`) replace raw MW counts cleanly ($\max |r| = 0.612$, $\max \text{VIF} = 3.35$).

### Dynamic `past_only` Covariates (6 channels — observed up to $\tau = 0$)
| Column | Meaning & Unit |
| :--- | :--- |
| `load_mw` | **[TARGET]** Hourly system electricity demand (`Demand (MW)`, EIA-930) |
| `load_change_1h` | 1-hour backward load ramp (`load[t] - load[t-1]`, MW/hr) |
| `gas_share` | Natural gas fraction of net generation (`0–1`) |
| `solar_share` | Utility solar PV fraction of net generation (`0–1`) |
| `wind_share` | Wind fraction of net generation (`0–1`) |
| `import_share` | Net electricity import ratio (`-interchange / load`) |

### Dynamic `past_future` Covariates (8 channels — weather forecasts & local schedules)
| Column | Meaning & Unit |
| :--- | :--- |
| `temp_c` | 2m air temperature (°C, NASA POWER MERRA-2) |
| `thermal_discomfort_c` | Absolute departure from 18 °C comfort baseline (`\|temp_c - 18.0\|` °C) |
| `humidity_pct` | 2m relative humidity (%) |
| `solar_wm2` | Shortwave solar irradiance (W/m²) |
| `wind_ms` | 10m wind speed (m/s) |
| `pressure_hpa` | Surface barometric pressure (hPa) |
| `local_hour` | Local hour of day (`0–23`) |
| `day_of_week` | Local day of week (`0=Mon .. 6=Sun`) |

### Static & Metadata Columns
- **Static (`3`):** `latitude`, `longitude`, `installed_cap_mw`.
- **Metadata (`8`):** `sample_id`, `ba`, `region`, `split`, `event_type`, `onset_utc`, `timestamp_utc`, `tau_hour` (`-72` to `+72`).

---

## Chronological Split Summary

| Split | Events ($N$) | Unique BAs | Heatwaves | Freezes | Earliest Window | Latest Window | Median Pre-Peak (MW) | Median Post-Peak (MW) | Median Peak Ratio |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **Train** (`2021–2023`) | `439` | `48` | `214` | `225` | `2021-01-02 09:00` | `2024-01-02 06:00` | `3,495.0` | `4,270.0` | `1.208x` |
| **Test** (`2024`) | `130` | `43` | `73` | `57` | `2024-01-07 09:00` | `2024-12-24 12:00` | `3,517.5` | `4,342.5` | `1.211x` |

- Zero window overlap (minimum `14-day` spacing within any BA) and strict chronological separation (`max train window end 2024-01-02 06:00 < min test window start 2024-01-07 09:00`).

---

## Quickstart

```python
import pandas as pd

df = pd.read_csv("data/power_grid_simple.csv")
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]
```
