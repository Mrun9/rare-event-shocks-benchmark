# 🌫️ Reworked Urban Wildfire Smoke `PM2.5` Benchmark (Hourly, Plume-Onset Aligned)

Multi-station U.S. wildfire smoke air-quality benchmark built for **surge / disruption time-series forecasting** around wildfire plume arrival ($\tau = 0$), with clean 72-hour pre-smoke baselines, sustained multi-day plumes (excluding Independence Day fireworks spikes), **100% unique physical EPA stations**, strict chronological separation (`2022–2023` Train vs. `2024` Test), raw physical concentrations ($\mu\text{g/m}^3$), and explicit `past_only` vs. `past_future` vs. `static` dispersion roles.

![Wildfire Smoke Benchmark Verification](data/wildfire_smoke_verification.png)

---

## Notebooks & Datasets

| File | Description | Size / Shape |
| :--- | :--- | :--- |
| [`01_wildfire_smoke_dataset_and_correlation.ipynb`](01_wildfire_smoke_dataset_and_correlation.ipynb) | Loads full dataset (`13` raw dynamic + `4` static channels), engineers `smoke_trapping_index = aerosol_optical_depth / (PBLH_km + 0.1)`, uses the 14-feature correlation matrix (`\|r\| > 0.90`) to prune `2` redundant variables, saves `wildfire_smoke_simple.csv`, documents covariate roles, checks `0%` station overlap, and plots East Syracuse (`NY`) + all `439` stations. | 15 cells (executed) |
| [`02_wildfire_smoke_visualize_features_around_onset.ipynb`](02_wildfire_smoke_visualize_features_around_onset.ipynb) | Commented, readable visualization walkthrough built strictly on `wildfire_smoke_simple.csv` (4-panel overview, 4x4 grid of all `12` retained dynamic features + `4` static/event panels around $\tau = 0$, Banning `CA` 2024 trace, before/after summary table). | 11 cells (executed) |
| `data/wildfire_smoke_simple.csv` | **Recommended modeling & EDA table:** `439` unique US EPA monitors $\times$ `145` hours ($\tau \in [-72, +72]$), zero redundant features ($\max \|r\| = 0.760$, max $\text{VIF} = 3.43$), 0 NaNs. | `63,655 rows x 24 cols` (`11.46 MB`) |
| `data/reworked_wildfire_smoke_hourly.npz` / `.pt` | Pre-split multivariate tensors containing all `13` candidate dynamic channels (`5 past_covariates`, `8 past_future_covariates`), `4 static_features`, `tau_steps`, `train_mask`, and `test_mask`. | `1.46 MB` (`.npz`) / `3.27 MB` (`.pt`) |
| `data/reworked_wildfire_smoke_metadata.csv` | One row per station plume event with EPA `site_id`, city, state, `onset_utc`, pre-smoke median, post-onset peak `PM2.5`, peak `AOD`, elevation, and urban/rural setting. | `439` events |

---

## Data Sources (Public Archives)

1. **U.S. EPA Air Quality System (`AQS hourly_88101` 2022–2024 & `aqs_sites`)** — Hourly ground-truth `PM2.5` monitor readings ($\mu\text{g/m}^3$) and station elevation / land-setting codes.
2. **Open-Meteo ERA5 Reanalysis Archive** — Hourly planetary boundary layer height (`boundary_layer_height_m`), 10m wind speed & direction vectors (`wind_ms`, `wind_dir_sin`, `wind_dir_cos`), and 2m temperature (`temp_c`).
3. **Open-Meteo CAMS Air Quality Archive** — Hourly columnar smoke aerosol optical depth (`aerosol_optical_depth`) and combustion co-pollutant tracers (`pm10_cams_ug_m3`, `co_ug_m3`).

---

## Feature Hygiene & Correlation Pruning (`12` Dynamic + `4` Static + `8` Metadata)

- **Non-Manipulative Feature Hygiene:** Dropped flat-zero, diurnal-noise, and East-vs-West geographic-leak covariates (`dust_ug_m3`, `rain_mm_hr`, `pressure_hpa`, `humidity_pct`, `no2_ug_m3`, `ozone_ug_m3`) while keeping all `439` unique smoke events and raw ground `PM2.5` observations intact (`0` events removed, `0` NaNs).
- **Collinearity Pruning (`|r| > 0.90`):** Across the `14` candidate dynamic features (`5 past + 8 past_future + 1 engineered`), `2` redundant features were dropped: `pm25_rolling_mean_6h` ($r = +0.921$ vs. `pm25_ug_m3`) and `apparent_temp_c` ($r = +0.947$ vs. `temp_c`). All `12` retained dynamic features maintain $\max |r| = 0.760$ and $\max \text{VIF} = 3.43$ (`aerosol_optical_depth`).

### Dynamic `past_only` Covariates (`4` channels — observed up to $\tau = 0$)
| Column | Meaning & Unit | Pre-Onset Median | Post-Onset Median | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `pm25_ug_m3` | **[TARGET]** Hourly EPA ground `PM2.5` ($\mu\text{g/m}^3$) | `7.35` | `27.60` | Peak `22.00 -> 89.80` (`+20.25`) |
| `pm25_change_1h` | 1-hour concentration rate of change ($\mu\text{g/m}^3/\text{hr}$) | `0.00` | `0.00` | Peak `8.10 -> 31.50` |
| `pm10_cams_ug_m3` | Coarse particulate matter `PM10` ($\mu\text{g/m}^3$, CAMS) | `14.80` | `29.90` | Peak `32.00 -> 75.70` (`+15.10`) |
| `co_ug_m3` | Carbon monoxide biomass combustion tracer ($\mu\text{g/m}^3$, CAMS) | `191.00` | `263.00` | Peak `294.00 -> 470.00` (`+72.00`) |

### Dynamic `past_future` Covariates (`8` channels — dispersion & smoke transport forecasts)
| Column | Meaning & Unit | Pre-Onset Median | Post-Onset Median | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `aerosol_optical_depth` | Columnar overhead smoke Aerosol Optical Depth forecast (`AOD`, CAMS) | `0.22` | `0.42` | Peak `0.58 -> 1.16` (`+0.20`) |
| `boundary_layer_height_m` | Planetary Boundary Layer Height (`m`, ERA5) — vertical mixing / trapping | `395.00` | `395.00` | Diurnal inversion driver |
| `smoke_trapping_index` | Engineered trapping proxy `aerosol_optical_depth / (PBLH_km + 0.1)` | `0.38` | `0.76` | Peak `2.64 -> 5.45` (`+0.37`) |
| `wind_ms` | 10m wind speed (m/s, ERA5) | `2.56` | `2.52` | Peak `5.52 -> 5.46` |
| `wind_dir_sin` | Sine of 10m wind direction (continuous E–W transport vector) | `-0.26` | `-0.33` | Shift `-0.07` |
| `wind_dir_cos` | Cosine of 10m wind direction (continuous N–S transport vector) | `0.00` | `0.10` | Shift `+0.10` |
| `temp_c` | 2m air temperature (°C, ERA5) | `21.70` | `21.10` | Peak `29.90 -> 29.60` |
| `local_hour` | Local hour of day (`0–23`, diurnal boundary-layer inversion cycle) | `11.50` | `12.00` | Diurnal phase |

### Static & Metadata Columns
- **Static (`4`):** `latitude`, `longitude`, `elevation_m`, `setting_code` (`1`=Urban Center, `2`=Suburban, `3`=Rural).
- **Metadata (`8`):** `sample_id`, `site_id`, `city`, `state`, `split`, `onset_utc`, `timestamp_utc`, `tau_hour` (`-72` to `+72`).

---

## Chronological & Zero-Overlap Split Summary

| Split | Unique EPA Stations ($N$) | States | Earliest Window | Latest Window | Median Baseline (`ug/m^3`) | Median Peak (`ug/m^3`) | Median Peak Hour ($\tau$) |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **Train** (`2022–2023`) | `358` | `44` | `2022-08-10 05:00` | `2023-10-27 23:00` | `7.23` | `89.15` | `+24.0h` |
| **Test** (`2024`) | `81` | `20` | `2024-06-23 01:00` | `2024-10-28 07:00` | `7.10` | `104.55` | `+23.0h` |

- **Zero shared sensors:** `0` station IDs shared between `train` and `test` (`439/439` distinct monitors across `49` US states).
- **Strict chronological gap:** `> 7` months (`max train end 2023-10-27 23:00:00 < min test start 2024-06-23 01:00:00`).

---

## Quickstart

```python
import pandas as pd

df = pd.read_csv("data/wildfire_smoke_simple.csv")
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]
```
