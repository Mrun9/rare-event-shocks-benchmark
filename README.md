# Daily Power Grid Demand Single-Surge Onset Benchmark (`reworked-power-grid-onset`)

Daily US Balancing Authority (EIA-930) electricity demand surges paired with Open-Meteo daily weather telemetry around heatwave and cold-snap onset days (`N = 286` single-surge events, `T = 29` days per window, `tau_day in [-14, +14]`).

---

## Dataset Overview

- **Events (`N`):** `286` daily single-surge grid load windows (`8,294` total rows; zero `NaN` values)
- **Window Resolution (`T`):** `29` daily steps (`tau_day = -14` to `+14`, relative to surge onset day `tau_day = 0`), eliminating 24-hour diurnal sawtooth oscillations
- **Chronological Split (Zero Temporal Leakage):** `max train end = 2023-12-12 < min test start = 2023-12-19`
  - **Train (`2021–2023`):** `219` events across `48` Balancing Authorities (`80` heatwaves, `139` freezes) — median baseline `2,660.8 MW`, median peak `3,435.0 MW` (`1.245x`)
  - **Test (`2024`):** `67` events across `39` Balancing Authorities (`31` heatwaves, `36` freezes) — median baseline `3,238.6 MW`, median peak `4,642.0 MW` (`1.300x`)

---

## Feature Hygiene & Correlation Pruning

To keep physical signal high without cherry-picking events or altering raw demand values:

1. **Feature Hygiene Cleanup:** Dropped flat, near-zero, or terrain-noise covariates (`solar_share`, `wind_share`, `wind_ms`, `pressure_hpa`) while retaining all `286` surge events.
2. **Collinearity Pruning (`|r| > 0.90`):** From `15` candidate dynamic covariates, `6` redundant features were dropped (`demand_forecast_mw`, `net_gen_mw`, `load_rolling_mean_7d`, `gas_gen_mw`, `coal_gen_mw`, `apparent_temp_c`).
3. **Final Retained Schema (`20` columns in `data/power_grid_simple.csv`):**
   - **4 Past-Only Dynamic Features:** `load_mw` (target), `load_change_1d`, `gas_share`, `import_share`
   - **5 Past-Future Known Dynamic Features:** `temp_c`, `thermal_discomfort_c`, `humidity_pct`, `solar_wm2`, `day_of_week`
   - **3 Static Balancing Authority Features:** `latitude`, `longitude`, `installed_cap_mw`
   - **8 Event Metadata Columns:** Event identifiers, Balancing Authority code, weather regime (`heatwave`/`freeze`), timestamps, and split labels
   - **Multicollinearity Diagnostics:** Max off-diagonal `|r| = 0.780`, Max `VIF = 4.68` (`temp_c`)

### Pre-Onset (`tau_day < 0`) vs Post-Onset (`tau_day >= 0`) Medians

| Feature | Role | Pre-Onset Median | Post-Onset Median | Notes |
| :--- | :--- | ---: | ---: | :--- |
| `load_mw` | `PAST-ONLY (TARGET)` | `2797.81` | `3080.21` | Peak `3070.54 -> 3562.96` (`+282.40`) |
| `load_change_1d` | `PAST-ONLY` | `4.25` | `-2.75` | Peak `276.38 -> 398.08` |
| `gas_share` | `PAST-ONLY` | `0.30` | `0.32` | Shift `+0.02` |
| `import_share` | `PAST-ONLY` | `0.05` | `0.06` | Shift `+0.02` |
| `temp_c` | `PAST-FUTURE` | `10.96` | `6.99` | Shift `-3.97` |
| `thermal_discomfort_c` | `PAST-FUTURE` | `10.18` | `13.68` | Peak `13.82 -> 18.70` (`+3.51`) |
| `humidity_pct` | `PAST-FUTURE` | `76.42` | `72.91` | Shift `-3.51` |
| `solar_wm2` | `PAST-FUTURE` | `134.85` | `144.77` | Shift `+9.92` |
| `day_of_week` | `PAST-FUTURE` | `3.00` | `3.00` | Calendar weekday |

---

## Repository Contents

- `01_power_grid_dataset_and_correlation.ipynb` — Loads full arrays, engineers `thermal_discomfort_c`, prunes `|r| > 0.90` pairs, verifies VIF and zero chronological overlap, and exports `power_grid_simple.csv` plus compressed `.npz` / `.pt` tensors.
- `02_power_grid_visualize_features_around_onset.ipynb` — Reads `data/power_grid_simple.csv` to plot the `3 x 4` aligned onset grid (`train` vs `test`), an individual `ERCO 2024-01-14` freeze event trace, and prints pre/post-onset summary statistics.
- `data/power_grid_simple.csv` (`1.24 MB`, `8,294 x 20`) — Clean tabular CSV of all `286` events.
- `data/reworked_power_grid_daily.npz` (`372.8 KB`) & `data/reworked_power_grid_daily.pt` (`507.9 KB`) — Ready-to-train tensor bundles.
- `data/power_grid_correlation_matrix.png` — Correlation matrix before/after pruning.
- `data/power_grid_all_features_around_surge.png` — `3 x 4` median + IQR panel grid across all `9` retained features plus metadata summaries.
- `data/power_grid_single_region_features.png` — Single-region `ERCO` (`2024-01-14`) freeze event trace.
