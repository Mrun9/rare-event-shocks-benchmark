# Reworked COVID-19 Weekly Mortality Benchmark

Multi-regional European all-cause mortality benchmark designed for **surge / disruption time-series forecasting** with an exogenous, justifiable $\tau = 0$ onset alignment, normal pre-onset baseline, sustained multi-week surge, strict chronological & spatial train/test separation, raw physical units, and **pure epidemiological/demographic covariates** (zero extraneous weather noise).

![COVID-19 Mortality Benchmark Verification](data/covid_mortality_verification.png)

---

## Notebooks & Datasets

| File | Description | Size / Shape |
| :--- | :--- | :--- |
| [`01_covid_mortality_dataset_and_correlation.ipynb`](01_covid_mortality_dataset_and_correlation.ipynb) | Loads full dataset, engineers `male_death_share`, uses the 16-feature correlation matrix (`\|r\| > 0.90`) to justify pruning 6 redundant covariates, creates the clean CSV, documents `past_only` vs. `past_future` vs. `static` roles, checks train/test split, and plots before/after onset. | 15 cells (executed) |
| [`02_covid_visualize_features_around_onset.ipynb`](02_covid_visualize_features_around_onset.ipynb) | Commented, readable visualization walkthrough built strictly on `covid_mortality_simple.csv` (4-panel overview, 10-feature + 2-static grid around $\tau = 0$, single-region Warsaw trace vs. 2015–2018 baseline, before/after summary table). | 11 cells (executed) |
| `data/covid_mortality_simple.csv` | **Recommended modeling & EDA table:** 495 European NUTS-3 regions $\times$ 41 weeks ($\tau \in [-24, +16]$), zero redundant features ($\max \|r\| = 0.818$, max $\text{VIF} = 3.85$), 0 NaNs. | `20,295 rows x 24 cols` (`3.11 MB`) |
| `data/reworked_covid_mortality_weekly.npz` / `.pt` | Pre-split multivariate tensors containing all 15 full candidate dynamic channels (`past_covariates`, `past_future_covariates`), `static_features`, `tau_steps`, `train_mask`, and `test_mask`. | `(495, 41, F)` |
| `data/reworked_covid_mortality_metadata.csv` | One row per region window with NUTS-3 ID, country, `onset_week`, chronological bounds, `hist_mean_weekly_deaths`, baseline median deaths, and peak ratios. | `495 rows x 21 cols` |

---

## Data Sources (Public Archives)

1. **Eurostat (`demo_r_mwk3_ts`)** — Regional NUTS-3 weekly all-cause deaths (`2019–2021` Total, Male, Female) plus pre-COVID **`2015–2018` historical weekly baselines**.
2. **Eurostat GISCO (`NUTS 2021`)** — NUTS-3 region centroid coordinates (`latitude`, `longitude`) and official `urban_type`, `coast_type`, `mount_type` codes.
3. **Oxford COVID-19 Government Response Tracker (`OxCGRT`)** — Daily national confirmed COVID-19 cases, confirmed COVID-19 deaths, and containment policy indices aggregated by ISO week.
4. **Our World in Data (`OWID`)** — National population counts used to normalize confirmed cases and deaths per 100,000 inhabitants.

---

## Feature Roles (Simple Clean CSV: 10 Dynamic + 6 Static + 8 Metadata)

Correlation pruning on the 16 candidate dynamic features removed **6 redundant variables** where $|r| > 0.90$ against a primary covariate: `deaths_male` ($r = +0.994$), `deaths_female` ($r = +0.994$), `deaths_rolling_mean_4w` ($r = +0.942$), `containment_index` ($r = +0.982$), `gov_response_index` ($r = +0.977$), and `month` ($r = +0.922$). Every remaining dynamic feature has $\text{VIF} \le 3.85$.

### Dynamic `past_only` Covariates (6 channels — observed up to forecast origin $\tau = 0$)
| Column | Meaning & Unit |
| :--- | :--- |
| `deaths_total` | **[TARGET]** Weekly all-cause regional deaths (Eurostat, raw count/week) |
| `deaths_change_1w` | 1-week backward mortality change (`deaths[t] - deaths[t-1]`, count/week) |
| `male_death_share` | Male fraction of weekly deaths (`deaths_male / deaths_total`, `0–1`) |
| `covid_cases_100k` | National weekly confirmed COVID-19 cases per 100k population |
| `covid_deaths_100k` | National weekly confirmed COVID-19 deaths per 100k population |
| `stringency_index` | OxCGRT Stringency Index (`0–100`) |

### Dynamic `past_future` Covariates (4 channels — known epidemiological baseline & seasonal clock)
| Column | Meaning & Unit |
| :--- | :--- |
| `hist_expected_deaths` | Pre-COVID (`2015–2018`) mean weekly deaths for the same `(region_id, week_of_year)` (count/week) |
| `hist_seasonal_index` | Dimensionless seasonal baseline index (`hist_expected_deaths / hist_mean_weekly_deaths`) |
| `daylight_hours` | Astronomical solar photoperiod at region latitude (hours/day) |
| `week_of_year` | ISO calendar week of year (`1–53`) |

### Static & Metadata Columns
- **Static (`6`):** `latitude`, `longitude`, `urban_type` (`1`=Urban, `2`=Intermediate, `3`=Rural), `coast_type` (`1`=Coastal, `2`=Semi-coastal, `3`=Inland), `mount_type` (`1..3`=Mountainous, `4`=Non-mountainous), `hist_mean_weekly_deaths` (`2015–2018` overall regional weekly death mean).
- **Metadata (`8`):** `sample_id`, `region_id`, `region_name`, `country_code`, `split`, `onset_week`, `iso_week`, `tau_week` (`-24` to `+16`).

---

## Chronological & Disjoint Split Summary

| Split | Regions ($N$) | Countries | Window Range | Onset Range ($\tau = 0$) | Median Baseline Deaths/wk | Median Peak Deaths/wk | Median Peak Ratio |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **Train** (Spring 2020) | `305` | `9` (`UK, IT, FR, ES, NL, BE, SE, CH, PT`) | `2019-W38` .. `2020-W29` | `2020-W10` .. `2020-W13` | `77.50` | `167.0` | `1.911x` |
| **Test** (Fall 2020) | `190` | `12` (`PL, AT, HU, RS, EL, CZ, SK, LT, LV, DK, EE, CY`) | `2020-W14` .. `2021-W10` | `2020-W38` .. `2020-W47` | `85.25` | `178.0` | `2.125x` |

- **Zero entity overlap** and a **$\ge 9$-week chronological separation** (`max train week 2020-W29 < min test onset 2020-W38`).

---

## Quickstart

```python
import pandas as pd

df = pd.read_csv("data/covid_mortality_simple.csv")
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]
```
