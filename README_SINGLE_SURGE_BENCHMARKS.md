# 📈 Multi-Domain Event-Aligned Single-Surge Benchmark Datasets
## 100% Single Surge per Trajectory • Zero Overlap • Diverse Natural Baselines • Zero Heuristic Manipulation

This branch (`single-surge-benchmarks`) provides **four curated, unmanipulated benchmark datasets** specifically structured to train, evaluate, and benchmark machine learning, deep learning, and statistical models on **isolated single-surge dynamics**.

---

### Key Properties & Scientific Guarantees

1. **100% Single Surge per Sample:**
   * Sliced relative to each entity's specific surge event $[t_{\text{peak}} - \tau_{\text{pre}}, t_{\text{peak}} + \tau_{\text{post}}]$.
   * In **100.0% of samples (500/500 across every domain)**, the surge apex is anchored strictly at internal time horizon $\tau = 0$.
   * Every single sample strictly rises from baseline ($y_{\text{peak}} > y_{\text{start}}$) and falls back down ($y_{\text{peak}} > y_{\text{end}}$). Zero monotonic drifts, flatlines, or boundary cut-offs.

2. **Zero Overlap Within Datasets (0% Shared Data):**
   * Every single sequence sample corresponds to a **unique, independent physical, financial, or geographic entity** (500 distinct companies, 500 distinct river stations, 500 distinct US counties).
   * Zero sliding window overlaps, zero repeated sequence segments.

3. **Naturally Dispersed, Authentic Baselines (No Artificial Flattening):**
   * Real-world resting levels differ by orders of magnitude across entities.
   * **No artificial normalization** (no dividing by baseline, no min-max squashing to $[0, 1]$). Each curve preserves its native physical and financial units.

4. **Zero Heuristic Preprocessing / Thresholding:**
   * No arbitrary threshold cutoffs, no clipping, and no synthetic smoothing.
   * Pure raw recorded observations directly from official sensor feeds, government repositories, and financial exchanges.

---

### Dataset Summary

| Domain | Entity Type (Zero Overlap) | Samples ($N$) | Horizon ($T$) | Surge Apex Anchor | Single Surge Rate | Natural Baseline Range (Median [Min, Max]) | Natural Peak Range (Median [Min, Max]) | File Path |
|---|---|---|---|---|---|---|---|---|
| **1. Stock Prices** | 500 Distinct US Companies | **500** | 51 trading days | $\tau = 0$ (day 20) | **100.0% (500/500)** | \$75.81 [\$1.00, \$86,400] | \$89.72 [\$1.71, \$97,800] | `data/domain1_equities_event_aligned_500.csv` |
| **2. River Hydrology** | 500 USGS Stream Gages | **500** | 61 (15-min intervals) | $\tau = 0$ (step 24) | **100.0% (500/500)** | 8.64 ft [-0.21 ft, 843.7 ft] | 11.28 ft [1.26 ft, 852.4 ft] | `data/domain2_hydrology_event_aligned_500.csv` |
| **3. County Epidemiology** | 500 Distinct US Counties | **500** | 61 daily counts | $\tau = 0$ (day 25) | **100.0% (500/500)** | 10.0 cases [0.0, 7,567.0] | 93.0 cases [10.0, 29,423.0] | `data/domain3_epidemiology_event_aligned_500.csv` |
| **4. Trading Volume** | 500 Distinct US Companies | **500** | 41 trading days | $\tau = 0$ (day 15) | **100.0% (500/500)** | 2.41M shares [300, 397.8M] | 13.33M shares [71.5K, 1.46B] | `data/domain4_trading_volume_event_aligned_500.csv` |

---

### Visualization

The benchmark visualization is available at `data/four_domains_event_aligned_single_surges.png`:
* **Left Column (Cross-Sectional Ensembles):** Overlays 40 raw curves per domain, showing that baseline levels are naturally dispersed across diverse elevations and scales, with every trajectory converging to its distinct peak at $\tau = 0$.
* **Right Column (Individual Sample Curves):** Three representative individual curves per domain plotted with their raw values, clearly exhibiting the authentic pre-surge resting level, the clean single peak at $\tau = 0$, and the subsequent return to baseline.

---

### Quickstart Loading Snippet (Python / Pandas / PyTorch)

```python
import pandas as pd
import numpy as np

# Load any of the 4 benchmark domains:
df_stocks = pd.read_csv('data/domain1_equities_event_aligned_500.csv', index_col=0)
df_hydro  = pd.read_csv('data/domain2_hydrology_event_aligned_500.csv', index_col=0)
df_covid  = pd.read_csv('data/domain3_epidemiology_event_aligned_500.csv', index_col=0)
df_volume = pd.read_csv('data/domain4_trading_volume_event_aligned_500.csv', index_col=0)

# Matrix shape: [500, T]
print("Stock Prices shape:", df_stocks.shape)   # (500, 51)
print("Hydrology Stage shape:", df_hydro.shape)  # (500, 61)
print("Epidemiology Cases shape:", df_covid.shape) # (500, 61)
print("Trading Volume shape:", df_volume.shape) # (500, 41)

# Convert to NumPy array or PyTorch tensor for model training:
# X_stocks = torch.tensor(df_stocks.values, dtype=torch.float32)
```
