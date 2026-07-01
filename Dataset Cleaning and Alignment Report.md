# Dataset Cleaning and Alignment Report

**Project:** Stock Performance Prediction (RNN)  
**Notebook:** `notebooks/01_exploratory_data_analysis.ipynb`  
**Last updated:** 2026-06-10  
**Status:** Living document — updated as EDA and cleaning progress

---

## 1. Purpose

This report separates two kinds of problems encountered so far:

| Category | Description |
|----------|-------------|
| **Dataset issues** | Problems inherent in the raw files under `Stock Market Dataset/` |
| **EDA / alignment challenges** | Problems discovered while exploring, plotting, or combining symbols — including bugs in analysis code |

The goal is to document what must be cleaned or aligned **before** building the RNN, and what we learned from the exploration workflow itself.

---

## 2. Dataset Overview

| Component | Count | Notes |
|-----------|------:|-------|
| Metadata symbols (`symbols_valid_meta.csv`) | 8,049 | Exchange, ETF flag, financial status |
| Stock CSV files (`stocks/`) | 5,884 | One file per symbol |
| ETF CSV files (`etfs/`) | 2,165 | One file per symbol |
| Columns per OHLCV file | 7 | `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` |

**Typical date range:** Varies by symbol (~1962–2020). The vast majority of files end on **2020-04-01**.

---

## 3. Dataset Issues (Raw Data)

These are properties of the dataset itself, independent of notebook code.

### 3.1 Uneven trading history across symbols

Each symbol has a different number of trading days because of IPO date, delisting, mergers, and data availability.

**Benchmark example (verified):**

| Symbol | Trading days | Approx. span |
|--------|-------------:|--------------|
| XOM | 14,663 | 1962–2020 |
| JPM | 10,098 | 1980–2020 |
| AAPL | 9,909 | 1980–2020 |
| MSFT | 8,584 | 1986–2020 |
| TSLA | 2,457 | 2010–2020 |

**Cross-sectional sample (500 random stocks):**

- Median history: ~3,461 trading days
- Range: 1 to 12,676 days
- ~10% of sampled stocks have fewer than 2 years (~504 trading days) of data

**Impact on modeling:** Sequence models need a minimum lookback window (e.g. 60–120 days). Symbols with very short history should be filtered out or handled separately.

**Recommended action:** Enforce a minimum history threshold (e.g. ≥ 504 or ≥ 756 trading days) when building the training universe.

---

### 3.2 No shared calendar index across symbols

Files are stored per symbol with their own `Date` column. There is **no panel-wide date index** and no guarantee that row *i* in one file corresponds to the same calendar day in another.

**Impact on modeling:**

- Cannot stack `.values` from multiple symbols into one DataFrame without alignment
- Multi-symbol features (correlation, market beta) require an explicit **merge on `Date`**
- Train/validation splits must be **time-based**, not random row splits

**Recommended action:** Use `pd.merge(..., on="Date", how="inner")` or reindex to a business-day calendar when combining symbols.

---

### 3.3 Metadata ↔ file inventory mismatches

| Check | Result |
|-------|--------|
| Symbols in metadata without a CSV file | 3 |
| CSV files not listed in metadata | 3 |

**Impact:** Low for bulk analysis, but symbol lookups can fail for those edge cases.

**Recommended action:** Log and exclude orphan/missing symbols; optionally reconcile the 6 mismatched entries manually.

---

### 3.4 Missing values in OHLCV fields

From a random sample of 500 stock files:

| Issue | % of files affected |
|-------|--------------------:|
| Any null in OHLCV columns | ~5% |
| Zero-volume trading days | ~58% |
| Duplicate dates | ~0% |

Benchmark symbols (AAPL, MSFT, JPM, XOM, TSLA) have **no nulls, no zero-volume days, and no duplicate dates** — but the broader universe is noisier.

**Impact:**

- Nulls break return calculations and RNN sequences
- Zero-volume days may indicate illiquid stocks, halts, or data artifacts

**Recommended action:**

- Drop or forward-fill rows with null OHLCV (document choice)
- Filter illiquid symbols (e.g. exclude if zero-volume days exceed a threshold)
- Validate `Date` is unique per symbol after load

---

### 3.5 Incomplete / unknown financial status in metadata

`Financial Status` in metadata:

| Status | Count (approx.) |
|--------|----------------:|
| Unknown (NaN) | 4,666 |
| Normal | 3,180 |
| Deficient | 191 |
| Delinquent | 8 |
| Halted | 4 |

**Impact:** Including deficient, delinquent, or halted symbols may add noise or survivorship bias.

**Recommended action:** For a first model, consider restricting to `Financial Status == Normal` or excluding flagged symbols.

---

### 3.6 Mixed asset types (stocks vs ETFs)

- ~5,884 stocks and ~2,165 ETFs in separate folders
- ETFs behave differently (tracking error, lower idiosyncratic vol, different liquidity)

**Impact:** Mixing stocks and ETFs in one model without an asset-type feature may blur patterns.

**Recommended action:** Decide early whether the RNN targets stocks only, ETFs only, or both with an explicit type indicator.

---

### 3.7 Stale end date / regime cutoff

~99% of sampled files end on **2020-04-01** (early COVID period). A few end earlier or on 2020-04-02.

**Impact:**

- No data beyond April 2020 for out-of-sample tests on recent markets
- Feb–Apr 2020 is an extreme volatility regime (useful stress test, risky as training-only period)

**Recommended action:** Hold out 2020 Q1 as a stress-test window; do not shuffle time when splitting.

---

### 3.8 Non-stationary prices; stationary returns

Raw `Close` / `Adj Close` are non-stationary (trending, scale differs across symbols and time).

Log returns:

- Fat-tailed (high excess kurtosis on AAPL)
- Slight negative skew
- Volatility clustering (autocorrelation in squared returns)

**Impact:** RNN inputs/targets should use **returns or normalized features**, not raw price levels.

**Recommended action:** Model on `log_return` or `pct_change` of `Adj Close`; normalize features per symbol or with rolling z-scores.

---

### 3.9 First-row NaN in return series

`log_return = log(price / price.shift(1))` produces one NaN per symbol on the first trading day. This is expected, not a data defect.

**Recommended action:** Drop the first row after computing returns, or mask it in sequence construction.

---

### 3.10 Trading calendar gaps

Consecutive observations are usually 1 calendar day apart (Mon–Fri), with 3-day gaps over weekends and longer gaps for holidays or halts.

**Impact:** Fixed-length sequence windows by **row count** do not always equal fixed **calendar time**. For an RNN using last 60 rows, the calendar span can vary slightly.

**Recommended action:** Document whether sequences are **trading-day-based** (row count) or **calendar-based** (reindexed business days).

---

## 4. EDA & Analysis Challenges (Exploration Process)

These are issues encountered while running `01_exploratory_data_analysis.ipynb`, distinct from raw data defects.

### 4.1 ✅ RESOLVED — Benchmark return boxplot: unequal array lengths (Cell 19)

**Error:**

```
ValueError: All arrays must be of the same length
```

**Code that failed:**

```python
returns_wide = pd.DataFrame({
    sym: add_returns(df)["log_return"].values
    for sym, df in benchmark_data.items()
})
```

**Root cause:**

- `pd.DataFrame({symbol: array})` requires all arrays to have **equal length**
- Benchmark symbols have **different history lengths** (2,457 to 14,663 rows)
- The failure occurs on `returns_wide` **before** the intended alignment step runs
- The follow-up logic (`dropna()` + `.tail(min_len)`) would also be flawed: it aligns by **row index**, not **calendar date**, so day 500 for AAPL and day 500 for TSLA are unrelated dates

**Category:** EDA code bug + dataset alignment issue (Section 3.1, 3.2)

**Planned fix:**

- For distribution comparison only: take `tail(min_length)` from each symbol’s return series, or
- For date-aligned comparison: merge on `Date` over a shared window (e.g. 2015–2020)

**Status:** Fixed — uses `min(len)` across dropna'd return series, then `.tail(min_len)` per symbol

---

### 4.2 Project path resolution across environments

**Issue:** `Path.cwd()` differs depending on whether Jupyter is started from project root, `notebooks/`, or a remote workbench path.

**Resolution applied:** `find_project_root()` walks up the directory tree looking for `Stock Market Dataset/`.

**Remaining risk:** On remote server, dataset must be uploaded to the expected project folder or `DATA_DIR` will not be found.

---

### 4.3 Python environment / missing packages (local Mac)

**Issue:** Notebook initially used a kernel from another course (`Programing for Data Analytics/.venv`) that lacked `seaborn`, `scipy`, and even `pip`.

**Resolution applied:**

- Project-local `.venv` with `requirements.txt`
- Registered kernel `Python (stock-performance-prediction)`
- Bootstrap cell to install missing packages if needed

**Category:** Environment setup — not a dataset issue

---

### 4.4 Remote JupyterLab upload / File ID error

**Issue:** Uploading the notebook to a university JupyterHub failed with “File ID could not be retrieved.”

**Category:** Infrastructure — not a dataset issue

**Workaround:** Use RTX 5090 workbench via SSH tunnel (`remote-jupyter-rtx5090.md`) or run locally in Cursor/VS Code.

---

### 4.5 Cross-sectional scan runtime

**Issue:** Scanning all ~8,000 CSV files for quality metrics is slow if done in-notebook without sampling.

**Resolution in notebook:** `scan_directory(..., sample_size=500)` for stocks and 300 for ETFs.

**Trade-off:** Sample statistics are approximate; full scan may be needed before final model training.

---

### 4.6 Feature correlation heatmap on engineered features

Rolling features (MA, volatility, RSI) are **highly correlated** by construction. This is expected but can mislead if interpreted as independent signals.

**Category:** Analysis interpretation — informs feature selection for RNN, not a cleaning defect

---

## 5. Summary Matrix

| ID | Issue | Source | Severity | Status |
|----|-------|--------|----------|--------|
| D1 | Uneven history length per symbol | Dataset | High | Open — filter policy needed |
| D2 | No shared date index across symbols | Dataset | High | Open — merge on Date required |
| D3 | Metadata/file mismatches (6 symbols) | Dataset | Low | Open |
| D4 | Nulls in ~5% of files | Dataset | Medium | Open |
| D5 | Zero-volume days in ~58% of files | Dataset | Medium | Open |
| D6 | Unknown/deficient financial status | Metadata | Medium | Open |
| D7 | Stocks vs ETFs mixed universe | Dataset | Medium | Design decision |
| D8 | Data ends ~2020-04-01 | Dataset | Medium | Accepted |
| D9 | Non-stationary prices | Dataset | High | Use returns (documented) |
| D10 | First-row return NaN | Expected | Low | Drop first row |
| D11 | Calendar vs trading-day gaps | Dataset | Low | Document sequence definition |
| E1 | Cell 19 unequal array lengths | EDA code | High | Resolved |
| E2 | Path resolution local vs remote | EDA setup | Medium | Mitigated |
| E3 | Wrong Python kernel / missing libs | Environment | Medium | Mitigated |
| E4 | JupyterHub File ID upload error | Infrastructure | Low | Workaround documented |
| E5 | Full-universe scan performance | EDA workflow | Low | Sampled in notebook |

---

## 6. Recommended Cleaning Pipeline (Pre-Modeling)

Proposed order of operations before RNN training:

1. **Define universe** — stocks only vs ETFs; minimum history; optional financial-status filter  
2. **Load & validate** — parse `Date`, sort ascending, check duplicate dates, drop null OHLCV  
3. **Filter liquidity** — optional zero-volume threshold  
4. **Compute features** — log returns from `Adj Close`, volume transforms, rolling stats  
5. **Align panel data** — merge on `Date` when combining symbols; never stack raw `.values`  
6. **Build sequences** — fixed lookback on trading days; drop rows with NaN from rolling windows  
7. **Split by time** — train / val / test with no shuffle; consider holding out 2020 Q1  
8. **Normalize** — per-symbol or rolling z-score on features  

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-06-10 | Initial report created from EDA notebook findings |
| 2026-06-10 | Added Cell 19 `ValueError` (unequal array lengths) as open EDA issue E1 |
| 2026-06-10 | Cell 19 fixed — equal-length tail alignment for benchmark boxplot |

---

## 8. Next Steps

- [x] Fix Cell 19 benchmark return boxplot with equal-length alignment (last N trading days)  
- [ ] Decide minimum history and liquidity filters for training universe  
- [ ] Run full-universe quality scan (or stratified sample) and append results to Section 3  
- [ ] Create `02_data_cleaning.ipynb` implementing the pipeline in Section 6  
