# freight-rate-ml-assessment
End-to-end ML pipeline for freight rate prediction on the Spotter platform.  Features 29 engineered signals, GradientBoosting model (MAE $160.88, MAPE 6.87%),  and time-based validation. Includes 12K validation predictions and December  pricing chart for fixed Lexington → Fort Wayne lane.

# Spotter ML Assessment — Freight Rate Prediction

**Candidate:** Khalil Benbrahim  
**Role:** Machine Learning Engineer  
**Date:** August 2026

---

## Overview

End-to-end ML pipeline for predicting freight rates on the Spotter platform.  
Given historical shipment data (distance, weight, equipment, location, date), the model predicts the `posted_rate` for new loads and generates a fixed December pricing chart for a specific lane.

---

## Dataset

| File | Records | Period | Description |
|---|---|---|---|
| `train-test.csv` | 48,000 | Jan–Oct 2025 | Historical shipments with known rates |
| `validation.csv` | 12,000 | Nov 2025 | Unseen shipments requiring predictions |
| `december-chart-inputs.csv` | 31 | Dec 2025 | Fixed lane: Lexington → Fort Wayne |

**Target variable:** `posted_rate` (USD)

---

## Approach

### 1. Data Validation
- Identified 300 missing `weight` values → filled with median by equipment type
- Identified 374 missing `market_index` values → filled with global median
- December chart lacked GPS coordinates → merged lat/lon from training data via city lookup

### 2. Train/Validation Split
**Time-based chronological split (80/20):**
- **Train:** January – August 2025
- **Validation:** September – October 2025

*Rationale:* Freight rates exhibit strong temporal patterns (seasonality, weekly cycles). Random splitting would leak future information into training.

### 3. Feature Engineering (29 features)

| Category | Features |
|---|---|
| **Temporal** | month, day-of-week, quarter, weekend flag, peak-season flag |
| **Geographic** | haversine distance, lat/lon differences, cardinal direction |
| **Freight-specific** | weight-per-mile, weight×distance, distance bins |
| **Categorical** | equipment one-hot encoding, lane/city label encoding |

### 4. Model Selection

Trained and compared three models on time-based validation:

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **GradientBoosting** ✅ | **$160.88** | $678.23 | 0.802 | **6.87%** |
| RandomForest | $163.94 | $669.85 | 0.807 | 7.21% |
| Ridge (baseline) | $178.92 | $639.27 | 0.824 | 9.90% |

**Selected:** GradientBoostingRegressor — best MAE and MAPE.

**Top features:** `distance` (37%), `dist_sq` (37%), `haversine_dist` (13%)

---

## Key Results

- **Validation MAE:** $160.88 (6.87% MAPE)
- **December predictions:** $792–$827 for fixed Lexington → Fort Wayne lane
- **December pattern:** Weekly cyclicality — mid-week peaks (~$826), weekend troughs (~$792)

---

## Files in This Repo

| File | Description |
|---|---|
| `spotter_assessment.ipynb` | Full Jupyter notebook with EDA, feature engineering, modeling, predictions |
| `validation_predictions.csv` | 12,000 predictions for validation set (`load_id`, `predicted_rate`) |
| `december_predictions.csv` | 31-day December chart predictions (7 columns, fixed inputs) |
| `december_prediction_chart.png` | Visual output of December rate predictions |
| `report.pdf` | Detailed PDF report (approach, data quality, findings) |

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
