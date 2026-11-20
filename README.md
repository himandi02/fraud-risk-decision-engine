# Fraud Risk Decision Engine : README

This project builds a transaction fraud decision engine: a calibrated fraud probability, an **Approve / Review / Block** decision, and an explanation for each risky transaction. It also covers validation, cost analysis, robustness testing, monitoring and governance.

All money figures are **hypothetical**. The hold-out set (600 rows, 73 frauds) has **not been used for any choice** and is reserved for one final check.

---

## 1. What is in this folder

| File | What it is | Produced by |
|---|---|---|
| `01_fraud_risk_analysis.ipynb` | Complete analysis, Parts A to K (297 cells, with outputs) | the notebook itself |
| `02_processed_dataset.csv` | Raw columns plus the 29 derived features (3,000 rows, 55 columns) | Part C |
| `03_fraud_model_pipeline.joblib` | Uncalibrated Random Forest pipeline (preprocessing + model), fitted on the 2,400 development rows | Part F8 |
| `04_calibrated_model.joblib` | Production candidate: the same pipeline plus the Platt calibration layer | Part F8 |
| `05_threshold_cost_analysis.csv` | Threshold and cost sensitivity results (257 rows, 34 columns; the `analysis` column separates policy by scenario, operating limits, ratio sweep, regret, tornado, recall-workload) | Part E8 |
| `06_model_card.pdf` | Model card: intended use, limitations, data assumptions, evaluation method, known risks, monitoring, triggers | Part K |
| `07_business_fraud_strategy.pdf` | 9-page management report | Parts A to K |
| `08_README.md` | This file | |
| `fraud_governance.py` | Helper module used by the Part K cells (cost framework re-implementation and governance evidence functions) | Part K |
| `fraud_features.py` | Feature engineering and feature-set selector (needed to load the `.joblib` files) | Part C |
| `fraud_calibration.py` | Platt calibrator and the `CalibratedFraudModel` wrapper stored in `04_calibrated_model.joblib` | Part F |
| `fraud_costs.py` | Cost scenarios, threshold search and honest cost evaluation behind `05_threshold_cost_analysis.csv` | Part E |


---

## 2. Environment

* Python 3.10 or newer (the Part K cells were last run with Python 3.13).
* The saved models were written with **scikit-learn 1.9.1**. Load them with the same version, because joblib files are not guaranteed to open across scikit-learn versions.
* Packages used by the notebook: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `imbalanced-learn` (resampling arms in Part G), `shap` (Part H), `matplotlib`, `joblib`, `nbformat` and `nbclient` (only to run the notebook headlessly).
* I did not freeze exact versions of the packages other than scikit-learn. After a successful run, record the environment with `pip freeze > requirements.txt` and keep it with the results.

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scipy statsmodels scikit-learn==1.9.1 imbalanced-learn shap matplotlib joblib jupyter
```

### Helper modules the notebook imports

The notebook calls project modules that must sit **in the same folder as the notebook**:

`fraud_features`, `fraud_models`, `fraud_eval`, `fraud_costs`, `fraud_calibration`, `fraud_imbalance`, `fraud_explain`, `fraud_ops`, `fraud_stress`, and (for Part K) `fraud_governance` (included here).

`fraud_features.py`, `fraud_calibration.py` and `fraud_costs.py` are included in this folder. They were rebuilt from the Part C feature catalogue, the Part E cost definitions and the Part F calibration settings, and were checked against the notebook's own results: the learned constants and the per-feature missing-value counts match exactly, all 457 Part H probabilities reproduce, the Platt coefficients match to 9 decimals, and every printed Part E table matches. The remaining modules (`fraud_models`, `fraud_eval`, `fraud_imbalance`, `fraud_explain`, `fraud_ops`, `fraud_stress`) are not in this folder.

Two things follow from this:

1. The notebook will not run end to end without the remaining modules. Copy the module files used during development next to the notebook; if your own copies of `fraud_features`, `fraud_calibration` or `fraud_costs` differ in internals from the versions here, load the `.joblib` files with the versions in this folder.
2. The `.joblib` files contain classes from `fraud_features` and `fraud_calibration`, so **both files must be importable** (same folder or on `PYTHONPATH`) when loading them.

---

## 3. How to run

1. Put the raw data file. 
2. Put all helper modules next to the notebook.
3. Run the notebook **top to bottom in order**.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_fraud_risk_analysis.ipynb
```

| Part | Content | Main outputs |
|---|---|---|
| A | Data audit, quality report, leakage register, hold-out split | `02a_audited_dataset.csv`, `split_assignment.csv`, `dq_*.csv` |
| B | Exploratory analysis (at least 10 figures, interaction plots) | `B_*.csv` |
| C | Feature engineering (29 derived features, with / without comparison) | `02_processed_dataset.csv`, `C_*.csv`, `C_learned_constants.json` |
| D | Logistic regression baseline, Random Forest, HistGradientBoosting; nested CV and tuning | `D_leaderboard.csv`, `D_oof_predictions.csv`, `D_final_models.json`, `D_*.joblib` |
| E | Cost matrix, cost-based thresholds, seven scenarios, sensitivity | `05_threshold_cost_analysis.csv`, `E_chosen_policy.json` |
| F | Calibration (raw, Platt, isotonic) and the calibrated model | `03_...joblib`, `04_...joblib`, `F_calibration_choice.json` |
| G | Class imbalance: weights, undersampling, oversampling, SMOTE inside the pipeline | `G_*.csv` |
| H | SHAP global and transaction-level explanations, reason codes | `H_*.csv` |
| I | Operations simulation, comparison with simpler policies | `I_*.csv`, `I_decision_policy.json` |
| J | Stress tests (49 scenarios), fold variability, degradation causes | `J_*.csv` |
| K | Model governance: monitoring, drift, triggers, eight questions | `K_*.csv`, `K0*_*.png`, `06_model_card.pdf` |

Part K, run alone (it reads only saved Part D to J outputs and takes about two minutes), needs `fraud_governance.py` in the working folder and the `outputs/` files listed above.

---

## 4. Using the saved models

```python
import joblib, pandas as pd
# fraud_features.py must be importable
model = joblib.load("04_calibrated_model.joblib")        # calibrated production candidate
X = pd.read_csv("new_transactions.csv")                   # same raw columns as the training data, without 'fraud'
p = model.predict_proba(X)[:, 1]                          # calibrated fraud probability
```

* The pipeline does its own cleaning, feature engineering, imputation and encoding, so pass **raw columns**.
* The decision step is `fraud_ops.decide(p, amount, scenario, shadow_price_review, shadow_price_block)`. The policy settings are in `I_decision_policy.json`. The simple fallback is: review from a calibrated probability of 0.158, block from 0.212.
* `authentication_level` contains the literal text **`None`** (authentication not performed) in 289 of 3,000 rows. `pandas.read_csv` turns this into a missing value by default. The pipeline expects that and keeps it as its own category, so keep the default reading, but do not drop or impute those rows.

---

## 5. Reproducibility and rules followed

* **Seed:** 42 everywhere (hold-out split, cross-validation, model fitting, bootstraps, simulations). The seed is set at the top of each part.
* **Hold-out set:** a stratified 20% split (600 rows, 73 frauds), created before any analysis that looks at the label and saved in `split_assignment.csv`. It was not used for feature selection, tuning, threshold choice, calibration, resampling or any figure in the reports. The final check is still to be done once, and its result recorded in the model card.
* **No leakage:** all learning (imputation, scaling, encoding, derived-feature constants, resampling, calibration) happens inside training folds. Resampling is tested to run only on training rows (Part G2).
* **Nested cross-validation:** 5 folds x 2 repeats, tuning in the inner loop only.
* **Difficult observations were kept.** No rows were removed to improve metrics.
* **Honest cost:** thresholds are chosen on training rows and scored on unseen rows.

---

## 6. Assumptions

**Cost framework (hypothetical; base scenario S1, all sensitivity-tested in Parts E, I and K):**

| Assumption | Base value | Range tested |
|---|---|---|
| Loss share of an approved fraud | 100% of the amount | 50% to 100% |
| Chargeback fee | 15 | 5 to 60 |
| Margin lost on a blocked genuine customer | 5% of the amount | 0% to 10% |
| Friction cost of a block | 20 | 5 to 60 |
| Cost of a manual review | 6 (8 minutes at 45 per hour) | 2 to 15 |
| Share of fraud caught in review | 85% | 60% to 95% |
| Genuine reviewed transaction wrongly declined | 3% | 0% to 10% |
| Review delay cost (genuine) | 1.5 | 0 to 5 |
| Review capacity / block limit | 10% / 10% of traffic | 5% to 20% / 2% to 20% one at a time, no limits in S7 |

Seven named scenarios (S1 to S7) combine these, and Part K also draws 300 random cost structures. **Operations:** 100,000 transactions a month, 22 working days, 6.5 productive hours per analyst-day.

**Data:** 3,000 transactions, 25 columns, fraud rate 12.1% (291 of 2,400 development rows). There are no timestamps (only hour and day of week), so true temporal drift could not be tested. About 40 missing amounts are filled with the development median for **cost accounting only**.

---

## 7. Known limitations

* Only 291 development frauds: model differences are mostly within noise and the interaction findings are exploratory.
* Costs are assumptions and the 12.1% fraud rate is far above a real portfolio. At a much lower rate the probabilities need the prior-shift correction (Part J7) and the cost analysis must be re-run.
* The amount-aware rule was selected on the same development rows it is evaluated on (probabilities and fixed thresholds are out-of-fold). The hold-out check is meant to test this.
* The advantage over simple business rules (4.2%) is not statistically reliable.
* No fairness assessment by customer group was possible (no protected attributes in the data).
* Part K re-implements the Part E cost framework in `fraud_governance.py` and agrees with Part E to within 0.4% (table `K_validation_vs_partE.csv`).
* I have not recorded how long the full notebook takes to run end to end; Part K alone takes about two minutes.
