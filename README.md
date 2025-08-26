# Fraud Detection POC — Predicting Card-Payment Fraud with ML

## Overview
This repository contains a proof‑of‑concept (POC) machine learning system for detecting fraudulent card payments. It was developed as part of a university coursework brief: a bank asked for an initial solution that can take details about a payment and predict whether it is fraudulent, along with clear evidence of expected performance on unseen transactions.

**Why this project exists**
- Explore whether ML can help detect fraudulent payments reliably.
- Compare multiple models and explain trade‑offs (recall vs precision) for business decision‑making.
- Deliver a transparent, reproducible workflow in Python, from data prep to evaluation and validation.

## Dataset
- Source file: `bs140513_032310.csv` (publicly available on Kaggle / Blackboard in the coursework context).
- Size: **594,643** rows, **10** columns, no missing values reported. Class balance is highly skewed: **587,443** genuine vs **7,200** fraud cases (~**1.21%** fraud rate). fileciteturn0file0
- Columns used: `step, customer, age, gender, merchant, category, amount, fraud`.  
  Two zipcode columns (`zipcodeOri`, `zipMerchant`) were dropped as non‑informative. fileciteturn0file0

## Problem Statement
Given basic attributes of a payment, predict whether the payment is fraudulent. From a client perspective, the model must:
- Generalise to **new** transactions.
- Perform well for **both** classes (fraud and genuine), not just overall accuracy.
- Offer interpretable trade‑offs so the business can tune thresholds to meet operational targets (loss reduction vs review workload).

## Approach (What I built)
1. **Exploratory Data Analysis (EDA)**: shape, class imbalance, basic stats. Fraud is rare (~1.2%). fileciteturn0file0  
2. **Train/Validation Strategy**:  
   - Reserve a **hold‑out validation** set (~94,643 rows).  
   - From the remaining 500k rows, create a train/test split (**400k** train, **100k** test) for model selection. fileciteturn0file0
3. **Pre‑processing**:  
   - Drop zipcode columns.  
   - **Label encode** high‑cardinality IDs: `customer`, `merchant`.  
   - **One‑hot encode** low‑cardinality categoricals: `age`, `gender`, `category`. fileciteturn0file0
4. **Class Imbalance Handling**: Apply **SMOTE** **only on the training split** (not on test/validation) to reduce bias against the minority class. Resulting training set is balanced. fileciteturn0file0
5. **Modeling**: Train and compare **Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**. fileciteturn0file0
6. **Evaluation**: Use precision, recall, F1 for each class, and confusion matrices. Prioritise the **fraud class recall** while keeping **precision** reasonable to limit false positives. Validate the selected model on the unseen hold‑out set. fileciteturn0file0

## Key Results (Test Set → then Validated on Hold‑out)
**Test set comparisons** (summary):  
- **Logistic Regression**: strong on genuine, weaker fraud precision (many false positives). fileciteturn0file0  
- **Decision Tree**: improved fraud metrics over Logistic. fileciteturn0file0  
- **Random Forest**: high overall, fraud **precision ~0.85**, **recall ~0.78**, **F1 ~0.82**. Slight train‑set overfitting noted. fileciteturn0file0  
- **XGBoost**: balanced and robust; fraud **precision ~0.78–0.80**, **recall ~0.84–0.86**, **F1 ~0.81–0.83**. Minimal overfitting. fileciteturn0file0

**Hold‑out validation (unseen data)** — **XGBoost** selected:  
- Fraud **precision ≈ 0.80**, **recall ≈ 0.86**, **F1 ≈ 0.83**; overall accuracy ~**0.996**.  
- Confusion matrix counts (validation): **TN 93,271**, **FP 244**, **FN 159**, **TP 969**.  
- False‑positive rate ≈ **0.26%** (244 / 93,515 non‑fraud). fileciteturn0file0

> **Why XGBoost?** It delivered the best balance: high recall to catch fraud and good precision to limit false alarms, and it validated strongly on unseen data. fileciteturn0file0

## Business View (So what?)
- On a portfolio with a similar fraud rate (~1.2%), the validated **recall ~86%** means the system can catch **most** fraudulent payments before completion, reducing direct losses and downstream chargeback costs. Precision ~**80%** implies about **1 in 5** flagged transactions may be genuine, which is a manageable manual‑review burden for many banks and can be reduced by threshold tuning or a second‑stage check. fileciteturn0file0
- **Operational knobs**:  
  - **Threshold tuning** to trade recall vs precision depending on review team capacity.  
  - **Cost‑sensitive thresholding** (use expected fraud loss vs review cost).  
  - **Two‑stage review** (fast lightweight rules after ML flag, or vice‑versa) to further trim false positives.
- **Guardrails**: monitor drift, re‑train periodically, and track KPIs such as fraud capture rate, false‑positive rate, and average review time.

## What’s in this repo
```
.
├─ data/                  # place bs140513_032310.csv here (not committed if large/private)
├─ notebooks/
│  └─ Fraud Detection Model.ipynb
├─ reports/
│  └─ Fraud Detection Model.pdf
├─ src/                   # (optional) helper scripts if you modularise the notebook
└─ README.md
```
The core work is in the Jupyter notebook. A PDF of results is provided for quick review. fileciteturn0file0

## How to Reproduce
1. **Clone** the repo and (optionally) create a fresh environment.  
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If you don’t use a `requirements.txt`, install the needed libraries:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
   ```
3. **Get data**: download `bs140513_032310.csv` and put it in `data/`.
4. **Run the notebook**:
   ```bash
   jupyter notebook "notebooks/Fraud Detection Model.ipynb"
   ```
   The notebook performs EDA, preprocessing, SMOTE on the training split, model training, and evaluation on test and hold‑out validation sets.

## Method Details
- **Encoding**:  
  - Label‑encoding IDs: `customer`, `merchant` (high cardinality).  
  - One‑hot for: `age`, `gender`, `category`. fileciteturn0file0
- **Imbalance**: **SMOTE** applied only on the training set. fileciteturn0file0
- **Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, compared via per‑class precision/recall/F1 and confusion matrices. fileciteturn0file0
- **Selection**: XGBoost chosen for strongest hold‑out validation (fraud recall ~0.86, precision ~0.80). fileciteturn0file0

## Results Snapshot (Validation, XGBoost)
- **Precision (fraud)**: ~**0.80**  
- **Recall (fraud)**: ~**0.86**  
- **F1 (fraud)**: ~**0.83**  
- **Accuracy**: ~**0.996**  
- **Confusion matrix**: TN **93,271**, FP **244**, FN **159**, TP **969**. fileciteturn0file0

## Limitations & Next Steps
- **Feature scope** is intentionally minimal (POC). Enrich with device, velocity, graph features, and recent behaviour windows.  
- **Explainability**: add SHAP/feature importance for investigator‑friendly reasons on flags.  
- **Threshold & cost**: calibrate decision threshold to a cost matrix (fraud loss vs review cost) and maximise expected savings.  
- **Robustness**: k‑fold CV, temporal splits (by `step`) to mimic production time‑ordering, and hyper‑parameter tuning.  
- **Monitoring**: drift detection and periodic re‑training; track precision/recall by segment (merchant, amount, category).

## Ethical Use
- Use only on legally obtained data with proper consent and governance.  
- Beware of indirect bias from proxies (e.g., location, merchant types). Regularly audit fairness across customer groups.  
- Keep humans in the loop for disputed/edge cases and allow appeal paths.

## Getting Started Fast
If you simply want the trained XGBoost from the notebook and to run quick predictions on a CSV in the same schema, see the last notebook cells. You can export the model with `joblib` and load it in a small script for batch scoring.

## Acknowledgements
- Coursework brief and dataset reference provided by the university; all modelling and write‑up here were implemented from scratch by me.  
- Dataset filename: `bs140513_032310.csv` (Kaggle / Blackboard context).

## License
MIT (or your preferred license).

---

**Author**: Mustafa Scentwala  
If you have questions or ideas to improve the POC, feel free to open an issue or reach out.
