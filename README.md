# Fraud Detection POC - Predicting Card-Payment Fraud with ML

## Overview
This repository contains a proof-of-concept (POC) machine learning system for detecting fraudulent card payments. It was developed as part of a university coursework brief: a bank asked for an initial solution that can take details about a payment and predict whether it is fraudulent, along with clear evidence of expected performance on unseen transactions.

**Why this project exists**
- Explore whether ML can help detect fraudulent payments reliably.
- Compare multiple models and explain trade-offs (recall vs precision) for business decision-making.
- Deliver a transparent, reproducible workflow in Python, from data prep to evaluation and validation.

## Dataset
- Source file: bs140513_032310.csv (publicly available on Kaggle or Blackboard in the coursework context).
- Size: 594,643 rows, 10 columns, no missing values reported. Class balance is highly skewed: 587,443 genuine vs 7,200 fraud cases (about 1.21 percent fraud rate).
- Columns used: step, customer, age, gender, merchant, category, amount, fraud.
  Two zipcode columns (zipcodeOri, zipMerchant) were dropped as non-informative.

## Problem Statement
Given basic attributes of a payment, predict whether the payment is fraudulent. From a client perspective, the model must:
- Generalise to new transactions.
- Perform well for both classes (fraud and genuine), not just overall accuracy.
- Offer interpretable trade-offs so the business can tune thresholds to meet operational targets (loss reduction vs review workload).

## Approach (What I built)
1. Exploratory Data Analysis (EDA): shape, class imbalance, basic stats. Fraud is rare (about 1.2 percent).
2. Train and validation strategy:
   - Reserve a hold-out validation set (about 94,643 rows).
   - From the remaining 500k rows, create a train and test split (about 400k train, 100k test) for model selection.
3. Pre-processing:
   - Drop zipcode columns.
   - Label encode high-cardinality IDs: customer, merchant.
   - One-hot encode low-cardinality categoricals: age, gender, category.
4. Class imbalance handling: apply SMOTE only on the training split (not on test or validation) to reduce bias against the minority class. The resulting training set is balanced.
5. Modeling: train and compare Logistic Regression, Decision Tree, Random Forest, and XGBoost.
6. Evaluation: use precision, recall, and F1 for each class, and confusion matrices. Prioritise the fraud class recall while keeping precision reasonable to limit false positives. Validate the selected model on the unseen hold-out set.

## Key Results (Test set then validated on hold-out)
Test set comparisons (summary):
- Logistic Regression: strong on genuine, weaker fraud precision (many false positives).
- Decision Tree: improved fraud metrics over Logistic.
- Random Forest: high overall, fraud precision about 0.85, recall about 0.78, F1 about 0.82. Slight train-set overfitting noted.
- XGBoost: balanced and robust; fraud precision about 0.78 to 0.80, recall about 0.84 to 0.86, F1 about 0.81 to 0.83. Minimal overfitting.

Hold-out validation (unseen data) - XGBoost selected:
- Fraud precision approx 0.80, recall approx 0.86, F1 approx 0.83; overall accuracy approx 0.996.
- Confusion matrix counts (validation): TN 93,271, FP 244, FN 159, TP 969.
- False-positive rate approx 0.26 percent (244 of 93,515 non-fraud).

Why XGBoost? It delivered the best balance: high recall to catch fraud and good precision to limit false alarms, and it validated strongly on unseen data.

## Business View (So what?)
- On a portfolio with a similar fraud rate (about 1.2 percent), the validated recall around 86 percent means the system can catch most fraudulent payments before completion, reducing direct losses and downstream chargeback costs. Precision around 80 percent implies about 1 in 5 flagged transactions may be genuine, which is a manageable manual-review burden for many banks and can be reduced by threshold tuning or a second-stage check.
- Operational knobs:
  - Threshold tuning to trade recall vs precision depending on review team capacity.
  - Cost-sensitive thresholding (use expected fraud loss vs review cost).
  - Two-stage review (fast lightweight rules after an ML flag, or the reverse) to further trim false positives.
- Guardrails: monitor drift, re-train periodically, and track KPIs such as fraud capture rate, false-positive rate, and average review time.

## What is in this repo
.
- data/                  place bs140513_032310.csv here (not committed if large or private)
- notebooks/             Fraud Detection Model.ipynb
- reports/               Fraud Detection Model.pdf
- src/                   optional helper scripts if you modularise the notebook
- README.md

The core work is in the Jupyter notebook. A PDF of results is provided for quick review.

## How to Reproduce
1. Clone the repo and optionally create a fresh environment.
2. Install dependencies:
   pip install -r requirements.txt

   If you do not use a requirements.txt, install the needed libraries:
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

3. Get data: download bs140513_032310.csv and put it in data/.
4. Run the notebook:
   jupyter notebook "notebooks/Fraud Detection Model.ipynb"

   The notebook performs EDA, preprocessing, SMOTE on the training split, model training, and evaluation on test and hold-out validation sets.

## Method Details
- Encoding:
  - Label-encoding IDs: customer, merchant (high cardinality).
  - One-hot for: age, gender, category.
- Imbalance: SMOTE applied only on the training set.
- Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, compared via per-class precision, recall, F1 and confusion matrices.
- Selection: XGBoost chosen for strongest hold-out validation (fraud recall around 0.86, precision around 0.80).

## Results Snapshot (Validation, XGBoost)
- Precision (fraud): approx 0.80
- Recall (fraud): approx 0.86
- F1 (fraud): approx 0.83
- Accuracy: approx 0.996
- Confusion matrix: TN 93,271, FP 244, FN 159, TP 969

## Limitations and Next Steps
- Feature scope is intentionally minimal (POC). Enrich with device, velocity, graph features, and recent behaviour windows.
- Explainability: add SHAP or feature importance for investigator-friendly reasons on flags.
- Threshold and cost: calibrate decision threshold to a cost matrix (fraud loss vs review cost) and maximise expected savings.
- Robustness: k-fold cross-validation, temporal splits by step to mimic production time-ordering, and hyper-parameter tuning.
- Monitoring: drift detection and periodic re-training; track precision and recall by segment (merchant, amount, category).

## Ethical Use
- Use only on legally obtained data with proper consent and governance.
- Beware of indirect bias from proxies (for example, location, merchant types). Regularly audit fairness across customer groups.
- Keep humans in the loop for disputed or edge cases and allow appeal paths.

## Getting Started Fast
If you simply want the trained XGBoost from the notebook and to run quick predictions on a CSV with the same schema, see the last notebook cells. You can export the model with joblib and load it in a small script for batch scoring.

## Acknowledgements
- Coursework brief and dataset reference were provided by the university; all modelling and write-up here were implemented from scratch by me.
- Dataset filename: bs140513_032310.csv (Kaggle or Blackboard context).

## License
MIT.

Author: Mustafa Scentwala
