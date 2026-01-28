# Give Me Some Credit — Credit Risk Classification (Logistic Regression)

## Project Overview
This project is a beginner-friendly case study on **credit risk scoring** using the Kaggle dataset **“Give Me Some Credit”**.

**Goal:** predict the probability of serious delinquency within 2 years  
Target: `SeriousDlqin2yrs` (1 = default risk, 0 = non-default)

Because the positive class is rare, this is an **imbalanced classification** problem.

## What I focused on (Junior-friendly)
- probability-based predictions (`predict_proba`)
- ROC-AUC and PR-AUC (instead of accuracy only)
- handling missing values
- handling class imbalance (`class_weight="balanced"`)
- decision threshold analysis (trade-off between FN and FP)

## Model
A clean scikit-learn Pipeline:
- median imputation (+ missing indicators)
- StandardScaler
- Logistic Regression (`class_weight="balanced"`)

Simple feature idea:
- `age_x_income_missing = age * (MonthlyIncome is missing)`

## Results (Holdout validation)
- ROC-AUC: **0.7825**
- PR-AUC: **0.2971**

### Confusion matrix @ threshold = 0.50
TN = 15026, FP = 4545  
FN = 501,  TP = 889  

This threshold increases recall for the positive class (fewer missed risky cases) but produces more false positives.

### Threshold tuning (F1)
Best threshold by F1 on validation: **0.65**

TN = 18643, FP = 928  
FN = 869,  TP = 521  

This reduces false positives but increases false negatives.  
Final threshold depends on the goal (e.g., minimizing missed default cases vs. reducing false alarms).

## Kaggle Notebook
 👉 [View the Kaggle Notebook](https://www/kaggle.com/code/ksenia395/notebooke7b5b5de72)

