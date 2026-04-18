# Classical ML Classification — Gesture Phase Segmentation

> **MSc Data Science | Kingston University**
> Group project — team of 4 | My contributions: K-Nearest Neighbours (KNN) implementation & tuning, Gradient Boosting implementation & tuning, data preprocessing pipeline

---

## Overview

This project applies classical machine learning classification techniques to the **Gesture Phase Segmentation** dataset from OpenML (data_id: 4538). The dataset contains features extracted from videos of people gesticulating, with the goal of classifying gesture phases (Rest, Preparation, Stroke, Hold, Retraction — labelled D, H, P, R, S).

The project covers the full machine learning pipeline from data loading and preprocessing through to model comparison and evaluation across 8 classifiers.

---

## Tools & Technologies

- Python 3
- scikit-learn
- pandas, NumPy
- Matplotlib, Seaborn
- XGBoost
- OpenML API

---

## Results — All 8 Models

| Model | Balanced Accuracy | ROC AUC (Macro) | ROC AUC (Micro) |
|---|---|---|---|
| XGBoost | 0.6242 | 0.8960 | 0.9082 |
| **KNN** ⬅ my work | **0.6148** | **0.8469** | **0.8610** |
| Random Forest | 0.5939 | 0.8941 | 0.9032 |
| **Gradient Boosting** ⬅ my work | **0.5531** | **0.8557** | **0.8727** |
| SVM | 0.4693 | 0.8087 | 0.8334 |
| Decision Tree | 0.4734 | 0.6948 | 0.7148 |
| AdaBoost | 0.3907 | 0.7664 | 0.7963 |
| Logistic Regression | 0.3582 | 0.7599 | 0.7887 |

---

## My Contributions

### K-Nearest Neighbours (KNN)
- Built a full scikit-learn **Pipeline** with median imputation, StandardScaler, and KNeighborsClassifier
- Tuned hyperparameters using **GridSearchCV** across:
  - `n_neighbors`: [3, 5, 7, 9, 11, 15, 21, 31]
  - `weights`: [uniform, distance]
  - `p`: Manhattan (1) vs Euclidean (2) distance
- Best params: `n_neighbors=3`, `weights=distance`, `p=1` (Manhattan distance)
- Best CV Balanced Accuracy: **0.5553** → Test Balanced Accuracy: **0.6148**
- ROC AUC Macro: 0.8469 | ROC AUC Micro: 0.8610
- Evaluated with confusion matrix and ROC AUC scores

### Gradient Boosting
- Built a Pipeline with median imputation and GradientBoostingClassifier
- Tuned using **GridSearchCV** (3-fold CV) across:
  - `n_estimators`: [150, 200]
  - `learning_rate`: [0.05, 0.1]
  - `max_depth`: [2, 3]
  - `subsample`: [0.8]
- Best params: `learning_rate=0.1`, `max_depth=3`, `n_estimators=200`, `subsample=0.8`
- Best CV Balanced Accuracy: **0.5168** → Test Balanced Accuracy: **0.5531**
- ROC AUC Macro: 0.8557 | ROC AUC Micro: 0.8727
- Evaluated with confusion matrix and ROC AUC scores

### Data Preprocessing
- Applied **StandardScaler** normalisation for distance-sensitive models (KNN)
- **70/30 stratified train-test split** — training: 6,911 samples | test: 2,962 samples
- **LabelEncoder** for target class encoding

---

## Key Techniques

- scikit-learn Pipelines (imputer → scaler → classifier)
- GridSearchCV hyperparameter tuning with cross-validation
- Balanced Accuracy and ROC AUC (Macro + Micro) evaluation
- Confusion matrix visualisation
- StandardScaler normalisation
- Stratified train-test split for class balance preservation

---

## Dataset

**Gesture Phase Segmentation (Processed)** — OpenML dataset ID: 4538
32 numeric features | 5 gesture phase classes | 9,873 total samples
Source: UCI Machine Learning Repository / OpenML
Creators: Renata Cristina Barros Madeo, Priscilla Koch Wagner, Sarajane Marques Peres — University of São Paulo, Brazil

---

MSc Data Science, Kingston University, London
