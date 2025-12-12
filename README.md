# Cas13 Protein Activity Prediction  
*(Titanic Dataset-based AutoML & XAI Practice)*

---

## 1. Team Information
- **Course**: Introduction to Machine Learning (머신러닝입문)
- **Team Name**: Cas13 Protein Activity
- **Member**: 김태현

---

## 2. Project Overview
This project aims to build a binary classification model to predict passenger survival
on the Titanic dataset, while demonstrating a complete machine learning pipeline
including **EDA, preprocessing, AutoML-based modeling, evaluation, and XAI**.

Although the team’s primary research interest is **Cas13 protein activity prediction
based on sequence variants**, the Titanic dataset was used for AutoML and XAI practice,
as allowed by the course.  
The overall analysis pipeline developed in this project can be directly extended to
biological sequence-based prediction tasks such as Cas13 activity modeling.

---

## 3. Dataset
- **Dataset**: Titanic (Kaggle train/test)
- **Target Variable**: `Survived` (binary classification)
- **Train Shape**: (892, 12)
- **Test Shape**: (418, 11)

### Main Features
- `Pclass`: Passenger class (1–3)
- `Sex`: Gender
- `Age`: Age
- `SibSp`, `Parch`: Number of family members aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation

---

## 4. Analysis Pipeline

### 4.1 Exploratory Data Analysis (EDA)
- Automated EDA was performed using **ydata-profiling**.
- Data distribution, missing values, and feature characteristics were explored.
- Both **train-only EDA** and **train vs test comparison EDA** were generated.

**Outputs**
- `reports/eda/titanic_train_eda.html`
- `reports/eda/titanic_train_test_eda.html`

---

### 4.2 Preprocessing & Feature Engineering
- Removed non-informative or high-cardinality columns  
  (`PassengerId`, `Name`, `Ticket`, `Cabin`)
- Missing values handled using training-data statistics:
  - `Age`: median
  - `Embarked`: mode
- Categorical encoding:
  - `Sex`: binary encoding
  - `Embarked`: one-hot encoding
- Ensured feature alignment between train and test datasets.

**Outputs**
- `data/interim/titanic_cleaned.csv`
- `data/processed/titanic_features.csv`

---

### 4.3 Modeling (AutoML & Baseline)
The following models were applied:

- **H2O AutoML**
  - Configuration: `max_models=10`, `balance_classes=True`
  - Validation metric: ROC-AUC
- **AutoGluon TabularPredictor**
  - Preset: `medium_quality`
  - Best model: `WeightedEnsemble_L2`
- **Baseline Model**
  - RandomForestClassifier (`n_estimators=300`)

AutoML results were compared using validation ROC-AUC scores.

---

### 4.4 Evaluation
Model performance was evaluated on the validation set using:
- Threshold-based Confusion Matrix (thresholds 0.1–0.9)
- Accuracy vs Threshold curve
- ROC Curve & ROC-AUC
- Precision-Recall Curve & PR-AUC

**Key Result**
- RandomForest ROC-AUC: **0.837**

---

### 4.5 Explainable AI (XAI)
- SHAP was applied to the RandomForest model for interpretability.
- **Global explanation**: SHAP summary bar plot
- **Local explanation**: Individual prediction explanations for selected samples

Key influential features:
- `Sex`
- `Fare`
- `Age`
- `Pclass`

These results are consistent with known historical factors affecting Titanic survival.

---

## 5. Repository Structure

```text
project-root/
├─ README.md
├─ data/
│  ├─ raw/
│  │   ├─ cas13_variants.csv
│  │   ├─ train.csv
│  │   └─ test.csv
│  ├─ interim/
│  │   └─ titanic_cleaned.csv
│  └─ processed/
│      └─ titanic_features.csv
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_modeling_analysis.ipynb
│  └─ 04_visualization_report.ipynb
├─ reports/
│  ├─ eda/
│  │   ├─ titanic_train_eda.html
│  │   └─ titanic_train_test_eda.html
│  ├─ figures/
│  │   ├─ cas13_example_diagram.png
│  │   ├─ figure1.png
│  │   ├─ figure2.png
│  │   ├─ figure3.png
│  │   ├─ figure4.png
│  │   ├─ figure5.png
│  │   ├─ figure6.png
│  │   └─ figure7.png
│  └─ final_report.md
└─ src/
   └─ preprocessing.py
```
---

## 6. Main Results Summary
- **Best AutoML Model**: AutoGluon `WeightedEnsemble_L2`
- **Baseline ROC-AUC**: 0.837
- **Most important features**: Sex, Fare, Age, Pclass
- Successfully demonstrated an end-to-end ML workflow with AutoML and XAI.

---

## 7. Notes
- All notebooks were executed and validated in **Google Colab**.
- Environment-specific commands (e.g., `pip install`) are included for reproducibility.
- The analysis pipeline can be extended to **Cas13 protein activity prediction**
  by replacing tabular features with sequence-based encodings.

---

## 8. Limitations & Future Work
- Titanic dataset contains limited contextual variables.
- TPOT AutoML was only partially explored due to library version constraints.
- Future work includes applying the same pipeline to:
  - Cas13 sequence variants
  - k-mer or embedding-based features
  - Biological activity prediction and interpretation
