# Cas13 Protein Activity Prediction  
*(Titanic Dataset-based AutoML & XAI Practice)*

---

## 1. 팀 정보
- **팀 번호**: 개인 프로젝트
- **팀명**: Cas13 Protein Activity
- **팀원**:
  - 김태현 (EDA, 전처리, AutoML, 평가, XAI, 보고서)

---

## 2. 프로젝트 개요

- **한 줄 설명**:  
  Titanic 승객 데이터를 활용하여 생존 여부를 예측하는 이진 분류 모델을 구축하고,  
  EDA부터 AutoML, 성능 평가, XAI까지 머신러닝 분석 전 과정을 실습하는 프로젝트입니다.

- **키워드**:  
  #Titanic #BinaryClassification #AutoML #H2O #AutoGluon #XAI #SHAP

---

## 3. 데이터 소개

- **출처**:  
  - Kaggle Titanic Dataset (train / test)

- **주요 컬럼**:  
  - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Survived`

- **기간**:  
  - 1912년 Titanic 탑승객 데이터

- **전처리 개요**:  
  - 불필요 컬럼 제거 (`PassengerId`, `Name`, `Ticket`, `Cabin`)  
  - 결측치 처리 (`Age`: median, `Embarked`: mode)  
  - 범주형 변수 인코딩 (`Sex`, `Embarked`)  
  - train/test feature 정합성 유지

---

## 4. 분석/모델링 목표

- **분석 질문**:
  1. 승객의 어떤 특성이 생존 여부에 가장 큰 영향을 미치는가?
  2. AutoML 모델은 기본 머신러닝 모델 대비 어떤 성능을 보이는가?

- **사용 방법**:
  - **EDA**: ydata-profiling 기반 자동 EDA
  - **모델링**:
    - H2O AutoML
    - AutoGluon TabularPredictor
    - RandomForest (baseline)
  - **평가**:
    - Threshold 기반 Confusion Matrix
    - ROC / PR Curve
  - **XAI**:
    - SHAP global feature importance
    - SHAP local explanation

---

## 5. 폴더 구조

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

## 6. 주요 결과 요약
- **Best AutoML Model**: AutoGluon `WeightedEnsemble_L2`
- **Baseline ROC-AUC (RandomForest)**: 0.837
- **중요 변수**: Sex, Fare, Age, Pclass
- SHAP 분석을 통해 모델 예측 근거를 정량적으로 해석함.

---

##  7. Limitations & Future Work
- Titanic 데이터는 제한된 변수만 포함함.
- TPOT은 라이브러리 버전 이슈로 제한적 사용.
- 향후 과제로 Cas13 단백질 서열 데이터에 대해:
  - Cas13 sequence variants
  - k-mer / embedding 기반 feature 구축
  - AutoML 적용
  - Biological activity prediction and interpretation
  - SHAP을 통한 활성 결정 부위 해석으로 확장 가능

---

##  8. Notes
- All notebooks were executed and validated in **Google Colab**.
- Environment-specific commands (e.g., `pip install`) are included for reproducibility.
- The analysis pipeline can be extended to **Cas13 protein activity prediction**
  by replacing tabular features with sequence-based encodings.

