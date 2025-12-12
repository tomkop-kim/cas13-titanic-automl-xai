1. 프로젝트 정보
- 과목: 머신러닝입문
- 팀명: Cas13 Protein Activity
- 주제: Cas13 변이 서열 기반 단백질 활성 예측
- 실습 데이터/과제 수행: Titanic train/test 데이터셋 기반 AutoML + 평가 + XAI

2. 목표 및 문제 정의
- 본 프로젝트의 목적은 Titanic 승객 데이터를 이용해 **생존 여부(Survived)**를 예측하는 이진 분류 모델을 구축하고,
(1) 자동 EDA를 통한 데이터 이해, (2) 전처리 및 feature engineering, (3) AutoML 기반 모델 탐색, (4) 다양한 지표 기반 성능 평가, (5) SHAP 기반 XAI를 통해 모델 해석까지 전 과정을 재현 가능하게 구현하는 것이다.

3. 데이터 소개
- 데이터셋: Titanic (Kaggle train/test)
- Train shape: (892,12)
- Test shape: (418,11)
- 주요 변수:
1) Pclass: 객실 등급(1~3)
2) Sex: 성별
3) Age: 나이
4) SibSp, Parch: 동승 가족 수
5) Fare: 요금
6) Embarked: 승선 항구
7) Survived: 생존 여부(Train에만 존재)

4. EDA (자동 리포트 기반 탐색):
본 프로젝트에서는 자동 EDA 도구인 ydata-profiling을 사용하여 데이터 분포 및 결측, feature 특성, train/test 차이를 확인하였다.
4.1. EDA 산출물
- titanic_train_eda.html: train 데이터 자동 EDA 리포트
- titanic_train_test_eda.html: train+test 결합 비교 리포트(구분 컬럼 _dataset 추가)
 
Figure1. Automated EDA summary of Titanic training dataset
4.2. 주요 관찰 요약
- Age 컬럼에 결측치가 존재하며, Embarked에도 일부 결측치가 존재함을 확인하였다.
- 범주형 변수(Sex, Embarked)는 모델 입력을 위해 인코딩이 필요하다.
- 생존 여부는 일부 변수(성별, 객실 등급 등)와 강한 연관성을 가질 가능성이 있으며, 이후 모델링 단계에서 중요 변수로 작용할 수 있다.

5. 전처리 및 Feature Engineering:
모델 입력 형태로 변환하기 위해 아래 전처리 과정을 수행하였다.
5.1. 불필요 컬럼 제거: 
분석/예측에 직접적으로 도움이 적거나(고유 ID), 고차원 문자열 특성을 유발하는 변수는 제거하였다.
 - 제거 컬럼: PassengerId, Name, Ticket, Cabin
5.2. 결측치 처리
- Age: 중앙값(median)으로 대체
- Embarked: 최빈값(mode)으로 대체
5.3. 범주형 변수 인코딩
- Sex: male=0, female=1로 변환
- Embarked: one-hot encoding 적용 (drop_first=True)
5.4. train/test 컬럼 정합성 유지:
one-hot encoding 이후 train/test의 컬럼이 달라질 수 있으므로, test에 대해 train의 feature 컬럼 기준으로 reindex하여 컬럼을 맞추었다.
- test_proc = test_proc.reindex(columns=train_cols, fill_value=0)

6. 모델링 (AutoML + Baseline):
본 프로젝트에서는 과제 요구에 따라 AutoML 3종을 활용하여 모델을 학습하였다
6.1. 데이터 분할
- Train/Validation split: test_size = 0.2, stratify 적용
- 목적: 클래스 비율을 유지한 상태로 검증 성능을 안정적으로 비교
6.2. AutoML 모델
(1) H2O AutoML
- 설정: max_models=10, balance_classes=True, sort_metric=“AUC”
- Valid AUC: (0.8538866930171278) 
 
Figure 2. H2O AutoML leaderboard on validation dataset
(2) AutoGluon TabularPredictor
- 설정: eval_metric=“roc_auc”, presets=“medium_quality”, time_limit=600
- Leaderboard 상위 모델: (WeightedEnsemble_L2) 
- Valid ROC-AUC: (0.928305785123967)
 
Figure 3. AutoML leaderboard comparison (AutoGluon)
(3) TPOT:
TPOT은 라이브러리 버전(API) 변경으로 인해 옵션 파라미터 호환 문제가 발생하여, 제한적 실행/로그 확인 방식으로 AutoML 적용을 확인하였다.
- 실행 로그 기반으로 TPOT 탐색 시도 확인
- AutoML 비교/결론은 H2O AutoML과 AutoGluon 중심으로 진행
6.3. 최종 평가용 모델 (Baseline):
AutoML과 별도로, 평가/시각화/SHAP 적용의 안정성을 위해 RandomForestClassifier를 최종 평가 모델로 사용하였다.
- n_estimators=300, random_state=42

7. 모델 성능 평가 (Evaluation): 
검증 데이터(X_valid)에 대해 확률 예측값(y_proba)을 기반으로 다양한 평가를 수행하였다.
  7.1. Threshold 기반 Confusion Matrix (0.1~0.9):
임계값(threshold)을 0.1부터 0.9까지 변화시키며 예측 클래스를 생성하고, 각 임계값에 대한 Confusion Matrix를 3×3 grid로 시각화하였다.
- 관찰: threshold가 증가할수록 보통 precision은 증가, recall은 감소하는 trade-off가 나타남
 
Figure 4. Confusion matrices under different decision thresholds (0.1–0.9)
 7.2. Accuracy vs Threshold:
각 threshold에서 accuracy를 계산하여 곡선으로 시각화하였다.
- 최고 accuracy threshold: (0.5) (그래프에서 최대값 위치)
- 해당 accuracy: (0.83)
 7.3. ROC Curve + ROC AUC
 - ROC AUC (RandomForest): (0.837)
 
Figure 5. ROC curve of the RandomForest classifier
 7.4. Precision-Recall Curve + PR AUC
 - PR AUC (RandomForest): (0.791)

8. XAI (SHAP 기반 모델 해석):
RandomForest 모델에 대해 SHAP을 적용하여 모델이 어떤 feature를 근거로 예측하는지 해석하였다.
8.1 Global feature importance (SHAP summary bar):
- SHAP summary bar plot을 통해 전역적으로 영향력이 큰 변수를 확인하였다.
- 주요 영향 변수는 Sex, Fare, Age, Pclass 등이었으며, 이는 Titanic 생존과 관련된 직관적 요인과도 일치한다.
- SHAP 분석 결과, Sex(성별) 변수는 생존 여부 예측에 가장 큰 영향력을 보였다.
- 또한 Fare(요금) 및 **Pclass(객실 등급)**는 승객의 사회·경제적 지위를 반영하는 변수로서 생존 확률에 유의미한 기여를 하였고, Age(나이) 역시 생존 예측에 중요한 요인으로 작용하였다.
- 이러한 결과는 실제 사고 당시 구조 우선순위 및 탑승 계층 차이가 생존율에 영향을 미쳤다는 역사적 맥락과도 일관된다.
 
Figure 6. SHAP global feature importance (bar plot)
8.2 Local explanation (개별 샘플 3건):
검증 데이터에서 임의의 샘플 3개를 선택하여 SHAP 기반 force/waterfall plot으로 개별 예측 근거를 확인하였다.
- 샘플 1: (index=180) / 예측확률=0.040 / 실제 라벨=0
- 샘플 2: (index=485) / 예측확률=0.030 / 실제 라벨=0
- 샘플 3: (index=38) / 예측확률=0.167/ 실제 라벨=0
개별 분석 결과, 특정 샘플은 Sex=1(여성), 높은 Fare, 낮은 Pclass(1등석) 등 요인으로 생존 확률이 상승하는 방향의 SHAP 기여가 관찰되었고,
반대로 남성 + 3등석 + 낮은 Fare 조합에서는 생존 확률을 낮추는 방향의 기여가 누적되는 것을 확인하였다.
 
Figure 7. SHAP local explanation for an individual prediction
9. Test dataset 예측
최종 모델(RandomForest)을 이용해 test 데이터(test_proc)에 대해 생존 여부를 예측하고 submission_rf.csv로 저장하였다.
- 저장 파일: submission_rf.csv
- 기준 threshold: 0.5

10. 한계 및 향후 과제
- Titanic 데이터는 제한된 변수로 구성되어 있어, 실제 생존 여부에 영향을 주는 다양한 외생 변수를 포함하지 못한다.
- AutoML 비교에서 TPOT은 최신 버전 API 변경 이슈로 인해 탐색 설정을 충분히 조정하지 못했다.
- 향후 과제로는 (1) 더 정교한 feature engineering, (2) calibration/threshold 최적화, (3) 다른 모델(GBM 계열)과의 비교 및 앙상블 등을 고려할 수 있다.
또한 본 팀의 도메인 주제인 Cas13 단백질 활성 예측 문제에도,
이번 프로젝트에서 구축한 EDA→전처리→모델링→평가→XAI 파이프라인을 확장 적용할 수 있다.
예를 들어, 서열 기반 feature(k-mer, embedding 등) 구축 후 AutoML 및 SHAP 분석을 통해 활성 결정 부위를 해석하는 방식으로 응용 가능하다.
