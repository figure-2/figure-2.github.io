---
title: XGBoost (eXtreme Gradient Boosting)
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- XGBoost
- 그라디언트부스팅
- 머신러닝
- 하이퍼파라미터
- 성능최적화
- SHAP
toc: true
date: 2023-09-15 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# XGBoost (eXtreme Gradient Boosting)

> 230926 학습한 내용 정리

## XGBoost 개요

### 정의
- **eXtreme Gradient Boosting**의 줄임말
- 그래디언트 부스팅 알고리즘의 최적화된 구현
- 높은 성능과 빠른 속도로 유명한 머신러닝 라이브러리

### 특징
- **높은 성능**: 많은 머신러닝 대회에서 우승
- **빠른 속도**: 병렬 처리와 최적화로 빠른 학습
- **메모리 효율**: 압축된 데이터 구조 사용
- **유연성**: 다양한 목적 함수와 평가 지표 지원

### 장점
- **정확도**: 높은 예측 정확도
- **속도**: 빠른 학습 및 예측 속도
- **안정성**: 과적합 방지 기능 내장
- **확장성**: 대용량 데이터 처리 가능

## XGBoost 설치 및 기본 사용법

### 1. 설치
```bash
# pip 설치
pip install xgboost

# conda 설치
conda install -c conda-forge xgboost
```

### 2. 기본 사용법
```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 샘플 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 생성
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 모델 학습
xgb_model.fit(X_train, y_train)

# 예측
y_pred = xgb_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost 정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))
```

## XGBoost 하이퍼파라미터

### 1. 주요 하이퍼파라미터
```python
# XGBoost 하이퍼파라미터 설정
xgb_params = {
    # 기본 파라미터
    'n_estimators': 100,        # 부스팅 라운드 수
    'max_depth': 6,             # 트리의 최대 깊이
    'learning_rate': 0.1,       # 학습률
    'subsample': 0.8,           # 샘플링 비율
    'colsample_bytree': 0.8,    # 특성 샘플링 비율
    
    # 정규화 파라미터
    'reg_alpha': 0,             # L1 정규화
    'reg_lambda': 1,            # L2 정규화
    'gamma': 0,                 # 최소 손실 감소
    
    # 기타 파라미터
    'random_state': 42,         # 랜덤 시드
    'n_jobs': -1,               # 병렬 처리
    'verbosity': 1              # 출력 레벨
}

# 파라미터를 적용한 모델
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)
```

### 2. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# 그리드 서치
grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")
```

## XGBoost 고급 기능

### 1. 조기 종료 (Early Stopping)
```python
# 조기 종료를 사용한 모델
xgb_early_stop = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    early_stopping_rounds=10,
    eval_metric='logloss'
)

# 검증 세트 분할
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 조기 종료와 함께 학습
xgb_early_stop.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"최적 라운드: {xgb_early_stop.best_iteration}")
```

### 2. 커스텀 목적 함수
```python
# 커스텀 목적 함수 정의
def custom_objective(y_pred, y_true):
    """커스텀 목적 함수"""
    y_true = y_true.get_label()
    grad = 2 * (y_pred - y_true)
    hess = 2 * np.ones_like(y_pred)
    return grad, hess

# 커스텀 목적 함수를 사용한 모델
xgb_custom = xgb.XGBRegressor(
    n_estimators=100,
    objective=custom_objective
)
```

### 3. 특성 중요도
```python
# 특성 중요도 계산
feature_importance = xgb_model.feature_importances_

# 특성 중요도 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('XGBoost 특성 중요도')
plt.xlabel('특성 인덱스')
plt.ylabel('중요도')
plt.show()

# 특성 중요도 순으로 정렬
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("특성 중요도 (상위 10개):")
print(importance_df.head(10))
```

## XGBoost 성능 최적화

### 1. 메모리 최적화
```python
# 메모리 효율적인 설정
xgb_memory_opt = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',          # 히스토그램 기반 방법
    grow_policy='lossguide',     # 손실 가이드 정책
    max_leaves=31,               # 최대 리프 수
    max_bin=256                  # 최대 빈 수
)
```

### 2. 병렬 처리
```python
# 병렬 처리 설정
xgb_parallel = xgb.XGBClassifier(
    n_estimators=100,
    n_jobs=-1,                   # 모든 CPU 코어 사용
    tree_method='gpu_hist',      # GPU 사용 (GPU가 있는 경우)
    gpu_id=0                     # GPU ID
)
```

### 3. 캐싱 최적화
```python
# DMatrix를 사용한 최적화
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# DMatrix를 사용한 학습
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'logloss'
}

xgb_model_dmatrix = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)
```

## 실무 적용 예시

### 1. 분류 문제
```python
# 분류 문제 예시
from sklearn.datasets import load_breast_cancer

# 유방암 데이터 로드
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 분류 모델
xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 모델 학습
xgb_classifier.fit(X_train, y_train)

# 예측
y_pred = xgb_classifier.predict(X_test)
y_pred_proba = xgb_classifier.predict_proba(X_test)

# 성능 평가
from sklearn.metrics import roc_auc_score, precision_recall_curve

auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"AUC 점수: {auc_score:.4f}")

# 정밀도-재현율 곡선
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(recall, precision)
plt.xlabel('재현율')
plt.ylabel('정밀도')
plt.title('정밀도-재현율 곡선')
plt.grid(True)
plt.show()
```

### 2. 회귀 문제
```python
# 회귀 문제 예시
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

# 보스턴 주택 가격 데이터 로드
boston = load_boston()
X, y = boston.data, boston.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 회귀 모델
xgb_regressor = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 모델 학습
xgb_regressor.fit(X_train, y_train)

# 예측
y_pred = xgb_regressor.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"평균제곱오차: {mse:.4f}")
print(f"결정계수: {r2:.4f}")

# 예측 vs 실제 값 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('XGBoost 회귀 예측 결과')
plt.grid(True)
plt.show()
```

### 3. 불균형 데이터 처리
```python
# 불균형 데이터 처리
from sklearn.utils.class_weight import compute_class_weight

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 가중치를 적용한 XGBoost
xgb_weighted = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],  # 양성 클래스 가중치
    random_state=42
)

xgb_weighted.fit(X_train, y_train)
y_pred_weighted = xgb_weighted.predict(X_test)

# 성능 비교
from sklearn.metrics import f1_score

f1_original = f1_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred_weighted)

print(f"원본 모델 F1 점수: {f1_original:.4f}")
print(f"가중치 적용 모델 F1 점수: {f1_weighted:.4f}")
```

## XGBoost 모델 해석

### 1. SHAP 값
```python
import shap

# SHAP 설명자 생성
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# SHAP 요약 플롯
shap.summary_plot(shap_values, X_test, show=False)
plt.title('XGBoost SHAP 요약 플롯')
plt.show()

# SHAP 의존성 플롯
shap.dependence_plot(0, shap_values, X_test, show=False)
plt.title('XGBoost SHAP 의존성 플롯')
plt.show()
```

### 2. 부분 의존성 플롯
```python
from sklearn.inspection import plot_partial_dependence

# 부분 의존성 플롯
plot_partial_dependence(xgb_model, X_test, features=[0, 1], grid_resolution=20)
plt.suptitle('XGBoost 부분 의존성 플롯')
plt.show()
```

## 주의사항 및 모범 사례

### 1. 과적합 방지
- 적절한 `max_depth` 설정
- `learning_rate` 조정
- 조기 종료 활용
- 교차 검증 사용

### 2. 하이퍼파라미터 튜닝
- 그리드 서치 또는 랜덤 서치 사용
- 교차 검증으로 성능 평가
- 검증 세트로 최종 모델 선택

### 3. 메모리 관리
- 대용량 데이터는 배치 처리
- `tree_method='hist'` 사용
- 적절한 `max_bin` 설정

## 마무리

XGBoost는 높은 성능과 빠른 속도로 유명한 그래디언트 부스팅 라이브러리입니다. 적절한 하이퍼파라미터 설정과 성능 최적화를 통해 실무에서 뛰어난 예측 성능을 달성할 수 있습니다. 조기 종료, 특성 중요도, SHAP 값 등 다양한 기능을 활용하여 모델의 성능과 해석성을 모두 확보할 수 있습니다.
