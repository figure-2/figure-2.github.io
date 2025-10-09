---
title: LightGBM (Light Gradient Boosting Machine)
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- LightGBM
- 그라디언트부스팅
- 머신러닝
- XGBoost
- 성능최적화
- 메모리효율
toc: true
date: 2023-09-15 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# LightGBM (Light Gradient Boosting Machine)

> 230926 학습한 내용 정리

## LightGBM 개요

### 정의
- **Light Gradient Boosting Machine**의 줄임말
- Microsoft에서 개발한 그래디언트 부스팅 프레임워크
- 빠른 속도와 낮은 메모리 사용량이 특징

### 특징
- **빠른 속도**: XGBoost보다 빠른 학습 속도
- **낮은 메모리**: 효율적인 메모리 사용
- **높은 정확도**: 높은 예측 정확도
- **대용량 데이터**: 대용량 데이터 처리에 최적화

### 장점
- **속도**: 빠른 학습 및 예측 속도
- **메모리 효율**: 낮은 메모리 사용량
- **정확도**: 높은 예측 정확도
- **GPU 지원**: GPU 가속 지원

## LightGBM 설치 및 기본 사용법

### 1. 설치
```bash
# pip 설치
pip install lightgbm

# conda 설치
conda install -c conda-forge lightgbm
```

### 2. 기본 사용법
```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 샘플 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 모델 생성
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 모델 학습
lgb_model.fit(X_train, y_train)

# 예측
y_pred = lgb_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM 정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))
```

## LightGBM 하이퍼파라미터

### 1. 주요 하이퍼파라미터
```python
# LightGBM 하이퍼파라미터 설정
lgb_params = {
    # 기본 파라미터
    'n_estimators': 100,        # 부스팅 라운드 수
    'max_depth': 6,             # 트리의 최대 깊이
    'learning_rate': 0.1,       # 학습률
    'subsample': 0.8,           # 샘플링 비율
    'colsample_bytree': 0.8,    # 특성 샘플링 비율
    
    # LightGBM 특화 파라미터
    'num_leaves': 31,           # 최대 리프 수
    'min_child_samples': 20,    # 리프당 최소 샘플 수
    'min_child_weight': 0.001,  # 리프당 최소 가중치
    'reg_alpha': 0,             # L1 정규화
    'reg_lambda': 0,            # L2 정규화
    
    # 기타 파라미터
    'random_state': 42,         # 랜덤 시드
    'n_jobs': -1,               # 병렬 처리
    'verbosity': -1             # 출력 레벨
}

# 파라미터를 적용한 모델
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train)
```

### 2. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [10, 20, 30]
}

# 그리드 서치
grid_search = GridSearchCV(
    lgb.LGBMClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")
```

## LightGBM 고급 기능

### 1. 조기 종료 (Early Stopping)
```python
# 조기 종료를 사용한 모델
lgb_early_stop = lgb.LGBMClassifier(
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
lgb_early_stop.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

print(f"최적 라운드: {lgb_early_stop.best_iteration_}")
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
lgb_custom = lgb.LGBMRegressor(
    n_estimators=100,
    objective=custom_objective
)
```

### 3. 특성 중요도
```python
# 특성 중요도 계산
feature_importance = lgb_model.feature_importances_

# 특성 중요도 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('LightGBM 특성 중요도')
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

## LightGBM 성능 최적화

### 1. 메모리 최적화
```python
# 메모리 효율적인 설정
lgb_memory_opt = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.8,        # 특성 샘플링
    bagging_fraction=0.8,        # 데이터 샘플링
    bagging_freq=5,              # 배깅 빈도
    verbose=-1
)
```

### 2. 병렬 처리
```python
# 병렬 처리 설정
lgb_parallel = lgb.LGBMClassifier(
    n_estimators=100,
    n_jobs=-1,                   # 모든 CPU 코어 사용
    device='gpu',                # GPU 사용 (GPU가 있는 경우)
    gpu_platform_id=0,           # GPU 플랫폼 ID
    gpu_device_id=0              # GPU 디바이스 ID
)
```

### 3. Dataset 객체 사용
```python
# Dataset 객체를 사용한 최적화
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 파라미터 설정
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# 모델 학습
lgb_model_dataset = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
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

# LightGBM 분류 모델
lgb_classifier = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 모델 학습
lgb_classifier.fit(X_train, y_train)

# 예측
y_pred = lgb_classifier.predict(X_test)
y_pred_proba = lgb_classifier.predict_proba(X_test)

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

# LightGBM 회귀 모델
lgb_regressor = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 모델 학습
lgb_regressor.fit(X_train, y_train)

# 예측
y_pred = lgb_regressor.predict(X_test)

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
plt.title('LightGBM 회귀 예측 결과')
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

# 가중치를 적용한 LightGBM
lgb_weighted = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    class_weight='balanced',     # 클래스 가중치 자동 조정
    random_state=42
)

lgb_weighted.fit(X_train, y_train)
y_pred_weighted = lgb_weighted.predict(X_test)

# 성능 비교
from sklearn.metrics import f1_score

f1_original = f1_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred_weighted)

print(f"원본 모델 F1 점수: {f1_original:.4f}")
print(f"가중치 적용 모델 F1 점수: {f1_weighted:.4f}")
```

## LightGBM 모델 해석

### 1. SHAP 값
```python
import shap

# SHAP 설명자 생성
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)

# SHAP 요약 플롯
shap.summary_plot(shap_values, X_test, show=False)
plt.title('LightGBM SHAP 요약 플롯')
plt.show()

# SHAP 의존성 플롯
shap.dependence_plot(0, shap_values, X_test, show=False)
plt.title('LightGBM SHAP 의존성 플롯')
plt.show()
```

### 2. 부분 의존성 플롯
```python
from sklearn.inspection import plot_partial_dependence

# 부분 의존성 플롯
plot_partial_dependence(lgb_model, X_test, features=[0, 1], grid_resolution=20)
plt.suptitle('LightGBM 부분 의존성 플롯')
plt.show()
```

## XGBoost vs LightGBM 비교

### 1. 성능 비교
```python
import xgboost as xgb
import time

# XGBoost 모델
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# LightGBM 모델
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# 학습 시간 측정
start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

start_time = time.time()
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start_time

# 예측 시간 측정
start_time = time.time()
xgb_pred = xgb_model.predict(X_test)
xgb_pred_time = time.time() - start_time

start_time = time.time()
lgb_pred = lgb_model.predict(X_test)
lgb_pred_time = time.time() - start_time

# 성능 비교
xgb_accuracy = accuracy_score(y_test, xgb_pred)
lgb_accuracy = accuracy_score(y_test, lgb_pred)

print("성능 비교:")
print(f"XGBoost - 정확도: {xgb_accuracy:.4f}, 학습 시간: {xgb_time:.4f}s, 예측 시간: {xgb_pred_time:.4f}s")
print(f"LightGBM - 정확도: {lgb_accuracy:.4f}, 학습 시간: {lgb_time:.4f}s, 예측 시간: {lgb_pred_time:.4f}s")
```

### 2. 메모리 사용량 비교
```python
import psutil
import os

def get_memory_usage():
    """메모리 사용량 반환"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# 메모리 사용량 측정
memory_before = get_memory_usage()

# XGBoost 메모리 사용량
xgb_model.fit(X_train, y_train)
xgb_memory = get_memory_usage() - memory_before

# LightGBM 메모리 사용량
lgb_model.fit(X_train, y_train)
lgb_memory = get_memory_usage() - memory_before

print(f"XGBoost 메모리 사용량: {xgb_memory:.2f} MB")
print(f"LightGBM 메모리 사용량: {lgb_memory:.2f} MB")
```

## 주의사항 및 모범 사례

### 1. 과적합 방지
- 적절한 `num_leaves` 설정
- `learning_rate` 조정
- 조기 종료 활용
- 교차 검증 사용

### 2. 하이퍼파라미터 튜닝
- 그리드 서치 또는 랜덤 서치 사용
- 교차 검증으로 성능 평가
- 검증 세트로 최종 모델 선택

### 3. 메모리 관리
- 대용량 데이터는 배치 처리
- `feature_fraction`과 `bagging_fraction` 조정
- 적절한 `num_leaves` 설정

## 마무리

LightGBM은 빠른 속도와 낮은 메모리 사용량으로 유명한 그래디언트 부스팅 라이브러리입니다. XGBoost보다 빠른 학습 속도와 효율적인 메모리 사용으로 대용량 데이터 처리에 최적화되어 있습니다. 적절한 하이퍼파라미터 설정과 성능 최적화를 통해 실무에서 뛰어난 예측 성능을 달성할 수 있습니다. 조기 종료, 특성 중요도, SHAP 값 등 다양한 기능을 활용하여 모델의 성능과 해석성을 모두 확보할 수 있습니다.
