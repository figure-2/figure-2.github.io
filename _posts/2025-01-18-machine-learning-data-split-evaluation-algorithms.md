---
title: 머신러닝 데이터분리, 평가척도, 알고리즘 - 실전 머신러닝의 핵심
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- 머신러닝
- 데이터분리
- 평가척도
- 알고리즘
- 선형회귀
- 결정트리
- KNN
- 로지스틱회귀
toc: true
date: 2023-09-13 11:00:00 +0900
comments: false
mermaid: true
math: true
---
# 머신러닝 데이터분리, 평가척도, 알고리즘 - 실전 머신러닝의 핵심

## 개요

실전 머신러닝의 핵심 요소들을 학습합니다:

- **데이터 분리**: 훈련/테스트 데이터 분할과 계층적 샘플링
- **평가 척도**: 회귀와 분류 문제의 성능 평가 지표
- **핵심 알고리즘**: 선형회귀, 결정트리, KNN, 로지스틱회귀
- **실무 적용**: 각 알고리즘의 특성과 활용 시나리오

## 1. 데이터 분리 (데이터셋 분할)

### 1-1. 기본 데이터 분리

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== 데이터 분리 기본 ===")

# 샘플 데이터 생성
x_data = np.array([
    [1, 1], [2, 2], [3, 4], [4, 5], [5, 5],
    [6, 5], [7, 9], [8, 10], [9, 12], [10, 2],
    [11, 10], [12, 4]
])
y_data = np.array([3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6])

print(f"전체 데이터 크기: {x_data.shape}")
print(f"레이블 크기: {y_data.shape}")

# 기본 데이터 분할 (70% 훈련, 30% 테스트)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, 
    test_size=0.3, 
    random_state=777
)

print(f"훈련 데이터 크기: {x_train.shape}")
print(f"테스트 데이터 크기: {x_test.shape}")
print(f"훈련 레이블 크기: {y_train.shape}")
print(f"테스트 레이블 크기: {y_test.shape}")
```

### 1-2. 계층적 샘플링 (Stratified Sampling)

```python
print("=== 계층적 샘플링 ===")

# 분류 문제용 데이터
x_data_class = np.array([
    [2, 1], [3, 2], [3, 4], [5, 5], [7, 5], [2, 5],
    [8, 9], [9, 10], [6, 12], [9, 2], [6, 10], [2, 4]
])
y_data_class = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0])

# 계층적 샘플링 적용
x_train, x_test, y_train, y_test = train_test_split(
    x_data_class, y_data_class, 
    test_size=0.3, 
    random_state=777,
    stratify=y_data_class  # 클래스 비율 유지
)

print(f"원본 클래스 분포: {np.bincount(y_data_class)}")
print(f"훈련 클래스 분포: {np.bincount(y_train)}")
print(f"테스트 클래스 분포: {np.bincount(y_test)}")
```

## 2. 평가 척도

### 2-1. 회귀 평가 지표

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

print("=== 회귀 평가 지표 ===")

# 회귀 모델 학습
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# 1. R² (결정계수)
r2 = r2_score(y_test, y_predict)
print(f"R² Score: {r2:.4f}")

# 2. MSE (평균 제곱 오차)
mse = mean_squared_error(y_test, y_predict)
print(f"MSE: {mse:.4f}")

# 3. RMSE (평균 제곱근 오차)
rmse = mean_squared_error(y_test, y_predict, squared=False)
print(f"RMSE: {rmse:.4f}")

# 4. MAE (평균 절대 오차)
mae = mean_absolute_error(y_test, y_predict)
print(f"MAE: {mae:.4f}")

# 5. MSLE (평균 제곱 로그 오차)
msle = mean_squared_log_error(y_test, y_predict)
print(f"MSLE: {msle:.4f}")

# 통합 평가 함수
def evaluate_regression(y_test, y_predict):
    """회귀 모델 통합 평가"""
    mse = mean_squared_error(y_test, y_predict, squared=True)
    rmse = mean_squared_error(y_test, y_predict, squared=False)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    
    print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

print("\n통합 평가:")
evaluate_regression(y_test, y_predict)
```

### 2-2. 분류 평가 지표

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("=== 분류 평가 지표 ===")

# 분류 모델 학습
x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
    x_data_class, y_data_class, 
    test_size=0.3, 
    random_state=777,
    stratify=y_data_class
)

model = LogisticRegression()
model.fit(x_train_class, y_train_class)
y_predict_class = model.predict(x_test_class)

# 1. 정확도 (Accuracy)
accuracy = accuracy_score(y_test_class, y_predict_class)
print(f"정확도: {accuracy:.4f}")

# 2. 정밀도 (Precision)
precision = precision_score(y_test_class, y_predict_class)
print(f"정밀도: {precision:.4f}")

# 3. 재현율 (Recall)
recall = recall_score(y_test_class, y_predict_class)
print(f"재현율: {recall:.4f}")

# 4. F1 Score
f1 = f1_score(y_test_class, y_predict_class)
print(f"F1 Score: {f1:.4f}")

# 5. 혼동 행렬 (Confusion Matrix)
cm = confusion_matrix(y_test_class, y_predict_class)
print(f"\n혼동 행렬:")
print(cm)

# 6. 분류 보고서
print(f"\n분류 보고서:")
print(classification_report(y_test_class, y_predict_class, target_names=['Fail', 'Pass']))
```

### 2-3. 평가 지표 해석

```python
print("=== 평가 지표 해석 ===")

print("회귀 평가 지표:")
print("- R²: 1에 가까울수록 좋음 (설명력)")
print("- MSE, RMSE, MAE: 작을수록 좋음 (오차)")
print("- MSLE: 로그 스케일에서의 오차")

print("\n분류 평가 지표:")
print("- 정확도: 전체 예측 중 맞춘 비율")
print("- 정밀도: 양성 예측 중 실제 양성 비율")
print("- 재현율: 실제 양성 중 예측한 양성 비율")
print("- F1 Score: 정밀도와 재현율의 조화평균")

print("\n혼동 행렬:")
print("- TN (True Negative): 정답 (0,0)")
print("- TP (True Positive): 정답 (1,1)")
print("- FP (False Positive): 오답 (0,1)")
print("- FN (False Negative): 오답 (1,0)")
```

## 3. 핵심 알고리즘

### 3-1. 선형 회귀 (Linear Regression)

```python
print("=== 선형 회귀 ===")

# 회귀 데이터 준비
x_data_reg = np.array([
    [1, 1], [2, 2], [3, 4], [4, 5], [5, 5],
    [6, 5], [7, 9], [8, 10], [9, 12], [10, 2],
    [11, 10], [12, 4]
])
y_data_reg = np.array([3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data_reg, y_data_reg, 
    test_size=0.3, 
    random_state=777
)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(x_train, y_train)

# 모델 평가
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print(f"훈련 데이터 R²: {train_score:.4f}")
print(f"테스트 데이터 R²: {test_score:.4f}")

# 회귀 계수 확인
print(f"회귀 계수: {model.coef_}")
print(f"절편: {model.intercept_:.4f}")

# 예측
x_new = np.array([[4, 6]])
y_predict = model.predict(x_new)
print(f"새로운 데이터 예측값: {y_predict[0]:.4f}")

# 상세 평가
y_predict_test = model.predict(x_test)
evaluate_regression(y_test, y_predict_test)
```

### 3-2. 결정트리 (Decision Tree)

```python
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

print("=== 결정트리 ===")

# Iris 데이터셋 사용
iris_data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, 
    test_size=0.2, 
    random_state=11
)

# 다양한 하이퍼파라미터 설정
models = {
    'max_depth=3': DecisionTreeClassifier(random_state=156, max_depth=3),
    'min_samples_leaf=5': DecisionTreeClassifier(random_state=156, min_samples_leaf=5),
    'combined': DecisionTreeClassifier(random_state=111, min_samples_leaf=2, max_depth=4)
}

print("결정트리 하이퍼파라미터 비교:")
for name, model in models.items():
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"{name}: 훈련={train_score:.4f}, 테스트={test_score:.4f}")

# 최적 모델 선택 및 시각화
best_model = DecisionTreeClassifier(random_state=156, max_depth=3)
best_model.fit(x_train, y_train)

plt.figure(figsize=(15, 10))
plot_tree(best_model, filled=True, feature_names=iris_data.feature_names)
plt.title('Decision Tree Visualization')
plt.show()

print("\n결정트리 특징:")
print("- 장점: 해석이 직관적, 피처 스케일링 불필요")
print("- 단점: 과적합 위험, 하이퍼파라미터 튜닝 필요")
```

### 3-3. K-최근접 이웃 (K-Nearest Neighbors)

```python
print("=== K-최근접 이웃 (KNN) ===")

# 분류 데이터 준비
x_data_knn = np.array([
    [2, 1], [3, 2], [3, 4], [5, 5], [7, 5], [2, 5],
    [8, 9], [9, 10], [6, 12], [9, 2], [6, 10], [2, 4]
])
y_data_knn = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data_knn, y_data_knn, 
    test_size=0.3, 
    random_state=777, 
    stratify=y_data_knn
)

# 다양한 K 값 테스트
k_values = [3, 5, 7, 9]
labels = ['fail', 'pass']

print("K 값에 따른 성능 비교:")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"K={k}: 훈련={train_score:.4f}, 테스트={test_score:.4f}")

# 최적 K 값으로 예측
best_k = 7
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(x_train, y_train)

# 예측 및 확률
x_new = np.array([[4, 6]])
y_predict = model.predict(x_new)
y_proba = model.predict_proba(x_new)

predicted_label = labels[y_predict[0]]
confidence = y_proba[0][y_predict[0]]

print(f"\n예측 결과: {predicted_label}")
print(f"신뢰도: {confidence:.4f}")

print("\nKNN 특징:")
print("- 장점: 간단하고 직관적, 비모수적 방법")
print("- 단점: 계산 비용 높음, 차원의 저주")
```

### 3-4. 로지스틱 회귀 (Logistic Regression)

```python
print("=== 로지스틱 회귀 ===")

# 분류 데이터 준비 (낮잠시간, 공부시간)
x_data_log = np.array([
    [2, 1], [3, 2], [3, 4], [5, 5], [7, 5], [2, 5],
    [8, 9], [9, 10], [6, 12], [9, 2], [6, 10], [2, 4]
])
y_data_log = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data_log, y_data_log, 
    test_size=0.3, 
    random_state=777, 
    stratify=y_data_log
)

# 모델 생성 및 학습
model = LogisticRegression()
model.fit(x_train, y_train)

# 모델 평가
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print(f"훈련 정확도: {train_score:.4f}")
print(f"테스트 정확도: {test_score:.4f}")

# 예측 및 확률
x_new = np.array([[4, 6]])
y_predict = model.predict(x_new)
y_proba = model.predict_proba(x_new)

predicted_label = labels[y_predict[0]]
confidence = y_proba[0][y_predict[0]]

print(f"\n예측 결과: {predicted_label}")
print(f"신뢰도: {confidence:.4f}")
print(f"확률 분포: {y_proba[0]}")

# 상세 평가
y_predict_test = model.predict(x_test)
print(f"\n상세 평가:")
print(classification_report(y_test, y_predict_test, target_names=['Fail', 'Pass']))

print("\n로지스틱 회귀 특징:")
print("- 시그모이드 함수 사용")
print("- 확률 기반 분류")
print("- 선형 결정 경계")
print("- 해석 가능한 계수")
```

## 4. 알고리즘 비교 및 선택

### 4-1. 알고리즘 성능 비교

```python
print("=== 알고리즘 성능 비교 ===")

# 동일한 데이터로 모든 알고리즘 테스트
from sklearn.datasets import make_classification

# 샘플 데이터 생성
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 알고리즘 정의
algorithms = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# 성능 비교
results = {}
for name, model in algorithms.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    results[name] = {'train': train_score, 'test': test_score}

# 결과 출력
print("알고리즘별 성능 비교:")
for name, scores in results.items():
    print(f"{name}:")
    print(f"  훈련 정확도: {scores['train']:.4f}")
    print(f"  테스트 정확도: {scores['test']:.4f}")
    print(f"  과적합 정도: {scores['train'] - scores['test']:.4f}")
    print()
```

### 4-2. 알고리즘 선택 가이드

```python
print("=== 알고리즘 선택 가이드 ===")

print("1. 선형 회귀 (Linear Regression)")
print("   - 사용 시기: 연속형 타겟 변수, 선형 관계")
print("   - 장점: 해석 가능, 빠른 학습")
print("   - 단점: 비선형 관계 처리 어려움")

print("\n2. 로지스틱 회귀 (Logistic Regression)")
print("   - 사용 시기: 이진 분류, 확률 예측 필요")
print("   - 장점: 해석 가능, 확률 출력")
print("   - 단점: 선형 관계 가정")

print("\n3. 결정트리 (Decision Tree)")
print("   - 사용 시기: 해석 가능성 중요, 비선형 관계")
print("   - 장점: 직관적, 피처 스케일링 불필요")
print("   - 단점: 과적합 위험")

print("\n4. K-최근접 이웃 (KNN)")
print("   - 사용 시기: 지역적 패턴, 비모수적 방법")
print("   - 장점: 간단, 직관적")
print("   - 단점: 계산 비용, 차원의 저주")
```

### 4-3. 실무 적용 시나리오

```python
print("=== 실무 적용 시나리오 ===")

print("1. 금융 분야")
print("   - 신용 평가: 로지스틱 회귀 (해석 가능성)")
print("   - 주가 예측: 선형 회귀 (시계열 특성)")
print("   - 이상 거래 탐지: 결정트리 (규칙 기반)")

print("\n2. 의료 분야")
print("   - 질병 진단: 로지스틱 회귀 (확률 출력)")
print("   - 약물 효과 예측: 선형 회귀")
print("   - 환자 분류: KNN (유사 사례 기반)")

print("\n3. 마케팅 분야")
print("   - 고객 세분화: 결정트리 (해석 가능)")
print("   - 구매 예측: 로지스틱 회귀")
print("   - 추천 시스템: KNN (유사 고객)")

print("\n4. 제조업")
print("   - 품질 예측: 선형 회귀")
print("   - 불량품 분류: 결정트리")
print("   - 유지보수 예측: 로지스틱 회귀")
```

## 5. 모델 성능 개선 전략

### 5-1. 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

print("=== 하이퍼파라미터 튜닝 ===")

# 결정트리 하이퍼파라미터 튜닝
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 예측
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"테스트 점수: {test_score:.4f}")
```

### 5-2. 교차 검증

```python
from sklearn.model_selection import cross_val_score

print("=== 교차 검증 ===")

# 5-fold 교차 검증
cv_scores = cross_val_score(
    LogisticRegression(),
    X_train, y_train,
    cv=5,
    scoring='accuracy'
)

print(f"교차 검증 점수: {cv_scores}")
print(f"평균 점수: {cv_scores.mean():.4f}")
print(f"표준편차: {cv_scores.std():.4f}")
```

## 마무리

머신러닝의 핵심 요소들을 종합적으로 학습했습니다:

### 핵심 학습 내용
- **데이터 분리**: 훈련/테스트 분할과 계층적 샘플링의 중요성
- **평가 척도**: 회귀와 분류 문제의 적절한 성능 지표
- **핵심 알고리즘**: 선형회귀, 결정트리, KNN, 로지스틱회귀의 특성
- **실무 적용**: 각 알고리즘의 장단점과 활용 시나리오

### 실무 적용
- **알고리즘 선택**: 문제 유형과 요구사항에 따른 적절한 알고리즘 선택
- **성능 평가**: 다양한 평가 지표를 통한 종합적 성능 분석
- **하이퍼파라미터 튜닝**: 체계적인 모델 최적화
- **교차 검증**: 안정적인 성능 평가

이러한 기초 알고리즘들을 바탕으로 더 복잡한 앙상블 방법과 딥러닝 모델로 발전시킬 수 있습니다.
