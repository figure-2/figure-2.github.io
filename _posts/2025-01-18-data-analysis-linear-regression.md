---
title: 선형회귀분석 - 머신러닝의 기초와 농어 무게 예측
categories:
- 1.TIL
- 1-7.DATA_ANALYSIS
tags:
- 머신러닝
- 선형회귀
- 데이터분리
- scikit-learn
- 농어예측
- 회귀분석
toc: true
date: 2023-09-23 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 선형회귀분석 - 머신러닝의 기초와 농어 무게 예측

## 개요

머신러닝의 기초인 선형회귀분석을 통해 데이터 예측의 핵심 원리를 학습합니다:

- **머신러닝 이론**: 지도학습, 비지도학습, 강화학습
- **데이터 분리**: 훈련 데이터와 테스트 데이터 분할
- **선형회귀**: 농어 길이로 무게 예측 실습
- **모델 평가**: 성능 지표와 과적합/과소적합 이해

## 1. 머신러닝 이론

### 1-1. AI 용어 및 종류

```python
# 머신러닝의 기본 개념
print("=== AI 용어 정리 ===")
print("인공지능: 사람처럼 똑똑한 기계")
print("머신러닝: 인공지능을 구현하는데 성공한 방법, 기계학습")
print("딥러닝: 머신러닝보다 더 똑똑한 인공지능을 구현하는데 성공한 방법")
print("  - 인간의 뉴런과 비슷한 인공신경망 기반")
```

### 1-2. 머신러닝 분류

```python
# 머신러닝 학습 방법 분류
print("=== 머신러닝 학습 방법 ===")

print("1. 지도학습 (Supervised Learning)")
print("   - 정답을 알려주며 학습시키는 방법")
print("   - 분류: 이메일 스팸/정상 분류")
print("   - 회귀: 집값 예측, 농어 무게 예측")

print("\n2. 비지도학습 (Unsupervised Learning)")
print("   - 정답을 알려주지 않고 규칙을 스스로 발견")
print("   - 군집화: 고객 세분화")
print("   - 연관규칙: 장바구니 분석")

print("\n3. 강화학습 (Reinforcement Learning)")
print("   - 실패와 성공의 과정을 반복하며 학습")
print("   - 보상 기반 학습 (알파고)")
print("   - 게임 AI, 자율주행")
```

### 1-3. 머신러닝 기본 개념

```python
# 데이터 속성과 워크플로우
print("=== 머신러닝 기본 개념 ===")

print("데이터 속성:")
print("- 피처(Feature): 레이블을 제외한 나머지 속성")
print("- 레이블(Label): 학습을 위한 정답 데이터")

print("\n머신러닝 워크플로우:")
print("1. 데이터 수집 및 전처리")
print("2. 데이터 분할 (훈련/테스트)")
print("3. 모델 학습")
print("4. 모델 평가")
print("5. 모델 예측")

print("\n과적합 vs 과소적합:")
print("- 과소적합(Underfitting): 학습이 부족한 상태")
print("- 과적합(Overfitting): 학습 데이터에 과도하게 적합")
```

## 2. 데이터 분리 (데이터셋 분할)

### 2-1. 기본 데이터 분리

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 샘플 데이터 생성
x_data = np.array([
    [1, 1], [2, 2], [3, 4], [4, 5], [5, 5],
    [6, 5], [7, 9], [8, 10], [9, 12], [10, 2],
    [11, 10], [12, 4]
])
y_data = np.array([3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6])

print("=== 데이터 분리 ===")
print(f"전체 데이터 크기: {x_data.shape}")
print(f"레이블 크기: {y_data.shape}")

# 데이터 분할 (70% 훈련, 30% 테스트)
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

### 2-2. 계층적 샘플링 (Stratified Sampling)

```python
# 분류 문제에서 클래스 비율 유지
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

print("=== 계층적 샘플링 결과 ===")
print(f"원본 클래스 분포: {np.bincount(y_data_class)}")
print(f"훈련 클래스 분포: {np.bincount(y_train)}")
print(f"테스트 클래스 분포: {np.bincount(y_test)}")
```

## 3. 회귀 모델

### 3-1. 선형 회귀 모델

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(x_train, y_train)

# 모델 성능 평가
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("=== 선형 회귀 모델 결과 ===")
print(f"훈련 데이터 정확도: {train_score:.4f}")
print(f"테스트 데이터 정확도: {test_score:.4f}")

# 예측
x_new = np.array([[4, 6]])
y_predict = model.predict(x_new)
print(f"새로운 데이터 예측값: {y_predict[0]:.4f}")

# 회귀 계수 확인
print(f"회귀 계수: {model.coef_}")
print(f"절편: {model.intercept_:.4f}")
```

### 3-2. 분류 모델 (로지스틱 회귀)

```python
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델
x_data_binary = np.array([
    [2, 1], [3, 2], [3, 4], [5, 5], [7, 5], [2, 5],
    [8, 9], [9, 10], [6, 12], [9, 2], [6, 10], [2, 4]
])
y_data_binary = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data_binary, y_data_binary, 
    test_size=0.3, 
    random_state=777, 
    stratify=y_data_binary
)

# 로지스틱 회귀 모델
model = LogisticRegression()
model.fit(x_train, y_train)

# 예측 및 확률
x_new = np.array([[4, 6]])
y_predict = model.predict(x_new)
y_proba = model.predict_proba(x_new)

labels = ['fail', 'pass']
predicted_label = labels[y_predict[0]]
confidence = y_proba[0][y_predict[0]]

print("=== 로지스틱 회귀 모델 결과 ===")
print(f"예측 결과: {predicted_label}")
print(f"신뢰도: {confidence:.4f}")
print(f"각 클래스 확률: {y_proba[0]}")
```

### 3-3. 다중 분류 모델

```python
# 다중 분류 데이터
x_data_multi = np.array([
    [2, 1], [3, 2], [3, 4], [5, 5], [7, 5], [2, 5],
    [8, 9], [9, 10], [6, 12], [9, 2], [6, 10], [2, 4]
])
y_data_multi = np.array([2, 2, 2, 1, 1, 2, 0, 0, 0, 1, 0, 2])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data_multi, y_data_multi, 
    test_size=0.3, 
    random_state=777, 
    stratify=y_data_multi
)

# 다중 분류 모델
model = LogisticRegression()
model.fit(x_train, y_train)

# 예측
x_new = np.array([[4, 6]])
y_predict = model.predict(x_new)
y_proba = model.predict_proba(x_new)

labels = ['A', 'B', 'C']
predicted_label = labels[y_predict[0]]
confidence = y_proba[0][y_predict[0]]

print("=== 다중 분류 모델 결과 ===")
print(f"예측 결과: {predicted_label}")
print(f"신뢰도: {confidence:.4f}")
print(f"각 클래스 확률: {dict(zip(labels, y_proba[0]))}")
```

## 4. 회귀 분석 이론

### 4-1. 회귀의 핵심 개념

```python
print("=== 회귀 분석 핵심 개념 ===")
print("회귀 예측의 목표:")
print("- 주어진 피처와 레이블을 기반으로 학습")
print("- 최적의 회귀 계수(W1, W2, ..., Wn) 찾기")
print("- 새로운 데이터에 대한 정확한 예측")

print("\n회귀 모델 종류:")
print("1. 선형 회귀: 1차 방정식, 직선 관계")
print("2. 다항 회귀: 2/3차 방정식, 곡선 관계")
```

### 4-2. 모델 학습 과정

```python
print("=== 모델 학습 과정 ===")
print("1. TrainData를 model에 input")
print("2. 모델 learning")
print("3. predict")
print("4. 예측과 정답 비교해서 손실함수(RSS) 계산")
print("5. 손실함수가 최소가 될 때까지 반복:")
print("   - 가중치(weight)와 절편(bias) 업데이트")
print("   - learning → predict → 손실함수 계산")
print("6. 손실함수가 최소가 되면 학습 종료")

print("\n중요한 점:")
print("- 100% 정확도는 과적합의 신호")
print("- 적절한 오차는 정상적인 모델의 특성")
```

## 5. 농어 길이로 무게 예측 실습

### 5-1. 데이터 수집 및 확인

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings(action='ignore')

# 샘플 농어 데이터 생성 (실제 데이터가 없는 경우)
np.random.seed(42)
lengths = np.random.uniform(15, 50, 100)
weights = 39.3 * lengths - 718.4 + np.random.normal(0, 50, 100)
weights = np.maximum(weights, 50)  # 최소 무게 50g

fish_df = pd.DataFrame({
    '길이': lengths,
    '무게': weights
})

print("=== 농어 데이터 정보 ===")
print(fish_df.info())
print(f"\n데이터 미리보기:")
print(fish_df.head())

print(f"\n기본 통계:")
print(fish_df.describe())
```

### 5-2. 데이터 시각화

```python
# 산점도로 길이와 무게의 관계 확인
plt.figure(figsize=(10, 6))
plt.scatter(fish_df['길이'], fish_df['무게'], alpha=0.6)
plt.xlabel('길이 (cm)')
plt.ylabel('무게 (g)')
plt.title('농어 길이와 무게의 관계')
plt.grid(True, alpha=0.3)
plt.show()

# 박스플롯으로 이상치 확인
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(fish_df['길이'])
axes[0].set_title('길이 분포')
axes[0].set_ylabel('길이 (cm)')

axes[1].boxplot(fish_df['무게'])
axes[1].set_title('무게 분포')
axes[1].set_ylabel('무게 (g)')

plt.tight_layout()
plt.show()
```

### 5-3. 단순 선형 회귀 모델

```python
# 데이터 전처리
y_data = fish_df['무게']
x_data = fish_df[['길이']]  # 2차원 배열로 변환

print(f"피처 데이터 형태: {x_data.shape}")
print(f"레이블 데이터 형태: {y_data.shape}")

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, 
    test_size=0.3, 
    random_state=42
)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(x_train, y_train)

# 모델 성능 평가
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("=== 모델 성능 ===")
print(f"훈련 데이터 R²: {train_score:.4f}")
print(f"테스트 데이터 R²: {test_score:.4f}")

# 회귀 계수 확인
print(f"\n회귀 계수 (기울기): {model.coef_[0]:.4f}")
print(f"절편: {model.intercept_:.4f}")

# 예측 방정식
print(f"\n농어 무게 예측 방정식:")
print(f"무게 = {model.coef_[0]:.2f} × 길이 + {model.intercept_:.2f}")
```

### 5-4. 모델 예측 및 평가

```python
# 새로운 농어 길이로 무게 예측
new_length = 50
x_new = np.array([[new_length]])
y_predict = model.predict(x_new)

print(f"=== 농어 무게 예측 ===")
print(f"농어 길이: {new_length}cm")
print(f"예측 무게: {y_predict[0]:.2f}g")

# 상세 성능 지표
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print(f"\n=== 상세 성능 지표 ===")
print("훈련 데이터:")
print(f"  MSE: {mean_squared_error(y_train, y_train_pred):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
print(f"  R²: {r2_score(y_train, y_train_pred):.4f}")

print("\n테스트 데이터:")
print(f"  MSE: {mean_squared_error(y_test, y_test_pred):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"  R²: {r2_score(y_test, y_test_pred):.4f}")
```

### 5-5. 회귀선 시각화

```python
# 회귀선과 함께 산점도 그리기
plt.figure(figsize=(12, 8))

# 훈련 데이터 산점도
plt.scatter(x_train, y_train, alpha=0.6, label='훈련 데이터', color='blue')
plt.scatter(x_test, y_test, alpha=0.6, label='테스트 데이터', color='red')

# 회귀선 그리기
x_range = np.linspace(fish_df['길이'].min(), fish_df['길이'].max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_range, 'g-', linewidth=2, label='회귀선')

# 예측점 표시
plt.scatter(new_length, y_predict[0], marker='^', s=100, color='orange', label=f'예측점 ({new_length}cm)')

plt.xlabel('길이 (cm)')
plt.ylabel('무게 (g)')
plt.title('농어 길이와 무게 예측 모델')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5-6. 모델 분석 및 개선 방향

```python
# 잔차 분석
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 잔차 vs 예측값
axes[0, 0].scatter(y_train_pred, residuals_train, alpha=0.6)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('예측값')
axes[0, 0].set_ylabel('잔차')
axes[0, 0].set_title('훈련 데이터 잔차 분석')
axes[0, 0].grid(True, alpha=0.3)

# 잔차 히스토그램
axes[0, 1].hist(residuals_train, bins=20, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('잔차')
axes[0, 1].set_ylabel('빈도')
axes[0, 1].set_title('훈련 데이터 잔차 분포')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q 플롯
from scipy import stats
stats.probplot(residuals_train, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q 플롯 (정규성 검정)')
axes[1, 0].grid(True, alpha=0.3)

# 실제값 vs 예측값
axes[1, 1].scatter(y_train, y_train_pred, alpha=0.6)
axes[1, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('실제값')
axes[1, 1].set_ylabel('예측값')
axes[1, 1].set_title('실제값 vs 예측값')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 모델 분석 결과
print("=== 모델 분석 결과 ===")
print("1. 과적합 여부:")
if train_score - test_score > 0.1:
    print("   - 과적합 의심: 훈련 성능이 테스트 성능보다 크게 높음")
else:
    print("   - 적절한 일반화: 훈련과 테스트 성능이 비슷함")

print("\n2. 모델 한계:")
print("   - 선형 모델의 한계: 곡선 관계를 직선으로 근사")
print("   - 음수 예측 가능성: 매우 작은 농어의 경우 음수 무게 예측")
print("   - 개선 방향: 다항 회귀, 로그 변환 등 고려 필요")
```

## 6. 고급 회귀 기법

### 6-1. 다항 회귀

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 다항 회귀 모델 (2차)
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_model.fit(x_train, y_train)

# 성능 비교
poly_train_score = poly_model.score(x_train, y_train)
poly_test_score = poly_model.score(x_test, y_test)

print("=== 다항 회귀 성능 ===")
print(f"훈련 데이터 R²: {poly_train_score:.4f}")
print(f"테스트 데이터 R²: {poly_test_score:.4f}")

# 시각화
plt.figure(figsize=(12, 6))

# 원본 데이터
plt.scatter(x_train, y_train, alpha=0.6, label='훈련 데이터')
plt.scatter(x_test, y_test, alpha=0.6, label='테스트 데이터')

# 선형 회귀선
x_range = np.linspace(fish_df['길이'].min(), fish_df['길이'].max(), 100)
y_linear = model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_linear, 'g-', linewidth=2, label='선형 회귀')

# 다항 회귀선
y_poly = poly_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_poly, 'r-', linewidth=2, label='다항 회귀 (2차)')

plt.xlabel('길이 (cm)')
plt.ylabel('무게 (g)')
plt.title('선형 vs 다항 회귀 비교')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 6-2. 정규화 회귀

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge 회귀 (L2 정규화)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

# Lasso 회귀 (L1 정규화)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(x_train, y_train)

# 성능 비교
models = {
    'Linear': model,
    'Ridge': ridge_model,
    'Lasso': lasso_model,
    'Polynomial': poly_model
}

print("=== 모델 성능 비교 ===")
for name, model_obj in models.items():
    train_score = model_obj.score(x_train, y_train)
    test_score = model_obj.score(x_test, y_test)
    print(f"{name:12} - 훈련: {train_score:.4f}, 테스트: {test_score:.4f}")
```

## 마무리

선형회귀분석을 통해 머신러닝의 핵심 원리를 학습했습니다:

### 핵심 학습 내용
- **머신러닝 이론**: 지도학습, 비지도학습, 강화학습의 차이점
- **데이터 분리**: 훈련/테스트 데이터 분할과 계층적 샘플링
- **회귀 분석**: 선형회귀의 원리와 회귀 계수 해석
- **모델 평가**: R², MSE, RMSE, MAE 등 성능 지표
- **실습 프로젝트**: 농어 길이로 무게 예측

### 실무 적용
- **과적합/과소적합**: 모델 성능 분석과 개선 방향
- **다항 회귀**: 비선형 관계 모델링
- **정규화**: Ridge, Lasso를 통한 모델 안정성 향상
- **잔차 분석**: 모델 가정 검증과 개선점 발견

선형회귀는 머신러닝의 기초이자 가장 직관적인 모델로, 복잡한 알고리즘을 이해하기 전에 반드시 익혀야 할 핵심 개념입니다.
