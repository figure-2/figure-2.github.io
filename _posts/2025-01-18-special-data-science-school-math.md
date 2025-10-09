---
title: 데이터사이언스스쿨 수학편
categories:
- 1.TIL
- 1-10.SPECIAL
tags:
- 특강
- 데이터사이언스스쿨
- 수학편
- 선형대수
- 통계학
- 확률론
- 최적화
- 미분적분
- 정보이론
- PCA
- 선형회귀
toc: true
date: 2023-08-15 14:00:00 +0900
comments: false
mermaid: true
math: true
---
# 데이터사이언스스쿨 수학편

> 230918~230921 학습한 내용 정리

## 데이터사이언스스쿨 개요

### 정의
- **데이터사이언스스쿨**: 데이터 사이언스 학습을 위한 온라인 교육 플랫폼
- **수학편**: 데이터 사이언스에 필요한 수학적 기초 지식
- **이론 학습**: 선형대수, 통계학, 확률론 등 핵심 수학 개념
- **실무 적용**: 데이터 분석에 필요한 수학적 도구들

### 특징
- **체계적 학습**: 기초부터 고급까지 단계별 학습
- **실무 중심**: 실제 데이터 분석에 필요한 수학 지식
- **이론과 실습**: 수학적 이론과 Python 구현 병행
- **무료 교육**: 온라인 무료 교육 자료 제공

## 선형대수 (Linear Algebra)

### 1. 벡터와 행렬
```python
import numpy as np
import matplotlib.pyplot as plt

# 벡터 생성
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 벡터 연산
print(f"벡터 덧셈: {v1 + v2}")
print(f"벡터 스칼라 곱: {3 * v1}")
print(f"내적: {np.dot(v1, v2)}")

# 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 연산
print(f"행렬 덧셈:\n{A + B}")
print(f"행렬 곱셈:\n{A @ B}")
print(f"행렬 전치:\n{A.T}")
```

### 2. 고유값과 고유벡터
```python
# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"고유값: {eigenvalues}")
print(f"고유벡터:\n{eigenvectors}")

# 고유값 분해
A_eigen = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print(f"원본 행렬 복원:\n{A_eigen}")
```

### 3. 특이값 분해 (SVD)
```python
# 특이값 분해
U, s, Vt = np.linalg.svd(A)

print(f"U 행렬:\n{U}")
print(f"특이값: {s}")
print(f"V^T 행렬:\n{Vt}")

# 행렬 복원
A_svd = U @ np.diag(s) @ Vt
print(f"원본 행렬 복원:\n{A_svd}")
```

## 통계학 (Statistics)

### 1. 기술통계
```python
import pandas as pd
from scipy import stats

# 샘플 데이터 생성
data = np.random.normal(100, 15, 1000)

# 기술통계 계산
print(f"평균: {np.mean(data):.2f}")
print(f"중앙값: {np.median(data):.2f}")
print(f"표준편차: {np.std(data):.2f}")
print(f"분산: {np.var(data):.2f}")
print(f"왜도: {stats.skew(data):.2f}")
print(f"첨도: {stats.kurtosis(data):.2f}")

# 분위수
print(f"25% 분위수: {np.percentile(data, 25):.2f}")
print(f"75% 분위수: {np.percentile(data, 75):.2f}")
```

### 2. 확률분포
```python
# 정규분포
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, 0, 1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='표준정규분포')
plt.xlabel('x')
plt.ylabel('확률밀도')
plt.title('정규분포')
plt.legend()
plt.grid(True)
plt.show()

# 이항분포
n, p = 10, 0.5
x_binom = np.arange(0, n+1)
y_binom = stats.binom.pmf(x_binom, n, p)

plt.figure(figsize=(10, 6))
plt.bar(x_binom, y_binom, label='이항분포')
plt.xlabel('성공 횟수')
plt.ylabel('확률')
plt.title('이항분포 (n=10, p=0.5)')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. 가설검정
```python
# t-검정
sample1 = np.random.normal(100, 15, 30)
sample2 = np.random.normal(105, 15, 30)

# 독립표본 t-검정
t_stat, p_value = stats.ttest_ind(sample1, sample2)

print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("귀무가설 기각: 두 그룹 간 유의한 차이가 있습니다.")
else:
    print("귀무가설 채택: 두 그룹 간 유의한 차이가 없습니다.")

# 카이제곱 검정
observed = np.array([[10, 20], [15, 25]])
chi2_stat, p_value = stats.chi2_contingency(observed)

print(f"카이제곱 통계량: {chi2_stat:.4f}")
print(f"p-값: {p_value:.4f}")
```

## 확률론 (Probability Theory)

### 1. 조건부 확률
```python
# 조건부 확률 계산
def conditional_probability(p_a, p_b_given_a):
    """조건부 확률 P(B|A) = P(A∩B) / P(A)"""
    return p_b_given_a

# 베이즈 정리
def bayes_theorem(p_a, p_b, p_b_given_a):
    """베이즈 정리 P(A|B) = P(B|A) * P(A) / P(B)"""
    return (p_b_given_a * p_a) / p_b

# 예시: 의료 진단
p_disease = 0.01  # 질병 유병률
p_positive_given_disease = 0.95  # 질병이 있을 때 양성 반응 확률
p_positive_given_no_disease = 0.05  # 질병이 없을 때 양성 반응 확률

p_positive = p_positive_given_disease * p_disease + p_positive_given_no_disease * (1 - p_disease)
p_disease_given_positive = bayes_theorem(p_disease, p_positive, p_positive_given_disease)

print(f"양성 반응을 받았을 때 실제 질병에 걸릴 확률: {p_disease_given_positive:.4f}")
```

### 2. 확률변수와 기댓값
```python
# 이산확률변수
x_values = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

# 기댓값 계산
expected_value = np.sum(x_values * probabilities)
print(f"기댓값: {expected_value}")

# 분산 계산
variance = np.sum((x_values - expected_value)**2 * probabilities)
print(f"분산: {variance:.4f}")

# 연속확률변수 (정규분포)
def normal_expected_value(mu, sigma):
    """정규분포의 기댓값"""
    return mu

def normal_variance(mu, sigma):
    """정규분포의 분산"""
    return sigma**2

print(f"정규분포 기댓값: {normal_expected_value(0, 1)}")
print(f"정규분포 분산: {normal_variance(0, 1)}")
```

## 최적화 (Optimization)

### 1. 경사하강법
```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """경사하강법 구현"""
    x = x0
    history = [x]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        
        if np.linalg.norm(x_new - x) < tolerance:
            break
            
        x = x_new
        history.append(x)
    
    return x, history

# 예시: 2차 함수 최적화
def quadratic_function(x):
    """f(x) = x^2 + 2x + 1"""
    return x**2 + 2*x + 1

def quadratic_gradient(x):
    """f'(x) = 2x + 2"""
    return 2*x + 2

# 최적화 실행
x0 = 5
x_opt, history = gradient_descent(quadratic_function, quadratic_gradient, x0)

print(f"최적해: {x_opt}")
print(f"최적값: {quadratic_function(x_opt)}")

# 시각화
x_range = np.linspace(-6, 6, 100)
y_range = [quadratic_function(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='f(x) = x² + 2x + 1')
plt.plot(history, [quadratic_function(x) for x in history], 'ro-', label='경사하강법 경로')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('경사하강법을 이용한 최적화')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. 라그랑주 승수법
```python
from scipy.optimize import minimize

def lagrange_multiplier_example():
    """라그랑주 승수법 예시"""
    # 목적함수: f(x, y) = x² + y²
    def objective(x):
        return x[0]**2 + x[1]**2
    
    # 제약조건: g(x, y) = x + y - 1 = 0
    def constraint(x):
        return x[0] + x[1] - 1
    
    # 초기값
    x0 = [0, 0]
    
    # 제약조건 설정
    constraints = {'type': 'eq', 'fun': constraint}
    
    # 최적화 실행
    result = minimize(objective, x0, constraints=constraints)
    
    print(f"최적해: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"최적값: {result.fun:.4f}")
    
    return result

# 라그랑주 승수법 실행
lagrange_result = lagrange_multiplier_example()
```

## 미분과 적분

### 1. 수치미분
```python
def numerical_derivative(f, x, h=1e-5):
    """수치미분 (중앙차분법)"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_second_derivative(f, x, h=1e-5):
    """수치이계미분"""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# 예시: sin(x)의 미분
def sin_function(x):
    return np.sin(x)

x = np.pi/4
analytical_derivative = np.cos(x)
numerical_derivative_value = numerical_derivative(sin_function, x)

print(f"해석적 미분값: {analytical_derivative:.6f}")
print(f"수치적 미분값: {numerical_derivative_value:.6f}")
print(f"오차: {abs(analytical_derivative - numerical_derivative_value):.6f}")
```

### 2. 수치적분
```python
def trapezoidal_rule(f, a, b, n=1000):
    """사다리꼴 공식"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def simpson_rule(f, a, b, n=1000):
    """심슨 공식"""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])

# 예시: ∫₀^π sin(x) dx = 2
def sin_integrand(x):
    return np.sin(x)

a, b = 0, np.pi
analytical_integral = 2

trapezoidal_result = trapezoidal_rule(sin_integrand, a, b)
simpson_result = simpson_rule(sin_integrand, a, b)

print(f"해석적 적분값: {analytical_integral}")
print(f"사다리꼴 공식: {trapezoidal_result:.6f}")
print(f"심슨 공식: {simpson_result:.6f}")
```

## 정보이론 (Information Theory)

### 1. 엔트로피
```python
def entropy(probabilities):
    """엔트로피 계산"""
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]  # 0인 확률 제거
    return -np.sum(probabilities * np.log2(probabilities))

# 예시: 동전 던지기
fair_coin = [0.5, 0.5]
biased_coin = [0.7, 0.3]

print(f"공정한 동전의 엔트로피: {entropy(fair_coin):.4f}")
print(f"편향된 동전의 엔트로피: {entropy(biased_coin):.4f}")

# 상호정보량
def mutual_information(p_xy):
    """상호정보량 계산"""
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi

# 예시: 두 변수 간 상호정보량
p_xy = np.array([[0.3, 0.1], [0.2, 0.4]])
mi = mutual_information(p_xy)
print(f"상호정보량: {mi:.4f}")
```

## 실무 적용 예시

### 1. 주성분 분석 (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 샘플 데이터 생성
np.random.seed(42)
X = np.random.randn(100, 5)

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산 비율: {np.cumsum(pca.explained_variance_ratio_)}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('첫 번째 주성분')
plt.ylabel('두 번째 주성분')
plt.title('PCA 결과')
plt.grid(True)
plt.show()
```

### 2. 선형회귀 분석
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 샘플 데이터 생성
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)

# 선형회귀 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측
y_pred = model.predict(X)

# 성능 평가
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"평균제곱오차: {mse:.4f}")
print(f"결정계수: {r2:.4f}")
print(f"회귀계수: {model.coef_[0]:.4f}")
print(f"절편: {model.intercept_:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='실제 데이터')
plt.plot(X, y_pred, color='red', linewidth=2, label='회귀선')
plt.xlabel('X')
plt.ylabel('y')
plt.title('선형회귀 분석')
plt.legend()
plt.grid(True)
plt.show()
```

## 마무리

데이터사이언스스쿨 수학편은 데이터 사이언스에 필요한 핵심 수학 지식을 체계적으로 학습할 수 있는 자료입니다. 선형대수, 통계학, 확률론, 최적화, 미적분, 정보이론 등 다양한 수학 분야의 이론과 실무 적용 방법을 통해 데이터 분석의 수학적 기초를 다질 수 있습니다. 이러한 수학적 지식은 머신러닝, 딥러닝, 데이터 분석 등 다양한 분야에서 필수적인 기초가 됩니다.
