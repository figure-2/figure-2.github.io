---
title: '고급 머신러닝: 모델 선택과 평가'
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- 머신러닝
- 모델선택
- 교차검증
- 하이퍼파라미터
- 앙상블
- 그리드서치
- 베이지안최적화
toc: true
date: 2023-09-14 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 고급 머신러닝: 모델 선택과 평가
> 최적의 모델을 선택하고 성능을 평가하는 고급 기법

## 모델 선택의 중요성

### 모델 선택이란?
- 여러 알고리즘 중 최적의 모델을 선택하는 과정
- 데이터와 문제에 가장 적합한 모델 찾기
- 성능, 복잡성, 해석가능성의 균형 고려

### 모델 선택의 도전과제
- **과적합**: 훈련 데이터에만 잘 맞는 모델
- **과소적합**: 데이터의 패턴을 제대로 학습하지 못하는 모델
- **편향-분산 트레이드오프**: 편향과 분산의 균형
- **일반화**: 새로운 데이터에 대한 성능

## 교차 검증 (Cross Validation)

### 1. K-Fold 교차 검증
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 5-Fold 교차 검증
rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print(f"교차 검증 점수: {cv_scores}")
print(f"평균 점수: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### 2. Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

# 계층적 K-Fold (클래스 비율 유지)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
```

### 3. 시계열 교차 검증
```python
from sklearn.model_selection import TimeSeriesSplit

# 시계열 데이터용 교차 검증
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='accuracy')
```

## 하이퍼파라미터 튜닝

### 1. 그리드 서치 (Grid Search)
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 하이퍼파라미터 그리드 정의
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 그리드 서치 실행
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, 
    scoring='accuracy', n_jobs=-1
)
grid_search.fit(X, y)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 점수: {grid_search.best_score_:.3f}")
```

### 2. 랜덤 서치 (Random Search)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 랜덤 서치 파라미터 분포
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# 랜덤 서치 실행
random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=100, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)
random_search.fit(X, y)
```

### 3. 베이지안 최적화
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# 베이지안 최적화 파라미터 공간
param_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10)
}

# 베이지안 최적화 실행
bayes_search = BayesSearchCV(
    rf, param_space, n_iter=100, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)
bayes_search.fit(X, y)
```

## 모델 평가 기법

### 1. 학습 곡선 (Learning Curve)
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# 학습 곡선 생성
train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

### 2. 검증 곡선 (Validation Curve)
```python
from sklearn.model_selection import validation_curve

# 특정 파라미터에 대한 검증 곡선
param_range = np.logspace(-6, -1, 5)
train_scores, val_scores = validation_curve(
    rf, X, y, param_name='min_samples_leaf',
    param_range=param_range, cv=5
)

# 시각화
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_scores.mean(axis=1), 'o-', label='Training Score')
plt.semilogx(param_range, val_scores.mean(axis=1), 'o-', label='Validation Score')
plt.xlabel('min_samples_leaf')
plt.ylabel('Score')
plt.legend()
plt.title('Validation Curve')
plt.show()
```

## 앙상블 기법

### 1. 보팅 (Voting)
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 여러 모델 결합
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ],
    voting='soft'  # 'hard' 또는 'soft'
)

voting_clf.fit(X, y)
```

### 2. 배깅 (Bagging)
```python
from sklearn.ensemble import BaggingClassifier

# 배깅 앙상블
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)

bagging_clf.fit(X, y)
```

### 3. 부스팅 (Boosting)
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# AdaBoost
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

## 모델 선택 전략

### 1. 단계별 접근
```python
# 1단계: 기본 모델들 비교
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier()
}

# 2단계: 교차 검증으로 성능 비교
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    results[name] = cv_scores.mean()

# 3단계: 최고 성능 모델 선택
best_model = max(results, key=results.get)
print(f"최고 성능 모델: {best_model}")
```

### 2. 성능 지표 고려
```python
# 다양한 성능 지표로 평가
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for metric in scoring_metrics:
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring=metric)
    print(f"{metric}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### 3. 비즈니스 요구사항 고려
```python
# 비즈니스 요구사항에 따른 모델 선택
def evaluate_model_for_business(model, X, y, cost_fp=1, cost_fn=10):
    """
    비즈니스 비용을 고려한 모델 평가
    cost_fp: False Positive 비용
    cost_fn: False Negative 비용
    """
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    total_cost = fp * cost_fp + fn * cost_fn
    return total_cost
```

## 주요 학습 포인트

### 1. 편향-분산 트레이드오프
- **편향**: 모델의 단순함으로 인한 오류
- **분산**: 모델의 복잡성으로 인한 오류
- **최적점**: 편향과 분산의 균형점

### 2. 과적합 방지
- 교차 검증으로 일반화 성능 확인
- 정규화 기법 활용
- 조기 종료 (Early Stopping)

### 3. 모델 해석가능성
- 비즈니스 요구사항 고려
- 특성 중요도 분석
- 모델 설명 가능성

### 4. 실무 고려사항
- 계산 비용과 성능의 균형
- 모델 배포의 복잡성
- 지속적인 모니터링 필요

고급 머신러닝 기법을 통해 더 정확하고 안정적인 모델을 구축할 수 있습니다.
