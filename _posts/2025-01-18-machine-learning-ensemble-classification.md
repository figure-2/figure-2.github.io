---
title: 앙상블 분류 모델
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- 앙상블
- 배깅
- 부스팅
- 스태킹
- 랜덤포레스트
- AdaBoost
- 그라디언트부스팅
toc: true
date: 2023-09-14 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# 앙상블 분류 모델

> 230926 학습한 내용 정리

## 앙상블 학습 (Ensemble Learning)

### 개념
- 여러 개의 모델을 조합하여 더 나은 예측 성능을 달성하는 기법
- 단일 모델보다 일반적으로 더 높은 정확도와 안정성을 제공

### 장점
- **정확도 향상**: 여러 모델의 예측을 결합하여 오류 감소
- **과적합 방지**: 다양한 모델의 조합으로 일반화 성능 향상
- **안정성**: 개별 모델의 단점을 보완

### 단점
- **복잡성**: 모델 해석이 어려움
- **계산 비용**: 여러 모델을 학습해야 하므로 시간과 자원 소모
- **메모리 사용량**: 여러 모델을 저장해야 함

## 앙상블 기법 분류

### 1. 배깅 (Bagging)
- **Bootstrap Aggregating**의 줄임말
- 동일한 알고리즘을 사용하여 여러 모델 생성
- 각 모델은 부트스트랩 샘플로 학습
- 예측 시 모든 모델의 평균 또는 투표

### 2. 부스팅 (Boosting)
- 순차적으로 모델을 학습
- 이전 모델의 오류를 다음 모델이 보완
- 약한 학습기를 강한 학습기로 만드는 기법

### 3. 스태킹 (Stacking)
- 서로 다른 알고리즘을 사용
- 메타 모델이 개별 모델의 예측을 학습
- 2단계 학습 과정

## 배깅 기법

### 1. 랜덤 포레스트 (Random Forest)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 샘플 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# 모델 학습
rf_model.fit(X_train, y_train)

# 예측
y_pred = rf_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"랜덤 포레스트 정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 특성 중요도
feature_importance = rf_model.feature_importances_
print(f"\n특성 중요도 (상위 5개):")
for i, importance in enumerate(feature_importance.argsort()[-5:][::-1]):
    print(f"특성 {importance}: {feature_importance[importance]:.4f}")
```

### 2. 배깅 분류기
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 배깅 분류기
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)

# 모델 학습
bagging_model.fit(X_train, y_train)

# 예측
y_pred_bagging = bagging_model.predict(X_test)

# 성능 평가
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"배깅 분류기 정확도: {accuracy_bagging:.4f}")
```

## 부스팅 기법

### 1. AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost 모델
adaboost_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# 모델 학습
adaboost_model.fit(X_train, y_train)

# 예측
y_pred_adaboost = adaboost_model.predict(X_test)

# 성능 평가
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print(f"AdaBoost 정확도: {accuracy_adaboost:.4f}")
```

### 2. 그라디언트 부스팅
```python
from sklearn.ensemble import GradientBoostingClassifier

# 그라디언트 부스팅 모델
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# 모델 학습
gb_model.fit(X_train, y_train)

# 예측
y_pred_gb = gb_model.predict(X_test)

# 성능 평가
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"그라디언트 부스팅 정확도: {accuracy_gb:.4f}")
```

## 스태킹 기법

### 1. 기본 스태킹
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 기본 모델들
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

# 스태킹 모델
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

# 모델 학습
stacking_model.fit(X_train, y_train)

# 예측
y_pred_stacking = stacking_model.predict(X_test)

# 성능 평가
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"스태킹 모델 정확도: {accuracy_stacking:.4f}")
```

## 앙상블 모델 비교

### 1. 성능 비교
```python
import matplotlib.pyplot as plt

# 모델별 정확도
models = ['Random Forest', 'Bagging', 'AdaBoost', 'Gradient Boosting', 'Stacking']
accuracies = [accuracy, accuracy_bagging, accuracy_adaboost, accuracy_gb, accuracy_stacking]

# 시각화
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
plt.title('앙상블 모델별 정확도 비교')
plt.ylabel('정확도')
plt.ylim(0, 1)

# 막대 위에 값 표시
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.4f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 2. 교차 검증
```python
from sklearn.model_selection import cross_val_score

# 교차 검증
cv_scores = {}
models_dict = {
    'Random Forest': rf_model,
    'Bagging': bagging_model,
    'AdaBoost': adaboost_model,
    'Gradient Boosting': gb_model,
    'Stacking': stacking_model
}

for name, model in models_dict.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## 실무 적용 예시

### 1. 신용카드 사기 탐지
```python
# 불균형 데이터 처리
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 가중치를 적용한 랜덤 포레스트
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)

rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred_weighted)
print("혼동 행렬:")
print(cm)

# AUC 점수
auc_score = roc_auc_score(y_test, rf_weighted.predict_proba(X_test)[:, 1])
print(f"AUC 점수: {auc_score:.4f}")
```

### 2. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

# 랜덤 포레스트 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
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
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"최적 모델 정확도: {accuracy_best:.4f}")
```

## 앙상블 모델 선택 가이드

### 1. 데이터 특성에 따른 선택
- **대용량 데이터**: 랜덤 포레스트, 그라디언트 부스팅
- **소용량 데이터**: AdaBoost, 스태킹
- **고차원 데이터**: 랜덤 포레스트 (특성 선택 효과)
- **불균형 데이터**: 가중치 적용 또는 샘플링 기법

### 2. 성능 vs 해석성
- **성능 우선**: 스태킹, 그라디언트 부스팅
- **해석성 우선**: 랜덤 포레스트 (특성 중요도)
- **균형**: 배깅, AdaBoost

### 3. 계산 자원 고려
- **빠른 학습**: 랜덤 포레스트 (병렬 처리)
- **메모리 효율**: AdaBoost
- **정확도 우선**: 그라디언트 부스팅, 스태킹

## 주의사항 및 모범 사례

### 1. 과적합 방지
- 교차 검증을 통한 모델 평가
- 적절한 하이퍼파라미터 설정
- 조기 종료 (Early Stopping) 활용

### 2. 데이터 전처리
- 특성 스케일링
- 결측값 처리
- 이상치 탐지 및 처리

### 3. 모델 해석
- 특성 중요도 분석
- SHAP 값 활용
- 부분 의존성 플롯

## 마무리

앙상블 학습은 단일 모델의 한계를 극복하고 더 나은 예측 성능을 달성하는 강력한 기법입니다. 배깅, 부스팅, 스태킹 등 다양한 앙상블 방법을 이해하고 적절히 활용하면 실무에서 높은 성능의 모델을 구축할 수 있습니다. 데이터의 특성과 요구사항에 맞는 앙상블 방법을 선택하고, 하이퍼파라미터 튜닝과 교차 검증을 통해 최적의 모델을 찾는 것이 중요합니다.
