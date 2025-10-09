---
title: LDA (Linear Discriminant Analysis)
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- LDA
- 선형판별분석
- 차원축소
- 지도학습
- 분류성능
- 판별함수
toc: true
date: 2023-09-15 12:00:00 +0900
comments: false
mermaid: true
math: true
---
# LDA (Linear Discriminant Analysis)

> 231004 학습한 내용 정리

## LDA 개요

### 정의
- **Linear Discriminant Analysis** (선형 판별 분석)
- 지도 학습 기반 차원 축소 기법
- 클래스 간 분리는 최대화하고 클래스 내 분산은 최소화

### 특징
- **지도 학습**: 레이블 정보를 활용한 차원 축소
- **선형 변환**: 선형 변환을 통한 차원 축소
- **분류 성능**: 분류 성능을 향상시키는 차원 축소
- **해석성**: 변환된 차원의 해석이 가능

### 장점
- **분류 성능**: 분류 성능을 향상시키는 차원 축소
- **해석성**: 변환된 차원의 의미를 해석 가능
- **계산 효율성**: 빠른 계산 속도
- **과적합 방지**: 차원 축소로 과적합 방지

## LDA 수학적 원리

### 1. 클래스 간 분산과 클래스 내 분산
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 아이리스 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LDA 적용
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 결과 확인
print(f"원본 데이터 형태: {X.shape}")
print(f"LDA 후 데이터 형태: {X_lda.shape}")
print(f"설명 분산 비율: {lda.explained_variance_ratio_}")
print(f"누적 설명 분산 비율: {np.cumsum(lda.explained_variance_ratio_)}")

# 시각화
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], 
                c=color, label=iris.target_names[i], alpha=0.7)
plt.xlabel('첫 번째 판별 함수')
plt.ylabel('두 번째 판별 함수')
plt.title('LDA 결과 시각화')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. 판별 함수 계산
```python
# 판별 함수 계수 확인
coef = lda.coef_
intercept = lda.intercept_

print("판별 함수 계수:")
for i, (coef_i, intercept_i) in enumerate(zip(coef, intercept)):
    print(f"판별 함수 {i+1}: {coef_i}")
    print(f"절편: {intercept_i}")

# 판별 함수 시각화
plt.figure(figsize=(10, 6))
feature_names = iris.feature_names
x_pos = np.arange(len(feature_names))

for i, coef_i in enumerate(coef):
    plt.bar(x_pos + i*0.25, coef_i, width=0.25, label=f'판별 함수 {i+1}')

plt.xlabel('특성')
plt.ylabel('계수')
plt.title('판별 함수 계수')
plt.xticks(x_pos + 0.125, feature_names, rotation=45)
plt.legend()
plt.grid(True)
plt.show()
```

## LDA 구현 및 사용법

### 1. 기본 LDA
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA 모델 생성
lda_model = LinearDiscriminantAnalysis(n_components=2)

# 모델 학습 및 변환
X_lda = lda_model.fit_transform(X_scaled, y)

# 결과 확인
print(f"설명 분산 비율: {lda_model.explained_variance_ratio_}")
print(f"판별 함수 수: {lda_model.n_components_}")

# 시각화
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], 
                c=color, label=iris.target_names[i], alpha=0.7)
plt.xlabel('첫 번째 판별 함수')
plt.ylabel('두 번째 판별 함수')
plt.title('LDA 결과')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. LDA를 이용한 분류
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 훈련 데이터 표준화
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

# LDA 모델 학습
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train_scaled, y_train)

# 예측
y_pred = lda_classifier.predict(X_test_scaled)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"LDA 분류 정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 3. LDA vs PCA 비교
```python
from sklearn.decomposition import PCA

# PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# LDA와 PCA 결과 비교
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PCA 결과
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0].set_title('PCA 결과')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# LDA 결과
axes[1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[1].set_title('LDA 결과')
axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')

plt.tight_layout()
plt.show()

# 설명 분산 비율 비교
print("PCA 설명 분산 비율:", pca.explained_variance_ratio_)
print("LDA 설명 분산 비율:", lda.explained_variance_ratio_)
```

## LDA 고급 기능

### 1. 다중 클래스 LDA
```python
from sklearn.datasets import make_classification

# 다중 클래스 데이터 생성
X_multi, y_multi = make_classification(
    n_samples=1000, n_features=20, n_classes=5, n_informative=15, random_state=42
)

# 데이터 표준화
X_multi_scaled = StandardScaler().fit_transform(X_multi)

# LDA 적용
lda_multi = LinearDiscriminantAnalysis(n_components=4)
X_multi_lda = lda_multi.fit_transform(X_multi_scaled, y_multi)

# 결과 확인
print(f"원본 데이터 형태: {X_multi.shape}")
print(f"LDA 후 데이터 형태: {X_multi_lda.shape}")
print(f"설명 분산 비율: {lda_multi.explained_variance_ratio_}")

# 시각화
plt.figure(figsize=(10, 6))
colors = plt.cm.Set1(np.linspace(0, 1, 5))
for i, color in enumerate(colors):
    plt.scatter(X_multi_lda[y_multi == i, 0], X_multi_lda[y_multi == i, 1], 
                c=[color], label=f'클래스 {i}', alpha=0.7)
plt.xlabel('첫 번째 판별 함수')
plt.ylabel('두 번째 판별 함수')
plt.title('다중 클래스 LDA 결과')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. LDA를 이용한 특성 선택
```python
# 특성 중요도 계산
feature_importance = np.abs(lda.coef_).mean(axis=0)
feature_names = iris.feature_names

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.title('LDA 특성 중요도')
plt.xlabel('특성')
plt.ylabel('중요도')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 특성 중요도 순으로 정렬
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("특성 중요도 순위:")
print(importance_df)
```

### 3. LDA를 이용한 이상치 탐지
```python
# 이상치가 포함된 데이터 생성
np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
X_outliers = np.random.multivariate_normal([3, 3], [[0.1, 0], [0, 0.1]], 10)
X_mixed = np.vstack([X_normal, X_outliers])
y_mixed = np.hstack([np.zeros(100), np.ones(10)])

# LDA 적용
lda_outlier = LinearDiscriminantAnalysis()
X_lda_outlier = lda_outlier.fit_transform(X_mixed, y_mixed)

# 시각화
plt.figure(figsize=(15, 5))

# 원본 데이터
plt.subplot(1, 3, 1)
plt.scatter(X_mixed[:, 0], X_mixed[:, 1], c=y_mixed, cmap='viridis', alpha=0.7)
plt.title('원본 데이터')
plt.xlabel('특성 1')
plt.ylabel('특성 2')

# LDA 결과
plt.subplot(1, 3, 2)
plt.scatter(X_lda_outlier, np.zeros_like(X_lda_outlier), c=y_mixed, cmap='viridis', alpha=0.7)
plt.title('LDA 결과')
plt.xlabel('판별 함수')
plt.ylabel('')

# 판별 점수 분포
plt.subplot(1, 3, 3)
plt.hist(X_lda_outlier[y_mixed == 0], alpha=0.7, label='정상', bins=20)
plt.hist(X_lda_outlier[y_mixed == 1], alpha=0.7, label='이상치', bins=20)
plt.title('판별 점수 분포')
plt.xlabel('판별 점수')
plt.ylabel('빈도')
plt.legend()

plt.tight_layout()
plt.show()
```

## 실무 적용 예시

### 1. 텍스트 분류
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# 뉴스그룹 데이터 로드
newsgroups = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'])
X_text, y_text = newsgroups.data, newsgroups.target

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text).toarray()

# LDA 적용
lda_text = LinearDiscriminantAnalysis(n_components=3)
X_text_lda = lda_text.fit_transform(X_tfidf, y_text)

# 시각화
plt.figure(figsize=(10, 6))
colors = plt.cm.Set1(np.linspace(0, 1, 4))
for i, color in enumerate(colors):
    plt.scatter(X_text_lda[y_text == i, 0], X_text_lda[y_text == i, 1], 
                c=[color], label=newsgroups.target_names[i], alpha=0.7)
plt.xlabel('첫 번째 판별 함수')
plt.ylabel('두 번째 판별 함수')
plt.title('텍스트 분류 LDA 결과')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. 이미지 분류
```python
from sklearn.datasets import fetch_olivetti_faces

# 얼굴 이미지 데이터 로드
faces = fetch_olivetti_faces()
X_faces, y_faces = faces.data, faces.target

# 데이터 표준화
X_faces_scaled = StandardScaler().fit_transform(X_faces)

# LDA 적용
lda_faces = LinearDiscriminantAnalysis(n_components=39)  # 클래스 수 - 1
X_faces_lda = lda_faces.fit_transform(X_faces_scaled, y_faces)

# 결과 확인
print(f"원본 데이터 형태: {X_faces.shape}")
print(f"LDA 후 데이터 형태: {X_faces_lda.shape}")
print(f"설명 분산 비율: {lda_faces.explained_variance_ratio_[:5]}")

# 시각화
plt.figure(figsize=(10, 6))
colors = plt.cm.Set1(np.linspace(0, 1, 40))
for i in range(40):
    plt.scatter(X_faces_lda[y_faces == i, 0], X_faces_lda[y_faces == i, 1], 
                c=[colors[i]], label=f'Person {i}', alpha=0.7, s=20)
plt.xlabel('첫 번째 판별 함수')
plt.ylabel('두 번째 판별 함수')
plt.title('얼굴 이미지 LDA 결과')
plt.grid(True)
plt.show()
```

### 3. LDA를 이용한 분류 성능 향상
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 훈련 데이터 표준화
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

# 원본 데이터로 분류
rf_original = RandomForestClassifier(random_state=42)
scores_original = cross_val_score(rf_original, X_train_scaled, y_train, cv=5)
print(f"원본 데이터 분류 정확도: {scores_original.mean():.4f} (+/- {scores_original.std() * 2:.4f})")

# LDA 적용 후 분류
lda_classifier = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda_classifier.fit_transform(X_train_scaled, y_train)
X_test_lda = lda_classifier.transform(X_test_scaled)

rf_lda = RandomForestClassifier(random_state=42)
scores_lda = cross_val_score(rf_lda, X_train_lda, y_train, cv=5)
print(f"LDA 적용 후 분류 정확도: {scores_lda.mean():.4f} (+/- {scores_lda.std() * 2:.4f})")

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(['원본 데이터', 'LDA 적용'], [scores_original.mean(), scores_lda.mean()])
plt.ylabel('정확도')
plt.title('LDA 적용 전후 분류 성능 비교')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
```

## LDA 주의사항 및 모범 사례

### 1. 데이터 전처리
- **표준화 필수**: LDA는 스케일에 민감하므로 반드시 표준화
- **결측값 처리**: LDA 적용 전 결측값 제거 또는 대체
- **이상치 처리**: 이상치가 판별 함수에 영향을 줄 수 있음

### 2. 클래스 수 제한
- **최대 차원**: 클래스 수 - 1개까지만 차원 축소 가능
- **클래스 불균형**: 클래스 불균형이 심하면 성능 저하
- **클래스 분리**: 클래스가 잘 분리되어야 효과적

### 3. 해석 주의사항
- **선형 관계**: LDA는 선형 관계만 포착
- **비선형 관계**: 비선형 관계는 LDA로 포착 불가
- **해석성**: 판별 함수의 해석이 가능하지만 복잡할 수 있음

## 마무리

LDA는 지도 학습 기반 차원 축소 기법으로, 클래스 간 분리를 최대화하고 클래스 내 분산을 최소화하여 분류 성능을 향상시키는 강력한 도구입니다. PCA와 달리 레이블 정보를 활용하여 분류에 최적화된 차원 축소를 수행할 수 있습니다. 적절한 데이터 전처리와 클래스 수 고려를 통해 실무에서 효과적으로 활용할 수 있습니다. 다만 선형 관계만 포착할 수 있다는 한계를 인지하고, 필요에 따라 비선형 차원 축소 기법도 고려해야 합니다.
