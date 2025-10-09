---
title: PCA (Principal Component Analysis)
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- PCA
- 차원축소
- 주성분분석
- 데이터분석
- 시각화
- 노이즈제거
toc: true
date: 2023-09-15 11:00:00 +0900
comments: false
mermaid: true
math: true
---
# PCA (Principal Component Analysis)

> 231004 학습한 내용 정리

## PCA 개요

### 정의
- **Principal Component Analysis** (주성분 분석)
- 고차원 데이터를 저차원으로 변환하는 차원 축소 기법
- 데이터의 분산을 최대한 보존하면서 차원을 줄임

### 특징
- **분산 보존**: 데이터의 분산을 최대한 유지
- **직교성**: 주성분들은 서로 직교
- **선형 변환**: 선형 변환을 통한 차원 축소
- **비지도 학습**: 레이블이 필요하지 않음

### 장점
- **차원 축소**: 고차원 데이터를 저차원으로 변환
- **노이즈 제거**: 주요 성분만 추출하여 노이즈 감소
- **시각화**: 고차원 데이터를 2D/3D로 시각화
- **계산 효율성**: 차원 축소로 계산 속도 향상

## PCA 수학적 원리

### 1. 주성분 (Principal Components)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 샘플 데이터 생성
np.random.seed(42)
X = np.random.randn(100, 3)

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 주성분 확인
print(f"주성분 수: {pca.n_components_}")
print(f"설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산 비율: {np.cumsum(pca.explained_variance_ratio_)}")
```

### 2. 공분산 행렬과 고유값 분해
```python
# 공분산 행렬 계산
cov_matrix = np.cov(X_scaled.T)
print(f"공분산 행렬:\n{cov_matrix}")

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 고유값 정렬
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"고유값: {eigenvalues}")
print(f"고유벡터:\n{eigenvectors}")

# 설명 분산 비율 계산
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print(f"설명 분산 비율: {explained_variance_ratio}")
```

## PCA 구현 및 사용법

### 1. 기본 PCA
```python
from sklearn.datasets import load_iris

# 아이리스 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 결과 확인
print(f"원본 데이터 형태: {X.shape}")
print(f"PCA 후 데이터 형태: {X_pca.shape}")
print(f"설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산 비율: {np.cumsum(pca.explained_variance_ratio_)}")

# 시각화
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                c=color, label=iris.target_names[i], alpha=0.7)
plt.xlabel('첫 번째 주성분')
plt.ylabel('두 번째 주성분')
plt.title('PCA 결과 시각화')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. 주성분 수 선택
```python
# 모든 주성분에 대한 설명 분산 비율 계산
pca_full = PCA()
pca_full.fit(X_scaled)

# 스크리 플롯
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_, 'bo-')
plt.xlabel('주성분 수')
plt.ylabel('설명 분산 비율')
plt.title('스크리 플롯')
plt.grid(True)
plt.show()

# 누적 설명 분산 비율
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         np.cumsum(pca_full.explained_variance_ratio_), 'ro-')
plt.xlabel('주성분 수')
plt.ylabel('누적 설명 분산 비율')
plt.title('누적 설명 분산 비율')
plt.axhline(y=0.95, color='k', linestyle='--', label='95%')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. 주성분 수 자동 선택
```python
# 95% 분산을 설명하는 주성분 수 선택
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"95% 분산을 설명하는 주성분 수: {pca_95.n_components_}")
print(f"실제 설명 분산 비율: {np.sum(pca_95.explained_variance_ratio_):.4f}")

# 2개 주성분으로 제한
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

print(f"2개 주성분 설명 분산 비율: {np.sum(pca_2.explained_variance_ratio_):.4f}")
```

## PCA 고급 기능

### 1. 역변환 (Inverse Transform)
```python
# PCA 역변환
X_reconstructed = pca_2.inverse_transform(X_pca_2)

# 원본과 재구성된 데이터 비교
mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"재구성 오차 (MSE): {mse:.6f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 원본 데이터
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0].set_title('원본 데이터 (첫 2개 특성)')
axes[0].set_xlabel('특성 1')
axes[0].set_ylabel('특성 2')

# 재구성된 데이터
axes[1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[1].set_title('재구성된 데이터')
axes[1].set_xlabel('특성 1')
axes[1].set_ylabel('특성 2')

plt.tight_layout()
plt.show()
```

### 2. 주성분 분석
```python
# 주성분 계수 분석
components = pca_2.components_
feature_names = iris.feature_names

print("주성분 계수:")
for i, component in enumerate(components):
    print(f"PC{i+1}: {dict(zip(feature_names, component))}")

# 주성분 계수 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), components[0], alpha=0.7, label='PC1')
plt.bar(range(len(feature_names)), components[1], alpha=0.7, label='PC2')
plt.xlabel('특성')
plt.ylabel('계수')
plt.title('주성분 계수')
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.legend()
plt.grid(True)
plt.show()
```

### 3. 희소 PCA (Sparse PCA)
```python
from sklearn.decomposition import SparsePCA

# 희소 PCA 적용
sparse_pca = SparsePCA(n_components=2, alpha=0.1, random_state=42)
X_sparse_pca = sparse_pca.fit_transform(X_scaled)

# 결과 비교
plt.figure(figsize=(15, 5))

# 일반 PCA
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('일반 PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 희소 PCA
plt.subplot(1, 3, 2)
plt.scatter(X_sparse_pca[:, 0], X_sparse_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('희소 PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 주성분 계수 비교
plt.subplot(1, 3, 3)
plt.bar(range(len(feature_names)), sparse_pca.components_[0], alpha=0.7, label='Sparse PC1')
plt.bar(range(len(feature_names)), pca_2.components_[0], alpha=0.7, label='PC1')
plt.xlabel('특성')
plt.ylabel('계수')
plt.title('주성분 계수 비교')
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 실무 적용 예시

### 1. 이미지 압축
```python
from sklearn.datasets import fetch_olivetti_faces

# 얼굴 이미지 데이터 로드
faces = fetch_olivetti_faces()
X_faces = faces.data
y_faces = faces.target

# 데이터 표준화
X_faces_scaled = StandardScaler().fit_transform(X_faces)

# PCA 적용
pca_faces = PCA(n_components=100)
X_faces_pca = pca_faces.fit_transform(X_faces_scaled)

# 재구성
X_faces_reconstructed = pca_faces.inverse_transform(X_faces_pca)

# 원본과 재구성된 이미지 비교
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # 원본 이미지
    axes[0, i].imshow(X_faces[i].reshape(64, 64), cmap='gray')
    axes[0, i].set_title('원본')
    axes[0, i].axis('off')
    
    # 재구성된 이미지
    axes[1, i].imshow(X_faces_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].set_title('재구성')
    axes[1, i].axis('off')

plt.suptitle('PCA를 이용한 이미지 압축')
plt.tight_layout()
plt.show()

# 압축률 계산
compression_ratio = pca_faces.n_components_ / X_faces.shape[1]
print(f"압축률: {compression_ratio:.2%}")
print(f"설명 분산 비율: {np.sum(pca_faces.explained_variance_ratio_):.4f}")
```

### 2. 차원 축소 후 분류
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 원본 데이터로 분류
rf_original = RandomForestClassifier(random_state=42)
scores_original = cross_val_score(rf_original, X_train, y_train, cv=5)
print(f"원본 데이터 분류 정확도: {scores_original.mean():.4f} (+/- {scores_original.std() * 2:.4f})")

# PCA 적용 후 분류
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

pca_classifier = PCA(n_components=2)
X_train_pca = pca_classifier.fit_transform(X_train_scaled)
X_test_pca = pca_classifier.transform(X_test_scaled)

rf_pca = RandomForestClassifier(random_state=42)
scores_pca = cross_val_score(rf_pca, X_train_pca, y_train, cv=5)
print(f"PCA 적용 후 분류 정확도: {scores_pca.mean():.4f} (+/- {scores_pca.std() * 2:.4f})")

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(['원본 데이터', 'PCA 적용'], [scores_original.mean(), scores_pca.mean()])
plt.ylabel('정확도')
plt.title('PCA 적용 전후 분류 성능 비교')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
```

### 3. 이상치 탐지
```python
# 이상치가 포함된 데이터 생성
np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
X_outliers = np.random.multivariate_normal([3, 3], [[0.1, 0], [0, 0.1]], 10)
X_mixed = np.vstack([X_normal, X_outliers])

# PCA 적용
pca_outlier = PCA(n_components=2)
X_pca_outlier = pca_outlier.fit_transform(X_mixed)

# 재구성 오차 계산
X_reconstructed_outlier = pca_outlier.inverse_transform(X_pca_outlier)
reconstruction_error = np.sum((X_mixed - X_reconstructed_outlier) ** 2, axis=1)

# 이상치 탐지
threshold = np.percentile(reconstruction_error, 95)
outliers = reconstruction_error > threshold

# 시각화
plt.figure(figsize=(15, 5))

# 원본 데이터
plt.subplot(1, 3, 1)
plt.scatter(X_mixed[:, 0], X_mixed[:, 1], c=outliers, cmap='viridis', alpha=0.7)
plt.title('원본 데이터')
plt.xlabel('특성 1')
plt.ylabel('특성 2')

# PCA 결과
plt.subplot(1, 3, 2)
plt.scatter(X_pca_outlier[:, 0], X_pca_outlier[:, 1], c=outliers, cmap='viridis', alpha=0.7)
plt.title('PCA 결과')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 재구성 오차
plt.subplot(1, 3, 3)
plt.scatter(range(len(reconstruction_error)), reconstruction_error, c=outliers, cmap='viridis', alpha=0.7)
plt.axhline(y=threshold, color='r', linestyle='--', label='임계값')
plt.title('재구성 오차')
plt.xlabel('샘플 인덱스')
plt.ylabel('재구성 오차')
plt.legend()

plt.tight_layout()
plt.show()

print(f"탐지된 이상치 수: {np.sum(outliers)}")
print(f"실제 이상치 수: {len(X_outliers)}")
```

## PCA 주의사항 및 모범 사례

### 1. 데이터 전처리
- **표준화 필수**: PCA는 스케일에 민감하므로 반드시 표준화
- **결측값 처리**: PCA 적용 전 결측값 제거 또는 대체
- **이상치 처리**: 이상치가 주성분에 영향을 줄 수 있음

### 2. 주성분 수 선택
- **스크리 플롯**: 설명 분산 비율이 급격히 감소하는 지점
- **누적 분산**: 95% 또는 99% 분산을 설명하는 주성분 수
- **도메인 지식**: 비즈니스 요구사항에 맞는 주성분 수

### 3. 해석 주의사항
- **선형 관계**: PCA는 선형 관계만 포착
- **비선형 관계**: 비선형 관계는 PCA로 포착 불가
- **해석성**: 주성분의 해석이 어려울 수 있음

## 마무리

PCA는 고차원 데이터를 저차원으로 변환하는 강력한 차원 축소 기법입니다. 데이터의 분산을 최대한 보존하면서 차원을 줄여 시각화, 노이즈 제거, 계산 효율성 향상 등의 목적을 달성할 수 있습니다. 적절한 데이터 전처리와 주성분 수 선택을 통해 실무에서 효과적으로 활용할 수 있습니다. 다만 선형 관계만 포착할 수 있다는 한계를 인지하고, 필요에 따라 비선형 차원 축소 기법도 고려해야 합니다.
