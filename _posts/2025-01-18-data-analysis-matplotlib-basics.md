---
title: Matplotlib 기초 - 데이터 시각화의 핵심 도구
categories:
- 1.TIL
- 1-7.DATA_ANALYSIS
tags:
- matplotlib
- 시각화
- 히스토그램
- 박스플롯
- 히트맵
- 데이터분석
toc: true
date: 2023-09-22 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Matplotlib 기초 - 데이터 시각화의 핵심 도구

## 개요

Matplotlib은 Python에서 가장 널리 사용되는 데이터 시각화 라이브러리입니다:

- **기본 구조**: 그래프 생성과 커스터마이징
- **폰트 설정**: 한글 폰트 설정과 스타일링
- **시각화**: 히스토그램, 박스플롯, 히트맵 활용
- **실무 적용**: 효과적인 데이터 시각화 기법

## 1. Matplotlib 기본 구조

### 1-1. 기본 그래프 생성

```python
import matplotlib.pyplot as plt
import numpy as np

# 기본 선 그래프
plt.plot([10, 20, 30, 40])
plt.show()

# y값만 지정 (x는 자동으로 0, 1, 2, 3...)
y = [10, 20, 30, 40]
plt.plot(y)
plt.show()

# x, y값 모두 지정
x = [1, 2, 3, 4]
y = [12, 43, 25, 15]
plt.plot(x, y)
plt.show()
```

### 1-2. 그래프 스타일링

```python
# 제목, 색상, 선 스타일, 범례 설정
plt.title('Color and Line Style Example')
plt.plot([10, 20, 30, 40], 
         color='skyblue', 
         linestyle='--',  # 또는 ls='--'
         label='skyblue line')
plt.plot([40, 30, 20, 10], 
         color='pink', 
         label='pink line')
plt.legend()  # 범례 표시
plt.show()
```

### 1-3. 마커 설정

```python
plt.title('Marker Examples')

# 빨간색 원형 마커
plt.plot([10, 20, 30, 40], 'r.', label='circle marker')
# 초록색 삼각형 마커
plt.plot([40, 30, 20, 10], 'g^', label='triangle up marker')

plt.legend()
plt.show()

# 다양한 마커 스타일
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

plt.figure(figsize=(12, 8))
for i, (marker, color) in enumerate(zip(markers, colors)):
    plt.subplot(3, 5, i+1)
    x = np.linspace(0, 10, 10)
    y = np.sin(x) + i
    plt.plot(x, y, marker=marker, color=color, markersize=8)
    plt.title(f'{marker} - {color}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1-4. 축 범위 설정

```python
y = [10, 20, 30, 40]
plt.plot(y)
plt.xlim((-5, 5))  # x축 범위 지정
plt.ylim((-10, 60))  # y축 범위 지정
plt.grid(True)
plt.show()
```

## 2. 여러 개의 그래프

### 2-1. subplot을 활용한 다중 그래프

```python
import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(1, 11)
y1 = np.exp(-x1)

# 2행 1열 구조
plt.subplot(2, 1, 1)  # nrows=2, ncols=1, index=1
plt.plot(x1, y1, 'o-')
plt.title('1st Graph')
plt.grid(True)

plt.subplot(2, 1, 2)  # nrows=2, ncols=1, index=2
plt.plot(x1, y1, '.-')
plt.title('2nd Graph')
plt.grid(True)

plt.tight_layout()  # 전체 레이아웃 조정
plt.show()
```

### 2-2. for문을 활용한 다중 그래프

```python
# 1행 5열 구조
plt.figure(figsize=(15, 3))

for i in range(5):
    x1 = np.arange(1, 11)
    y1 = np.exp(-x1) * (i + 1)  # 각 그래프마다 다른 스케일
    
    plt.subplot(1, 5, i+1)
    plt.plot(x1, y1, 'o-')
    plt.title(f'{i+1}st Graph')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2-3. 고급 다중 그래프 설정

```python
x = np.arange(1, 5)  # [1, 2, 3, 4]

# 3행 2열 구조, 축 공유 설정
fig, ax = plt.subplots(3, 2, 
                      sharex=True,  # x축 공유
                      sharey=True,  # y축 공유
                      squeeze=True,
                      figsize=(10, 12))

# 각 서브플롯에 다른 그래프 그리기
ax[0][0].plot(x, np.sqrt(x), 'gray', linewidth=3)
ax[0][0].set_title('Square Root')

ax[0][1].plot(x, x, 'g^-', markersize=10)
ax[0][1].set_title('Linear')

ax[1][0].plot(x, -x+5, 'ro--')
ax[1][0].set_title('Negative Linear')

ax[1][1].plot(x, np.sqrt(-x+5), 'b.-.')
ax[1][1].set_title('Square Root (Negative)')

ax[2][0].plot(x, x**2, 'purple', marker='s')
ax[2][0].set_title('Quadratic')

ax[2][1].plot(x, np.log(x), 'orange', marker='D')
ax[2][1].set_title('Logarithm')

plt.tight_layout()
plt.show()
```

## 3. 히스토그램 (Histogram)

### 3-1. 기본 히스토그램

```python
import pandas as pd
import seaborn as sns

# 샘플 데이터 생성
np.random.seed(42)
data = np.random.normal(100, 15, 1000)  # 평균 100, 표준편차 15

# 기본 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
plt.title('Basic Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
```

### 3-2. 다양한 bins 설정

```python
# 다양한 bins 설정으로 분포 비교
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

bins_list = [10, 30, 50, 100]
titles = ['Bins = 10', 'Bins = 30', 'Bins = 50', 'Bins = 100']

for i, (bins, title) in enumerate(zip(bins_list, titles)):
    row = i // 2
    col = i % 2
    
    axes[row, col].hist(data, bins=bins, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(title)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3-3. 실제 데이터 활용

```python
# Iris 데이터셋 활용
iris = sns.load_dataset('iris')

# 각 특성별 히스토그램
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    
    axes[row, col].hist(iris[feature], bins=20, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'{feature.replace("_", " ").title()} Distribution')
    axes[row, col].set_xlabel(feature.replace("_", " ").title())
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 4. 박스플롯 (Boxplot)

### 4-1. 기본 박스플롯

```python
# 단일 컬럼 박스플롯
np.random.seed(42)
data = np.random.normal(100, 15, 100)

plt.figure(figsize=(8, 6))
plt.boxplot(data)
plt.title('Basic Boxplot')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4-2. 여러 컬럼 박스플롯

```python
# Iris 데이터셋의 수치형 컬럼들
iris_numeric = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# 방법 1: pandas의 boxplot 메서드
plt.figure(figsize=(10, 6))
iris_numeric.boxplot()
plt.title('Iris Dataset - All Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# 방법 2: matplotlib의 boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(iris_numeric.values, labels=iris_numeric.columns)
plt.title('Iris Dataset - All Features (Matplotlib)')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
```

### 4-3. 박스플롯 통계값 계산

```python
# IQR (Interquartile Range) 계산
feature = 'sepal_width'
Q1 = iris[feature].quantile(0.25)
Q3 = iris[feature].quantile(0.75)
IQR = Q3 - Q1

print(f"=== {feature} 박스플롯 통계값 ===")
print(f"Q1 (1사분위수): {Q1:.2f}")
print(f"Q3 (3사분위수): {Q3:.2f}")
print(f"IQR (사분위수 범위): {IQR:.2f}")

# 이상치 경계값 계산
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"이상치 하한: {lower_bound:.2f}")
print(f"이상치 상한: {upper_bound:.2f}")

# 이상치 확인
outliers = iris[(iris[feature] < lower_bound) | (iris[feature] > upper_bound)]
print(f"이상치 개수: {len(outliers)}개")

# 박스플롯과 이상치 시각화
plt.figure(figsize=(10, 6))
plt.boxplot(iris[feature], vert=True)
plt.title(f'{feature.replace("_", " ").title()} Boxplot with Outliers')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

# 이상치를 별도로 표시
if len(outliers) > 0:
    plt.scatter([1] * len(outliers), outliers[feature], 
               color='red', s=50, alpha=0.7, label='Outliers')
    plt.legend()

plt.show()
```

## 5. 히트맵 (Heatmap)

### 5-1. 기본 히트맵

```python
import seaborn as sns

# 상관관계 행렬 생성
titanic = sns.load_dataset('titanic')
correlation_matrix = titanic.corr(numeric_only=True)

# 기본 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Titanic Dataset Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### 5-2. 고급 히트맵 설정

```python
# 상세한 히트맵 설정
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,  # 수치 표시
            annot_kws={'size': 8},  # 수치 크기
            cmap=plt.cm.RdYlBu_r,  # 색상 맵
            fmt=".2f",  # 소수점 둘째 자리
            linewidth=0.5,  # 선 두께
            vmin=-1.0,  # 최소값
            vmax=1.0,  # 최대값
            square=False,  # 정사각형 모양
            cbar_kws={'shrink': 0.8})  # 컬러바 설정

plt.title('Titanic Dataset - Detailed Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### 5-3. 생존률과의 상관관계 분석

```python
# 생존(Survived)과 가장 상관도가 높은 피처 찾기
survived_corr = correlation_matrix['survived'].abs().sort_values(ascending=False)
print("생존률과의 상관관계 (절댓값 기준):")
print(survived_corr)

# 상위 상관관계 피처들만 시각화
top_features = survived_corr.head(6).index
top_corr_matrix = correlation_matrix.loc[top_features, top_features]

plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt=".2f")
plt.title('Top Features Correlation with Survival')
plt.tight_layout()
plt.show()
```

## 6. 폰트 설정

### 6-1. 한글 폰트 설정

```python
import matplotlib.font_manager as fm

# 시스템에 설치된 한글 폰트 확인
font_list = [f.name for f in fm.fontManager.ttflist if '한글' in f.name or 'Korean' in f.name]
print("사용 가능한 한글 폰트:")
for font in font_list:
    print(f"- {font}")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 또는 특정 폰트 지정
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'    # Mac
# plt.rcParams['font.family'] = 'NanumGothic'    # Linux
```

### 6-2. 폰트 스타일 설정

```python
# 전역 폰트 설정
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# 테스트 그래프
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='sin(x)')
plt.title('폰트 설정 테스트')
plt.xlabel('X축 라벨')
plt.ylabel('Y축 라벨')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 7. 실무 활용 예제

### 7-1. 종합 시각화 대시보드

```python
def create_dashboard(data):
    """종합 시각화 대시보드 생성"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 히스토그램
    axes[0, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 박스플롯
    axes[0, 1].boxplot(data)
    axes[0, 1].set_title('Boxplot')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q 플롯 (정규성 검정)
    from scipy import stats
    stats.probplot(data, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 누적 분포 함수
    sorted_data = np.sort(data)
    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 0].plot(sorted_data, y_vals, 'b-', linewidth=2)
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 시계열 플롯 (인덱스를 시간으로 가정)
    axes[1, 1].plot(data, 'g-', alpha=0.7)
    axes[1, 1].set_title('Time Series')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 통계 요약
    axes[1, 2].axis('off')
    stats_text = f"""
    통계 요약:
    
    평균: {np.mean(data):.2f}
    중앙값: {np.median(data):.2f}
    표준편차: {np.std(data):.2f}
    최소값: {np.min(data):.2f}
    최대값: {np.max(data):.2f}
    왜도: {stats.skew(data):.2f}
    첨도: {stats.kurtosis(data):.2f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle('Data Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 샘플 데이터로 대시보드 생성
np.random.seed(42)
sample_data = np.random.normal(100, 15, 1000)
create_dashboard(sample_data)
```

### 7-2. 커스텀 스타일 함수

```python
def set_plot_style(style='default'):
    """그래프 스타일 설정 함수"""
    
    if style == 'professional':
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    elif style == 'minimal':
        plt.style.use('default')
        plt.rcParams.update({
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    else:  # default
        plt.style.use('default')

# 스타일 적용 예제
set_plot_style('professional')

plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.title('Professional Style Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
```

## 마무리

Matplotlib은 데이터 시각화의 핵심 도구입니다:

### 핵심 학습 내용
- **기본 구조**: 그래프 생성, 스타일링, 다중 그래프 구성
- **히스토그램**: 데이터 분포 시각화와 bins 설정의 중요성
- **박스플롯**: 데이터 분포와 이상치 탐지
- **히트맵**: 상관관계 분석과 패턴 발견
- **폰트 설정**: 한글 폰트 설정과 스타일링

### 실무 적용
- **시각화 대시보드**: 종합적인 데이터 분석 시각화
- **스타일 관리**: 일관된 그래프 스타일 설정
- **통계 시각화**: 기술 통계와 분포 분석
- **상관관계 분석**: 변수 간 관계 파악

Matplotlib의 기본기를 익혀 효과적인 데이터 시각화를 통해 인사이트를 도출할 수 있습니다.
