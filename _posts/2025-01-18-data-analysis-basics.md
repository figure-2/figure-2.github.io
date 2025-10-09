---
title: 데이터 분석 기초 - 자동차 결함 리콜 데이터 분석 실습
categories:
- 1.TIL
- 1-7.DATA_ANALYSIS
tags:
- 데이터분석
- pandas
- matplotlib
- 데이터정제
- 시각화
- 리콜데이터
toc: true
date: 2023-09-20 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 데이터 분석 기초 - 자동차 결함 리콜 데이터 분석 실습

## 개요

데이터 분석의 기본 과정을 자동차 결함 리콜 데이터를 통해 실습합니다:

- **데이터 읽기**: DataFrame 구조 확인
- **데이터 정제**: 결측치, 중복값 확인 및 처리
- **데이터 시각화**: 데이터 특성 파악을 위한 시각화
- **실무 적용**: 실제 데이터 분석 프로젝트 경험

## 1. 데이터 읽기

### 기본 데이터 로딩

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 읽기 (한글 인코딩 주의)
df = pd.read_csv('데이터경로', encoding='euc_kr')

# 데이터 기본 정보 확인
df.head()      # 상위 5행 확인
df.tail()      # 하위 5행 확인
df.info()      # 데이터 타입, 결측치 정보
df.describe()  # 기초 통계 정보 (수치형 데이터)
```

### 데이터 구조 파악

```python
# 데이터 형태 확인
print(f"데이터 크기: {df.shape}")
print(f"컬럼 수: {df.shape[1]}")
print(f"행 수: {df.shape[0]}")

# 컬럼 정보
print("컬럼 목록:")
print(df.columns.tolist())

# 데이터 타입 확인
print("\n데이터 타입:")
print(df.dtypes)
```

## 2. 데이터 정제

### 2-1. 결측치 확인 및 처리

```python
# 각 열별 결측치 개수 확인
missing_data = df.isnull().sum()
print("결측치 현황:")
print(missing_data[missing_data > 0])

# 결측치 비율 확인
missing_ratio = (df.isnull().sum() / len(df)) * 100
print("\n결측치 비율:")
print(missing_ratio[missing_ratio > 0])

# 결측치 시각화
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('결측치 분포')
plt.show()
```

### 2-2. 중복값 확인 및 처리

```python
# 중복값 확인
duplicated_rows = df[df.duplicated(keep=False)]
print(f"중복된 행 수: {len(duplicated_rows)}")

# 중복값 상세 확인
if len(duplicated_rows) > 0:
    print("중복된 행들:")
    print(duplicated_rows)

# 중복값 제거
df_cleaned = df.drop_duplicates()
print(f"중복 제거 후 데이터 크기: {df_cleaned.shape}")
```

### 2-3. 데이터 타입 변환 및 새로운 컬럼 생성

```python
# 날짜 타입 변환
df['생산기간(부터)'] = pd.to_datetime(df['생산기간(부터)'])
df['생산기간(까지)'] = pd.to_datetime(df['생산기간(까지)'])
df['리콜개시일'] = pd.to_datetime(df['리콜개시일'])

# 새로운 컬럼 생성 (년, 월, 일 추출)
df['생산_년'] = df['생산기간(부터)'].dt.year
df['생산_월'] = df['생산기간(부터)'].dt.month
df['생산_일'] = df['생산기간(부터)'].dt.day

df['리콜_년'] = df['리콜개시일'].dt.year
df['리콜_월'] = df['리콜개시일'].dt.month
df['리콜_일'] = df['리콜개시일'].dt.day

# 불필요한 컬럼 삭제
columns_to_drop = ['생산기간(부터)', '생산기간(까지)', '리콜개시일']
df = df.drop(columns_to_drop, axis=1)
```

### 2-4. 데이터 값 분석

```python
# 특정 컬럼의 고유값 확인
print("리콜 연도별 고유값:")
print(df['리콜_년'].unique())

print(f"\n리콜 연도 고유값 개수: {df['리콜_년'].nunique()}")

# 값 분포 확인
print("\n리콜 연도별 빈도:")
print(df['리콜_년'].value_counts().sort_index())

# 제작자별 리콜 건수 집계
manufacturer_recalls = df.groupby('제작자').size().sort_values(ascending=False)
print("\n제작자별 리콜 건수:")
print(manufacturer_recalls.head(10))
```

## 3. 데이터 시각화

### 3-1. 막대 그래프 (Bar Chart)

```python
# 제작자별 리콜 건수 막대 그래프
top_manufacturers = df.groupby('제작자').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(x=top_manufacturers.index, height=top_manufacturers.values)
plt.xticks(rotation=45, ha='right')
plt.title('제작자별 리콜 건수 현황 (Top 10)', fontsize=14, fontweight='bold')
plt.xlabel('제작자', fontsize=12)
plt.ylabel('리콜 건수', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 3-2. 원형 그래프 (Pie Chart)

```python
# 제작자별 리콜 건수 원형 그래프 (Top 10)
top_10_manufacturers = df.groupby('제작자').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 8))
plt.pie(top_10_manufacturers.values, 
        labels=top_10_manufacturers.index,
        autopct='%.1f%%',
        startangle=90)
plt.title('제작자별 리콜 건수 비율 (Top 10)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()
```

### 3-3. 히스토그램

```python
# 리콜 연도별 분포 히스토그램
plt.figure(figsize=(12, 6))
plt.hist(df['리콜_년'], bins=20, edgecolor='black', alpha=0.7)
plt.title('리콜 연도별 분포', fontsize=14, fontweight='bold')
plt.xlabel('리콜 연도', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### 3-4. 박스 플롯

```python
# 제작자별 리콜 건수 박스 플롯
plt.figure(figsize=(15, 8))
top_5_manufacturers = df.groupby('제작자').size().sort_values(ascending=False).head(5).index
df_top5 = df[df['제작자'].isin(top_5_manufacturers)]

sns.boxplot(data=df_top5, x='제작자', y='리콜_년')
plt.title('주요 제작자별 리콜 연도 분포', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 4. 워드 클라우드

### 4-1. 기본 워드 클라우드

```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# 기본 stopwords 확인
print("기본 stopwords:")
print(set(STOPWORDS))
```

### 4-2. 의미 있는 중복을 포함한 워드 클라우드

```python
# 리콜 사유 텍스트 결합
text = ' '.join(df['리콜사유'].values)

# 한국어 stopwords 설정
korean_stopwords = set([
    "동안", "인하여", "있는", "경우", "있습니다", "가능성이", "않을", "차량의", 
    "가", "에", "될", "이", "인해", "수", "중", "시", "또는", "있음", "의", 
    "및", "있으며", "발생할", "이로", "오류로", "해당", "때문에", "위해", 
    "통해", "관련", "대한", "하여", "되는", "될", "수", "있는"
])

# 워드 클라우드 생성
wordcloud = WordCloud(
    max_font_size=200,
    stopwords=korean_stopwords,
    font_path='./NanumGothic.ttf',  # 한글 폰트 경로
    background_color='white',
    width=800,
    height=600,
    colormap='viridis'
).generate(text)

# 워드 클라우드 시각화
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('리콜 사유 워드 클라우드', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
```

### 4-3. 중복 제거 후 워드 클라우드

```python
# 중복 제거된 리콜 사유 텍스트
unique_text = ' '.join(df['리콜사유'].drop_duplicates().values)

# 워드 클라우드 생성 (중복 제거 버전)
wordcloud_unique = WordCloud(
    max_font_size=200,
    stopwords=korean_stopwords,
    font_path='./NanumGothic.ttf',
    background_color='white',
    width=800,
    height=600,
    colormap='plasma'
).generate(unique_text)

# 워드 클라우드 시각화
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud_unique, interpolation='bilinear')
plt.axis('off')
plt.title('리콜 사유 워드 클라우드 (중복 제거)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
```

## 5. 고급 데이터 분석

### 5-1. 시계열 분석

```python
# 월별 리콜 건수 추이
monthly_recalls = df.groupby(['리콜_년', '리콜_월']).size().reset_index(name='recall_count')
monthly_recalls['date'] = pd.to_datetime(monthly_recalls[['리콜_년', '리콜_월']].assign(day=1))

plt.figure(figsize=(15, 6))
plt.plot(monthly_recalls['date'], monthly_recalls['recall_count'], marker='o')
plt.title('월별 리콜 건수 추이', fontsize=14, fontweight='bold')
plt.xlabel('날짜', fontsize=12)
plt.ylabel('리콜 건수', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5-2. 상관관계 분석

```python
# 수치형 데이터 간 상관관계
numeric_columns = ['생산_년', '생산_월', '리콜_년', '리콜_월']
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('수치형 변수 간 상관관계', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 5-3. 제작자별 상세 분석

```python
# 제작자별 리콜 연도 분포
top_manufacturers = df.groupby('제작자').size().sort_values(ascending=False).head(5).index

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, manufacturer in enumerate(top_manufacturers):
    if i < 5:
        manufacturer_data = df[df['제작자'] == manufacturer]
        axes[i].hist(manufacturer_data['리콜_년'], bins=15, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{manufacturer} 리콜 연도 분포', fontweight='bold')
        axes[i].set_xlabel('리콜 연도')
        axes[i].set_ylabel('빈도')
        axes[i].grid(alpha=0.3)

# 빈 subplot 제거
if len(top_manufacturers) < 6:
    axes[5].remove()

plt.suptitle('주요 제작자별 리콜 연도 분포', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## 6. 데이터 분석 결과 요약

### 6-1. 주요 통계 요약

```python
# 전체 데이터 요약
print("=== 데이터 분석 결과 요약 ===")
print(f"총 리콜 건수: {len(df):,}건")
print(f"분석 기간: {df['리콜_년'].min()}년 ~ {df['리콜_년'].max()}년")
print(f"제작자 수: {df['제작자'].nunique()}개")

print(f"\n리콜이 가장 많은 제작자 Top 5:")
top_5 = df.groupby('제작자').size().sort_values(ascending=False).head(5)
for i, (manufacturer, count) in enumerate(top_5.items(), 1):
    print(f"{i}. {manufacturer}: {count:,}건")

print(f"\n연도별 리콜 건수:")
yearly_recalls = df.groupby('리콜_년').size().sort_index()
for year, count in yearly_recalls.items():
    print(f"{year}년: {count:,}건")
```

### 6-2. 인사이트 도출

```python
# 리콜 패턴 분석
print("\n=== 리콜 패턴 분석 ===")

# 가장 많은 리콜이 발생한 연도
max_recall_year = yearly_recalls.idxmax()
max_recall_count = yearly_recalls.max()
print(f"가장 많은 리콜이 발생한 연도: {max_recall_year}년 ({max_recall_count:,}건)")

# 제작자별 리콜 비율
total_recalls = len(df)
top_manufacturer = top_5.index[0]
top_manufacturer_count = top_5.iloc[0]
top_manufacturer_ratio = (top_manufacturer_count / total_recalls) * 100
print(f"리콜이 가장 많은 제작자: {top_manufacturer} ({top_manufacturer_ratio:.1f}%)")

# 리콜 증가/감소 추이
recent_years = yearly_recalls.tail(3)
if len(recent_years) >= 2:
    trend = "증가" if recent_years.iloc[-1] > recent_years.iloc[-2] else "감소"
    print(f"최근 리콜 추이: {trend}")
```

## 7. 실무 적용 팁

### 7-1. 데이터 품질 관리

```python
# 데이터 품질 체크리스트
def data_quality_check(df):
    """데이터 품질 체크 함수"""
    print("=== 데이터 품질 체크 ===")
    
    # 1. 결측치 체크
    missing_data = df.isnull().sum()
    missing_ratio = (missing_data / len(df)) * 100
    
    print("1. 결측치 현황:")
    for col, ratio in missing_ratio.items():
        if ratio > 0:
            print(f"   {col}: {missing_data[col]}개 ({ratio:.1f}%)")
    
    # 2. 중복값 체크
    duplicates = df.duplicated().sum()
    print(f"\n2. 중복값: {duplicates}개")
    
    # 3. 데이터 타입 체크
    print("\n3. 데이터 타입:")
    for col, dtype in df.dtypes.items():
        print(f"   {col}: {dtype}")
    
    # 4. 이상값 체크 (수치형 데이터)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\n4. 이상값 체크 (수치형 데이터):")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"   {col}: {len(outliers)}개 이상값")

# 데이터 품질 체크 실행
data_quality_check(df)
```

### 7-2. 시각화 모범 사례

```python
# 시각화 설정 함수
def setup_plot_style():
    """시각화 스타일 설정"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

# 스타일 적용
setup_plot_style()
```

## 마무리

데이터 분석의 기본 과정을 자동차 결함 리콜 데이터를 통해 실습했습니다:

### 핵심 학습 내용
- **데이터 읽기**: 다양한 형식의 데이터 로딩과 기본 정보 확인
- **데이터 정제**: 결측치, 중복값 처리 및 데이터 타입 변환
- **데이터 시각화**: 막대 그래프, 원형 그래프, 히스토그램, 워드 클라우드
- **고급 분석**: 시계열 분석, 상관관계 분석, 패턴 분석

### 실무 적용
- **데이터 품질 관리**: 체계적인 데이터 검증 과정
- **시각화 모범 사례**: 효과적인 데이터 시각화 방법
- **인사이트 도출**: 데이터에서 의미 있는 패턴 발견

이러한 기본 과정을 통해 실제 데이터 분석 프로젝트에서 활용할 수 있는 실무 역량을 기를 수 있습니다.
