---
title: EDA(탐색적 데이터 분석) - 데이터 이해의 첫 걸음
categories:
- 1.TIL
- 1-7.DATA_ANALYSIS
tags:
- EDA
- 탐색적데이터분석
- 기술통계
- sweetviz
- 데이터타입
- 시각화
toc: true
date: 2023-09-21 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# EDA(탐색적 데이터 분석) - 데이터 이해의 첫 걸음

## 개요

탐색적 데이터 분석(EDA)은 데이터를 이해하고 패턴을 발견하는 핵심 과정입니다:

- **데이터 타입**: 수치형, 문자형, 범주형, 불리언형 분류
- **기술 통계**: 중심 경향성과 분산성 측정
- **EDA 도구**: Sweetviz를 활용한 자동화된 분석
- **실무 적용**: 체계적인 데이터 탐색 방법론

## 1. 데이터 타입 분류

### 1-1. 데이터 타입 개요

데이터는 크게 4가지 타입으로 분류됩니다:

- **수치형(Numerical)**: 연속형, 이산형
- **문자형(String)**: 텍스트 데이터
- **범주형(Categorical)**: 순서형, 명목형
- **불리언형(Boolean)**: True/False 값

### 1-2. Iris 데이터셋 분석

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Iris 데이터 로드
iris = sns.load_dataset('iris')

# 데이터 타입 확인
print("Iris 데이터셋 정보:")
print(iris.info())
print("\n데이터 타입별 분류:")
print(iris.dtypes)

# 수치형 데이터
numerical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(f"\n수치형 변수: {numerical_features}")

# 범주형 데이터
categorical_features = ['species']
print(f"범주형 변수: {categorical_features}")

# 데이터 미리보기
print("\n데이터 미리보기:")
print(iris.head())
```

### 1-3. Titanic 데이터셋 분석

```python
# Titanic 데이터 로드
titanic = sns.load_dataset('titanic')

print("Titanic 데이터셋 정보:")
print(titanic.info())

# 데이터 타입별 분류
print("\n=== 데이터 타입 분류 ===")

# 수치형(연속형)
continuous_numerical = ['age', 'fare']
print(f"수치형(연속형): {continuous_numerical}")

# 수치형(이산형)
discrete_numerical = ['sibsp', 'parch']
print(f"수치형(이산형): {discrete_numerical}")

# 범주형(순서형)
ordinal_categorical = ['pclass']
print(f"범주형(순서형): {ordinal_categorical}")

# 범주형(명목형)
nominal_categorical = ['sex', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone']
print(f"범주형(명목형): {nominal_categorical}")

# 불리언형
boolean_features = ['adult_male', 'alone']
print(f"불리언형: {boolean_features}")

# 데이터 미리보기
print("\n데이터 미리보기:")
print(titanic.head())
```

## 2. 기술 통계 (Descriptive Statistics)

### 2-1. 중심 경향성 측정

```python
# 수치형 데이터의 중심 경향성
def calculate_central_tendency(data):
    """중심 경향성 계산"""
    import numpy as np
    from scipy import stats
    
    # 결측치 제거
    clean_data = data.dropna()
    
    mean_val = np.mean(clean_data)
    median_val = np.median(clean_data)
    mode_val = stats.mode(clean_data, keepdims=True)[0][0] if len(clean_data) > 0 else None
    
    return {
        'Mean (평균)': mean_val,
        'Median (중앙값)': median_val,
        'Mode (최빈값)': mode_val
    }

# Titanic 데이터의 연령 분석
print("=== 연령 데이터 중심 경향성 ===")
age_stats = calculate_central_tendency(titanic['age'])
for stat, value in age_stats.items():
    print(f"{stat}: {value:.2f}")

# 요금 데이터의 중심 경향성
print("\n=== 요금 데이터 중심 경향성 ===")
fare_stats = calculate_central_tendency(titanic['fare'])
for stat, value in fare_stats.items():
    print(f"{stat}: {value:.2f}")
```

### 2-2. 분산성 측정

```python
def calculate_dispersion(data):
    """분산성 측정"""
    import numpy as np
    
    # 결측치 제거
    clean_data = data.dropna()
    
    min_val = np.min(clean_data)
    max_val = np.max(clean_data)
    var_val = np.var(clean_data)
    std_val = np.std(clean_data)
    q1 = np.percentile(clean_data, 25)
    q2 = np.percentile(clean_data, 50)  # 중앙값
    q3 = np.percentile(clean_data, 75)
    iqr = q3 - q1
    
    return {
        'Min (최소값)': min_val,
        'Max (최대값)': max_val,
        'Variance (분산)': var_val,
        'Std (표준편차)': std_val,
        'Q1 (1사분위수)': q1,
        'Q2 (2사분위수)': q2,
        'Q3 (3사분위수)': q3,
        'IQR (사분위수 범위)': iqr
    }

# 연령 데이터 분산성
print("=== 연령 데이터 분산성 ===")
age_dispersion = calculate_dispersion(titanic['age'])
for stat, value in age_dispersion.items():
    print(f"{stat}: {value:.2f}")

# 요금 데이터 분산성
print("\n=== 요금 데이터 분산성 ===")
fare_dispersion = calculate_dispersion(titanic['fare'])
for stat, value in fare_dispersion.items():
    print(f"{stat}: {value:.2f}")
```

### 2-3. 분포 형태 분석

```python
from scipy import stats

def analyze_distribution(data):
    """분포 형태 분석"""
    clean_data = data.dropna()
    
    # 왜도 (Skewness)
    skewness = stats.skew(clean_data)
    
    # 첨도 (Kurtosis)
    kurtosis = stats.kurtosis(clean_data)
    
    # 분포 해석
    skew_interpretation = "왼쪽 치우침" if skewness < -0.5 else "오른쪽 치우침" if skewness > 0.5 else "대칭"
    kurt_interpretation = "뾰족함" if kurtosis > 0 else "평평함" if kurtosis < 0 else "정규분포와 유사"
    
    return {
        'Skewness (왜도)': skewness,
        'Kurtosis (첨도)': kurtosis,
        '분포 해석': f"{skew_interpretation}, {kurt_interpretation}"
    }

# 연령 분포 분석
print("=== 연령 분포 형태 ===")
age_dist = analyze_distribution(titanic['age'])
for stat, value in age_dist.items():
    print(f"{stat}: {value}")

# 요금 분포 분석
print("\n=== 요금 분포 형태 ===")
fare_dist = analyze_distribution(titanic['fare'])
for stat, value in fare_dist.items():
    print(f"{stat}: {value}")
```

## 3. 시각화를 통한 EDA

### 3-1. 박스 플롯

```python
# 결측치 처리 방법 1: 결측치 제거
titanic_clean = titanic.dropna(subset=['age'])

plt.figure(figsize=(12, 6))

# 서브플롯 1: 연령 박스 플롯
plt.subplot(1, 2, 1)
plt.boxplot(titanic_clean['age'])
plt.title('연령 분포 (Box Plot)')
plt.ylabel('연령')

# 서브플롯 2: 요금 박스 플롯
plt.subplot(1, 2, 2)
titanic['fare'].plot(kind='box')
plt.title('요금 분포 (Box Plot)')
plt.ylabel('요금')

plt.tight_layout()
plt.show()
```

### 3-2. 히스토그램

```python
plt.figure(figsize=(15, 10))

# 연령 히스토그램
plt.subplot(2, 3, 1)
plt.hist(titanic_clean['age'], bins=20, alpha=0.7, edgecolor='black')
plt.title('연령 분포')
plt.xlabel('연령')
plt.ylabel('빈도')

# 요금 히스토그램
plt.subplot(2, 3, 2)
plt.hist(titanic['fare'], bins=20, alpha=0.7, edgecolor='black')
plt.title('요금 분포')
plt.xlabel('요금')
plt.ylabel('빈도')

# 생존자별 연령 분포
plt.subplot(2, 3, 3)
survived_age = titanic_clean[titanic_clean['survived'] == 1]['age']
not_survived_age = titanic_clean[titanic_clean['survived'] == 0]['age']

plt.hist(survived_age, bins=15, alpha=0.7, label='생존', color='green')
plt.hist(not_survived_age, bins=15, alpha=0.7, label='사망', color='red')
plt.title('생존자별 연령 분포')
plt.xlabel('연령')
plt.ylabel('빈도')
plt.legend()

# 성별별 요금 분포
plt.subplot(2, 3, 4)
male_fare = titanic[titanic['sex'] == 'male']['fare']
female_fare = titanic[titanic['sex'] == 'female']['fare']

plt.hist(male_fare, bins=15, alpha=0.7, label='남성', color='blue')
plt.hist(female_fare, bins=15, alpha=0.7, label='여성', color='pink')
plt.title('성별별 요금 분포')
plt.xlabel('요금')
plt.ylabel('빈도')
plt.legend()

# 등급별 연령 분포
plt.subplot(2, 3, 5)
for pclass in sorted(titanic_clean['pclass'].unique()):
    class_age = titanic_clean[titanic_clean['pclass'] == pclass]['age']
    plt.hist(class_age, bins=10, alpha=0.6, label=f'{pclass}등급')

plt.title('등급별 연령 분포')
plt.xlabel('연령')
plt.ylabel('빈도')
plt.legend()

# 요금 vs 연령 산점도
plt.subplot(2, 3, 6)
plt.scatter(titanic_clean['age'], titanic_clean['fare'], alpha=0.6)
plt.title('연령 vs 요금')
plt.xlabel('연령')
plt.ylabel('요금')

plt.tight_layout()
plt.show()
```

### 3-3. 상관관계 분석

```python
# 수치형 변수 간 상관관계
numerical_cols = ['age', 'fare', 'sibsp', 'parch']
correlation_matrix = titanic[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('수치형 변수 간 상관관계')
plt.tight_layout()
plt.show()
```

## 4. Sweetviz를 활용한 자동화된 EDA

### 4-1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate

# pip 업그레이드
python.exe -m pip install --upgrade pip

# Sweetviz 설치
pip install sweetviz

# 설치 확인
pip show sweetviz
```

### 4-2. 단일 데이터셋 분석

```python
import pandas as pd
import sweetviz as sv
import warnings

# 경고 메시지 무시
warnings.filterwarnings(action='ignore')

# 데이터 로드
df_train = pd.read_csv('./data/train.csv')

# Sweetviz 분석 실행
my_report = sv.analyze(df_train)

# HTML 리포트 생성
my_report.show_html('./reports/sweet_report.html')

# Jupyter Notebook에서 표시 (선택사항)
# my_report.show_notebook(layout='widescreen', scale=0.8)
```

### 4-3. 비교 분석

```python
# 훈련 데이터와 테스트 데이터 비교
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# 비교 분석 실행
compare_report = sv.compare([df_train, 'Train'], [df_test, 'Test'], target_feat='Survived')

# HTML 리포트 생성
compare_report.show_html('./reports/compare_report.html')

# Jupyter Notebook에서 표시 (선택사항)
# compare_report.show_notebook(layout='widescreen', scale=0.8)
```

### 4-4. 고급 Sweetviz 활용

```python
# 특정 변수만 분석
feature_config = sv.FeatureConfig(
    force_text=['Name', 'Ticket', 'Cabin'],  # 텍스트로 강제 변환
    force_cat=['Pclass', 'SibSp', 'Parch'],  # 범주형으로 강제 변환
    force_num=['Age', 'Fare']  # 수치형으로 강제 변환
)

# 설정을 적용한 분석
my_report = sv.analyze(df_train, feature_config=feature_config)
my_report.show_html('./reports/advanced_report.html')

# 특정 변수 제외
my_report = sv.analyze(df_train, target_feat='Survived', feat_cfg=feature_config)
my_report.show_html('./reports/target_analysis.html')
```

## 5. 실무 EDA 체크리스트

### 5-1. 데이터 품질 검사

```python
def data_quality_check(df):
    """데이터 품질 종합 검사"""
    print("=== 데이터 품질 검사 ===")
    
    # 1. 기본 정보
    print(f"데이터 크기: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. 결측치 분석
    missing_data = df.isnull().sum()
    missing_ratio = (missing_data / len(df)) * 100
    
    print("\n=== 결측치 분석 ===")
    for col, ratio in missing_ratio.items():
        if ratio > 0:
            print(f"{col}: {missing_data[col]}개 ({ratio:.1f}%)")
    
    # 3. 중복값 분석
    duplicates = df.duplicated().sum()
    print(f"\n중복값: {duplicates}개 ({duplicates/len(df)*100:.1f}%)")
    
    # 4. 데이터 타입 분석
    print("\n=== 데이터 타입 분석 ===")
    print(df.dtypes.value_counts())
    
    # 5. 수치형 변수 이상값 검사
    numeric_cols = df.select_dtypes(include=['number']).columns
    print("\n=== 이상값 검사 ===")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"{col}: {len(outliers)}개 이상값 ({len(outliers)/len(df)*100:.1f}%)")

# 데이터 품질 검사 실행
data_quality_check(titanic)
```

### 5-2. EDA 자동화 함수

```python
def automated_eda(df, target_column=None, save_path='./eda_results/'):
    """자동화된 EDA 실행"""
    import os
    
    # 결과 저장 폴더 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 기본 통계 요약
    print("=== 기본 통계 요약 ===")
    print(df.describe())
    
    # 2. 데이터 타입별 분석
    print("\n=== 데이터 타입별 분석 ===")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns
        print(f"{dtype}: {list(cols)}")
    
    # 3. Sweetviz 리포트 생성
    print("\n=== Sweetviz 리포트 생성 중... ===")
    if target_column and target_column in df.columns:
        report = sv.analyze(df, target_feat=target_column)
    else:
        report = sv.analyze(df)
    
    report.show_html(f'{save_path}eda_report.html')
    print(f"Sweetviz 리포트가 {save_path}eda_report.html에 저장되었습니다.")
    
    # 4. 주요 시각화 생성
    create_eda_visualizations(df, save_path)
    
    return report

def create_eda_visualizations(df, save_path):
    """EDA 시각화 생성"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 수치형 변수 히스토그램
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numeric_cols):
            plt.subplot(n_rows, n_cols, i+1)
            plt.hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'{col} 분포')
            plt.xlabel(col)
            plt.ylabel('빈도')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}numeric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 범주형 변수 막대 그래프
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 5*n_rows))
        for i, col in enumerate(categorical_cols):
            plt.subplot(n_rows, n_cols, i+1)
            value_counts = df[col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'{col} 분포 (Top 10)')
            plt.xlabel(col)
            plt.ylabel('빈도')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

# 자동화된 EDA 실행
eda_report = automated_eda(titanic, target_column='survived')
```

## 6. EDA 모범 사례

### 6-1. 체계적인 EDA 접근법

```python
def systematic_eda(df):
    """체계적인 EDA 접근법"""
    
    print("=== 1단계: 데이터 개요 ===")
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼 수: {df.shape[1]}")
    print(f"행 수: {df.shape[0]}")
    
    print("\n=== 2단계: 데이터 타입 분석 ===")
    print(df.dtypes.value_counts())
    
    print("\n=== 3단계: 결측치 분석 ===")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(missing_data[missing_data > 0])
    else:
        print("결측치 없음")
    
    print("\n=== 4단계: 수치형 변수 분석 ===")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print("\n=== 5단계: 범주형 변수 분석 ===")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  고유값 수: {df[col].nunique()}")
        print(f"  최빈값: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}")
        print(f"  상위 5개 값:")
        print(df[col].value_counts().head())

# 체계적인 EDA 실행
systematic_eda(titanic)
```

### 6-2. EDA 결과 문서화

```python
def document_eda_findings(df, findings_file='eda_findings.md'):
    """EDA 결과를 마크다운으로 문서화"""
    
    with open(findings_file, 'w', encoding='utf-8') as f:
        f.write("# EDA 분석 결과\n\n")
        
        # 기본 정보
        f.write("## 1. 데이터 개요\n")
        f.write(f"- 데이터 크기: {df.shape}\n")
        f.write(f"- 컬럼 수: {df.shape[1]}\n")
        f.write(f"- 행 수: {df.shape[0]}\n\n")
        
        # 데이터 타입
        f.write("## 2. 데이터 타입 분포\n")
        for dtype, count in df.dtypes.value_counts().items():
            f.write(f"- {dtype}: {count}개\n")
        f.write("\n")
        
        # 결측치
        f.write("## 3. 결측치 현황\n")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            for col, count in missing_data[missing_data > 0].items():
                ratio = (count / len(df)) * 100
                f.write(f"- {col}: {count}개 ({ratio:.1f}%)\n")
        else:
            f.write("- 결측치 없음\n")
        f.write("\n")
        
        # 수치형 변수 요약
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            f.write("## 4. 수치형 변수 요약\n")
            f.write("| 변수 | 평균 | 중앙값 | 표준편차 | 최소값 | 최대값 |\n")
            f.write("|------|------|--------|----------|--------|--------|\n")
            for col in numeric_cols:
                stats = df[col].describe()
                f.write(f"| {col} | {stats['mean']:.2f} | {stats['50%']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n")
            f.write("\n")
        
        # 범주형 변수 요약
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            f.write("## 5. 범주형 변수 요약\n")
            for col in categorical_cols:
                f.write(f"### {col}\n")
                f.write(f"- 고유값 수: {df[col].nunique()}\n")
                f.write(f"- 최빈값: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}\n")
                f.write("- 상위 5개 값:\n")
                for value, count in df[col].value_counts().head().items():
                    f.write(f"  - {value}: {count}개\n")
                f.write("\n")
    
    print(f"EDA 결과가 {findings_file}에 저장되었습니다.")

# EDA 결과 문서화
document_eda_findings(titanic)
```

## 마무리

EDA(탐색적 데이터 분석)는 데이터 과학 프로젝트의 핵심 단계입니다:

### 핵심 학습 내용
- **데이터 타입 분류**: 수치형, 범주형, 문자형, 불리언형의 특성 이해
- **기술 통계**: 중심 경향성과 분산성을 통한 데이터 특성 파악
- **시각화**: 박스 플롯, 히스토그램을 통한 분포 분석
- **자동화 도구**: Sweetviz를 활용한 효율적인 EDA

### 실무 적용
- **체계적인 접근**: 단계별 EDA 프로세스 구축
- **자동화**: 반복적인 분석 작업의 효율성 향상
- **문서화**: 분석 결과의 체계적인 기록과 공유
- **품질 관리**: 데이터 품질 검사와 이상값 탐지

EDA를 통해 데이터의 특성을 정확히 파악하고, 후속 분석 단계에서 올바른 방향으로 진행할 수 있는 기반을 마련할 수 있습니다.
