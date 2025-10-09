---
title: EDA 도구 활용 - 효율적인 탐색적 데이터 분석
categories:
- 1.TIL
- 1-7.DATA_ANALYSIS
tags:
- EDA
- sweetviz
- pandas-profiling
- 자동화
- 데이터분석
- 탐색적분석
toc: true
date: 2023-09-25 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# EDA 도구 활용 - 효율적인 탐색적 데이터 분석

## 개요

효율적인 탐색적 데이터 분석을 위한 자동화 도구를 학습합니다:

- **EDA 도구**: Sweetviz, Pandas Profiling, DataPrep, AutoViz
- **자동화 분석**: 데이터 개요, 통계 분석, 시각화
- **실무 활용**: 데이터 품질 검사, 모델링 전 탐색
- **고급 기법**: 커스터마이징, 대용량 데이터 처리

## 1. EDA 도구 개요

### 1-1. 주요 EDA 도구

```python
print("=== 주요 EDA 도구 ===")

print("1. Sweetviz")
print("   - 자동 EDA 보고서 생성")
print("   - HTML 형태의 인터랙티브 리포트")
print("   - 데이터셋 비교 분석 지원")
print("   - 타겟 변수 지정 가능")

print("\n2. Pandas Profiling")
print("   - 상세한 데이터 프로파일링")
print("   - 통계적 분석과 시각화")
print("   - 상관관계 분석")
print("   - 이상치 탐지")

print("\n3. DataPrep")
print("   - 빠른 데이터 탐색")
print("   - 데이터 클리닝 기능")
print("   - 시각화 자동 생성")
print("   - 웹 기반 인터페이스")

print("\n4. AutoViz")
print("   - 자동 시각화 생성")
print("   - 다양한 차트 타입")
print("   - 최적 차트 선택")
print("   - 빠른 인사이트 도출")
```

### 1-2. EDA 도구의 장점

```python
print("=== EDA 도구의 장점 ===")

print("1. 시간 절약")
print("   - 수동 분석 대비 빠른 인사이트 도출")
print("   - 반복적인 분석 작업 자동화")
print("   - 표준화된 분석 프로세스")

print("\n2. 일관성")
print("   - 동일한 분석 기준 적용")
print("   - 누락 없는 체계적 분석")
print("   - 재현 가능한 결과")

print("\n3. 완전성")
print("   - 놓치기 쉬운 패턴 발견")
print("   - 포괄적인 데이터 검사")
print("   - 숨겨진 관계성 탐지")

print("\n4. 시각화")
print("   - 자동 생성되는 차트와 그래프")
print("   - 인터랙티브 시각화")
print("   - 보고서 형태의 결과")
```

## 2. Sweetviz 활용

### 2-1. 설치 및 기본 사용

```python
import pandas as pd
import numpy as np
import sweetviz as sv
import warnings
warnings.filterwarnings(action='ignore')

# 샘플 데이터 생성 (실제 데이터가 없는 경우)
np.random.seed(42)
n_samples = 1000

# 다양한 타입의 데이터 생성
data = {
    'id': range(1, n_samples + 1),
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], n_samples),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples),
    'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

df_train = pd.DataFrame(data)

print("=== Sweetviz 기본 사용 ===")
print(f"데이터 크기: {df_train.shape}")
print(f"컬럼: {list(df_train.columns)}")

# 단일 데이터셋 분석
my_report = sv.analyze(df_train)
my_report.show_html('./sweet_report.html')

print("Sweetviz 리포트가 'sweet_report.html'로 생성되었습니다.")
```

### 2-2. 비교 분석

```python
# 테스트 데이터 생성 (약간 다른 분포)
df_test = pd.DataFrame({
    'id': range(1001, 1501),
    'age': np.random.normal(32, 8, 500),  # 약간 다른 분포
    'income': np.random.lognormal(9.5, 0.6, 500),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
    'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], 500),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5], 500),
    'target': np.random.choice([0, 1], 500, p=[0.6, 0.4])  # 다른 비율
})

# 두 데이터셋 비교
compare_report = sv.compare([df_train, 'Train'], [df_test, 'Test'])
compare_report.show_html('./compare_report.html')

print("비교 분석 리포트가 'compare_report.html'로 생성되었습니다.")
```

### 2-3. 타겟 변수 지정

```python
# 타겟 변수 지정하여 분석
target_report = sv.analyze(df_train, target_feat='target')
target_report.show_html('./target_report.html')

print("타겟 변수 분석 리포트가 'target_report.html'로 생성되었습니다.")
print("주요 분석 내용:")
print("- 타겟 변수 분포")
print("- 각 피처와 타겟의 관계")
print("- 클래스별 통계")
print("- 상관관계 분석")
```

### 2-4. 특정 변수만 분석

```python
# 특정 변수들만 선택하여 분석
feature_config = sv.FeatureConfig(
    skip=['id'],  # ID 컬럼 제외
    force_cat=['education', 'city'],  # 범주형으로 강제 변환
    force_num=['satisfaction']  # 수치형으로 강제 변환
)

filtered_report = sv.analyze(df_train, feature_config=feature_config)
filtered_report.show_html('./filtered_report.html')

print("필터링된 분석 리포트가 'filtered_report.html'로 생성되었습니다.")
```

## 3. Pandas Profiling 활용

### 3-1. 기본 사용

```python
try:
    from pandas_profiling import ProfileReport
    
    # 프로파일링 리포트 생성
    profile = ProfileReport(df_train, title="데이터 프로파일링 리포트")
    profile.to_file("profile_report.html")
    
    print("Pandas Profiling 리포트가 'profile_report.html'로 생성되었습니다.")
    
except ImportError:
    print("Pandas Profiling이 설치되지 않았습니다.")
    print("설치 명령: pip install pandas-profiling")
```

### 3-2. 설정 커스터마이징

```python
try:
    # 상세 설정
    profile = ProfileReport(
        df_train,
        title="커스텀 프로파일링 리포트",
        explorative=True,  # 탐색적 분석 활성화
        minimal=False,     # 상세 분석
        progress_bar=True, # 진행률 표시
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True}
        }
    )
    profile.to_file("custom_report.html")
    
    print("커스텀 프로파일링 리포트가 'custom_report.html'로 생성되었습니다.")
    
except ImportError:
    print("Pandas Profiling을 사용할 수 없습니다.")
```

### 3-3. Jupyter 노트북에서 표시

```python
# 노트북에서 직접 표시 (실제 노트북 환경에서만 동작)
try:
    # profile.to_notebook_iframe()  # 주석 해제하여 사용
    print("노트북에서 profile.to_notebook_iframe()을 사용하여 리포트를 표시할 수 있습니다.")
except:
    print("Jupyter 노트북 환경이 아닙니다.")
```

## 4. 주요 분석 내용

### 4-1. 데이터 개요

```python
def analyze_data_overview(df):
    """데이터 개요 분석"""
    print("=== 데이터 개요 ===")
    print(f"데이터셋 크기: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"변수 수: {df.shape[1]}")
    print(f"관측치 수: {df.shape[0]}")
    
    print("\n=== 변수 타입 ===")
    print(df.dtypes.value_counts())
    
    print("\n=== 중복 데이터 ===")
    duplicates = df.duplicated().sum()
    print(f"중복 행 수: {duplicates}개 ({duplicates/len(df)*100:.1f}%)")
    
    print("\n=== 결측값 ===")
    missing_data = df.isnull().sum()
    missing_ratio = (missing_data / len(df)) * 100
    for col, ratio in missing_ratio.items():
        if ratio > 0:
            print(f"{col}: {missing_data[col]}개 ({ratio:.1f}%)")
    
    if missing_ratio.sum() == 0:
        print("결측값 없음")

# 데이터 개요 분석 실행
analyze_data_overview(df_train)
```

### 4-2. 수치형 변수 분석

```python
def analyze_numeric_variables(df):
    """수치형 변수 분석"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print("=== 수치형 변수 분석 ===")
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  평균: {df[col].mean():.2f}")
        print(f"  중앙값: {df[col].median():.2f}")
        print(f"  표준편차: {df[col].std():.2f}")
        print(f"  최소값: {df[col].min():.2f}")
        print(f"  최대값: {df[col].max():.2f}")
        
        # 이상치 탐지 (IQR 방법)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"  이상치: {len(outliers)}개 ({len(outliers)/len(df)*100:.1f}%)")

# 수치형 변수 분석 실행
analyze_numeric_variables(df_train)
```

### 4-3. 범주형 변수 분석

```python
def analyze_categorical_variables(df):
    """범주형 변수 분석"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print("=== 범주형 변수 분석 ===")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  고유값 수: {df[col].nunique()}")
        print(f"  최빈값: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}")
        print(f"  최빈값 빈도: {df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0}")
        
        # 클래스 불균형 확인
        value_counts = df[col].value_counts()
        max_ratio = value_counts.iloc[0] / len(df)
        print(f"  최대 클래스 비율: {max_ratio:.1%}")
        
        if max_ratio > 0.8:
            print("  ⚠️  클래스 불균형 심각")
        elif max_ratio > 0.6:
            print("  ⚠️  클래스 불균형 있음")

# 범주형 변수 분석 실행
analyze_categorical_variables(df_train)
```

### 4-4. 변수 간 관계 분석

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_variable_relationships(df):
    """변수 간 관계 분석"""
    print("=== 변수 간 관계 분석 ===")
    
    # 수치형 변수 간 상관관계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        # 상관관계 히트맵
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('수치형 변수 간 상관관계')
        plt.tight_layout()
        plt.show()
        
        # 높은 상관관계 찾기
        high_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append((correlation_matrix.columns[i], 
                                    correlation_matrix.columns[j], corr_val))
        
        if high_corr:
            print("높은 상관관계 (|r| > 0.7):")
            for var1, var2, corr in high_corr:
                print(f"  {var1} - {var2}: {corr:.3f}")
        else:
            print("높은 상관관계 없음")
    
    # 범주형-수치형 변수 관계
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        print(f"\n범주형-수치형 변수 관계 분석 가능:")
        print(f"  범주형: {list(categorical_cols)}")
        print(f"  수치형: {list(numeric_cols)}")

# 변수 간 관계 분석 실행
analyze_variable_relationships(df_train)
```

## 5. 실무 활용 사례

### 5-1. 데이터 품질 검사

```python
def data_quality_check(df, target_col=None):
    """데이터 품질 검사"""
    print("=== 데이터 품질 검사 ===")
    
    # 1. 기본 정보
    print(f"데이터 크기: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. 결측값 패턴
    missing_data = df.isnull().sum()
    missing_ratio = (missing_data / len(df)) * 100
    
    print("\n결측값 패턴:")
    if missing_ratio.sum() > 0:
        for col, ratio in missing_ratio.items():
            if ratio > 0:
                print(f"  {col}: {missing_data[col]}개 ({ratio:.1f}%)")
    else:
        print("  결측값 없음")
    
    # 3. 이상치 분포
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\n이상치 분포:")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        outlier_ratio = len(outliers) / len(df) * 100
        print(f"  {col}: {len(outliers)}개 ({outlier_ratio:.1f}%)")
    
    # 4. 데이터 타입 일관성
    print("\n데이터 타입 일관성:")
    for col in df.columns:
        if df[col].dtype == 'object':
            # 숫자로 변환 가능한 문자열 확인
            try:
                pd.to_numeric(df[col], errors='raise')
                print(f"  {col}: 숫자로 변환 가능한 문자열 포함")
            except:
                pass
    
    # 5. 중복 데이터
    duplicates = df.duplicated().sum()
    print(f"\n중복 데이터: {duplicates}개 ({duplicates/len(df)*100:.1f}%)")
    
    # 6. 타겟 변수 분석 (지정된 경우)
    if target_col and target_col in df.columns:
        print(f"\n타겟 변수 ({target_col}) 분석:")
        if df[target_col].dtype in ['object', 'category']:
            print(f"  클래스 분포: {df[target_col].value_counts().to_dict()}")
        else:
            print(f"  평균: {df[target_col].mean():.2f}")
            print(f"  표준편차: {df[target_col].std():.2f}")

# 데이터 품질 검사 실행
data_quality_check(df_train, target_col='target')
```

### 5-2. 모델링 전 데이터 탐색

```python
def pre_modeling_exploration(df, target_col):
    """모델링 전 데이터 탐색"""
    print(f"=== 모델링 전 데이터 탐색 (타겟: {target_col}) ===")
    
    # 1. 타겟 변수 분포
    print("\n1. 타겟 변수 분포:")
    if df[target_col].dtype in ['object', 'category']:
        target_dist = df[target_col].value_counts()
        print(target_dist)
        
        # 클래스 불균형 확인
        max_ratio = target_dist.iloc[0] / len(df)
        print(f"최대 클래스 비율: {max_ratio:.1%}")
        
        if max_ratio > 0.8:
            print("⚠️  심각한 클래스 불균형 - 샘플링 전략 필요")
        elif max_ratio > 0.6:
            print("⚠️  클래스 불균형 - 가중치 조정 고려")
    else:
        print(f"평균: {df[target_col].mean():.2f}")
        print(f"표준편차: {df[target_col].std():.2f}")
        print(f"최소값: {df[target_col].min():.2f}")
        print(f"최대값: {df[target_col].max():.2f}")
    
    # 2. 피처-타겟 관계
    print("\n2. 피처-타겟 관계:")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = numeric_features.drop(target_col) if target_col in numeric_features else numeric_features
    
    for feature in numeric_features:
        if df[target_col].dtype in ['object', 'category']:
            # 범주형 타겟의 경우 그룹별 통계
            group_stats = df.groupby(target_col)[feature].agg(['mean', 'std'])
            print(f"  {feature}:")
            for target_val in group_stats.index:
                mean_val = group_stats.loc[target_val, 'mean']
                std_val = group_stats.loc[target_val, 'std']
                print(f"    {target_val}: {mean_val:.2f} ± {std_val:.2f}")
        else:
            # 수치형 타겟의 경우 상관관계
            corr = df[feature].corr(df[target_col])
            print(f"  {feature}: 상관계수 = {corr:.3f}")
    
    # 3. 피처 중요도 (간단한 방법)
    print("\n3. 피처 중요도 (상관관계 기반):")
    if df[target_col].dtype in ['object', 'category']:
        # 범주형 타겟의 경우 ANOVA F-score 계산
        from sklearn.feature_selection import f_classif
        X = df[numeric_features]
        y = df[target_col]
        f_scores, _ = f_classif(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': numeric_features,
            'f_score': f_scores
        }).sort_values('f_score', ascending=False)
        
        print(feature_importance)
    else:
        # 수치형 타겟의 경우 상관관계
        correlations = df[numeric_features].corrwith(df[target_col]).abs().sort_values(ascending=False)
        print(correlations)

# 모델링 전 데이터 탐색 실행
pre_modeling_exploration(df_train, 'target')
```

### 5-3. 데이터 전처리 계획 수립

```python
def preprocessing_plan(df):
    """데이터 전처리 계획 수립"""
    print("=== 데이터 전처리 계획 ===")
    
    preprocessing_steps = []
    
    # 1. 결측값 처리
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("1. 결측값 처리 필요:")
        for col, count in missing_data.items():
            if count > 0:
                ratio = count / len(df)
                if ratio > 0.5:
                    preprocessing_steps.append(f"  - {col}: 삭제 (결측률 {ratio:.1%})")
                elif ratio > 0.1:
                    preprocessing_steps.append(f"  - {col}: 고급 대체 방법 (결측률 {ratio:.1%})")
                else:
                    preprocessing_steps.append(f"  - {col}: 단순 대체 (결측률 {ratio:.1%})")
    
    # 2. 이상치 처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) / len(df) > 0.05:  # 5% 이상
            outlier_cols.append(col)
    
    if outlier_cols:
        preprocessing_steps.append("2. 이상치 처리:")
        for col in outlier_cols:
            preprocessing_steps.append(f"  - {col}: 이상치 탐지 및 처리")
    
    # 3. 데이터 타입 변환
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        preprocessing_steps.append("3. 범주형 변수 인코딩:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:
                preprocessing_steps.append(f"  - {col}: 원핫 인코딩 ({unique_count}개 카테고리)")
            else:
                preprocessing_steps.append(f"  - {col}: 타겟 인코딩 또는 삭제 고려 ({unique_count}개 카테고리)")
    
    # 4. 스케일링
    if len(numeric_cols) > 0:
        preprocessing_steps.append("4. 수치형 변수 스케일링:")
        preprocessing_steps.append("  - StandardScaler 또는 MinMaxScaler 적용")
    
    # 5. 피처 선택
    preprocessing_steps.append("5. 피처 선택:")
    preprocessing_steps.append("  - 상관관계가 높은 피처 제거")
    preprocessing_steps.append("  - 분산이 낮은 피처 제거")
    preprocessing_steps.append("  - 도메인 지식 기반 피처 선택")
    
    # 결과 출력
    if preprocessing_steps:
        for step in preprocessing_steps:
            print(step)
    else:
        print("전처리가 필요하지 않습니다.")

# 전처리 계획 수립
preprocessing_plan(df_train)
```

## 6. 고급 활용 기법

### 6-1. 커스텀 설정

```python
def advanced_sweetviz_analysis(df):
    """고급 Sweetviz 분석"""
    print("=== 고급 Sweetviz 분석 ===")
    
    # 커스텀 설정
    feature_config = sv.FeatureConfig(
        skip=['id'],  # ID 컬럼 제외
        force_cat=['education', 'city'],  # 범주형으로 강제 변환
        force_num=['satisfaction'],  # 수치형으로 강제 변환
        force_text=[],  # 텍스트로 강제 변환
        force_date=[]  # 날짜로 강제 변환
    )
    
    # 고급 분석 설정
    advanced_report = sv.analyze(
        df,
        target_feat='target',
        feature_config=feature_config,
        pairwise_analysis='on'  # 쌍별 분석 활성화
    )
    
    advanced_report.show_html('./advanced_report.html')
    print("고급 분석 리포트가 'advanced_report.html'로 생성되었습니다.")

# 고급 분석 실행
advanced_sweetviz_analysis(df_train)
```

### 6-2. 대용량 데이터 처리

```python
def large_data_analysis(df, sample_size=10000):
    """대용량 데이터 분석"""
    print("=== 대용량 데이터 분석 ===")
    
    if len(df) > sample_size:
        print(f"데이터 크기: {len(df)} → {sample_size} (샘플링)")
        
        # 계층적 샘플링 (타겟 변수가 있는 경우)
        if 'target' in df.columns:
            sample_df = df.groupby('target').apply(
                lambda x: x.sample(min(len(x), sample_size // df['target'].nunique()))
            ).reset_index(drop=True)
        else:
            sample_df = df.sample(n=sample_size, random_state=42)
        
        # 샘플 데이터로 분석
        sample_report = sv.analyze(sample_df)
        sample_report.show_html('./sample_report.html')
        print("샘플 분석 리포트가 'sample_report.html'로 생성되었습니다.")
        
        return sample_df
    else:
        print(f"데이터 크기가 작아 샘플링이 필요하지 않습니다: {len(df)}")
        return df

# 대용량 데이터 분석 실행
sample_data = large_data_analysis(df_train)
```

### 6-3. 시계열 데이터 분석

```python
def time_series_analysis():
    """시계열 데이터 분석"""
    print("=== 시계열 데이터 분석 ===")
    
    # 시계열 데이터 생성
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    time_series_data = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(365)) + 100,
        'category': np.random.choice(['A', 'B', 'C'], 365),
        'target': np.random.choice([0, 1], 365, p=[0.6, 0.4])
    })
    
    print(f"시계열 데이터 크기: {time_series_data.shape}")
    
    # 시계열 특화 분석
    time_series_report = sv.analyze(
        time_series_data,
        target_feat='target',
        pairwise_analysis='on'
    )
    
    time_series_report.show_html('./timeseries_report.html')
    print("시계열 분석 리포트가 'timeseries_report.html'로 생성되었습니다.")
    
    return time_series_data

# 시계열 데이터 분석 실행
ts_data = time_series_analysis()
```

## 7. 보고서 활용

### 7-1. HTML 보고서 활용

```python
def html_report_utilization():
    """HTML 보고서 활용 방법"""
    print("=== HTML 보고서 활용 ===")
    
    print("1. HTML 보고서의 장점:")
    print("   - 인터랙티브한 탐색")
    print("   - 클라이언트와의 공유 용이")
    print("   - 팀 내 협업 도구")
    print("   - 웹 브라우저에서 직접 확인")
    
    print("\n2. 활용 방법:")
    print("   - 데이터 품질 검사 결과 공유")
    print("   - 모델링 전 데이터 탐색 보고")
    print("   - 정기적인 데이터 모니터링")
    print("   - 새로운 데이터셋 검증")
    
    print("\n3. 보고서 구성 요소:")
    print("   - 데이터 개요 및 통계")
    print("   - 변수별 상세 분석")
    print("   - 변수 간 관계 분석")
    print("   - 이상치 및 결측값 정보")
    print("   - 시각화 차트")

html_report_utilization()
```

### 7-2. 자동화된 EDA 파이프라인

```python
def automated_eda_pipeline(df, project_name="EDA_Analysis"):
    """자동화된 EDA 파이프라인"""
    print(f"=== 자동화된 EDA 파이프라인: {project_name} ===")
    
    import os
    from datetime import datetime
    
    # 결과 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./eda_reports/{project_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 기본 분석
    print("1. 기본 분석 실행 중...")
    basic_report = sv.analyze(df)
    basic_report.show_html(f"{output_dir}/basic_analysis.html")
    
    # 2. 타겟 변수 분석 (있는 경우)
    target_cols = [col for col in df.columns if 'target' in col.lower() or 'label' in col.lower()]
    if target_cols:
        print(f"2. 타겟 변수 분석 실행 중... (타겟: {target_cols[0]})")
        target_report = sv.analyze(df, target_feat=target_cols[0])
        target_report.show_html(f"{output_dir}/target_analysis.html")
    
    # 3. 데이터 품질 리포트
    print("3. 데이터 품질 리포트 생성 중...")
    quality_summary = {
        'data_size': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_data': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # 품질 리포트를 텍스트 파일로 저장
    with open(f"{output_dir}/quality_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"데이터 품질 요약\n")
        f.write(f"================\n\n")
        f.write(f"데이터 크기: {quality_summary['data_size']}\n")
        f.write(f"메모리 사용량: {quality_summary['memory_usage']:.2f} MB\n")
        f.write(f"중복 행 수: {quality_summary['duplicate_rows']}\n\n")
        f.write(f"결측값:\n")
        for col, count in quality_summary['missing_data'].items():
            if count > 0:
                f.write(f"  {col}: {count}개\n")
        f.write(f"\n데이터 타입:\n")
        for col, dtype in quality_summary['data_types'].items():
            f.write(f"  {col}: {dtype}\n")
    
    print(f"EDA 파이프라인 완료!")
    print(f"결과 저장 위치: {output_dir}")
    print(f"생성된 파일:")
    print(f"  - basic_analysis.html")
    if target_cols:
        print(f"  - target_analysis.html")
    print(f"  - quality_summary.txt")
    
    return output_dir

# 자동화된 EDA 파이프라인 실행
output_directory = automated_eda_pipeline(df_train, "Sample_Project")
```

## 마무리

EDA 도구를 효과적으로 활용하면 데이터 분석의 효율성을 크게 향상시킬 수 있습니다:

### 핵심 학습 내용
- **EDA 도구**: Sweetviz, Pandas Profiling의 특성과 활용법
- **자동화 분석**: 데이터 개요, 통계 분석, 시각화 자동 생성
- **실무 활용**: 데이터 품질 검사, 모델링 전 탐색, 전처리 계획
- **고급 기법**: 커스터마이징, 대용량 데이터 처리, 시계열 분석

### 실무 적용
- **자동화 파이프라인**: 반복적인 EDA 작업의 자동화
- **품질 관리**: 체계적인 데이터 품질 검사
- **협업 도구**: HTML 보고서를 통한 팀 내 공유
- **지속적 모니터링**: 정기적인 데이터 품질 모니터링

EDA 도구는 데이터 분석의 시작점으로, 도메인 지식과 결합하여 더 깊이 있는 인사이트를 도출할 수 있습니다.
