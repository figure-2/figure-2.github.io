---
title: 머신러닝 전처리 - 데이터 품질 향상을 위한 핵심 기법
categories:
- 1.TIL
- 1-8.MACHINE_LEARNING
tags:
- 머신러닝
- 전처리
- 결측치
- 이상치
- 인코딩
- 데이터정제
toc: true
date: 2023-09-13 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# 머신러닝 전처리 - 데이터 품질 향상을 위한 핵심 기법

## 개요

머신러닝 모델의 성능을 높이기 위한 데이터 전처리 기법을 학습합니다:

- **결측치 처리**: 다양한 결측치 처리 방법과 시각화
- **이상치 처리**: IQR 방법을 통한 이상치 탐지와 제거
- **인코딩**: 레이블 인코딩과 원-핫 인코딩
- **실무 활용**: 실제 데이터에 적용하는 전처리 파이프라인

## 1. 결측치 처리

### 1-1. 결측치 확인

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 샘플 데이터 생성 (결측치 포함)
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], n_samples),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples),
    'temp': np.random.normal(20, 5, n_samples),
    'humidity': np.random.normal(60, 15, n_samples)
}

df = pd.DataFrame(data)

# 의도적으로 결측치 생성
df.loc[df.sample(50).index, 'age'] = np.nan
df.loc[df.sample(30).index, 'income'] = np.nan
df.loc[df.sample(20).index, 'education'] = np.nan
df.loc[df.sample(15).index, 'temp'] = np.nan
df.loc[df.sample(10).index, 'humidity'] = np.nan

print("=== 결측치 확인 ===")
print("1. 데이터 정보 확인")
print(df.info())

print("\n2. 결측값 수 확인")
missing_data = df.isnull().sum()
print(missing_data)

print("\n3. 결측값 비율 확인")
missing_ratio = (missing_data / len(df)) * 100
for col, ratio in missing_ratio.items():
    if ratio > 0:
        print(f"{col}: {missing_data[col]}개 ({ratio:.1f}%)")
```

### 1-2. 결측치 시각화

```python
# 결측치 시각화 (missingno 패키지 사용)
try:
    import missingno as msno
    
    print("=== 결측치 시각화 ===")
    
    # 1. 매트릭스 시각화
    plt.figure(figsize=(12, 8))
    msno.matrix(df)
    plt.title('Missing Data Matrix')
    plt.show()
    
    # 2. 바 차트
    plt.figure(figsize=(10, 6))
    msno.bar(df)
    plt.title('Missing Data Bar Chart')
    plt.show()
    
    # 3. 히트맵
    plt.figure(figsize=(10, 6))
    msno.heatmap(df)
    plt.title('Missing Data Heatmap')
    plt.show()
    
except ImportError:
    print("missingno 패키지가 설치되지 않았습니다.")
    print("설치 명령: pip install missingno")
    
    # 대안: matplotlib을 사용한 결측치 시각화
    plt.figure(figsize=(12, 6))
    missing_data.plot(kind='bar')
    plt.title('Missing Data Count by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

### 1-3. 결측치 제거

```python
print("=== 결측치 제거 방법 ===")

# 1. 모든 컬럼이 결측값인 행 제거
df_drop_all = df.dropna(how='all')
print(f"모든 컬럼이 결측값인 행 제거 후: {len(df_drop_all)}행")

# 2. 세 개 이상의 컬럼이 결측값인 행 제거
df_drop_3 = df.dropna(thresh=len(df.columns)-3)
print(f"세 개 이상의 컬럼이 결측값인 행 제거 후: {len(df_drop_3)}행")

# 3. 특정 컬럼(temp)이 결측값인 행 제거
df_drop_temp = df.dropna(subset=['temp'])
print(f"temp 컬럼이 결측값인 행 제거 후: {len(df_drop_temp)}행")

# 4. 한 컬럼이라도 결측치가 있는 행 제거
df_drop_any = df.dropna(how='any')
print(f"한 컬럼이라도 결측치가 있는 행 제거 후: {len(df_drop_any)}행")

# 결측치 제거 전후 비교
print(f"\n원본 데이터: {len(df)}행")
print(f"완전 제거 후: {len(df_drop_any)}행")
print(f"데이터 손실률: {(len(df) - len(df_drop_any)) / len(df) * 100:.1f}%")
```

### 1-4. 결측치 채우기

```python
print("=== 결측치 채우기 방법 ===")

# 1. 특정값으로 대치
print("1. 특정값(0)으로 대치")
df_0_all = df.fillna(0)
print(f"결측치를 0으로 대치 후 결측치 수: {df_0_all.isnull().sum().sum()}")

# 특정 컬럼만 0으로 대치
df_0_select = df.fillna({'temp': 0, 'humidity': 0})
print(f"temp, humidity만 0으로 대치 후 결측치 수: {df_0_select.isnull().sum().sum()}")

# 2. 평균값 대치
print("\n2. 평균값 대치")
df_mean_all = df.fillna(df.mean(numeric_only=True))
print(f"수치형 컬럼을 평균값으로 대치 후 결측치 수: {df_mean_all.isnull().sum().sum()}")

# 특정 컬럼만 평균값으로 대치
df_mean_select = df.fillna({'age': df['age'].mean(), 'income': df['income'].mean()})
print(f"age, income만 평균값으로 대치 후 결측치 수: {df_mean_select.isnull().sum().sum()}")

# 3. 중앙값 대치
print("\n3. 중앙값 대치")
df_median_all = df.fillna(df.median(numeric_only=True))
print(f"수치형 컬럼을 중앙값으로 대치 후 결측치 수: {df_median_all.isnull().sum().sum()}")

# 4. 최빈값 대치
print("\n4. 최빈값 대치")
# 범주형 변수의 경우 최빈값으로 대치
df_mode_select = df.fillna({'education': df['education'].mode()[0]})
print(f"education을 최빈값으로 대치 후 결측치 수: {df_mode_select.isnull().sum().sum()}")

# 5. 전진 채우기 (Forward Fill)
print("\n5. 전진 채우기")
df_ffill = df.fillna(method='ffill')
print(f"전진 채우기 후 결측치 수: {df_ffill.isnull().sum().sum()}")

# 6. 후진 채우기 (Backward Fill)
print("\n6. 후진 채우기")
df_bfill = df.fillna(method='bfill')
print(f"후진 채우기 후 결측치 수: {df_bfill.isnull().sum().sum()}")
```

### 1-5. 고급 결측치 처리

```python
from sklearn.impute import SimpleImputer, KNNImputer

print("=== 고급 결측치 처리 방법 ===")

# 1. SimpleImputer 사용
print("1. SimpleImputer 사용")
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_imputed = df.copy()
df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print(f"SimpleImputer 사용 후 결측치 수: {df_imputed.isnull().sum().sum()}")

# 2. KNN Imputer 사용
print("\n2. KNN Imputer 사용")
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = df.copy()
df_knn[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
print(f"KNN Imputer 사용 후 결측치 수: {df_knn.isnull().sum().sum()}")

# 3. 조건부 평균 대치
print("\n3. 조건부 평균 대치")
df_conditional = df.copy()
# education별 age 평균으로 대치
education_age_mean = df.groupby('education')['age'].mean()
for edu in df['education'].dropna().unique():
    mask = (df['education'] == edu) & (df['age'].isnull())
    df_conditional.loc[mask, 'age'] = education_age_mean[edu]
print(f"조건부 평균 대치 후 age 결측치 수: {df_conditional['age'].isnull().sum()}")
```

## 2. 이상치 처리

### 2-1. 이상치 확인

```python
# Titanic 데이터셋 사용
titanic = sns.load_dataset('titanic')

print("=== 이상치 확인 ===")

# 1. 박스플롯으로 이상치 시각화
plt.figure(figsize=(15, 10))

# 전체 수치형 컬럼 박스플롯
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    titanic.boxplot(column=col)
    plt.title(f'{col} Boxplot')

plt.tight_layout()
plt.show()

# 2. 특정 컬럼(fare) 박스플롯
plt.figure(figsize=(8, 6))
titanic.boxplot(column=['fare'])
plt.title('Fare Boxplot')
plt.show()

# 3. 통계적 이상치 확인
print("\n=== 통계적 이상치 확인 ===")
for col in ['age', 'fare']:
    if col in titanic.columns:
        Q1 = titanic[col].quantile(0.25)
        Q3 = titanic[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = titanic[(titanic[col] < lower_bound) | (titanic[col] > upper_bound)]
        
        print(f"\n{col} 컬럼 이상치:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  하한: {lower_bound:.2f}, 상한: {upper_bound:.2f}")
        print(f"  이상치 개수: {len(outliers)}개 ({len(outliers)/len(titanic)*100:.1f}%)")
```

### 2-2. 이상치 제거

```python
print("=== 이상치 제거 ===")

# IQR 방법으로 이상치 제거
def remove_outliers_iqr(df, column, multiplier=1.5):
    """IQR 방법으로 이상치 제거"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # 이상치가 아닌 데이터만 반환
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    return df[mask]

# fare 컬럼의 이상치 제거
print("1. fare 컬럼 이상치 제거")
original_size = len(titanic)
df_no_outliers = remove_outliers_iqr(titanic, 'fare', multiplier=1.5)
print(f"원본 크기: {original_size}")
print(f"이상치 제거 후 크기: {len(df_no_outliers)}")
print(f"제거된 데이터: {original_size - len(df_no_outliers)}개 ({((original_size - len(df_no_outliers))/original_size)*100:.1f}%)")

# 제거 전후 분포 비교
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 원본 데이터
axes[0].boxplot(titanic['fare'].dropna())
axes[0].set_title('Original Fare Distribution')
axes[0].set_ylabel('Fare')

# 이상치 제거 후
axes[1].boxplot(df_no_outliers['fare'].dropna())
axes[1].set_title('After Outlier Removal')
axes[1].set_ylabel('Fare')

plt.tight_layout()
plt.show()

# 2. Z-score 방법으로 이상치 제거
from scipy import stats

def remove_outliers_zscore(df, column, threshold=3):
    """Z-score 방법으로 이상치 제거"""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores < threshold
    return df[df[column].notna()][mask]

print("\n2. Z-score 방법으로 이상치 제거")
df_zscore = remove_outliers_zscore(titanic, 'fare', threshold=3)
print(f"Z-score 방법 제거 후 크기: {len(df_zscore)}")
```

### 2-3. 이상치 처리 전략

```python
print("=== 이상치 처리 전략 ===")

# 1. 이상치 제거
print("1. 이상치 제거")
print("   - 장점: 모델 성능 향상 가능")
print("   - 단점: 중요한 정보 손실 가능")
print("   - 사용 시기: 명백한 오류인 경우")

# 2. 이상치 변환
print("\n2. 이상치 변환")
# 로그 변환
titanic['fare_log'] = np.log1p(titanic['fare'])  # log1p는 log(1+x)로 0값 처리

# 제한 변환 (Winsorization)
def winsorize_data(series, limits=(0.05, 0.05)):
    """Winsorization: 극값을 제한값으로 대체"""
    return series.clip(lower=series.quantile(limits[0]), 
                      upper=series.quantile(1-limits[1]))

titanic['fare_winsorized'] = winsorize_data(titanic['fare'])

# 변환 전후 비교
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].boxplot(titanic['fare'].dropna())
axes[0].set_title('Original Fare')

axes[1].boxplot(titanic['fare_log'].dropna())
axes[1].set_title('Log Transformed Fare')

axes[2].boxplot(titanic['fare_winsorized'].dropna())
axes[2].set_title('Winsorized Fare')

plt.tight_layout()
plt.show()

# 3. 이상치를 별도 카테고리로 처리
print("\n3. 이상치를 별도 카테고리로 처리")
# 이상치 여부를 나타내는 새로운 피처 생성
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

titanic['fare_outlier'] = ((titanic['fare'] < lower_bound) | 
                          (titanic['fare'] > upper_bound)).astype(int)

print(f"이상치로 분류된 데이터: {titanic['fare_outlier'].sum()}개")
```

## 3. 인코딩

### 3-1. 레이블 인코딩

```python
from sklearn.preprocessing import LabelEncoder

print("=== 레이블 인코딩 ===")

# 1. 기본 레이블 인코딩
print("1. 기본 레이블 인코딩")
train_data = ['TV', '냉장고', '전자렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# LabelEncoder 객체 생성
le = LabelEncoder()
encoded_labels = le.fit_transform(train_data)

print(f"원본 데이터: {train_data}")
print(f"인코딩 결과: {encoded_labels}")
print(f"클래스 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 2. 새로운 데이터에 적용
print("\n2. 새로운 데이터에 적용")
real_data = ['TV', '냉장고', '전자렌지', '냉장고']
real_encoded = le.transform(real_data)
print(f"새로운 데이터: {real_data}")
print(f"인코딩 결과: {real_encoded}")

# 3. DataFrame에 적용
print("\n3. DataFrame에 적용")
train_df = pd.DataFrame({
    'price': [120, 400, 23, 300, 12, 16, 22, 24],
    'item': ['TV', '냉장고', '전자렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
})

# LabelEncoder 적용
le_df = LabelEncoder()
train_df['item_label'] = le_df.fit_transform(train_df['item'])
print("훈련 데이터:")
print(train_df)

# 실제 데이터에 적용
real_df = pd.DataFrame({'item': ['TV', '냉장고', '전자렌지']})
real_df['item_label'] = le_df.transform(real_df['item'])
print("\n실제 데이터:")
print(real_df)
```

### 3-2. 원-핫 인코딩

```python
print("=== 원-핫 인코딩 ===")

# 1. Pandas get_dummies 사용
print("1. Pandas get_dummies 사용")
titanic = sns.load_dataset('titanic')

# 특정 컬럼만 원-핫 인코딩
categorical_cols = ['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive']
dummy_df = pd.get_dummies(titanic, columns=categorical_cols)

print(f"원본 데이터 크기: {titanic.shape}")
print(f"원-핫 인코딩 후 크기: {dummy_df.shape}")

# 2. Scikit-learn OneHotEncoder 사용
print("\n2. Scikit-learn OneHotEncoder 사용")
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

# ColumnTransformer 생성
transformer = make_column_transformer(
    (OneHotEncoder(), categorical_cols),  # 해당 컬럼에 원-핫 인코딩 적용
    remainder='passthrough'  # 나머지 컬럼은 그대로 통과
)

# 변환 적용
transformer.fit(titanic)
trans_data = transformer.transform(titanic)

# DataFrame으로 변환
trans_df = pd.DataFrame(
    data=trans_data.toarray() if hasattr(trans_data, 'toarray') else trans_data,
    columns=transformer.get_feature_names_out()
)

print(f"Scikit-learn 원-핫 인코딩 후 크기: {trans_df.shape}")

# 3. 원-핫 인코딩 결과 비교
print("\n3. 원-핫 인코딩 결과 비교")
print("Pandas get_dummies vs Scikit-learn OneHotEncoder 차이점:")
print("- Scikit-learn은 결측치(NaN)도 별도 카테고리로 인코딩")
print("- Pandas는 결측치를 무시하고 인코딩")
print("- Scikit-learn은 파이프라인에서 일관성 있게 사용 가능")
```

### 3-3. 고급 인코딩 기법

```python
print("=== 고급 인코딩 기법 ===")

# 1. 타겟 인코딩 (Target Encoding)
print("1. 타겟 인코딩")
# 범주형 변수의 각 카테고리를 타겟 변수의 평균으로 인코딩
def target_encoding(df, categorical_col, target_col):
    """타겟 인코딩 함수"""
    target_mean = df.groupby(categorical_col)[target_col].mean()
    return df[categorical_col].map(target_mean)

# 예제 데이터
sample_df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
    'target': [1, 0, 1, 0, 1, 0, 1, 0]
})

sample_df['category_target_encoded'] = target_encoding(sample_df, 'category', 'target')
print("타겟 인코딩 결과:")
print(sample_df)

# 2. 빈도 인코딩 (Frequency Encoding)
print("\n2. 빈도 인코딩")
def frequency_encoding(df, categorical_col):
    """빈도 인코딩 함수"""
    freq_map = df[categorical_col].value_counts().to_dict()
    return df[categorical_col].map(freq_map)

sample_df['category_freq_encoded'] = frequency_encoding(sample_df, 'category')
print("빈도 인코딩 결과:")
print(sample_df)

# 3. 순서 인코딩 (Ordinal Encoding)
print("\n3. 순서 인코딩")
from sklearn.preprocessing import OrdinalEncoder

# 순서가 있는 범주형 변수
ordinal_data = pd.DataFrame({
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master']
})

# 순서 정의
education_order = [['High School', 'Bachelor', 'Master', 'PhD']]

ordinal_encoder = OrdinalEncoder(categories=education_order)
ordinal_data['education_encoded'] = ordinal_encoder.fit_transform(ordinal_data[['education']])

print("순서 인코딩 결과:")
print(ordinal_data)
```

## 4. 전처리 파이프라인

### 4-1. 통합 전처리 파이프라인

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("=== 통합 전처리 파이프라인 ===")

def create_preprocessing_pipeline():
    """전처리 파이프라인 생성"""
    
    # 수치형 컬럼 전처리
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 범주형 컬럼 전처리
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 컬럼별 전처리 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['age', 'fare']),
            ('cat', categorical_transformer, ['sex', 'embarked', 'class'])
        ]
    )
    
    return preprocessor

# 파이프라인 생성 및 적용
preprocessor = create_preprocessing_pipeline()

# Titanic 데이터 준비
titanic_clean = titanic[['age', 'fare', 'sex', 'embarked', 'class', 'survived']].copy()

# 전처리 적용
X = titanic_clean.drop('survived', axis=1)
y = titanic_clean['survived']

# 전처리 파이프라인 적용
X_processed = preprocessor.fit_transform(X)

print(f"원본 데이터 크기: {X.shape}")
print(f"전처리 후 데이터 크기: {X_processed.shape}")
print(f"전처리된 데이터 타입: {type(X_processed)}")
```

### 4-2. 실무 전처리 체크리스트

```python
print("=== 실무 전처리 체크리스트 ===")

def preprocessing_checklist(df):
    """전처리 체크리스트"""
    
    print("1. 데이터 품질 검사")
    print(f"   - 데이터 크기: {df.shape}")
    print(f"   - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 결측치 검사
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("   - 결측치 발견:")
        for col, count in missing_data.items():
            if count > 0:
                ratio = count / len(df) * 100
                print(f"     {col}: {count}개 ({ratio:.1f}%)")
    else:
        print("   - 결측치 없음")
    
    # 중복 데이터 검사
    duplicates = df.duplicated().sum()
    print(f"   - 중복 데이터: {duplicates}개")
    
    print("\n2. 데이터 타입 검사")
    for col, dtype in df.dtypes.items():
        print(f"   - {col}: {dtype}")
    
    print("\n3. 이상치 검사")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        outlier_ratio = len(outliers) / len(df) * 100
        print(f"   - {col}: {len(outliers)}개 이상치 ({outlier_ratio:.1f}%)")
    
    print("\n4. 범주형 변수 검사")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"   - {col}: {unique_count}개 고유값")
        if unique_count > 20:
            print(f"     ⚠️ 고유값이 많음 - 인코딩 전략 재검토 필요")

# 체크리스트 실행
preprocessing_checklist(titanic)
```

### 4-3. 전처리 성능 최적화

```python
print("=== 전처리 성능 최적화 ===")

# 1. 메모리 최적화
def optimize_memory(df):
    """메모리 사용량 최적화"""
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # 정수형 컬럼 최적화
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].max() < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                df[col] = df[col].astype('int32')
    
    # 실수형 컬럼 최적화
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # 범주형 컬럼 최적화
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (original_memory - optimized_memory) / original_memory * 100
    
    print(f"메모리 사용량 최적화:")
    print(f"  원본: {original_memory:.2f} MB")
    print(f"  최적화 후: {optimized_memory:.2f} MB")
    print(f"  절약: {reduction:.1f}%")
    
    return df

# 메모리 최적화 실행
titanic_optimized = optimize_memory(titanic.copy())
```

## 마무리

머신러닝 전처리의 핵심 기법을 학습했습니다:

### 핵심 학습 내용
- **결측치 처리**: 제거, 대치, 고급 방법론
- **이상치 처리**: IQR, Z-score, 변환 방법
- **인코딩**: 레이블, 원-핫, 고급 인코딩 기법
- **파이프라인**: 체계적인 전처리 프로세스

### 실무 적용
- **데이터 품질**: 체계적인 데이터 검사와 최적화
- **성능 향상**: 메모리 최적화와 처리 속도 개선
- **일관성**: 재현 가능한 전처리 파이프라인
- **확장성**: 새로운 데이터에 적용 가능한 전처리 방법

전처리는 머신러닝 모델의 성능에 직접적인 영향을 미치므로, 도메인 지식과 함께 신중하게 수행해야 합니다.
