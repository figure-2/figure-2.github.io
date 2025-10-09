---
title: Apache Spark 상세 내용
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- Apache Spark
- DataFrame
- 윈도우함수
- UDF
- 캐싱
- 스트리밍
- MLlib
- 성능최적화
toc: true
date: 2023-11-07 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# Spark 상세 내용

> 231114~231124 학습한 내용 정리

## Apache Spark 개요

### 정의
- **Apache Spark**: 대용량 데이터 처리를 위한 통합 분석 엔진
- **인메모리 처리**: 메모리 기반의 빠른 데이터 처리
- **분산 처리**: 여러 노드에서 병렬로 데이터 처리
- **통합 플랫폼**: 배치, 스트리밍, 머신러닝, 그래프 처리 통합

### 특징
- **고성능**: 인메모리 처리로 빠른 성능
- **확장성**: 수천 개의 노드로 확장 가능
- **유연성**: 다양한 데이터 소스 지원
- **통합성**: 여러 작업을 하나의 플랫폼에서 처리

### 장점
- **속도**: 메모리 기반 처리로 빠른 성능
- **편의성**: 다양한 프로그래밍 언어 지원
- **통합**: 배치, 스트리밍, ML, 그래프 처리 통합
- **생태계**: 풍부한 라이브러리와 도구

## Spark 설치 및 설정

### 1. Spark 설치
```bash
# Spark 다운로드
wget https://archive.apache.org/dist/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz

# 압축 해제
tar -xzf spark-3.4.0-bin-hadoop3.tgz

# 환경변수 설정
export SPARK_HOME=/path/to/spark-3.4.0-bin-hadoop3
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
```

### 2. PySpark 설정
```python
# PySpark 설치
pip install pyspark

# PySpark 기본 사용법
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd

# SparkSession 생성
spark = SparkSession.builder \
    .appName("SparkExample") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# SparkContext 확인
sc = spark.sparkContext
print(f"Spark 버전: {sc.version}")
print(f"애플리케이션 이름: {sc.appName}")
```

## Spark 기본 사용법

### 1. DataFrame 생성 및 조작
```python
# 샘플 데이터 생성
data = [
    ("Alice", 25, "Engineer", 50000),
    ("Bob", 30, "Manager", 60000),
    ("Charlie", 35, "Analyst", 55000),
    ("Diana", 28, "Engineer", 52000),
    ("Eve", 32, "Manager", 65000)
]

# 스키마 정의
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("salary", IntegerType(), True)
])

# DataFrame 생성
df = spark.createDataFrame(data, schema)

# DataFrame 기본 정보
df.show()
df.printSchema()
df.describe().show()

# 컬럼 선택
df.select("name", "salary").show()

# 조건 필터링
df.filter(df.age > 30).show()
df.filter((df.job == "Engineer") & (df.salary > 50000)).show()

# 정렬
df.orderBy(df.salary.desc()).show()

# 그룹화 및 집계
df.groupBy("job").agg(
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary"),
    count("*").alias("count")
).show()
```

### 2. 데이터 변환
```python
# 컬럼 추가
df_with_bonus = df.withColumn("bonus", df.salary * 0.1)

# 컬럼 이름 변경
df_renamed = df.withColumnRenamed("job", "position")

# 컬럼 삭제
df_dropped = df.drop("age")

# 조건부 컬럼 추가
df_with_grade = df.withColumn(
    "grade",
    when(df.salary >= 60000, "A")
    .when(df.salary >= 55000, "B")
    .otherwise("C")
)

# 문자열 함수
df_with_upper = df.withColumn("name_upper", upper(df.name))

# 날짜 함수
from datetime import datetime
df_with_date = df.withColumn("created_date", current_timestamp())

# 수학 함수
df_with_math = df.withColumn("salary_sqrt", sqrt(df.salary))
```

### 3. 데이터 조인
```python
# 두 번째 DataFrame 생성
dept_data = [
    ("Engineer", "Engineering"),
    ("Manager", "Management"),
    ("Analyst", "Analytics")
]

dept_schema = StructType([
    StructField("job", StringType(), True),
    StructField("department", StringType(), True)
])

dept_df = spark.createDataFrame(dept_data, dept_schema)

# 내부 조인
joined_df = df.join(dept_df, "job", "inner")
joined_df.show()

# 왼쪽 외부 조인
left_joined_df = df.join(dept_df, "job", "left")
left_joined_df.show()

# 오른쪽 외부 조인
right_joined_df = df.join(dept_df, "job", "right")
right_joined_df.show()

# 전체 외부 조인
full_joined_df = df.join(dept_df, "job", "full")
full_joined_df.show()
```

## Spark 고급 기능

### 1. 윈도우 함수
```python
from pyspark.sql.window import Window

# 윈도우 정의
window_spec = Window.partitionBy("job").orderBy("salary")

# 윈도우 함수 적용
df_with_window = df.withColumn(
    "rank", rank().over(window_spec)
).withColumn(
    "dense_rank", dense_rank().over(window_spec)
).withColumn(
    "row_number", row_number().over(window_spec)
).withColumn(
    "lag_salary", lag("salary", 1).over(window_spec)
).withColumn(
    "lead_salary", lead("salary", 1).over(window_spec)
)

df_with_window.show()

# 누적 합계
window_spec_cumulative = Window.partitionBy("job").orderBy("salary").rowsBetween(Window.unboundedPreceding, Window.currentRow)

df_with_cumulative = df.withColumn(
    "cumulative_salary", sum("salary").over(window_spec_cumulative)
)

df_with_cumulative.show()
```

### 2. UDF (User Defined Function)
```python
# UDF 정의
def calculate_bonus(salary):
    if salary >= 60000:
        return salary * 0.15
    elif salary >= 55000:
        return salary * 0.10
    else:
        return salary * 0.05

# UDF 등록
from pyspark.sql.functions import udf
bonus_udf = udf(calculate_bonus, IntegerType())

# UDF 적용
df_with_bonus_udf = df.withColumn("bonus", bonus_udf(df.salary))
df_with_bonus_udf.show()

# 람다 UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

name_length_udf = udf(lambda name: f"Name length: {len(name)}", StringType())
df_with_name_length = df.withColumn("name_info", name_length_udf(df.name))
df_with_name_length.show()
```

### 3. 캐싱 및 지속성
```python
# DataFrame 캐싱
df.cache()
df.persist()

# 메모리와 디스크에 저장
df.persist(StorageLevel.MEMORY_AND_DISK)

# 메모리만 사용
df.persist(StorageLevel.MEMORY_ONLY)

# 압축된 메모리 사용
df.persist(StorageLevel.MEMORY_ONLY_SER)

# 캐시 해제
df.unpersist()

# 캐시 상태 확인
print(f"캐시된 DataFrame 수: {spark.catalog.listTables()}")
```

## Spark 실무 적용 예시

### 1. 대용량 데이터 처리
```python
# 대용량 CSV 파일 읽기
large_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("large_dataset.csv")

# 파티션 수 확인
print(f"파티션 수: {large_df.rdd.getNumPartitions()}")

# 파티션 재구성
repartitioned_df = large_df.repartition(10)

# 파티션별 데이터 분포 확인
partition_counts = repartitioned_df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()
print(f"파티션별 데이터 수: {partition_counts}")

# 데이터 샘플링
sampled_df = large_df.sample(0.1, seed=42)

# 데이터 필터링 및 집계
filtered_df = large_df.filter(col("age") > 25)
aggregated_df = filtered_df.groupBy("department").agg(
    avg("salary").alias("avg_salary"),
    count("*").alias("employee_count")
)

# 결과 저장
aggregated_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("output/aggregated_data")
```

### 2. 데이터 품질 관리
```python
# 데이터 품질 검사
def data_quality_check(df):
    """데이터 품질 검사 함수"""
    
    # 전체 행 수
    total_rows = df.count()
    print(f"전체 행 수: {total_rows}")
    
    # 결측값 확인
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        print(f"{col_name} 결측값 수: {null_count}")
    
    # 중복 행 확인
    duplicate_count = df.count() - df.dropDuplicates().count()
    print(f"중복 행 수: {duplicate_count}")
    
    # 데이터 타입 확인
    df.printSchema()
    
    # 통계 정보
    df.describe().show()

# 데이터 품질 검사 실행
data_quality_check(df)

# 데이터 정리
def clean_data(df):
    """데이터 정리 함수"""
    
    # 결측값 처리
    cleaned_df = df.fillna({
        "age": 0,
        "salary": 0,
        "job": "Unknown"
    })
    
    # 중복 제거
    cleaned_df = cleaned_df.dropDuplicates()
    
    # 이상값 처리 (IQR 방법)
    for col_name in ["age", "salary"]:
        quantiles = cleaned_df.approxQuantile(col_name, [0.25, 0.75], 0.0)
        Q1, Q3 = quantiles[0], quantiles[1]
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        cleaned_df = cleaned_df.filter(
            (col(col_name) >= lower_bound) & (col(col_name) <= upper_bound)
        )
    
    return cleaned_df

# 데이터 정리 실행
cleaned_df = clean_data(df)
cleaned_df.show()
```

### 3. 성능 최적화
```python
# 성능 최적화 설정
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")

# 브로드캐스트 조인
small_df = spark.createDataFrame([("A", "Small"), ("B", "Medium")], ["id", "size"])
large_df = spark.createDataFrame([("A", 1), ("B", 2), ("C", 3)], ["id", "value"])

# 작은 DataFrame을 브로드캐스트
broadcasted_df = broadcast(small_df)
joined_df = large_df.join(broadcasted_df, "id")
joined_df.show()

# 파티션 최적화
optimized_df = df.repartition(4, "job")
optimized_df.write \
    .mode("overwrite") \
    .partitionBy("job") \
    .parquet("output/partitioned_data")

# 컬럼 기반 저장
df.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet("output/compressed_data")
```

## Spark 스트리밍

### 1. 구조화 스트리밍
```python
from pyspark.sql.streaming import StreamingQuery

# 스트리밍 데이터 읽기
streaming_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic_name") \
    .load()

# 스트리밍 데이터 처리
processed_stream = streaming_df \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .filter(col("age") > 25) \
    .groupBy("job") \
    .agg(count("*").alias("count"))

# 스트리밍 쿼리 시작
query = processed_stream \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# 쿼리 상태 확인
print(f"쿼리 ID: {query.id}")
print(f"쿼리 상태: {query.status}")

# 쿼리 중지
query.stop()
```

### 2. 배치 처리
```python
# 배치 데이터 읽기
batch_df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic_name") \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load()

# 배치 데이터 처리
processed_batch = batch_df \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .filter(col("age") > 25) \
    .groupBy("job") \
    .agg(count("*").alias("count"))

# 결과 저장
processed_batch.write \
    .mode("overwrite") \
    .parquet("output/batch_processed")
```

## Spark 머신러닝

### 1. MLlib 기본 사용법
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

# 샘플 데이터 생성
ml_data = [
    (25, "Engineer", 50000, 1),
    (30, "Manager", 60000, 0),
    (35, "Analyst", 55000, 1),
    (28, "Engineer", 52000, 0),
    (32, "Manager", 65000, 1)
]

ml_schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("salary", IntegerType(), True),
    StructField("label", IntegerType(), True)
])

ml_df = spark.createDataFrame(ml_data, ml_schema)

# 특성 변환
string_indexer = StringIndexer(inputCol="job", outputCol="job_index")
one_hot_encoder = OneHotEncoder(inputCol="job_index", outputCol="job_encoded")
vector_assembler = VectorAssembler(
    inputCols=["age", "job_encoded", "salary"],
    outputCol="features"
)

# 파이프라인 생성
pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler])

# 파이프라인 실행
transformed_df = pipeline.fit(ml_df).transform(ml_df)
transformed_df.show()

# 선형 회귀 모델
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(transformed_df)

# 예측
predictions = lr_model.transform(transformed_df)
predictions.show()

# 모델 평가
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")
```

### 2. 분류 모델
```python
# 분류 데이터 생성
classification_data = [
    (25, "Engineer", 50000, 1),
    (30, "Manager", 60000, 0),
    (35, "Analyst", 55000, 1),
    (28, "Engineer", 52000, 0),
    (32, "Manager", 65000, 1)
]

classification_schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("salary", IntegerType(), True),
    StructField("label", IntegerType(), True)
])

classification_df = spark.createDataFrame(classification_data, classification_schema)

# 특성 변환
string_indexer = StringIndexer(inputCol="job", outputCol="job_index")
one_hot_encoder = OneHotEncoder(inputCol="job_index", outputCol="job_encoded")
vector_assembler = VectorAssembler(
    inputCols=["age", "job_encoded", "salary"],
    outputCol="features"
)

# 랜덤 포레스트 분류기
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# 파이프라인 생성
pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler, rf])

# 모델 학습
model = pipeline.fit(classification_df)

# 예측
predictions = model.transform(classification_df)
predictions.show()

# 모델 평가
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"정확도: {accuracy}")
```

## 주의사항 및 모범 사례

### 1. 성능 최적화
- **파티션 수 조정**: 적절한 파티션 수 설정
- **캐싱 활용**: 자주 사용하는 DataFrame 캐싱
- **브로드캐스트 조인**: 작은 DataFrame 조인 시 브로드캐스트 사용
- **컬럼 기반 저장**: Parquet 형식 사용

### 2. 메모리 관리
- **적절한 스토리지 레벨**: 메모리 사용량에 맞는 스토리지 레벨 선택
- **가비지 컬렉션**: GC 튜닝으로 성능 향상
- **메모리 모니터링**: 메모리 사용량 모니터링

### 3. 에러 처리
- **예외 처리**: 적절한 예외 처리 구현
- **재시도 로직**: 일시적 오류에 대한 재시도
- **로깅**: 상세한 로깅으로 디버깅 지원

## 마무리

Apache Spark는 대용량 데이터 처리를 위한 강력한 통합 분석 엔진입니다. 인메모리 처리, 분산 처리, 통합 플랫폼 등의 특징을 통해 빠르고 효율적인 데이터 처리가 가능합니다. 적절한 설정과 최적화를 통해 실무에서 요구되는 높은 성능의 데이터 처리 시스템을 구축할 수 있습니다.
