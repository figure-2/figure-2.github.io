---
title: Apache Spark 기초 개념
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- Apache Spark
- RDD
- SparkContext
- 분산컴퓨팅
- 메모리기반연산
- 지연실행
- 장애복구
toc: true
date: 2023-11-07 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Apache Spark 기초 개념
> 대용량 데이터 처리를 위한 분산 컴퓨팅 프레임워크

## Spark란?
Apache Spark는 대용량 데이터를 빠르게 처리하기 위한 오픈소스 분산 컴퓨팅 프레임워크입니다. 메모리 기반 연산을 통해 기존 MapReduce보다 10-100배 빠른 성능을 제공합니다.

## 주요 특징
- **메모리 기반 연산**: 디스크 I/O 최소화로 빠른 처리
- **분산 처리**: 여러 노드에서 병렬 처리
- **다양한 언어 지원**: Python, Scala, Java, R
- **풍부한 라이브러리**: SQL, MLlib, GraphX, Streaming

## Spark 아키텍처
```
Driver Program (SparkContext)
    ↓
Cluster Manager (YARN, Mesos, Standalone)
    ↓
Worker Nodes (Executors)
```

## PySpark 기본 설정

### 1. SparkConf 설정
```python
from pyspark import SparkConf, SparkContext

# SparkConf: 스파크 실행환경 설정 클래스
conf = SparkConf().setMaster("local").setAppName("country-student-count")
```

**설정 옵션:**
- `setMaster("local")`: 로컬 모드 실행
- `setAppName("app-name")`: 애플리케이션 이름 설정

### 2. SparkContext 생성
```python
# SparkContext: DriverProgram 실행 환경 구성을 위한 클래스
sc = SparkContext(conf=conf)
```

**주의사항:**
- 변수명을 항상 `sc`로 설정
- 하나의 JVM당 하나의 SparkContext만 생성 가능

## RDD (Resilient Distributed Dataset)

### RDD란?
- **Resilient**: 장애 복구 능력
- **Distributed**: 분산 저장
- **Dataset**: 데이터 집합

RDD는 Spark의 기본 데이터 구조로, 불변(immutable)하고 분산된 데이터 컬렉션입니다.

### RDD 생성
```python
# 텍스트 파일에서 RDD 생성
filepath = "/path/to/data.csv"
lines = sc.textFile(f"file:///{filepath}")

# 첫 번째 줄 (헤더) 불러오기
header = lines.first()
print(header)
```

### RDD 기본 연산
```python
# RDD 정보 확인
print(f"파티션 수: {lines.getNumPartitions()}")
print(f"총 라인 수: {lines.count()}")

# 처음 5개 라인 출력
lines.take(5)
```

## Spark 실행 모드

### 1. Local 모드
```python
conf = SparkConf().setMaster("local")
```
- 단일 JVM에서 실행
- 개발 및 테스트용

### 2. Standalone 모드
```python
conf = SparkConf().setMaster("spark://master:7077")
```
- Spark 자체 클러스터 매니저 사용

### 3. YARN 모드
```python
conf = SparkConf().setMaster("yarn")
```
- Hadoop YARN 클러스터에서 실행

## 로그 레벨 설정
```python
# 로그 레벨 조정
sc.setLogLevel("ERROR")  # WARN, INFO, DEBUG
```

## Spark UI
- 기본 포트: 4040
- 포트 충돌 시 자동으로 4041, 4042... 사용
- 웹 브라우저에서 `http://localhost:4040` 접속

## 주요 학습 포인트

### 1. 분산 컴퓨팅 개념
- **Driver**: 메인 프로그램 실행
- **Executor**: 실제 작업 수행
- **Task**: 실행 단위

### 2. 메모리 기반 연산
- 디스크 I/O 최소화
- 중간 결과를 메모리에 캐싱
- 반복 연산에서 큰 성능 향상

### 3. 지연 실행 (Lazy Evaluation)
- Transformation은 즉시 실행되지 않음
- Action이 호출될 때 실제 연산 수행
- 최적화된 실행 계획 수립

### 4. 장애 복구
- Lineage 정보로 데이터 재생성
- 체크포인트로 복구 시간 단축

## 다음 단계
- RDD Transformation과 Action
- DataFrame API
- Spark SQL
- Spark MLlib
- Spark Streaming

Spark는 빅데이터 처리의 핵심 도구로, 대용량 데이터를 효율적으로 처리할 수 있는 강력한 기능들을 제공합니다.
