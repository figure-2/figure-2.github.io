---
title: Apache Airflow 기초 개념
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- 데이터엔지니어링
- Airflow
- 워크플로우
- 오케스트레이션
- DAG
- Task
- Operator
- Sensor
- XCom
- 스케줄링
toc: true
date: 2023-11-09 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Apache Airflow 기초 개념
> 워크플로우 오케스트레이션을 위한 플랫폼

## Airflow란?
Apache Airflow는 워크플로우를 프로그래밍 방식으로 작성, 스케줄링, 모니터링할 수 있는 플랫폼입니다. 데이터 파이프라인의 복잡한 의존성과 스케줄링을 관리하는 데 특화되어 있습니다.

## 주요 특징
- **프로그래밍 방식**: Python 코드로 워크플로우 정의
- **스케줄링**: Cron 표현식 기반 스케줄링
- **모니터링**: 웹 UI를 통한 실시간 모니터링
- **확장성**: 다양한 연산자와 센서 제공
- **재시도**: 실패한 작업의 자동 재시도

## 핵심 개념

### 1. DAG (Directed Acyclic Graph)
- 워크플로우를 정의하는 Python 스크립트
- 방향성이 있는 비순환 그래프
- 각 노드는 Task, 각 엣지는 의존성

### 2. Task
- DAG 내의 개별 작업 단위
- Operator를 통해 실행되는 작업

### 3. Operator
- 실제 작업을 수행하는 컴포넌트
- BashOperator, PythonOperator, HttpOperator 등

### 4. Sensor
- 특정 조건이 만족될 때까지 대기
- 파일 존재, API 응답, 데이터베이스 상태 등

## 기본 DAG 구조

### 1. 기본 설정
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# 기본 인수 설정
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 11, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# DAG 정의
dag = DAG(
    'my_first_dag',
    default_args=default_args,
    description='My first Airflow DAG',
    schedule_interval=timedelta(days=1),
    catchup=False
)
```

### 2. Task 정의
```python
# Bash 작업
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello Airflow!"',
    dag=dag
)

# Python 작업
def python_function():
    print("This is a Python task")
    return "Task completed"

python_task = PythonOperator(
    task_id='python_task',
    python_callable=python_function,
    dag=dag
)
```

### 3. 의존성 설정
```python
# Task 간 의존성 설정
bash_task >> python_task

# 또는
python_task.set_upstream(bash_task)
```

## 주요 Operator

### 1. BashOperator
```python
from airflow.operators.bash import BashOperator

bash_task = BashOperator(
    task_id='run_bash_command',
    bash_command='ls -la /tmp',
    dag=dag
)
```

### 2. PythonOperator
```python
from airflow.operators.python import PythonOperator

def process_data():
    import pandas as pd
    df = pd.read_csv('/tmp/data.csv')
    processed_df = df.dropna()
    processed_df.to_csv('/tmp/processed_data.csv', index=False)

python_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)
```

### 3. HttpOperator
```python
from airflow.providers.http.operators.http import SimpleHttpOperator

http_task = SimpleHttpOperator(
    task_id='call_api',
    http_conn_id='my_api_connection',
    endpoint='api/data',
    method='GET',
    dag=dag
)
```

### 4. SqliteOperator
```python
from airflow.providers.sqlite.operators.sqlite import SqliteOperator

sqlite_task = SqliteOperator(
    task_id='create_table',
    sqlite_conn_id='sqlite_default',
    sql='CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)',
    dag=dag
)
```

## XCom을 통한 데이터 전달

### 1. 데이터 푸시
```python
def extract_data(ti):
    data = {'key': 'value', 'number': 42}
    ti.xcom_push(key='extracted_data', value=data)
    return data

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)
```

### 2. 데이터 풀
```python
def process_data(ti):
    data = ti.xcom_pull(key='extracted_data', task_ids='extract_data')
    print(f"Received data: {data}")
    return data

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)
```

## 실무 예제: 카카오 API 파이프라인

### 1. API 데이터 추출
```python
from airflow.providers.http.operators.http import SimpleHttpOperator

def extract_kakao(ti):
    import requests
    
    api_key = "your_kakao_api_key"
    url = f"https://dapi.kakao.com/v2/search/web?query=파이썬"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    return data

extract_task = PythonOperator(
    task_id='extract_kakao',
    python_callable=extract_kakao,
    dag=dag
)
```

### 2. 데이터 전처리
```python
def preprocessing(ti):
    import pandas as pd
    
    # 이전 태스크에서 데이터 가져오기
    search_result = ti.xcom_pull(task_ids=['extract_kakao'])
    documents = search_result[0]['documents']
    
    # DataFrame으로 변환
    df = pd.DataFrame(documents)
    
    # CSV로 저장
    df.to_csv('/tmp/processed_result.csv', index=False)
    
    return "Preprocessing completed"

preprocess_task = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    dag=dag
)
```

### 3. 데이터베이스 저장
```python
def load_to_db(ti):
    import pandas as pd
    import sqlite3
    
    # CSV 파일 읽기
    df = pd.read_csv('/tmp/processed_result.csv')
    
    # 데이터베이스 연결
    conn = sqlite3.connect('/tmp/results.db')
    
    # 테이블에 저장
    df.to_sql('search_results', conn, if_exists='replace', index=False)
    
    conn.close()
    return "Data loaded to database"

load_task = PythonOperator(
    task_id='load_to_db',
    python_callable=load_to_db,
    dag=dag
)
```

### 4. 전체 파이프라인
```python
# 의존성 설정
extract_task >> preprocess_task >> load_task
```

## 스케줄링

### 1. Cron 표현식
```python
dag = DAG(
    'scheduled_dag',
    default_args=default_args,
    schedule_interval='0 9 * * *',  # 매일 오전 9시
    catchup=False
)
```

### 2. 상대적 간격
```python
dag = DAG(
    'interval_dag',
    default_args=default_args,
    schedule_interval=timedelta(hours=2),  # 2시간마다
    catchup=False
)
```

### 3. 조건부 실행
```python
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor

# 파일 존재 확인
file_sensor = FileSensor(
    task_id='wait_for_file',
    filepath='/tmp/input_data.csv',
    dag=dag
)

# 조건부 실행
dummy_task = DummyOperator(
    task_id='dummy_task',
    dag=dag
)

file_sensor >> dummy_task
```

## 모니터링과 로깅

### 1. 웹 UI
- **DAGs**: 모든 DAG 목록 및 상태
- **Graph**: DAG의 그래프 뷰
- **Task Instance**: 개별 태스크 상세 정보
- **Logs**: 태스크 실행 로그

### 2. 알림 설정
```python
default_args = {
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['admin@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}
```

### 3. 로깅
```python
def logging_task():
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
```

## 주요 학습 포인트

### 1. DAG 설계 원칙
- **단일 책임**: 각 DAG는 하나의 목적
- **재사용성**: 공통 로직은 별도 모듈로 분리
- **테스트 가능성**: 단위 테스트 작성

### 2. 에러 처리
- **재시도 전략**: 적절한 재시도 횟수 설정
- **실패 처리**: 실패 시 알림 및 복구 로직
- **의존성 관리**: 실패한 태스크의 영향 최소화

### 3. 성능 최적화
- **병렬 처리**: 독립적인 태스크의 병렬 실행
- **리소스 관리**: 적절한 리소스 할당
- **캐싱**: 중복 계산 방지

### 4. 보안
- **연결 관리**: 민감한 정보의 안전한 저장
- **권한 관리**: 적절한 접근 권한 설정
- **암호화**: 데이터 전송 및 저장 시 암호화

Airflow는 복잡한 데이터 파이프라인을 체계적으로 관리할 수 있는 강력한 도구입니다.
