---
title: Apache Airflow 상세 내용
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- 데이터엔지니어링
- Airflow
- 워크플로우
- 오케스트레이션
- DAG
- ETL
- 데이터품질검사
- 머신러닝파이프라인
- 모니터링
- 성능최적화
- 보안
toc: true
date: 2023-11-09 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# Airflow 상세 내용

> 231120~231121, 231215~231222 학습한 내용 정리

## Apache Airflow 개요

### 정의
- **Apache Airflow**: 워크플로우 오케스트레이션 플랫폼
- **DAG**: Directed Acyclic Graph (방향성 비순환 그래프)
- **스케줄링**: 복잡한 데이터 파이프라인 스케줄링 및 모니터링
- **오케스트레이션**: 작업 간 의존성 관리 및 실행 순서 제어

### 특징
- **시각화**: 웹 UI를 통한 워크플로우 시각화
- **스케줄링**: Cron 기반 스케줄링
- **모니터링**: 실시간 작업 모니터링
- **확장성**: 다양한 연산자와 플러그인 지원

### 장점
- **유연성**: 다양한 작업 유형 지원
- **모니터링**: 상세한 실행 로그 및 모니터링
- **재사용성**: 재사용 가능한 컴포넌트
- **확장성**: 커스텀 연산자 및 플러그인 개발

## Airflow 설치 및 설정

### 1. Airflow 설치
```bash
# pip 설치
pip install apache-airflow

# 특정 버전 설치
pip install apache-airflow==2.5.0

# 추가 패키지 설치
pip install apache-airflow[postgres,celery,redis]
```

### 2. Airflow 초기화
```bash
# Airflow 홈 디렉토리 설정
export AIRFLOW_HOME=~/airflow

# 데이터베이스 초기화
airflow db init

# 관리자 사용자 생성
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 3. Airflow 시작
```bash
# 웹 서버 시작
airflow webserver --port 8080

# 스케줄러 시작
airflow scheduler
```

## Airflow 기본 사용법

### 1. DAG 생성
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

# 기본 인수 설정
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'example_dag',
    default_args=default_args,
    description='예제 DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['example', 'tutorial'],
)

# 작업 정의
start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

# Bash 작업
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello Airflow!"',
    dag=dag,
)

# Python 작업
def python_function():
    print("Python 작업 실행")
    return "작업 완료"

python_task = PythonOperator(
    task_id='python_task',
    python_callable=python_function,
    dag=dag,
)

# 작업 의존성 설정
start_task >> bash_task >> python_task
```

### 2. Python 작업
```python
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python import ShortCircuitOperator

# 기본 Python 작업
def data_processing():
    """데이터 처리 함수"""
    import pandas as pd
    import numpy as np
    
    # 샘플 데이터 생성
    data = {
        'id': range(1, 101),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    # 데이터 처리
    processed_df = df.groupby('category').agg({
        'value': ['mean', 'std', 'count']
    }).round(2)
    
    print("데이터 처리 완료:")
    print(processed_df)
    
    return processed_df.to_dict()

python_task = PythonOperator(
    task_id='data_processing',
    python_callable=data_processing,
    dag=dag,
)

# 분기 작업
def decide_branch():
    """분기 결정 함수"""
    import random
    
    if random.random() > 0.5:
        return 'branch_a'
    else:
        return 'branch_b'

branch_task = BranchPythonOperator(
    task_id='branch_task',
    python_callable=decide_branch,
    dag=dag,
)

# 분기 작업들
branch_a_task = DummyOperator(
    task_id='branch_a',
    dag=dag,
)

branch_b_task = DummyOperator(
    task_id='branch_b',
    dag=dag,
)

# 단축 회로 작업
def should_continue():
    """계속 진행할지 결정"""
    import random
    return random.random() > 0.3

short_circuit_task = ShortCircuitOperator(
    task_id='short_circuit_task',
    python_callable=should_continue,
    dag=dag,
)

# 작업 의존성
python_task >> branch_task
branch_task >> [branch_a_task, branch_b_task]
branch_a_task >> short_circuit_task
branch_b_task >> short_circuit_task
```

### 3. 데이터베이스 작업
```python
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# PostgreSQL 작업
postgres_task = PostgresOperator(
    task_id='postgres_task',
    postgres_conn_id='postgres_default',
    sql='''
    CREATE TABLE IF NOT EXISTS test_table (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''',
    dag=dag,
)

# PostgreSQL Hook 사용
def postgres_hook_example():
    """PostgreSQL Hook 사용 예시"""
    hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # 데이터 삽입
    insert_sql = """
    INSERT INTO test_table (name) VALUES (%s)
    """
    
    hook.run(insert_sql, parameters=['test_name'])
    
    # 데이터 조회
    select_sql = "SELECT * FROM test_table LIMIT 10"
    records = hook.get_records(select_sql)
    
    print("조회된 데이터:")
    for record in records:
        print(record)

postgres_hook_task = PythonOperator(
    task_id='postgres_hook_task',
    python_callable=postgres_hook_example,
    dag=dag,
)
```

## Airflow 고급 기능

### 1. XCom (Cross-Communication)
```python
from airflow.operators.python import PythonOperator

# XCom을 사용한 데이터 전달
def push_data():
    """데이터를 XCom에 푸시"""
    return {
        'processed_data': [1, 2, 3, 4, 5],
        'metadata': {
            'processed_at': datetime.now().isoformat(),
            'record_count': 5
        }
    }

def pull_data(**context):
    """XCom에서 데이터를 풀"""
    # 이전 작업의 결과 가져오기
    data = context['task_instance'].xcom_pull(task_ids='push_data_task')
    
    print("받은 데이터:")
    print(f"처리된 데이터: {data['processed_data']}")
    print(f"메타데이터: {data['metadata']}")
    
    # 데이터 처리
    processed_data = [x * 2 for x in data['processed_data']]
    
    return processed_data

def final_processing(**context):
    """최종 처리"""
    data = context['task_instance'].xcom_pull(task_ids='pull_data_task')
    
    print(f"최종 처리된 데이터: {data}")
    
    return sum(data)

# XCom 작업들
push_data_task = PythonOperator(
    task_id='push_data_task',
    python_callable=push_data,
    dag=dag,
)

pull_data_task = PythonOperator(
    task_id='pull_data_task',
    python_callable=pull_data,
    dag=dag,
)

final_processing_task = PythonOperator(
    task_id='final_processing_task',
    python_callable=final_processing,
    dag=dag,
)

# 작업 의존성
push_data_task >> pull_data_task >> final_processing_task
```

### 2. 동적 DAG 생성
```python
from airflow import DAG
from airflow.operators.bash import BashOperator

# 동적 DAG 생성 함수
def create_dynamic_dag(dag_id, schedule_interval, tasks):
    """동적 DAG 생성"""
    
    default_args = {
        'owner': 'data_team',
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
    
    dag = DAG(
        dag_id,
        default_args=default_args,
        description=f'동적 생성된 DAG: {dag_id}',
        schedule_interval=schedule_interval,
        catchup=False,
    )
    
    # 동적으로 작업 생성
    previous_task = None
    
    for i, task_config in enumerate(tasks):
        task = BashOperator(
            task_id=f'task_{i}',
            bash_command=task_config['command'],
            dag=dag,
        )
        
        if previous_task:
            previous_task >> task
        
        previous_task = task
    
    return dag

# 동적 DAG 생성 예시
tasks_config = [
    {'command': 'echo "작업 1 실행"'},
    {'command': 'echo "작업 2 실행"'},
    {'command': 'echo "작업 3 실행"'},
]

dynamic_dag = create_dynamic_dag(
    'dynamic_example_dag',
    timedelta(hours=1),
    tasks_config
)
```

### 3. 커스텀 연산자
```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class CustomOperator(BaseOperator):
    """커스텀 연산자"""
    
    @apply_defaults
    def __init__(self, custom_param, *args, **kwargs):
        super(CustomOperator, self).__init__(*args, **kwargs)
        self.custom_param = custom_param
    
    def execute(self, context):
        """연산자 실행"""
        print(f"커스텀 연산자 실행: {self.custom_param}")
        
        # 커스텀 로직 구현
        result = self.process_data()
        
        return result
    
    def process_data(self):
        """데이터 처리 로직"""
        # 실제 데이터 처리 로직
        processed_data = {
            'input': self.custom_param,
            'output': f"processed_{self.custom_param}",
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_data

# 커스텀 연산자 사용
custom_task = CustomOperator(
    task_id='custom_task',
    custom_param='test_value',
    dag=dag,
)
```

## Airflow 실무 적용 예시

### 1. ETL 파이프라인
```python
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

# ETL 파이프라인 DAG
etl_dag = DAG(
    'etl_pipeline',
    default_args=default_args,
    description='ETL 파이프라인',
    schedule_interval=timedelta(hours=1),
    catchup=False,
)

# Extract 작업
def extract_data():
    """데이터 추출"""
    import requests
    import json
    
    # API에서 데이터 추출
    response = requests.get('https://api.example.com/data')
    data = response.json()
    
    # 데이터를 파일로 저장
    with open('/tmp/extracted_data.json', 'w') as f:
        json.dump(data, f)
    
    print(f"추출된 데이터 수: {len(data)}")
    return len(data)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=etl_dag,
)

# Transform 작업
def transform_data():
    """데이터 변환"""
    import json
    import pandas as pd
    
    # 데이터 로드
    with open('/tmp/extracted_data.json', 'r') as f:
        data = json.load(f)
    
    # DataFrame으로 변환
    df = pd.DataFrame(data)
    
    # 데이터 변환
    df['processed_at'] = datetime.now()
    df['value_squared'] = df['value'] ** 2
    
    # 변환된 데이터 저장
    df.to_csv('/tmp/transformed_data.csv', index=False)
    
    print(f"변환된 데이터 수: {len(df)}")
    return len(df)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=etl_dag,
)

# Load 작업
def load_data():
    """데이터 로드"""
    import pandas as pd
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    
    # 데이터 로드
    df = pd.read_csv('/tmp/transformed_data.csv')
    
    # PostgreSQL에 데이터 삽입
    hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # 테이블 생성
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS processed_data (
        id SERIAL PRIMARY KEY,
        value FLOAT,
        processed_at TIMESTAMP,
        value_squared FLOAT
    );
    """
    
    hook.run(create_table_sql)
    
    # 데이터 삽입
    for _, row in df.iterrows():
        insert_sql = """
        INSERT INTO processed_data (value, processed_at, value_squared)
        VALUES (%s, %s, %s)
        """
        
        hook.run(insert_sql, parameters=[
            row['value'],
            row['processed_at'],
            row['value_squared']
        ])
    
    print(f"로드된 데이터 수: {len(df)}")
    return len(df)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=etl_dag,
)

# ETL 파이프라인 의존성
extract_task >> transform_task >> load_task
```

### 2. 데이터 품질 검사
```python
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator

# 데이터 품질 검사 DAG
quality_dag = DAG(
    'data_quality_check',
    default_args=default_args,
    description='데이터 품질 검사',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# 데이터 품질 검사
def data_quality_check():
    """데이터 품질 검사"""
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import pandas as pd
    
    hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # 데이터 조회
    query = "SELECT * FROM processed_data"
    df = pd.read_sql(query, hook.get_conn())
    
    # 품질 검사
    quality_report = {
        'total_records': len(df),
        'null_values': df.isnull().sum().to_dict(),
        'duplicate_records': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'statistics': df.describe().to_dict()
    }
    
    # 품질 검사 결과 저장
    with open('/tmp/quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print("데이터 품질 검사 완료:")
    print(f"총 레코드 수: {quality_report['total_records']}")
    print(f"중복 레코드 수: {quality_report['duplicate_records']}")
    
    return quality_report

quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=quality_dag,
)

# 품질 검사 결과 알림
def send_quality_report(**context):
    """품질 검사 결과 알림"""
    import json
    
    # 품질 검사 결과 가져오기
    quality_report = context['task_instance'].xcom_pull(task_ids='data_quality_check')
    
    # 알림 메시지 생성
    message = f"""
    데이터 품질 검사 결과:
    
    총 레코드 수: {quality_report['total_records']}
    중복 레코드 수: {quality_report['duplicate_records']}
    
    상세 결과는 첨부 파일을 참조하세요.
    """
    
    # 이메일 전송
    email_task = EmailOperator(
        task_id='send_quality_report',
        to=['data_team@example.com'],
        subject='데이터 품질 검사 결과',
        html_content=message,
        files=['/tmp/quality_report.json'],
        dag=quality_dag,
    )
    
    return email_task

# 품질 검사 파이프라인 의존성
quality_check_task >> send_quality_report()
```

### 3. 머신러닝 파이프라인
```python
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# 머신러닝 파이프라인 DAG
ml_dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='머신러닝 파이프라인',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# 데이터 전처리
def data_preprocessing():
    """데이터 전처리"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # 데이터 로드
    df = pd.read_csv('/tmp/raw_data.csv')
    
    # 데이터 전처리
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 전처리된 데이터 저장
    pd.DataFrame(X_train).to_csv('/tmp/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('/tmp/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('/tmp/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('/tmp/y_test.csv', index=False)
    
    print("데이터 전처리 완료")
    return "preprocessing_complete"

preprocessing_task = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    dag=ml_dag,
)

# 모델 훈련
def model_training():
    """모델 훈련"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    # 데이터 로드
    X_train = pd.read_csv('/tmp/X_train.csv')
    y_train = pd.read_csv('/tmp/y_train.csv')
    X_test = pd.read_csv('/tmp/X_test.csv')
    y_test = pd.read_csv('/tmp/y_test.csv')
    
    # 모델 훈련
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 모델 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 모델 저장
    joblib.dump(model, '/tmp/trained_model.pkl')
    
    print(f"모델 훈련 완료, 정확도: {accuracy:.4f}")
    return accuracy

training_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=ml_dag,
)

# 모델 배포
def model_deployment():
    """모델 배포"""
    import shutil
    import os
    
    # 모델 파일 복사
    source = '/tmp/trained_model.pkl'
    destination = '/opt/models/production_model.pkl'
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # 모델 파일 복사
    shutil.copy2(source, destination)
    
    print("모델 배포 완료")
    return "deployment_complete"

deployment_task = PythonOperator(
    task_id='model_deployment',
    python_callable=model_deployment,
    dag=ml_dag,
)

# 머신러닝 파이프라인 의존성
preprocessing_task >> training_task >> deployment_task
```

## Airflow 모니터링 및 관리

### 1. 알림 설정
```python
from airflow.operators.email import EmailOperator
from airflow.operators.slack import SlackAPIPostOperator

# 실패 시 알림
def failure_callback(context):
    """실패 시 콜백"""
    task_instance = context['task_instance']
    
    # 이메일 알림
    email_task = EmailOperator(
        task_id='failure_email',
        to=['admin@example.com'],
        subject=f'Airflow 작업 실패: {task_instance.task_id}',
        html_content=f"""
        작업 실패 정보:
        - DAG: {context['dag'].dag_id}
        - 작업: {task_instance.task_id}
        - 실행 날짜: {context['ds']}
        - 오류: {context['exception']}
        """,
    )
    
    # Slack 알림
    slack_task = SlackAPIPostOperator(
        task_id='failure_slack',
        channel='#alerts',
        text=f'🚨 Airflow 작업 실패: {task_instance.task_id}',
        username='airflow-bot',
    )
    
    return [email_task, slack_task]

# DAG에 실패 콜백 설정
dag = DAG(
    'monitored_dag',
    default_args=default_args,
    description='모니터링되는 DAG',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    on_failure_callback=failure_callback,
)
```

### 2. 성능 모니터링
```python
from airflow.operators.python import PythonOperator
from airflow.models import TaskInstance

# 성능 모니터링
def performance_monitoring(**context):
    """성능 모니터링"""
    task_instance = context['task_instance']
    
    # 실행 시간 측정
    start_time = task_instance.start_date
    end_time = task_instance.end_date
    
    if start_time and end_time:
        execution_time = (end_time - start_time).total_seconds()
        
        # 성능 메트릭 저장
        metrics = {
            'task_id': task_instance.task_id,
            'execution_time': execution_time,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'status': task_instance.state
        }
        
        # 메트릭을 파일로 저장
        with open(f'/tmp/metrics_{task_instance.task_id}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"성능 메트릭: {metrics}")
    
    return metrics

performance_task = PythonOperator(
    task_id='performance_monitoring',
    python_callable=performance_monitoring,
    dag=dag,
)
```

### 3. 리소스 모니터링
```python
import psutil
import os

# 리소스 모니터링
def resource_monitoring():
    """리소스 모니터링"""
    
    # 시스템 리소스 정보
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Airflow 프로세스 정보
    airflow_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if 'airflow' in proc.info['name'].lower():
            airflow_processes.append(proc.info)
    
    # 리소스 정보 저장
    resource_info = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': cpu_percent,
        'memory': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent
        },
        'disk': {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100
        },
        'airflow_processes': airflow_processes
    }
    
    # 리소스 정보를 파일로 저장
    with open('/tmp/resource_info.json', 'w') as f:
        json.dump(resource_info, f, indent=2)
    
    print("리소스 모니터링 완료:")
    print(f"CPU 사용률: {cpu_percent}%")
    print(f"메모리 사용률: {memory.percent}%")
    print(f"디스크 사용률: {(disk.used / disk.total) * 100:.2f}%")
    
    return resource_info

resource_monitoring_task = PythonOperator(
    task_id='resource_monitoring',
    python_callable=resource_monitoring,
    dag=dag,
)
```

## 주의사항 및 모범 사례

### 1. DAG 설계
- **단순성**: DAG를 단순하고 이해하기 쉽게 설계
- **재사용성**: 재사용 가능한 컴포넌트 사용
- **의존성**: 명확한 작업 의존성 정의
- **에러 처리**: 적절한 에러 처리 및 재시도

### 2. 성능 최적화
- **리소스 관리**: 적절한 리소스 할당
- **병렬 처리**: 가능한 경우 병렬 처리 활용
- **캐싱**: 중복 작업 방지를 위한 캐싱
- **모니터링**: 지속적인 성능 모니터링

### 3. 보안
- **접근 제어**: 적절한 접근 권한 관리
- **비밀 관리**: 민감한 정보 보호
- **감사**: 작업 실행 로그 및 감사
- **백업**: 정기적인 백업 수행

## 마무리

Apache Airflow는 복잡한 데이터 파이프라인을 오케스트레이션하기 위한 강력한 플랫폼입니다. DAG를 통한 워크플로우 정의, 스케줄링, 모니터링 등의 기능을 통해 안정적이고 효율적인 데이터 파이프라인을 구축할 수 있습니다. 적절한 설계와 모니터링을 통해 실무에서 요구되는 높은 품질의 데이터 파이프라인을 운영할 수 있습니다.
