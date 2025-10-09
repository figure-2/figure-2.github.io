---
title: Apache Kafka 기초 개념
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- Apache Kafka
- Topic
- Partition
- Producer
- Consumer
- Broker
- 스트리밍
- 실시간처리
toc: true
date: 2023-11-08 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Apache Kafka 기초 개념
> 실시간 스트리밍 데이터 플랫폼

## Kafka란?
Apache Kafka는 실시간으로 대용량 스트림 데이터를 처리하기 위한 분산 이벤트 스트리밍 플랫폼입니다. 높은 처리량과 낮은 지연시간을 제공하며, 마이크로서비스 간의 데이터 파이프라인 구축에 널리 사용됩니다.

## 주요 특징
- **고성능**: 초당 수백만 개의 메시지 처리
- **확장성**: 수평적 확장 가능
- **내구성**: 메시지 영속성 보장
- **실시간성**: 낮은 지연시간
- **분산**: 여러 브로커로 구성된 클러스터

## Kafka 아키텍처
```
Producer → Topic → Partition → Consumer
    ↓         ↓        ↓         ↓
  Broker1   Broker2  Broker3  Consumer Group
```

## 핵심 개념

### 1. Topic
- 메시지가 저장되는 카테고리
- 여러 파티션으로 분할 가능
- 예: `sample-topic`, `user-events`, `payment-logs`

### 2. Partition
- Topic을 여러 개로 분할
- 병렬 처리 및 확장성 제공
- 순서 보장 (파티션 내에서만)

### 3. Producer
- 메시지를 Topic에 전송
- 파티션 선택 전략 제공
- 배치 처리로 성능 최적화

### 4. Consumer
- Topic에서 메시지 수신
- Consumer Group으로 병렬 처리
- 오프셋 관리로 중복/누락 방지

### 5. Broker
- Kafka 서버 인스턴스
- Topic과 Partition 관리
- 클러스터로 구성

## Python Kafka 구현

### Producer 구현
```python
from kafka import KafkaProducer

# 브로커 서버 설정
BROKER_SERVERS = ["172.31.36.132:9092"]
TOPIC_NAME = "sample-topic"

# Producer 객체 생성
producer = KafkaProducer(bootstrap_servers=BROKER_SERVERS)

# 메시지 전송
producer.send(TOPIC_NAME, b'hello Kafka')

# 버퍼 플러싱 (즉시 전송)
producer.flush()
```

**주요 설정:**
- `bootstrap_servers`: 브로커 서버 목록
- `value_serializer`: 메시지 직렬화 방식
- `acks`: 메시지 전송 확인 설정

### Consumer 구현
```python
from kafka import KafkaConsumer

BROKER_SERVERS = ["172.31.36.132:9092"]
TOPIC_NAME = "sample-topic"

# Consumer 객체 생성
consumer = KafkaConsumer(
    TOPIC_NAME, 
    bootstrap_servers=BROKER_SERVERS
)

print("Wait.....")

# 무한 루프로 메시지 수신
for message in consumer:
    print(message)
```

**Consumer 특징:**
- Generator 패턴으로 구현
- 무한 대기 상태
- 자동 오프셋 관리

## JSON 메시지 처리

### JSON Producer
```python
import json
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=BROKER_SERVERS,
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# JSON 메시지 전송
message = {
    "user_id": 123,
    "action": "login",
    "timestamp": "2023-11-15T10:30:00Z"
}

producer.send(TOPIC_NAME, message)
producer.flush()
```

### JSON Consumer
```python
import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=BROKER_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    print(f"Received: {message.value}")
```

## 실무 활용 사례

### 1. 실시간 데이터 파이프라인
```
Web App → Kafka → Spark Streaming → Database
```

### 2. 마이크로서비스 통신
```
Service A → Kafka → Service B
Service A → Kafka → Service C
```

### 3. 이벤트 소싱
```
User Action → Event → Kafka → Event Store
```

### 4. 로그 수집
```
Application Logs → Kafka → Log Analysis System
```

## 고급 기능

### 1. 파티션 전략
```python
# 키 기반 파티션
producer.send(TOPIC_NAME, value=message, key=user_id)

# 라운드 로빈 파티션
producer.send(TOPIC_NAME, value=message)
```

### 2. Consumer Group
```python
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=BROKER_SERVERS,
    group_id='my-consumer-group'
)
```

### 3. 오프셋 관리
```python
# 수동 오프셋 커밋
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=BROKER_SERVERS,
    enable_auto_commit=False
)

for message in consumer:
    process_message(message)
    consumer.commit()
```

## 성능 최적화

### 1. 배치 처리
```python
producer = KafkaProducer(
    bootstrap_servers=BROKER_SERVERS,
    batch_size=16384,  # 배치 크기
    linger_ms=10       # 대기 시간
)
```

### 2. 압축
```python
producer = KafkaProducer(
    bootstrap_servers=BROKER_SERVERS,
    compression_type='gzip'
)
```

### 3. 파티션 수 조정
- Topic의 파티션 수 = Consumer 수
- 병렬 처리 성능 향상

## 모니터링

### 1. Kafka Manager
- 웹 기반 관리 도구
- Topic, Consumer Group 모니터링

### 2. JMX 메트릭
- Producer/Consumer 성능 지표
- 브로커 상태 모니터링

### 3. 로그 분석
- 브로커 로그 확인
- Consumer 지연 모니터링

## 주요 학습 포인트

### 1. 분산 시스템 이해
- 파티션과 복제
- 리더-팔로워 구조
- 장애 복구 메커니즘

### 2. 메시지 순서 보장
- 파티션 내 순서 보장
- 키 기반 라우팅
- 멱등성 보장

### 3. 확장성 설계
- 수평적 확장
- 부하 분산
- 리소스 최적화

### 4. 실시간 처리
- 스트리밍 데이터 처리
- 이벤트 기반 아키텍처
- 마이크로서비스 통신

Kafka는 현대적인 데이터 아키텍처의 핵심 구성 요소로, 실시간 데이터 처리와 마이크로서비스 통신에 필수적인 도구입니다.
