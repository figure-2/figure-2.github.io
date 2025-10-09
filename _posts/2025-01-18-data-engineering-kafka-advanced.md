---
title: Apache Kafka 상세 내용
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- Apache Kafka
- 파티셔닝
- 컨슈머그룹
- 오프셋관리
- 이벤트스트리밍
- 로그수집
- 데이터파이프라인
- 모니터링
toc: true
date: 2023-11-08 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# Kafka 상세 내용

> 231122 학습한 내용 정리

## Apache Kafka 개요

### 정의
- **Apache Kafka**: 분산 스트리밍 플랫폼
- **메시지 큐**: 대용량 실시간 데이터 스트리밍
- **이벤트 스트리밍**: 이벤트 기반 아키텍처 지원
- **로그 기반**: 분산 로그 시스템

### 특징
- **고성능**: 초당 수백만 메시지 처리
- **확장성**: 수평적 확장 가능
- **내구성**: 데이터 지속성 보장
- **실시간**: 낮은 지연시간

### 장점
- **처리량**: 높은 처리량과 낮은 지연시간
- **확장성**: 클러스터로 확장 가능
- **내구성**: 데이터 손실 방지
- **통합**: 다양한 시스템과 통합

## Kafka 설치 및 설정

### 1. Kafka 설치
```bash
# Kafka 다운로드
wget https://archive.apache.org/dist/kafka/2.8.1/kafka_2.13-2.8.1.tgz

# 압축 해제
tar -xzf kafka_2.13-2.8.1.tgz

# 환경변수 설정
export KAFKA_HOME=/path/to/kafka_2.13-2.8.1
export PATH=$PATH:$KAFKA_HOME/bin
```

### 2. Kafka 시작
```bash
# Zookeeper 시작
bin/zookeeper-server-start.sh config/zookeeper.properties

# Kafka 서버 시작
bin/kafka-server-start.sh config/server.properties
```

### 3. Python Kafka 클라이언트
```python
# kafka-python 설치
pip install kafka-python

# 기본 사용법
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
import time

# 프로듀서 설정
producer_config = {
    'bootstrap_servers': ['localhost:9092'],
    'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
    'key_serializer': lambda x: x.encode('utf-8') if x else None,
    'acks': 'all',
    'retries': 3,
    'batch_size': 16384,
    'linger_ms': 10,
    'buffer_memory': 33554432
}

# 컨슈머 설정
consumer_config = {
    'bootstrap_servers': ['localhost:9092'],
    'group_id': 'my_consumer_group',
    'auto_offset_reset': 'earliest',
    'enable_auto_commit': True,
    'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
    'key_deserializer': lambda x: x.decode('utf-8') if x else None
}
```

## Kafka 기본 사용법

### 1. 프로듀서 (Producer)
```python
def create_producer():
    """Kafka 프로듀서 생성"""
    try:
        producer = KafkaProducer(**producer_config)
        return producer
    except Exception as e:
        print(f"프로듀서 생성 오류: {e}")
        return None

def send_message(producer, topic, message, key=None):
    """메시지 전송"""
    try:
        future = producer.send(topic, value=message, key=key)
        record_metadata = future.get(timeout=10)
        
        print(f"메시지 전송 성공:")
        print(f"  토픽: {record_metadata.topic}")
        print(f"  파티션: {record_metadata.partition}")
        print(f"  오프셋: {record_metadata.offset}")
        
        return True
    except KafkaError as e:
        print(f"메시지 전송 오류: {e}")
        return False

# 프로듀서 사용 예시
def producer_example():
    """프로듀서 사용 예시"""
    producer = create_producer()
    
    if producer:
        # 단일 메시지 전송
        message = {"user_id": 1, "action": "login", "timestamp": time.time()}
        send_message(producer, "user_events", message, key="user_1")
        
        # 여러 메시지 전송
        for i in range(10):
            message = {
                "user_id": i,
                "action": "page_view",
                "page": f"/page_{i}",
                "timestamp": time.time()
            }
            send_message(producer, "user_events", message, key=f"user_{i}")
        
        # 프로듀서 종료
        producer.close()

# 프로듀서 예시 실행
# producer_example()
```

### 2. 컨슈머 (Consumer)
```python
def create_consumer():
    """Kafka 컨슈머 생성"""
    try:
        consumer = KafkaConsumer(**consumer_config)
        return consumer
    except Exception as e:
        print(f"컨슈머 생성 오류: {e}")
        return None

def consume_messages(consumer, topics, timeout_ms=1000):
    """메시지 소비"""
    try:
        consumer.subscribe(topics)
        
        while True:
            message_batch = consumer.poll(timeout_ms=timeout_ms)
            
            if not message_batch:
                continue
            
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    print(f"메시지 수신:")
                    print(f"  토픽: {message.topic}")
                    print(f"  파티션: {message.partition}")
                    print(f"  오프셋: {message.offset}")
                    print(f"  키: {message.key}")
                    print(f"  값: {message.value}")
                    print(f"  타임스탬프: {message.timestamp}")
                    print("-" * 50)
            
            # 오프셋 커밋
            consumer.commit()
    
    except KeyboardInterrupt:
        print("컨슈머 중지")
    except Exception as e:
        print(f"메시지 소비 오류: {e}")
    finally:
        consumer.close()

# 컨슈머 사용 예시
def consumer_example():
    """컨슈머 사용 예시"""
    consumer = create_consumer()
    
    if consumer:
        topics = ["user_events"]
        consume_messages(consumer, topics)

# 컨슈머 예시 실행
# consumer_example()
```

### 3. 토픽 관리
```python
from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType
from kafka.admin.config_resource import ConfigResource
from kafka.errors import TopicAlreadyExistsError

def create_admin_client():
    """Kafka 관리자 클라이언트 생성"""
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=['localhost:9092'],
            client_id='admin_client'
        )
        return admin_client
    except Exception as e:
        print(f"관리자 클라이언트 생성 오류: {e}")
        return None

def create_topic(admin_client, topic_name, num_partitions=1, replication_factor=1):
    """토픽 생성"""
    from kafka.admin import NewTopic
    
    try:
        topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        
        admin_client.create_topics([topic])
        print(f"토픽 '{topic_name}' 생성 완료")
        return True
    
    except TopicAlreadyExistsError:
        print(f"토픽 '{topic_name}'이 이미 존재합니다")
        return False
    except Exception as e:
        print(f"토픽 생성 오류: {e}")
        return False

def list_topics(admin_client):
    """토픽 목록 조회"""
    try:
        metadata = admin_client.describe_topics()
        topics = list(metadata.keys())
        print(f"토픽 목록: {topics}")
        return topics
    except Exception as e:
        print(f"토픽 목록 조회 오류: {e}")
        return []

def delete_topic(admin_client, topic_name):
    """토픽 삭제"""
    try:
        admin_client.delete_topics([topic_name])
        print(f"토픽 '{topic_name}' 삭제 완료")
        return True
    except Exception as e:
        print(f"토픽 삭제 오류: {e}")
        return False

# 토픽 관리 예시
def topic_management_example():
    """토픽 관리 예시"""
    admin_client = create_admin_client()
    
    if admin_client:
        # 토픽 생성
        create_topic(admin_client, "test_topic", num_partitions=3, replication_factor=1)
        
        # 토픽 목록 조회
        list_topics(admin_client)
        
        # 토픽 삭제
        delete_topic(admin_client, "test_topic")

# 토픽 관리 예시 실행
# topic_management_example()
```

## Kafka 고급 기능

### 1. 파티셔닝
```python
def custom_partitioner(key, all_partitions, available_partitions):
    """커스텀 파티셔너"""
    if key is None:
        return available_partitions[0]
    
    # 키의 해시값을 사용하여 파티션 선택
    partition = hash(key) % len(available_partitions)
    return available_partitions[partition]

def create_partitioned_producer():
    """파티셔닝이 적용된 프로듀서 생성"""
    config = producer_config.copy()
    config['partitioner'] = custom_partitioner
    
    try:
        producer = KafkaProducer(**config)
        return producer
    except Exception as e:
        print(f"파티셔닝 프로듀서 생성 오류: {e}")
        return None

def send_partitioned_messages():
    """파티셔닝된 메시지 전송"""
    producer = create_partitioned_producer()
    
    if producer:
        # 키를 사용하여 파티션 지정
        messages = [
            ("user_1", {"user_id": 1, "action": "login"}),
            ("user_2", {"user_id": 2, "action": "logout"}),
            ("user_1", {"user_id": 1, "action": "page_view"}),
            ("user_3", {"user_id": 3, "action": "purchase"})
        ]
        
        for key, message in messages:
            send_message(producer, "partitioned_topic", message, key=key)
        
        producer.close()

# 파티셔닝 예시 실행
# send_partitioned_messages()
```

### 2. 컨슈머 그룹
```python
def create_consumer_group(group_id, topics):
    """컨슈머 그룹 생성"""
    config = consumer_config.copy()
    config['group_id'] = group_id
    
    try:
        consumer = KafkaConsumer(**config)
        consumer.subscribe(topics)
        return consumer
    except Exception as e:
        print(f"컨슈머 그룹 생성 오류: {e}")
        return None

def consumer_group_example():
    """컨슈머 그룹 예시"""
    # 여러 컨슈머가 같은 그룹에 속하면 메시지를 분할하여 처리
    topics = ["user_events"]
    
    # 컨슈머 1
    consumer1 = create_consumer_group("my_group", topics)
    
    # 컨슈머 2
    consumer2 = create_consumer_group("my_group", topics)
    
    # 각 컨슈머는 다른 파티션의 메시지를 처리
    if consumer1 and consumer2:
        print("컨슈머 그룹이 생성되었습니다")
        print("각 컨슈머는 다른 파티션의 메시지를 처리합니다")
        
        # 컨슈머 종료
        consumer1.close()
        consumer2.close()

# 컨슈머 그룹 예시 실행
# consumer_group_example()
```

### 3. 오프셋 관리
```python
def manual_offset_commit():
    """수동 오프셋 커밋"""
    config = consumer_config.copy()
    config['enable_auto_commit'] = False
    
    try:
        consumer = KafkaConsumer(**config)
        consumer.subscribe(["user_events"])
        
        message_count = 0
        
        for message in consumer:
            print(f"메시지 처리: {message.value}")
            message_count += 1
            
            # 10개 메시지마다 오프셋 커밋
            if message_count % 10 == 0:
                consumer.commit()
                print(f"{message_count}개 메시지 처리 완료, 오프셋 커밋")
    
    except KeyboardInterrupt:
        print("컨슈머 중지")
    except Exception as e:
        print(f"수동 오프셋 커밋 오류: {e}")
    finally:
        consumer.close()

# 수동 오프셋 커밋 예시 실행
# manual_offset_commit()
```

## Kafka 실무 적용 예시

### 1. 이벤트 스트리밍
```python
def event_streaming_example():
    """이벤트 스트리밍 예시"""
    
    # 이벤트 프로듀서
    def event_producer():
        producer = create_producer()
        
        if producer:
            events = [
                {"event_type": "user_registration", "user_id": 1, "timestamp": time.time()},
                {"event_type": "user_login", "user_id": 1, "timestamp": time.time()},
                {"event_type": "product_view", "user_id": 1, "product_id": 100, "timestamp": time.time()},
                {"event_type": "add_to_cart", "user_id": 1, "product_id": 100, "timestamp": time.time()},
                {"event_type": "purchase", "user_id": 1, "product_id": 100, "amount": 50.0, "timestamp": time.time()}
            ]
            
            for event in events:
                send_message(producer, "user_events", event, key=str(event["user_id"]))
                time.sleep(1)  # 1초 간격으로 이벤트 전송
            
            producer.close()
    
    # 이벤트 컨슈머
    def event_consumer():
        consumer = create_consumer()
        
        if consumer:
            consumer.subscribe(["user_events"])
            
            for message in consumer:
                event = message.value
                print(f"이벤트 처리: {event['event_type']} - 사용자 {event['user_id']}")
                
                # 이벤트 타입별 처리
                if event["event_type"] == "purchase":
                    print(f"구매 완료: 상품 {event['product_id']}, 금액 {event['amount']}")
                elif event["event_type"] == "user_registration":
                    print(f"신규 사용자 등록: {event['user_id']}")
    
    # 이벤트 스트리밍 실행
    # event_producer()
    # event_consumer()
```

### 2. 로그 수집
```python
def log_collection_example():
    """로그 수집 예시"""
    
    # 로그 프로듀서
    def log_producer():
        producer = create_producer()
        
        if producer:
            import logging
            
            # 로그 메시지 생성
            log_messages = [
                {"level": "INFO", "message": "사용자 로그인 성공", "user_id": 1},
                {"level": "WARNING", "message": "로그인 시도 실패", "user_id": 2},
                {"level": "ERROR", "message": "데이터베이스 연결 실패", "service": "db"},
                {"level": "INFO", "message": "API 요청 처리 완료", "endpoint": "/api/users"},
                {"level": "ERROR", "message": "외부 API 호출 실패", "service": "external_api"}
            ]
            
            for log_msg in log_messages:
                send_message(producer, "application_logs", log_msg, key=log_msg["level"])
                time.sleep(0.5)
            
            producer.close()
    
    # 로그 컨슈머
    def log_consumer():
        consumer = create_consumer()
        
        if consumer:
            consumer.subscribe(["application_logs"])
            
            for message in consumer:
                log_msg = message.value
                
                # 로그 레벨별 처리
                if log_msg["level"] == "ERROR":
                    print(f"🚨 오류 발생: {log_msg['message']}")
                    # 오류 알림 시스템에 전송
                elif log_msg["level"] == "WARNING":
                    print(f"⚠️ 경고: {log_msg['message']}")
                else:
                    print(f"ℹ️ 정보: {log_msg['message']}")
    
    # 로그 수집 실행
    # log_producer()
    # log_consumer()
```

### 3. 데이터 파이프라인
```python
def data_pipeline_example():
    """데이터 파이프라인 예시"""
    
    # 데이터 소스 (프로듀서)
    def data_source():
        producer = create_producer()
        
        if producer:
            # 다양한 데이터 소스에서 데이터 수집
            data_sources = [
                {"source": "web_analytics", "data": {"page_views": 1000, "unique_visitors": 500}},
                {"source": "mobile_app", "data": {"app_opens": 2000, "active_users": 800}},
                {"source": "api_usage", "data": {"api_calls": 5000, "response_time": 150}},
                {"source": "database", "data": {"queries": 3000, "slow_queries": 50}}
            ]
            
            for source_data in data_sources:
                send_message(producer, "raw_data", source_data, key=source_data["source"])
                time.sleep(1)
            
            producer.close()
    
    # 데이터 처리 (컨슈머)
    def data_processor():
        consumer = create_consumer()
        
        if consumer:
            consumer.subscribe(["raw_data"])
            
            for message in consumer:
                raw_data = message.value
                source = raw_data["source"]
                data = raw_data["data"]
                
                # 데이터 전처리
                processed_data = {
                    "source": source,
                    "timestamp": time.time(),
                    "processed_at": time.time(),
                    "data": data
                }
                
                print(f"데이터 처리 완료: {source}")
                
                # 처리된 데이터를 다른 토픽으로 전송
                producer = create_producer()
                if producer:
                    send_message(producer, "processed_data", processed_data, key=source)
                    producer.close()
    
    # 데이터 파이프라인 실행
    # data_source()
    # data_processor()
```

## Kafka 모니터링 및 관리

### 1. 메트릭 수집
```python
def collect_kafka_metrics():
    """Kafka 메트릭 수집"""
    
    # 토픽별 메트릭
    def get_topic_metrics(admin_client, topic_name):
        try:
            metadata = admin_client.describe_topics([topic_name])
            topic_metadata = metadata[topic_name]
            
            metrics = {
                "topic": topic_name,
                "partitions": len(topic_metadata.partitions),
                "replication_factor": len(topic_metadata.partitions[0].replicas)
            }
            
            return metrics
        except Exception as e:
            print(f"토픽 메트릭 수집 오류: {e}")
            return None
    
    # 컨슈머 그룹 메트릭
    def get_consumer_group_metrics(admin_client, group_id):
        try:
            # 컨슈머 그룹 정보 조회
            group_info = admin_client.describe_consumer_groups([group_id])
            
            metrics = {
                "group_id": group_id,
                "state": group_info[group_id].state,
                "members": len(group_info[group_id].members)
            }
            
            return metrics
        except Exception as e:
            print(f"컨슈머 그룹 메트릭 수집 오류: {e}")
            return None
    
    # 메트릭 수집 실행
    admin_client = create_admin_client()
    
    if admin_client:
        # 토픽 메트릭
        topic_metrics = get_topic_metrics(admin_client, "user_events")
        if topic_metrics:
            print(f"토픽 메트릭: {topic_metrics}")
        
        # 컨슈머 그룹 메트릭
        group_metrics = get_consumer_group_metrics(admin_client, "my_consumer_group")
        if group_metrics:
            print(f"컨슈머 그룹 메트릭: {group_metrics}")

# 메트릭 수집 실행
# collect_kafka_metrics()
```

### 2. 성능 모니터링
```python
def performance_monitoring():
    """성능 모니터링"""
    
    # 메시지 처리 속도 측정
    def measure_throughput():
        producer = create_producer()
        consumer = create_consumer()
        
        if producer and consumer:
            # 메시지 전송 속도 측정
            start_time = time.time()
            message_count = 1000
            
            for i in range(message_count):
                message = {"id": i, "data": f"message_{i}"}
                send_message(producer, "performance_test", message)
            
            end_time = time.time()
            throughput = message_count / (end_time - start_time)
            
            print(f"전송 처리량: {throughput:.2f} 메시지/초")
            
            producer.close()
            consumer.close()
    
    # 지연시간 측정
    def measure_latency():
        producer = create_producer()
        consumer = create_consumer()
        
        if producer and consumer:
            # 메시지 전송 시간 기록
            send_time = time.time()
            message = {"timestamp": send_time, "data": "latency_test"}
            
            send_message(producer, "latency_test", message)
            
            # 메시지 수신 시간 측정
            consumer.subscribe(["latency_test"])
            
            for message in consumer:
                receive_time = time.time()
                latency = receive_time - message.value["timestamp"]
                
                print(f"지연시간: {latency:.4f}초")
                break
            
            producer.close()
            consumer.close()
    
    # 성능 모니터링 실행
    # measure_throughput()
    # measure_latency()
```

## 주의사항 및 모범 사례

### 1. 성능 최적화
- **배치 크기**: 적절한 배치 크기 설정
- **압축**: 메시지 압축 사용
- **파티션 수**: 적절한 파티션 수 설정
- **복제 팩터**: 적절한 복제 팩터 설정

### 2. 안정성
- **오프셋 관리**: 적절한 오프셋 관리
- **에러 처리**: 적절한 에러 처리
- **재시도**: 실패 시 재시도 로직
- **모니터링**: 지속적인 모니터링

### 3. 보안
- **인증**: 적절한 인증 설정
- **암호화**: 데이터 암호화
- **접근 제어**: 접근 권한 관리
- **감사**: 로그 및 감사

## 마무리

Apache Kafka는 대용량 실시간 데이터 스트리밍을 위한 강력한 분산 플랫폼입니다. 높은 처리량, 낮은 지연시간, 확장성 등의 특징을 통해 실시간 데이터 처리 시스템을 구축할 수 있습니다. 적절한 설정과 모니터링을 통해 안정적이고 효율적인 스트리밍 시스템을 운영할 수 있습니다.
