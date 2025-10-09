---
title: NoSQL/MongoDB 상세 내용
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
tags:
- 데이터엔지니어링
- NoSQL
- MongoDB
- 문서데이터베이스
- 키값데이터베이스
- 컬럼패밀리
- 그래프데이터베이스
- 트랜잭션
- 변경스트림
- 지리공간쿼리
- 성능최적화
- 실무적용
toc: true
date: 2023-11-10 11:00:00 +0900
comments: false
mermaid: true
math: true
---
# NoSQL/MongoDB 상세 내용

> 231228~231229 학습한 내용 정리

## NoSQL 개요

### 정의
- **NoSQL**: Not Only SQL 또는 Non-SQL
- **비관계형 데이터베이스**: 관계형 데이터베이스가 아닌 데이터 저장 방식
- **유연한 스키마**: 고정된 스키마 없이 데이터 저장
- **수평적 확장**: 여러 서버에 데이터 분산 저장

### 특징
- **유연성**: 다양한 데이터 구조 지원
- **확장성**: 수평적 확장 가능
- **성능**: 높은 성능과 가용성
- **분산**: 분산 환경에서 동작

### 장점
- **유연성**: 스키마 변경 없이 데이터 구조 변경
- **확장성**: 대용량 데이터 처리
- **성능**: 빠른 읽기/쓰기 성능
- **비용**: 오픈소스 솔루션 활용

## NoSQL 데이터베이스 유형

### 1. 문서 데이터베이스 (Document Database)
```python
# MongoDB 예시
from pymongo import MongoClient
import json

# MongoDB 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['example_db']
collection = db['users']

# 문서 데이터 예시
user_document = {
    "_id": 1,
    "name": "홍길동",
    "email": "hong@example.com",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "Seoul",
        "country": "South Korea"
    },
    "hobbies": ["reading", "swimming", "coding"],
    "created_at": "2023-12-01T10:00:00Z"
}

# 문서 삽입
result = collection.insert_one(user_document)
print(f"문서 삽입 완료: {result.inserted_id}")

# 문서 조회
user = collection.find_one({"name": "홍길동"})
print(f"조회된 사용자: {user}")

# 문서 업데이트
collection.update_one(
    {"name": "홍길동"},
    {"$set": {"age": 31}}
)

# 문서 삭제
collection.delete_one({"name": "홍길동"})
```

### 2. 키-값 데이터베이스 (Key-Value Database)
```python
# Redis 예시
import redis

# Redis 연결
r = redis.Redis(host='localhost', port=6379, db=0)

# 키-값 저장
r.set('user:1:name', '홍길동')
r.set('user:1:email', 'hong@example.com')
r.set('user:1:age', 30)

# 키-값 조회
name = r.get('user:1:name')
email = r.get('user:1:email')
age = r.get('user:1:age')

print(f"이름: {name.decode('utf-8')}")
print(f"이메일: {email.decode('utf-8')}")
print(f"나이: {age.decode('utf-8')}")

# 해시 저장
r.hset('user:1', mapping={
    'name': '홍길동',
    'email': 'hong@example.com',
    'age': 30
})

# 해시 조회
user_data = r.hgetall('user:1')
print(f"사용자 데이터: {user_data}")

# 리스트 저장
r.lpush('recent_users', 'user:1', 'user:2', 'user:3')

# 리스트 조회
recent_users = r.lrange('recent_users', 0, -1)
print(f"최근 사용자: {recent_users}")

# 세트 저장
r.sadd('active_users', 'user:1', 'user:2', 'user:3')

# 세트 조회
active_users = r.smembers('active_users')
print(f"활성 사용자: {active_users}")
```

### 3. 컬럼 패밀리 데이터베이스 (Column Family Database)
```python
# Cassandra 예시
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# Cassandra 연결
cluster = Cluster(['localhost'])
session = cluster.connect()

# 키스페이스 생성
session.execute("""
    CREATE KEYSPACE IF NOT EXISTS example_keyspace
    WITH REPLICATION = {
        'class': 'SimpleStrategy',
        'replication_factor': 1
    }
""")

# 키스페이스 사용
session.set_keyspace('example_keyspace')

# 테이블 생성
session.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id UUID PRIMARY KEY,
        name TEXT,
        email TEXT,
        age INT,
        created_at TIMESTAMP
    )
""")

# 데이터 삽입
session.execute("""
    INSERT INTO users (user_id, name, email, age, created_at)
    VALUES (uuid(), '홍길동', 'hong@example.com', 30, toTimestamp(now()))
""")

# 데이터 조회
result = session.execute("SELECT * FROM users WHERE name = '홍길동'")
for row in result:
    print(f"사용자: {row.name}, 이메일: {row.email}, 나이: {row.age}")

# 클러스터 종료
cluster.shutdown()
```

### 4. 그래프 데이터베이스 (Graph Database)
```python
# Neo4j 예시
from neo4j import GraphDatabase

# Neo4j 연결
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# 세션 생성
with driver.session() as session:
    # 노드 생성
    session.run("""
        CREATE (u:User {name: '홍길동', email: 'hong@example.com', age: 30})
    """)
    
    # 관계 생성
    session.run("""
        MATCH (u:User {name: '홍길동'})
        CREATE (u)-[:FRIENDS_WITH]->(friend:User {name: '김철수', email: 'kim@example.com'})
    """)
    
    # 쿼리 실행
    result = session.run("""
        MATCH (u:User)-[:FRIENDS_WITH]->(friend:User)
        WHERE u.name = '홍길동'
        RETURN u.name, friend.name
    """)
    
    for record in result:
        print(f"{record['u.name']}의 친구: {record['friend.name']}")

# 드라이버 종료
driver.close()
```

## MongoDB 상세

### 1. MongoDB 설치 및 설정
```bash
# MongoDB 설치 (Ubuntu)
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

# MongoDB 시작
sudo systemctl start mongod
sudo systemctl enable mongod

# MongoDB 상태 확인
sudo systemctl status mongod
```

### 2. MongoDB 기본 사용법
```python
from pymongo import MongoClient
from bson import ObjectId
import datetime

# MongoDB 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['example_db']

# 컬렉션 생성
users_collection = db['users']
products_collection = db['products']

# 문서 삽입
def insert_user():
    """사용자 문서 삽입"""
    user = {
        "name": "홍길동",
        "email": "hong@example.com",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Seoul",
            "country": "South Korea"
        },
        "hobbies": ["reading", "swimming", "coding"],
        "created_at": datetime.datetime.now()
    }
    
    result = users_collection.insert_one(user)
    print(f"사용자 삽입 완료: {result.inserted_id}")
    return result.inserted_id

# 여러 문서 삽입
def insert_multiple_users():
    """여러 사용자 문서 삽입"""
    users = [
        {
            "name": "김철수",
            "email": "kim@example.com",
            "age": 25,
            "hobbies": ["gaming", "music"]
        },
        {
            "name": "이영희",
            "email": "lee@example.com",
            "age": 28,
            "hobbies": ["cooking", "traveling"]
        }
    ]
    
    result = users_collection.insert_many(users)
    print(f"사용자들 삽입 완료: {result.inserted_ids}")
    return result.inserted_ids

# 문서 조회
def find_users():
    """사용자 문서 조회"""
    # 모든 사용자 조회
    all_users = list(users_collection.find())
    print(f"전체 사용자 수: {len(all_users)}")
    
    # 조건부 조회
    young_users = list(users_collection.find({"age": {"$lt": 30}}))
    print(f"30세 미만 사용자 수: {len(young_users)}")
    
    # 특정 필드만 조회
    user_names = list(users_collection.find({}, {"name": 1, "email": 1}))
    print(f"사용자 이름과 이메일: {user_names}")
    
    # 정렬 조회
    sorted_users = list(users_collection.find().sort("age", -1))
    print(f"나이 순 정렬: {sorted_users}")

# 문서 업데이트
def update_users():
    """사용자 문서 업데이트"""
    # 단일 문서 업데이트
    result = users_collection.update_one(
        {"name": "홍길동"},
        {"$set": {"age": 31}}
    )
    print(f"업데이트된 문서 수: {result.modified_count}")
    
    # 여러 문서 업데이트
    result = users_collection.update_many(
        {"age": {"$lt": 30}},
        {"$set": {"category": "young"}}
    )
    print(f"업데이트된 문서 수: {result.modified_count}")
    
    # 배열에 요소 추가
    result = users_collection.update_one(
        {"name": "홍길동"},
        {"$push": {"hobbies": "gaming"}}
    )
    print(f"취미 추가 완료: {result.modified_count}")

# 문서 삭제
def delete_users():
    """사용자 문서 삭제"""
    # 단일 문서 삭제
    result = users_collection.delete_one({"name": "홍길동"})
    print(f"삭제된 문서 수: {result.deleted_count}")
    
    # 여러 문서 삭제
    result = users_collection.delete_many({"age": {"$lt": 25}})
    print(f"삭제된 문서 수: {result.deleted_count}")

# 집계 파이프라인
def aggregate_users():
    """사용자 데이터 집계"""
    pipeline = [
        # 나이별 그룹화
        {"$group": {
            "_id": "$age",
            "count": {"$sum": 1},
            "names": {"$push": "$name"}
        }},
        # 정렬
        {"$sort": {"_id": 1}}
    ]
    
    result = list(users_collection.aggregate(pipeline))
    print("나이별 사용자 분포:")
    for item in result:
        print(f"나이 {item['_id']}: {item['count']}명 - {item['names']}")

# 인덱스 생성
def create_indexes():
    """인덱스 생성"""
    # 단일 필드 인덱스
    users_collection.create_index("email")
    
    # 복합 인덱스
    users_collection.create_index([("name", 1), ("age", -1)])
    
    # 텍스트 인덱스
    users_collection.create_index([("name", "text"), ("email", "text")])
    
    print("인덱스 생성 완료")

# 인덱스 조회
def list_indexes():
    """인덱스 목록 조회"""
    indexes = list(users_collection.list_indexes())
    print("인덱스 목록:")
    for index in indexes:
        print(f"- {index['name']}: {index['key']}")

# 함수 실행
# insert_user()
# insert_multiple_users()
# find_users()
# update_users()
# delete_users()
# aggregate_users()
# create_indexes()
# list_indexes()
```

### 3. MongoDB 고급 기능
```python
# 트랜잭션
def transaction_example():
    """트랜잭션 예시"""
    with client.start_session() as session:
        with session.start_transaction():
            try:
                # 사용자 생성
                user_result = users_collection.insert_one(
                    {"name": "트랜잭션 사용자", "email": "transaction@example.com"},
                    session=session
                )
                
                # 제품 생성
                product_result = products_collection.insert_one(
                    {"name": "트랜잭션 제품", "price": 100},
                    session=session
                )
                
                print("트랜잭션 성공")
                
            except Exception as e:
                print(f"트랜잭션 실패: {e}")
                session.abort_transaction()

# 변경 스트림
def change_stream_example():
    """변경 스트림 예시"""
    pipeline = [
        {"$match": {"operationType": {"$in": ["insert", "update", "delete"]}}}
    ]
    
    with users_collection.watch(pipeline) as stream:
        print("변경 스트림 시작...")
        for change in stream:
            print(f"변경 감지: {change}")

# 지리공간 쿼리
def geospatial_example():
    """지리공간 쿼리 예시"""
    # 지리공간 인덱스 생성
    users_collection.create_index([("location", "2dsphere")])
    
    # 위치 정보가 있는 사용자 삽입
    users_collection.insert_many([
        {
            "name": "서울 사용자",
            "location": {"type": "Point", "coordinates": [126.9780, 37.5665]}
        },
        {
            "name": "부산 사용자",
            "location": {"type": "Point", "coordinates": [129.0756, 35.1796]}
        }
    ])
    
    # 특정 위치에서 100km 이내의 사용자 조회
    query = {
        "location": {
            "$near": {
                "$geometry": {
                    "type": "Point",
                    "coordinates": [126.9780, 37.5665]
                },
                "$maxDistance": 100000  # 100km
            }
        }
    }
    
    nearby_users = list(users_collection.find(query))
    print(f"근처 사용자: {nearby_users}")

# 텍스트 검색
def text_search_example():
    """텍스트 검색 예시"""
    # 텍스트 인덱스 생성
    users_collection.create_index([("name", "text"), ("email", "text")])
    
    # 텍스트 검색
    search_result = users_collection.find(
        {"$text": {"$search": "홍길동"}}
    )
    
    print("텍스트 검색 결과:")
    for user in search_result:
        print(user)

# 함수 실행
# transaction_example()
# change_stream_example()
# geospatial_example()
# text_search_example()
```

## MongoDB 실무 적용 예시

### 1. 사용자 관리 시스템
```python
class UserManager:
    """사용자 관리 클래스"""
    
    def __init__(self, db):
        self.db = db
        self.users_collection = db['users']
        self.sessions_collection = db['sessions']
    
    def create_user(self, user_data):
        """사용자 생성"""
        # 이메일 중복 확인
        if self.users_collection.find_one({"email": user_data["email"]}):
            raise ValueError("이미 존재하는 이메일입니다")
        
        # 사용자 데이터 검증
        required_fields = ["name", "email", "password"]
        for field in required_fields:
            if field not in user_data:
                raise ValueError(f"필수 필드가 누락되었습니다: {field}")
        
        # 사용자 생성
        user_data["created_at"] = datetime.datetime.now()
        user_data["updated_at"] = datetime.datetime.now()
        
        result = self.users_collection.insert_one(user_data)
        return result.inserted_id
    
    def get_user(self, user_id):
        """사용자 조회"""
        user = self.users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])
        return user
    
    def update_user(self, user_id, update_data):
        """사용자 정보 업데이트"""
        update_data["updated_at"] = datetime.datetime.now()
        
        result = self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    def delete_user(self, user_id):
        """사용자 삭제"""
        result = self.users_collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0
    
    def search_users(self, query, limit=10):
        """사용자 검색"""
        search_criteria = {
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"email": {"$regex": query, "$options": "i"}}
            ]
        }
        
        users = list(self.users_collection.find(search_criteria).limit(limit))
        for user in users:
            user["_id"] = str(user["_id"])
        
        return users
    
    def get_user_statistics(self):
        """사용자 통계"""
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_users": {"$sum": 1},
                    "avg_age": {"$avg": "$age"},
                    "min_age": {"$min": "$age"},
                    "max_age": {"$max": "$age"}
                }
            }
        ]
        
        result = list(self.users_collection.aggregate(pipeline))
        return result[0] if result else {}

# 사용자 관리 시스템 사용
user_manager = UserManager(db)

# 사용자 생성
user_data = {
    "name": "홍길동",
    "email": "hong@example.com",
    "password": "hashed_password",
    "age": 30
}

user_id = user_manager.create_user(user_data)
print(f"사용자 생성 완료: {user_id}")

# 사용자 조회
user = user_manager.get_user(user_id)
print(f"사용자 정보: {user}")

# 사용자 검색
search_results = user_manager.search_users("홍")
print(f"검색 결과: {search_results}")

# 사용자 통계
stats = user_manager.get_user_statistics()
print(f"사용자 통계: {stats}")
```

### 2. 제품 카탈로그 시스템
```python
class ProductCatalog:
    """제품 카탈로그 클래스"""
    
    def __init__(self, db):
        self.db = db
        self.products_collection = db['products']
        self.categories_collection = db['categories']
    
    def create_product(self, product_data):
        """제품 생성"""
        # 카테고리 존재 확인
        if "category_id" in product_data:
            category = self.categories_collection.find_one(
                {"_id": ObjectId(product_data["category_id"])}
            )
            if not category:
                raise ValueError("존재하지 않는 카테고리입니다")
        
        # 제품 데이터 검증
        required_fields = ["name", "price", "description"]
        for field in required_fields:
            if field not in product_data:
                raise ValueError(f"필수 필드가 누락되었습니다: {field}")
        
        # 제품 생성
        product_data["created_at"] = datetime.datetime.now()
        product_data["updated_at"] = datetime.datetime.now()
        
        result = self.products_collection.insert_one(product_data)
        return result.inserted_id
    
    def get_products(self, filters=None, sort=None, limit=20, skip=0):
        """제품 목록 조회"""
        query = filters or {}
        
        cursor = self.products_collection.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        
        products = list(cursor)
        for product in products:
            product["_id"] = str(product["_id"])
        
        return products
    
    def get_product(self, product_id):
        """제품 상세 조회"""
        product = self.products_collection.find_one({"_id": ObjectId(product_id)})
        if product:
            product["_id"] = str(product["_id"])
        return product
    
    def update_product(self, product_id, update_data):
        """제품 정보 업데이트"""
        update_data["updated_at"] = datetime.datetime.now()
        
        result = self.products_collection.update_one(
            {"_id": ObjectId(product_id)},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    def delete_product(self, product_id):
        """제품 삭제"""
        result = self.products_collection.delete_one({"_id": ObjectId(product_id)})
        return result.deleted_count > 0
    
    def search_products(self, search_query, limit=20):
        """제품 검색"""
        # 텍스트 검색
        text_search = {
            "$text": {"$search": search_query}
        }
        
        # 정규식 검색 (텍스트 인덱스가 없는 경우)
        regex_search = {
            "$or": [
                {"name": {"$regex": search_query, "$options": "i"}},
                {"description": {"$regex": search_query, "$options": "i"}}
            ]
        }
        
        # 텍스트 인덱스가 있으면 텍스트 검색 사용
        try:
            products = list(self.products_collection.find(text_search).limit(limit))
        except:
            products = list(self.products_collection.find(regex_search).limit(limit))
        
        for product in products:
            product["_id"] = str(product["_id"])
        
        return products
    
    def get_products_by_category(self, category_id, limit=20):
        """카테고리별 제품 조회"""
        products = list(self.products_collection.find(
            {"category_id": ObjectId(category_id)}
        ).limit(limit))
        
        for product in products:
            product["_id"] = str(product["_id"])
        
        return products
    
    def get_price_range_products(self, min_price, max_price, limit=20):
        """가격 범위별 제품 조회"""
        products = list(self.products_collection.find(
            {"price": {"$gte": min_price, "$lte": max_price}}
        ).limit(limit))
        
        for product in products:
            product["_id"] = str(product["_id"])
        
        return products

# 제품 카탈로그 시스템 사용
product_catalog = ProductCatalog(db)

# 제품 생성
product_data = {
    "name": "스마트폰",
    "price": 800000,
    "description": "최신 스마트폰",
    "category_id": "category_id_here",
    "stock": 100
}

product_id = product_catalog.create_product(product_data)
print(f"제품 생성 완료: {product_id}")

# 제품 조회
products = product_catalog.get_products(limit=10)
print(f"제품 목록: {products}")

# 제품 검색
search_results = product_catalog.search_products("스마트폰")
print(f"검색 결과: {search_results}")

# 가격 범위별 제품 조회
price_range_products = product_catalog.get_price_range_products(500000, 1000000)
print(f"가격 범위 제품: {price_range_products}")
```

### 3. 로그 분석 시스템
```python
class LogAnalyzer:
    """로그 분석 클래스"""
    
    def __init__(self, db):
        self.db = db
        self.logs_collection = db['logs']
    
    def insert_log(self, log_data):
        """로그 삽입"""
        log_data["timestamp"] = datetime.datetime.now()
        
        result = self.logs_collection.insert_one(log_data)
        return result.inserted_id
    
    def get_logs_by_level(self, level, limit=100):
        """로그 레벨별 조회"""
        logs = list(self.logs_collection.find(
            {"level": level}
        ).sort("timestamp", -1).limit(limit))
        
        for log in logs:
            log["_id"] = str(log["_id"])
        
        return logs
    
    def get_logs_by_time_range(self, start_time, end_time, limit=100):
        """시간 범위별 로그 조회"""
        logs = list(self.logs_collection.find(
            {"timestamp": {"$gte": start_time, "$lte": end_time}}
        ).sort("timestamp", -1).limit(limit))
        
        for log in logs:
            log["_id"] = str(log["_id"])
        
        return logs
    
    def get_log_statistics(self, start_time, end_time):
        """로그 통계"""
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": "$level",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]
        
        result = list(self.logs_collection.aggregate(pipeline))
        return result
    
    def get_error_logs(self, limit=50):
        """오류 로그 조회"""
        error_logs = list(self.logs_collection.find(
            {"level": {"$in": ["ERROR", "FATAL"]}}
        ).sort("timestamp", -1).limit(limit))
        
        for log in error_logs:
            log["_id"] = str(log["_id"])
        
        return error_logs
    
    def get_logs_by_source(self, source, limit=100):
        """소스별 로그 조회"""
        logs = list(self.logs_collection.find(
            {"source": source}
        ).sort("timestamp", -1).limit(limit))
        
        for log in logs:
            log["_id"] = str(log["_id"])
        
        return logs

# 로그 분석 시스템 사용
log_analyzer = LogAnalyzer(db)

# 로그 삽입
log_data = {
    "level": "INFO",
    "message": "사용자 로그인 성공",
    "source": "auth_service",
    "user_id": "user_123"
}

log_id = log_analyzer.insert_log(log_data)
print(f"로그 삽입 완료: {log_id}")

# 로그 레벨별 조회
info_logs = log_analyzer.get_logs_by_level("INFO")
print(f"INFO 로그: {info_logs}")

# 오류 로그 조회
error_logs = log_analyzer.get_error_logs()
print(f"오류 로그: {error_logs}")

# 로그 통계
start_time = datetime.datetime.now() - datetime.timedelta(days=1)
end_time = datetime.datetime.now()
stats = log_analyzer.get_log_statistics(start_time, end_time)
print(f"로그 통계: {stats}")
```

## MongoDB 성능 최적화

### 1. 인덱스 최적화
```python
def optimize_indexes():
    """인덱스 최적화"""
    
    # 단일 필드 인덱스
    users_collection.create_index("email", unique=True)
    users_collection.create_index("age")
    users_collection.create_index("created_at")
    
    # 복합 인덱스
    users_collection.create_index([("name", 1), ("age", -1)])
    users_collection.create_index([("category", 1), ("price", 1)])
    
    # 부분 인덱스
    users_collection.create_index(
        [("email", 1)],
        partialFilterExpression={"email": {"$exists": True}}
    )
    
    # 텍스트 인덱스
    users_collection.create_index([("name", "text"), ("description", "text")])
    
    # 지리공간 인덱스
    users_collection.create_index([("location", "2dsphere")])
    
    print("인덱스 최적화 완료")

# 인덱스 최적화 실행
# optimize_indexes()
```

### 2. 쿼리 최적화
```python
def optimize_queries():
    """쿼리 최적화"""
    
    # 프로젝션 사용
    users = list(users_collection.find(
        {"age": {"$gte": 18}},
        {"name": 1, "email": 1, "_id": 0}
    ))
    
    # 제한 사용
    recent_users = list(users_collection.find().sort("created_at", -1).limit(10))
    
    # 스킵 사용
    paginated_users = list(users_collection.find().skip(20).limit(10))
    
    # 집계 파이프라인 사용
    pipeline = [
        {"$match": {"age": {"$gte": 18}}},
        {"$group": {"_id": "$age", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    
    age_distribution = list(users_collection.aggregate(pipeline))
    
    print("쿼리 최적화 완료")

# 쿼리 최적화 실행
# optimize_queries()
```

### 3. 연결 풀링
```python
from pymongo import MongoClient
from pymongo.pool import Pool

# 연결 풀 설정
client = MongoClient(
    'mongodb://localhost:27017/',
    maxPoolSize=50,
    minPoolSize=10,
    maxIdleTimeMS=30000,
    waitQueueTimeoutMS=5000
)

# 연결 풀 상태 확인
def check_connection_pool():
    """연결 풀 상태 확인"""
    pool = client._pool
    print(f"연결 풀 크기: {pool.max_pool_size}")
    print(f"활성 연결 수: {len(pool.sockets)}")
    print(f"대기 중인 연결 수: {pool.wait_queue.qsize()}")

# 연결 풀 상태 확인
# check_connection_pool()
```

## 주의사항 및 모범 사례

### 1. 데이터 모델링
- **문서 구조**: 적절한 문서 구조 설계
- **관계**: 문서 간 관계 설계
- **정규화**: 적절한 정규화 수준
- **스키마**: 유연한 스키마 설계

### 2. 성능 최적화
- **인덱스**: 적절한 인덱스 사용
- **쿼리**: 효율적인 쿼리 작성
- **연결**: 연결 풀링 활용
- **모니터링**: 성능 모니터링

### 3. 보안
- **인증**: 적절한 인증 설정
- **권한**: 접근 권한 관리
- **암호화**: 데이터 암호화
- **백업**: 정기적인 백업

## 마무리

NoSQL과 MongoDB는 유연하고 확장 가능한 데이터 저장 솔루션을 제공합니다. 문서 기반의 데이터 구조, 높은 성능, 수평적 확장성 등의 특징을 통해 다양한 애플리케이션에서 활용할 수 있습니다. 적절한 데이터 모델링과 성능 최적화를 통해 실무에서 요구되는 높은 성능의 데이터베이스 시스템을 구축할 수 있습니다.
