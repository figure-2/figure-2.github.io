---
title: MongoDB 기초
categories:
- 1.TIL
- 1-9.DATA_ENGINEERING
- NoSQL
tags:
- 데이터엔지니어링
- MongoDB
- NoSQL
- 문서데이터베이스
- CRUD
- 집계
- 인덱싱
- 성능최적화
- 확장성
toc: true
date: 2023-11-10 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# MongoDB 기초
> 가장 인기 있는 NoSQL 문서 데이터베이스

## MongoDB란?
MongoDB는 문서 데이터베이스(Document DB)의 하나로, 가장 인기 있는 NoSQL 데이터베이스입니다. JSON 기반의 Document 기반 데이터 관리를 제공합니다.

## MongoDB 특징
- **Schema-less**: 고정된 스키마가 없고, 같은 Collection 안에서도 문서가 다른 구조를 가질 수 있음
- **No JOIN**: RDBMS의 Join이 없음
- **효율적인 CRUD**: Create, Read, Update, Delete가 매우 효율적이고 빠르게 작동
- **Scale-Out**: 수평적 확장이 용이하고, Sharding으로 클러스터 구축 가능

## MongoDB 데이터 구조
```
Database → Collection (Table) → Document (Row)
```

### 예시 문서
```json
{
    "_id": ObjectId("65165168464546541651"),
    "username": "HiHiHi",
    "name": {
        "first": "Yujin", 
        "last": "Jeong"
    },
    "age": 25,
    "interests": ["programming", "music"],
    "address": {
        "city": "Seoul",
        "country": "Korea"
    }
}
```

## 핵심 개념

### 1. Document (문서)
- MongoDB의 레코드
- 필드와 값으로 구성
- JSON 형태로 데이터 표기
- `_id`: 자동 생성되는 Primary Key

### 2. Collection (컬렉션)
- 문서들의 집합
- RDBMS의 Table과 유사
- 스키마가 정의되어 있지 않음

### 3. Database
- 컬렉션의 집합
- RDBMS의 Database와 동일

### 4. Dot Notation (점 표기법)
MongoDB에서는 `.`을 찍어 세부 레벨의 객체에 접근 가능

**예시 1:**
```json
"contribs": ["Turing machine", "Turing test", "Turingery"]
```
→ `contribs.2`로 "Turingery"에 접근

**예시 2:**
```json
"contact": {
    "phone": {
        "type": "cell", 
        "number": "111-222-3333"
    }
}
```
→ `contact.phone.number`로 전화번호에 접근

## MongoDB vs RDBMS 비교

| MongoDB | RDBMS |
|---------|-------|
| Database | Database |
| Collection | Table |
| Document | Tuple/Row |
| Key/Field | Column |
| Embedded Documents | Table Join |
| Primary Key (_id) | Primary Key |

## 기본 명령어

### 데이터베이스 관리
```javascript
// 데이터베이스 목록 보기
show dbs

// 현재 데이터베이스 확인
db

// 데이터베이스 통계
db.stats()

// 데이터베이스 사용/생성
use mydatabase

// 컬렉션 목록 보기
show collections

// 코드를 읽기 쉽게 표시
pretty()
```

### 컬렉션 생성
```javascript
// 일반 컬렉션 생성
db.createCollection("employees")

// Capped 컬렉션 생성 (크기 제한)
db.createCollection("logs", {
    capped: true, 
    size: 10000
})
```

## CRUD 연산

### 1. Create (생성)

#### 단일 문서 삽입
```javascript
db.student.insert({
    name: '아무개1',
    age: 12,
    major: 'Data Science',
    interest: ['Playing game'],
    course: {
        course_name: 'Big Data',
        credit: 3,
        grade: 'A'
    }
})
```

#### 여러 문서 삽입
```javascript
db.student.insertMany([
    {
        name: '아무개2',
        age: 23,
        major: 'Data Base',
        interest: ['Watching TV', 'Playing game'],
        course: {
            course_name: 'Data Base',
            credit: 3,
            grade: 'A+'
        }
    },
    {
        name: '아무개3',
        age: 23,
        major: 'Programming',
        interest: ['Youtube', 'Instagram'],
        course: {
            course_name: 'Programming3',
            credit: 3,
            grade: 'A+'
        }
    }
])
```

### 2. Read (조회)

#### 기본 조회
```javascript
// 모든 문서 조회
db.student.find()

// 조건에 맞는 문서 조회
db.student.find({age: 23})

// 첫 번째 문서만 조회
db.student.findOne({major: 'Data Science'})
```

#### 비교 연산자
| Operator | Meaning | Description |
|----------|---------|-------------|
| $eq | equals | 일치하는 값 |
| $gt | greater than | 큰 값 |
| $gte | greater than or equals | 크거나 같은 값 |
| $lt | less than | 작은 값 |
| $lte | less than or equals | 작거나 같은 값 |
| $ne | not equal | 일치하지 않는 값 |
| $in | in an array | 배열 안에 속하는 값 |
| $nin | none in an array | 배열 안에 속하지 않는 값 |

**예시:**
```javascript
// 나이가 23 이상 27 이하인 학생
db.student.find({age: {$gte: 23, $lte: 27}})

// 특정 전공에 속하는 학생
db.student.find({major: {$in: ['Data Science', 'Programming']}})
```

#### 논리 연산자
| Operator | Description |
|----------|-------------|
| $or | OR |
| $and | AND |
| $not | NOT |
| $nor | A도 B도 아님 |

**예시:**
```javascript
db.student.find({
    $or: [
        {major: '데이터사이언스'},
        {GPA: {$gt: 3.0}}
    ]
})
```

#### 정렬, 제한, 건너뛰기
```javascript
// 정렬 (1: 오름차순, -1: 내림차순)
db.student.find().sort({name: -1}).pretty()

// 개수 제한
db.student.find().limit(3)

// 건너뛰기
db.student.find().skip(3)

// 조합
db.student.find().sort({age: -1}).limit(5).skip(10)
```

### 3. Update (수정)

#### 기본 수정
```javascript
db.collection_name.update(query, update)
```

**예시:**
```javascript
// 나이 변경
db.student.update(
    {name: '홍길동'},
    {$set: {age: 30}}
)

// 나이 증가
db.student.update(
    {name: '홍길동'},
    {$inc: {age: 1}}
)
```

#### Update 연산자
| Operator | Description |
|----------|-------------|
| $inc | field의 value를 지정한 수만큼 증가 |
| $rename | field 이름을 변경 |
| $set | update할 field의 value 설정 |
| $unset | document에서 설정된 field 삭제 |
| $min | 설정값이 field value보다 작은 경우만 update |
| $max | 설정값이 field value보다 큰 경우만 update |
| $currentDate | 현재 시간 설정 |

### 4. Delete (삭제)

#### 문서 삭제
```javascript
// 조건에 맞는 문서 삭제
db.student.remove({gpa: {$lt: 1.0}})

// 모든 문서 삭제
db.student.remove({})
```

#### 컬렉션/데이터베이스 삭제
```javascript
// 컬렉션 삭제
db.student.drop()

// 데이터베이스 삭제
db.dropDatabase()
```

## 고급 기능

### 1. 집계 (Aggregation)
```javascript
// 전공별 학생 수 집계
db.student.aggregate([{
    $group: {
        _id: '$major',
        number: {'$sum': 1}
    }
}])

// 음식 종류별 평점 합계
db.restaurant.aggregate([{
    $group: {
        _id: '$type_of_food',
        total_rating: {'$sum': '$rating'}
    }
}])
```

### 2. 사용자 정의 함수
```javascript
// forEach 함수
db.student.find().forEach(function(doc) {
    print("Students: " + doc.name)
})

// 사용자 정의 함수
exports = function() {
    return "Hello, world!";
};
```

## 인덱싱
```javascript
// 단일 필드 인덱스
db.student.createIndex({name: 1})

// 복합 인덱스
db.student.createIndex({major: 1, age: -1})

// 텍스트 인덱스
db.student.createIndex({content: "text"})
```

## 주요 학습 포인트

### 1. 문서 기반 모델링
- 중첩된 객체 구조 활용
- 배열 데이터 처리
- 관계형 데이터의 내장 문서화

### 2. 유연한 스키마
- 동적 필드 추가/제거
- 다양한 데이터 타입 지원
- 스키마 마이그레이션 불필요

### 3. 성능 최적화
- 적절한 인덱스 설계
- 쿼리 최적화
- 집계 파이프라인 활용

### 4. 확장성
- 샤딩을 통한 수평적 확장
- 복제를 통한 고가용성
- 클러스터 관리

MongoDB는 현대적인 웹 애플리케이션과 빅데이터 처리에 최적화된 강력한 NoSQL 데이터베이스입니다.
