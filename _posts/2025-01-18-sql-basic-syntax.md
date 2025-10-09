---
title: SQL 기본 문법 - 데이터베이스 조작의 핵심
categories:
- 1.TIL
- 1-6.SQL
tags:
- sql
- 데이터베이스
- 테이블
- CRUD
- django
- ORM
toc: true
date: 2023-09-01 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# SQL 기본 문법 - 데이터베이스 조작의 핵심

## 개요

SQL(Structured Query Language)은 데이터베이스를 조작하는 표준 언어입니다:

- **데이터베이스 관리**: 테이블 생성, 수정, 삭제
- **데이터 조작**: 데이터 추가, 조회, 수정, 삭제
- **Django 연동**: Django ORM과 SQL 구문 비교
- **실무 활용**: 실제 프로젝트에서의 SQL 활용

## 1. SQL 환경 설정

### 데이터베이스 파일 생성

1. **mysql.sqlite3 파일 생성**
2. **table.sql 파일 생성** → Use Database mysql.sqlite3 연결 (마우스 오른쪽 버튼)
3. **Run Query** (마우스 오른쪽 버튼 OR Ctrl+Shift+Enter)

### 기본 설정

```sql
-- 데이터베이스 연결 확인
SELECT sqlite_version();

-- 현재 데이터베이스 정보 확인
PRAGMA database_list;
```

## 2. 테이블 생성 (CREATE TABLE)

### 기본 테이블 생성

```sql
-- 기본 테이블 생성
CREATE TABLE Post(
    id INTEGER,
    title TEXT,
    content TEXT
);
```

### 제약조건이 포함된 테이블 생성

```sql
-- Post 테이블 생성 (제약조건 포함)
CREATE TABLE Post(
   id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
   title TEXT NOT NULL,
   content TEXT NOT NULL
);

-- Comment 테이블 생성 (외래키 포함)
CREATE TABLE Comment(
   id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
   post_id INTEGER NOT NULL,
   content TEXT NOT NULL,
   FOREIGN KEY (post_id) REFERENCES Post (id)
);
```

### 제약조건 설명

- **PRIMARY KEY**: 기본키 (고유 식별자)
- **AUTOINCREMENT**: 자동 증가
- **NOT NULL**: NULL 값 허용하지 않음
- **FOREIGN KEY**: 외래키 (다른 테이블 참조)

## 3. 테이블 삭제 (DROP TABLE)

```sql
-- 테이블 삭제
DROP TABLE Post;

-- 조건부 테이블 삭제 (테이블이 존재할 때만)
DROP TABLE IF EXISTS Post;
```

## 4. 테이블 변경 (ALTER TABLE)

### 컬럼 추가

```sql
-- 컬럼 추가
ALTER TABLE Post
ADD COLUMN test TEXT;
```

### 컬럼 이름 변경

```sql
-- 컬럼 이름 변경
ALTER TABLE Post
RENAME COLUMN test TO email;
```

### 컬럼 삭제

```sql
-- 컬럼 삭제 (SQLite 3.35.0 이상)
ALTER TABLE Post
DROP COLUMN email;
```

## 5. 데이터 추가 (INSERT)

### 단일 데이터 추가

```sql
-- 단일 데이터 추가
INSERT INTO Post (title, content)
VALUES ('first post', 'hihi');
```

### 다중 데이터 추가

```sql
-- 다중 데이터 추가
INSERT INTO Post (title, content) 
VALUES 
    ('1', '1'), 
    ('2', '2'), 
    ('3', '3');
```

### 외래키가 있는 테이블에 데이터 추가

```sql
-- Comment 테이블에 데이터 추가
INSERT INTO Comment (post_id, content) 
VALUES
    (1, 'hello'),
    (1, 'hello'),
    (2, 'hello'),
    (2, 'hello'),
    (3, 'hello'),
    (3, 'hello'),
    (4, 'hello'),
    (4, 'hello'),
    (5, 'hello'),
    (5, 'hello');
```

## 6. 데이터 조회 (SELECT)

### 기본 조회

```sql
-- 모든 데이터 조회
SELECT * FROM Post;

-- 특정 컬럼만 조회
SELECT title FROM Post;

-- 여러 컬럼 조회
SELECT title, content FROM Post;
```

### 중복 제거

```sql
-- 중복 제거하여 조회
SELECT DISTINCT content FROM Comment;
```

### 조건부 조회

```sql
-- WHERE 조건 사용
SELECT * FROM Post WHERE id = 1;
SELECT * FROM Post WHERE title = 'first post';

-- 비교 연산자 사용
SELECT * FROM Post WHERE id > 2;
SELECT * FROM Post WHERE id BETWEEN 1 AND 3;
```

## 7. 데이터 수정 (UPDATE)

```sql
-- 데이터 수정
UPDATE Post 
SET title = 'updated title', content = 'updated content'
WHERE id = 1;

-- 여러 조건으로 수정
UPDATE Post 
SET content = 'modified'
WHERE id IN (1, 2, 3);
```

## 8. 데이터 삭제 (DELETE)

```sql
-- 특정 데이터 삭제
DELETE FROM Post WHERE id = 1;

-- 모든 데이터 삭제
DELETE FROM Post;

-- 조건부 삭제
DELETE FROM Post WHERE title = 'first post';
```

## 9. JOIN 연산

### INNER JOIN

```sql
-- Post와 Comment 테이블 조인
SELECT p.title, p.content, c.content as comment_content
FROM Post p
INNER JOIN Comment c ON p.id = c.post_id;
```

### LEFT JOIN

```sql
-- 모든 Post와 해당하는 Comment 조회
SELECT p.title, p.content, c.content as comment_content
FROM Post p
LEFT JOIN Comment c ON p.id = c.post_id;
```

## 10. 집계 함수

```sql
-- COUNT: 개수 세기
SELECT COUNT(*) FROM Post;
SELECT COUNT(DISTINCT content) FROM Comment;

-- SUM, AVG, MIN, MAX
SELECT 
    COUNT(*) as total_posts,
    AVG(LENGTH(content)) as avg_content_length
FROM Post;

-- GROUP BY 사용
SELECT post_id, COUNT(*) as comment_count
FROM Comment
GROUP BY post_id;
```

## 11. 정렬 및 제한

```sql
-- ORDER BY: 정렬
SELECT * FROM Post ORDER BY id DESC;
SELECT * FROM Post ORDER BY title ASC, id DESC;

-- LIMIT: 결과 개수 제한
SELECT * FROM Post LIMIT 5;
SELECT * FROM Post LIMIT 5 OFFSET 10;  -- 11번째부터 5개
```

## 12. Django와 SQL 연동

### Django에서 SQL 직접 실행

```python
from django.db import connection

def execute_raw_sql():
    """Django에서 SQL 직접 실행"""
    
    with connection.cursor() as cursor:
        # 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS CustomTable(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 데이터 삽입
        cursor.execute("""
            INSERT INTO CustomTable (name) VALUES (%s)
        """, ['Test Name'])
        
        # 데이터 조회
        cursor.execute("SELECT * FROM CustomTable")
        results = cursor.fetchall()
        
        return results
```

### Django ORM과 SQL 비교

```python
# Django ORM
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()

# ORM 쿼리
posts = Post.objects.filter(title__icontains='first')

# 해당하는 SQL
# SELECT * FROM Post WHERE title LIKE '%first%'
```

## 13. 실무 활용 팁

### 1. 인덱스 생성

```sql
-- 인덱스 생성 (성능 향상)
CREATE INDEX idx_post_title ON Post(title);
CREATE INDEX idx_comment_post_id ON Comment(post_id);

-- 복합 인덱스
CREATE INDEX idx_post_title_content ON Post(title, content);
```

### 2. 뷰 생성

```sql
-- 뷰 생성
CREATE VIEW PostWithCommentCount AS
SELECT 
    p.id,
    p.title,
    p.content,
    COUNT(c.id) as comment_count
FROM Post p
LEFT JOIN Comment c ON p.id = c.post_id
GROUP BY p.id, p.title, p.content;

-- 뷰 사용
SELECT * FROM PostWithCommentCount;
```

### 3. 트랜잭션 처리

```sql
-- 트랜잭션 시작
BEGIN TRANSACTION;

-- 여러 작업 수행
INSERT INTO Post (title, content) VALUES ('Post 1', 'Content 1');
INSERT INTO Post (title, content) VALUES ('Post 2', 'Content 2');

-- 커밋 (변경사항 저장)
COMMIT;

-- 롤백 (변경사항 취소)
-- ROLLBACK;
```

### 4. 데이터 백업 및 복원

```sql
-- 데이터 백업 (SQLite)
.backup main backup.db

-- 스키마만 백업
.schema Post

-- 데이터만 백업
.dump Post
```

## 14. 성능 최적화

### 1. 쿼리 최적화

```sql
-- 비효율적인 쿼리
SELECT * FROM Post WHERE LOWER(title) = 'first post';

-- 효율적인 쿼리
SELECT * FROM Post WHERE title = 'First Post';
```

### 2. EXPLAIN QUERY PLAN 사용

```sql
-- 쿼리 실행 계획 확인
EXPLAIN QUERY PLAN 
SELECT * FROM Post p 
JOIN Comment c ON p.id = c.post_id 
WHERE p.title = 'first post';
```

## 15. 에러 처리 및 디버깅

### 1. 일반적인 에러들

```sql
-- 테이블이 존재하지 않는 경우
-- Error: no such table: NonExistentTable
SELECT * FROM NonExistentTable;

-- 컬럼이 존재하지 않는 경우
-- Error: no such column: non_existent_column
SELECT non_existent_column FROM Post;

-- 제약조건 위반
-- Error: NOT NULL constraint failed
INSERT INTO Post (title) VALUES ('test');  -- content가 NOT NULL인데 누락
```

### 2. 디버깅 팁

```sql
-- 현재 데이터베이스의 모든 테이블 확인
SELECT name FROM sqlite_master WHERE type='table';

-- 특정 테이블의 스키마 확인
.schema Post

-- 테이블 정보 확인
PRAGMA table_info(Post);
```

## 마무리

SQL 기본 문법을 통해 데이터베이스를 효과적으로 조작할 수 있습니다. 테이블 생성부터 데이터 조작까지의 기본적인 SQL 구문을 익히고, Django ORM과의 연동을 통해 실무에서 활용할 수 있는 기반을 마련했습니다.

성능 최적화, 에러 처리, 디버깅 등의 고급 기법을 통해 더욱 효율적인 데이터베이스 관리가 가능합니다.
