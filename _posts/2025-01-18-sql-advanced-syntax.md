---
title: SQL 고급 문법 - GROUP BY, JOIN, 서브쿼리 완전 가이드
categories:
- 1.TIL
- 1-6.SQL
tags:
- sql
- groupby
- join
- 서브쿼리
- 집계함수
- 조건문
- 날짜함수
toc: true
date: 2023-09-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# SQL 고급 문법 - GROUP BY, JOIN, 서브쿼리 완전 가이드

## 개요

SQL의 고급 문법을 통해 복잡한 데이터 분석과 조작이 가능합니다:

- **GROUP BY**: 데이터 그룹화 및 집계
- **JOIN**: 여러 테이블 연결
- **서브쿼리**: 중첩된 쿼리 구조
- **조건문**: IF, CASE WHEN 활용
- **날짜 함수**: DATE_FORMAT 등

## 1. GROUP BY - 데이터 그룹화

### 기본 GROUP BY

```sql
-- 국가코드별 평균 인구수
SELECT CountryCode, AVG(Population) AS "avg_pop"
FROM city
GROUP BY CountryCode;

-- 여러 컬럼으로 그룹화
SELECT COUNTRYCODE, DISTRICT, COUNT(*)
FROM city
GROUP BY COUNTRYCODE, DISTRICT;
```

### 집계 함수와 GROUP BY

```sql
-- 국가코드별 다양한 집계 통계
SELECT COUNTRYCODE,
    AVG(POPULATION) AS "AVG_POP",
    STDDEV(POPULATION) AS "STD_POP",
    MAX(POPULATION) AS "MAX_POP",
    MIN(POPULATION) AS "MIN_POP"
FROM city
GROUP BY COUNTRYCODE
ORDER BY STD_POP DESC;
```

### WHERE와 HAVING의 차이

```sql
-- WHERE: 그룹화 전 필터링
-- HAVING: 그룹화 후 필터링
SELECT COUNTRYCODE,
    AVG(POPULATION) AS "AVG_POP",
    STDDEV(POPULATION) AS "STD_POP",
    MAX(POPULATION) AS "MAX_POP",
    MIN(POPULATION) AS "MIN_POP"
FROM city
WHERE POPULATION >= 5000000  -- 그룹화 전 필터링
GROUP BY COUNTRYCODE
HAVING STD_POP >= 300000    -- 그룹화 후 필터링
ORDER BY STD_POP DESC;
```

### 복합 조건 GROUP BY

```sql
-- 나라코드에 K가 들어가는 나라들의 인구수 총합이 5백만 이상인 나라만 조회
SELECT COUNTRYCODE, SUM(POPULATION) 'SUM_POP'
FROM city
WHERE COUNTRYCODE LIKE '%K%'
GROUP BY COUNTRYCODE
HAVING SUM_POP >= 5000000;
```

### 실무 예제 - 대륙별 분석

```sql
-- 1. 대륙별 인구수와 GNP 최대값
SELECT Continent, SUM(population), MAX(gnp)
FROM country
GROUP BY Continent;

-- 2. 대륙별 인구수와 GNP 평균 (GNP와 인구수가 0이 아닌 데이터)
SELECT Continent, SUM(population), AVG(gnp)
FROM country
WHERE GNP != 0 AND POPULATION != 0 
GROUP BY Continent;

-- 3. 대륙별 평균 인구수와 평균 GNP (인구수로 내림차순 정렬)
SELECT Continent, AVG(population) AVG_POP, AVG(gnp)
FROM country
WHERE GNP != 0 AND POPULATION != 0 
GROUP BY Continent
ORDER BY AVG_POP DESC;

-- 4. 대륙별 1인당 GNP 계산
SELECT Continent, 
    AVG(population) AVG_POP, 
    AVG(gnp) avg_gnp, 
    AVG(gnp)/AVG(population)*1000 GNP_POP_AVG
FROM country
WHERE GNP != 0 AND POPULATION != 0 
GROUP BY Continent
HAVING GNP_POP_AVG >= 0.01
ORDER BY GNP_POP_AVG DESC;
```

## 2. SELECT 절 고급 문법

### 수학 함수

#### CEIL (올림)

```sql
-- 실수 데이터에서 올림
SELECT CEIL(12.345);  -- 결과: 13
SELECT CEIL(-12.345); -- 결과: -12
```

#### ROUND (반올림)

```sql
-- 반올림 (소수점 둘째 자리까지)
SELECT ROUND(12.345, 2), ROUND(12.343, 2);
-- 결과: 12.35, 12.34

-- 국가별 언어 사용 비율을 소수 첫 번째 자리에서 반올림하여 정수로 표현
SELECT COUNTRYCODE, LANGUAGE, ISOFFICIAL, ROUND(PERCENTAGE, 0) 
FROM countrylanguage;
```

#### TRUNCATE (버림)

```sql
-- 실수 데이터를 버림(내림)
SELECT TRUNCATE(12.345, 2);  -- 결과: 12.34
SELECT TRUNCATE(12.345, 1);  -- 결과: 12.3
```

### 조건문

#### IF 함수

```sql
-- 기본 IF 문법: IF(조건, 참일 때 값, 거짓일 때 값)
SELECT IF(10 > 11, "참입니다", "거짓입니다");
-- 결과: "거짓입니다"

-- 실무 예제: 인구수에 따른 도시 규모 분류
SELECT Name, Population,
    IF(Population >= 1000000, "대도시", "소도시") AS city_type
FROM city;
```

#### CASE WHEN (고급 조건문)

```sql
-- CASE WHEN 기본 문법
CASE
    WHEN (조건1) THEN expr
    WHEN (조건2) THEN expr
    WHEN (조건3) THEN expr
    ELSE expr
END

-- 도시 규모 분류 (다단계 조건)
SELECT Name, Population,
    CASE
        WHEN POPULATION >= 1000000 THEN "Big City"
        WHEN POPULATION BETWEEN 500000 AND 999999 THEN "Middle City"
        ELSE "Small City"
    END CITY_SCALE
FROM city;

-- 등급 분류 예제
SELECT Name, Population,
    CASE
        WHEN Population >= 10000000 THEN "A급"
        WHEN Population >= 5000000 THEN "B급"
        WHEN Population >= 1000000 THEN "C급"
        WHEN Population >= 500000 THEN "D급"
        ELSE "E급"
    END city_grade
FROM city
ORDER BY Population DESC;
```

### 날짜 함수

#### DATE_FORMAT

```sql
-- 날짜 데이터 포맷 변경
USE sakila;

SELECT payment_date, 
    DATE_FORMAT(payment_date, "%Y") as "year",
    DATE_FORMAT(payment_date, "%m") as "month",
    DATE_FORMAT(payment_date, "%d") as "day",
    DATE_FORMAT(payment_date, "%Y년 %m월 %d일") as "formatted_date"
FROM payment;

-- 다양한 날짜 포맷
SELECT 
    payment_date,
    DATE_FORMAT(payment_date, "%Y-%m-%d") as "date_only",
    DATE_FORMAT(payment_date, "%H:%i:%s") as "time_only",
    DATE_FORMAT(payment_date, "%Y년 %m월 %d일 %H시 %i분") as "korean_format"
FROM payment;
```

#### 날짜 관련 함수들

```sql
-- 현재 날짜/시간
SELECT NOW(), CURDATE(), CURTIME();

-- 날짜 계산
SELECT 
    payment_date,
    DATE_ADD(payment_date, INTERVAL 1 DAY) as "next_day",
    DATE_SUB(payment_date, INTERVAL 1 MONTH) as "last_month",
    DATEDIFF(NOW(), payment_date) as "days_ago"
FROM payment;

-- 날짜 추출
SELECT 
    payment_date,
    YEAR(payment_date) as "year",
    MONTH(payment_date) as "month",
    DAY(payment_date) as "day",
    DAYOFWEEK(payment_date) as "day_of_week"
FROM payment;
```

## 3. JOIN - 테이블 연결

### Cartesian Join (카테시안 조인)

```sql
-- Cartesian Join (교차곱)
-- 클라이언트 서비스에서는 사용하지 않음
-- 데이터 분석에서 집계용 코드 부여 시 사용
SELECT * FROM user, addr;
-- 결과: user 테이블의 모든 행 × addr 테이블의 모든 행
```

### INNER JOIN

```sql
-- INNER JOIN: 두 테이블의 공통 데이터만 조회
SELECT *
FROM user  -- FROM절에 오는 테이블이 기준 테이블
JOIN addr ON user.user_id = addr.user_id;  -- 대상 테이블

-- 명시적 INNER JOIN
SELECT u.name, u.age, a.addr
FROM user u
INNER JOIN addr a ON u.user_id = a.user_id;
```

### LEFT JOIN

```sql
-- LEFT JOIN: 왼쪽 테이블의 모든 데이터 + 오른쪽 테이블의 매칭 데이터
-- 모든 사용자의 이름과 주소를 출력 (주소가 없는 사람은 "주소없음"으로 출력)
SELECT user.name, IFNULL(addr.addr, "주소없음") as address
FROM user
LEFT JOIN addr ON user.user_id = addr.user_id;

-- 실무 예제: 고객별 주문 내역 (주문이 없는 고객도 포함)
SELECT c.customer_name, 
    IFNULL(COUNT(o.order_id), 0) as order_count,
    IFNULL(SUM(o.amount), 0) as total_amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name;
```

### RIGHT JOIN

```sql
-- RIGHT JOIN: 오른쪽 테이블의 모든 데이터 + 왼쪽 테이블의 매칭 데이터
SELECT u.name, a.addr
FROM user u
RIGHT JOIN addr a ON u.user_id = a.user_id;
```

### FULL OUTER JOIN

```sql
-- FULL OUTER JOIN: 양쪽 테이블의 모든 데이터
-- MySQL에서는 지원하지 않음 (UNION으로 구현)
SELECT u.name, a.addr
FROM user u
LEFT JOIN addr a ON u.user_id = a.user_id
UNION
SELECT u.name, a.addr
FROM user u
RIGHT JOIN addr a ON u.user_id = a.user_id;
```

### SELF JOIN

```sql
-- SELF JOIN: 같은 테이블을 조인
-- 직원과 상사의 관계 조회
SELECT e.employee_name, m.employee_name as manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;
```

## 4. 서브쿼리 (Subquery)

### SELECT 절 서브쿼리

```sql
-- SELECT 절에서 서브쿼리 사용
-- 도시 이름에 k가 들어가는 도시의 개수, 나라 이름에 A가 들어가는 나라들의 GNP 평균
USE world;

SELECT 
    (SELECT COUNT(*) FROM city WHERE name LIKE '%K%') as city_count,
    (SELECT AVG(gnp) FROM country WHERE name LIKE '%A%') as avg_gnp;

-- 상관 서브쿼리 예제
SELECT 
    c.name as country_name,
    (SELECT COUNT(*) FROM city WHERE CountryCode = c.Code) as city_count,
    (SELECT AVG(Population) FROM city WHERE CountryCode = c.Code) as avg_city_population
FROM country c;
```

### FROM 절 서브쿼리

```sql
-- FROM 절에서 서브쿼리 사용 (인라인 뷰)
SELECT city_sub.CountryCode,
    city_sub.Name as "city_name",
    city_sub.Population as "city_pop",
    country.Name as "country_name"
FROM (
    SELECT COUNTRYCODE, NAME, POPULATION
    FROM city
    WHERE POPULATION >= 8000000
) AS city_sub
JOIN country ON city_sub.CountryCode = country.Code;

-- 복잡한 집계 후 조인
SELECT 
    sales_summary.product_id,
    p.product_name,
    sales_summary.total_sales,
    sales_summary.avg_price
FROM (
    SELECT 
        product_id,
        SUM(quantity) as total_sales,
        AVG(price) as avg_price
    FROM sales
    WHERE sale_date >= '2023-01-01'
    GROUP BY product_id
    HAVING total_sales > 100
) AS sales_summary
JOIN products p ON sales_summary.product_id = p.product_id;
```

### WHERE 절 서브쿼리

```sql
-- WHERE 절에서 서브쿼리 사용
-- 도시 인구수 800만 이상 도시의 국가코드, 국가 이름, 대통령 이름 출력
SELECT code, NAME, headofstate
FROM country
WHERE CODE IN (
    SELECT DISTINCT(CountryCode)
    FROM city
    WHERE POPULATION >= 8000000
);

-- EXISTS 사용
SELECT c.name
FROM country c
WHERE EXISTS (
    SELECT 1
    FROM city
    WHERE CountryCode = c.Code AND Population > 10000000
);

-- 비교 연산자와 서브쿼리
SELECT name, population
FROM country
WHERE population > (
    SELECT AVG(population)
    FROM country
    WHERE continent = 'Asia'
);
```

### 서브쿼리 성능 최적화

```sql
-- IN vs EXISTS 성능 비교
-- IN: 서브쿼리 결과가 적을 때 유리
SELECT * FROM orders
WHERE customer_id IN (
    SELECT customer_id FROM customers WHERE status = 'active'
);

-- EXISTS: 서브쿼리 결과가 많을 때 유리
SELECT * FROM orders o
WHERE EXISTS (
    SELECT 1 FROM customers c 
    WHERE c.customer_id = o.customer_id AND c.status = 'active'
);

-- JOIN으로 변환 (가장 효율적)
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE c.status = 'active';
```

## 5. 고급 집계 함수

### 윈도우 함수 (Window Functions)

```sql
-- ROW_NUMBER: 순위 매기기
SELECT 
    name,
    population,
    ROW_NUMBER() OVER (ORDER BY population DESC) as rank
FROM city;

-- RANK: 동일한 값에 같은 순위 부여
SELECT 
    name,
    population,
    RANK() OVER (ORDER BY population DESC) as rank
FROM city;

-- DENSE_RANK: 동일한 값에 같은 순위, 다음 순위는 연속
SELECT 
    name,
    population,
    DENSE_RANK() OVER (ORDER BY population DESC) as rank
FROM city;

-- PARTITION BY: 그룹별 순위
SELECT 
    CountryCode,
    name,
    population,
    ROW_NUMBER() OVER (PARTITION BY CountryCode ORDER BY population DESC) as country_rank
FROM city;
```

### 누적 집계

```sql
-- 누적 합계
SELECT 
    name,
    population,
    SUM(population) OVER (ORDER BY population DESC) as cumulative_population
FROM city;

-- 이동 평균
SELECT 
    name,
    population,
    AVG(population) OVER (
        ORDER BY population DESC 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) as moving_avg
FROM city;
```

## 6. 실무 활용 예제

### 매출 분석 쿼리

```sql
-- 월별 매출 분석
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') as month,
    COUNT(*) as order_count,
    SUM(amount) as total_sales,
    AVG(amount) as avg_order_value,
    MAX(amount) as max_order_value
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;

-- 고객별 구매 패턴 분석
SELECT 
    c.customer_name,
    COUNT(o.order_id) as order_count,
    SUM(o.amount) as total_spent,
    AVG(o.amount) as avg_order_value,
    MAX(o.order_date) as last_order_date,
    DATEDIFF(NOW(), MAX(o.order_date)) as days_since_last_order
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name
HAVING total_spent > 1000
ORDER BY total_spent DESC;
```

### 재고 관리 쿼리

```sql
-- 재고 부족 상품 조회
SELECT 
    p.product_name,
    p.current_stock,
    p.min_stock_level,
    (p.min_stock_level - p.current_stock) as shortage_amount,
    CASE
        WHEN p.current_stock = 0 THEN '품절'
        WHEN p.current_stock <= p.min_stock_level THEN '재고부족'
        ELSE '정상'
    END as stock_status
FROM products p
WHERE p.current_stock <= p.min_stock_level
ORDER BY shortage_amount DESC;

-- 카테고리별 재고 현황
SELECT 
    c.category_name,
    COUNT(p.product_id) as product_count,
    SUM(p.current_stock) as total_stock,
    AVG(p.current_stock) as avg_stock,
    SUM(CASE WHEN p.current_stock <= p.min_stock_level THEN 1 ELSE 0 END) as low_stock_count
FROM categories c
LEFT JOIN products p ON c.category_id = p.category_id
GROUP BY c.category_id, c.category_name
ORDER BY low_stock_count DESC;
```

## 7. 성능 최적화 팁

### 인덱스 활용

```sql
-- 자주 사용되는 컬럼에 인덱스 생성
CREATE INDEX idx_city_countrycode ON city(CountryCode);
CREATE INDEX idx_city_population ON city(Population);
CREATE INDEX idx_country_continent ON country(Continent);

-- 복합 인덱스
CREATE INDEX idx_city_country_pop ON city(CountryCode, Population);
```

### 쿼리 최적화

```sql
-- 비효율적인 쿼리
SELECT * FROM city WHERE UPPER(name) = 'SEOUL';

-- 효율적인 쿼리
SELECT * FROM city WHERE name = 'Seoul';

-- LIMIT 사용으로 결과 제한
SELECT * FROM city ORDER BY population DESC LIMIT 10;

-- 필요한 컬럼만 선택
SELECT name, population FROM city WHERE CountryCode = 'KOR';
```

## 마무리

SQL의 고급 문법을 통해 복잡한 데이터 분석과 조작이 가능합니다:

### 핵심 포인트
- **GROUP BY**: 데이터 그룹화와 집계 분석
- **JOIN**: 여러 테이블의 데이터 연결
- **서브쿼리**: 중첩된 쿼리로 복잡한 조건 처리
- **조건문**: IF, CASE WHEN으로 데이터 분류
- **날짜 함수**: DATE_FORMAT으로 날짜 데이터 처리

### 실무 활용
- 매출 분석, 재고 관리, 고객 분석 등 다양한 비즈니스 요구사항 해결
- 성능 최적화를 통한 효율적인 데이터 처리
- 인덱스 활용과 쿼리 최적화로 시스템 성능 향상

이러한 고급 SQL 문법을 익혀 실무에서 데이터 분석과 비즈니스 인사이트 도출에 활용하세요.
