---
title: Django ORM과 SQL 비교 - 데이터베이스 조작의 두 가지 방법
categories:
- 1.TIL
- 1-6.SQL
tags:
- django
- orm
- sql
- 데이터베이스
- 쿼리
- 비교
toc: true
date: 2023-09-02 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django ORM과 SQL 비교 - 데이터베이스 조작의 두 가지 방법

## 개요

Django ORM과 SQL은 데이터베이스를 조작하는 두 가지 다른 접근 방식입니다:

- **Django ORM**: Python 객체 지향적 접근
- **SQL**: 직접적인 데이터베이스 쿼리
- **비교 분석**: 각각의 장단점과 사용 시기
- **실무 활용**: 프로젝트에서의 최적 선택

## 1. Django ORM 환경 설정

### 프로젝트 설정

```bash
# 1. 가상환경 설정 및 Django 설치
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install django

# 2. 프로젝트 생성
django-admin startproject ormsql .
# 또는
django-admin startproject movies

# 3. 앱 생성
python manage.py startapp movies
```

### settings.py 설정

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'movies',  # 앱 등록
    'django_extensions',  # ORM 작성을 위해 추가
]
```

### 모델 정의

```python
# models.py
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    country = models.CharField(max_length=100)

class Movie(models.Model):
    title = models.CharField(max_length=200)
    year = models.IntegerField()
    categories = models.ManyToManyField('Category')

class Category(models.Model):
    name = models.CharField(max_length=100)

class Score(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    value = models.FloatField()
```

### 더미 데이터 생성

```python
# movies/management/commands/generate.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from faker import Faker
from movies.models import User, Movie, Category, Score
import random

fake = Faker()

class Command(BaseCommand):
    help = 'Generate dummy data'

    def handle(self, *args, **options):
        # 더미 데이터 생성 로직
        for i in range(100):
            User.objects.create(
                name=fake.name(),
                age=random.randint(18, 80),
                country=fake.country()
            )
        
        # 영화, 카테고리, 점수 데이터도 생성...
        self.stdout.write('Dummy data generated successfully!')
```

### 필요한 패키지 설치

```bash
# 더미 데이터 생성
pip install faker

# ORM 작성을 위한 도구
pip install django-extensions
pip install ipython

# 실행
python manage.py shell_plus --ipython
```

## 2. 기본 조회 (SELECT)

### 모든 데이터 조회

```python
# Django ORM
Movie.objects.all()
User.objects.all()

# 해당하는 SQL
# SELECT * FROM movies_movie;
# SELECT * FROM movies_user;
```

### 정렬

```python
# Django ORM
Movie.objects.all().order_by('-year')

# 해당하는 SQL
# SELECT * FROM movies_movie ORDER BY year DESC;
```

## 3. 조건부 조회 (WHERE)

### 기본 필터링

```python
# Django ORM
User.objects.filter(age=41)

# 해당하는 SQL
# SELECT * FROM movies_user WHERE age=41;
```

### 비교 연산자

```python
# Django ORM - Lookup 방법
Movie.objects.filter(year__gt=2000)  # greater than
Movie.objects.filter(year__lt=2010)  # less than
Movie.objects.filter(year__gte=2000) # greater than or equal
Movie.objects.filter(year__lte=2010) # less than or equal

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE year > 2000;
# SELECT * FROM movies_movie WHERE year < 2010;
# SELECT * FROM movies_movie WHERE year >= 2000;
# SELECT * FROM movies_movie WHERE year <= 2010;
```

### 복합 조건

```python
# Django ORM
Movie.objects.filter(year__gt=2000, year__lt=2010)

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE year > 2000 AND year < 2010;
```

### OR, NOT 조건

```python
from django.db.models import Q

# Django ORM - OR 조건
Movie.objects.filter(Q(year__lt=2000) | Q(year__gt=2010))

# Django ORM - NOT 조건
User.objects.exclude(age__gt=30)

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE year < 2000 OR year > 2010;
# SELECT * FROM movies_user WHERE NOT age > 30;
```

## 4. 집계 함수

### MIN, MAX

```python
from django.db.models import Min, Max

# Django ORM
Movie.objects.aggregate(Min('year'))
Movie.objects.aggregate(Max('year'))
Score.objects.filter(movie_id=1).aggregate(Min('value'), Max('value'))

# 해당하는 SQL
# SELECT min(year) FROM movies_movie;
# SELECT max(year) FROM movies_movie;
# SELECT max(value), min(value) FROM movies_score WHERE movie_id=1;
```

### COUNT

```python
from django.db.models import Count

# Django ORM
Movie.objects.count()
Movie.objects.filter(year__gt=2000).count()
User.objects.values('country').distinct().count()

# 해당하는 SQL
# SELECT count(*) FROM movies_movie;
# SELECT count(*) FROM movies_movie WHERE year > 2000;
# SELECT count(DISTINCT country) FROM movies_user;
```

### SUM, AVG

```python
from django.db.models import Sum, Avg

# Django ORM
Movie.objects.aggregate(Sum('year'))
Score.objects.filter(movie_id=10).aggregate(Sum('value'))
Movie.objects.aggregate(Avg('year'))
Score.objects.filter(movie_id=10).aggregate(Avg('value'))

# 서브쿼리 활용
avg = Score.objects.aggregate(Avg('value'))
Score.objects.filter(value__gt=avg['value__avg'])

# 해당하는 SQL
# SELECT sum(year) FROM movies_movie;
# SELECT sum(value) FROM movies_score WHERE movie_id=10;
# SELECT avg(year) FROM movies_movie;
# SELECT avg(value) FROM movies_score WHERE movie_id=10;
# SELECT * FROM movies_score WHERE value > (SELECT avg(value) FROM movies_score);
```

## 5. 패턴 매칭 (LIKE)

### 문자열 검색

```python
# Django ORM
Movie.objects.filter(title__contains='the')
Movie.objects.filter(title__startswith='the')
Movie.objects.filter(title__endswith='on.')

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE title LIKE '%the%';
# SELECT * FROM movies_movie WHERE title LIKE 'the%';
# SELECT * FROM movies_movie WHERE title LIKE '%on.';
```

### 정규표현식

```python
# Django ORM - 정규표현식 사용
Movie.objects.filter(title__regex=r'g..d')

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE title LIKE '%g__d%';
```

## 6. 범위 검색

### IN, BETWEEN

```python
# Django ORM
Movie.objects.filter(year__in=[2000, 2001, 2002])
User.objects.filter(age__range=[20, 30])

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE year IN (2000, 2001, 2002);
# SELECT * FROM movies_user WHERE age BETWEEN 20 AND 30;
```

## 7. 데이터 수정 및 삭제

### UPDATE

```python
# Django ORM - 개별 수정
actors = Actor.objects.filter(id__range=[1, 10])
for actor in actors:
    actor.age = 50
    actor.save()

# Django ORM - 일괄 수정
Actor.objects.filter(id__range=[1, 10]).update(age=100)

# 해당하는 SQL
# UPDATE movies_actor SET age=100 WHERE id BETWEEN 1 AND 10;
```

### DELETE

```python
# Django ORM
Movie.objects.filter(year__lt=1990).delete()

# 해당하는 SQL
# DELETE FROM movies_movie WHERE year < 1990;
```

## 8. JOIN 연산

### 1:N 관계 조회

```python
# Django ORM - 역참조
User.objects.get(id=1).score_set.all()

# 해당하는 SQL
# SELECT * FROM movies_user
# JOIN movies_score ON movies_user.id = movies_score.user_id
# WHERE movies_score.user_id = 1;
```

### M:N 관계 조회

```python
# Django ORM
Movie.objects.get(id=1).categories.all()
Category.objects.get(name='drama').movies.all()

# 해당하는 SQL
# SELECT * FROM movies_movie
# JOIN movies_category_movies ON movies_movie.id = movies_category_movies.movie_id
# JOIN movies_category ON movies_category_movies.category_id = movies_category.id
# WHERE movies_movie.id = 1;
```

### 집계와 JOIN

```python
# Django ORM
User.objects.get(id=1).score_set.all().aggregate(Avg('value'))

# 해당하는 SQL
# SELECT avg(value) FROM movies_user
# JOIN movies_score ON movies_user.id = movies_score.user_id
# WHERE movies_user.id = 1;
```

## 9. GROUP BY

### 그룹화

```python
from django.db.models import Count

# Django ORM
User.objects.values('country').annotate(Count('id'))

# 해당하는 SQL
# SELECT country, count(*) FROM movies_user GROUP BY country;
```

### 복합 집계

```python
# Django ORM - 나라별 점수 평균
from django.db.models import Avg

User.objects.values('country').annotate(
    user_count=Count('id'),
    avg_score=Avg('score__value')
)

# 해당하는 SQL
# SELECT country, count(*), avg(value) FROM movies_user
# JOIN movies_score ON movies_user.id = movies_score.user_id
# GROUP BY country;
```

## 10. 고급 ORM 기법

### select_related (JOIN 최적화)

```python
# Django ORM - N+1 문제 해결
scores = Score.objects.select_related('user', 'movie').all()

# 해당하는 SQL
# SELECT * FROM movies_score
# JOIN movies_user ON movies_score.user_id = movies_user.id
# JOIN movies_movie ON movies_score.movie_id = movies_movie.id;
```

### prefetch_related (M:N 관계 최적화)

```python
# Django ORM
movies = Movie.objects.prefetch_related('categories').all()

# 해당하는 SQL
# SELECT * FROM movies_movie;
# SELECT * FROM movies_category_movies WHERE movie_id IN (...);
# SELECT * FROM movies_category WHERE id IN (...);
```

### F() 표현식

```python
from django.db.models import F

# Django ORM - 필드 간 연산
Movie.objects.filter(year__gt=F('id') + 1900)

# 해당하는 SQL
# SELECT * FROM movies_movie WHERE year > (id + 1900);
```

### Case/When

```python
from django.db.models import Case, When, IntegerField

# Django ORM
Movie.objects.annotate(
    decade=Case(
        When(year__lt=2000, then=1990),
        When(year__lt=2010, then=2000),
        default=2010,
        output_field=IntegerField()
    )
)

# 해당하는 SQL
# SELECT *, CASE
#   WHEN year < 2000 THEN 1990
#   WHEN year < 2010 THEN 2000
#   ELSE 2010
# END AS decade
# FROM movies_movie;
```

## 11. 성능 비교 및 최적화

### ORM vs SQL 성능

```python
# Django ORM - 느린 쿼리
movies = Movie.objects.all()
for movie in movies:
    print(movie.categories.all())  # N+1 문제

# Django ORM - 최적화된 쿼리
movies = Movie.objects.prefetch_related('categories').all()
for movie in movies:
    print(movie.categories.all())  # 미리 로드됨

# Raw SQL - 직접 제어
from django.db import connection

with connection.cursor() as cursor:
    cursor.execute("""
        SELECT m.title, c.name 
        FROM movies_movie m
        JOIN movies_category_movies cm ON m.id = cm.movie_id
        JOIN movies_category c ON cm.category_id = c.id
    """)
    results = cursor.fetchall()
```

### 쿼리 분석

```python
# Django ORM 쿼리 확인
from django.db import connection

# 쿼리 실행 전
connection.queries.clear()

# ORM 쿼리 실행
Movie.objects.filter(year__gt=2000).select_related('director')

# 실행된 SQL 확인
for query in connection.queries:
    print(query['sql'])
```

## 12. 실무 활용 가이드

### 언제 ORM을 사용할까?

```python
# ✅ ORM 사용 권장 상황
# 1. 간단한 CRUD 작업
Movie.objects.create(title="New Movie", year=2023)

# 2. 복잡한 관계형 데이터 조회
User.objects.prefetch_related('score_set__movie').filter(age__gt=30)

# 3. 데이터베이스 독립적인 코드
Movie.objects.filter(year__gt=2000)  # 모든 DB에서 동작
```

### 언제 Raw SQL을 사용할까?

```python
# ✅ Raw SQL 사용 권장 상황
# 1. 복잡한 집계 쿼리
from django.db import connection

with connection.cursor() as cursor:
    cursor.execute("""
        SELECT 
            u.country,
            COUNT(*) as user_count,
            AVG(s.value) as avg_score,
            MAX(s.value) as max_score
        FROM movies_user u
        LEFT JOIN movies_score s ON u.id = s.user_id
        GROUP BY u.country
        HAVING COUNT(*) > 10
        ORDER BY avg_score DESC
    """)
    results = cursor.fetchall()

# 2. 데이터베이스 특화 기능 사용
# 3. 성능이 중요한 대용량 데이터 처리
```

### 하이브리드 접근법

```python
# ORM과 Raw SQL 조합
def get_movie_statistics():
    # ORM으로 기본 데이터 조회
    movies = Movie.objects.filter(year__gt=2000)
    
    # Raw SQL로 복잡한 집계
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                m.id,
                m.title,
                COUNT(s.id) as score_count,
                AVG(s.value) as avg_score
            FROM movies_movie m
            LEFT JOIN movies_score s ON m.id = s.movie_id
            WHERE m.year > 2000
            GROUP BY m.id, m.title
            HAVING COUNT(s.id) > 5
        """)
        stats = cursor.fetchall()
    
    return movies, stats
```

## 13. 디버깅 및 모니터링

### 쿼리 로깅

```python
# settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'django_queries.log',
        },
    },
    'loggers': {
        'django.db.backends': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
```

### 쿼리 성능 측정

```python
import time
from django.db import connection

def measure_query_performance():
    # ORM 쿼리 성능 측정
    start_time = time.time()
    movies = Movie.objects.select_related('director').filter(year__gt=2000)
    list(movies)  # 쿼리 실행
    orm_time = time.time() - start_time
    
    # Raw SQL 쿼리 성능 측정
    start_time = time.time()
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT m.*, d.name 
            FROM movies_movie m
            JOIN movies_director d ON m.director_id = d.id
            WHERE m.year > 2000
        """)
        results = cursor.fetchall()
    sql_time = time.time() - start_time
    
    print(f"ORM: {orm_time:.4f}s, Raw SQL: {sql_time:.4f}s")
```

## 마무리

Django ORM과 SQL은 각각의 장단점이 있습니다:

### Django ORM의 장점
- **Python 네이티브**: Python 코드로 데이터베이스 조작
- **데이터베이스 독립성**: 다양한 DB에서 동일한 코드 사용
- **보안**: SQL 인젝션 공격 방지
- **유지보수성**: 코드 가독성과 유지보수성 향상

### Raw SQL의 장점
- **성능**: 복잡한 쿼리에서 더 나은 성능
- **유연성**: 데이터베이스 특화 기능 활용 가능
- **정밀한 제어**: 쿼리 최적화에 대한 완전한 제어

실무에서는 두 방식을 적절히 조합하여 사용하는 것이 가장 효과적입니다. 간단한 CRUD 작업은 ORM을, 복잡한 집계나 성능이 중요한 작업은 Raw SQL을 사용하는 것이 좋습니다.
