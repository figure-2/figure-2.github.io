---
title: Django CRUD - 게시판 만들기의 핵심
categories:
- 1.TIL
- 1-5.WEB
tags:
- django
- crud
- 게시판
- 모델
- 뷰
- 템플릿
- orm
toc: true
date: 2023-08-22 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django CRUD - 게시판 만들기의 핵심

## CRUD 개요

**CRUD**는 웹 애플리케이션의 기본 기능입니다:

- **C**reate (게시물 생성)
- **R**ead (게시물 읽기)
- **U**pdate (게시물 수정)
- **D**elete (게시물 삭제)

## 프로젝트 구조 작성

### 1. 프로젝트 생성 및 설정

```bash
# 1. 프로젝트 폴더 생성
mkdir crud_project
cd crud_project

# 2. Django 프로젝트 생성
django-admin startproject crud .

# 3. 가상환경 설정
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

# 4. Django 설치
pip install django

# 5. 서버 실행 확인
python manage.py runserver
```

### 2. 앱 생성 및 등록

```bash
# 앱 생성
django-admin startapp posts
```

`settings.py`에서 앱 등록:
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'posts',  # 추가
]
```

### 3. 모델 정의

`posts/models.py`:
```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
```

### 4. 데이터베이스 마이그레이션

```bash
# 마이그레이션 파일 생성
python manage.py makemigrations

# 데이터베이스에 반영
python manage.py migrate

# SQL 스키마 확인
python manage.py sqlmigrate posts 0001
```

### 5. 관리자 설정

`posts/admin.py`:
```python
from django.contrib import admin
from .models import Post

admin.site.register(Post)
```

관리자 계정 생성:
```bash
python manage.py createsuperuser
```

## CRUD 로직 구현

### 1. Read (읽기)

#### 전체 게시물 목록 보기

`posts/urls.py`:
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:id>/', views.detail, name='detail'),
]
```

`posts/views.py`:
```python
from django.shortcuts import render
from .models import Post

def index(request):
    posts = Post.objects.all()  # 모든 게시물 가져오기
    
    context = {
        'posts': posts,
    }
    
    return render(request, 'posts/index.html', context)
```

`templates/posts/index.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>게시판</title>
</head>
<body>
    <h1>게시판</h1>
    
    &#123;% for post in posts %&#125;
        <h3>&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;</h3>
        <p>&#123;% raw %&#125;&#123;&#123; post.content|truncatewords:20 &#125;&#125;&#123;% endraw %&#125;</p>
        <a href="&#123;% url 'detail' post.id %&#125;">자세히 보기</a>
        <hr>
    &#123;% endfor %&#125;
</body>
</html>
```

#### 개별 게시물 상세 보기

`posts/views.py`:
```python
def detail(request, id):
    post = Post.objects.get(id=id)  # 특정 게시물 하나 가져오기
    
    context = {
        'post': post,
    }
    
    return render(request, 'posts/detail.html', context)
```

`templates/posts/detail.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;</title>
</head>
<body>
    <h1>&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;</h1>
    <p>&#123;% raw %&#125;&#123;&#123; post.content &#125;&#125;&#123;% endraw %&#125;</p>
    <p>작성일: &#123;% raw %&#125;&#123;&#123; post.created_at &#125;&#125;&#123;% endraw %&#125;</p>
    <p>수정일: &#123;% raw %&#125;&#123;&#123; post.updated_at &#125;&#125;&#123;% endraw %&#125;</p>
    
    <a href="&#123;% url 'index' %&#125;">목록으로</a>
    <a href="&#123;% url 'edit' post.id %&#125;">수정</a>
    <a href="&#123;% url 'delete' post.id %&#125;">삭제</a>
</body>
</html>
```

### 2. Create (생성)

#### 게시물 작성 폼

`posts/urls.py`:
```python
urlpatterns = [
    path('', views.index, name='index'),
    path('new/', views.new, name='new'),
    path('create/', views.create, name='create'),
    path('<int:id>/', views.detail, name='detail'),
]
```

`posts/views.py`:
```python
from django.shortcuts import render, redirect

def new(request):
    return render(request, 'posts/new.html')

def create(request):
    title = request.POST.get('title')
    content = request.POST.get('content')
    
    post = Post()
    post.title = title
    post.content = content
    post.save()
    
    return redirect('detail', post.id)
```

`templates/posts/new.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>새 글 작성</title>
</head>
<body>
    <h1>새 글 작성</h1>
    
    <form method="POST" action="&#123;% url 'create' %&#125;">
        &#123;% csrf_token %&#125;
        <div>
            <label for="title">제목:</label>
            <input type="text" id="title" name="title" required>
        </div>
        <div>
            <label for="content">내용:</label>
            <textarea id="content" name="content" rows="10" cols="50" required></textarea>
        </div>
        <button type="submit">작성</button>
    </form>
    
    <a href="&#123;% url 'index' %&#125;">목록으로</a>
</body>
</html>
```

### 3. Delete (삭제)

`posts/urls.py`:
```python
urlpatterns = [
    path('', views.index, name='index'),
    path('new/', views.new, name='new'),
    path('create/', views.create, name='create'),
    path('<int:id>/', views.detail, name='detail'),
    path('<int:id>/delete/', views.delete, name='delete'),
]
```

`posts/views.py`:
```python
def delete(request, id):
    post = Post.objects.get(id=id)
    post.delete()
    
    return redirect('index')
```

### 4. Update (수정)

`posts/urls.py`:
```python
urlpatterns = [
    path('', views.index, name='index'),
    path('new/', views.new, name='new'),
    path('create/', views.create, name='create'),
    path('<int:id>/', views.detail, name='detail'),
    path('<int:id>/edit/', views.edit, name='edit'),
    path('<int:id>/update/', views.update, name='update'),
    path('<int:id>/delete/', views.delete, name='delete'),
]
```

`posts/views.py`:
```python
def edit(request, id):
    post = Post.objects.get(id=id)
    
    context = {
        'post': post,
    }
    
    return render(request, 'posts/edit.html', context)

def update(request, id):
    title = request.POST.get('title')
    content = request.POST.get('content')
    
    post = Post.objects.get(id=id)
    post.title = title
    post.content = content
    post.save()
    
    return redirect('detail', post.id)
```

`templates/posts/edit.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>글 수정</title>
</head>
<body>
    <h1>글 수정</h1>
    
    <form method="POST" action="&#123;% url 'update' post.id %&#125;">
        &#123;% csrf_token %&#125;
        <div>
            <label for="title">제목:</label>
            <input type="text" id="title" name="title" value="&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;" required>
        </div>
        <div>
            <label for="content">내용:</label>
            <textarea id="content" name="content" rows="10" cols="50" required>&#123;% raw %&#125;&#123;&#123; post.content &#125;&#125;&#123;% endraw %&#125;</textarea>
        </div>
        <button type="submit">수정</button>
    </form>
    
    <a href="&#123;% url 'detail' post.id %&#125;">취소</a>
</body>
</html>
```

## 데이터베이스와 ORM

### RDBMS (관계형 데이터베이스 관리 시스템)
- Oracle, MySQL, PostgreSQL, **SQLite (Django 기본)**

### Django 모델 필드 타입
```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=100)      # 짧은 텍스트
    content = models.TextField()                  # 긴 텍스트
    created_at = models.DateTimeField(auto_now_add=True)  # 생성 시간
    updated_at = models.DateTimeField(auto_now=True)      # 수정 시간
    is_published = models.BooleanField(default=False)     # 불린 값
    view_count = models.IntegerField(default=0)           # 정수
    price = models.DecimalField(max_digits=10, decimal_places=2)  # 소수
```

### ORM (객체 관계 매핑)
- **O**bject (객체): Python
- **R**elational (관계): SQL
- **M**apping (매핑): Django ORM이 Python과 SQL을 연결

**ORM 예시:**
```python
# Python 코드
posts = Post.objects.all()

# SQL로 변환
# SELECT * FROM posts_post;
```

## CRUD 흐름 정리

### 1. Read (읽기)
- `Post.objects.all()` - 모든 게시물 가져오기
- `Post.objects.get(id=1)` - 특정 게시물 하나 가져오기
- `Post.objects.filter(title__contains='Django')` - 조건에 맞는 게시물 가져오기

### 2. Create (생성)
- 사용자에게 입력할 수 있는 폼 제공
- 사용자가 입력한 데이터를 가져와서 DB에 저장

### 3. Delete (삭제)
- 특정 게시물을 찾아서 삭제

### 4. Update (수정)
- 기존 정보를 담은 폼 제공
- 사용자가 입력한 새로운 정보를 가져와서 기존 정보에 덮어씌우기

## 실무 팁

### 1. URL 네이밍
```python
# urls.py
urlpatterns = [
    path('posts/', views.post_list, name='post_list'),
    path('posts/create/', views.post_create, name='post_create'),
    path('posts/<int:id>/', views.post_detail, name='post_detail'),
    path('posts/<int:id>/edit/', views.post_edit, name='post_edit'),
    path('posts/<int:id>/delete/', views.post_delete, name='post_delete'),
]
```

### 2. 에러 처리
```python
from django.shortcuts import render, get_object_or_404

def detail(request, id):
    post = get_object_or_404(Post, id=id)  # 404 에러 처리
    return render(request, 'posts/detail.html', {'post': post})
```

### 3. 폼 검증
```python
def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        
        if title and content:  # 검증
            post = Post(title=title, content=content)
            post.save()
            return redirect('detail', post.id)
        else:
            return render(request, 'posts/new.html', {'error': '제목과 내용을 모두 입력해주세요.'})
    
    return render(request, 'posts/new.html')
```

이렇게 Django CRUD를 통해 완전한 게시판 기능을 구현할 수 있습니다!
