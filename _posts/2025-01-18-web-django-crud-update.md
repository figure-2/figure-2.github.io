---
title: Django CRUD 업데이트 - URL 구조화와 템플릿 상속
categories:
- 1.TIL
- 1-5.WEB
tags:
- django
- crud
- url구조화
- 템플릿상속
- post메서드
- base.html
toc: true
date: 2023-08-23 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django CRUD 업데이트 - URL 구조화와 템플릿 상속

## 개요

CRUD 로직을 업데이트하여 더 체계적이고 유지보수하기 쉬운 구조로 개선합니다:

- **APP(posts)에 urls.py 추가 생성**
- **공용 HTML 생성 (base.html)**
- **form method = POST 사용**

## 1. APP에 urls.py 추가 생성

### Django 프로젝트 구조 이해

- `django-admin startproject <pjt-name>` 실행시: pjt-name 폴더에 **settings.py**, **urls.py** 생성
- `django-admin startapp <app-name>` 실행시: app-name 폴더에 **admin.py**, **models.py**, **views.py** 생성
- **추가로 app-name 폴더에 urls.py 생성**

**이유**: 프로젝트 규모가 커질수록 URL이 추가되는데, 하나의 URL로만 관리하기에는 힘들어서 유지보수를 위해 main URL과 sub URL들로 나누는 것입니다.

### URL 구조 변경

#### 변경 전
`board/urls.py` (프로젝트 메인 urls.py):
```python
from django.contrib import admin
from django.urls import path
from posts import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index),
    path('posts/<int:id>/', views.detail),
]
```

#### 변경 후
`board/urls.py` (프로젝트 메인 urls.py):
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('posts/', include('posts.urls')),  # posts 앱의 urls.py로 위임
]
```

`posts/urls.py` (앱별 urls.py):
```python
from django.urls import path
from . import views  # 같은 posts 경로에 있기 때문에 .으로 표기

app_name = 'posts'  # URL 네임스페이스 설정

urlpatterns = [
    path('', views.index, name='index'),  # /posts/
    path('<int:id>/', views.detail, name='detail'),  # /posts/1/
]
```

## 2. 공용 HTML 작성 (base.html)

### 템플릿 상속 구조

공통으로 사용하는 base.html을 만들어 HTML 반복 코드를 줄입니다:

- **최상위 폴더** (예: crud, board)에 `templates` 폴더 생성
- **공통 코드**는 base.html로 분리
- **특정 기능**은 각각의 HTML에서 상속받아 사용

### settings.py 설정

상위 templates 폴더 경로를 찾기 위해 설정을 변경합니다:

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # 변경 후
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

### base.html 생성

`templates/base.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>&#123;% block title %&#125;Django Board&#123;% endblock %&#125;</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="&#123;% url 'posts:index' %&#125;">Django Board</a>
            <div class="navbar-nav">
                <a class="nav-link" href="&#123;% url 'posts:index' %&#125;">Home</a>
                <a class="nav-link" href="&#123;% url 'posts:new' %&#125;">New Post</a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container mt-4">
        &#123;% block content %&#125;
        &#123;% endblock %&#125;
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### 템플릿 상속 사용

`posts/templates/posts/index.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;게시판 - Django Board&#123;% endblock %&#125;

&#123;% block content %&#125;
<h1>게시판</h1>
<table class="table">
    <thead>
        <tr>
            <th>제목</th>
            <th>작성일</th>
            <th>상세보기</th>
        </tr>
    </thead>
    <tbody>
        &#123;% for post in posts %&#125;
        <tr>
            <td>&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;</td>
            <td>&#123;% raw %&#125;&#123;&#123; post.created_at|date:"Y-m-d" &#125;&#125;&#123;% endraw %&#125;</td>
            <td><a href="&#123;% url 'posts:detail' post.id %&#125;">상세보기</a></td>
        </tr>
        &#123;% endfor %&#125;
    </tbody>
</table>
&#123;% endblock %&#125;
```

## 3. POST 메서드 사용

### GET vs POST 메서드

#### 변경 전 (GET 메서드)
```html
<form action="/posts/create/">
    <input type="text" name="title">
    <textarea name="content" cols="30" rows="10"></textarea>
    <input type="submit">
</form>
```

#### 변경 후 (POST 메서드)
```html
<form action="&#123;% url 'posts:create' %&#125;" method="POST">
    &#123;% csrf_token %&#125;
    <div class="mb-3">
        <label for="title" class="form-label">제목</label>
        <input type="text" class="form-control" id="title" name="title">
    </div>
    <div class="mb-3">
        <label for="content" class="form-label">내용</label>
        <textarea class="form-control" name="content" id="content" cols="30" rows="10"></textarea>
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
</form>
```

### 변경사항 비교

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| URL 지정 | `action="/posts/create/"` | `action="&#123;% url 'posts:create' %&#125;"` |
| HTTP 메서드 | GET (기본값) | POST |
| CSRF 보호 | 없음 | `&#123;% csrf_token %&#125;` |
| 스타일링 | 기본 HTML | Bootstrap 클래스 |

### views.py 업데이트

```python
from django.shortcuts import render, redirect
from .models import Post

def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        
        post = Post(title=title, content=content)
        post.save()
        
        return redirect('posts:detail', id=post.id)
    
    return render(request, 'posts/new.html')
```

## 4. 완전한 CRUD 구현

### 프로젝트 구조

```
board/
├── board/
│   ├── settings.py
│   └── urls.py
├── posts/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── posts/
│           ├── index.html
│           ├── detail.html
│           ├── new.html
│           └── edit.html
└── templates/
    └── base.html
```

### URL 설정

`posts/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'posts'

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

### 모델 정의

`posts/models.py`:
```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
```

### 뷰 함수들

`posts/views.py`:
```python
from django.shortcuts import render, redirect, get_object_or_404
from .models import Post

def index(request):
    posts = Post.objects.all()
    context = {'posts': posts}
    return render(request, 'posts/index.html', context)

def detail(request, id):
    post = get_object_or_404(Post, id=id)
    context = {'post': post}
    return render(request, 'posts/detail.html', context)

def new(request):
    return render(request, 'posts/new.html')

def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        
        post = Post(title=title, content=content)
        post.save()
        
        return redirect('posts:detail', id=post.id)
    
    return redirect('posts:new')

def edit(request, id):
    post = get_object_or_404(Post, id=id)
    context = {'post': post}
    return render(request, 'posts/edit.html', context)

def update(request, id):
    if request.method == 'POST':
        post = get_object_or_404(Post, id=id)
        post.title = request.POST.get('title')
        post.content = request.POST.get('content')
        post.save()
        
        return redirect('posts:detail', id=post.id)
    
    return redirect('posts:edit', id=id)

def delete(request, id):
    post = get_object_or_404(Post, id=id)
    post.delete()
    return redirect('posts:index')
```

### 템플릿 파일들

#### detail.html
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125; - Django Board&#123;% endblock %&#125;

&#123;% block content %&#125;
<div class="card mt-4">
    <div class="card-header">
        <h2>&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;</h2>
    </div>
    <div class="card-body">
        <p>&#123;% raw %&#125;&#123;&#123; post.content|linebreaks &#125;&#125;&#123;% endraw %&#125;</p>
    </div>
    <div class="card-footer">
        <small class="text-muted">&#123;% raw %&#125;&#123;&#123; post.created_at &#125;&#125;&#123;% endraw %&#125;</small>
        <div class="float-end">
            <a href="&#123;% url 'posts:edit' post.id %&#125;" class="btn btn-warning">수정</a>
            <a href="&#123;% url 'posts:delete' post.id %&#125;" class="btn btn-danger">삭제</a>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

#### new.html
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;새 글 작성 - Django Board&#123;% endblock %&#125;

&#123;% block content %&#125;
<h1>새 글 작성</h1>

<form action="&#123;% url 'posts:create' %&#125;" method="POST">
    &#123;% csrf_token %&#125;
    <div class="mb-3">
        <label for="title" class="form-label">제목</label>
        <input type="text" class="form-control" id="title" name="title" required>
    </div>
    <div class="mb-3">
        <label for="content" class="form-label">내용</label>
        <textarea class="form-control" name="content" id="content" rows="10" required></textarea>
    </div>
    <button type="submit" class="btn btn-primary">작성</button>
    <a href="&#123;% url 'posts:index' %&#125;" class="btn btn-secondary">취소</a>
</form>
&#123;% endblock %&#125;
```

#### edit.html
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;글 수정 - Django Board&#123;% endblock %&#125;

&#123;% block content %&#125;
<h1>글 수정</h1>

<form action="&#123;% url 'posts:update' post.id %&#125;" method="POST">
    &#123;% csrf_token %&#125;
    <div class="mb-3">
        <label for="title" class="form-label">제목</label>
        <input type="text" class="form-control" id="title" name="title" value="&#123;% raw %&#125;&#123;&#123; post.title &#125;&#125;&#123;% endraw %&#125;" required>
    </div>
    <div class="mb-3">
        <label for="content" class="form-label">내용</label>
        <textarea class="form-control" id="content" name="content" rows="10" required>&#123;% raw %&#125;&#123;&#123; post.content &#125;&#125;&#123;% endraw %&#125;</textarea>
    </div>
    <button type="submit" class="btn btn-primary">수정</button>
    <a href="&#123;% url 'posts:detail' post.id %&#125;" class="btn btn-secondary">취소</a>
</form>
&#123;% endblock %&#125;
```

## 5. 실무 팁

### 1. URL 네임스페이스 활용
```python
# urls.py
app_name = 'posts'

# 템플릿에서 사용
&#123;% url 'posts:detail' post.id %&#125;
```

### 2. 에러 처리
```python
from django.shortcuts import get_object_or_404

def detail(request, id):
    post = get_object_or_404(Post, id=id)  # 404 에러 자동 처리
    return render(request, 'posts/detail.html', {'post': post})
```

### 3. 폼 검증
```python
def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        
        if title and content:  # 기본 검증
            post = Post(title=title, content=content)
            post.save()
            return redirect('posts:detail', id=post.id)
        else:
            # 에러 메시지와 함께 폼 다시 표시
            return render(request, 'posts/new.html', {
                'error': '제목과 내용을 모두 입력해주세요.'
            })
    
    return render(request, 'posts/new.html')
```

이렇게 Django CRUD를 업데이트하여 더 체계적이고 유지보수하기 쉬운 구조로 개선할 수 있습니다!
