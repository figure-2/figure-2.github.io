---
title: "Django 1:N 관계 완성 - 사용자와 게시물 연결"

categories:
  - WEB_DEVELOPMENT
tags:
  - django
  - 1n관계
  - foreignkey
  - login_required
  - django-bootstrap
  - requirements

toc: true

date: 2023-08-29 09:00:00 +0900
comments: false
mermaid: true
math: true
---

# Django 1:N 관계 완성 - 사용자와 게시물 연결

## 개요

Django에서 1:N 관계를 완성하고 사용자 인증과 연결합니다:

- **1:N 관계 설정**
- **Create 할 때 로그인 인증 처리**
- **django-bootstrap 실습**
- **협업을 위한 라이브러리 목록 저장**

## 1. 1:N 관계 설정

### User 모델 참조 방법

Django에서 User 모델을 참조하는 세 가지 방법:

```python
# articles/models.py
from django.db import models
from accounts.models import User  # 방법 1 (권장하지 않음)
from django.conf import settings  # 방법 2 (권장)
from django.contrib.auth import get_user_model  # 방법 3

class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    # comment_set = 장고가 자동으로 추가해주는 컬럼

    # 방법 1. (권장하지 않음) -> 유지보수가 편하지 않아서
    # user = models.ForeignKey(User, on_delete=models.CASCADE)

    # 방법 2. (권장) settings.AUTH_USER_MODEL == 'accounts.User'
    # user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    # 방법 3. (권장)
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    # user_id = 숫자 1개만 저장

class Comment(models.Model):
    content = models.TextField()
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    # article_id = 장고가 자동으로 추가해주는 컬럼

    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    # user_id = 장고가 자동으로 추가해주는 컬럼
```

### User 모델에서 역참조

```python
# accounts/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    pass
    # article_set = 만들어짐 article와 연결하여(1(User):N(Article))
    # comment_set = 만들어짐 Comment와 연결하여(1(User):N(Comment))
```

## 2. Read 기능 구현

### URL 설정

`project/urls.py`:
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('articles/', include('articles.urls')),
]
```

`articles/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'articles'

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create, name='create'),
    path('<int:id>/', views.detail, name='detail'),
    path('<int:article_id>/comments/create/', views.comment_create, name='comment_create'),
]
```

### 뷰와 템플릿

`articles/views.py`:
```python
from django.shortcuts import render, redirect, get_object_or_404
from .models import Article
from .forms import ArticleForm, CommentForm

def index(request):
    articles = Article.objects.all()
    form = CommentForm()  # 댓글 폼
    
    context = {
        'articles': articles,
        'form': form,
    }
    return render(request, 'articles/index.html', context)
```

`articles/templates/articles/index.html`:
```html
{% extends 'base.html' %}
{% load bootstrap5 %}

{% block title %}게시판 - Django Board{% endblock %}

{% block body %}
<h1>게시판</h1>

{% for article in articles %}
<div class="card mb-3">
    <div class="card-header">
        <h5>{% raw %}{{ article.title }}{% endraw %}</h5>
        <small class="text-muted">작성자: {% raw %}{{ article.user.username }}{% endraw %} | {% raw %}{{ article.created_at }}{% endraw %}</small>
    </div>
    <div class="card-body">
        <p>{% raw %}{{ article.content|truncatewords:20 }}{% endraw %}</p>
        <a href="{% url 'articles:detail' article.id %}" class="btn btn-primary">자세히 보기</a>
    </div>
    
    <!-- 댓글 작성 (로그인한 경우에만) -->
    {% if user.is_authenticated %}
    <div class="card-footer">
        <form action="{% url 'articles:comment_create' article_id=article.id %}" method="POST">
            {% csrf_token %}
            {% bootstrap_form form %}
            <button type="submit" class="btn btn-sm btn-outline-primary">댓글 작성</button>
        </form>
    </div>
    {% endif %}
</div>
{% endfor %}
{% endblock %}
```

## 3. Create 기능 구현

### ArticleForm 생성

`articles/forms.py`:
```python
from django import forms
from .models import Article, Comment

class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = '__all__'

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ('content',)
```

### Create 뷰 (로그인 인증 처리)

`articles/views.py`:
```python
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404

@login_required  # 로그인 필요 데코레이터
def create(request):
    # 방법 1: 수동으로 로그인 확인
    # if not request.user.is_authenticated:
    #     return redirect('accounts:login')
    
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)  # DB로는 아직 넣지 말고 객체에만 넣고 멈춰
            article.user = request.user  # 작성자 설정
            article.save()  # 모든 컬럼에 대해서 정보 채워넣었으니 저장
            return redirect('articles:index')
    else:
        form = ArticleForm()

    context = {
        'form': form
    }
    return render(request, 'articles/form.html', context)
```

### form.html 템플릿

`articles/templates/articles/form.html`:
```html
{% extends 'base.html' %}
{% load bootstrap5 %}

{% block title %}
    {% if article %}
        글 수정 - Django Board
    {% else %}
        새 글 작성 - Django Board
    {% endif %}
{% endblock %}

{% block body %}
<h1>
    {% if article %}
        글 수정
    {% else %}
        새 글 작성
    {% endif %}
</h1>

<form action="" method="POST">
    {% csrf_token %}
    {% bootstrap_form form %}
    <button type="submit" class="btn btn-primary">
        {% if article %}수정{% else %}작성{% endif %}
    </button>
    <a href="{% url 'articles:index' %}" class="btn btn-secondary">취소</a>
</form>
{% endblock %}
```

## 4. 로그인 인증 처리

### @login_required 데코레이터

- `@login_required`: create 함수 실행 전에 로그인 여부 확인
- 로그인 했으면: create 함수 실행
- 로그인 안했으면: login 페이지로 리다이렉트 (next 파라미터 포함)

### accounts/views.py 수정

```python
def login(request):
    if request.method == "POST":
        form = CustomAuthenticationForm(request, request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            
            # next 파라미터 처리
            next_url = request.GET.get('next')  # /articles/create/ 정보 담김
            return redirect(next_url or 'articles:index')  # 단축평가 사용
    else:
        form = CustomAuthenticationForm()

    context = {
        'form': form
    }
    return render(request, 'accounts/form.html', context)

def signup(request):
    if request.user.is_authenticated:  # 로그인이 되어있으면 signup 굳이?
        return redirect('articles:index')
    
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('accounts:login')
    else:
        form = CustomUserCreationForm()

    context = {
        'form': form,
    }
    return render(request, 'accounts/form.html', context)
```

### next 파라미터 동작 원리

1. 로그인하지 않은 상태에서 `/articles/create/` 접근
2. `@login_required` 데코레이터가 `/accounts/login/?next=/articles/create/`로 리다이렉트
3. 로그인 성공 후 `next` 파라미터의 URL로 이동

## 5. django-bootstrap으로 폼 꾸미기

### 설치

```bash
pip install django-bootstrap-v5
```

### settings.py 설정

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'bootstrap5',  # 추가
    'accounts',
    'articles',
]
```

### 템플릿에서 사용

```html
{% extends 'base.html' %}
{% load bootstrap5 %}

{% block body %}
<form action="" method="POST">
    {% csrf_token %}
    {% bootstrap_form form %}
    <button type="submit" class="btn btn-primary">제출</button>
</form>
{% endblock %}
```

### HTML 통일화

`accounts/templates/accounts/form.html`:
```html
{% extends 'base.html' %}
{% load bootstrap5 %}

{% block title %}
    {% if request.resolver_match.url_name == 'signup' %}
        회원가입 - Django Board
    {% else %}
        로그인 - Django Board
    {% endif %}
{% endblock %}

{% block body %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                {% if request.resolver_match.url_name == 'signup' %}
                    <h3>회원가입</h3>
                {% else %}
                    <h3>로그인</h3>
                {% endif %}
            </div>
            <div class="card-body">
                <form action="" method="POST">
                    {% csrf_token %}
                    {% bootstrap_form form %}
                    <button type="submit" class="btn btn-primary">
                        {% if request.resolver_match.url_name == 'signup' %}
                            회원가입
                        {% else %}
                            로그인
                        {% endif %}
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

## 6. 댓글 기능 구현

### 댓글 작성 뷰

`articles/views.py`:
```python
@login_required
def comment_create(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    form = CommentForm(request.POST)

    if form.is_valid():
        comment = form.save(commit=False)
        comment.user = request.user
        comment.article = article
        comment.save()

        return redirect('articles:index')
    
    return redirect('articles:index')
```

### 댓글 표시

`articles/templates/articles/index.html`:
```html
{% for article in articles %}
<div class="card mb-3">
    <div class="card-header">
        <h5>{% raw %}{{ article.title }}{% endraw %}</h5>
        <small class="text-muted">작성자: {% raw %}{{ article.user.username }}{% endraw %} | {% raw %}{{ article.created_at }}{% endraw %}</small>
    </div>
    <div class="card-body">
        <p>{% raw %}{{ article.content|truncatewords:20 }}{% endraw %}</p>
        <a href="{% url 'articles:detail' article.id %}" class="btn btn-primary">자세히 보기</a>
    </div>
    
    <!-- 댓글 목록 -->
    {% if article.comment_set.all %}
    <div class="card-footer">
        <h6>댓글 ({% raw %}{{ article.comment_set.count }}{% endraw %}개)</h6>
        {% for comment in article.comment_set.all|slice:":3" %}
        <div class="border-bottom pb-1 mb-1">
            <small class="text-muted">{% raw %}{{ comment.user.username }}{% endraw %}: {% raw %}{{ comment.content|truncatewords:10 }}{% endraw %}</small>
        </div>
        {% endfor %}
        {% if article.comment_set.count > 3 %}
        <small class="text-muted">... 더 보려면 자세히 보기 클릭</small>
        {% endif %}
    </div>
    {% endif %}
    
    <!-- 댓글 작성 (로그인한 경우에만) -->
    {% if user.is_authenticated %}
    <div class="card-footer">
        <form action="{% url 'articles:comment_create' article_id=article.id %}" method="POST">
            {% csrf_token %}
            {% bootstrap_form form %}
            <button type="submit" class="btn btn-sm btn-outline-primary">댓글 작성</button>
        </form>
    </div>
    {% endif %}
</div>
{% endfor %}
```

## 7. 협업을 위한 라이브러리 목록 저장

### requirements.txt 생성

```bash
# 현재 설치된 패키지 목록 확인
pip list

# requirements.txt 파일 생성
pip freeze > requirements.txt
```

### requirements.txt 내용 예시

```
Django==4.2.7
django-bootstrap-v5==23.3
Pillow==10.1.0
```

### 협업자 설치 방법

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

# 패키지 설치
pip install -r requirements.txt

# 데이터베이스 마이그레이션
python manage.py migrate

# 서버 실행
python manage.py runserver
```

## 8. 실무 팁

### 1. 모델 관계 최적화

```python
# select_related 사용 (1:1, 1:N 관계)
articles = Article.objects.select_related('user').all()

# prefetch_related 사용 (1:N, M:N 관계)
articles = Article.objects.prefetch_related('comment_set').all()
```

### 2. 사용자 권한 관리

```python
from django.contrib.auth.decorators import user_passes_test

def is_author(user):
    return user.is_authenticated

@user_passes_test(is_author)
def edit_article(request, id):
    article = get_object_or_404(Article, id=id)
    if article.user != request.user:
        return redirect('articles:index')
    # 수정 로직
```

### 3. 폼 커스터마이징

```python
class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '제목을 입력하세요'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 10
            })
        }
```

### 4. 에러 처리

```python
def create(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            try:
                article = form.save(commit=False)
                article.user = request.user
                article.save()
                return redirect('articles:index')
            except Exception as e:
                messages.error(request, f'게시물 작성 중 오류가 발생했습니다: {e}')
    else:
        form = ArticleForm()
    
    return render(request, 'articles/form.html', {'form': form})
```

이렇게 Django에서 1:N 관계를 완성하고 사용자 인증과 연결하여 완전한 게시판 시스템을 구축할 수 있습니다!
