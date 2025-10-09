---
title: Django ModelForm - 폼 자동화와 URL 통합
categories:
- 1.TIL
- 1-5.WEB
tags:
- django
- modelform
- 폼자동화
- url통합
- getpost
- forms.py
toc: true
date: 2023-08-24 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django ModelForm - 폼 자동화와 URL 통합

## 개요

CRUD 로직을 더욱 효율적으로 개선합니다:

- **URL 통합**: create(new, create) / update(edit, update)를 하나의 URL로 통합
- **Form 자동화**: forms.py를 통한 폼 자동 생성
- **GET/POST 방식 통합**: 하나의 뷰에서 GET과 POST 요청을 모두 처리

## 1. URL 통합

### 기존 방식 (URL 분리)
```python
# 기존: 2개의 URL 필요
path('new/', views.new, name='new'),      # 폼 표시
path('create/', views.create, name='create'),  # 데이터 저장
path('edit/<int:id>/', views.edit, name='edit'),  # 수정 폼 표시
path('update/<int:id>/', views.update, name='update'),  # 데이터 업데이트
```

### 개선된 방식 (URL 통합)
```python
# 개선: 1개의 URL로 통합
path('create/', views.create, name='create'),  # GET: 폼 표시, POST: 데이터 저장
path('update/<int:id>/', views.update, name='update'),  # GET: 수정 폼, POST: 데이터 업데이트
```

## 2. Form 자동화 (forms.py)

### forms.py 생성

`articles/forms.py`:
```python
from django import forms
from .models import Article

class ArticleForm(forms.ModelForm):
    
    class Meta:
        model = Article
        fields = '__all__'  # 모든 필드 포함
        # 또는 특정 필드만: fields = ['title', 'content']
```

### ModelForm의 장점

1. **자동 폼 생성**: 모델의 필드를 기반으로 폼 자동 생성
2. **검증 자동화**: 모델의 제약조건에 따른 자동 검증
3. **유지보수성**: 모델 변경 시 폼도 자동으로 업데이트

## 3. GET/POST 방식 통합 처리

### Create 기능 통합

`articles/views.py`:
```python
from django.shortcuts import render, redirect, get_object_or_404
from .models import Article
from .forms import ArticleForm

def create(request):
    # POST 요청: 사용자가 폼을 제출한 경우
    if request.method == 'POST':
        # 사용자가 입력한 데이터로 폼 생성
        form = ArticleForm(request.POST)
        
        # 폼 검증
        if form.is_valid():
            # 데이터 저장
            article = form.save()
            # 상세 페이지로 리다이렉트
            return redirect('articles:detail', id=article.id)
    
    # GET 요청: 사용자에게 빈 폼을 보여주는 경우
    else:
        # 빈 폼 생성
        form = ArticleForm()
    
    # 공통: 폼을 템플릿에 전달
    context = {
        'form': form,
    }
    
    return render(request, 'articles/create.html', context)
```

### 처리 흐름

1. **GET 요청**: 빈 폼을 사용자에게 표시
2. **POST 요청 (유효하지 않은 데이터)**: 검증 실패 시 폼을 다시 표시
3. **POST 요청 (유효한 데이터)**: 데이터 저장 후 리다이렉트

## 4. 템플릿 작성

### base.html 업데이트

`templates/base.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>&#123;% block title %&#125;Django Board&#123;% endblock %&#125;</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="&#123;% url 'articles:index' %&#125;">Django Board</a>
            <div class="navbar-nav">
                <a class="nav-link" href="&#123;% url 'articles:index' %&#125;">Home</a>
                <a class="nav-link" href="&#123;% url 'articles:create' %&#125;">Create</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        &#123;% block body %&#125;
        &#123;% endblock %&#125;
    </div>
</body>
</html>
```

### create.html

`articles/templates/articles/create.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;새 글 작성 - Django Board&#123;% endblock %&#125;

&#123;% block body %&#125;
<h1>새 글 작성</h1>

<form action="" method="POST">
    &#123;% csrf_token %&#125;
    &#123;% raw %&#125;&#123;&#123; form.as_p &#125;&#125;&#123;% endraw %&#125;
    <button type="submit" class="btn btn-primary">작성</button>
    <a href="&#123;% url 'articles:index' %&#125;" class="btn btn-secondary">취소</a>
</form>
&#123;% endblock %&#125;
```

### 폼 렌더링 옵션

```html
<!-- 기본 렌더링 -->
&#123;% raw %&#125;&#123;&#123; form &#125;&#125;&#123;% endraw %&#125;

<!-- 각 필드를 <p> 태그로 감싸기 -->
&#123;% raw %&#125;&#123;&#123; form.as_p &#125;&#125;&#123;% endraw %&#125;

<!-- 각 필드를 <table> 태그로 감싸기 -->
&#123;% raw %&#125;&#123;&#123; form.as_table &#125;&#125;&#123;% endraw %&#125;

<!-- 각 필드를 <ul> 태그로 감싸기 -->
&#123;% raw %&#125;&#123;&#123; form.as_ul &#125;&#125;&#123;% endraw %&#125;

<!-- 개별 필드 렌더링 -->
&#123;% raw %&#125;&#123;&#123; form.title &#125;&#125;&#123;% endraw %&#125;
&#123;% raw %&#125;&#123;&#123; form.content &#125;&#125;&#123;% endraw %&#125;
```

## 5. Update 기능 통합

### views.py

```python
def update(request, id):
    # 기존 게시물 가져오기
    article = get_object_or_404(Article, id=id)
    
    # POST 요청: 사용자가 수정 폼을 제출한 경우
    if request.method == 'POST':
        # 기존 데이터 + 사용자 입력 데이터로 폼 생성
        form = ArticleForm(request.POST, instance=article)
        
        if form.is_valid():
            # 데이터 저장
            form.save()
            return redirect('articles:detail', id=article.id)
    
    # GET 요청: 수정 폼을 보여주는 경우
    else:
        # 기존 데이터로 폼 생성
        form = ArticleForm(instance=article)
    
    context = {
        'form': form,
    }
    
    return render(request, 'articles/update.html', context)
```

### update.html

`articles/templates/articles/update.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;글 수정 - Django Board&#123;% endblock %&#125;

&#123;% block body %&#125;
<h1>글 수정</h1>

<form action="" method="POST">
    &#123;% csrf_token %&#125;
    &#123;% raw %&#125;&#123;&#123; form.as_p &#125;&#125;&#123;% endraw %&#125;
    <button type="submit" class="btn btn-primary">수정</button>
    <a href="&#123;% url 'articles:detail' article.id %&#125;" class="btn btn-secondary">취소</a>
</form>
&#123;% endblock %&#125;
```

## 6. 완전한 CRUD 구현

### URL 설정

`articles/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'articles'

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create, name='create'),
    path('<int:id>/', views.detail, name='detail'),
    path('<int:id>/update/', views.update, name='update'),
    path('<int:id>/delete/', views.delete, name='delete'),
]
```

### 모델 정의

`articles/models.py`:
```python
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
```

### 모든 뷰 함수

`articles/views.py`:
```python
from django.shortcuts import render, redirect, get_object_or_404
from .models import Article
from .forms import ArticleForm

def index(request):
    articles = Article.objects.all()
    context = {'articles': articles}
    return render(request, 'articles/index.html', context)

def detail(request, id):
    article = get_object_or_404(Article, id=id)
    context = {'article': article}
    return render(request, 'articles/detail.html', context)

def create(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save()
            return redirect('articles:detail', id=article.id)
    else:
        form = ArticleForm()
    
    context = {'form': form}
    return render(request, 'articles/create.html', context)

def update(request, id):
    article = get_object_or_404(Article, id=id)
    
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('articles:detail', id=article.id)
    else:
        form = ArticleForm(instance=article)
    
    context = {'form': form}
    return render(request, 'articles/update.html', context)

def delete(request, id):
    article = get_object_or_404(Article, id=id)
    article.delete()
    return redirect('articles:index')
```

## 7. 폼 커스터마이징

### forms.py 고급 설정

```python
from django import forms
from .models import Article

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
                'rows': 10,
                'placeholder': '내용을 입력하세요'
            })
        }
        labels = {
            'title': '제목',
            'content': '내용'
        }
        help_texts = {
            'title': '제목은 100자 이내로 입력해주세요.',
            'content': '내용을 자세히 작성해주세요.'
        }
```

### 폼 검증 추가

```python
class ArticleForm(forms.ModelForm):
    
    class Meta:
        model = Article
        fields = ['title', 'content']
    
    def clean_title(self):
        title = self.cleaned_data.get('title')
        if len(title) < 5:
            raise forms.ValidationError('제목은 5자 이상 입력해주세요.')
        return title
    
    def clean_content(self):
        content = self.cleaned_data.get('content')
        if len(content) < 10:
            raise forms.ValidationError('내용은 10자 이상 입력해주세요.')
        return content
```

## 8. 실무 팁

### 1. 폼 필드 제외
```python
class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        exclude = ['created_at', 'updated_at']  # 특정 필드 제외
```

### 2. 조건부 필드 표시
```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.instance.pk:  # 수정 모드
        self.fields['title'].widget.attrs['readonly'] = True
```

### 3. 폼 에러 처리
```html
<form method="POST">
    &#123;% csrf_token %&#125;
    &#123;% raw %&#125;&#123;&#123; form.as_p &#125;&#125;&#123;% endraw %&#125;
    
    &#123;% if form.errors %&#125;
        <div class="alert alert-danger">
            <ul>
                &#123;% for field, errors in form.errors.items %&#125;
                    &#123;% for error in errors %&#125;
                        <li>&#123;% raw %&#125;&#123;&#123; error &#125;&#125;&#123;% endraw %&#125;</li>
                    &#123;% endfor %&#125;
                &#123;% endfor %&#125;
            </ul>
        </div>
    &#123;% endif %&#125;
    
    <button type="submit">제출</button>
</form>
```

이렇게 Django ModelForm을 활용하면 폼 처리가 훨씬 간단하고 효율적이 됩니다!
