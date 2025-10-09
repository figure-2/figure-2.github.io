---
title: "Django 댓글 시스템 - 1:N 관계와 댓글 CRUD"

categories:
  - WEB_DEVELOPMENT
tags:
  - django
  - 댓글시스템
  - 1n관계
  - foreignkey
  - comment
  - crud

toc: true

date: 2023-08-25 09:00:00 +0900
comments: false
mermaid: true
math: true
---

# Django 댓글 시스템 - 1:N 관계와 댓글 CRUD

## 개요

CRUD 로직을 확장하여 댓글 시스템을 구현합니다:

- **Article, Comment 모델 2개 구현**
- **Comment 기능 추가**
- **1:N 관계 설정**
- **댓글 작성 및 삭제**

## 1. 모델 관계 설정

### 1:N 관계 이해

하나의 게시물(Article)에 여러 개의 댓글(Comment)이 달릴 수 있는 관계입니다.

```python
# models.py
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title

class Comment(models.Model):
    content = models.TextField()
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.article.title} - {self.content[:20]}"
```

### ForeignKey 설명

- `models.ForeignKey(Article, on_delete=models.CASCADE)`
- **Article**: 참조하는 모델
- **on_delete=models.CASCADE**: 게시물이 삭제되면 관련 댓글도 함께 삭제

## 2. 템플릿 통합 (form.html)

### 공통 템플릿 생성

create.html과 update.html이 완전히 동일해졌으므로 공통 템플릿을 만듭니다.

`articles/templates/articles/form.html`:
```html
{% extends 'base.html' %}

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
    {% raw %}{{ form.as_p }}{% endraw %}
    <button type="submit" class="btn btn-primary">
        {% if article %}수정{% else %}작성{% endif %}
    </button>
    <a href="{% url 'articles:index' %}" class="btn btn-secondary">취소</a>
</form>
{% endblock %}
```

### views.py 업데이트

```python
def create(request):
    if request.method == "POST":
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save()
            return redirect('articles:detail', id=article.id)
    else:
        form = ArticleForm()
    
    context = {
        'form': form,
    }
    return render(request, 'articles/form.html', context)

def update(request, id):
    article = get_object_or_404(Article, id=id)
    
    if request.method == "POST":
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('articles:detail', id=article.id)
    else:
        form = ArticleForm(instance=article)
    
    context = {
        'form': form,
        'article': article,  # 템플릿에서 수정/작성 구분용
    }
    return render(request, 'articles/form.html', context)
```

## 3. 댓글 작성 기능

### CommentForm 생성

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
        # fields = '__all__'  # article 필드를 제외해야 하므로 사용 불가
        fields = ('content',)  # content 필드만 포함
        # 또는 exclude = ('article',)  # article 필드 제외
```

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
    path('<int:article_id>/comments/create/', views.comment_create, name='comment_create'),
    path('<int:article_id>/comments/<int:id>/delete/', views.comment_delete, name='comment_delete'),
]
```

### 댓글 작성 뷰

`articles/views.py`:
```python
from django.shortcuts import render, redirect, get_object_or_404
from .models import Article, Comment
from .forms import ArticleForm, CommentForm

def comment_create(request, article_id):
    # 사용자가 입력한 정보를 form에 입력
    comment_form = CommentForm(request.POST)
    
    # 유효성 검사
    if comment_form.is_valid():
        # form을 저장하기 전에 추가 데이터 설정
        comment = comment_form.save(commit=False)
        
        # 방법 1: Article 객체를 가져와서 설정
        # article = Article.objects.get(id=article_id)
        # comment.article = article
        
        # 방법 2: article_id 직접 설정 (더 효율적)
        comment.article_id = article_id
        
        # 저장
        comment.save()
        
        return redirect('articles:detail', id=article_id)
    
    # 유효하지 않은 경우 detail 페이지로 리다이렉트
    return redirect('articles:detail', id=article_id)
```

### detail 뷰 업데이트

```python
def detail(request, id):
    article = get_object_or_404(Article, id=id)
    comment_form = CommentForm()  # 빈 댓글 폼
    
    context = {
        'article': article,
        'comment_form': comment_form,
    }
    return render(request, 'articles/detail.html', context)
```

### detail.html 템플릿

`articles/templates/articles/detail.html`:
```html
{% extends 'base.html' %}

{% block title %}{% raw %}{{ article.title }}{% endraw %} - Django Board{% endblock %}

{% block body %}
<div class="card mt-4">
    <div class="card-header">
        <h2>{% raw %}{{ article.title }}{% endraw %}</h2>
    </div>
    <div class="card-body">
        <p>{% raw %}{{ article.content|linebreaks }}{% endraw %}</p>
    </div>
    <div class="card-footer">
        <small class="text-muted">{% raw %}{{ article.created_at }}{% endraw %}</small>
        <div class="float-end">
            <a href="{% url 'articles:update' article.id %}" class="btn btn-warning">수정</a>
            <a href="{% url 'articles:delete' article.id %}" class="btn btn-danger">삭제</a>
        </div>
    </div>
</div>

<!-- 댓글 작성 폼 -->
<div class="card mt-4">
    <div class="card-header">
        <h4>댓글 작성</h4>
    </div>
    <div class="card-body">
        <form action="{% url 'articles:comment_create' article_id=article.id %}" method="POST">
            {% csrf_token %}
            {% raw %}{{ comment_form.as_p }}{% endraw %}
            <button type="submit" class="btn btn-primary">댓글 작성</button>
        </form>
    </div>
</div>

<!-- 댓글 목록 -->
<div class="card mt-4">
    <div class="card-header">
        <h4>댓글 목록</h4>
    </div>
    <div class="card-body">
        {% if not article.comment_set.all|length %}
            <p class="text-muted">댓글이 없습니다.</p>
        {% endif %}
        
        {% for comment in article.comment_set.all %}
        <div class="border-bottom pb-2 mb-2">
            <p>{% raw %}{{ comment.content }}{% endraw %}</p>
            <small class="text-muted">
                {% raw %}{{ comment.created_at }}{% endraw %}
                <a href="{% url 'articles:comment_delete' article_id=article.id id=comment.id %}" 
                   class="text-danger ms-2">삭제</a>
            </small>
        </div>
        {% endfor %}
    </div>
</div>
{% endblock %}
```

## 4. 댓글 삭제 기능

### 댓글 삭제 뷰

`articles/views.py`:
```python
def comment_delete(request, article_id, id):
    # article_id: 게시물 ID, id: 댓글 ID
    comment = get_object_or_404(Comment, id=id)
    comment.delete()
    
    return redirect('articles:detail', id=article_id)
```

## 5. 1:N 관계 활용

### 역참조 (Reverse Lookup)

Django에서 1:N 관계에서 "1"쪽에서 "N"쪽을 참조하는 방법:

```python
# Article에서 Comment들을 가져오기
article = Article.objects.get(id=1)
comments = article.comment_set.all()  # 역참조

# 또는 related_name 설정 후
comments = article.comments.all()  # related_name='comments'인 경우
```

### 모델에 related_name 추가

```python
class Comment(models.Model):
    content = models.TextField()
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='comments')
    created_at = models.DateTimeField(auto_now_add=True)
```

템플릿에서 사용:
```html
{% for comment in article.comments.all %}
    <p>{% raw %}{{ comment.content }}{% endraw %}</p>
{% endfor %}
```

## 6. 댓글 개수 표시

### 템플릿에서 댓글 개수

```html
<div class="card-header">
    <h4>댓글 목록 ({% raw %}{{ article.comment_set.count }}{% endraw %}개)</h4>
</div>
```

### 뷰에서 댓글 개수

```python
def detail(request, id):
    article = get_object_or_404(Article, id=id)
    comment_count = article.comment_set.count()
    comment_form = CommentForm()
    
    context = {
        'article': article,
        'comment_form': comment_form,
        'comment_count': comment_count,
    }
    return render(request, 'articles/detail.html', context)
```

## 7. 댓글 수정 기능 (선택사항)

### URL 추가

```python
path('<int:article_id>/comments/<int:id>/update/', views.comment_update, name='comment_update'),
```

### 뷰 함수

```python
def comment_update(request, article_id, id):
    comment = get_object_or_404(Comment, id=id)
    
    if request.method == 'POST':
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            form.save()
            return redirect('articles:detail', id=article_id)
    else:
        form = CommentForm(instance=comment)
    
    context = {
        'form': form,
        'comment': comment,
    }
    return render(request, 'articles/comment_update.html', context)
```

### 템플릿

`articles/templates/articles/comment_update.html`:
```html
{% extends 'base.html' %}

{% block title %}댓글 수정 - Django Board{% endblock %}

{% block body %}
<h1>댓글 수정</h1>

<form action="" method="POST">
    {% csrf_token %}
    {% raw %}{{ form.as_p }}{% endraw %}
    <button type="submit" class="btn btn-primary">수정</button>
    <a href="{% url 'articles:detail' comment.article.id %}" class="btn btn-secondary">취소</a>
</form>
{% endblock %}
```

## 8. 실무 팁

### 1. 댓글 정렬

```python
# 최신 댓글부터 표시
comments = article.comment_set.all().order_by('-created_at')

# 오래된 댓글부터 표시
comments = article.comment_set.all().order_by('created_at')
```

### 2. 댓글 페이징

```python
from django.core.paginator import Paginator

def detail(request, id):
    article = get_object_or_404(Article, id=id)
    comments = article.comment_set.all().order_by('-created_at')
    
    paginator = Paginator(comments, 10)  # 페이지당 10개
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'article': article,
        'comment_form': CommentForm(),
        'page_obj': page_obj,
    }
    return render(request, 'articles/detail.html', context)
```

### 3. 댓글 작성자 추가

```python
class Comment(models.Model):
    content = models.TextField()
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    author = models.CharField(max_length=50, default='익명')
    created_at = models.DateTimeField(auto_now_add=True)
```

### 4. 댓글 검증

```python
class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ('content',)
    
    def clean_content(self):
        content = self.cleaned_data.get('content')
        if len(content.strip()) < 5:
            raise forms.ValidationError('댓글은 5자 이상 입력해주세요.')
        return content
```

이렇게 Django에서 1:N 관계를 활용한 댓글 시스템을 구현할 수 있습니다!
