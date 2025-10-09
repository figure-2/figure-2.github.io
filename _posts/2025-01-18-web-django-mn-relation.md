---
title: "Django M:N 관계 - 좋아요와 팔로우 기능"

categories:
  - WEB_DEVELOPMENT
tags:
  - django
  - mn관계
  - manytomany
  - 좋아요
  - 팔로우
  - 댓글

toc: true

date: 2023-08-31 09:00:00 +0900
comments: false
mermaid: true
math: true
---

# Django M:N 관계 - 좋아요와 팔로우 기능

## 개요

Django에서 M:N 관계를 활용하여 소셜 기능을 구현합니다:

- **M:N 관계 설정**
- **좋아요, 팔로우 기능 추가**
- **댓글 시스템 완성**

## 1. Comment Create 기능 구현

### Comment 모델 생성

`posts/models.py`:
```python
from django.db import models
from django.conf import settings
from django_resized import ResizedImageField

class Post(models.Model):
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='image/%Y/%m'
    )
    # M:N 관계 - 좋아요 기능
    like_users = models.ManyToManyField(
        settings.AUTH_USER_MODEL, 
        related_name='like_posts'
    )

class Comment(models.Model):
    content = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    post = models.ForeignKey(Post, on_delete=models.CASCADE)  # 어떤 게시물과 연결
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)  # 작성자 저장
```

### 마이그레이션 실행

```bash
python manage.py makemigrations
python manage.py migrate
```

### CommentForm 생성

`posts/forms.py`:
```python
from django import forms
from .models import Post, Comment

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = '__all__'

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ('content',)
```

### 댓글 작성 뷰

`posts/views.py`:
```python
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from .models import Post, Comment
from .forms import PostForm, CommentForm

def index(request):
    posts = Post.objects.all().order_by('-id')
    comment_form = CommentForm()
    
    context = {
        'posts': posts,
        'comment_form': comment_form,
    }
    return render(request, 'posts/index.html', context)

@login_required
def comment_create(request, post_id):
    comment_form = CommentForm(request.POST)
    
    if comment_form.is_valid():
        comment = comment_form.save(commit=False)  # 저장 안됨 - 관계 설정 2개가 빠짐
        
        # 현재 로그인 유저 (user 정보)
        comment.user = request.user
        
        # post_id를 기준으로 찾은 post (post 정보)
        post = get_object_or_404(Post, id=post_id)
        comment.post = post
        
        comment.save()
        
        return redirect('posts:index')
    
    return redirect('posts:index')
```

### URL 설정

`posts/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'posts'

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create, name='create'),
    path('<int:post_id>/comment/', views.comment_create, name='comment_create'),
    path('<int:post_id>/like/', views.like, name='like'),
]
```

### _card.html 업데이트

`posts/templates/_card.html`:
```html
{% load bootstrap5 %}

<div class="card mt-5">
    <img src="{{ post.image.url }}" class="card-img-top" alt="...">
    <div class="card-body">
        <a href="{% url 'posts:like' post_id=post.id %}" class="text-reset text-decoration-none">
            {% if post in user.like_posts.all %}
                <i class="bi bi-heart-fill" style="color: red;"></i>
            {% else %}
                <i class="bi bi-heart"></i>
            {% endif %}
        </a> {{ post.like_users.all|length }}명이 좋아합니다.
        
        <p class="card-text">{{ post.content }}</p>
        <small class="text-muted">{{ post.created_at|timesince }} 전</small>
        <br>
        <a href="{% url 'accounts:profile' username=post.user %}" class="text-reset text-decoration-none">
            {{ post.user }}
        </a>
    </div>
    
    <!-- 댓글 목록 -->
    <div class="card-footer">
        <hr>
        {% for comment in post.comment_set.all %}
            <li>{{ comment.user }}: {{ comment.content }}</li>
        {% endfor %}
        
        <!-- 댓글 작성 (로그인한 경우에만) -->
        {% if user.is_authenticated %}
        <hr>
        <form action="{% url 'posts:comment_create' post_id=post.id %}" method="POST">
            {% csrf_token %}
            {% bootstrap_form comment_form %}
            <button type="submit" class="btn btn-sm btn-outline-primary">댓글 작성</button>
        </form>
        {% endif %}
    </div>
</div>
```

## 2. M:N 관계 연습

### 기본 M:N 관계 예시

`movies/models.py`:
```python
from django.db import models

class Actor(models.Model):
    name = models.CharField(max_length=100)

class Movie(models.Model):
    title = models.CharField(max_length=100)
    actors = models.ManyToManyField(Actor, related_name='movies')
```

### Django Shell에서 테스트

```bash
python manage.py shell
```

```python
from movies.models import Actor, Movie

# 배우 생성
a = Actor(name='정우성')
a.save()

# 영화 생성
m = Movie(title='신세계')
m.save()

# M:N 관계 설정
m.actors.add(a)  # 영화에 배우 추가
a.movies.all()   # 배우가 출연한 영화들
m.actors.all()   # 영화에 출연한 배우들
```

### Django Seed로 더미 데이터 생성

```bash
pip install django-seed
```

`settings.py`:
```python
INSTALLED_APPS = [
    # ... 기존 앱들 ...
    'django_seed',
]
```

더미 데이터 생성:
```bash
python manage.py seed movies --number=10
```

## 3. 좋아요 기능 구현

### Post 모델에 M:N 관계 추가

`posts/models.py`:
```python
class Post(models.Model):
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='image/%Y/%m'
    )
    # M:N 관계 - 좋아요 기능
    like_users = models.ManyToManyField(
        settings.AUTH_USER_MODEL, 
        related_name='like_posts'
    )
```

### 좋아요 뷰

`posts/views.py`:
```python
@login_required
def like(request, post_id):
    # 좋아요 버튼을 누른 유저
    user = request.user
    post = get_object_or_404(Post, id=post_id)
    
    # 이미 좋아요 버튼을 누른 경우 (좋아요 취소)
    if post in user.like_posts.all():
        post.like_users.remove(user)
    # 좋아요 버튼을 아직 안 누른 경우 (좋아요)
    else:
        post.like_users.add(user)
    
    return redirect('posts:index')
```

### 좋아요 버튼 UI

`_card.html`에서 좋아요 버튼:
```html
<div class="card-body">
    <a href="{% url 'posts:like' post_id=post.id %}" class="text-reset text-decoration-none">
        {% if post in user.like_posts.all %}
            <i class="bi bi-heart-fill" style="color: red;"></i>
        {% else %}
            <i class="bi bi-heart"></i>
        {% endif %}
    </a> {{ post.like_users.all|length }}명이 좋아합니다.
    
    <p class="card-text">{{ post.content }}</p>
    <!-- ... 기타 내용 ... -->
</div>
```

### Bootstrap Icons 추가

`base.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Instagram{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
</head>
<body>
    <!-- ... 기타 내용 ... -->
</body>
</html>
```

## 4. 팔로우 기능 구현

### User 모델에 M:N 관계 추가

`accounts/models.py`:
```python
from django.contrib.auth.models import AbstractUser
from django_resized import ResizedImageField

class User(AbstractUser):
    profile_image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='profile',
        blank=True,
        null=True
    )
    # M:N 관계 - 팔로우 기능 (자기 자신과의 관계)
    followings = models.ManyToManyField(
        'self', 
        related_name='followers', 
        symmetrical=False  # 내가 팔로우해도 상대방이 자동으로 팔로우하지 않음
    )
```

### 팔로우 뷰

`accounts/views.py`:
```python
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect, get_object_or_404

@login_required
def follow(request, username):
    User = get_user_model()
    
    me = request.user  # 현재 로그인한 사람
    you = get_user_model().objects.get(username=username)  # 팔로우하고 싶은 사람
    
    # 팔로잉이 이미 되어있는 경우
    if me in you.followers.all():
        me.followings.remove(you)  # 팔로우 취소
    # 팔로잉이 아직 안 된 경우
    else:
        me.followings.add(you)  # 팔로우 추가
    
    return redirect('accounts:profile', username=username)
```

### URL 설정

`accounts/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    path('<str:username>/', views.profile, name='profile'),
    path('<str:username>/follow/', views.follow, name='follow'),
]
```

### 프로필 페이지 업데이트

`accounts/templates/accounts/profile.html`:
```html
{% extends 'base.html' %}

{% block title %}{{ user_info.username }}의 프로필{% endblock %}

{% block body %}
<div class="row mb-4">
    <div class="col-4">
        {% if user_info.profile_image %}
            <img src="{{ user_info.profile_image.url }}" alt="" class="img-fluid rounded-circle" style="width: 150px; height: 150px; object-fit: cover;">
        {% else %}
            <div class="bg-secondary rounded-circle d-flex align-items-center justify-content-center" style="width: 150px; height: 150px;">
                <span class="text-white">{{ user_info.username|first|upper }}</span>
            </div>
        {% endif %}
    </div>

    <div class="col-8">
        <div class="row mb-3">
            <div class="col-3">
                <h4>{{ user_info.username }}</h4>
            </div>
            <div class="col-4">
                <!-- user: 로그인한 사람, user_info: 프로필 페이지 유저 -->
                {% if user != user_info %}
                    {% if user in user_info.followers.all %}
                        <a href="{% url 'accounts:follow' username=user_info.username %}" class="btn btn-primary btn-sm">팔로잉</a>
                    {% else %}
                        <a href="{% url 'accounts:follow' username=user_info.username %}" class="btn btn-secondary btn-sm">팔로우</a>
                    {% endif %}
                {% endif %}
            </div>
        </div>
        <div class="row">
            <div class="col">
                <strong>{{ user_info.post_set.all|length }}</strong><br>
                <span>게시물</span>
            </div>
            <div class="col">
                <strong>{{ user_info.followers.all|length }}</strong><br>
                <span>팔로워</span>
            </div>
            <div class="col">
                <strong>{{ user_info.followings.all|length }}</strong><br>
                <span>팔로잉</span>
            </div>
        </div>
    </div>
</div>

<div class="row row-cols-3 g-2">
    {% for post in user_info.post_set.all %}
    <div class="col">
        <div class="card">
            <img src="{{ post.image.url }}" alt="" class="card-img-top" style="height: 200px; object-fit: cover;">
        </div>
    </div>
    {% empty %}
    <div class="col-12 text-center">
        <p class="text-muted">아직 게시물이 없습니다.</p>
    </div>
    {% endfor %}
</div>
{% endblock %}
```

## 5. 팔로우 피드 구현

### 팔로우한 사용자의 게시물만 보기

`posts/views.py`:
```python
def index(request):
    if request.user.is_authenticated:
        # 팔로우한 사용자들의 게시물만 가져오기
        following_users = request.user.followings.all()
        posts = Post.objects.filter(user__in=following_users).order_by('-id')
    else:
        # 로그인하지 않은 경우 모든 게시물
        posts = Post.objects.all().order_by('-id')
    
    comment_form = CommentForm()
    
    context = {
        'posts': posts,
        'comment_form': comment_form,
    }
    return render(request, 'posts/index.html', context)
```

### 네비게이션에 피드 전환 버튼 추가

`_nav.html`:
```html
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
        <a class="navbar-brand" href="{% url 'posts:index' %}">Instagram</a>
        <div class="navbar-nav">
            <a class="nav-link" href="{% url 'posts:index' %}">Home</a>
            <a class="nav-link" href="{% url 'posts:all' %}">전체 게시물</a>
            <a class="nav-link" href="{% url 'posts:create' %}">Create</a>
            {% if user.is_authenticated %}
                <a class="nav-link" href="{% url 'accounts:profile' username=user.username %}">My Profile</a>
                <a class="nav-link" href="{% url 'accounts:logout' %}">Logout</a>
            {% else %}
                <a class="nav-link" href="{% url 'accounts:signup' %}">Signup</a>
                <a class="nav-link" href="{% url 'accounts:login' %}">Login</a>
            {% endif %}
        </div>
    </div>
</nav>
```

### 전체 게시물 보기 뷰

`posts/views.py`:
```python
def all_posts(request):
    posts = Post.objects.all().order_by('-id')
    comment_form = CommentForm()
    
    context = {
        'posts': posts,
        'comment_form': comment_form,
    }
    return render(request, 'posts/all.html', context)
```

`posts/urls.py`:
```python
urlpatterns = [
    path('', views.index, name='index'),
    path('all/', views.all_posts, name='all'),
    path('create/', views.create, name='create'),
    path('<int:post_id>/comment/', views.comment_create, name='comment_create'),
    path('<int:post_id>/like/', views.like, name='like'),
]
```

## 6. 실무 팁

### 1. M:N 관계 최적화

```python
# select_related와 prefetch_related 사용
def index(request):
    posts = Post.objects.select_related('user').prefetch_related(
        'like_users', 'comment_set__user'
    ).order_by('-id')
    
    context = {
        'posts': posts,
        'comment_form': CommentForm(),
    }
    return render(request, 'posts/index.html', context)
```

### 2. 좋아요 개수 캐싱

```python
# models.py
class Post(models.Model):
    # ... 기존 필드들 ...
    like_count = models.PositiveIntegerField(default=0)
    
    def update_like_count(self):
        self.like_count = self.like_users.count()
        self.save()
```

### 3. 팔로우 알림 기능

```python
# models.py
class FollowNotification(models.Model):
    from_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sent_follows')
    to_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='received_follows')
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
```

### 4. AJAX로 좋아요 기능 구현

```javascript
// 좋아요 버튼 클릭 시 AJAX 요청
function likePost(postId) {
    fetch(`/posts/${postId}/like/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
        },
    })
    .then(response => response.json())
    .then(data => {
        // 좋아요 버튼 상태 업데이트
        updateLikeButton(postId, data.liked, data.like_count);
    });
}
```

### 5. 팔로우 추천 기능

```python
# views.py
def recommend_users(request):
    if request.user.is_authenticated:
        # 팔로우하지 않은 사용자들 중에서 팔로워가 많은 순으로 추천
        following_ids = request.user.followings.values_list('id', flat=True)
        recommended_users = User.objects.exclude(
            id__in=following_ids
        ).exclude(id=request.user.id).annotate(
            follower_count=models.Count('followers')
        ).order_by('-follower_count')[:5]
        
        context = {
            'recommended_users': recommended_users,
        }
        return render(request, 'accounts/recommend.html', context)
```

이렇게 Django에서 M:N 관계를 활용하여 좋아요와 팔로우 기능을 완전히 구현할 수 있습니다!
