---
title: Django Instagram 클론 프로젝트 - 소셜미디어 플랫폼 개발
categories:
- 1.TIL
- 1-6.PROJECT
tags:
- django
- 프로젝트
- instagram
- 소셜미디어
- M:N관계
- 이미지처리
toc: true
date: 2023-08-30 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django Instagram 클론 프로젝트 - 소셜미디어 플랫폼 개발

## 프로젝트 개요

Django를 활용한 Instagram 클론 프로젝트로 소셜미디어 플랫폼의 핵심 기능을 구현합니다:

- **기술스택**: Django, django-resized, HTML/CSS/JavaScript
- **주요기능**: 게시물 CRUD, 좋아요, 댓글, 팔로우, 이미지 업로드
- **특징**: 커스텀 User 모델, 이미지 리사이징, M:N 관계 활용
- **학습목표**: Django 고급 기능과 소셜미디어 플랫폼 로직 구현

## 1. 프로젝트 구조

```
Instagram/
├── accounts/          # 사용자 관리 앱
│   ├── models.py      # 커스텀 User 모델
│   ├── views.py       # 회원가입, 로그인, 프로필
│   ├── forms.py       # 사용자 폼
│   ├── urls.py        # 사용자 관련 URL
│   └── templates/     # 사용자 템플릿
├── posts/             # 게시물 관리 앱
│   ├── models.py      # Post, Comment 모델
│   ├── views.py       # 게시물 CRUD, 좋아요
│   ├── forms.py       # 게시물 폼
│   ├── urls.py        # 게시물 관련 URL
│   └── templates/     # 게시물 템플릿
├── insta/             # 프로젝트 설정
│   ├── settings.py    # 프로젝트 설정
│   ├── urls.py        # 메인 URL
│   └── wsgi.py        # WSGI 설정
├── templates/         # 공통 템플릿
│   ├── base.html      # 기본 레이아웃
│   ├── _nav.html      # 네비게이션
│   └── _card.html     # 게시물 카드
└── static/            # 정적 파일
    ├── css/
    ├── js/
    └── images/
```

## 2. 데이터 모델 설계

### 커스텀 User 모델

```python
# accounts/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django_resized import ResizedImageField

class User(AbstractUser):
    """커스텀 User 모델"""
    
    profile_image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='profile/',
        blank=True,
        null=True,
        help_text="프로필 이미지 (500x500)"
    )
    
    bio = models.TextField(
        max_length=500,
        blank=True,
        help_text="자기소개"
    )
    
    website = models.URLField(
        blank=True,
        help_text="웹사이트 URL"
    )
    
    followings = models.ManyToManyField(
        'self',
        related_name='followers',
        symmetrical=False,
        blank=True,
        help_text="팔로우하는 사용자들"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "사용자"
        verbose_name_plural = "사용자들"
    
    def __str__(self):
        return self.username
    
    def get_followers_count(self):
        """팔로워 수 반환"""
        return self.followers.count()
    
    def get_followings_count(self):
        """팔로잉 수 반환"""
        return self.followings.count()
    
    def get_posts_count(self):
        """게시물 수 반환"""
        return self.post_set.count()
    
    def is_following(self, user):
        """특정 사용자를 팔로우하는지 확인"""
        return self.followings.filter(id=user.id).exists()
```

**핵심 특징:**
- `AbstractUser` 상속으로 기본 User 기능 확장
- `ResizedImageField`로 이미지 자동 리사이징
- `symmetrical=False`로 비대칭 팔로우 관계 구현
- 모델 메서드로 비즈니스 로직 캡슐화

### Post 모델

```python
# posts/models.py
from django.db import models
from django.conf import settings
from django_resized import ResizedImageField
from django.urls import reverse

class Post(models.Model):
    """게시물 모델"""
    
    content = models.TextField(
        help_text="게시물 내용"
    )
    
    image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='posts/%Y/%m/',
        help_text="게시물 이미지 (500x500)"
    )
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='posts',
        help_text="작성자"
    )
    
    like_users = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='like_posts',
        blank=True,
        help_text="좋아요를 누른 사용자들"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "게시물"
        verbose_name_plural = "게시물들"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username}의 게시물"
    
    def get_absolute_url(self):
        """게시물 상세 페이지 URL 반환"""
        return reverse('posts:detail', kwargs={'post_id': self.id})
    
    def get_likes_count(self):
        """좋아요 수 반환"""
        return self.like_users.count()
    
    def get_comments_count(self):
        """댓글 수 반환"""
        return self.comments.count()
    
    def is_liked_by(self, user):
        """특정 사용자가 좋아요를 눌렀는지 확인"""
        return self.like_users.filter(id=user.id).exists()
```

**핵심 특징:**
- 이미지 자동 리사이징 (500x500)
- `auto_now_add=True`: 생성 시 자동 시간 기록
- `auto_now=True`: 수정 시 자동 시간 업데이트
- M:N 관계로 좋아요 기능 구현
- 날짜별 폴더 구조로 파일 관리

### Comment 모델

```python
class Comment(models.Model):
    """댓글 모델"""
    
    content = models.CharField(
        max_length=200,
        help_text="댓글 내용"
    )
    
    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name='comments',
        help_text="게시물"
    )
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='comments',
        help_text="댓글 작성자"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "댓글"
        verbose_name_plural = "댓글들"
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.user.username}의 댓글: {self.content[:20]}"
```

## 3. 설정 파일

### settings.py 설정

```python
# insta/settings.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 커스텀 User 모델 설정
AUTH_USER_MODEL = 'accounts.User'

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'accounts',
    'posts',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'insta.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
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

# 미디어 파일 설정
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# 정적 파일 설정
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# 로그인/로그아웃 URL
LOGIN_URL = 'accounts:login'
LOGIN_REDIRECT_URL = 'posts:index'
LOGOUT_REDIRECT_URL = 'accounts:login'
```

## 4. 폼 설계

### Post 폼

```python
# posts/forms.py
from django import forms
from .models import Post, Comment

class PostForm(forms.ModelForm):
    """게시물 생성/수정 폼"""
    
    class Meta:
        model = Post
        fields = ['content', 'image']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': '무슨 일이 일어나고 있나요?'
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
        }
        labels = {
            'content': '내용',
            'image': '이미지',
        }

class CommentForm(forms.ModelForm):
    """댓글 작성 폼"""
    
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '댓글을 입력하세요...',
                'maxlength': 200
            }),
        }
        labels = {
            'content': '댓글',
        }
```

### User 폼

```python
# accounts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User

class CustomUserCreationForm(UserCreationForm):
    """커스텀 회원가입 폼"""
    
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': '이메일'
        })
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '사용자명'
            }),
        }

class UserUpdateForm(forms.ModelForm):
    """사용자 정보 수정 폼"""
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'bio', 'website', 'profile_image']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'bio': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '자기소개를 입력하세요...'
            }),
            'website': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://example.com'
            }),
            'profile_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
        }
```

## 5. 뷰 구현

### Posts 뷰

```python
# posts/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from .models import Post, Comment
from .forms import PostForm, CommentForm

@login_required
def index(request):
    """게시물 목록 페이지"""
    # 팔로우하는 사용자들의 게시물만 표시
    following_users = request.user.followings.all()
    posts = Post.objects.filter(
        user__in=following_users
    ).select_related('user').prefetch_related('like_users', 'comments').order_by('-created_at')
    
    # 페이지네이션
    paginator = Paginator(posts, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'posts': page_obj,
    }
    return render(request, 'posts/index.html', context)

@login_required
def create(request):
    """게시물 생성"""
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user
            post.save()
            messages.success(request, '게시물이 성공적으로 작성되었습니다.')
            return redirect('posts:index')
    else:
        form = PostForm()
    
    context = {'form': form}
    return render(request, 'posts/form.html', context)

@login_required
def detail(request, post_id):
    """게시물 상세 페이지"""
    post = get_object_or_404(Post, id=post_id)
    comments = post.comments.select_related('user').order_by('created_at')
    
    if request.method == 'POST':
        comment_form = CommentForm(request.POST)
        if comment_form.is_valid():
            comment = comment_form.save(commit=False)
            comment.post = post
            comment.user = request.user
            comment.save()
            return redirect('posts:detail', post_id=post.id)
    else:
        comment_form = CommentForm()
    
    context = {
        'post': post,
        'comments': comments,
        'comment_form': comment_form,
    }
    return render(request, 'posts/detail.html', context)

@login_required
def like(request, post_id):
    """좋아요 토글"""
    post = get_object_or_404(Post, id=post_id)
    
    if request.user in post.like_users.all():
        post.like_users.remove(request.user)
        is_liked = False
    else:
        post.like_users.add(request.user)
        is_liked = True
    
    # AJAX 요청인 경우 JSON 응답
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'is_liked': is_liked,
            'likes_count': post.get_likes_count()
        })
    
    return redirect('posts:index')

@login_required
def delete(request, post_id):
    """게시물 삭제"""
    post = get_object_or_404(Post, id=post_id)
    
    # 작성자만 삭제 가능
    if post.user == request.user:
        post.delete()
        messages.success(request, '게시물이 삭제되었습니다.')
    else:
        messages.error(request, '삭제 권한이 없습니다.')
    
    return redirect('posts:index')
```

### Accounts 뷰

```python
# accounts/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from .models import User
from .forms import CustomUserCreationForm, UserUpdateForm

def signup(request):
    """회원가입"""
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, '회원가입이 완료되었습니다.')
            return redirect('posts:index')
    else:
        form = CustomUserCreationForm()
    
    context = {'form': form}
    return render(request, 'accounts/signup.html', context)

@login_required
def profile(request, user_id):
    """사용자 프로필 페이지"""
    user = get_object_or_404(User, id=user_id)
    posts = user.posts.all().order_by('-created_at')
    
    # 페이지네이션
    paginator = Paginator(posts, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # 팔로우 상태 확인
    is_following = request.user.is_following(user) if request.user.is_authenticated else False
    
    context = {
        'profile_user': user,
        'page_obj': page_obj,
        'posts': page_obj,
        'is_following': is_following,
    }
    return render(request, 'accounts/profile.html', context)

@login_required
def follow(request, user_id):
    """팔로우/언팔로우"""
    user = get_object_or_404(User, id=user_id)
    
    if request.user == user:
        messages.error(request, '자기 자신을 팔로우할 수 없습니다.')
        return redirect('accounts:profile', user_id=user.id)
    
    if request.user in user.followers.all():
        user.followers.remove(request.user)
        messages.success(request, f'{user.username}을(를) 언팔로우했습니다.')
    else:
        user.followers.add(request.user)
        messages.success(request, f'{user.username}을(를) 팔로우했습니다.')
    
    return redirect('accounts:profile', user_id=user.id)

@login_required
def edit_profile(request):
    """프로필 수정"""
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, '프로필이 수정되었습니다.')
            return redirect('accounts:profile', user_id=request.user.id)
    else:
        form = UserUpdateForm(instance=request.user)
    
    context = {'form': form}
    return render(request, 'accounts/edit_profile.html', context)
```

## 6. URL 라우팅

### Posts URL

```python
# posts/urls.py
from django.urls import path
from . import views

app_name = 'posts'

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create, name='create'),
    path('<int:post_id>/', views.detail, name='detail'),
    path('<int:post_id>/like/', views.like, name='like'),
    path('<int:post_id>/delete/', views.delete, name='delete'),
]
```

### Accounts URL

```python
# accounts/urls.py
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'accounts'

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('profile/<int:user_id>/', views.profile, name='profile'),
    path('profile/<int:user_id>/follow/', views.follow, name='follow'),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
]
```

### 메인 URL

```python
# insta/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('posts.urls')),
    path('accounts/', include('accounts.urls')),
]

# 개발 환경에서 미디어 파일 서빙
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

## 7. 템플릿 구현

### 기본 템플릿

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>&#123;% block title %&#125;Instagram Clone&#123;% endblock %&#125;</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="&#123;% static 'css/style.css' %&#125;">
</head>
<body>
    &#123;% include '_nav.html' %&#125;
    
    <main class="container mt-4">
        &#123;% if messages %&#125;
            &#123;% for message in messages %&#125;
                <div class="alert alert-&#123;% raw %&#125;&#123;&#123; message.tags &#125;&#125;&#123;% endraw %&#125; alert-dismissible fade show" role="alert">
                    &#123;% raw %&#125;&#123;&#123; message &#125;&#125;&#123;% endraw %&#125;
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            &#123;% endfor %&#125;
        &#123;% endif %&#125;
        
        &#123;% block content %&#125;
        &#123;% endblock %&#125;
    </main>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="&#123;% static 'js/main.js' %&#125;"></script>
</body>
</html>
```

### 네비게이션

```html
<!-- templates/_nav.html -->
<nav class="navbar navbar-expand-lg navbar-light bg-white border-bottom">
    <div class="container">
        <a class="navbar-brand fw-bold" href="&#123;% url 'posts:index' %&#125;">
            <i class="fab fa-instagram"></i> Instagram
        </a>
        
        <div class="navbar-nav ms-auto">
            &#123;% if user.is_authenticated %&#125;
                <a class="nav-link" href="&#123;% url 'posts:create' %&#125;">
                    <i class="fas fa-plus"></i> 새 게시물
                </a>
                <a class="nav-link" href="&#123;% url 'accounts:profile' user.id %&#125;">
                    <i class="fas fa-user"></i> 프로필
                </a>
                <a class="nav-link" href="&#123;% url 'accounts:logout' %&#125;">
                    <i class="fas fa-sign-out-alt"></i> 로그아웃
                </a>
            &#123;% else %&#125;
                <a class="nav-link" href="&#123;% url 'accounts:login' %&#125;">로그인</a>
                <a class="nav-link" href="&#123;% url 'accounts:signup' %&#125;">회원가입</a>
            &#123;% endif %&#125;
        </div>
    </div>
</nav>
```

### 게시물 카드 컴포넌트

```html
<!-- templates/_card.html -->
<div class="card mb-4">
    <div class="card-header d-flex align-items-center">
        <img src="&#123;% raw %&#125;&#123;&#123; post.user.profile_image.url|default:'/static/images/default-profile.png' &#125;&#125;&#123;% endraw %&#125;" 
             class="rounded-circle me-2" width="32" height="32" alt="프로필">
        <div>
            <h6 class="mb-0">
                <a href="&#123;% url 'accounts:profile' post.user.id %&#125;" class="text-decoration-none">
                    &#123;% raw %&#125;&#123;&#123; post.user.username &#125;&#125;&#123;% endraw %&#125;
                </a>
            </h6>
            <small class="text-muted">&#123;% raw %&#125;&#123;&#123; post.created_at|timesince &#125;&#125;&#123;% endraw %&#125; 전</small>
        </div>
    </div>
    
    <div class="card-body p-0">
        <img src="&#123;% raw %&#125;&#123;&#123; post.image.url &#125;&#125;&#123;% endraw %&#125;" class="card-img-top" alt="게시물 이미지">
        
        <div class="p-3">
            <div class="d-flex align-items-center mb-2">
                <button class="btn btn-link p-0 me-2 like-btn" data-post-id="&#123;% raw %&#125;&#123;&#123; post.id &#125;&#125;&#123;% endraw %&#125;">
                    <i class="fas fa-heart &#123;% if post.is_liked_by:user %&#125;text-danger&#123;% else %&#125;text-muted&#123;% endif %&#125;"></i>
                </button>
                <button class="btn btn-link p-0 me-2" data-bs-toggle="collapse" data-bs-target="#comments-&#123;% raw %&#125;&#123;&#123; post.id &#125;&#125;&#123;% endraw %&#125;">
                    <i class="far fa-comment text-muted"></i>
                </button>
                <span class="text-muted">&#123;% raw %&#125;&#123;&#123; post.get_likes_count &#125;&#125;&#123;% endraw %&#125;명이 좋아합니다</span>
            </div>
            
            <div class="mb-2">
                <strong>&#123;% raw %&#125;&#123;&#123; post.user.username &#125;&#125;&#123;% endraw %&#125;</strong> &#123;% raw %&#125;&#123;&#123; post.content &#125;&#125;&#123;% endraw %&#125;
            </div>
            
            <div class="collapse" id="comments-&#123;% raw %&#125;&#123;&#123; post.id &#125;&#125;&#123;% endraw %&#125;">
                <div class="comments-section">
                    &#123;% for comment in post.comments.all %&#125;
                    <div class="mb-1">
                        <strong>&#123;% raw %&#125;&#123;&#123; comment.user.username &#125;&#125;&#123;% endraw %&#125;</strong> &#123;% raw %&#125;&#123;&#123; comment.content &#125;&#125;&#123;% endraw %&#125;
                    </div>
                    &#123;% endfor %&#125;
                </div>
                
                <form class="comment-form mt-2" data-post-id="&#123;% raw %&#125;&#123;&#123; post.id &#125;&#125;&#123;% endraw %&#125;">
                    &#123;% csrf_token %&#125;
                    <div class="input-group">
                        <input type="text" class="form-control" placeholder="댓글 달기..." maxlength="200">
                        <button class="btn btn-outline-primary" type="submit">게시</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
```

### 게시물 목록 페이지

```html
<!-- posts/templates/posts/index.html -->
&#123;% extends 'base.html' %&#125;
&#123;% load static %&#125;

&#123;% block title %&#125;홈 - Instagram Clone&#123;% endblock %&#125;

&#123;% block content %&#125;
<div class="row justify-content-center">
    <div class="col-md-6">
        &#123;% if posts %&#125;
            &#123;% for post in posts %&#125;
                &#123;% include '_card.html' %&#125;
            &#123;% endfor %&#125;
            
            <!-- 페이지네이션 -->
            &#123;% if page_obj.has_other_pages %&#125;
            <nav aria-label="Page navigation">
                <ul class="pagination justify-content-center">
                    &#123;% if page_obj.has_previous %&#125;
                        <li class="page-item">
                            <a class="page-link" href="?page=1">처음</a>
                        </li>
                        <li class="page-item">
                            <a class="page-link" href="?page=&#123;% raw %&#125;&#123;&#123; page_obj.previous_page_number &#125;&#125;&#123;% endraw %&#125;">이전</a>
                        </li>
                    &#123;% endif %&#125;
                    
                    <li class="page-item active">
                        <span class="page-link">&#123;% raw %&#125;&#123;&#123; page_obj.number &#125;&#125;&#123;% endraw %&#125; / &#123;% raw %&#125;&#123;&#123; page_obj.paginator.num_pages &#125;&#125;&#123;% endraw %&#125;</span>
                    </li>
                    
                    &#123;% if page_obj.has_next %&#125;
                        <li class="page-item">
                            <a class="page-link" href="?page=&#123;% raw %&#125;&#123;&#123; page_obj.next_page_number &#125;&#125;&#123;% endraw %&#125;">다음</a>
                        </li>
                        <li class="page-item">
                            <a class="page-link" href="?page=&#123;% raw %&#125;&#123;&#123; page_obj.paginator.num_pages &#125;&#125;&#123;% endraw %&#125;">마지막</a>
                        </li>
                    &#123;% endif %&#125;
                </ul>
            </nav>
            &#123;% endif %&#125;
        &#123;% else %&#125;
            <div class="text-center py-5">
                <h4>아직 팔로우하는 사용자가 없습니다</h4>
                <p class="text-muted">다른 사용자를 팔로우하여 게시물을 확인해보세요!</p>
                <a href="&#123;% url 'posts:create' %&#125;" class="btn btn-primary">첫 게시물 작성하기</a>
            </div>
        &#123;% endif %&#125;
    </div>
</div>
&#123;% endblock %&#125;
```

## 8. JavaScript 기능

```javascript
// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // 좋아요 버튼 이벤트
    document.querySelectorAll('.like-btn').forEach(button => {
        button.addEventListener('click', function() {
            const postId = this.dataset.postId;
            const heartIcon = this.querySelector('i');
            
            fetch(`/posts/${postId}/like/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                    'X-Requested-With': 'XMLHttpRequest',
                },
            })
            .then(response => response.json())
            .then(data => {
                if (data.is_liked) {
                    heartIcon.classList.remove('text-muted');
                    heartIcon.classList.add('text-danger');
                } else {
                    heartIcon.classList.remove('text-danger');
                    heartIcon.classList.add('text-muted');
                }
                
                // 좋아요 수 업데이트
                const likesCount = this.parentElement.querySelector('.text-muted');
                likesCount.textContent = `${data.likes_count}명이 좋아합니다`;
            })
            .catch(error => console.error('Error:', error));
        });
    });
    
    // 댓글 폼 이벤트
    document.querySelectorAll('.comment-form').forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const postId = this.dataset.postId;
            const content = this.querySelector('input[type="text"]').value;
            
            if (content.trim() === '') return;
            
            fetch(`/posts/${postId}/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                },
                body: new FormData(this)
            })
            .then(response => {
                if (response.ok) {
                    location.reload();
                }
            })
            .catch(error => console.error('Error:', error));
        });
    });
});
```

## 9. 스타일링

```css
/* static/css/style.css */
.card {
    border: 1px solid #dbdbdb;
    border-radius: 8px;
    box-shadow: none;
}

.card-header {
    background-color: white;
    border-bottom: 1px solid #dbdbdb;
    padding: 14px 16px;
}

.card-img-top {
    width: 100%;
    height: auto;
    object-fit: cover;
}

.like-btn {
    border: none;
    background: none;
    font-size: 1.5rem;
    transition: transform 0.2s;
}

.like-btn:hover {
    transform: scale(1.1);
}

.comment-form .input-group {
    border-top: 1px solid #dbdbdb;
    padding-top: 12px;
}

.comment-form .form-control {
    border: none;
    box-shadow: none;
    padding: 8px 0;
}

.comment-form .form-control:focus {
    border: none;
    box-shadow: none;
}

.comment-form .btn {
    border: none;
    color: #0095f6;
    font-weight: 600;
}

.navbar-brand {
    font-size: 1.5rem;
    color: #262626 !important;
}

.nav-link {
    color: #262626 !important;
    font-weight: 500;
}

.nav-link:hover {
    color: #8e8e8e !important;
}

.profile-image {
    width: 150px;
    height: 150px;
    object-fit: cover;
}

.stats-card {
    text-align: center;
    padding: 20px;
}

.stats-number {
    font-size: 1.5rem;
    font-weight: 600;
    color: #262626;
}

.stats-label {
    color: #8e8e8e;
    font-size: 0.9rem;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .container {
        padding: 0 10px;
    }
    
    .card {
        border-left: none;
        border-right: none;
        border-radius: 0;
    }
    
    .profile-image {
        width: 100px;
        height: 100px;
    }
}
```

## 10. 주요 학습 포인트

### 1. 커스텀 User 모델
- `AbstractUser` 상속으로 기본 기능 확장
- 프로필 이미지 필드 추가
- 팔로우 관계를 M:N으로 구현
- 모델 메서드로 비즈니스 로직 캡슐화

### 2. 이미지 처리
- `django-resized` 라이브러리 활용
- 자동 리사이징 및 크롭 기능
- 업로드 경로 동적 설정 (`%Y/%m`)
- 미디어 파일 서빙 설정

### 3. M:N 관계 활용
- **좋아요**: Post ↔ User (ManyToManyField)
- **팔로우**: User ↔ User (self-referential ManyToManyField)
- `related_name`으로 역참조 이름 지정
- `symmetrical=False`로 비대칭 관계 구현

### 4. 폼 처리
- `request.FILES`로 이미지 파일 처리
- `commit=False`로 추가 필드 설정 후 저장
- 폼 유효성 검사 및 에러 처리

### 5. 템플릿 시스템
- 템플릿 상속을 통한 코드 재사용
- `_card.html` 컴포넌트로 코드 중복 제거
- `_nav.html`로 네비게이션 일관성 유지

### 6. JavaScript 활용
- AJAX를 통한 비동기 처리
- 좋아요 버튼 토글
- 댓글 실시간 추가
- CSRF 토큰 처리

### 7. 성능 최적화
- `select_related`로 N+1 문제 해결
- `prefetch_related`로 M:N 관계 최적화
- 페이지네이션으로 대용량 데이터 처리

## 11. 프로젝트 확장 아이디어

### 기능 확장
```python
# 해시태그 시스템
class Hashtag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    posts = models.ManyToManyField(Post, related_name='hashtags')

# 스토리 기능
class Story(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = ResizedImageField(size=[400, 400], upload_to='stories/')
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

# 알림 시스템
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.CharField(max_length=200)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
```

### API 개발
```python
# REST API 추가
from rest_framework import serializers, viewsets

class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
```

## 마무리

Django Instagram 클론 프로젝트를 통해 다음과 같은 실무 경험을 쌓을 수 있습니다:

### 핵심 학습 내용
- **커스텀 User 모델**: 사용자 기능 확장과 프로필 관리
- **M:N 관계**: 복잡한 데이터 관계 설계와 구현
- **이미지 처리**: 파일 업로드와 자동 리사이징
- **소셜 기능**: 팔로우, 좋아요, 댓글 시스템
- **AJAX**: 비동기 처리와 사용자 경험 향상

### 실무 적용 가능성
- **소셜미디어 플랫폼**: Instagram, Facebook 등과 유사한 기능
- **커뮤니티 사이트**: 사용자 간 상호작용이 중요한 플랫폼
- **이미지 공유 서비스**: 사진 중심의 웹 애플리케이션
- **확장성**: 모듈화된 구조로 기능 확장 용이

이 프로젝트는 Django의 고급 기능들과 소셜미디어 플랫폼의 핵심 로직을 실무 수준으로 구현하는 완성도 높은 프로젝트입니다.
