---
title: Django 이미지 업로드 - 사진 업로드와 프로필 관리
categories:
- 1.TIL
- 1-5.WEB
tags:
- django
- 이미지업로드
- pillow
- media
- 프로필
- 컴포넌트화
toc: true
date: 2023-08-30 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django 이미지 업로드 - 사진 업로드와 프로필 관리

## 개요

Django에서 이미지 업로드 기능을 구현합니다:

- **사진 업로드**
  - admin을 통해 사진 등록
  - create를 통해 사진 등록
  - signup으로 프로필사진 등록
- **컴포넌트화(_*.html)**
- **여러가지 기능 추가** (timesince, 정렬, 사진규격통일)
- **My profile 만들기** (팔로우, 팔로잉 등)

## 1. 사진 업로드를 위한 설정

### Pillow 라이브러리 설치

```bash
pip install Pillow
```

### 모델 설정

`posts/models.py`:
```python
from django.db import models
from django.conf import settings
from django_resized import ResizedImageField  # 사진 규격 통일을 위해

class Post(models.Model):
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    
    # 기본 이미지 필드
    # image = models.ImageField(upload_to='image/')
    
    # 날짜별 폴더 구조로 저장
    # image = models.ImageField(upload_to='image/%Y/%m')
    
    # 사진 규격 통일 (django-resized 사용)
    image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='image/%Y/%m'
    )
```

### settings.py 설정

```python
# 업로드한 사진을 저장할 위치
MEDIA_ROOT = BASE_DIR / 'media'  # BASE_DIR: 프로젝트 위치

# 미디어 경로를 처리할 URL
MEDIA_URL = '/media/'
```

### django-resized 설치 (선택사항)

```bash
pip install django-resized
```

`settings.py`에 추가:
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_resized',  # 추가
    'posts',
    'accounts',
]
```

## 2. admin 페이지를 통한 사진 업로드

### admin.py 설정

`posts/admin.py`:
```python
from django.contrib import admin
from .models import Post

admin.site.register(Post)
```

### 관리자 계정 생성 및 테스트

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

1. `/admin/` 접속
2. Posts에서 사진 선택하여 업로드
3. 프로젝트 최상위 폴더에서 `media/image/년/월` 폴더에 사진 업로드 확인

## 3. Read 기능 구현

### URL 설정

`project/urls.py`:
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('posts/', include('posts.urls')),
    path('accounts/', include('accounts.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

`posts/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'posts'

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create, name='create'),
]
```

### 컴포넌트화 - _nav.html

`templates/_nav.html`:
```html
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container">
        <a class="navbar-brand" href="&#123;% url 'posts:index' %&#125;">Instagram</a>
        <div class="navbar-nav">
            <a class="nav-link" href="&#123;% url 'posts:index' %&#125;">Home</a>
            <a class="nav-link" href="&#123;% url 'posts:create' %&#125;">Create</a>
            <a class="nav-link" href="&#123;% url 'accounts:signup' %&#125;">Signup</a>
            <a class="nav-link" href="&#123;% url 'accounts:login' %&#125;">Login</a>
        </div>
    </div>
</nav>
```

### base.html 업데이트

`templates/base.html`:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>&#123;% block title %&#125;Instagram&#123;% endblock %&#125;</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    &#123;% include '_nav.html' %&#125;
    
    <div class="container mt-4">
        &#123;% block body %&#125;
        &#123;% endblock %&#125;
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### 컴포넌트화 - _card.html

`posts/templates/_card.html`:
```html
<div class="card mb-3" style="width: 18rem;">
    <img src="&#123;% raw %&#125;&#123;&#123; post.image.url &#125;&#125;&#123;% endraw %&#125;" class="card-img-top" alt="...">
    <div class="card-body">
        <p class="card-text">&#123;% raw %&#125;&#123;&#123; post.content &#125;&#125;&#123;% endraw %&#125;</p>
        <small class="text-muted">&#123;% raw %&#125;&#123;&#123; post.created_at|timesince &#125;&#125;&#123;% endraw %&#125; 전</small>
        <br>
        <a href="&#123;% url 'accounts:profile' username=post.user %&#125;" class="text-reset text-decoration-none">
            &#123;% raw %&#125;&#123;&#123; post.user &#125;&#125;&#123;% endraw %&#125;
        </a>
    </div>
</div>
```

### index.html

`posts/templates/posts/index.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;Instagram&#123;% endblock %&#125;

&#123;% block body %&#125;
<div class="row">
    &#123;% for post in posts %&#125;
        <div class="col-md-4 mb-3">
            &#123;% include '_card.html' %&#125;
        </div>
    &#123;% endfor %&#125;
</div>
&#123;% endblock %&#125;
```

### views.py

`posts/views.py`:
```python
from django.shortcuts import render, redirect
from .models import Post

def index(request):
    posts = Post.objects.all().order_by('-id')  # 최신순 정렬
    context = {
        'posts': posts,
    }
    return render(request, 'posts/index.html', context)
```

## 4. Create 기능 구현

### PostForm 생성

`posts/forms.py`:
```python
from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = '__all__'
```

### Create 뷰

`posts/views.py`:
```python
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .models import Post
from .forms import PostForm

@login_required
def create(request):
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user
            post.save()
            return redirect('posts:index')
    else:
        form = PostForm()

    context = {
        'form': form
    }
    return render(request, 'posts/form.html', context)
```

### form.html

`posts/templates/posts/form.html`:
```html
&#123;% extends 'base.html' %&#125;
&#123;% load bootstrap5 %&#125;

&#123;% block title %&#125;새 게시물 작성&#123;% endblock %&#125;

&#123;% block body %&#125;
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h3>새 게시물 작성</h3>
            </div>
            <div class="card-body">
                <form action="" method="POST" enctype="multipart/form-data">
                    &#123;% csrf_token %&#125;
                    &#123;% bootstrap_form form %&#125;
                    <button type="submit" class="btn btn-primary">게시</button>
                    <a href="&#123;% url 'posts:index' %&#125;" class="btn btn-secondary">취소</a>
                </form>
            </div>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

## 5. 여러가지 기능 추가

### timesince 필터

`_card.html`에서 사용:
```html
<small class="text-muted">&#123;% raw %&#125;&#123;&#123; post.created_at|timesince &#125;&#125;&#123;% endraw %&#125; 전</small>
```

### 정렬 기능

`views.py`:
```python
def index(request):
    # 최신순 정렬
    posts = Post.objects.all().order_by('-id')
    
    # 또는 특정 조건별 정렬
    # posts = Post.objects.all().order_by('-created_at')
    
    context = {
        'posts': posts,
    }
    return render(request, 'posts/index.html', context)
```

### 사진 규격 통일

`models.py`에서 ResizedImageField 사용:
```python
from django_resized import ResizedImageField

class Post(models.Model):
    image = ResizedImageField(
        size=[500, 500],  # 500x500 픽셀로 리사이즈
        crop=['middle', 'center'],  # 중앙에서 크롭
        upload_to='image/%Y/%m'
    )
```

## 6. Signup 기능 구현 (프로필 사진 포함)

### accounts 앱 생성

```bash
python manage.py startapp accounts
```

### User 모델 확장

`accounts/models.py`:
```python
from django.contrib.auth.models import AbstractUser
from django_resized import ResizedImageField

class User(AbstractUser):
    # 프로필 이미지 등록
    profile_image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='profile',
        blank=True,
        null=True
    )
    # 관계 설정을 함으로써 post_set = 자동으로 생성
```

### settings.py 설정

```python
AUTH_USER_MODEL = 'accounts.User'
```

### 1:N 관계 설정

`posts/models.py`:
```python
from django.db import models
from django.conf import settings

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
```

### CustomUserCreationForm

`accounts/forms.py`:
```python
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = get_user_model()  # 유지보수를 위해서 좋은 방법
        fields = ('username', 'profile_image')
```

### Signup 뷰

`accounts/views.py`:
```python
from django.shortcuts import render, redirect
from .forms import CustomUserCreationForm

def signup(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('posts:index')
    else:
        form = CustomUserCreationForm()

    context = {
        'form': form,
    }
    return render(request, 'accounts/form.html', context)
```

### accounts/form.html

`accounts/templates/accounts/form.html`:
```html
&#123;% extends 'base.html' %&#125;
&#123;% load bootstrap5 %&#125;

&#123;% block title %&#125;
    &#123;% if request.resolver_match.url_name == 'signup' %&#125;
        회원가입
    &#123;% else %&#125;
        로그인
    &#123;% endif %&#125;
&#123;% endblock %&#125;

&#123;% block body %&#125;
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                &#123;% if request.resolver_match.url_name == 'signup' %&#125;
                    <h3>회원가입</h3>
                &#123;% else %&#125;
                    <h3>로그인</h3>
                &#123;% endif %&#125;
            </div>
            <div class="card-body">
                <form action="" method="POST" enctype="multipart/form-data">
                    &#123;% csrf_token %&#125;
                    &#123;% bootstrap_form form %&#125;
                    <button type="submit" class="btn btn-primary">
                        &#123;% if request.resolver_match.url_name == 'signup' %&#125;
                            회원가입
                        &#123;% else %&#125;
                            로그인
                        &#123;% endif %&#125;
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

## 7. Login 기능 구현

### CustomAuthenticationForm

`accounts/forms.py`:
```python
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = ('username', 'profile_image')

class CustomAuthenticationForm(AuthenticationForm):
    pass
```

### Login 뷰

`accounts/views.py`:
```python
from django.contrib.auth import login as auth_login
from django.shortcuts import render, redirect
from .forms import CustomUserCreationForm, CustomAuthenticationForm

def login(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('posts:index')
    else:
        form = CustomAuthenticationForm()
    
    context = {
        'form': form,
    }
    return render(request, 'accounts/form.html', context)
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
]
```

## 8. My Profile 만들기

### Profile 뷰

`accounts/views.py`:
```python
from django.contrib.auth import get_user_model

def profile(request, username):
    User = get_user_model()
    user_info = User.objects.get(username=username)
    context = {
        'user_info': user_info
    }
    return render(request, 'accounts/profile.html', context)
```

### profile.html

`accounts/templates/accounts/profile.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;&#123;% raw %&#125;&#123;&#123; user_info.username &#125;&#125;&#123;% endraw %&#125;의 프로필&#123;% endblock %&#125;

&#123;% block body %&#125;
<div class="row mb-4">
    <div class="col-4">
        &#123;% if user_info.profile_image %&#125;
            <img src="&#123;% raw %&#125;&#123;&#123; user_info.profile_image.url &#125;&#125;&#123;% endraw %&#125;" alt="" class="img-fluid rounded-circle" style="width: 150px; height: 150px; object-fit: cover;">
        &#123;% else %&#125;
            <div class="bg-secondary rounded-circle d-flex align-items-center justify-content-center" style="width: 150px; height: 150px;">
                <span class="text-white">&#123;% raw %&#125;&#123;&#123; user_info.username|first|upper &#125;&#125;&#123;% endraw %&#125;</span>
            </div>
        &#123;% endif %&#125;
    </div>

    <div class="col-8">
        <div class="row mb-3">
            <div class="col-3">
                <h4>&#123;% raw %&#125;&#123;&#123; user_info.username &#125;&#125;&#123;% endraw %&#125;</h4>
            </div>
            <div class="col-4">
                <a href="" class="btn btn-outline-primary">팔로우</a>
            </div>
        </div>
        <div class="row">
            <div class="col">
                <strong>&#123;% raw %&#125;&#123;&#123; user_info.post_set.count &#125;&#125;&#123;% endraw %&#125;</strong><br>
                <span>게시물</span>
            </div>
            <div class="col">
                <strong>0</strong><br>
                <span>팔로잉</span>
            </div>
            <div class="col">
                <strong>0</strong><br>
                <span>팔로워</span>
            </div>
        </div>
    </div>
</div>

<div class="row row-cols-3 g-2">
    &#123;% for post in user_info.post_set.all %&#125;
    <div class="col">
        <div class="card">
            <img src="&#123;% raw %&#125;&#123;&#123; post.image.url &#125;&#125;&#123;% endraw %&#125;" alt="" class="card-img-top" style="height: 200px; object-fit: cover;">
        </div>
    </div>
    &#123;% empty %&#125;
    <div class="col-12 text-center">
        <p class="text-muted">아직 게시물이 없습니다.</p>
    </div>
    &#123;% endfor %&#125;
</div>
&#123;% endblock %&#125;
```

## 9. 실무 팁

### 1. 이미지 최적화

```python
# models.py
from django_resized import ResizedImageField

class Post(models.Model):
    image = ResizedImageField(
        size=[500, 500],
        crop=['middle', 'center'],
        upload_to='image/%Y/%m',
        quality=85,  # 이미지 품질 (85%)
        force_format='JPEG'  # 포맷 강제 지정
    )
```

### 2. 이미지 검증

```python
# forms.py
from django import forms
from django.core.exceptions import ValidationError

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = '__all__'
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            if image.size > 5 * 1024 * 1024:  # 5MB 제한
                raise ValidationError('이미지 크기는 5MB 이하여야 합니다.')
        return image
```

### 3. 이미지 삭제 시 파일도 삭제

```python
# models.py
import os
from django.db.models.signals import post_delete
from django.dispatch import receiver

@receiver(post_delete, sender=Post)
def delete_image_file(sender, instance, **kwargs):
    if instance.image:
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)
```

### 4. 이미지 썸네일 생성

```python
# models.py
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile

class Post(models.Model):
    # ... 기존 필드들 ...
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        
        if self.image:
            # 썸네일 생성
            img = Image.open(self.image.path)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            # 썸네일 저장
            thumb_io = BytesIO()
            img.save(thumb_io, format='JPEG', quality=85)
            thumb_io.seek(0)
            
            # 썸네일 파일명 생성
            thumb_name = f"thumb_{self.image.name}"
            self.thumbnail.save(thumb_name, ContentFile(thumb_io.getvalue()), save=False)
```

이렇게 Django에서 이미지 업로드 기능을 완전히 구현할 수 있습니다!
