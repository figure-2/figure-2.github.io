---
title: Django 사용자 인증 - Signup, Login, Logout
categories:
- 1.TIL
- 1-5.WEB
tags:
- django
- 사용자인증
- signup
- login
- logout
- abstractuser
toc: true
date: 2023-08-28 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django 사용자 인증 - Signup, Login, Logout

## 개요

Django의 내장 사용자 인증 시스템을 활용하여 다음 기능들을 구현합니다:

- **User Model 추가**
- **Signup, Login, Logout 기능 추가**

## 1. User Model 설정

### AbstractUser 상속

Django의 기본 User 모델을 확장하여 커스텀 사용자 모델을 만듭니다.

`accounts/models.py`:
```python
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    pass
```

### admin.py에 등록

`accounts/admin.py`:
```python
from django.contrib import admin
from .models import User

admin.site.register(User)
```

### settings.py 설정

`settings.py`에 맨 아래에 추가:
```python
AUTH_USER_MODEL = 'accounts.User'
```

### 마이그레이션 실행

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## 2. 회원가입 (Signup) 기능

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

`accounts/urls.py`:
```python
from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
]
```

### 커스텀 회원가입 폼

`accounts/forms.py`:
```python
from django.contrib.auth.forms import UserCreationForm
from .models import User

class CustomUserCreationForm(UserCreationForm):
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name')
```

### 회원가입 뷰

`accounts/views.py`:
```python
from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login
from .forms import CustomUserCreationForm

def signup(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)  # 회원가입 후 자동 로그인
            return redirect('articles:index')
    else:
        form = CustomUserCreationForm()

    context = {
        'form': form,
    }
    return render(request, 'accounts/signup.html', context)
```

### 회원가입 템플릿

`accounts/templates/accounts/signup.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;회원가입 - Django Board&#123;% endblock %&#125;

&#123;% block body %&#125;
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h3>회원가입</h3>
            </div>
            <div class="card-body">
                <form action="" method="POST">
                    &#123;% csrf_token %&#125;
                    &#123;% raw %&#125;&#123;&#123; form.as_p &#125;&#125;&#123;% endraw %&#125;
                    <button type="submit" class="btn btn-primary">회원가입</button>
                    <a href="&#123;% url 'accounts:login' %&#125;" class="btn btn-secondary">로그인</a>
                </form>
            </div>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

## 3. 로그인 (Login) 기능

### 로그인 폼

`accounts/forms.py`에 추가:
```python
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name')

class CustomAuthenticationForm(AuthenticationForm):
    pass  # 기본 AuthenticationForm 사용
```

### 로그인 뷰

`accounts/views.py`에 추가:
```python
from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout
from .forms import CustomUserCreationForm, CustomAuthenticationForm

def login(request):
    if request.method == "POST":
        form = CustomAuthenticationForm(request, request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            return redirect('articles:index')
    else:
        form = CustomAuthenticationForm()

    context = {
        'form': form
    }
    
    return render(request, 'accounts/login.html', context)
```

### 로그인 템플릿

`accounts/templates/accounts/login.html`:
```html
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;로그인 - Django Board&#123;% endblock %&#125;

&#123;% block body %&#125;
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h3>로그인</h3>
            </div>
            <div class="card-body">
                <form action="" method="POST">
                    &#123;% csrf_token %&#125;
                    &#123;% raw %&#125;&#123;&#123; form.as_p &#125;&#125;&#123;% endraw %&#125;
                    <button type="submit" class="btn btn-primary">로그인</button>
                    <a href="&#123;% url 'accounts:signup' %&#125;" class="btn btn-secondary">회원가입</a>
                </form>
            </div>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

## 4. 로그아웃 (Logout) 기능

### 로그아웃 뷰

`accounts/views.py`에 추가:
```python
def logout(request):
    auth_logout(request)
    return redirect('articles:index')
```

## 5. 네비게이션 바 업데이트

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
                
                <!-- 사용자 인증 상태에 따른 메뉴 표시 -->
                &#123;% if user.is_authenticated %&#125;
                    <a class="nav-link" href="&#123;% url 'accounts:logout' %&#125;">Logout</a>
                    <span class="navbar-text ms-3">안녕하세요, &#123;% raw %&#125;&#123;&#123; user.username &#125;&#125;&#123;% endraw %&#125;님!</span>
                &#123;% else %&#125;
                    <a class="nav-link" href="&#123;% url 'accounts:signup' %&#125;">Signup</a>
                    <a class="nav-link" href="&#123;% url 'accounts:login' %&#125;">Login</a>
                &#123;% endif %&#125;
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

## 6. 사용자 인증 데코레이터

### 로그인 필요 기능 보호

`articles/views.py`:
```python
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404

@login_required
def create(request):
    if request.method == "POST":
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)
            article.author = request.user  # 작성자 설정
            article.save()
            return redirect('articles:detail', id=article.id)
    else:
        form = ArticleForm()
    
    context = {'form': form}
    return render(request, 'articles/form.html', context)

@login_required
def update(request, id):
    article = get_object_or_404(Article, id=id)
    
    # 작성자만 수정 가능
    if article.author != request.user:
        return redirect('articles:detail', id=article.id)
    
    if request.method == "POST":
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('articles:detail', id=article.id)
    else:
        form = ArticleForm(instance=article)
    
    context = {'form': form, 'article': article}
    return render(request, 'articles/form.html', context)

@login_required
def delete(request, id):
    article = get_object_or_404(Article, id=id)
    
    # 작성자만 삭제 가능
    if article.author == request.user:
        article.delete()
    
    return redirect('articles:index')
```

### 모델에 작성자 필드 추가

`articles/models.py`:
```python
from django.db import models
from django.conf import settings

class Article(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title

class Comment(models.Model):
    content = models.TextField()
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.article.title} - {self.content[:20]}"
```

## 7. 댓글 작성자 표시

### 댓글 작성 뷰 업데이트

`articles/views.py`:
```python
@login_required
def comment_create(request, article_id):
    comment_form = CommentForm(request.POST)
    
    if comment_form.is_valid():
        comment = comment_form.save(commit=False)
        comment.article_id = article_id
        comment.author = request.user  # 댓글 작성자 설정
        comment.save()
        
        return redirect('articles:detail', id=article_id)
    
    return redirect('articles:detail', id=article_id)
```

### detail.html 업데이트

```html
&#123;% for comment in article.comment_set.all %&#125;
<div class="border-bottom pb-2 mb-2">
    <p>&#123;% raw %&#125;&#123;&#123; comment.content &#125;&#125;&#123;% endraw %&#125;</p>
    <small class="text-muted">
        &#123;% raw %&#125;&#123;&#123; comment.author.username &#125;&#125;&#123;% endraw %&#125; | &#123;% raw %&#125;&#123;&#123; comment.created_at &#125;&#125;&#123;% endraw %&#125;
        &#123;% if comment.author == user %&#125;
            <a href="&#123;% url 'articles:comment_delete' article_id=article.id id=comment.id %&#125;" 
               class="text-danger ms-2">삭제</a>
        &#123;% endif %&#125;
    </small>
</div>
&#123;% endfor %&#125;
```

## 8. 로그인 리다이렉트 설정

### settings.py 설정

```python
# 로그인 후 리다이렉트할 URL
LOGIN_REDIRECT_URL = '/articles/'

# 로그인이 필요한 페이지 접근 시 리다이렉트할 URL
LOGIN_URL = '/accounts/login/'
```

## 9. 사용자 프로필 기능 (선택사항)

### 프로필 모델 추가

`accounts/models.py`:
```python
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    bio = models.TextField(blank=True)
    profile_image = models.ImageField(upload_to='profiles/', blank=True)

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(blank=True)
    profile_image = models.ImageField(upload_to='profiles/', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}의 프로필"
```

### 프로필 뷰

`accounts/views.py`:
```python
@login_required
def profile(request):
    try:
        profile = request.user.userprofile
    except:
        profile = UserProfile.objects.create(user=request.user)
    
    context = {'profile': profile}
    return render(request, 'accounts/profile.html', context)
```

## 10. 실무 팁

### 1. 사용자 권한 관리

```python
from django.contrib.auth.decorators import user_passes_test

def is_staff(user):
    return user.is_staff

@user_passes_test(is_staff)
def admin_only_view(request):
    # 관리자만 접근 가능
    pass
```

### 2. 세션 관리

```python
# settings.py
SESSION_COOKIE_AGE = 1209600  # 2주
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
```

### 3. 비밀번호 정책

```python
# settings.py
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 8,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
```

### 4. 이메일 인증 (선택사항)

```python
from django.core.mail import send_mail
from django.conf import settings

def send_verification_email(user):
    subject = '이메일 인증'
    message = f'안녕하세요 {user.username}님, 이메일을 인증해주세요.'
    send_mail(subject, message, settings.EMAIL_HOST_USER, [user.email])
```

이렇게 Django의 사용자 인증 시스템을 활용하여 완전한 회원가입, 로그인, 로그아웃 기능을 구현할 수 있습니다!
