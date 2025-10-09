---
title: Django 기초 - Python 웹 프레임워크의 핵심
categories:
- 04_WEB_DEVELOPMENT
tags:
- django
- python
- 웹프레임워크
- mtv
- url
- 뷰
- 템플릿
toc: true
date: 2023-08-21 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django 기초 - Python 웹 프레임워크의 핵심

## Django 개요

웹사이트는 HTML, CSS로도 만들 수 있지만, **Django**로 더욱 동적이고 강력한 웹 애플리케이션을 만들 수 있습니다.

## Django 프로젝트 구조

### 주요 파일들
- `__init__.py` - 패키지로 인식하게 하는 파일
- `admin.py` - 관리자 페이지 설정
- `models.py` - 데이터베이스 모델 정의 (데이터 구조)
- `tests.py` - 테스트 코드
- **`views.py`** - 비즈니스 로직 처리
- **`urls.py`** - URL 라우팅 관리
- **`templates/`** - HTML 템플릿 폴더

### views.py
HTML 파일이 화면에 어떻게 보일지를 설정하는 공간입니다.
- 들어오는 요청 URL을 파싱하여 라우팅 처리된 특정 요청을 처리
- `render`: 결과물을 만들어냄

### urls.py
URL 관리를 담당합니다.
- URL을 만들어 연결
- 요청된 URL을 적절한 뷰 함수로 라우팅

### Template 폴더
View로부터 전달된 데이터를 템플릿에 적용하여 Dynamic한 웹페이지를 만드는데 사용됩니다.

## Django 아키텍처 패턴

### MVC vs MTV
- **MVC** (Model Views Controller) - 일반적인 웹 프레임워크 패턴
- **MTV** (Model Template Views) - Django식 패턴

**Django MTV 패턴:**
- **Model**: 데이터베이스와 상호작용
- **Template**: 사용자에게 보여지는 화면 (HTML)
- **Views**: 비즈니스 로직 처리

## Django 웹 요청 처리 흐름

![Django 흐름](/assets/230821_django_flow.jpg)

### 1. 클라이언트 요청
클라이언트가 특정 주소를 요청합니다:
- `127.0.0.1:8000/index`
- `127.0.0.1:8000/greeting/yujin`
- `127.0.0.1:8000/cube/3`

### 2. urls.py에서 URL 처리
URL 주소를 받아서 적절한 뷰로 라우팅합니다:

```python
from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index),
    path('greeting/<name>/', views.greeting),
    path('cube/<int:num>/', views.cube),
    # 바로 int 처리하기 위해서 <int:num>
]
```

**URL 패턴 설명:**
- `'index/'` - 정확한 매칭
- `'greeting/<name>/'` - name이라는 변수로 동적 URL 처리
- `'cube/<int:num>/'` - num을 정수로 변환하여 처리

### 3. views.py에서 요청 처리
URL 처리해서 응답을 반환합니다:

```python
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def greeting(request, name):
    result = {
        'name': name,
    }
    return render(request, 'greeting.html', result)

def cube(request, num):
    result = {
        'num': num,
        'cube': num ** 3,
    }
    return render(request, 'cube.html', result)
```

**뷰 함수 설명:**
- `index` 함수: 단순히 index.html을 렌더링
- `greeting` 함수: name 변수를 받아서 템플릿에 전달
- `cube` 함수: num을 받아서 세제곱 계산 후 결과를 템플릿에 전달

### 4. Models (데이터베이스)
데이터베이스에 있는 데이터를 꺼내옵니다 (이 예시에서는 사용하지 않음)

### 5. Template (HTML)
templates 폴더에서 HTML 파일들을 생성합니다:

#### index.html
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Index</title>
</head>
<body>
    <h1>Hello Django!</h1>
</body>
</html>
```

#### greeting.html
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Greeting</title>
</head>
<body>
    <h1>&#123;&#123;name&#125;&#125;님 안녕하세요</h1>
</body>
</html>
```

#### cube.html
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Cube</title>
</head>
<body>
    <h1>&#123;&#123;num&#125;&#125;을 세제곱한 결과는 &#123;&#123;cube&#125;&#125;입니다.</h1>
</body>
</html>
```

**템플릿 문법:**
- `&#123;&#123;변수명&#125;&#125;` - 변수 출력
- `&#123;% 태그 %&#125;` - 템플릿 태그 (조건문, 반복문 등)

### 6. 웹 실행 (HTML Response)
서버를 실행하고 결과를 확인합니다:

```bash
python manage.py runserver
```

**실행 결과:**

#### http://127.0.0.1:8000/index
![Index 페이지](/assets/230821_index_page.jpg)

#### http://127.0.0.1:8000/greeting/yujin
![Greeting 페이지](/assets/230821_greeting_page.jpg)

#### http://127.0.0.1:8000/cube/3
![Cube 페이지](/assets/230821_cube_page.jpg)

## Django 개발 팁

### 1. URL 패턴 설계
```python
# 좋은 URL 패턴
urlpatterns = [
    path('articles/', views.article_list),
    path('articles/<int:article_id>/', views.article_detail),
    path('articles/<int:article_id>/comments/', views.comment_list),
]
```

### 2. 뷰 함수 구조화
```python
def article_detail(request, article_id):
    # 1. 데이터 조회
    article = Article.objects.get(id=article_id)
    
    # 2. 비즈니스 로직 처리
    comments = article.comment_set.all()
    
    # 3. 컨텍스트 준비
    context = {
        'article': article,
        'comments': comments,
    }
    
    # 4. 템플릿 렌더링
    return render(request, 'articles/detail.html', context)
```

### 3. 템플릿 상속
```html
<!-- base.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>&#123;% block title %&#125;My Site&#123;% endblock %&#125;</title>
</head>
<body>
    <header>
        <h1>My Django Site</h1>
    </header>
    
    <main>
        &#123;% block content %&#125;
        &#123;% endblock %&#125;
    </main>
</body>
</html>
```

```html
<!-- index.html -->
&#123;% extends 'base.html' %&#125;

&#123;% block title %&#125;Home - My Site&#123;% endblock %&#125;

&#123;% block content %&#125;
<h2>Welcome to My Site!</h2>
<p>This is the home page.</p>
&#123;% endblock %&#125;
```

## 실무 적용 예시

### 동적 URL과 폼 처리
```python
# views.py
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        
        # 이메일 발송 로직
        send_email(name, email, message)
        
        return render(request, 'contact_success.html')
    
    return render(request, 'contact.html')
```

```html
<!-- contact.html -->
<form method="POST">
    &#123;% csrf_token %&#125;
    <input type="text" name="name" placeholder="이름" required>
    <input type="email" name="email" placeholder="이메일" required>
    <textarea name="message" placeholder="메시지" required></textarea>
    <button type="submit">전송</button>
</form>
```

이렇게 Django는 MTV 패턴을 통해 체계적이고 확장 가능한 웹 애플리케이션을 구축할 수 있게 해줍니다.
