---
title: Django 밸런스 게임 프로젝트 - 실무 웹 애플리케이션 개발
categories:
- 1.TIL
- 1-6.PROJECT
tags:
- django
- 프로젝트
- 밸런스게임
- 웹개발
- 실무
- CRUD
toc: true
date: 2023-09-01 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Django 밸런스 게임 프로젝트 - 실무 웹 애플리케이션 개발

## 프로젝트 개요

Django를 활용한 밸런스 게임 웹 애플리케이션을 개발합니다:

- **참고사이트**: [wouldurather.io](https://wouldurather.io/?id=433)
- **기술스택**: Django, Bootstrap, HTML/CSS
- **주요기능**: 질문 생성, 랜덤 질문 표시, 선택 결과 통계, 질문 관리
- **학습목표**: Django의 핵심 기능을 종합적으로 활용한 실무 프로젝트

## 1. 프로젝트 구조

```
Balance_Game/
├── accounts/          # 사용자 인증 앱
├── balance/           # 메인 게임 앱
│   ├── models.py      # 데이터 모델
│   ├── views.py       # 뷰 로직
│   ├── forms.py       # 폼 처리
│   ├── urls.py        # URL 라우팅
│   └── templates/     # HTML 템플릿
│       ├── base.html
│       ├── main.html
│       ├── index.html
│       ├── form.html
│       └── setting.html
├── final/             # 프로젝트 설정
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── static/            # 정적 파일
    ├── css/
    └── js/
```

## 2. 데이터 모델 설계

### Question 모델

```python
# balance/models.py
from django.db import models

class Question(models.Model):
    question_a = models.CharField(max_length=100, verbose_name="선택지 A")
    question_b = models.CharField(max_length=100, verbose_name="선택지 B")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "질문"
        verbose_name_plural = "질문들"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.question_a} vs {self.question_b}"
    
    def get_total_answers(self):
        """총 답변 수 반환"""
        return self.answer_set.count()
    
    def get_choice_a_count(self):
        """선택지 A 답변 수 반환"""
        return self.answer_set.filter(click='a').count()
    
    def get_choice_b_count(self):
        """선택지 B 답변 수 반환"""
        return self.answer_set.filter(click='b').count()
    
    def get_choice_a_percentage(self):
        """선택지 A 비율 반환"""
        total = self.get_total_answers()
        if total == 0:
            return 0
        return round((self.get_choice_a_count() / total) * 100, 1)
    
    def get_choice_b_percentage(self):
        """선택지 B 비율 반환"""
        total = self.get_total_answers()
        if total == 0:
            return 0
        return round((self.get_choice_b_count() / total) * 100, 1)
```

### Answer 모델

```python
class Answer(models.Model):
    CHOICES = [
        ('a', '선택지 A'),
        ('b', '선택지 B'),
    ]
    
    question = models.ForeignKey(
        Question, 
        on_delete=models.CASCADE,
        verbose_name="질문"
    )
    click = models.CharField(
        max_length=1, 
        choices=CHOICES,
        verbose_name="선택"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        verbose_name = "답변"
        verbose_name_plural = "답변들"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.question} - {self.get_click_display()}"
```

## 3. 폼 설계

```python
# balance/forms.py
from django import forms
from .models import Question, Answer

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['question_a', 'question_b']
        widgets = {
            'question_a': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '선택지 A를 입력하세요'
            }),
            'question_b': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '선택지 B를 입력하세요'
            }),
        }
        labels = {
            'question_a': '선택지 A',
            'question_b': '선택지 B',
        }

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['click']
        widgets = {
            'click': forms.HiddenInput()
        }
```

## 4. 뷰 구현

### 메인 페이지 뷰

```python
# balance/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count
import random
from .models import Question, Answer
from .forms import QuestionForm, AnswerForm

def main(request):
    """메인 페이지 - 게임 시작"""
    context = {
        'total_questions': Question.objects.count(),
        'total_answers': Answer.objects.count(),
    }
    return render(request, 'balance/main.html', context)
```

### 게임 페이지 뷰

```python
def index(request, question_id):
    """게임 페이지 - 질문 표시 및 통계"""
    question = get_object_or_404(Question, id=question_id)
    
    # 통계 계산
    total_answers = question.get_total_answers()
    choice_a_count = question.get_choice_a_count()
    choice_b_count = question.get_choice_b_count()
    
    # 0으로 나누기 방지
    if total_answers == 0:
        choice_a_percent = 0
        choice_b_percent = 0
    else:
        choice_a_percent = round((choice_a_count / total_answers) * 100, 1)
        choice_b_percent = round((choice_b_count / total_answers) * 100, 1)
    
    context = {
        'question': question,
        'choice_a_count': choice_a_count,
        'choice_b_count': choice_b_count,
        'choice_a_percent': choice_a_percent,
        'choice_b_percent': choice_b_percent,
        'total_answers': total_answers,
    }
    return render(request, 'balance/index.html', context)
```

### 선택 처리 뷰

```python
def answer_click(request, question_id):
    """답변 처리 및 다음 질문으로 이동"""
    if request.method == 'GET':
        choice = request.GET.get('choice')
        
        if choice in ['a', 'b']:
            # 답변 저장
            answer = Answer.objects.create(
                question_id=question_id,
                click=choice
            )
            
            # 다음 랜덤 질문 선택
            questions = Question.objects.all()
            if questions.exists():
                next_question = random.choice(questions)
                return redirect('balance:index', question_id=next_question.id)
            else:
                messages.warning(request, '질문이 없습니다. 먼저 질문을 생성해주세요.')
                return redirect('balance:create')
        else:
            messages.error(request, '잘못된 선택입니다.')
            return redirect('balance:index', question_id=question_id)
    
    return redirect('balance:main')
```

### 질문 생성 뷰

```python
def create(request):
    """질문 생성"""
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.save()
            messages.success(request, '질문이 성공적으로 생성되었습니다.')
            return redirect('balance:index', question_id=question.id)
    else:
        form = QuestionForm()
    
    context = {'form': form}
    return render(request, 'balance/form.html', context)
```

### 질문 관리 뷰

```python
def settings(request):
    """질문 관리 페이지"""
    questions = Question.objects.annotate(
        answer_count=Count('answer')
    ).order_by('-created_at')
    
    # 페이지네이션
    paginator = Paginator(questions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'questions': page_obj,
    }
    return render(request, 'balance/setting.html', context)

def delete(request, question_id):
    """질문 삭제"""
    question = get_object_or_404(Question, id=question_id)
    question.delete()
    messages.success(request, '질문이 삭제되었습니다.')
    return redirect('balance:settings')

def edit(request, question_id):
    """질문 수정"""
    question = get_object_or_404(Question, id=question_id)
    
    if request.method == 'POST':
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            form.save()
            messages.success(request, '질문이 수정되었습니다.')
            return redirect('balance:settings')
    else:
        form = QuestionForm(instance=question)
    
    context = {
        'form': form,
        'question': question,
    }
    return render(request, 'balance/form.html', context)
```

## 5. URL 라우팅

```python
# balance/urls.py
from django.urls import path
from . import views

app_name = 'balance'

urlpatterns = [
    path('', views.main, name='main'),
    path('game/<int:question_id>/', views.index, name='index'),
    path('answer/<int:question_id>/', views.answer_click, name='answer_click'),
    path('create/', views.create, name='create'),
    path('settings/', views.settings, name='settings'),
    path('delete/<int:question_id>/', views.delete, name='delete'),
    path('edit/<int:question_id>/', views.edit, name='edit'),
]
```

```python
# final/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('balance.urls')),
]
```

## 6. 템플릿 구현

### 기본 템플릿 (base.html)

```html
<!-- balance/templates/balance/base.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>&#123;% block title %&#125;밸런스 게임&#123;% endblock %&#125;</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="&#123;% static 'css/mystyle.css' %&#125;">
</head>
<body>
    <!-- 네비게이션 바 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="&#123;% url 'balance:main' %&#125;">밸런스 게임</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="&#123;% url 'balance:main' %&#125;">Home</a>
                <a class="nav-link" href="&#123;% url 'balance:create' %&#125;">Create</a>
                <a class="nav-link" href="&#123;% url 'balance:settings' %&#125;">Settings</a>
            </div>
        </div>
    </nav>

    <!-- 메시지 표시 -->
    &#123;% if messages %&#125;
        <div class="container mt-3">
            &#123;% for message in messages %&#125;
                <div class="alert alert-&#123;% raw %&#125;&#123;&#123; message.tags &#125;&#125;&#123;% endraw %&#125; alert-dismissible fade show" role="alert">
                    &#123;% raw %&#125;&#123;&#123; message &#125;&#125;&#123;% endraw %&#125;
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            &#123;% endfor %&#125;
        </div>
    &#123;% endif %&#125;

    <!-- 메인 콘텐츠 -->
    <main class="container mt-4">
        &#123;% block content %&#125;
        &#123;% endblock %&#125;
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### 메인 페이지 템플릿

```html
<!-- balance/templates/balance/main.html -->
&#123;% extends 'balance/base.html' %&#125;
&#123;% load static %&#125;

&#123;% block title %&#125;밸런스 게임 - 메인&#123;% endblock %&#125;

&#123;% block content %&#125;
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card text-center">
            <div class="card-body">
                <h1 class="card-title">밸런스 게임</h1>
                <p class="card-text">두 가지 선택지 중 하나를 선택하는 게임입니다.</p>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card bg-light">
                            <div class="card-body">
                                <h5 class="card-title">총 질문 수</h5>
                                <h2 class="text-primary">&#123;% raw %&#125;&#123;&#123; total_questions &#125;&#125;&#123;% endraw %&#125;</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card bg-light">
                            <div class="card-body">
                                <h5 class="card-title">총 답변 수</h5>
                                <h2 class="text-success">&#123;% raw %&#125;&#123;&#123; total_answers &#125;&#125;&#123;% endraw %&#125;</h2>
                            </div>
                        </div>
                    </div>
                </div>
                
                <a href="&#123;% url 'balance:index' question_id=1 %&#125;" class="btn btn-primary btn-lg mt-4">
                    게임 시작하기
                </a>
            </div>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

### 게임 페이지 템플릿

```html
<!-- balance/templates/balance/index.html -->
&#123;% extends 'balance/base.html' %&#125;
&#123;% load static %&#125;

&#123;% block title %&#125;밸런스 게임 - &#123;% raw %&#125;&#123;&#123; question.question_a &#125;&#125;&#123;% endraw %&#125; vs &#123;% raw %&#125;&#123;&#123; question.question_b &#125;&#125;&#123;% endraw %&#125;&#123;% endblock %&#125;

&#123;% block content %&#125;
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header text-center">
                <h4>밸런스 게임</h4>
                <small class="text-muted">총 &#123;% raw %&#125;&#123;&#123; total_answers &#125;&#125;&#123;% endraw %&#125;명이 참여했습니다</small>
            </div>
            <div class="card-body">
                <div class="row">
                    <!-- 선택지 A -->
                    <div class="col-md-6">
                        <div class="choice-card" data-choice="a">
                            <div class="choice-content">
                                <h5>&#123;% raw %&#125;&#123;&#123; question.question_a &#125;&#125;&#123;% endraw %&#125;</h5>
                                <div class="choice-stats">
                                    <span class="badge bg-primary">&#123;% raw %&#125;&#123;&#123; choice_a_count &#125;&#125;&#123;% endraw %&#125;명</span>
                                    <span class="badge bg-secondary">&#123;% raw %&#125;&#123;&#123; choice_a_percent &#125;&#125;&#123;% endraw %&#125;%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- VS -->
                    <div class="col-12 text-center my-3">
                        <h2 class="text-muted">VS</h2>
                    </div>
                    
                    <!-- 선택지 B -->
                    <div class="col-md-6">
                        <div class="choice-card" data-choice="b">
                            <div class="choice-content">
                                <h5>&#123;% raw %&#125;&#123;&#123; question.question_b &#125;&#125;&#123;% endraw %&#125;</h5>
                                <div class="choice-stats">
                                    <span class="badge bg-primary">&#123;% raw %&#125;&#123;&#123; choice_b_count &#125;&#125;&#123;% endraw %&#125;명</span>
                                    <span class="badge bg-secondary">&#123;% raw %&#125;&#123;&#123; choice_b_percent &#125;&#125;&#123;% endraw %&#125;%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 진행률 바 -->
                <div class="progress mt-4" style="height: 20px;">
                    <div class="progress-bar bg-primary" role="progressbar" 
                         style="width: &#123;% raw %&#125;&#123;&#123; choice_a_percent &#125;&#125;&#123;% endraw %&#125;%">
                        &#123;% raw %&#125;&#123;&#123; choice_a_percent &#125;&#125;&#123;% endraw %&#125;%
                    </div>
                    <div class="progress-bar bg-secondary" role="progressbar" 
                         style="width: &#123;% raw %&#125;&#123;&#123; choice_b_percent &#125;&#125;&#123;% endraw %&#125;%">
                        &#123;% raw %&#125;&#123;&#123; choice_b_percent &#125;&#125;&#123;% endraw %&#125;%
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
document.querySelectorAll('.choice-card').forEach(card => {
    card.addEventListener('click', function() {
        const choice = this.dataset.choice;
        window.location.href = `&#123;% url 'balance:answer_click' question_id=question.id %&#125;?choice=${choice}`;
    });
});
</script>
&#123;% endblock %&#125;
```

### 질문 생성 템플릿

```html
<!-- balance/templates/balance/form.html -->
&#123;% extends 'balance/base.html' %&#125;
&#123;% load static %&#125;

&#123;% block title %&#125;질문 생성&#123;% endblock %&#125;

&#123;% block content %&#125;
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h4>&#123;% if question %&#125;질문 수정&#123;% else %&#125;새 질문 생성&#123;% endif %&#125;</h4>
            </div>
            <div class="card-body">
                <form method="post">
                    &#123;% csrf_token %&#125;
                    <div class="mb-3">
                        <label for="&#123;% raw %&#125;&#123;&#123; form.question_a.id_for_label &#125;&#125;&#123;% endraw %&#125;" class="form-label">선택지 A</label>
                        &#123;% raw %&#125;&#123;&#123; form.question_a &#125;&#125;&#123;% endraw %&#125;
                        &#123;% if form.question_a.errors %&#125;
                            <div class="text-danger">&#123;% raw %&#125;&#123;&#123; form.question_a.errors &#125;&#125;&#123;% endraw %&#125;</div>
                        &#123;% endif %&#125;
                    </div>
                    <div class="mb-3">
                        <label for="&#123;% raw %&#125;&#123;&#123; form.question_b.id_for_label &#125;&#125;&#123;% endraw %&#125;" class="form-label">선택지 B</label>
                        &#123;% raw %&#125;&#123;&#123; form.question_b &#125;&#125;&#123;% endraw %&#125;
                        &#123;% if form.question_b.errors %&#125;
                            <div class="text-danger">&#123;% raw %&#125;&#123;&#123; form.question_b.errors &#125;&#125;&#123;% endraw %&#125;</div>
                        &#123;% endif %&#125;
                    </div>
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary">저장</button>
                        <a href="&#123;% url 'balance:settings' %&#125;" class="btn btn-secondary">취소</a>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
&#123;% endblock %&#125;
```

### 질문 관리 템플릿

```html
<!-- balance/templates/balance/setting.html -->
&#123;% extends 'balance/base.html' %&#125;
&#123;% load static %&#125;

&#123;% block title %&#125;질문 관리&#123;% endblock %&#125;

&#123;% block content %&#125;
<div class="d-flex justify-content-between align-items-center mb-4">
    <h2>질문 관리</h2>
    <a href="&#123;% url 'balance:create' %&#125;" class="btn btn-primary">새 질문 생성</a>
</div>

<div class="row">
    &#123;% for question in questions %&#125;
    <div class="col-md-6 mb-3">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">&#123;% raw %&#125;&#123;&#123; question.question_a &#125;&#125;&#123;% endraw %&#125; vs &#123;% raw %&#125;&#123;&#123; question.question_b &#125;&#125;&#123;% endraw %&#125;</h5>
                <p class="card-text">
                    <small class="text-muted">
                        답변 수: &#123;% raw %&#125;&#123;&#123; question.answer_count &#125;&#125;&#123;% endraw %&#125;개 | 
                        생성일: &#123;% raw %&#125;&#123;&#123; question.created_at|date:"Y-m-d H:i" &#125;&#125;&#123;% endraw %&#125;
                    </small>
                </p>
                <div class="btn-group" role="group">
                    <a href="&#123;% url 'balance:index' question_id=question.id %&#125;" 
                       class="btn btn-sm btn-outline-primary">게임하기</a>
                    <a href="&#123;% url 'balance:edit' question_id=question.id %&#125;" 
                       class="btn btn-sm btn-outline-secondary">수정</a>
                    <a href="&#123;% url 'balance:delete' question_id=question.id %&#125;" 
                       class="btn btn-sm btn-outline-danger"
                       onclick="return confirm('정말 삭제하시겠습니까?')">삭제</a>
                </div>
            </div>
        </div>
    </div>
    &#123;% empty %&#125;
    <div class="col-12">
        <div class="alert alert-info text-center">
            <h4>아직 질문이 없습니다</h4>
            <p>첫 번째 질문을 생성해보세요!</p>
            <a href="&#123;% url 'balance:create' %&#125;" class="btn btn-primary">질문 생성하기</a>
        </div>
    </div>
    &#123;% endfor %&#125;
</div>

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
&#123;% endblock %&#125;
```

## 7. 스타일링 (CSS)

```css
/* static/css/mystyle.css */
.choice-card {
    border: 2px solid #dee2e6;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.choice-card:hover {
    border-color: #007bff;
    background-color: #f8f9fa;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.choice-content h5 {
    margin-bottom: 15px;
    color: #333;
}

.choice-stats {
    margin-top: 10px;
}

.choice-stats .badge {
    margin: 0 5px;
    font-size: 0.9em;
}

.progress {
    border-radius: 10px;
}

.progress-bar {
    transition: width 0.6s ease;
}

.card {
    border: none;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-radius: 15px;
}

.card-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px 15px 0 0 !important;
}

.navbar-brand {
    font-weight: bold;
    font-size: 1.5rem;
}

.btn {
    border-radius: 25px;
    padding: 10px 20px;
}

.btn-lg {
    padding: 15px 30px;
    font-size: 1.2rem;
}

.alert {
    border-radius: 10px;
    border: none;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .choice-card {
        min-height: 120px;
        padding: 15px;
    }
    
    .choice-content h5 {
        font-size: 1.1rem;
    }
}
```

## 8. 주요 학습 포인트

### 1. Django 모델 관계
- **ForeignKey**: Answer가 Question을 참조하는 1:N 관계
- **related_name**: 역참조를 통한 답변 조회 (`question.answer_set.all()`)
- **모델 메서드**: 비즈니스 로직을 모델에 캡슐화

### 2. 통계 계산 로직
- 선택지별 답변 수 집계
- 백분율 계산 및 반올림 처리
- 0으로 나누기 방지 (안전한 계산)

### 3. 랜덤 질문 선택
- `random.choice()`를 활용한 랜덤 질문 선택
- 질문 존재 여부 확인

### 4. 폼 처리
- GET 방식으로 선택값 전달
- ModelForm을 활용한 데이터 저장
- 폼 유효성 검사

### 5. URL 라우팅
- 동적 URL 패턴 (`<int:question_id>`)
- 네임스페이스를 활용한 URL 참조
- RESTful URL 설계

### 6. 템플릿 시스템
- 템플릿 상속을 통한 코드 재사용
- 템플릿 태그와 필터 활용
- 반응형 디자인

### 7. 사용자 경험 (UX)
- 실시간 통계 표시
- 직관적인 인터페이스
- 메시지 시스템을 통한 피드백

## 9. 프로젝트 확장 아이디어

### 기능 확장
```python
# 사용자 인증 추가
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    total_answers = models.IntegerField(default=0)
    favorite_category = models.CharField(max_length=50, blank=True)

# 카테고리 시스템
class Category(models.Model):
    name = models.CharField(max_length=50)
    description = models.TextField(blank=True)

class Question(models.Model):
    # 기존 필드들...
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True)
    difficulty = models.IntegerField(choices=[(1, '쉬움'), (2, '보통'), (3, '어려움')], default=2)
```

### API 개발
```python
# REST API 추가
from rest_framework import serializers, viewsets

class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = '__all__'

class QuestionViewSet(viewsets.ModelViewSet):
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer
```

### 실시간 기능
```python
# WebSocket을 활용한 실시간 통계 업데이트
# Django Channels 사용
```

## 마무리

Django 밸런스 게임 프로젝트를 통해 다음과 같은 실무 경험을 쌓을 수 있습니다:

### 핵심 학습 내용
- **Django 기본 구조**: 모델, 뷰, 템플릿의 역할과 관계
- **데이터베이스 설계**: 모델 관계와 비즈니스 로직 구현
- **사용자 인터페이스**: Bootstrap을 활용한 반응형 디자인
- **프로젝트 관리**: 체계적인 폴더 구조와 코드 조직화

### 실무 적용 가능성
- **CRUD 애플리케이션**: 기본적인 웹 애플리케이션 개발 패턴
- **통계 및 분석**: 데이터 집계와 시각화
- **사용자 경험**: 직관적인 UI/UX 설계
- **확장성**: 모듈화된 구조로 기능 확장 용이

이 프로젝트는 Django의 핵심 기능들을 종합적으로 활용하는 실무 프로젝트의 좋은 예시입니다.
