---
title: Bootstrap 기초 - 빠른 웹 개발을 위한 CSS 프레임워크
categories:
- 04_WEB_DEVELOPMENT
tags:
- bootstrap
- css프레임워크
- 유틸리티
- 컴포넌트
- 반응형
toc: true
date: 2023-08-21 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Bootstrap 기초 - 빠른 웹 개발을 위한 CSS 프레임워크

## Bootstrap 개요

[Bootstrap](https://getbootstrap.com/)은 빠르고 반응형 웹 개발을 위한 가장 인기 있는 CSS 프레임워크입니다.

### Bootstrap 설정

사용하기 위해서는 link는 head 태그, script는 body 태그에 복사 붙여넣기 해주세요.

![Bootstrap 설정](/assets/230821_bootstrap.jpg)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bootstrap Example</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

## Utilities (유틸리티 클래스)

### Background (배경색)
[Background 유틸리티](https://getbootstrap.com/docs/5.3/utilities/background/)

```html
<h1 class="text-success bg-primary">world</h1>
```

**주요 클래스:**
- `.bg-primary` - 파란색 배경
- `.bg-success` - 초록색 배경
- `.bg-warning` - 노란색 배경
- `.bg-danger` - 빨간색 배경
- `.bg-info` - 하늘색 배경

### Borders (테두리)
[Borders 유틸리티](https://getbootstrap.com/docs/5.3/utilities/borders/)

```html
<div class="border border-primary">테두리 예시</div>
```

**마진과 패딩:**
- `m-3` - 마진을 3으로 설정
- `mx-3` - margin-x (좌우 마진)
- `my-3` - margin-y (상하 마진)
- `p-3` - 패딩을 3으로 설정

![마진과 패딩](/assets/230821_margin_padding.jpg)

### Colors (색상)
[Colors 유틸리티](https://getbootstrap.com/docs/5.3/utilities/colors/)

```html
<p class="text-success">성공 메시지</p>
<div class="border-warning">경고 테두리</div>
```

**주요 색상 클래스:**
- `text-success` - 초록색 텍스트
- `text-warning` - 노란색 텍스트
- `text-danger` - 빨간색 텍스트
- `border-warning` - 노란색 테두리

### Display (표시)
[Display 유틸리티](https://getbootstrap.com/docs/5.3/utilities/display/)

```html
<div class="d-inline">인라인 요소</div>
<div class="d-none">숨겨진 요소</div>
```

**주요 클래스:**
- `d-inline` - 인라인으로 표시
- `d-block` - 블록으로 표시
- `d-none` - 숨김
- `d-flex` - 플렉스로 표시

### Flex (플렉스박스)
[Flex 유틸리티](https://getbootstrap.com/docs/5.3/utilities/flex/)

개구리 게임에서 사용했던 것처럼 레이아웃을 쉽게 구성할 수 있습니다.

```html
<div class="d-flex justify-content-between border align-items-center">
    <div>왼쪽</div>
    <div>가운데</div>
    <div>오른쪽</div>
</div>
```

**주요 클래스:**
- `d-flex` - 플렉스 컨테이너
- `justify-content-between` - 좌우 정렬
- `justify-content-center` - 가운데 정렬
- `align-items-center` - 세로 가운데 정렬

## 코드 줄임 (Emmet)

HTML 코드를 빠르게 작성할 수 있는 Emmet 문법:

```html
<!-- 띄어쓰기 금지 -->
div.d-flex>div.p-2*3
```

위 코드는 다음과 같이 확장됩니다:

```html
<div class="d-flex">
    <div class="p-2"></div>
    <div class="p-2"></div>
    <div class="p-2"></div>
</div>
```

**Emmet 문법 예시:**
- `div.container>div.row>div.col-6*2` - 컨테이너 안에 행, 열 구조
- `ul>li*5` - 5개의 리스트 아이템
- `button.btn.btn-primary*3` - 3개의 버튼

## Components (컴포넌트)

### 1. Alerts (알림)
[Alerts 컴포넌트](https://getbootstrap.com/docs/5.3/components/alerts/)

```html
<div class="alert alert-success" role="alert">
    성공적으로 저장되었습니다!
</div>

<div class="alert alert-warning alert-dismissible fade show" role="alert">
    <strong>경고!</strong> 이 작업은 되돌릴 수 없습니다.
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
</div>
```

### 2. Buttons (버튼)
[Buttons 컴포넌트](https://getbootstrap.com/docs/5.3/components/buttons/)

```html
<button type="button" class="btn btn-primary">Primary</button>
<button type="button" class="btn btn-secondary">Secondary</button>
<button type="button" class="btn btn-success">Success</button>
<button type="button" class="btn btn-danger">Danger</button>
<button type="button" class="btn btn-warning">Warning</button>
<button type="button" class="btn btn-info">Info</button>
<button type="button" class="btn btn-light">Light</button>
<button type="button" class="btn btn-dark">Dark</button>
```

**버튼 크기:**
```html
<button type="button" class="btn btn-primary btn-lg">Large button</button>
<button type="button" class="btn btn-primary">Normal button</button>
<button type="button" class="btn btn-primary btn-sm">Small button</button>
```

### 3. Card (카드)
[Card 컴포넌트](https://getbootstrap.com/docs/5.3/components/card/)

```html
<div class="card" style="width: 18rem;">
    <img src="..." class="card-img-top" alt="...">
    <div class="card-body">
        <h5 class="card-title">Card title</h5>
        <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
        <a href="#" class="btn btn-primary">Go somewhere</a>
    </div>
</div>
```

### 4. Navbar (네비게이션 바)
[Navbar 컴포넌트](https://getbootstrap.com/docs/5.3/components/navbar/)

**중요**: container 태그 밖에 설정해야 합니다.

```html
<nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container-fluid">
        <a class="navbar-brand" href="#">Navbar</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
                <li class="nav-item">
                    <a class="nav-link active" href="#">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#">Features</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#">Pricing</a>
                </li>
            </ul>
        </div>
    </div>
</nav>
```

## 그리드 시스템

Bootstrap의 핵심인 12열 그리드 시스템:

```html
<div class="container">
    <div class="row">
        <div class="col-md-4">4열</div>
        <div class="col-md-4">4열</div>
        <div class="col-md-4">4열</div>
    </div>
    <div class="row">
        <div class="col-md-6">6열</div>
        <div class="col-md-6">6열</div>
    </div>
</div>
```

**반응형 브레이크포인트:**
- `col-sm-*` - 576px 이상
- `col-md-*` - 768px 이상
- `col-lg-*` - 992px 이상
- `col-xl-*` - 1200px 이상

## 실무 팁

### 1. 커스터마이징
```css
/* 커스텀 CSS로 Bootstrap 스타일 오버라이드 */
.btn-custom {
    background-color: #ff6b6b;
    border-color: #ff6b6b;
}

.btn-custom:hover {
    background-color: #ff5252;
    border-color: #ff5252;
}
```

### 2. 유틸리티 조합
```html
<!-- 여러 유틸리티 클래스 조합 -->
<div class="d-flex justify-content-center align-items-center bg-primary text-white p-4 rounded">
    중앙 정렬된 컨텐츠
</div>
```

### 3. 반응형 디자인
```html
<!-- 화면 크기에 따라 다른 레이아웃 -->
<div class="col-12 col-md-6 col-lg-4">
    모바일: 12열, 태블릿: 6열, 데스크톱: 4열
</div>
```
