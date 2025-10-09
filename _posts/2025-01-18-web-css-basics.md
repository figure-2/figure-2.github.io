---
title: CSS 기초 - 웹 스타일링의 핵심
categories:
- 1.TIL
- 1-5.WEB
tags:
- css
- 스타일링
- 선택자
- 색상
- 테두리
- 마진
toc: true
date: 2023-08-18 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# CSS 기초 - 웹 스타일링의 핵심

## CSS 색상 적용 방법

### 1. 인라인 스타일 (Inline Style)
HTML 태그에 직접 스타일 속성으로 색깔 지정

```html
<h1 style="color: red">hello</h1>
```

### 2. 내부 스타일 (Internal Style)
head 태그 안의 style 태그에 색깔 지정

```html
<head>
    <style>
        h2 {
            color: blue;
        }
    </style>
</head>
```

### 3. 외부 스타일 (External Style)
CSS 파일을 따로 만들어서 link 설정하여 태그별로 연결

```html
<link rel="stylesheet" href="./mystyle.css">
```

**우선순위**: 태그가 같더라도 `#id`로 지정해주면 id가 우선
- id는 한 클래스에 한번만 사용 가능

## CSS 선택자

### Class 선택자
```html
<!-- HTML -->
<p class="hello">하나</p>
```

```css
/* class 선택자 */
.hello {
    color: purple;
}
```

### ID 선택자
```html
<p id='here'>two</p>
```

```css
/* id 선택자 */
#here {
    color: gold;
    background-image: url('./assets/image/background.jpg');
}
```

### 태그의 Class 선택자
```html
<p class="center">three</p>
```

```css
/* p의 class 선택자 */
p.center {
    text-align: center;
}
```

### 태그의 ID 선택자
```html
<p class="center" id="size">one</p>
```

```css
/* p의 id 선택자 */
p#size {
    font-size: 50px;
}
```

### 하위 선택자 (Descendant Selector)
```html
<ul class="english">
    <li>python</li>
    <li>HTML</li>
    <li>CSS</li>
</ul>
```

```css
/* ul의 class 선택자의 li 모두 */
ul.english > li {
    color: royalblue;
}
```

## Border (테두리)

### 개별 속성
```css
border-width: 2px;
border-style: dashed;
border-color: bisque;
```

### 축약 속성
```css
border: 2px dashed bisque; /* 위 세개를 한 번에 축약 가능 */
```

### 사용 예시
```css
.korean {
    /* border-width: 2px;
    border-style: dashed;
    border-color: bisque; */
    border: 2px dashed bisque;
    border-radius: 5px;
}
```

### Border Radius
```css
border-radius: 5px; /* 모서리를 둥글게 */
```

## Margin (여백)

### 개별 속성
```css
margin-top: 50px;
```

### 축약 속성
```css
margin: 25px 50px 75px 100px; /* top right bottom left */
```

### 사용 예시
```css
.english {
    /* margin-top: 50px; */
    margin: 25px 50px 75px 100px;
}
```

## Font (폰트)

### Google Fonts 활용
[Google Fonts](https://developers.google.com/fonts/docs/getting_started?hl=ko)에서 다양한 폰트를 무료로 사용할 수 있습니다.

### 폰트 적용 방법
```html
<!-- Google Fonts 링크 추가 -->
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
```

```css
/* CSS에서 폰트 적용 */
body {
    font-family: 'Roboto', sans-serif;
}
```

## CSS 우선순위

1. **인라인 스타일** (가장 높음)
2. **ID 선택자**
3. **Class 선택자**
4. **태그 선택자** (가장 낮음)

### 우선순위 예시
```html
<p class="text" id="special" style="color: red;">텍스트</p>
```

```css
p { color: black; }        /* 우선순위: 1 */
.text { color: blue; }     /* 우선순위: 10 */
#special { color: green; } /* 우선순위: 100 */
/* 인라인 스타일이 가장 높은 우선순위 */
```

## 실무 팁

### 1. CSS 파일 구조화
```css
/* Reset/Normalize */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Typography */
body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
}

/* Layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Components */
.button {
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
```

### 2. 반응형 디자인
```css
/* 모바일 우선 접근법 */
.container {
    width: 100%;
    padding: 10px;
}

/* 태블릿 */
@media (min-width: 768px) {
    .container {
        width: 750px;
        margin: 0 auto;
    }
}

/* 데스크톱 */
@media (min-width: 1024px) {
    .container {
        width: 1000px;
    }
}
```

### 3. CSS 변수 활용
```css
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --font-size-base: 16px;
}

.button {
    background-color: var(--primary-color);
    font-size: var(--font-size-base);
}
```
