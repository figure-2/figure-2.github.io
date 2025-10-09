---
title: HTML 기초 - 웹 개발의 첫 걸음
categories:
- 04.WEB
tags:
- - html
  - 웹개발
  - 기초
  - 태그
  - form
  - 시맨틱
toc: true
date: 2023-08-18 09:00:00
comments: false
mermaid: true
math: true
---
# HTML 기초 - 웹 개발의 첫 걸음

## 웹서비스 개요

- **HTML**: 웹 언어
- **CSS**: 만든 웹을 사람이 보기 좋게 꾸며줌
- **HTTP**: 만든 HTML을 어떻게 전송할 것인지에 대한 프로토콜

![HTTP 프로토콜](./assets/230818_HTTP.png)

### 유용한 참고 사이트
- [W3schools](https://www.w3schools.com/)
- [MDN](https://developer.mozilla.org/ko/)

## 웹 만들기 기본 과정

1. **기본 틀 생성**: `! + tab/Enter`
2. **웹 새로고침**으로 변경사항 실시간 확인
3. **기본 코딩**
   - 주석: `Ctrl + /`
   - HTML의 기본구조

### HTML 요소(Element)
```html
<태그이름>내용</태그이름> == <여는태그>내용</닫는태그>
```
- 닫는태그 안하는 경우도 있음

### 속성(Attribute)
```html
<태그이름 속성명="속성값" 속성명2="속성값2">내용</태그이름>
```

## head 태그 안

문서 정보들을 담는 공간

```html
<link rel="icon" href="./assets/image/background.jpg">
```
- link 아이콘 설정

## Body 태그 안

브라우저 화면에 나타나는 정보들을 담는 공간
- 거의 모든 태그는 `<body></body>` 안에 들어감

### 주요 태그들

#### 1. 하이퍼링크 태그
```html
<a>하이퍼텍스트</a>
```
- **속성**: `target="_blank"` => 새로운 웹사이트로 열어줌

#### 2. 제목 태그
```html
<h1>제목</h1>
<h2>제목</h2>
...
<h6>제목</h6>
```
- h1~h6: 폰트사이즈 X **중요도**라고 생각해야 함

#### 3. 문단 태그
```html
<p>문단</p>
```

**유용한 팁:**
- `Lorem + tab`: 더미데이터 만들어줌 (예시로 긴 데이터를 만들 때)

**포맷팅 요소:**
- `<b>내용</b>`: 글자를 굵게
- `<strong>내용</strong>`: 의미적으로 중요할 때

**Q. `<b>`와 `<strong>`의 차이점?**
A. `strong` 태그는 **시맨틱 태그**임. 의미적으로 중요할 때 사용, `b`태그는 단지 글자를 굵게 만드는 용도. `strong` 태그는 웹에서 분석할 때 의미적으로 의미있는 부분이라는 걸 알기위함.

**속성:**
- `class=`, `id=`

#### 4. 순서 없는 리스트
```html
<ul>제목</ul>
```
- **속성**: `style="list-style-type: square;"`, `class=`
- **주요태그**: `<li>내용</li>`
- **한 번에 여러개 만들기**: `ul>li*3` (ul태그의 자식으로 li태그 3개 만들어달라)

#### 5. 순서 있는 리스트
```html
<ol>제목</ol>
```
- **속성**: `type="i"`
- **주요태그**: `<li>내용</li>`
- **한 번에 여러개 만들기**: `ol>li*3`

#### 6. 이미지 태그
```html
<img>
```
- 닫는 태그는 없음
- **속성**:
  - `src=`: 이미지경로 (src: source)
  - `alt=`: 이미지가 안떴을 때 사용할 단어
  - `width=`, `height=`: 사진 넓이

#### 7. 아이콘 태그
```html
<i></i>
```
- 아이콘 넣기
- **속성**: `class=`
- **참고**: [Font Awesome 아이콘](https://fontawesome.com/icons/palette?f=classic&s=solid)

## Table 태그

Body 태그 안에 들어가는 table 태그

### 속성
```html
<table style="border: 1px solid">
```

### 주요 태그
- `<tr></tr>`: table row
- `<th>내용</th>`: table 헤더
  - **속성**: `style="border: 1px solid"`
- `<td>내용</td>`: table 데이터
  - **속성**: `colspan="2"`, `rowspan="2"`

### 한 번에 여러개 만들기
`tr*3>td*3`: tr 3개 만들어주는데 안에 td 3개씩 만들어달라는 의미

## Form 태그

Body 태그 안에 들어가는 form 태그
- **목적**: 앱을 만들기 위함

![Form 예시](/assets/230818_HTTP.png)

### 주요 태그

#### 1. div 태그
```html
<div></div>
```
- 태그를 연속으로 적게 되면 가로로 되는데 div를 적으면 세로로 됨

#### 2. 구분선
```html
<hr>
```
- 임의의 선

#### 3. 선택 박스
```html
<select name="" id="">
    <option value="">내용</option>
</select>
```

#### 4. 텍스트 영역
```html
<textarea name="" id="" cols="" rows=""></textarea>
```
- 텍스트를 자유롭게 크기를 설정하여 작성

#### 5. 라벨
```html
<label for="">내용</label>
```
- input받는 내용의 제목
- 연결될 태그로 이동 (for, id를 연결)

#### 6. 입력 필드
```html
<input>
```
- 입력한 데이터가 자동으로 value 값으로 들어감
- 닫는 태그는 없음

**속성:**
- `type="checkbox"`, `"radio"`, `"text"`, `"email"`, `"password"`, `"date"`, `"submit"`
- `name=""`: 링크에 ?뒤로 나오는 **key**=value
- `value=""`: 링크에 key=**value**
- `id=""`

**checkbox, radio인 경우 value 사용 이유:**
다른 건 내가 직접 입력하는데 2개는 클릭하는 value가 뭔지 지정 필요

### label 태그와 input 태그 연결
`for`(label태그), `id`(input태그)로 연결

### 웹사이트에서 input을 넣고 Enter를 치면 나오는 링크 예시
```
file:///C:/Users/yujin/Desktop/camp29/web/2.form.html?username=yujin&pw=a1234
```

**링크 구조 분석:**
- `file:///C:/Users/figure.2/Desktop/camp29/web/2.form.html` (기본 URL)
- `?` (구분자)
- `username=figure.2` (**key** = *value* - 웹에서 직접 입력)
- `&` (구분자)
- `pw=a1234` (**key** = *value* - 웹에서 직접 입력)
- `&` (구분자)
- `choice=css` (**key** = *value* - HTML에서 value값 지정)

### label 태그와 input 태그 순서의 차이

<!-- 이미지 파일이 없어서 주석 처리: ![순서 차이](/assets/230818-input_label_order.png) -->

- **label - input인 경우**: label 내용 + input박스
- **input - label인 경우**: input박스 + label 내용
