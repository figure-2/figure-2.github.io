---
title: JavaScript 기초 - 웹 상호작용의 핵심
categories:
- 1.TIL
- 1-5.WEB
tags:
- javascript
- dom
- 이벤트
- 비동기
- ajax
- fetch
toc: true
date: 2023-09-05 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# JavaScript 기초 - 웹 상호작용의 핵심

## 개요

JavaScript는 웹사이트에 상호작용성을 더해주는 프로그래밍 언어입니다:

- **상호작용성**: 게임, 버튼 클릭, 폼 입력 반응
- **동적 스타일링**: 애니메이션, 스타일 변경
- **비동기 처리**: 서버와의 통신

## 1. JavaScript 기초 문법

### DOM 조작

```javascript
// HTML 요소 선택
let myHeading = document.querySelector("h1");  // h1 태그 선택
myHeading.textContent = "Hello world!";  // 텍스트 내용 변경
```

### JavaScript 파일 연결

```html
<!-- 외부 JavaScript 파일 연결 -->
<script src="scripts/main.js"></script>

<!-- HTML 내부에 JavaScript 작성 -->
<script>
    console.log('hello');
</script>
```

## 2. 변수 선언

### 변수 선언 방법

```javascript
// 변수 선언
let myVariable;

// 값 할당
myVariable = 'hello';
console.log(myVariable);

let myVariable2 = 'world';
console.log(myVariable2);

// var (옛날 방식, 거의 사용하지 않음)
var a = 1;

// let (재할당 가능)
let b = 2;

// const (재할당 불가능)
const c = 3;

console.log(a, b, c);

// 재할당
a = 10;
b = 20;
// c = 30;  // 오류: const는 재할당 불가능

// 재선언
var a = 100;  // 가능
// let b = 200;  // 오류: let은 재선언 불가능
// const c = 300;  // 오류: const는 재선언 불가능
```

### 변수 선언 규칙

- **var**: 함수 스코프, 재선언 가능 (사용 권장하지 않음)
- **let**: 블록 스코프, 재할당 가능, 재선언 불가능
- **const**: 블록 스코프, 재할당 불가능, 재선언 불가능

## 3. 데이터 타입

### 기본 타입들

```javascript
let stringVar = 'hello';        // 문자열
let numberVar = 10;             // 숫자
let boolVar = true;             // 불린
let arrayVar = [];              // 배열
let objectVar = {};             // 객체

console.log(stringVar, numberVar, boolVar, arrayVar, objectVar);

// 배열 조작
arrayVar.push('hello');
console.log(arrayVar);

// 객체 조작
objectVar.name = 'kim';
objectVar.location = 'seoul';
console.log(objectVar);

// 객체 생성자 사용
let myObject = new Object();
myObject.name = 'park';
console.log(myObject);
```

## 4. 연산자

### 비교 연산자

```javascript
let myVarA = 10;
let myVarB = '10';

console.log(myVarA == myVarB);   // true (타입 변환 후 비교)
console.log(myVarA === myVarB);  // false (타입까지 정확히 비교)

// 권장: === 사용 (엄격한 비교)
```

### 연산자 종류

- **==**: 동등 비교 (타입 변환)
- **===**: 일치 비교 (타입까지 정확히)
- **!=**: 부등 비교
- **!==**: 불일치 비교

## 5. 조건문

### if-else 문

```javascript
let iceCream = 'chocolate';

if (iceCream === 'chocolate') {
    alert('choco');
} else {
    alert('no choco');
}
```

### 조건문 패턴

```javascript
// 기본 if문
if (condition) {
    // 실행할 코드
}

// if-else문
if (condition) {
    // 조건이 true일 때
} else {
    // 조건이 false일 때
}

// if-else if-else문
if (condition1) {
    // 조건1이 true일 때
} else if (condition2) {
    // 조건2가 true일 때
} else {
    // 모든 조건이 false일 때
}
```

## 6. 반복문

### while 반복문

```javascript
let i = 0;  // 변수
while (i < 5) {  // 조건
    console.log(i);
    i++;  // 증가 (i += 1과 동일)
}
```

### for 반복문

```javascript
// 기본 for문 구조: for (변수; 조건; 증가)
for (let i = 0; i < 5; i++) {
    console.log(i);
}

// 배열과 함께 사용
let myArray = [1, 2, 3, 4, 5];
for (let i = 0; i < myArray.length; i++) {
    console.log(myArray[i]);
}
```

### for-in과 for-of

```javascript
let myArray = [1, 2, 3, 4, 5];

// for-in: 인덱스를 추출
for (let item in myArray) {
    console.log(item);  // 결과: 0, 1, 2, 3, 4
}

// for-of: 값을 추출 (많이 사용)
for (let item of myArray) {
    console.log(item);  // 결과: 1, 2, 3, 4, 5
}
```

### forEach 메서드

```javascript
// forEach: 가장 많이 사용 (Python의 map 함수와 유사)
myArray.forEach(function(item, index, array) {
    console.log(item);   // 값: 1, 2, 3, 4, 5
    console.log(index);  // 인덱스: 0, 1, 2, 3, 4
    console.log(array);  // 전체 배열: [1, 2, 3, 4, 5]
});
```

## 7. 함수

### 함수 선언

```javascript
// 함수 선언식
function multiply(num1, num2) {
    let result = num1 * num2;
    console.log(this);  // this: 현재 객체 (self와 유사)
    return result;
}
console.log(multiply(3, 4));
```

### 함수 표현식

```javascript
// 함수 표현식
let multiply2 = function (num1, num2) {
    let result = num1 * num2;
    console.log(this);
    return result;
};
console.log(multiply2(3, 4));
```

### 화살표 함수

```javascript
// 화살표 함수
let multiply3 = (num1, num2) => {
    let result = num1 * num2;
    console.log(this);
    return result;
};
console.log(multiply3(3, 4));

// 화살표 함수 생략 1: return 문만 있을 때
let multiply4 = (num1, num2) => {
    return num1 * num2;
};

let multiply4_1 = (num1, num2) => num1 * num2;

// 화살표 함수 생략 2: 매개변수가 하나일 때
let multiply5 = (num1) => num1 * 2;
let multiply5_1 = num1 => num1 * 2;
```

### 객체에서 함수 사용

```javascript
let testObj = {
    mul1: multiply,    // 함수
    mul2: multiply2,   // 함수
    mul3: multiply3    // 화살표 함수
};
```

## 8. 이벤트 처리

### 기본 이벤트 처리

```javascript
// HTML 요소 클릭 이벤트
document.querySelector('html').onclick = function() {
    alert('hihi');
};

document.querySelector('h1').onclick = function() {
    alert('hihello');
};
```

### addEventListener 사용

```javascript
// h1 태그 클릭 이벤트
let myH1 = document.querySelector('h1');
myH1.addEventListener('click', function(e) {
    console.log('hihi');
    console.log(e);  // 이벤트 객체
});

// h1 태그에 마우스 오버 이벤트
myH1.addEventListener('mouseover', function(e) {
    console.log('hihi');
    console.log(e);
});
```

### 이미지 클릭으로 사진 변경

```javascript
let myImage = document.querySelector('img');
console.log(myImage);  // 요소 확인

myImage.addEventListener('click', function() {
    let src = myImage.getAttribute('src');  // 속성 가져오기
    console.log(src);
    
    if (src === 'images/firefox-icon.png') {
        myImage.setAttribute('src', 'images/푸바옹.jpg');  // 속성 변경
    } else {
        myImage.setAttribute('src', 'images/firefox-icon.png');
    }
});
```

### 여러 요소에 이벤트 추가

```javascript
// 모든 li 태그 선택
const liElements = document.querySelectorAll('li');
console.log(liElements);  // NodeList(3) [li, li, li]

// for-of로 순회
for (const li of liElements) {
    console.log(li);
}

// 스타일 적용
liElements.forEach(function(li) {
    li.style.color = 'red';
    li.style.backgroundColor = 'blue';
});

// 모든 li에 같은 이벤트 추가
liElements.forEach(function(li) {
    li.addEventListener('click', function() {
        console.log('hello');
    });
});

// 각 li에 다른 동작 지정
liElements.forEach(function(li) {
    li.addEventListener('click', function(e) {
        console.log(e);
        console.log(e.target);
        console.log(e.target.textContent);
        
        if (e.target.textContent === 'thinkers') {
            e.target.style.color = 'black';
        }
    });
});
```

## 9. 비동기 처리

### 동기 vs 비동기

- **동기**: 작업이 시작하고 끝나기 전까지는 아무것도 할 수 없음
- **비동기**: 작업이 돌발적으로 들어올 때 처리 (JavaScript의 특징)

### setTimeout 사용

```javascript
console.log('hi');
// setTimeout(실행할 함수, 시간(밀리초))
setTimeout(function() {
    console.log('??');
}, 2000);  // 2초 후 실행
console.log('bye');  // 결과: hi, bye, ??
```

### fetch API 사용

```javascript
const URL = 'https://jsonplaceholder.typicode.com/posts/1';

// 잘못된 방법 (오류 발생)
let response = fetch(URL);
let result = response.json();
console.log(response);
console.log(result);  // 오류 발생

// 올바른 방법 1: Promise 체이닝
fetch(URL)
    .then(response => response.json())
    .then(json => console.log(json));

// 올바른 방법 2: async/await
async function fetchAndPrint() {
    let response = await fetch(URL);  // fetch 완료까지 대기
    let result = await response.json();  // JSON 변환 완료까지 대기
    console.log(result);
}
fetchAndPrint();
```

## 10. Django와 JavaScript 연동

### AJAX로 좋아요 기능 구현

#### HTML 수정

```html
<!-- 수정 전 -->
<a href="&#123;% url 'posts:like' post_id=post.id %&#125;" class="text-reset text-decoration-none">
    &#123;% if post in user.like_posts.all %&#125;
        <i class="bi bi-heart-fill" style="color: red;"></i>
    &#123;% else %&#125;
        <i class="bi bi-heart"></i>
    &#123;% endif %&#125;
</a> &#123;% raw %&#125;&#123;&#123; post.like_users.all|length &#125;&#125;&#123;% endraw %&#125;명이 좋아합니다.

<!-- 수정 후 -->
&#123;% if post in user.like_posts.all %&#125;
    <i class="bi bi-heart-fill heart" style="color: red;" data-post-id="&#123;% raw %&#125;&#123;&#123; post.id &#125;&#125;&#123;% endraw %&#125;"></i>
&#123;% else %&#125;
    <i class="bi bi-heart heart" data-post-id="&#123;% raw %&#125;&#123;&#123; post.id &#125;&#125;&#123;% endraw %&#125;"></i>
&#123;% endif %&#125;
&#123;% raw %&#125;&#123;&#123; post.like_users.all|length &#125;&#125;&#123;% endraw %&#125;명이 좋아합니다.
```

#### JavaScript 코드

```html
<script>
    const likeButtons = document.querySelectorAll('i.heart');
    
    likeButtons.forEach((likeButton) => {
        likeButton.addEventListener('click', (event) => {
            let postId = event.target.dataset.postId;
            likeRequest(event.target, postId);
        });
    });
    
    let likeRequest = async (button, postId) => {
        console.log(button, postId);
        
        // Django 서버에 요청 보내기
        let likeURL = `/posts/${postId}/like-async/`;
        let response = await fetch(likeURL);
        let result = await response.json();
        
        console.log(result);
        
        // Django 서버 응답에 따라 좋아요 버튼 수정
        if (result.status) {
            // true => 좋아요가 눌린 경우
            button.classList.remove('bi-heart');
            button.classList.add('bi-heart-fill');
            button.style.color = 'red';
        } else {
            // false => 좋아요 취소
            button.classList.remove('bi-heart-fill');
            button.classList.add('bi-heart');
            button.style.color = 'black';
        }
        
        // 좋아요 개수 업데이트
        button.nextSibling.textContent = ` ${result.count}명이 좋아합니다.`;
    };
</script>
```

#### Django 뷰 수정

```python
# urls.py
urlpatterns = [
    path('<int:post_id>/like-async/', views.like_async, name='like-async'),
]

# views.py
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

@login_required
def like_async(request, post_id):
    user = request.user;
    post = get_object_or_404(Post, id=post_id);
    
    if post in user.like_posts.all():
        post.like_users.remove(user);
        status = False;
    else:
        post.like_users.add(user);
        status = True;
    
    context = {
        'status': status,
        'count': post.like_users.count(),
    };
    return JsonResponse(context);
```

## 11. 실무 팁

### 1. 이벤트 위임 (Event Delegation)

```javascript
// 부모 요소에 이벤트 리스너를 추가하여 자식 요소들의 이벤트 처리
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('like-button')) {
        handleLike(e.target);
    }
});
```

### 2. 디바운싱 (Debouncing)

```javascript
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 사용 예시
const debouncedSearch = debounce(function(query) {
    // 검색 로직
}, 300);
```

### 3. 에러 처리

```javascript
async function fetchData() {
    try {
        const response = await fetch('/api/data/');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        // 에러 처리 로직
    }
}
```

### 4. 로컬 스토리지 활용

```javascript
// 데이터 저장
localStorage.setItem('user', JSON.stringify({name: 'John', age: 30}));

// 데이터 불러오기
const user = JSON.parse(localStorage.getItem('user'));

// 데이터 삭제
localStorage.removeItem('user');
```

### 5. 모듈화

```javascript
// utils.js
export function formatDate(date) {
    return new Date(date).toLocaleDateString();
}

export function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// main.js
import { formatDate, validateEmail } from './utils.js';
```

이렇게 JavaScript의 기초부터 Django와의 연동까지 완전히 마스터할 수 있습니다!
