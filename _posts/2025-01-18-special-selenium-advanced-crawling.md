---
title: Selenium을 활용한 고급 크롤링
categories:
- 1.TIL
- 1-10.SPECIAL
tags:
- 특강
- Selenium
- 고급크롤링
- 동적웹페이지
- JavaScript
- WebDriver
- 대기처리
- 스크롤
- 팝업처리
- 성능최적화
- 에러처리
toc: true
date: 2023-08-15 13:00:00 +0900
comments: false
mermaid: true
math: true
---
# Selenium을 활용한 고급 크롤링
> 동적 웹 페이지와 JavaScript 기반 사이트 크롤링

## Selenium이란?
Selenium은 웹 브라우저를 자동화하는 도구로, JavaScript로 동적으로 생성되는 콘텐츠를 크롤링할 수 있습니다. requests와 BeautifulSoup으로 처리할 수 없는 동적 웹사이트에 필수적인 도구입니다.

## Selenium의 특징
- **브라우저 자동화**: 실제 브라우저를 조작
- **JavaScript 실행**: 동적 콘텐츠 처리 가능
- **사용자 행동 시뮬레이션**: 클릭, 스크롤, 입력 등
- **다양한 브라우저 지원**: Chrome, Firefox, Safari 등

## 설치 및 설정

### 1. 라이브러리 설치
```bash
pip install selenium
```

### 2. WebDriver 설치
```python
# Chrome WebDriver 자동 설치
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# WebDriver 자동 설치 및 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
```

### 3. 기본 설정
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument('--headless')  # 백그라운드 실행
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=chrome_options)
```

## 기본 사용법

### 1. 페이지 접속
```python
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get('https://example.com')

# 페이지 제목 확인
print(driver.title)

# 현재 URL 확인
print(driver.current_url)
```

### 2. 요소 찾기
```python
# ID로 요소 찾기
element = driver.find_element(By.ID, 'element_id')

# 클래스명으로 요소 찾기
elements = driver.find_elements(By.CLASS_NAME, 'class_name')

# CSS 선택자로 요소 찾기
element = driver.find_element(By.CSS_SELECTOR, 'div.content')

# XPath로 요소 찾기
element = driver.find_element(By.XPATH, '//div[@class="content"]')
```

### 3. 요소 조작
```python
# 텍스트 입력
search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys('검색어')

# 클릭
button = driver.find_element(By.ID, 'submit')
button.click()

# 텍스트 가져오기
text = element.text

# 속성 가져오기
href = element.get_attribute('href')
```

## 고급 기법

### 1. 대기 처리
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 명시적 대기
wait = WebDriverWait(driver, 10)
element = wait.until(EC.presence_of_element_located((By.ID, 'dynamic_element')))

# 요소가 클릭 가능할 때까지 대기
clickable_element = wait.until(EC.element_to_be_clickable((By.ID, 'button')))
```

### 2. 스크롤 처리
```python
# 페이지 끝까지 스크롤
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# 특정 요소까지 스크롤
element = driver.find_element(By.ID, 'target')
driver.execute_script("arguments[0].scrollIntoView();", element)

# 점진적 스크롤
for i in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
```

### 3. 팝업 및 알림 처리
```python
# 알림 창 처리
alert = driver.switch_to.alert
alert.accept()  # 확인
# alert.dismiss()  # 취소

# 새 탭으로 전환
driver.switch_to.window(driver.window_handles[1])

# 원래 탭으로 돌아가기
driver.switch_to.window(driver.window_handles[0])
```

### 4. 파일 다운로드
```python
# 다운로드 디렉토리 설정
chrome_options = Options()
prefs = {
    "download.default_directory": "/path/to/download",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=chrome_options)
```

## 실무 활용 예제

### 1. 동적 콘텐츠 크롤링
```python
def crawl_dynamic_content(url):
    driver = webdriver.Chrome()
    driver.get(url)
    
    # 페이지 로딩 대기
    wait = WebDriverWait(driver, 10)
    
    # 더보기 버튼 클릭 (여러 번)
    for i in range(5):
        try:
            more_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.more-button'))
            )
            more_button.click()
            time.sleep(2)
        except:
            break
    
    # 모든 콘텐츠 수집
    items = driver.find_elements(By.CSS_SELECTOR, '.content-item')
    data = []
    
    for item in items:
        title = item.find_element(By.CSS_SELECTOR, '.title').text
        content = item.find_element(By.CSS_SELECTOR, '.content').text
        data.append({'title': title, 'content': content})
    
    driver.quit()
    return data
```

### 2. 로그인이 필요한 사이트 크롤링
```python
def login_and_crawl(username, password, target_url):
    driver = webdriver.Chrome()
    
    # 로그인 페이지 접속
    driver.get('https://example.com/login')
    
    # 로그인 정보 입력
    username_field = driver.find_element(By.NAME, 'username')
    password_field = driver.find_element(By.NAME, 'password')
    
    username_field.send_keys(username)
    password_field.send_keys(password)
    
    # 로그인 버튼 클릭
    login_button = driver.find_element(By.CSS_SELECTOR, 'input[type="submit"]')
    login_button.click()
    
    # 로그인 성공 대기
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'dashboard')))
    
    # 대상 페이지로 이동
    driver.get(target_url)
    
    # 데이터 크롤링
    data = extract_data(driver)
    
    driver.quit()
    return data
```

### 3. 무한 스크롤 페이지 크롤링
```python
def crawl_infinite_scroll(url, max_scrolls=10):
    driver = webdriver.Chrome()
    driver.get(url)
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    
    while scroll_count < max_scrolls:
        # 페이지 끝까지 스크롤
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # 새 콘텐츠 로딩 대기
        time.sleep(2)
        
        # 새로운 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            break
            
        last_height = new_height
        scroll_count += 1
    
    # 모든 콘텐츠 수집
    items = driver.find_elements(By.CSS_SELECTOR, '.item')
    data = [item.text for item in items]
    
    driver.quit()
    return data
```

## 성능 최적화

### 1. 헤드리스 모드
```python
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
```

### 2. 이미지 로딩 비활성화
```python
chrome_options = Options()
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)
```

### 3. 불필요한 플러그인 비활성화
```python
chrome_options = Options()
chrome_options.add_argument('--disable-plugins')
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-images')
```

## 에러 처리

### 1. 요소 찾기 실패 처리
```python
from selenium.common.exceptions import NoSuchElementException

try:
    element = driver.find_element(By.ID, 'non-existent')
    print(element.text)
except NoSuchElementException:
    print("요소를 찾을 수 없습니다.")
```

### 2. 타임아웃 처리
```python
from selenium.common.exceptions import TimeoutException

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'slow-loading'))
    )
except TimeoutException:
    print("요소 로딩 시간 초과")
```

### 3. 재시도 로직
```python
import time
from selenium.common.exceptions import WebDriverException

def retry_operation(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except WebDriverException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # 지수 백오프
```

## 주요 학습 포인트

### 1. 동적 콘텐츠 처리
- JavaScript 실행 대기
- AJAX 요청 완료 확인
- 동적 요소 로딩 대기

### 2. 사용자 행동 시뮬레이션
- 자연스러운 클릭 패턴
- 적절한 대기 시간
- 인간적인 행동 패턴

### 3. 성능 최적화
- 불필요한 리소스 로딩 방지
- 효율적인 요소 선택
- 메모리 사용량 관리

### 4. 안정성 확보
- 예외 처리
- 재시도 로직
- 로깅 및 모니터링

Selenium은 복잡한 동적 웹사이트를 크롤링하는 강력한 도구로, 적절한 기법을 사용하면 안정적이고 효율적인 크롤링이 가능합니다.
