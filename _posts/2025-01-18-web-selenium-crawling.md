---
title: Selenium 크롤링 - 동적 웹 페이지 자동화
categories:
- 1.TIL
- 1-5.WEB
tags:
- selenium
- 웹자동화
- webdriver
- 동적콘텐츠
- 크롤링
- 브라우저자동화
toc: true
date: 2023-10-20 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Selenium 크롤링 - 동적 웹 페이지 자동화

## 개요

Selenium은 웹 브라우저 자동화를 위한 강력한 도구입니다:

- **웹 브라우저 자동화**: 실제 브라우저를 제어하여 웹 페이지 조작
- **JavaScript 실행**: JavaScript 코드 실행 및 동적 콘텐츠 처리
- **사용자 시뮬레이션**: 클릭, 입력, 스크롤 등 사용자 행동 시뮬레이션
- **다양한 브라우저**: Chrome, Firefox, Safari 등 다양한 브라우저 지원

## 1. Selenium 개요

### 주요 특징

- **동적 콘텐츠**: JavaScript로 생성되는 콘텐츠 처리 가능
- **실제 브라우저**: 실제 사용자 환경과 동일한 조건
- **강력한 기능**: 복잡한 웹 애플리케이션 자동화
- **디버깅**: 브라우저에서 직접 확인 가능

### 장점

- **완전한 렌더링**: JavaScript로 생성되는 모든 콘텐츠 접근 가능
- **사용자 행동 시뮬레이션**: 실제 사용자처럼 상호작용
- **디버깅 용이**: 브라우저에서 직접 확인 가능
- **강력한 선택자**: CSS 선택자, XPath 등 다양한 요소 선택 방법

## 2. Selenium 설치 및 설정

### Selenium 설치

```bash
# pip 설치
pip install selenium

# WebDriver 설치 (Chrome)
# ChromeDriver 다운로드: https://chromedriver.chromium.org/
# 또는 자동 설치
pip install webdriver-manager
```

### 기본 설정

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# Chrome 옵션 설정
def setup_chrome_options():
    """Chrome 옵션 설정"""
    options = Options()
    
    # 헤드리스 모드 (브라우저 창 숨기기)
    options.add_argument('--headless')
    
    # 성능 최적화
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    
    # User-Agent 설정
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    # 창 크기 설정
    options.add_argument('--window-size=1920,1080')
    
    return options

# WebDriver 생성
def create_driver():
    """WebDriver 생성"""
    options = setup_chrome_options()
    
    # ChromeDriver 자동 설치 및 생성
    driver = webdriver.Chrome(
        service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
        options=options
    )
    
    return driver

# 기본 WebDriver 생성
driver = create_driver()
```

## 3. Selenium 기본 사용법

### 페이지 접근 및 요소 찾기

```python
def basic_selenium_example():
    """Selenium 기본 사용법 예시"""
    
    try:
        # 페이지 접근
        driver.get("https://example.com")
        
        # 페이지 제목 확인
        title = driver.title
        print(f"페이지 제목: {title}")
        
        # 요소 찾기 (다양한 방법)
        # ID로 찾기
        element_by_id = driver.find_element(By.ID, "element_id")
        
        # 클래스명으로 찾기
        element_by_class = driver.find_element(By.CLASS_NAME, "class_name")
        
        # CSS 선택자로 찾기
        element_by_css = driver.find_element(By.CSS_SELECTOR, ".class_name")
        
        # XPath로 찾기
        element_by_xpath = driver.find_element(By.XPATH, "//div[@class='class_name']")
        
        # 텍스트로 찾기
        element_by_text = driver.find_element(By.LINK_TEXT, "링크 텍스트")
        
        # 부분 텍스트로 찾기
        element_by_partial_text = driver.find_element(By.PARTIAL_LINK_TEXT, "부분 텍스트")
        
        # 여러 요소 찾기
        elements = driver.find_elements(By.CLASS_NAME, "class_name")
        print(f"찾은 요소 수: {len(elements)}")
        
        return driver
    
    except Exception as e:
        print(f"Selenium 오류: {e}")
        return None

# 기본 예시 실행
# basic_selenium_example()
```

### 요소 조작

```python
def element_interaction_example():
    """요소 조작 예시"""
    
    try:
        # 페이지 접근
        driver.get("https://example.com")
        
        # 텍스트 입력
        input_element = driver.find_element(By.ID, "input_id")
        input_element.clear()  # 기존 텍스트 지우기
        input_element.send_keys("입력할 텍스트")
        
        # 클릭
        button_element = driver.find_element(By.ID, "button_id")
        button_element.click()
        
        # 체크박스 선택/해제
        checkbox = driver.find_element(By.ID, "checkbox_id")
        if not checkbox.is_selected():
            checkbox.click()
        
        # 라디오 버튼 선택
        radio_button = driver.find_element(By.ID, "radio_id")
        radio_button.click()
        
        # 드롭다운 선택
        from selenium.webdriver.support.ui import Select
        dropdown = Select(driver.find_element(By.ID, "dropdown_id"))
        dropdown.select_by_visible_text("옵션 텍스트")
        dropdown.select_by_value("옵션 값")
        dropdown.select_by_index(0)
        
        # 키보드 입력
        input_element.send_keys(Keys.ENTER)  # Enter 키
        input_element.send_keys(Keys.TAB)    # Tab 키
        input_element.send_keys(Keys.ARROW_DOWN)  # 화살표 키
        
        return True
    
    except Exception as e:
        print(f"요소 조작 오류: {e}")
        return False

# 요소 조작 예시 실행
# element_interaction_example()
```

### 대기 및 조건부 실행

```python
def wait_and_conditional_example():
    """대기 및 조건부 실행 예시"""
    
    try:
        # 페이지 접근
        driver.get("https://example.com")
        
        # 명시적 대기 (WebDriverWait 사용)
        wait = WebDriverWait(driver, 10)  # 최대 10초 대기
        
        # 요소가 클릭 가능할 때까지 대기
        clickable_element = wait.until(
            EC.element_to_be_clickable((By.ID, "button_id"))
        )
        clickable_element.click()
        
        # 요소가 보일 때까지 대기
        visible_element = wait.until(
            EC.visibility_of_element_located((By.ID, "element_id"))
        )
        
        # 요소가 사라질 때까지 대기
        wait.until(
            EC.invisibility_of_element_located((By.ID, "loading_id"))
        )
        
        # 텍스트가 나타날 때까지 대기
        wait.until(
            EC.text_to_be_present_in_element((By.ID, "status_id"), "완료")
        )
        
        # URL이 변경될 때까지 대기
        wait.until(
            EC.url_contains("success")
        )
        
        # 암시적 대기 (전역 설정)
        driver.implicitly_wait(10)  # 모든 요소 찾기 시 최대 10초 대기
        
        # 강제 대기
        time.sleep(2)  # 2초 대기
        
        return True
    
    except Exception as e:
        print(f"대기 및 조건부 실행 오류: {e}")
        return False

# 대기 및 조건부 실행 예시
# wait_and_conditional_example()
```

## 4. Selenium 고급 기능

### JavaScript 실행

```python
def javascript_execution_example():
    """JavaScript 실행 예시"""
    
    try:
        # 페이지 접근
        driver.get("https://example.com")
        
        # JavaScript 실행
        result = driver.execute_script("return document.title;")
        print(f"JavaScript 실행 결과: {result}")
        
        # JavaScript로 요소 스크롤
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # JavaScript로 요소 클릭
        element = driver.find_element(By.ID, "button_id")
        driver.execute_script("arguments[0].click();", element)
        
        # JavaScript로 요소에 텍스트 입력
        input_element = driver.find_element(By.ID, "input_id")
        driver.execute_script("arguments[0].value = 'JavaScript로 입력';", input_element)
        
        # JavaScript로 요소 스타일 변경
        driver.execute_script("arguments[0].style.border = '2px solid red';", element)
        
        # JavaScript로 요소 제거
        driver.execute_script("arguments[0].remove();", element)
        
        # JavaScript로 새 창 열기
        driver.execute_script("window.open('https://example.com', '_blank');")
        
        # JavaScript로 팝업 차단
        driver.execute_script("window.alert = function() {};")
        
        return True
    
    except Exception as e:
        print(f"JavaScript 실행 오류: {e}")
        return False

# JavaScript 실행 예시
# javascript_execution_example()
```

### 액션 체인 (ActionChains)

```python
def action_chains_example():
    """액션 체인 예시"""
    
    try:
        # 페이지 접근
        driver.get("https://example.com")
        
        # ActionChains 객체 생성
        actions = ActionChains(driver)
        
        # 요소 찾기
        element = driver.find_element(By.ID, "element_id")
        
        # 마우스 오버
        actions.move_to_element(element).perform()
        
        # 더블 클릭
        actions.double_click(element).perform()
        
        # 우클릭 (컨텍스트 메뉴)
        actions.context_click(element).perform()
        
        # 드래그 앤 드롭
        source_element = driver.find_element(By.ID, "source_id")
        target_element = driver.find_element(By.ID, "target_id")
        actions.drag_and_drop(source_element, target_element).perform()
        
        # 키보드 조합
        actions.key_down(Keys.CONTROL).send_keys('c').key_up(Keys.CONTROL).perform()
        
        # 복합 액션
        actions.move_to_element(element).click().send_keys("텍스트").perform()
        
        # 액션 체인 저장 및 실행
        actions.move_to_element(element).click().perform()
        
        return True
    
    except Exception as e:
        print(f"액션 체인 오류: {e}")
        return False

# 액션 체인 예시
# action_chains_example()
```

### 쿠키 및 세션 관리

```python
def cookie_session_example():
    """쿠키 및 세션 관리 예시"""
    
    try:
        # 페이지 접근
        driver.get("https://example.com")
        
        # 쿠키 추가
        driver.add_cookie({
            'name': 'cookie_name',
            'value': 'cookie_value',
            'domain': 'example.com'
        })
        
        # 쿠키 가져오기
        cookies = driver.get_cookies()
        print(f"쿠키 수: {len(cookies)}")
        
        for cookie in cookies:
            print(f"쿠키: {cookie['name']} = {cookie['value']}")
        
        # 특정 쿠키 가져오기
        specific_cookie = driver.get_cookie('cookie_name')
        print(f"특정 쿠키: {specific_cookie}")
        
        # 쿠키 삭제
        driver.delete_cookie('cookie_name')
        
        # 모든 쿠키 삭제
        driver.delete_all_cookies()
        
        # 로컬 스토리지 접근
        driver.execute_script("localStorage.setItem('key', 'value');")
        value = driver.execute_script("return localStorage.getItem('key');")
        print(f"로컬 스토리지 값: {value}")
        
        # 세션 스토리지 접근
        driver.execute_script("sessionStorage.setItem('key', 'value');")
        value = driver.execute_script("return sessionStorage.getItem('key');")
        print(f"세션 스토리지 값: {value}")
        
        return True
    
    except Exception as e:
        print(f"쿠키 및 세션 관리 오류: {e}")
        return False

# 쿠키 및 세션 관리 예시
# cookie_session_example()
```

## 5. Selenium 실무 적용 예시

### 로그인 자동화

```python
def automated_login():
    """로그인 자동화"""
    
    try:
        # 로그인 페이지 접근
        driver.get("https://example.com/login")
        
        # 로그인 폼 요소 찾기
        username_input = driver.find_element(By.ID, "username")
        password_input = driver.find_element(By.ID, "password")
        login_button = driver.find_element(By.ID, "login_button")
        
        # 로그인 정보 입력
        username_input.clear()
        username_input.send_keys("your_username")
        
        password_input.clear()
        password_input.send_keys("your_password")
        
        # 로그인 버튼 클릭
        login_button.click()
        
        # 로그인 성공 확인
        wait = WebDriverWait(driver, 10)
        success_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "success_message"))
        )
        
        print("로그인 성공!")
        return True
    
    except Exception as e:
        print(f"로그인 오류: {e}")
        return False

# 로그인 자동화 실행
# automated_login()
```

### 폼 자동 작성

```python
def automated_form_filling():
    """폼 자동 작성"""
    
    try:
        # 폼 페이지 접근
        driver.get("https://example.com/form")
        
        # 폼 요소들 찾기
        name_input = driver.find_element(By.ID, "name")
        email_input = driver.find_element(By.ID, "email")
        phone_input = driver.find_element(By.ID, "phone")
        
        # 체크박스들
        checkbox1 = driver.find_element(By.ID, "checkbox1")
        checkbox2 = driver.find_element(By.ID, "checkbox2")
        
        # 라디오 버튼
        radio_button = driver.find_element(By.ID, "radio_option1")
        
        # 드롭다운
        from selenium.webdriver.support.ui import Select
        dropdown = Select(driver.find_element(By.ID, "dropdown"))
        
        # 텍스트 영역
        textarea = driver.find_element(By.ID, "textarea")
        
        # 폼 작성
        name_input.send_keys("홍길동")
        email_input.send_keys("hong@example.com")
        phone_input.send_keys("010-1234-5678")
        
        # 체크박스 선택
        if not checkbox1.is_selected():
            checkbox1.click()
        if not checkbox2.is_selected():
            checkbox2.click()
        
        # 라디오 버튼 선택
        radio_button.click()
        
        # 드롭다운 선택
        dropdown.select_by_visible_text("옵션 1")
        
        # 텍스트 영역 작성
        textarea.send_keys("자동으로 작성된 텍스트입니다.")
        
        # 폼 제출
        submit_button = driver.find_element(By.ID, "submit_button")
        submit_button.click()
        
        # 제출 성공 확인
        wait = WebDriverWait(driver, 10)
        success_message = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "success_message"))
        )
        
        print("폼 제출 성공!")
        return True
    
    except Exception as e:
        print(f"폼 작성 오류: {e}")
        return False

# 폼 자동 작성 실행
# automated_form_filling()
```

### 데이터 수집

```python
def data_collection_example():
    """데이터 수집 예시"""
    
    try:
        # 데이터 페이지 접근
        driver.get("https://example.com/data")
        
        # 데이터 요소들 찾기
        data_elements = driver.find_elements(By.CLASS_NAME, "data-item")
        
        collected_data = []
        
        for element in data_elements:
            try:
                # 각 데이터 항목에서 정보 추출
                title = element.find_element(By.CLASS_NAME, "title").text
                price = element.find_element(By.CLASS_NAME, "price").text
                description = element.find_element(By.CLASS_NAME, "description").text
                link = element.find_element(By.TAG_NAME, "a").get_attribute("href")
                
                data_item = {
                    "title": title,
                    "price": price,
                    "description": description,
                    "link": link
                }
                
                collected_data.append(data_item)
                
            except Exception as e:
                print(f"데이터 항목 추출 오류: {e}")
                continue
        
        # 데이터를 DataFrame으로 변환
        df = pd.DataFrame(collected_data)
        
        # CSV 파일로 저장
        df.to_csv("collected_data.csv", index=False, encoding="utf-8-sig")
        
        print(f"수집된 데이터 수: {len(collected_data)}")
        print("데이터가 'collected_data.csv'에 저장되었습니다.")
        
        return collected_data
    
    except Exception as e:
        print(f"데이터 수집 오류: {e}")
        return []

# 데이터 수집 실행
# data_collection_example()
```

### 페이지네이션 처리

```python
def pagination_handling():
    """페이지네이션 처리"""
    
    try:
        # 첫 페이지 접근
        driver.get("https://example.com/list")
        
        all_data = []
        page_num = 1
        
        while True:
            print(f"페이지 {page_num} 처리 중...")
            
            # 현재 페이지 데이터 수집
            data_elements = driver.find_elements(By.CLASS_NAME, "data-item")
            
            for element in data_elements:
                try:
                    title = element.find_element(By.CLASS_NAME, "title").text
                    price = element.find_element(By.CLASS_NAME, "price").text
                    
                    data_item = {
                        "page": page_num,
                        "title": title,
                        "price": price
                    }
                    
                    all_data.append(data_item)
                    
                except Exception as e:
                    print(f"데이터 항목 추출 오류: {e}")
                    continue
            
            # 다음 페이지 버튼 찾기
            try:
                next_button = driver.find_element(By.CLASS_NAME, "next-page")
                
                # 버튼이 비활성화되어 있으면 종료
                if not next_button.is_enabled():
                    break
                
                # 다음 페이지로 이동
                next_button.click()
                
                # 페이지 로딩 대기
                wait = WebDriverWait(driver, 10)
                wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "data-item"))
                )
                
                page_num += 1
                
            except Exception as e:
                print(f"다음 페이지 버튼 찾기 오류: {e}")
                break
        
        # 데이터를 DataFrame으로 변환
        df = pd.DataFrame(all_data)
        
        # CSV 파일로 저장
        df.to_csv("paginated_data.csv", index=False, encoding="utf-8-sig")
        
        print(f"총 수집된 데이터 수: {len(all_data)}")
        print("데이터가 'paginated_data.csv'에 저장되었습니다.")
        
        return all_data
    
    except Exception as e:
        print(f"페이지네이션 처리 오류: {e}")
        return []

# 페이지네이션 처리 실행
# pagination_handling()
```

## 6. Selenium 최적화 및 모범 사례

### 성능 최적화

```python
def optimize_selenium_performance():
    """Selenium 성능 최적화"""
    
    # Chrome 옵션 최적화
    options = Options()
    
    # 헤드리스 모드
    options.add_argument('--headless')
    
    # 성능 최적화 옵션들
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-plugins')
    options.add_argument('--disable-images')  # 이미지 로딩 비활성화
    options.add_argument('--disable-javascript')  # JavaScript 비활성화 (필요한 경우)
    
    # 메모리 사용량 최적화
    options.add_argument('--memory-pressure-off')
    options.add_argument('--max_old_space_size=4096')
    
    # 네트워크 최적화
    options.add_argument('--aggressive-cache-discard')
    options.add_argument('--disable-background-timer-throttling')
    
    # 창 크기 최적화
    options.add_argument('--window-size=1920,1080')
    
    return options

# 최적화된 WebDriver 생성
def create_optimized_driver():
    """최적화된 WebDriver 생성"""
    options = optimize_selenium_performance()
    
    driver = webdriver.Chrome(
        service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
        options=options
    )
    
    # 암시적 대기 시간 설정
    driver.implicitly_wait(5)
    
    # 페이지 로드 타임아웃 설정
    driver.set_page_load_timeout(30)
    
    return driver

# 최적화된 WebDriver 사용
# optimized_driver = create_optimized_driver()
```

### 에러 처리 및 재시도

```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """실패 시 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"최대 재시도 횟수 초과: {e}")
                        raise
                    print(f"시도 {attempt + 1} 실패, {delay}초 후 재시도: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# 재시도 기능이 있는 Selenium 함수
@retry_on_failure(max_retries=3, delay=2)
def robust_selenium_operation():
    """견고한 Selenium 작업"""
    
    try:
        # WebDriver 생성
        driver = create_optimized_driver()
        
        # 페이지 접근
        driver.get("https://example.com")
        
        # 요소 찾기 및 조작
        element = driver.find_element(By.ID, "element_id")
        element.click()
        
        # 결과 반환
        result = driver.find_element(By.CLASS_NAME, "result").text
        
        # WebDriver 종료
        driver.quit()
        
        return result
    
    except Exception as e:
        print(f"Selenium 작업 오류: {e}")
        if 'driver' in locals():
            driver.quit()
        raise

# 견고한 Selenium 작업 실행
# result = robust_selenium_operation()
```

### 스크린샷 및 디버깅

```python
def screenshot_and_debugging():
    """스크린샷 및 디버깅"""
    
    try:
        # WebDriver 생성
        driver = create_optimized_driver()
        
        # 페이지 접근
        driver.get("https://example.com")
        
        # 스크린샷 촬영
        screenshot_path = "screenshot.png"
        driver.save_screenshot(screenshot_path)
        print(f"스크린샷 저장: {screenshot_path}")
        
        # 특정 요소 스크린샷
        element = driver.find_element(By.ID, "element_id")
        element.screenshot("element_screenshot.png")
        print("요소 스크린샷 저장: element_screenshot.png")
        
        # 페이지 소스 저장
        page_source = driver.page_source
        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(page_source)
        print("페이지 소스 저장: page_source.html")
        
        # 콘솔 로그 가져오기
        logs = driver.get_log('browser')
        for log in logs:
            print(f"브라우저 로그: {log}")
        
        # 네트워크 로그 가져오기
        performance_logs = driver.get_log('performance')
        for log in performance_logs:
            print(f"성능 로그: {log}")
        
        # WebDriver 종료
        driver.quit()
        
        return True
    
    except Exception as e:
        print(f"스크린샷 및 디버깅 오류: {e}")
        if 'driver' in locals():
            driver.quit()
        return False

# 스크린샷 및 디버깅 실행
# screenshot_and_debugging()
```

## 7. 실무 활용 팁

### 1. 동적 콘텐츠 처리

```python
def handle_dynamic_content():
    """동적 콘텐츠 처리"""
    
    driver = create_optimized_driver()
    
    try:
        # 페이지 접근
        driver.get("https://spa.example.com")
        
        # 동적 콘텐츠 로딩 대기
        wait = WebDriverWait(driver, 20)
        
        # 특정 요소가 나타날 때까지 대기
        dynamic_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "dynamic-content"))
        )
        
        # 스크롤하여 더 많은 콘텐츠 로드
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # 무한 스크롤 처리
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # 스크롤 다운
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # 새로운 높이 계산
            new_height = driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                break
            
            last_height = new_height
        
        # 동적 콘텐츠 수집
        dynamic_elements = driver.find_elements(By.CLASS_NAME, "dynamic-content")
        print(f"동적 콘텐츠 수: {len(dynamic_elements)}")
        
        return dynamic_elements
    
    finally:
        driver.quit()
```

### 2. 복잡한 폼 처리

```python
def handle_complex_forms():
    """복잡한 폼 처리"""
    
    driver = create_optimized_driver()
    
    try:
        # 폼 페이지 접근
        driver.get("https://example.com/complex-form")
        
        # 파일 업로드
        file_input = driver.find_element(By.ID, "file-upload")
        file_input.send_keys("/path/to/file.pdf")
        
        # 날짜 선택기
        date_input = driver.find_element(By.ID, "date-picker")
        date_input.click()
        
        # 달력에서 날짜 선택
        calendar = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "calendar"))
        )
        
        target_date = calendar.find_element(By.XPATH, "//td[text()='15']")
        target_date.click()
        
        # 자동완성 입력
        autocomplete_input = driver.find_element(By.ID, "autocomplete")
        autocomplete_input.send_keys("검색어")
        
        # 자동완성 옵션 선택
        wait = WebDriverWait(driver, 10)
        option = wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, "autocomplete-option"))
        )
        option.click()
        
        # 슬라이더 조작
        slider = driver.find_element(By.ID, "slider")
        actions = ActionChains(driver)
        actions.drag_and_drop_by_offset(slider, 100, 0).perform()
        
        return True
    
    finally:
        driver.quit()
```

### 3. 다중 창 처리

```python
def handle_multiple_windows():
    """다중 창 처리"""
    
    driver = create_optimized_driver()
    
    try:
        # 첫 번째 창
        driver.get("https://example.com")
        main_window = driver.current_window_handle
        
        # 새 창 열기
        driver.execute_script("window.open('https://example2.com', '_blank');")
        
        # 모든 창 핸들 가져오기
        all_windows = driver.window_handles
        
        for window in all_windows:
            if window != main_window:
                # 새 창으로 전환
                driver.switch_to.window(window)
                
                # 새 창에서 작업
                title = driver.find_element(By.TAG_NAME, "h1").text
                print(f"새 창 제목: {title}")
                
                # 새 창 닫기
                driver.close()
        
        # 원래 창으로 돌아가기
        driver.switch_to.window(main_window)
        
        return True
    
    finally:
        driver.quit()
```

### 4. 모바일 시뮬레이션

```python
def mobile_simulation():
    """모바일 시뮬레이션"""
    
    # 모바일 옵션 설정
    mobile_emulation = {
        "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 2.0},
        "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1"
    }
    
    options = Options()
    options.add_experimental_option("mobileEmulation", mobile_emulation)
    
    driver = webdriver.Chrome(
        service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
        options=options
    )
    
    try:
        # 모바일 페이지 접근
        driver.get("https://m.example.com")
        
        # 모바일 전용 요소 찾기
        mobile_menu = driver.find_element(By.CLASS_NAME, "mobile-menu")
        mobile_menu.click()
        
        # 터치 제스처 시뮬레이션
        actions = ActionChains(driver)
        actions.move_to_element(mobile_menu).click_and_hold().move_by_offset(100, 0).release().perform()
        
        return True
    
    finally:
        driver.quit()
```

## 8. 주의사항 및 모범 사례

### 법적 고려사항

- **robots.txt 확인**: 웹사이트의 robots.txt 파일 확인
- **이용약관 준수**: 웹사이트 이용약관 준수
- **저작권 존중**: 저작권이 있는 콘텐츠 사용 시 주의
- **개인정보 보호**: 개인정보 수집 시 관련 법규 준수

### 기술적 고려사항

- **요청 제한**: 서버에 과부하를 주지 않도록 요청 제한
- **User-Agent 설정**: 적절한 User-Agent 설정
- **에러 처리**: 적절한 에러 처리 및 재시도 로직
- **리소스 관리**: WebDriver 리소스 적절한 관리

### 성능 최적화

- **헤드리스 모드**: 불필요한 UI 렌더링 비활성화
- **이미지 로딩**: 필요하지 않은 경우 이미지 로딩 비활성화
- **JavaScript**: 필요하지 않은 경우 JavaScript 비활성화
- **캐싱**: 중복 요청 방지를 위한 캐싱

### 실무 팁

1. **적절한 대기 시간 설정**: 너무 짧으면 요소를 찾지 못하고, 너무 길면 성능 저하
2. **요소 선택자 최적화**: ID > CSS 선택자 > XPath 순으로 우선순위
3. **리소스 정리**: 작업 완료 후 반드시 driver.quit() 호출
4. **로깅 및 모니터링**: 스크래핑 과정을 적절히 로깅하고 모니터링

## 마무리

Selenium은 웹 브라우저 자동화를 위한 강력한 도구입니다. JavaScript로 생성되는 동적 콘텐츠를 처리할 수 있고, 실제 사용자 행동을 시뮬레이션할 수 있습니다. 

적절한 설정과 최적화를 통해 안정적이고 효율적인 웹 자동화 시스템을 구축할 수 있으며, 실무에서는 동적 콘텐츠 처리, 복잡한 폼 처리, 다중 창 처리 등의 고급 기능을 활용할 수 있습니다. 

다만 법적, 윤리적 고려사항을 준수하여 사용해야 하며, 성능 최적화와 에러 처리를 통해 안정적인 시스템을 구축해야 합니다.
