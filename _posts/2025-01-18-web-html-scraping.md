---
title: HTML 스크래핑 - 웹 데이터 수집의 핵심
categories:
- 1.TIL
- 1-5.WEB
tags:
- html
- 스크래핑
- beautifulsoup
- requests
- selenium
- 웹크롤링
toc: true
date: 2023-10-13 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# HTML 스크래핑 - 웹 데이터 수집의 핵심

## 개요

HTML 스크래핑은 웹 페이지의 HTML 구조를 분석하여 원하는 데이터를 추출하는 기술입니다:

- **웹 크롤링**: 웹 사이트를 자동으로 탐색하고 데이터를 수집하는 과정
- **데이터 추출**: HTML 태그, 속성, 텍스트에서 필요한 정보 추출
- **자동화**: 반복적인 데이터 수집 작업 자동화

## 1. HTML 스크래핑 개요

### 주요 특징

- **구조 분석**: HTML DOM 구조를 분석하여 데이터 추출
- **선택자 활용**: CSS 선택자나 XPath를 사용한 요소 선택
- **데이터 변환**: 추출된 데이터를 원하는 형태로 변환
- **확장성**: 대량의 데이터 수집 가능

### 장점

- **효율성**: 수동 데이터 수집보다 빠르고 정확
- **자동화**: 24시간 무인 데이터 수집
- **정확성**: 일관된 데이터 추출
- **비용 절약**: 수동 작업 대비 비용 절약

## 2. HTML 스크래핑 기본 도구

### BeautifulSoup 설치 및 기본 사용법

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import json

# BeautifulSoup 기본 사용법
def basic_beautifulsoup_example():
    """BeautifulSoup 기본 사용법 예시"""
    
    # 샘플 HTML
    html_content = """
    <html>
        <head>
            <title>웹 스크래핑 예시</title>
        </head>
        <body>
            <div class="container">
                <h1>제품 목록</h1>
                <div class="product">
                    <h2>노트북</h2>
                    <p class="price">1,200,000원</p>
                    <p class="description">고성능 게이밍 노트북</p>
                </div>
                <div class="product">
                    <h2>스마트폰</h2>
                    <p class="price">800,000원</p>
                    <p class="description">최신 스마트폰</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 제목 추출
    title = soup.find('title').text
    print(f"페이지 제목: {title}")
    
    # 제품 목록 추출
    products = soup.find_all('div', class_='product')
    
    product_list = []
    for product in products:
        name = product.find('h2').text
        price = product.find('p', class_='price').text
        description = product.find('p', class_='description').text
        
        product_info = {
            'name': name,
            'price': price,
            'description': description
        }
        product_list.append(product_info)
    
    return product_list

# 기본 예시 실행
products = basic_beautifulsoup_example()
for product in products:
    print(f"제품: {product['name']}, 가격: {product['price']}")
```

### requests와 BeautifulSoup 조합

```python
def scrape_web_page(url):
    """웹 페이지 스크래핑"""
    
    # User-Agent 설정 (봇 차단 방지)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # 웹 페이지 요청
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # HTML 파싱
        soup = BeautifulSoup(response.content, 'html.parser')
        
        return soup
    
    except requests.exceptions.RequestException as e:
        print(f"웹 페이지 요청 오류: {e}")
        return None

# 웹 페이지 스크래핑 예시
def scrape_example_website():
    """예시 웹사이트 스크래핑"""
    
    # 실제 웹사이트 URL (예시)
    url = "https://example.com"
    
    soup = scrape_web_page(url)
    
    if soup:
        # 제목 추출
        title = soup.find('title')
        if title:
            print(f"페이지 제목: {title.text}")
        
        # 모든 링크 추출
        links = soup.find_all('a', href=True)
        print(f"링크 수: {len(links)}")
        
        # 모든 이미지 추출
        images = soup.find_all('img', src=True)
        print(f"이미지 수: {len(images)}")
        
        return soup
    
    return None

# 웹 페이지 스크래핑 실행
# scrape_example_website()
```

### CSS 선택자 활용

```python
def advanced_css_selectors():
    """고급 CSS 선택자 활용"""
    
    # 샘플 HTML
    html_content = """
    <div class="news-container">
        <article class="news-item featured">
            <h2 class="title">주요 뉴스</h2>
            <p class="summary">요약 내용</p>
            <span class="date">2023-12-01</span>
            <a href="/news/1" class="read-more">더보기</a>
        </article>
        <article class="news-item">
            <h2 class="title">일반 뉴스 1</h2>
            <p class="summary">요약 내용 1</p>
            <span class="date">2023-12-01</span>
            <a href="/news/2" class="read-more">더보기</a>
        </article>
        <article class="news-item">
            <h2 class="title">일반 뉴스 2</h2>
            <p class="summary">요약 내용 2</p>
            <span class="date">2023-12-01</span>
            <a href="/read-more">더보기</a>
        </article>
    </div>
    """
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 다양한 CSS 선택자 사용
    print("=== CSS 선택자 예시 ===")
    
    # 클래스 선택자
    featured_news = soup.select('.news-item.featured')
    print(f"주요 뉴스 수: {len(featured_news)}")
    
    # 자식 선택자
    titles = soup.select('.news-item > .title')
    print(f"뉴스 제목 수: {len(titles)}")
    
    # 속성 선택자
    read_more_links = soup.select('a[class="read-more"]')
    print(f"더보기 링크 수: {len(read_more_links)}")
    
    # 복합 선택자
    news_items = soup.select('.news-container .news-item')
    print(f"전체 뉴스 수: {len(news_items)}")
    
    # 뉴스 데이터 추출
    news_data = []
    for item in news_items:
        title = item.select_one('.title').text if item.select_one('.title') else ""
        summary = item.select_one('.summary').text if item.select_one('.summary') else ""
        date = item.select_one('.date').text if item.select_one('.date') else ""
        link = item.select_one('a[class="read-more"]')
        link_url = link['href'] if link else ""
        
        news_info = {
            'title': title,
            'summary': summary,
            'date': date,
            'link': link_url
        }
        news_data.append(news_info)
    
    return news_data

# CSS 선택자 예시 실행
news_data = advanced_css_selectors()
for news in news_data:
    print(f"제목: {news['title']}, 링크: {news['link']}")
```

## 3. 실무 HTML 스크래핑 예시

### 뉴스 사이트 스크래핑

```python
def scrape_news_website():
    """뉴스 사이트 스크래핑"""
    
    # 실제 뉴스 사이트 URL (예시)
    url = "https://news.example.com"
    
    soup = scrape_web_page(url)
    
    if not soup:
        return []
    
    # 뉴스 기사 추출
    news_articles = []
    
    # 뉴스 기사 선택자 (사이트마다 다름)
    articles = soup.select('.news-article, .article-item, .news-item')
    
    for article in articles:
        try:
            # 제목 추출
            title_element = article.select_one('.title, .headline, h2, h3')
            title = title_element.text.strip() if title_element else ""
            
            # 링크 추출
            link_element = article.select_one('a')
            link = link_element['href'] if link_element else ""
            
            # 날짜 추출
            date_element = article.select_one('.date, .published, .time')
            date = date_element.text.strip() if date_element else ""
            
            # 요약 추출
            summary_element = article.select_one('.summary, .excerpt, .description')
            summary = summary_element.text.strip() if summary_element else ""
            
            # 이미지 추출
            img_element = article.select_one('img')
            image_url = img_element['src'] if img_element else ""
            
            if title:  # 제목이 있는 경우만 추가
                news_info = {
                    'title': title,
                    'link': link,
                    'date': date,
                    'summary': summary,
                    'image_url': image_url
                }
                news_articles.append(news_info)
        
        except Exception as e:
            print(f"기사 추출 오류: {e}")
            continue
    
    return news_articles

# 뉴스 스크래핑 실행
def run_news_scraping():
    """뉴스 스크래핑 실행"""
    news_articles = scrape_news_website()
    
    print(f"추출된 뉴스 기사 수: {len(news_articles)}")
    
    if news_articles:
        # DataFrame으로 변환
        df_news = pd.DataFrame(news_articles)
        print("\n=== 뉴스 기사 목록 ===")
        print(df_news.head())
        
        # CSV 파일로 저장
        df_news.to_csv('news_articles.csv', index=False, encoding='utf-8-sig')
        print("\n뉴스 기사가 'news_articles.csv'에 저장되었습니다.")
    
    return news_articles

# 뉴스 스크래핑 실행
# run_news_scraping()
```

### 전자상거래 사이트 스크래핑

```python
def scrape_ecommerce_website():
    """전자상거래 사이트 스크래핑"""
    
    # 실제 전자상거래 사이트 URL (예시)
    url = "https://shop.example.com"
    
    soup = scrape_web_page(url)
    
    if not soup:
        return []
    
    # 상품 정보 추출
    products = []
    
    # 상품 선택자 (사이트마다 다름)
    product_elements = soup.select('.product, .item, .goods')
    
    for product in product_elements:
        try:
            # 상품명 추출
            name_element = product.select_one('.name, .title, .product-name, h3')
            name = name_element.text.strip() if name_element else ""
            
            # 가격 추출
            price_element = product.select_one('.price, .cost, .amount')
            price_text = price_element.text.strip() if price_element else ""
            
            # 가격에서 숫자만 추출
            price = re.sub(r'[^\d,]', '', price_text)
            
            # 이미지 추출
            img_element = product.select_one('img')
            image_url = img_element['src'] if img_element else ""
            
            # 링크 추출
            link_element = product.select_one('a')
            product_url = link_element['href'] if link_element else ""
            
            # 평점 추출
            rating_element = product.select_one('.rating, .score, .stars')
            rating = rating_element.text.strip() if rating_element else ""
            
            # 리뷰 수 추출
            review_element = product.select_one('.review-count, .reviews')
            review_count = review_element.text.strip() if review_element else ""
            
            if name:  # 상품명이 있는 경우만 추가
                product_info = {
                    'name': name,
                    'price': price,
                    'image_url': image_url,
                    'product_url': product_url,
                    'rating': rating,
                    'review_count': review_count
                }
                products.append(product_info)
        
        except Exception as e:
            print(f"상품 추출 오류: {e}")
            continue
    
    return products

# 전자상거래 스크래핑 실행
def run_ecommerce_scraping():
    """전자상거래 스크래핑 실행"""
    products = scrape_ecommerce_website()
    
    print(f"추출된 상품 수: {len(products)}")
    
    if products:
        # DataFrame으로 변환
        df_products = pd.DataFrame(products)
        print("\n=== 상품 목록 ===")
        print(df_products.head())
        
        # CSV 파일로 저장
        df_products.to_csv('products.csv', index=False, encoding='utf-8-sig')
        print("\n상품 정보가 'products.csv'에 저장되었습니다.")
        
        # 가격 분석
        if 'price' in df_products.columns:
            # 가격 데이터 정리
            df_products['price_numeric'] = df_products['price'].str.replace(',', '').str.replace('원', '')
            df_products['price_numeric'] = pd.to_numeric(df_products['price_numeric'], errors='coerce')
            
            # 가격 통계
            price_stats = df_products['price_numeric'].describe()
            print("\n=== 가격 통계 ===")
            print(price_stats)
    
    return products

# 전자상거래 스크래핑 실행
# run_ecommerce_scraping()
```

### 부동산 사이트 스크래핑

```python
def scrape_real_estate_website():
    """부동산 사이트 스크래핑"""
    
    # 실제 부동산 사이트 URL (예시)
    url = "https://realestate.example.com"
    
    soup = scrape_web_page(url)
    
    if not soup:
        return []
    
    # 부동산 정보 추출
    properties = []
    
    # 부동산 선택자 (사이트마다 다름)
    property_elements = soup.select('.property, .house, .apartment')
    
    for property in property_elements:
        try:
            # 매물명 추출
            title_element = property.select_one('.title, .name, .property-name')
            title = title_element.text.strip() if title_element else ""
            
            # 가격 추출
            price_element = property.select_one('.price, .cost, .amount')
            price = price_element.text.strip() if price_element else ""
            
            # 위치 추출
            location_element = property.select_one('.location, .address, .area')
            location = location_element.text.strip() if location_element else ""
            
            # 면적 추출
            area_element = property.select_one('.area, .size, .square')
            area = area_element.text.strip() if area_element else ""
            
            # 방 수 추출
            rooms_element = property.select_one('.rooms, .bedrooms, .room-count')
            rooms = rooms_element.text.strip() if rooms_element else ""
            
            # 이미지 추출
            img_element = property.select_one('img')
            image_url = img_element['src'] if img_element else ""
            
            # 링크 추출
            link_element = property.select_one('a')
            property_url = link_element['href'] if link_element else ""
            
            if title:  # 매물명이 있는 경우만 추가
                property_info = {
                    'title': title,
                    'price': price,
                    'location': location,
                    'area': area,
                    'rooms': rooms,
                    'image_url': image_url,
                    'property_url': property_url
                }
                properties.append(property_info)
        
        except Exception as e:
            print(f"부동산 정보 추출 오류: {e}")
            continue
    
    return properties

# 부동산 스크래핑 실행
def run_real_estate_scraping():
    """부동산 스크래핑 실행"""
    properties = scrape_real_estate_website()
    
    print(f"추출된 부동산 수: {len(properties)}")
    
    if properties:
        # DataFrame으로 변환
        df_properties = pd.DataFrame(properties)
        print("\n=== 부동산 목록 ===")
        print(df_properties.head())
        
        # CSV 파일로 저장
        df_properties.to_csv('properties.csv', index=False, encoding='utf-8-sig')
        print("\n부동산 정보가 'properties.csv'에 저장되었습니다.")
        
        # 지역별 분석
        if 'location' in df_properties.columns:
            location_counts = df_properties['location'].value_counts()
            print("\n=== 지역별 매물 수 ===")
            print(location_counts.head(10))
    
    return properties

# 부동산 스크래핑 실행
# run_real_estate_scraping()
```

## 4. 고급 HTML 스크래핑 기법

### 동적 콘텐츠 처리 (Selenium)

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def scrape_dynamic_content(url):
    """동적 콘텐츠 스크래핑 (Selenium 사용)"""
    
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 브라우저 창 숨기기
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    try:
        # WebDriver 생성
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # 페이지 로딩 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # JavaScript 실행 대기
        time.sleep(3)
        
        # 페이지 소스 가져오기
        page_source = driver.page_source
        
        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(page_source, 'html.parser')
        
        driver.quit()
        
        return soup
    
    except Exception as e:
        print(f"동적 콘텐츠 스크래핑 오류: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

# 동적 콘텐츠 스크래핑 예시
def scrape_spa_website():
    """SPA 웹사이트 스크래핑"""
    
    url = "https://spa.example.com"
    
    soup = scrape_dynamic_content(url)
    
    if soup:
        # 동적으로 로드된 콘텐츠 추출
        dynamic_elements = soup.select('.dynamic-content, .loaded-content')
        
        print(f"동적 콘텐츠 요소 수: {len(dynamic_elements)}")
        
        for element in dynamic_elements:
            print(f"콘텐츠: {element.text.strip()}")
    
    return soup

# SPA 웹사이트 스크래핑 실행
# scrape_spa_website()
```

### 폼 데이터 처리

```python
def handle_form_submission():
    """폼 데이터 처리"""
    
    # 로그인 폼 예시
    login_data = {
        'username': 'your_username',
        'password': 'your_password'
    }
    
    # 세션 생성
    session = requests.Session()
    
    # 로그인 페이지 접근
    login_url = "https://example.com/login"
    response = session.get(login_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # CSRF 토큰 추출 (있는 경우)
        csrf_token = soup.find('input', {'name': 'csrf_token'})
        if csrf_token:
            login_data['csrf_token'] = csrf_token['value']
        
        # 로그인 요청
        login_response = session.post(login_url, data=login_data)
        
        if login_response.status_code == 200:
            print("로그인 성공")
            
            # 로그인 후 페이지 스크래핑
            protected_url = "https://example.com/protected"
            protected_response = session.get(protected_url)
            
            if protected_response.status_code == 200:
                protected_soup = BeautifulSoup(protected_response.content, 'html.parser')
                return protected_soup
    
    return None

# 폼 데이터 처리 실행
# handle_form_submission()
```

### 이미지 및 파일 다운로드

```python
import os
from urllib.parse import urljoin, urlparse

def download_images(soup, base_url, download_dir="images"):
    """이미지 다운로드"""
    
    # 다운로드 디렉토리 생성
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # 이미지 요소 찾기
    images = soup.find_all('img', src=True)
    
    downloaded_images = []
    
    for img in images:
        try:
            # 이미지 URL 생성
            img_url = urljoin(base_url, img['src'])
            
            # 이미지 파일명 생성
            img_filename = os.path.basename(urlparse(img_url).path)
            if not img_filename:
                img_filename = f"image_{len(downloaded_images)}.jpg"
            
            # 이미지 다운로드
            img_response = requests.get(img_url, timeout=10)
            img_response.raise_for_status()
            
            # 파일 저장
            img_path = os.path.join(download_dir, img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_response.content)
            
            downloaded_images.append({
                'url': img_url,
                'filename': img_filename,
                'path': img_path
            })
            
            print(f"이미지 다운로드 완료: {img_filename}")
        
        except Exception as e:
            print(f"이미지 다운로드 오류: {e}")
            continue
    
    return downloaded_images

# 이미지 다운로드 예시
def download_website_images():
    """웹사이트 이미지 다운로드"""
    
    url = "https://example.com"
    soup = scrape_web_page(url)
    
    if soup:
        images = download_images(soup, url)
        print(f"다운로드된 이미지 수: {len(images)}")
        
        return images
    
    return []

# 이미지 다운로드 실행
# download_website_images()
```

## 5. 스크래핑 최적화 및 모범 사례

### 요청 제한 및 지연

```python
import time
from functools import wraps

def rate_limit(seconds):
    """요청 제한 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(seconds)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 요청 제한 적용
@rate_limit(1)  # 1초 대기
def limited_scrape(url):
    """제한된 스크래핑"""
    return scrape_web_page(url)

# 여러 페이지 스크래핑
def scrape_multiple_pages(urls):
    """여러 페이지 스크래핑"""
    
    all_data = []
    
    for i, url in enumerate(urls):
        print(f"스크래핑 중: {url} ({i+1}/{len(urls)})")
        
        soup = limited_scrape(url)
        
        if soup:
            # 페이지별 데이터 추출 로직
            page_data = extract_page_data(soup)
            all_data.extend(page_data)
        
        # 진행률 표시
        progress = (i + 1) / len(urls) * 100
        print(f"진행률: {progress:.1f}%")
    
    return all_data

def extract_page_data(soup):
    """페이지 데이터 추출 (구현 필요)"""
    # 실제 데이터 추출 로직 구현
    return []
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

# 재시도 기능이 있는 스크래핑
@retry_on_failure(max_retries=3, delay=2)
def robust_scrape(url):
    """견고한 스크래핑"""
    return scrape_web_page(url)

# 견고한 스크래핑 실행
def run_robust_scraping():
    """견고한 스크래핑 실행"""
    
    urls = [
        "https://example1.com",
        "https://example2.com",
        "https://example3.com"
    ]
    
    results = []
    
    for url in urls:
        print(f"스크래핑 중: {url}")
        
        soup = robust_scrape(url)
        
        if soup:
            # 데이터 추출
            data = extract_page_data(soup)
            results.extend(data)
            print(f"성공: {len(data)}개 항목 추출")
        else:
            print(f"실패: {url}")
    
    return results

# 견고한 스크래핑 실행
# run_robust_scraping()
```

### 데이터 검증 및 정리

```python
import re
from datetime import datetime

def clean_text(text):
    """텍스트 정리"""
    if not text:
        return ""
    
    # 공백 제거
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 특수 문자 제거
    text = re.sub(r'[^\w\s가-힣]', '', text)
    
    return text

def validate_data(data):
    """데이터 검증"""
    
    required_fields = ['title', 'content']
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    # 텍스트 길이 검증
    if len(data['title']) < 5 or len(data['content']) < 10:
        return False
    
    return True

def process_scraped_data(raw_data):
    """스크래핑된 데이터 처리"""
    
    processed_data = []
    
    for item in raw_data:
        # 텍스트 정리
        cleaned_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                cleaned_item[key] = clean_text(value)
            else:
                cleaned_item[key] = value
        
        # 데이터 검증
        if validate_data(cleaned_item):
            # 추가 처리
            cleaned_item['scraped_at'] = datetime.now().isoformat()
            processed_data.append(cleaned_item)
        else:
            print(f"유효하지 않은 데이터: {item}")
    
    return processed_data

# 데이터 처리 예시
def run_data_processing():
    """데이터 처리 실행"""
    
    # 원시 데이터 (예시)
    raw_data = [
        {
            'title': '  제목 1  ',
            'content': '내용 1입니다.',
            'date': '2023-12-01'
        },
        {
            'title': '제목 2',
            'content': '내용 2입니다.',
            'date': '2023-12-02'
        }
    ]
    
    # 데이터 처리
    processed_data = process_scraped_data(raw_data)
    
    print(f"처리된 데이터 수: {len(processed_data)}")
    
    # DataFrame으로 변환
    df = pd.DataFrame(processed_data)
    print(df)
    
    return processed_data

# 데이터 처리 실행
# run_data_processing()
```

## 6. 실무 활용 팁

### 1. 병렬 처리

```python
import concurrent.futures
import threading

def parallel_scraping(urls, max_workers=5):
    """병렬 스크래핑"""
    
    def scrape_single_url(url):
        """단일 URL 스크래핑"""
        soup = scrape_web_page(url)
        if soup:
            return extract_page_data(soup)
        return []
    
    all_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 URL에 대해 스크래핑 작업 제출
        future_to_url = {executor.submit(scrape_single_url, url): url for url in urls}
        
        # 완료된 작업 결과 수집
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                all_data.extend(data)
                print(f"완료: {url} - {len(data)}개 항목")
            except Exception as e:
                print(f"오류: {url} - {e}")
    
    return all_data
```

### 2. 데이터베이스 저장

```python
import sqlite3
import json

def save_to_database(data, db_name="scraped_data.db"):
    """데이터베이스에 저장"""
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraped_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            url TEXT,
            scraped_at TIMESTAMP,
            data_json TEXT
        )
    ''')
    
    # 데이터 삽입
    for item in data:
        cursor.execute('''
            INSERT INTO scraped_data (title, content, url, scraped_at, data_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            item.get('title', ''),
            item.get('content', ''),
            item.get('url', ''),
            datetime.now(),
            json.dumps(item, ensure_ascii=False)
        ))
    
    conn.commit()
    conn.close()
    
    print(f"데이터가 {db_name}에 저장되었습니다.")
```

### 3. 모니터링 및 로깅

```python
import logging
from datetime import datetime

def setup_logging():
    """로깅 설정"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scraping.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def monitor_scraping():
    """스크래핑 모니터링"""
    
    logger = setup_logging()
    
    start_time = datetime.now()
    logger.info("스크래핑 시작")
    
    try:
        # 스크래핑 로직 실행
        results = run_robust_scraping()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"스크래핑 완료: {len(results)}개 항목, 소요시간: {duration}")
        
        return results
    
    except Exception as e:
        logger.error(f"스크래핑 오류: {e}")
        return []
```

## 7. 주의사항 및 모범 사례

### 법적 고려사항

- **robots.txt 확인**: 웹사이트의 robots.txt 파일 확인
- **이용약관 준수**: 웹사이트 이용약관 준수
- **저작권 존중**: 저작권이 있는 콘텐츠 사용 시 주의
- **개인정보 보호**: 개인정보 수집 시 관련 법규 준수

### 기술적 고려사항

- **요청 제한**: 서버에 과부하를 주지 않도록 요청 제한
- **User-Agent 설정**: 적절한 User-Agent 설정
- **에러 처리**: 적절한 에러 처리 및 재시도 로직
- **데이터 검증**: 스크래핑된 데이터의 유효성 검증

### 성능 최적화

- **병렬 처리**: 여러 페이지를 병렬로 스크래핑
- **캐싱**: 중복 요청 방지를 위한 캐싱
- **데이터베이스**: 대량 데이터 저장을 위한 데이터베이스 사용
- **모니터링**: 스크래핑 과정 모니터링

## 마무리

HTML 스크래핑은 웹에서 데이터를 수집하는 강력한 기술입니다. BeautifulSoup, requests, Selenium 등의 도구를 활용하여 다양한 웹사이트에서 데이터를 추출할 수 있습니다. 

적절한 요청 제한, 에러 처리, 데이터 검증 등을 통해 안정적이고 효율적인 스크래핑 시스템을 구축할 수 있으며, 병렬 처리, 데이터베이스 저장, 모니터링 등을 통해 실무에서 더욱 효과적으로 활용할 수 있습니다. 

다만 법적, 윤리적 고려사항을 준수하여 스크래핑을 수행해야 합니다.
