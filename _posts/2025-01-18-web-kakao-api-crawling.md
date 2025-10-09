---
title: 카카오 API 크롤링 - 다양한 서비스 활용하기
categories:
- 1.TIL
- 1-5.WEB
tags:
- kakao
- api
- 크롤링
- 카카오맵
- 카카오뉴스
- 카카오톡
- 데이터수집
toc: true
date: 2023-10-12 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 카카오 API 크롤링 - 다양한 서비스 활용하기

## 개요

카카오 API는 다양한 서비스를 제공하는 강력한 도구입니다:

- **카카오맵 API**: 장소 검색, 좌표 변환, 주소 검색
- **카카오뉴스 API**: 뉴스 검색 및 분석
- **카카오톡 API**: 메시지 전송, 친구 목록 조회
- **REST API**: HTTP 기반의 RESTful API 제공

## 1. 카카오 API 개요

### 주요 특징

- **다양한 서비스**: 맵, 메시지, 결제, 검색 등 다양한 기능
- **RESTful**: 표준 HTTP 메서드 사용
- **인증**: OAuth 2.0 기반 인증 시스템
- **제한**: API 호출 횟수 및 데이터 사용량 제한

### 장점

- **안정성**: 안정적인 API 서비스 제공
- **문서화**: 상세한 API 문서 제공
- **지원**: 다양한 프로그래밍 언어 지원
- **무료**: 기본적인 사용량은 무료

## 2. 카카오 API 설정

### 기본 설정

```python
import requests
import json
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# API 키 설정 (환경변수에서 가져오기)
KAKAO_API_KEY = os.getenv('KAKAO_API_KEY')
KAKAO_REST_API_KEY = os.getenv('KAKAO_REST_API_KEY')

# 헤더 설정
headers = {
    "Authorization": f"KakaoAK {KAKAO_API_KEY}",
    "Content-Type": "application/json"
}

# REST API 헤더
rest_headers = {
    "Authorization": f"Bearer {KAKAO_REST_API_KEY}",
    "Content-Type": "application/json"
}
```

### API 키 검증

```python
def validate_api_key():
    """API 키 유효성 검증"""
    if not KAKAO_API_KEY:
        raise ValueError("KAKAO_API_KEY가 설정되지 않았습니다.")
    if not KAKAO_REST_API_KEY:
        raise ValueError("KAKAO_REST_API_KEY가 설정되지 않았습니다.")
    
    print("API 키가 올바르게 설정되었습니다.")
    return True

# API 키 검증 실행
validate_api_key()
```

## 3. 카카오맵 API

### 장소 검색

```python
def search_places(query, category=None, x=None, y=None, radius=None, page=1, size=15):
    """카카오맵 장소 검색"""
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    
    params = {
        "query": query,
        "page": page,
        "size": size
    }
    
    if category:
        params["category_group_code"] = category
    
    if x and y:
        params["x"] = x
        params["y"] = y
    
    if radius:
        params["radius"] = radius
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return None

# 서울의 맛집 검색 예시
def search_restaurants_in_seoul():
    """서울의 맛집 검색"""
    # 서울 중심 좌표
    seoul_x = "126.9780"
    seoul_y = "37.5665"
    
    # 맛집 검색
    results = search_places(
        query="맛집",
        category="FD6",  # 음식점 카테고리
        x=seoul_x,
        y=seoul_y,
        radius=5000,  # 5km 반경
        size=15
    )
    
    if results and results.get("documents"):
        restaurants = []
        for place in results["documents"]:
            restaurant = {
                "name": place["place_name"],
                "address": place["address_name"],
                "road_address": place["road_address_name"],
                "phone": place.get("phone", ""),
                "category": place["category_name"],
                "x": place["x"],
                "y": place["y"],
                "url": place.get("place_url", "")
            }
            restaurants.append(restaurant)
        
        return restaurants
    
    return []

# 맛집 검색 실행
restaurants = search_restaurants_in_seoul()
print(f"검색된 맛집 수: {len(restaurants)}")

# 결과를 DataFrame으로 변환
if restaurants:
    df_restaurants = pd.DataFrame(restaurants)
    print(df_restaurants.head())
```

### 좌표 변환

```python
def convert_coordinates(x, y, output_coord="WGS84"):
    """좌표 변환 (WGS84, WCONGNAMUL, CONGNAMUL, WTM, TM)"""
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    
    params = {
        "x": x,
        "y": y,
        "output_coord": output_coord
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"좌표 변환 오류: {e}")
        return None

# 서울 좌표 변환 예시
def convert_seoul_coordinates():
    """서울 좌표 변환 예시"""
    # 서울 시청 좌표 (WGS84)
    seoul_x = "126.9780"
    seoul_y = "37.5665"
    
    # 좌표 변환
    result = convert_coordinates(seoul_x, seoul_y, "WCONGNAMUL")
    
    if result and result.get("documents"):
        address_info = result["documents"][0]
        print(f"원본 좌표: {seoul_x}, {seoul_y}")
        print(f"변환된 좌표: {address_info['x']}, {address_info['y']}")
        print(f"주소: {address_info['address']['address_name']}")
        print(f"도로명 주소: {address_info['road_address']['address_name']}")
    
    return result

# 좌표 변환 실행
convert_seoul_coordinates()
```

### 주소 검색

```python
def search_address(query, page=1, size=15):
    """주소 검색"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    
    params = {
        "query": query,
        "page": page,
        "size": size
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"주소 검색 오류: {e}")
        return None

# 주소 검색 예시
def search_address_example():
    """주소 검색 예시"""
    query = "서울시 강남구"
    
    result = search_address(query)
    
    if result and result.get("documents"):
        addresses = []
        for address in result["documents"]:
            addr_info = {
                "address_name": address["address_name"],
                "road_address_name": address.get("road_address_name", ""),
                "x": address["x"],
                "y": address["y"],
                "address_type": address["address_type"]
            }
            addresses.append(addr_info)
        
        return addresses
    
    return []

# 주소 검색 실행
addresses = search_address_example()
print(f"검색된 주소 수: {len(addresses)}")

if addresses:
    df_addresses = pd.DataFrame(addresses)
    print(df_addresses.head())
```

## 4. 카카오뉴스 API

### 뉴스 검색

```python
def search_news(query, sort="accuracy", page=1, size=10):
    """카카오뉴스 검색"""
    url = "https://dapi.kakao.com/v2/search/news"
    
    params = {
        "query": query,
        "sort": sort,  # accuracy, recency
        "page": page,
        "size": size
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"뉴스 검색 오류: {e}")
        return None

# 기술 뉴스 검색 예시
def search_tech_news():
    """기술 뉴스 검색"""
    query = "인공지능 머신러닝"
    
    result = search_news(query, sort="recency", size=20)
    
    if result and result.get("documents"):
        news_list = []
        for news in result["documents"]:
            news_item = {
                "title": news["title"],
                "contents": news["contents"],
                "url": news["url"],
                "datetime": news["datetime"],
                "publisher": news.get("publisher", ""),
                "category": news.get("category", "")
            }
            news_list.append(news_item)
        
        return news_list
    
    return []

# 뉴스 검색 실행
tech_news = search_tech_news()
print(f"검색된 뉴스 수: {len(tech_news)}")

if tech_news:
    df_news = pd.DataFrame(tech_news)
    print(df_news.head())
```

### 뉴스 데이터 분석

```python
def analyze_news_data(news_list):
    """뉴스 데이터 분석"""
    if not news_list:
        return None
    
    df = pd.DataFrame(news_list)
    
    # 날짜 변환
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    # 분석 결과
    analysis = {
        "total_news": len(df),
        "date_range": {
            "start": df['date'].min(),
            "end": df['date'].max()
        },
        "publisher_counts": df['publisher'].value_counts().head(10),
        "hourly_distribution": df['hour'].value_counts().sort_index(),
        "category_distribution": df['category'].value_counts()
    }
    
    return analysis, df

# 뉴스 데이터 분석 실행
if tech_news:
    analysis_result, news_df = analyze_news_data(tech_news)
    
    print("=== 뉴스 데이터 분석 결과 ===")
    print(f"총 뉴스 수: {analysis_result['total_news']}")
    print(f"날짜 범위: {analysis_result['date_range']['start']} ~ {analysis_result['date_range']['end']}")
    
    print("\n=== 주요 발행사 ===")
    print(analysis_result['publisher_counts'])
    
    print("\n=== 시간대별 분포 ===")
    print(analysis_result['hourly_distribution'])
```

## 5. 카카오톡 API

### 메시지 전송

```python
def send_kakao_message(friend_name, message):
    """카카오톡 메시지 전송 (친구에게)"""
    url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"
    
    data = {
        "receiver_uuids": [friend_name],
        "template_object": {
            "object_type": "text",
            "text": message,
            "link": {
                "web_url": "https://developers.kakao.com",
                "mobile_web_url": "https://developers.kakao.com"
            }
        }
    }
    
    try:
        response = requests.post(url, headers=rest_headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"메시지 전송 오류: {e}")
        return None

# 알림 메시지 전송 예시
def send_notification_message():
    """알림 메시지 전송"""
    message = "안녕하세요! 카카오 API를 통한 메시지입니다."
    
    # 실제 사용 시 친구 이름을 입력
    friend_name = "친구이름"
    
    result = send_kakao_message(friend_name, message)
    
    if result:
        print("메시지가 성공적으로 전송되었습니다.")
        print(result)
    else:
        print("메시지 전송에 실패했습니다.")
    
    return result

# 메시지 전송 실행 (실제 사용 시 주석 해제)
# send_notification_message()
```

### 친구 목록 조회

```python
def get_friend_list():
    """카카오톡 친구 목록 조회"""
    url = "https://kapi.kakao.com/v1/api/talk/friends"
    
    try:
        response = requests.get(url, headers=rest_headers)
        response.raise_for_status()
        
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"친구 목록 조회 오류: {e}")
        return None

# 친구 정보 조회 예시
def get_friends_info():
    """친구 정보 조회"""
    result = get_friend_list()
    
    if result and result.get("elements"):
        friends = []
        for friend in result["elements"]:
            friend_info = {
                "id": friend["id"],
                "uuid": friend["uuid"],
                "profile_nickname": friend["profile_nickname"],
                "profile_thumbnail_image": friend.get("profile_thumbnail_image", "")
            }
            friends.append(friend_info)
        
        return friends
    
    return []

# 친구 목록 조회 실행
friends_list = get_friends_info()
print(f"친구 수: {len(friends_list)}")

if friends_list:
    df_friends = pd.DataFrame(friends_list)
    print(df_friends.head())
```

## 6. 실무 적용 사례

### 위치 기반 서비스

```python
def create_location_service():
    """위치 기반 서비스 구현"""
    
    def find_nearby_places(latitude, longitude, category, radius=1000):
        """주변 장소 찾기"""
        places = search_places(
            query=category,
            x=longitude,
            y=latitude,
            radius=radius
        )
        
        if places and places.get("documents"):
            nearby_places = []
            for place in places["documents"]:
                place_info = {
                    "name": place["place_name"],
                    "address": place["address_name"],
                    "distance": place.get("distance", 0),
                    "category": place["category_name"],
                    "phone": place.get("phone", ""),
                    "x": place["x"],
                    "y": place["y"]
                }
                nearby_places.append(place_info)
            
            return nearby_places
        
        return []
    
    def get_route_info(start_x, start_y, end_x, end_y):
        """경로 정보 조회"""
        # 실제 경로 API는 별도로 구현 필요
        # 여기서는 기본적인 장소 검색으로 대체
        start_place = search_places("", x=start_x, y=start_y, radius=100)
        end_place = search_places("", x=end_x, y=end_y, radius=100)
        
        return {
            "start": start_place,
            "end": end_place
        }
    
    return find_nearby_places, get_route_info

# 위치 기반 서비스 사용
find_nearby, get_route = create_location_service()

# 주변 맛집 찾기
nearby_restaurants = find_nearby(
    latitude="37.5665",
    longitude="126.9780",
    category="맛집",
    radius=500
)

print(f"주변 맛집 수: {len(nearby_restaurants)}")
if nearby_restaurants:
    for restaurant in nearby_restaurants[:5]:
        print(f"- {restaurant['name']}: {restaurant['address']}")
```

### 뉴스 모니터링 시스템

```python
def create_news_monitoring_system():
    """뉴스 모니터링 시스템"""
    
    def monitor_keywords(keywords, interval_minutes=30):
        """키워드 모니터링"""
        import time
        
        all_news = []
        
        for keyword in keywords:
            print(f"'{keyword}' 키워드 모니터링 중...")
            
            news = search_news(keyword, sort="recency", size=10)
            
            if news and news.get("documents"):
                for article in news["documents"]:
                    article["keyword"] = keyword
                    article["monitored_at"] = datetime.now()
                    all_news.append(article)
        
        return all_news
    
    def analyze_trending_topics(news_list):
        """트렌딩 토픽 분석"""
        if not news_list:
            return None
        
        df = pd.DataFrame(news_list)
        
        # 키워드별 뉴스 수
        keyword_counts = df['keyword'].value_counts()
        
        # 발행사별 뉴스 수
        publisher_counts = df['publisher'].value_counts()
        
        # 시간대별 분포
        df['datetime'] = pd.to_datetime(df['datetime'])
        hourly_distribution = df['datetime'].dt.hour.value_counts().sort_index()
        
        return {
            "keyword_counts": keyword_counts,
            "publisher_counts": publisher_counts,
            "hourly_distribution": hourly_distribution
        }
    
    return monitor_keywords, analyze_trending_topics

# 뉴스 모니터링 시스템 사용
monitor_keywords, analyze_trends = create_news_monitoring_system()

# 키워드 모니터링
keywords = ["인공지능", "머신러닝", "딥러닝", "데이터사이언스"]
monitored_news = monitor_keywords(keywords)

print(f"모니터링된 뉴스 수: {len(monitored_news)}")

# 트렌드 분석
trend_analysis = analyze_trends(monitored_news)
if trend_analysis:
    print("\n=== 키워드별 뉴스 수 ===")
    print(trend_analysis['keyword_counts'])
    
    print("\n=== 주요 발행사 ===")
    print(trend_analysis['publisher_counts'].head())
```

### 데이터 수집 및 저장

```python
def create_data_collection_system():
    """데이터 수집 및 저장 시스템"""
    
    def collect_place_data(queries, categories=None):
        """장소 데이터 수집"""
        all_places = []
        
        for query in queries:
            print(f"'{query}' 장소 데이터 수집 중...")
            
            places = search_places(query, size=15)
            
            if places and places.get("documents"):
                for place in places["documents"]:
                    place["collected_at"] = datetime.now()
                    all_places.append(place)
        
        return all_places
    
    def save_to_csv(data, filename):
        """CSV 파일로 저장"""
        if not data:
            print("저장할 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"데이터가 {filename}에 저장되었습니다.")
    
    def save_to_json(data, filename):
        """JSON 파일로 저장"""
        if not data:
            print("저장할 데이터가 없습니다.")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"데이터가 {filename}에 저장되었습니다.")
    
    return collect_place_data, save_to_csv, save_to_json

# 데이터 수집 시스템 사용
collect_places, save_csv, save_json = create_data_collection_system()

# 장소 데이터 수집
place_queries = ["카페", "맛집", "병원", "약국", "은행"]
collected_places = collect_places(place_queries)

print(f"수집된 장소 수: {len(collected_places)}")

# 데이터 저장
if collected_places:
    save_csv(collected_places, "kakao_places.csv")
    save_json(collected_places, "kakao_places.json")
```

## 7. API 사용 제한 및 최적화

### API 호출 제한 관리

```python
import time
from functools import wraps

def rate_limit(calls_per_second=10):
    """API 호출 제한 데코레이터"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# API 호출 제한 적용
@rate_limit(calls_per_second=5)
def limited_search_places(query, **kwargs):
    """호출 제한이 적용된 장소 검색"""
    return search_places(query, **kwargs)

# 제한된 API 호출 테스트
def test_rate_limited_api():
    """제한된 API 호출 테스트"""
    queries = ["카페", "맛집", "병원", "약국", "은행"]
    
    for query in queries:
        print(f"'{query}' 검색 중...")
        result = limited_search_places(query, size=5)
        if result and result.get("documents"):
            print(f"  - {len(result['documents'])}개 장소 발견")
        time.sleep(1)  # 추가 대기

# 제한된 API 호출 실행
# test_rate_limited_api()
```

### 에러 처리 및 재시도

```python
def retry_on_failure(max_retries=3, delay=1):
    """실패 시 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        print(f"최대 재시도 횟수 초과: {e}")
                        raise
                    print(f"시도 {attempt + 1} 실패, {delay}초 후 재시도: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# 재시도 기능이 적용된 API 호출
@retry_on_failure(max_retries=3, delay=2)
def robust_search_places(query, **kwargs):
    """재시도 기능이 있는 장소 검색"""
    return search_places(query, **kwargs)

# 견고한 API 호출 테스트
def test_robust_api():
    """견고한 API 호출 테스트"""
    query = "맛집"
    
    result = robust_search_places(query, size=10)
    
    if result and result.get("documents"):
        print(f"'{query}' 검색 성공: {len(result['documents'])}개 장소")
        return result
    else:
        print(f"'{query}' 검색 실패")
        return None

# 견고한 API 호출 실행
# test_robust_api()
```

## 8. 주의사항 및 모범 사례

### API 키 보안

- **환경변수 사용**: API 키를 코드에 직접 포함하지 않기
- **권한 관리**: 필요한 권한만 부여
- **정기적 갱신**: API 키 정기적 갱신

### 사용량 관리

- **호출 제한**: API 호출 제한 준수
- **캐싱**: 중복 요청 방지를 위한 캐싱
- **배치 처리**: 대량 데이터 처리 시 배치 처리

### 에러 처리

- **예외 처리**: 적절한 예외 처리 구현
- **재시도 로직**: 일시적 오류에 대한 재시도
- **로깅**: 오류 로깅 및 모니터링

## 9. 실무 활용 팁

### 1. 데이터 품질 관리

```python
def validate_place_data(place_data):
    """장소 데이터 유효성 검증"""
    required_fields = ['place_name', 'address_name', 'x', 'y']
    
    for field in required_fields:
        if field not in place_data or not place_data[field]:
            return False
    
    # 좌표 유효성 검증
    try:
        float(place_data['x'])
        float(place_data['y'])
    except (ValueError, TypeError):
        return False
    
    return True

def clean_place_data(places):
    """장소 데이터 정제"""
    cleaned_places = []
    
    for place in places:
        if validate_place_data(place):
            # 불필요한 HTML 태그 제거
            place['place_name'] = place['place_name'].replace('<b>', '').replace('</b>', '')
            place['address_name'] = place['address_name'].replace('<b>', '').replace('</b>', '')
            cleaned_places.append(place)
    
    return cleaned_places
```

### 2. 배치 처리 최적화

```python
def batch_process_places(queries, batch_size=5):
    """배치 처리로 장소 데이터 수집"""
    all_results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        print(f"배치 {i//batch_size + 1} 처리 중...")
        
        batch_results = []
        for query in batch:
            result = search_places(query, size=10)
            if result and result.get("documents"):
                batch_results.extend(result["documents"])
        
        all_results.extend(batch_results)
        
        # 배치 간 대기
        if i + batch_size < len(queries):
            time.sleep(2)
    
    return all_results
```

### 3. 데이터 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_place_data(places_df):
    """장소 데이터 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 카테고리별 분포
    plt.subplot(2, 2, 1)
    places_df['category_name'].value_counts().head(10).plot(kind='bar')
    plt.title('카테고리별 장소 분포')
    plt.xticks(rotation=45)
    
    # 지역별 분포
    plt.subplot(2, 2, 2)
    places_df['address_name'].str.split(' ').str[0].value_counts().head(10).plot(kind='bar')
    plt.title('지역별 장소 분포')
    plt.xticks(rotation=45)
    
    # 좌표 분포
    plt.subplot(2, 2, 3)
    plt.scatter(places_df['x'].astype(float), places_df['y'].astype(float), alpha=0.6)
    plt.title('장소 좌표 분포')
    plt.xlabel('경도')
    plt.ylabel('위도')
    
    # 거리 분포
    plt.subplot(2, 2, 4)
    if 'distance' in places_df.columns:
        places_df['distance'].astype(float).hist(bins=20)
        plt.title('거리 분포')
        plt.xlabel('거리 (m)')
    
    plt.tight_layout()
    plt.show()
```

## 마무리

카카오 API는 다양한 서비스를 제공하는 강력한 도구입니다. 카카오맵, 카카오뉴스, 카카오톡 등의 API를 활용하여 위치 기반 서비스, 뉴스 모니터링, 메시지 전송 등의 기능을 구현할 수 있습니다. 

적절한 API 키 관리, 호출 제한 준수, 에러 처리 등을 통해 안정적이고 효율적인 서비스를 구축할 수 있으며, 실무에서는 데이터 품질 관리, 배치 처리 최적화, 시각화 등을 통해 더욱 효과적으로 활용할 수 있습니다.
