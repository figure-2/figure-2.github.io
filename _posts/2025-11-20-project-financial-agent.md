---
title: "Financial-Agent: AI 기반 금융 투자 비서"
categories:
- 2.PROJECT
- 2-4. Financial-Agent
tags:
- LangChain
- LangGraph
- Financial Analysis
- AI Agent
- LLM
- Docker
- Project
toc: true
date: 2025-11-23
comments: true
mermaid: true
math: true
---

## Financial-Agent: AI 기반 금융 투자 비서

> **프로젝트 기간**: 2025.10 ~ 2025.11 (2개월)  
> **목적**: 자연어 질의를 통해 복잡한 금융 데이터를 조회, 분석하고 개인화된 투자 위험을 관리하는 AI Agent 개발  


<br>

## 🎯 프로젝트 개요

Financial-Agent는 사용자의 자연어 질문을 이해하고, 복잡한 주식 시장 데이터를 분석하여 인사이트를 제공하는 지능형 에이전트입니다. 단순한 가격 조회를 넘어, 복합 조건 검색, 기술적 시그널 감지, 그리고 개인 투자 성향을 고려한 리스크 관리까지 수행합니다.

<br>

## 🏗 시스템 아키텍처 및 주요 기능

LangGraph를 기반으로 상태(State)를 관리하며, 5가지 주요 Task를 수행하는 서브 그래프(Sub-graph) 구조로 설계되었습니다.

graph TB
    User[User Input] --> Router{Intent Router}
    
    %% Task 1: 단순 조회
    Router -->|Simple Query| Task1[Task 1: Info Retrieval<br/>SQL Generation]
    
    %% Task 2: 조건 검색
    Router -->|Screening| Task2[Task 2: Screening<br/>Complex Conditions]
    
    %% Task 3: 시그널 감지
    Router -->|Signal| Task3[Task 3: Signal Detection<br/>Technical Analysis]
    
    %% Task 4: 모호성 해결
    Router -->|Ambiguous| Task4[Task 4: Clarification<br/>Rewrite & Ask]
    Task4 -->|Rewritten| Router
    
    %% Task 5: 위험 관리
    Router -->|Risk Analysis| Task5[Task 5: Risk Alert<br/>Personalized Analysis]
    
    %% 결과 통합
    Task1 --> Response[Final Response]
    Task2 --> Response
    Task3 --> Response
    Task5 --> Response### Task 1: 단순 조회 (Information Retrieval)
주식 시장의 다양한 금융 정보(가격, 등락률, 시가총액 등)를 자연어 질문으로 조회합니다.
- **기능**: 가격 조회, 시장 통계, 순위 확인, 종목 간 비교
- **예시**: "동부건설우의 2024-11-06 시가는?", "2025-03-15에 KOSDAQ에서 상승한 종목은 몇 개?"

### Task 2: 조건 검색 (Stock Screening)
복잡한 조건을 만족하는 종목들을 필터링하여 검색합니다. 여러 조건을 AND 연산으로 결합할 수 있습니다.
- **기능**: 등락률, 거래량 급증, 특정 가격대 종목 필터링
- **예시**: "2025-09-05에 KOSPI에서 종가가 10만원 이상이고 거래량이 50만주 이상인 종목 알려줘"

### Task 3: 시그널 감지 (Technical Signal Detection)
기술적 분석 지표(RSI, 이동평균선, 볼린저밴드 등)를 활용하여 매매 시점을 포착합니다.
- **지원 지표**: RSI 과매수/과매도, 골든/데드크로스, 볼린저밴드 터치, 이동평균선 돌파
- **예시**: "2025-01-20에 RSI 과매수(70 이상) 종목을 알려줘"

### Task 4: 모호한 의미 해석 (Ambiguity Resolution)
사용자의 불완전하거나 모호한 질문(축약어, 은어, 정보 누락 등)을 자동으로 해석하고 명확하게 변환합니다.
- **처리 방식**:
    - **Rewriting**: 축약어("삼전" -> "삼성전자"), 은어("떡상" -> "폭등") 변환
    - **Clarification**: 누락된 정보(날짜, 종목명)에 대해 역질문 생성

### Task 5: 집중 투자 위험 알림 (Risk Management)
투자자의 매매 패턴과 마이데이터(투자 성향, 자산 규모)를 분석하여 과도한 집중 투자 위험을 경고합니다.
- **PTPRA 모델**: Personalized Trading Pattern Risk Alert
- **기능**: 개인별 위험 임계치 산출, 뉴스 기반 매매 동기 분석, 위험 경고 리포트 생성

<br>

## 💻 기술 스택

### Core Logic
- **LangChain & LangGraph**: 에이전트 상태 관리 및 워크플로우 오케스트레이션
- **OpenAI GPT-4o**: 자연어 이해 및 SQL/JSON 생성
- **Pandas & NumPy**: 금융 데이터 전처리 및 분석

### Data Engineering
- **yfinance**: 주식 시장 데이터 수집
- **SQLAlchemy & SQLite**: 로컬 데이터베이스 구축 및 ORM
- **BeautifulSoup & Selenium**: 뉴스 데이터 크롤링 및 하이라이팅

### Infrastructure
- **Docker**: 실행 환경 컨테이너화
- **FastAPI**: REST API 엔드포인트 제공

<br>

## 📊 구현 상세 (Code Snippet)

### 자연어 -> SQL 변환 (Task 1)
LLM을 통해 자연어를 구조화된 JSON으로 변환하고, 이를 SQL 쿼리로 매핑하여 실행합니다.

def parse_question_with_llm(state: AgentState) -> Dict[str, Any]:
    # LLM을 통해 자연어를 구조화된 JSON으로 변환
    # 지원 task_type: PRICE_INQUIRY, MARKET_STATISTICS, RANKING 등
    pass

def execute_plan(state: AgentState) -> Dict[str, Any]:
    # 분석된 JSON 계획을 SQL 쿼리로 변환하여 데이터베이스 조회 실행
    # 예: {"task_type": "PRICE_INQUIRY", "stock": "삼성전자"} 
    # -> SELECT open, close FROM stocks WHERE name='삼성전자' ...
    pass### 위험도 분석 로직 (Task 5)
투자자의 성향과 생애주기(나이)를 고려하여 개인화된 위험 임계치를 계산합니다.

def analyze_risk_patterns(state: AgentState) -> Dict[str, Any]:
    # 1. 개인화 임계치 계산
    # 투자성향 한도(예: 위험중립형 60%) × 생애주기 계수(예: 20대 0.3)
    personalized_threshold = profile_limit * age_factor
    
    # 2. 포트폴리오 집중도 계산
    concentration = stock_value / total_asset
    
    # 3. 위험 경고 여부 판단
    if concentration > personalized_threshold:
        return create_risk_alert(stock_name, concentration, personalized_threshold)
    return {"status": "SAFE"}<br>

## 🎓 학습한 점

1. **Agentic Workflow**: LangGraph를 활용하여 단순한 체인이 아닌, 순환하고 분기하는 에이전트 워크플로우를 설계하며 LLM 애플리케이션의 제어 흐름을 깊이 이해했습니다.
2. **Text-to-SQL**: 자연어를 정확한 SQL로 변환하기 위해 스키마 정보를 프롬프트에 효율적으로 주입하고, LLM의 환각을 제어하는 프롬프트 엔지니어링 기술을 익혔습니다.
3. **Domain Knowledge Integration**: 금융 도메인의 특수성(은어, 기술적 지표, 리스크 관리 이론)을 로직에 녹여내어, 단순 챗봇이 아닌 전문성 있는 도구로 발전시켰습니다.

<br>

## 🔗 관련 링크
- **GitHub Repository**: [Financial-Agent](https://github.com/figure-2/Financial-Agent) (Private)
- **API Endpoint**: `http://211.188.58.134:8000/agent` (데모 기간 한정)
