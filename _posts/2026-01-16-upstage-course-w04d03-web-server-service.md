---
title: "웹서버 사용자 서비스"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-4.NETWORK_CLOUD
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- http
- course-note
- network-cloud
toc: true
date: 2026-01-16 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 웹서버 사용자 서비스

## 핵심 개념

> **요약**
> FastAPI를 사용하여 웹서버를 고도화한다. APIRouter로 라우트를 분리하고, Pydantic으로 요청/응답 모델을 정의하여 타입 안전성을 확보한다. Repository 패턴을 도입하여 데이터 접근 계층을 비즈니스 로직과 분리하고, Dependency Injection으로 서비스 간 의존성을 관리한다.

## 주요 내용

### 1. FastAPI 개발 - Route와 Service
- 웹서버 구조의 **책임별 계층화**: router - service - repository
- **APIRouter**로 API 분리: `prefix`, `tags` 설정
- uvicorn으로 FastAPI 실행
- 관련: FastAPI, API 라우팅

### 2. Pydantic 모델링
- Python **타입 힌트** 기반 데이터 검증/변환 라이브러리
- `BaseModel`을 상속하여 요청/응답 모델 정의
- `UserCreate(name: str, email: str)`, `UserResponse(id: int, name: str, ...)`
- JSON 타입 안전성 확보
- 관련: Pydantic

### 3. Global Exception Handler
- 에러/예외 처리 전역 핸들러
- 로그 시스템 구축
- 관련: 에러 처리

### 4. Repository 패턴
- **Service가 데이터를 직접 처리**하면 문제 발생
  - DB 접근 코드와 비즈니스 로직 혼재
  - Service의 책임이 과도하게 커짐
- **Repository**: 데이터 접근 추상화 계층
  - 데이터 저장/조회 등 담당
  - Service는 비즈니스에 집중, Repository는 데이터 접근에 집중
- 관련: Repository 패턴

### 5. Dependency Injection
- FastAPI의 `Depends`를 통한 의존성 주입
- 객체지향 원칙: 책임, 위임, 의존성 관리
- Repository + MySQL + ORM 연동
- 관련: Dependency Injection

## 흐름도

![웹서버 사용자 서비스 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-network-cloud-w04d03-web-server-service-diagram-1.svg)

## 연결된 개념
- FastAPI
- Pydantic
- Repository 패턴
- Dependency Injection
- API 라우팅
