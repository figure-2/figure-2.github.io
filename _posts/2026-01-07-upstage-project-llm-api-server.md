---
title: "Solar FastAPI 앱"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-10.PROJECTS
- PROJECT_NOTE
tags:
- upstage
- sesac
- ai-agent
- fastapi
- project-note
- projects
toc: true
date: 2026-01-07 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Solar FastAPI 앱

> **프로젝트 정보**
> - **위치**: `Week03/solar-project/`
> - **기술 스택**: FastAPI, Python, uv
> - **주차**: Week 03

## 아키텍처

![Solar FastAPI 앱 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/03-projects-llm-api-server-diagram-1.svg)

## 프로젝트 구조

```
solar-project/
  main.py           # FastAPI 앱 진입점
  source/            # 소스 모듈
  pyproject.toml     # 의존성 관리 (uv)
```

## 핵심 구현 포인트

### 1. FastAPI 엔드포인트
- RESTful API 설계
- 비동기 요청 처리

### 2. Solar LLM 연동
- Solar API 키 기반 인증
- Solar LLM 호출 및 응답 처리

## 사용된 개념
- FastAPI - 비동기 웹 프레임워크
- HTTP - RESTful API 설계

## 회고
- FastAPI를 활용한 실전 API 서버 구축 경험
