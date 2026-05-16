---
title: "idol-agent v0.6 - LiteLLM + Docker + CI/CD"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-10.PROJECTS
- PROJECT_NOTE
tags:
- upstage
- sesac
- ai-agent
- litellm
- docker
- ci-cd
- project-note
- projects
toc: true
date: 2026-03-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# idol-agent v0.6 - LiteLLM + Docker + CI/CD

> **프로젝트 정보**
> - **위치**: `Week09/Day01/day6-mission/`
> - **기술 스택**: FastAPI, LangGraph, LiteLLM, Docker, Supabase
> - **주차**: Week 09

## 아키텍처

![idol-agent v0.6 - LiteLLM + Docker + CI/CD 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/03-projects-idol-agent-v06-diagram-1.svg)

## 문제 정의

v0.6의 목표는 에이전트 기능을 추가하는 것보다 운영 중 실패할 수 있는 지점을 줄이는 것이다. 이전 버전의 핵심이 LangGraph 기반 에이전트 구조였다면, v0.6에서는 모델 호출과 tool 실행을 더 안전하게 다루는 데 초점이 있다.

운영 관점에서 문제가 되는 지점은 다음과 같다.

- LLM API가 rate limit, timeout, provider 장애로 실패할 수 있다.
- tool call이 외부 I/O에 묶이면 응답 전체가 멈출 수 있다.
- 배포 환경마다 API key, model, timeout 설정이 달라질 수 있다.
- CI/CD가 없으면 작은 수정도 수동 검증에 의존하게 된다.

따라서 v0.6은 LiteLLM, Docker, GitHub Actions를 통해 “실행되는 에이전트”를 “운영 준비가 된 에이전트”에 가깝게 만드는 단계로 볼 수 있다.

## v0.2 대비 추가 사항

### 1. LiteLLM 통합
- **RetryPolicy**: 재시도 정책 설정
- **Router**: Primary(Solar) + Fallback(Gemini) 모델 라우팅
- **Config**: use_litellm, num_retries, timeout 등 환경변수

### 2. Tool 에러 핸들링
- `asyncio.wait_for`로 타임아웃 적용
- TimeoutError, 일반 에러 분기 처리

### 3. Docker 컨테이너화
- Dockerfile + docker-compose.yml
- 환경변수 기반 설정

### 4. CI/CD 연결
- Ruff 기반 lint/format check
- pytest 기반 테스트 실행
- PR 상태 comment
- AI code review workflow

## 구현 포인트

`day6-mission` 코드 기준으로 핵심은 `app/core/llm.py`와 `app/graph/nodes.py`다.

`app/core/llm.py`는 모델 호출 계층을 담당한다. LiteLLM Router를 이 계층에 두면 Graph node가 provider별 SDK와 직접 결합하지 않는다. `app/graph/nodes.py`는 router, rag, tool, response node를 담당하며, tool timeout과 error handling을 이곳에서 정리한다.

```text
FastAPI endpoint
  -> LangGraph graph
  -> router/rag/tool/response node
  -> LiteLLM wrapper
  -> primary 또는 fallback model
```

이 구조의 장점은 장애 대응 로직이 흩어지지 않는다는 점이다. 모델 fallback은 LLM wrapper에서 관리하고, tool timeout은 tool node에서 관리한다. API handler는 최종 상태와 응답만 다룬다.

## 운영 관점에서 배운 점

- fallback은 장애 대응이지만 품질 차이를 만들 수 있으므로 출력 형식 검증이 필요하다.
- timeout은 사용자 경험과 비용 사이의 절충이다.
- Docker 환경에서는 `.env.example`의 placeholder와 실제 secret 주입 경로를 분리해야 한다.
- CI는 코드 품질 자동화의 시작점이고, Observability는 운영 중 품질 확인의 시작점이다.

## 사용된 개념

- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %})
- [Docker]({% post_url 2026-01-12-upstage-tech-docker %})
- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %})
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [FastAPI]({% post_url 2026-01-07-upstage-tech-fastapi %})
