---
title: "Agent Architecture 수업 기록"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-8.AGENT_ARCHITECTURE
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- agent-architecture
- course-note
toc: true
date: 2026-02-09 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Agent Architecture 수업 기록

## 수업 위치

이 수업은 Product Engineering 단계에서 만든 에이전트 아이디어를 실제 서비스 구조로 옮기는 출발점이다. 강의자료의 핵심은 “노트북에서 돌아가는 코드”와 “다른 사람이 접속해서 쓸 수 있는 서비스”는 다르다는 점이다.

노트북은 실험에는 좋지만 서비스 운영에는 부족하다. 요청을 받는 API 계층, 설정을 관리하는 core 계층, LangGraph 실행 계층, DB 접근 계층, tool 실행 계층을 분리해야 수정과 장애 대응이 쉬워진다.

## 핵심 개념

> **요약**
> 에이전트 시스템을 프로덕션 서비스로 배포하기 위한 아키텍처를 설계한다. 노트북 기반 프로토타입에서 실제 서비스로 전환하기 위해 계층 분리, 아키텍처 특성, trade-off, Supabase 설정, API/Graph 구조를 함께 다룬다.

강의자료에서 특히 강조된 관점은 **AI Product Engineer는 모델을 직접 학습시키는 사람이라기보다 모델 API를 안정적인 서비스 파이프라인으로 감싸는 사람**이라는 점이다. 그래서 모델 성능만 보는 것이 아니라 latency, 비용, 안정성, 관측 가능성, 페르소나 일관성을 함께 봐야 한다.

## 주요 내용

### 1. 서비스 디플로이먼트 개요
- 노트북 → 스크립트 변환의 필요성
- 서비스 구성 요소: 백엔드, 프론트엔드, 인프라, CI/CD
- 현업 수준의 서비스 배포 프로세스
- MVP → 클라우드 배포 → 운영 개선으로 이어지는 학습 흐름

### 2. 아키텍처적 사고
- 모든 기술 선택은 trade-off를 가진다.
- LLM 에이전트에서는 품질, 속도, 비용을 동시에 최대로 잡기 어렵다.
- 프로젝트 시작 전에 중요한 아키텍처 특성을 정해야 한다.
- 결정은 ADR처럼 나중에 다시 이해할 수 있게 기록해야 한다.

### 3. 에이전트 서비스 아키텍처
- FastAPI 기반 백엔드 구조
- LangGraph 그래프 설계 패턴
- 프로젝트 구조화:
  - `app/api/routes/` - HTTP 엔드포인트
  - `app/core/` - 설정, LLM, 프롬프트
  - `app/graph/` - LangGraph 노드/엣지
  - `app/repositories/` - DB 접근
  - `app/tools/` - 외부 행동 실행

### 4. idol-agent 프로젝트
- 버추얼 아이돌 AI 에이전트 서비스
- Solar LLM + Supabase(pgvector)
- RAG 기반 세계관 지식 검색
- 스케줄 조회, 팬레터 저장, 추천, 날씨 조회 같은 tool action

## 설계 기준

Lumi 에이전트 기준으로 보면 중요한 아키텍처 특성은 다음과 같이 정리할 수 있다.

| 특성 | 의미 | 설계에 미치는 영향 |
|---|---|---|
| Persona Consistency | 루미답게 말하는가 | prompt, response node, 평가 기준에 반영 |
| Latency | 첫 응답까지 얼마나 걸리는가 | streaming, 모델 선택, RAG 검색 범위에 영향 |
| Cost Efficiency | 대화당 비용이 감당 가능한가 | prompt 길이, context trimming, 모델 routing에 영향 |
| Reliability | 외부 API 장애를 견디는가 | retry, fallback, timeout 설계 필요 |
| Observability | 문제 원인을 찾을 수 있는가 | node별 log, trace, token usage 기록 필요 |

이 기준을 먼저 잡아야 이후 LangGraph MVP, Streaming, CI/CD, LLMOps의 선택이 임의적인 기술 나열이 아니라 하나의 서비스 설계로 연결된다.

## 수업에서 남길 체크포인트

- `main.py` 하나에 모든 코드를 넣지 않고 계층을 나눠야 하는 이유를 설명할 수 있다.
- router, rag, tool, response node가 각각 어떤 책임을 갖는지 구분할 수 있다.
- Supabase는 단순 DB가 아니라 RAG 저장소와 서비스 데이터 저장소 역할을 함께 한다.
- architecture decision은 “왜 이렇게 했는가”가 남아야 나중에 운영 중 수정할 수 있다.
- secret 값은 `.env`와 배포 환경 secret으로만 다루고 블로그나 코드에 직접 쓰지 않는다.

## 흐름도

![Agent Architecture 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agent-architecture-w08d01-agent-architecture-diagram-1.svg)

## 관련 글

- [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [FastAPI]({% post_url 2026-01-07-upstage-tech-fastapi %})
- [Supabase]({% post_url 2026-02-10-upstage-tech-supabase %})
