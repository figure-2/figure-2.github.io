---
title: "idol-agent v0.7 - 상태관리 + 비용 추적"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-10.PROJECTS
- PROJECT_NOTE
tags:
- upstage
- sesac
- ai-agent
- state-management
- langgraph
- project-note
- projects
toc: true
date: 2026-03-04 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# idol-agent v0.7 - 상태관리 + 비용 추적

> **프로젝트 정보**
> - **위치**: `Week09/Day02/Day7_mission/`
> - **기술 스택**: FastAPI, LangGraph, Supabase, PostgreSQL, tiktoken
> - **주차**: Week 09

## 아키텍처

![idol-agent v0.7 - 상태관리 + 비용 추적 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/03-projects-idol-agent-v07-diagram-1.svg)

## 문제 정의

v0.7의 목표는 에이전트가 대화 상태와 운영 비용을 관리하도록 만드는 것이다. v0.6에서 모델 호출 안정성을 높였다면, v0.7에서는 “대화가 길어질수록 무엇을 기억하고 무엇을 줄일 것인가”를 다룬다.

LLM 에이전트는 이전 대화, RAG 결과, tool 실행 결과를 기억해야 자연스럽게 동작한다. 하지만 모든 내용을 계속 prompt에 넣으면 token 비용이 증가하고 context window를 초과한다. 그래서 상태 영속화, token counting, 비용 제한, 메시지 trimming을 함께 설계해야 한다.

## v0.6 대비 추가 사항

### 1. 체크포인터 (상태 영속화)
- **MemorySaver**: 개발 환경용 인메모리 체크포인터
- **PostgreSQL Checkpointer**: 프로덕션 Supabase 연결
- 환경변수로 전환: `CHECKPOINTER_TYPE`

### 2. 토큰 카운터
- tiktoken으로 정확한 토큰 수 계산
- 비용 산정 기초 데이터 제공

### 3. 비용 추적 + Discord 알림
- daily_cost_limit, max_context_tokens 설정
- Discord Webhook으로 비용 초과 알림

### 4. 메시지 트리밍
- 컨텍스트 윈도우 초과 방지
- 오래된 메시지 자동 제거

## 구현 포인트

상태 관리에서 중요한 것은 저장소 선택보다 상태 모델을 정리하는 것이다. 대화 이력, graph 실행 상태, 운영 메트릭을 구분하지 않으면 나중에 trimming이나 비용 추적 로직이 복잡해진다.

| 상태 종류 | 예시 | 사용처 |
|---|---|---|
| 대화 상태 | messages, session_id | 멀티 턴 대화 유지 |
| 실행 상태 | intent, retrieved_docs, tool_result | LangGraph node 간 전달 |
| 운영 상태 | token_usage, estimated_cost, fallback_used | 비용 추적, 알림, 분석 |

checkpointer는 이 상태를 저장하고 복원하는 역할을 한다. MemorySaver는 빠른 개발 확인에 적합하지만, 운영에서는 서버 재시작 후 상태가 사라지면 안 되므로 PostgreSQL 기반 저장소가 필요하다.

## 비용 추적과 trimming

비용 추적은 나중에 로그를 보고 추정하는 방식보다 호출 단위로 남기는 방식이 안전하다. 모델 호출마다 prompt token, completion token, model name, retry/fallback 여부를 함께 기록해야 실제 비용에 가까워진다.

메시지 trimming은 오래된 메시지를 단순 삭제하는 방식이면 위험하다. 예를 들어 사용자의 목표, system instruction, tool 결과, RAG 근거가 사라지면 응답 품질이 급격히 떨어질 수 있다. 따라서 trimming 전후로 “답변에 필요한 최소 맥락이 남아 있는가”를 확인해야 한다.

## 운영 관점에서 배운 점

- 상태 영속화는 사용자 경험을 높이지만 개인정보와 보관 기간 리스크를 만든다.
- token budget은 비용 관리와 답변 품질을 동시에 좌우한다.
- 비용 알림은 secret이나 raw prompt를 포함하지 않는 sanitized summary로 보내야 한다.
- 상태 관리와 Observability를 연결해야 나중에 장애 원인 분석이 가능하다.

## 사용된 개념

- [상태관리]({% post_url 2026-03-04-upstage-tech-state-management %})
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [Supabase]({% post_url 2026-02-10-upstage-tech-supabase %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
