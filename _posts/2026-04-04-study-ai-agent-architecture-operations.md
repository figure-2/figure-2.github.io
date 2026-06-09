---
title: "AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- memory
- guardrails
- mcp
- a2a
toc: true
date: 2026-04-04 01:20:00 +0900
comments: false
mermaid: true
math: true
---

AI Agent를 실제 서비스로 만들 때는 모델 호출보다 운영 구조가 더 중요해진다. Memory, RAG, guardrails, protocol, cost control이 붙는 순간 Agent는 단순 챗봇이 아니라 상태를 가진 시스템이 된다.

## 핵심 아키텍처 개념

에이전트 시스템의 내부를 구성하는 핵심 개념을 memory, retrieval, safety, protocol 관점으로 나눠 정리한다.

### Agent Memory 시스템

인간의 기억 체계에서 영감을 받은 에이전트 메모리 시스템은 세 가지 유형으로 나뉩니다.

#### Short-term Memory

컨텍스트 윈도우 내의 작업 기억. 현재 대화와 즉시 필요한 정보를 유지합니다.

구현: Context Window, Working Memory

#### Long-term Memory

세션을 넘어 지속되는 기억. 사실, 정의, 규칙 등 구조화된 지식을 저장합니다.

구현: Vector DB, Knowledge Graph

#### Episodic Memory

과거 경험과 에피소드를 기록. 유사한 상황에서 과거 경험을 참조합니다.

구현: Vector DB + Semantic Retrieval

Generative Agents와 AI Agent Memory 관련 정리에서 공통적으로 강조하는 부분은, Agent memory가 단순 로그 저장이 아니라 현재 작업 기억과 장기 지식을 분리해야 한다는 점이다.

### Agentic RAG vs Traditional RAG

#### Traditional RAG

Query

↓

Vector Search

↓

Retrieve Docs

↓

Generate Answer

단일 패스. 검색 결과가 부족해도 재시도 없음. 도서관에서 책 한 권 빌리는 것과 같음.

vs

#### Agentic RAG

Plan

↓

Retrieve

↓

Evaluate

↓ / ↺

Re-retrieve / Tool Use

↓

Synthesize

반복적 검색, 평가, 재검색. 연구 조교가 여러 자료를 찾아 교차 검증하는 것과 같음.

### Guardrails 아키텍처

Safety

가드레일은 계층적 방어(Layered Defense) 원칙으로 설계됩니다. 하나의 가드레일로 모든 것을 잡을 수 없습니다.

Input Guardrails

PII 감지

Prompt Injection 방어

유해성 필터링

↓

Agent Core (LLM + Tools)

↓

Output Guardrails

할루시네이션 탐지

콘텐츠 검수

PII 제거

Tool Guardrails

실행 전 검증

권한 확인

Human-in-the-Loop

### 비용 최적화 전략

Production

에이전트 루프는 단일 호출 대비 10~100배 더 많은 토큰을 소비할 수 있습니다. 주요 최적화 전략:

Prompt Caching

60-80% 절감

캐시된 토큰은 75% 저렴. 시스템 프롬프트, 도구 스키마 재사용

Multi-Model Routing

30-60% 절감

단순 작업은 저렴한 모델, 복잡한 추론만 고급 모델 사용

Batch Processing

~50% 절감

비동기 배치 처리로 할인 적용 (OpenAI, Google, Mistral)

Prompt Engineering

15-40% 절감

간결한 프롬프트, JSON 구조화 출력, 사용하지 않는 도구 제거

Protocols

## 에이전트 통신 프로토콜

에이전트 생태계를 연결하는 두 가지 핵심 프로토콜

MCP

### Model Context Protocol

by Anthropic (Nov 2024)

Vertical

Agent ↔ Tools & Data

에이전트가 외부 도구와 데이터에 접근하는 방법을 표준화합니다. N x M 통합 문제를 M + N으로 줄입니다.

Tools

LLM이 호출할 수 있는 함수

Resources

접근할 수 있는 데이터 소스

Prompts

최적 사용을 위한 템플릿

JSON-RPC 2.0 | stdio / HTTP+SSE

A2A

### Agent-to-Agent Protocol

by Google (Apr 2025)

Horizontal

Agent ↔ Agent

에이전트 간 작업을 위임하고 결과를 교환하는 방법을 표준화합니다. Agent Card로 능력을 광고합니다.

Agent Cards

능력 광고 JSON 문서

Tasks

작업 단위 & 라이프사이클

Messages

컨텍스트, 결과, 아티팩트 교환

HTTP + JSON | SSE Streaming | Apache 2.0

MCP

Agent ↔ Tools

Complementary

A2A

Agent ↔ Agent

두 프로토콜은 경쟁이 아닌 상호 보완 관계입니다. 2025년 12월 Linux Foundation의 AAIF(Agentic AI Foundation)에서 OpenAI, Anthropic, Google, Microsoft, AWS가 공동 거버넌스에 합류했습니다.

---

## 추가 정리

### 핵심 요약

이 글의 중심은 에이전트가 단순히 LLM을 호출하는 구조가 아니라는 점이다. 실무 에이전트는 Memory, RAG, Guardrails, Cost Control을 함께 설계해야 운영 가능한 시스템이 된다.

### 보충 해설

Memory는 상태를 유지하기 위한 장치이고, RAG는 외부 지식을 가져오는 장치다. Guardrails는 행동 범위를 제한하고, Cost Control은 반복 실행이 비용 폭주로 이어지지 않게 막는다. 네 요소 중 하나라도 빠지면 데모는 가능해도 운영 안정성이 낮아진다.
