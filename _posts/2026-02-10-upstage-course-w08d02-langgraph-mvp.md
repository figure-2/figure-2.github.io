---
title: "LangGraph MVP"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-8.AGENT_ARCHITECTURE
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- langgraph
- course-note
- agent-architecture
toc: true
date: 2026-02-10 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# LangGraph MVP

## 수업 위치

이 수업은 전날 만든 서비스 뼈대에 실제 에이전트의 “두뇌”를 넣는 단계다. 강의자료 기준으로 핵심 흐름은 사용자의 질문을 router가 분류하고, 필요에 따라 RAG나 tool을 거친 뒤, response node가 최종 답변을 만드는 구조다.

예를 들어 사용자가 스케줄을 물어보면 router는 이것을 일반 대화가 아니라 tool 실행이 필요한 요청으로 분류한다. tool node는 Supabase에서 데이터를 조회하고, response node는 조회 결과를 루미 페르소나에 맞게 정리한다.

## 핵심 개념

> **요약**
> LangGraph를 활용하여 에이전트 MVP를 구현한다. State, Node, Edge, Router를 기반으로 `chat`, `rag`, `tool` 경로를 나누고, FastAPI와 연결해 실제 요청을 처리하는 구조를 만든다.

## 주요 내용

### 1. LangGraph 핵심 개념
- **State**: 그래프 전체에서 공유되는 상태
- **Node**: 각 처리 단계 (함수)
- **Edge**: 노드 간 연결 (조건부 분기 포함)
- **Graph**: 노드와 엣지의 조합
- **Router**: 현재 상태를 보고 다음 실행 경로를 결정하는 함수

### 2. MVP 구현
- 그래프 상태 정의 (TypedDict)
- 노드 구현: chat, rag, tool_call
- 조건부 엣지를 통한 라우팅
- 체크포인터를 통한 대화 이력 관리

### 3. idol-agent v0.2
- FastAPI + LangGraph 통합
- RAG 노드: Supabase pgvector 검색
- 채팅 노드: Solar LLM 호출
- API 엔드포인트 설계

## 그래프 흐름

LangGraph MVP의 기본 흐름은 다음처럼 정리된다.

```text
START
  -> router
  -> intent=chat  -> response
  -> intent=rag   -> rag      -> response
  -> intent=tool  -> tool     -> response
  -> END
```

이 구조에서 핵심은 router가 모든 것을 직접 처리하지 않는다는 점이다. router는 “어디로 보낼지”만 결정하고, 실제 검색은 rag node, 실제 행동은 tool node, 말투와 최종 응답은 response node가 담당한다.

## 구현 관점

강의자료의 개발 흐름은 `state -> nodes -> router/prompts -> repository -> executor -> graph -> ui`로 이어진다. 이 순서를 지키면 Graph의 책임이 분명해진다.

| 구성요소 | 역할 |
|---|---|
| State | messages, intent, retrieved_docs, tool_result 같은 공유 데이터 |
| Router node | 사용자 의도 분류 |
| RAG node | Supabase pgvector에서 관련 문서 검색 |
| Tool node | 스케줄 조회, 팬레터 저장 같은 action 수행 |
| Response node | 최종 답변 생성 |
| Repository | DB 접근 로직 분리 |

MVP 단계에서는 모든 예외를 완벽히 처리하기보다, 경로가 명확히 나뉘고 각 node가 상태를 어떻게 읽고 쓰는지 확인하는 것이 중요하다. 이후 streaming, CI/CD, LLMOps에서 이 구조를 점진적으로 강화한다.

## 실습/코드

- Day2 Mission: idol-agent v0.2 (LangGraph MVP) - `Week08/Day02/day2-mission/`

## 흐름도

![LangGraph MVP 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agent-architecture-w08d02-langgraph-mvp-diagram-1.svg)

## 체크포인트

- router는 응답을 만드는 node가 아니라 경로를 결정하는 node다.
- RAG와 Tool은 둘 다 외부 지식을 쓰지만 목적이 다르다. RAG는 문서를 찾고, Tool은 행동을 수행한다.
- State에 어떤 값을 넣을지 정하지 않으면 node 간 계약이 흐려진다.
- MVP에서는 동작하는 end-to-end 흐름을 먼저 만들고, 다음 단계에서 streaming과 운영 안정성을 붙인다.

## 관련 글

- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [FastAPI]({% post_url 2026-01-07-upstage-tech-fastapi %})
- [Supabase]({% post_url 2026-02-10-upstage-tech-supabase %})
