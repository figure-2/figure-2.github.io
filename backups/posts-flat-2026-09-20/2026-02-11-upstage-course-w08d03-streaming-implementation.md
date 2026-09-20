---
title: "Streaming 구현"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-8.AGENT_ARCHITECTURE
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- streaming
- course-note
- agent-architecture
toc: true
date: 2026-02-11 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Streaming 구현

## 수업 위치

이 수업은 LangGraph MVP가 동작한 이후 사용자 경험을 개선하는 단계다. MVP에서는 `ainvoke`로 그래프 전체 실행이 끝난 뒤 응답을 받기 때문에, 처리 시간이 길면 사용자는 아무것도 보지 못한다. Streaming은 이 대기 시간을 줄이는 것이 아니라, 사용자가 진행 상황을 느끼게 만드는 방식이다.

## 핵심 개념

> **요약**
> 챗봇 서비스에서 사용자 경험을 향상시키기 위해 노드 스트리밍과 토큰 스트리밍을 구현한다. LangGraph의 streaming API, FastAPI `StreamingResponse`, SSE 이벤트 포맷, Gradio 또는 클라이언트 UI 수신 처리를 함께 다룬다.

## 주요 내용

### 1. 스트리밍 기초
- 일반 응답 vs 스트리밍 응답의 차이
- 사용자 경험 관점에서의 스트리밍 필요성
- SSE (Server-Sent Events) 프로토콜
- node streaming과 token streaming의 차이

### 2. FastAPI 스트리밍
- `StreamingResponse` 활용
- async generator를 통한 토큰 단위 전송
- SSE 이벤트 포맷팅
- 에러 핸들링과 연결 관리

### 3. LangGraph 스트리밍
- LangGraph의 `astream` / `astream_events`
- 노드별 스트리밍 출력 처리
- 중간 상태 전달 (node 시작/완료 이벤트)
- 관련: LangGraph

### 4. 프론트엔드 통합
- Gradio에서의 스트리밍 수신
- EventSource API 활용
- 실시간 UI 업데이트

## 두 가지 스트리밍

강의자료에서 구분한 스트리밍은 두 종류다.

| 종류 | 의미 | 사용자에게 보이는 것 |
|---|---|---|
| Node streaming | LangGraph의 각 node 진행 상태를 전송 | “의도 파악 중”, “스케줄 조회 중”, “응답 생성 중” |
| Token streaming | 모델이 생성한 토큰을 순차적으로 전송 | 답변 문장이 조금씩 나타남 |

Node streaming은 에이전트가 지금 무엇을 하고 있는지 알려준다. Token streaming은 최종 답변이 만들어지는 과정을 바로 보여준다. 둘은 경쟁 관계가 아니라 함께 쓰면 좋다. 긴 tool call이나 RAG 검색이 있는 서비스에서는 node streaming만 있어도 체감 대기 시간이 줄어든다.

## 구현 흐름

서비스 구조로 보면 streaming은 “문지기 역할의 API 서버가 클라이언트에게 진행 상황을 계속 보내는 것”이다.

```text
client
  -> FastAPI streaming endpoint
  -> LangGraph astream / astream_events
  -> SSE event
  -> UI update
```

SSE는 서버에서 클라이언트로 이어지는 단방향 스트림에 적합하다. 채팅 입력은 일반 HTTP 요청으로 받고, 응답 진행 상황은 SSE로 흘려보내는 구조를 만들 수 있다.

## 체크포인트

- streaming은 실제 처리 시간을 줄이는 기능이 아니라 체감 대기 시간을 줄이는 기능이다.
- node streaming과 token streaming은 목적이 다르므로 이벤트 타입을 구분해야 한다.
- 연결이 중간에 끊겼을 때 서버 리소스가 계속 소모되지 않도록 처리해야 한다.
- error event도 스트림 안에서 안전하게 전달해야 한다.
- raw 내부 상태나 secret이 이벤트 payload로 나가지 않도록 주의한다.

## 흐름도

![Streaming 구현 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agent-architecture-w08d03-streaming-implementation-diagram-1.svg)

## 관련 글

- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [FastAPI]({% post_url 2026-01-07-upstage-tech-fastapi %})
- [Gradio]({% post_url 2026-02-11-upstage-tech-gradio %})
