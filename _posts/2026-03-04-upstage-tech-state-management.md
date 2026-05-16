---
title: "상태관리 (State Management)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- state-management
- tech-note
- llmops
toc: true
date: 2026-03-04 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 상태관리 (State Management)

> **한줄 정의**
> LangGraph 에이전트의 실행 상태를 저장, 복원, trimming, 영속화하여 멀티 턴 대화와 운영 추적을 가능하게 하는 메커니즘.

## 핵심 이해

LangGraph에서 상태는 그래프 전체가 공유하는 데이터 구조다. 각 node는 상태를 읽고 필요한 값을 업데이트한다. 예를 들어 router node는 `intent`를 기록하고, rag node는 `retrieved_docs`를 추가하고, response node는 최종 답변과 모델 호출 결과를 남길 수 있다.

상태관리가 중요한 이유는 LLM이 기본적으로 stateless하기 때문이다. 모델은 이전 요청을 기억하지 못하므로, 애플리케이션이 필요한 맥락을 상태로 관리하고 다시 prompt에 넣어줘야 한다.

다만 모든 상태를 무한히 저장하는 것은 답이 아니다. context window, 비용, 개인정보, 응답 품질을 함께 고려해야 한다.

## 언제 쓰는지

상태관리는 다음 상황에서 필요하다.

- 멀티 턴 대화에서 이전 발화를 기억해야 할 때
- tool 실행 결과를 다음 node에서 사용해야 할 때
- RAG 검색 결과를 response node까지 전달해야 할 때
- 사용자의 세션을 분리해야 할 때
- 서비스 재시작 후에도 대화 상태를 복원해야 할 때
- 비용 추적과 token trimming이 필요한 때

## 구현 관점

상태관리는 세 층으로 나눠서 보면 정리하기 쉽다.

| 층 | 예시 | 목적 |
|---|---|---|
| 실행 상태 | intent, selected_tool, retrieved_docs | 현재 graph 실행에 필요한 데이터 |
| 대화 상태 | messages, session_id | 멀티 턴 대화 유지 |
| 운영 상태 | token_usage, cost, latency, error | 비용 추적과 장애 분석 |

LangGraph checkpoint는 이 상태를 저장하고 복원하는 기능이다. 개발 단계에서는 MemorySaver가 빠르지만, 운영에서는 프로세스 재시작과 서버 확장을 고려해야 하므로 PostgreSQL 기반 checkpointer가 더 적합하다.

## 주의점

상태를 오래 저장할수록 사용자 경험은 좋아질 수 있지만, 보안과 비용 리스크도 커진다. 특히 prompt에 넣었던 사용자 입력, 검색 문서, tool 결과가 상태 저장소에 남을 수 있으므로 저장 범위와 보관 기간을 정해야 한다.

메시지 trimming은 단순히 오래된 메시지를 잘라내는 작업이 아니다. 시스템 지시문, 최근 사용자 의도, tool 결과, RAG 근거처럼 답변 품질에 직접 영향을 주는 정보는 보존해야 한다. 반대로 감탄사, 중복 질문, 이미 요약된 내용은 줄일 수 있다.

## 관련 강의

- [상태 관리 수업 기록]({% post_url 2026-03-04-upstage-course-w09d02-state-management %})
- [상태관리 + 비용 추적 실습]({% post_url 2026-03-04-upstage-practice-state-management-practice %})
- [idol-agent v0.7]({% post_url 2026-03-04-upstage-project-idol-agent-v07 %})

## 구조/흐름

![상태관리 (State Management) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-state-management-diagram-1.svg)

## 관련 개념

- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [Supabase]({% post_url 2026-02-10-upstage-tech-supabase %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
