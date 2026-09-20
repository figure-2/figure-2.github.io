---
title: "상태관리 + 비용 추적 실습"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- PRACTICE
tags:
- upstage
- sesac
- ai-agent
- state-management
- langgraph
- practice
- llmops
toc: true
date: 2026-03-04 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## 상태관리 + 비용 추적 실습

> **실습 정보**
> - **주차**: Week 09, Day 02
> - **유형**: 코드 구현 (Python)
> - **상태**: 완료

## 실습 목표

LangGraph checkpoint를 이용해 에이전트의 대화 상태를 저장하고, token usage 기반으로 비용을 추적하며, context window 초과를 막기 위해 메시지를 trimming하는 흐름을 구현한다.

## 핵심 구현 사항
- Checkpointer 선택 로직 (MemorySaver vs PostgreSQL)
- tiktoken 기반 토큰 수 계산
- Discord Webhook 비용 알림
- 메시지 트리밍 for 루프

## 시작점

이 실습은 LiteLLM 실습 다음 단계다. Day01에서 모델 호출 안정성을 다뤘다면, Day02에서는 “호출 전후에 쌓이는 상태와 비용을 어떻게 관리할 것인가”를 다룬다.

에이전트의 상태에는 단순 대화 이력만 있는 것이 아니다. router가 판단한 intent, RAG 검색 결과, tool 실행 결과, 모델 응답, token 사용량이 모두 다음 판단의 근거가 될 수 있다. 그래서 상태 관리는 memory 기능이 아니라 운영 데이터 관리에 가깝다.

## 진행 순서

1. Checkpointer 선택 기준 정리

   개발 단계에서는 MemorySaver로 빠르게 확인할 수 있다. 운영 단계에서는 프로세스가 재시작되어도 상태가 남아야 하므로 PostgreSQL 또는 Supabase 기반 저장소가 필요하다.

2. 세션 단위 정하기

   상태는 사용자 전체에 하나로 묶으면 안 된다. 사용자, 대화방, 작업 단위 중 무엇을 session id로 볼지 정해야 한다. 이 기준이 흐리면 서로 다른 대화가 섞일 수 있다.

3. Checkpoint 직렬화 확인

   LangChain message 객체는 단순 dict와 다르게 복원 과정이 중요하다. 저장할 때는 DB에 들어갈 수 있는 형태로 바꾸고, 꺼낼 때는 HumanMessage, AIMessage 같은 원래 객체 형태로 돌아와야 LangGraph가 정상 동작한다.

4. Token usage 계산

   모델 호출 전후로 token 사용량을 기록한다. 비용 추적은 나중에 한 번 계산하는 것이 아니라 호출 단위로 남겨야 한다.

5. 메시지 trimming 구현

   모든 대화 이력을 prompt에 넣으면 비용이 증가하고 context window를 초과할 수 있다. 오래된 메시지를 제거하되, 시스템 지시문, 최근 사용자 요청, tool 결과처럼 중요한 정보는 보존해야 한다.

6. 알림 기준 만들기

   daily cost limit 또는 max context token을 초과하면 Discord Webhook 같은 알림 채널로 알려준다. 단, 알림에는 API key, raw prompt, 개인정보를 포함하지 않는다.

## 체크포인트

- [ ] MemorySaver와 PostgreSQL checkpointer의 차이를 설명할 수 있다.
- [ ] session id 기준을 정하고 그 이유를 설명할 수 있다.
- [ ] token usage가 비용 계산에 어떻게 연결되는지 설명할 수 있다.
- [ ] trimming 후에도 필요한 대화 맥락이 보존되는지 확인했다.
- [ ] 비용 알림에 secret이나 raw prompt가 포함되지 않는지 확인했다.

## 회고 질문

- 상태를 많이 저장할수록 좋아지는가, 아니면 리스크도 커지는가?
- 사용자의 이전 발화를 어느 시점까지 유지해야 하는가?
- 비용 초과 알림은 사용자 경험을 해치지 않으면서 어디까지 자동화할 수 있는가?
- trimming 로직이 답변 품질을 떨어뜨렸는지 확인하려면 어떤 평가가 필요한가?

## 관련 글

- [상태 관리]({% post_url 2026-03-04-upstage-course-w09d02-state-management %})
- [상태관리 기술 노트]({% post_url 2026-03-04-upstage-tech-state-management %})
- [idol-agent v0.7]({% post_url 2026-03-04-upstage-project-idol-agent-v07 %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
