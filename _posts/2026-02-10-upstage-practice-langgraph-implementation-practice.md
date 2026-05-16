---
title: "LangGraph 에이전트 구현 실습"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-8.AGENT_ARCHITECTURE
- PRACTICE
tags:
- upstage
- sesac
- ai-agent
- langgraph
- agent-architecture
- practice
toc: true
date: 2026-02-10 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# LangGraph 에이전트 구현 실습

> **실습 정보**
> - **주차**: Week 08, Day 02
> - **유형**: 코드 구현 (Python)
> - **상태**: 완료

## 실습 목표
LangGraph 기반 에이전트 MVP 구현. State, Nodes, Edges, Graph 구성.

## 핵심 구현 사항
- AgentState 정의 (messages, intent, retrieved_docs 등)
- 노드 함수 구현 (intent_classifier, rag_retriever, tool_executor, response_generator)
- 조건부 엣지 라우팅 (chat/rag/tool)
- StateGraph 컴파일 및 FastAPI 연동

## 진행 순서

1. `LangGraph 에이전트 구현 실습`에서 확인할 핵심 개념을 먼저 정리한다.
2. 실습 목표를 작은 작업 단위로 나누고 필요한 입력, 출력, 제약 조건을 확인한다.
3. 답안 작성 또는 구현을 진행하면서 실행 결과와 판단 근거를 함께 남긴다.
4. 마지막에 체크포인트를 기준으로 빠진 부분과 다음 보완점을 정리한다.

## 체크포인트

- [ ] 실습 목표를 한 문장으로 설명할 수 있다.
- [ ] 핵심 학습 포인트가 실제 작업의 어느 부분에 쓰였는지 연결했다.
- [ ] 관련 개념 또는 수업 기록을 다시 확인했다.
- [ ] 실행 결과, 답안 근거, 회고 중 하나 이상을 남겼다.

## 회고 질문

- 이번 실습에서 가장 헷갈린 개념은 무엇이었나?
- 수업 노트만 읽을 때와 직접 실습할 때 다르게 느껴진 점은 무엇인가?
- 같은 유형의 문제를 다시 만났을 때 먼저 확인할 기준은 무엇인가?

## 관련 개념
- LangGraph · Agent-Architecture · RAG · Tool-Calling
- W08D02-LangGraph-MVP · idol-agent-v02
