---
title: "AI Agent 개인 학습 로드맵"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- agent-architecture
- tool-calling
- memory
- guide-review
- reference-note
toc: true
date: 2026-04-04 00:50:00 +0900
comments: false
mermaid: true
math: true
---
# AI Agent 개인 학습 로드맵

## 학습 목적

AI Agent를 모델 호출의 확장이 아니라, memory, planning, tool use, orchestration을 가진 시스템 구조로 정리한다.

## 정리 범위

| 소주제 | 정리 관점 |
| --- | --- |
| Workflow vs Agent | 고정 흐름과 동적 의사결정의 차이 |
| Agent 구성요소 | LLM, memory, planning, tools |
| 성숙도 단계 | simple LLM call에서 multi-agent까지 |
| Memory System | short-term, long-term, episodic memory |
| Protocol | MCP, A2A 등 에이전트 연결 방식 |
| Framework | LangGraph, CrewAI, AutoGen 등 비교 기준 |
| Agent Engineering | 모델 호출 바깥의 실행 시스템 설계 |
| Agent Case Study | Hermes, OpenClaw 같은 구현 사례 비교 |
| Assistant Engineering | LLM, RAG, Agent, fine-tuning을 하나의 assistant로 묶는 실습 경로 |

## 작성할 글

| 순서 | 게시글 | 소주제 | 상태 |
| --- | --- | --- | --- |
| 1 | [AI Agent 완벽 가이드 1: 정의와 Workflow 구분]({% post_url 2026-04-04-study-ai-agent-definition-workflow %}) | Agent 4요소, Workflow vs Agent, 도입 기준 | 작성 |
| 2 | [AI Agent 완벽 가이드 2: Agent 성숙도 7단계]({% post_url 2026-04-04-study-ai-agent-maturity-levels %}) | L0~L6, ReAct, planning, multi-agent | 작성 |
| 3 | [AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost]({% post_url 2026-04-04-study-ai-agent-architecture-operations %}) | memory, agentic RAG, guardrails, MCP/A2A, framework, benchmark | 작성 |
| 4 | [Agent Engineering]({% post_url 2026-05-23-study-agent-engineering %}) | 중심 객체, plan-act-observe-verify, human-gated action, tool permission matrix, state machine, evaluator | 작성 |
| 5 | [Hermes Agent vs OpenClaw]({% post_url 2026-05-23-study-hermes-vs-openclaw %}) | Agent-first vs Gateway-first, memory, tool registry, skill system, multi-agent, MCP, 배포/운영 | 작성 |
| 6 | [AI Assistant Engineering]({% post_url 2026-04-26-study-ai-assistant-engineering %}) | LLM 기초, RAG, Agent pattern, fine-tuning, assistant 구축 실습 경로 | 작성 |

## 후속 분리 후보

| 후보 글 | 분리 이유 |
| --- | --- |
| Agent Framework 비교 | 1차 글에 반영 완료. 추후 실제 구현 비교로 분리 가능 |
| Agent 핵심 논문 읽기 | 1차 글에 요약 반영. 논문별 상세 읽기는 후속 가능 |
| Agent Benchmark 읽는 법 | 1차 글에 요약 반영. 최신성 검증이 필요한 별도 관리 대상 |

## 작성 기준

AI Agent 글은 "자율성"을 과장하지 않는다. 어떤 판단을 모델에게 맡기고, 어떤 경계는 코드와 정책으로 고정할지 분리한다.
