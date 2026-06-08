---
title: "AI Agent 완벽 가이드 1: 정의와 Workflow 구분"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- workflow
- tools
- memory
toc: true
date: 2026-04-04 01:00:00 +0900
comments: false
mermaid: true
math: true
---
2026 Complete Guide

# AI Agent 완벽 가이드

단순 LLM 호출부터 멀티 에이전트 시스템까지 7단계로 이해하는 AI Agent의 모든 것

7

Agent Levels

10+

핵심 논문

6

주요 프레임워크

레벨별 가이드 보기

Overview 먼저 보기

Overview

## AI Agent란 무엇인가?

Lilian Weng(OpenAI)의 정의에 따르면, AI Agent는 네 가지 핵심 요소의 조합입니다

Agent

=

LLM

+

Memory

+

Planning

+

Tools

Source: Lilian Weng, "LLM Powered Autonomous Agents" (June 2023)

### LLM (두뇌)

추론과 의사결정의 핵심 엔진. 자연어를 이해하고, 계획을 세우고, 도구 사용을 결정합니다.

Core Engine

### Memory (기억)

단기 기억(컨텍스트 윈도우)과 장기 기억(벡터 DB). 경험을 축적하고 과거를 참조합니다.

State Management

### Planning (계획)

작업 분해(Task Decomposition)와 자기 반성(Reflection). 복잡한 목표를 실행 가능한 단계로 쪼갭니다.

Strategy

### Tools (도구)

외부 API, 검색 엔진, 코드 실행기 등. LLM의 능력을 실제 세계로 확장합니다.

External Actions

### Workflow vs Agent: 핵심 구분

Anthropic의 "Building Effective Agents"(2024)에서는 Workflow와 Agent를 명확히 구분합니다

#### Workflow

Deterministic

- 실행 흐름이 코드로 미리 정의됨

- 같은 입력 = 같은 경로

- 예측 가능하고 디버깅이 쉬움

- 대부분의 비즈니스 문제에 적합

- 비용이 예측 가능

예시:

문서 번역 파이프라인, 이메일 분류 시스템

vs

#### Agent

Dynamic

- 실행 흐름을 LLM이 동적으로 결정

- 같은 입력이라도 다른 경로 가능

- 관찰(Observability) 도구 필요

- Open-ended 문제에 강함

- 비용이 가변적

예시:

코드 디버깅 에이전트, 리서치 에이전트

---

## 추가 정리

### 핵심 요약

Workflow는 사람이 정한 경로를 LLM이 따라가는 구조이고, Agent는 실행 중에 다음 행동을 스스로 선택하는 구조다. 두 개념을 섞어 쓰면 설계 복잡도가 불필요하게 올라간다.

### 보충 해설

실무에서는 먼저 Workflow로 충분한지 판단해야 한다. 입력 유형이 명확하고 단계가 고정되어 있으면 Workflow가 더 안전하고 저렴하다. Agent가 필요한 경우는 도구 선택, 경로 선택, 재시도 판단이 실행 중에 달라지는 문제다.
