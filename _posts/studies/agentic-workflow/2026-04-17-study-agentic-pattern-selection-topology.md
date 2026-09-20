---
title: "Agentic AI 패턴 가이드 3: 선택 기준, 비용, 토폴로지"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- agentic-patterns
- topology
- cost
- latency
toc: true
date: 2026-04-17 22:20:00 +0900
comments: false
mermaid: true
math: true
---

Agentic pattern은 복잡한 구조부터 고르는 것이 아니다. 비용, latency, 예측 가능성, 실패 복구 가능성을 기준으로 가장 단순한 구조부터 선택해야 한다.

## 어떤 패턴을 써야 할까?

#### 원칙

위로 갈수록 단순 · 저렴 · 예측 가능, 아래로 갈수록 복잡 · 비싸지만 강력. 항상 위쪽부터 시도하고, 한계에 부딪힐 때만 아래로 내려가세요.

### 비용/레이턴시 비교

| 패턴 | 복잡도 | 비용 | 레이턴시 | 예측가능성 |
| --- | --- | --- | --- | --- |
| Single LLM | 낮음 | 낮음 | 낮음 | 높음 |
| Prompt Chaining | 낮음~중간 | 낮음~중간 | 중간 | 높음 |
| Routing | 중간 | 중간 | 중간 | 중간~높음 |
| Parallelization | 중간 | 중간~높음 | 낮음~중간 | 중간 |
| Orchestrator-Workers | 높음 | 높음 | 높음 | 낮음~중간 |
| Evaluator-Optimizer | 높음 | 높음 | 높음 | 중간 |
| Autonomous Agent | 매우 높음 | 매우 높음 | 높음 | 낮음 |
| Human-in-the-Loop | 중간 | 중간 | 사람 대기 포함 | 높음 |

### Multi-agent 토폴로지 선택

Orchestrator-Workers를 확장할 때 3가지 구조 중 선택 — 규모와 자율성 수준이 기준.

| 토폴로지 | 제어 구조 | 자율성 | 디버깅 | 적합한 규모 |
| --- | --- | --- | --- | --- |
| Supervisor | 중앙 집중 | 낮음 | 쉬움 | 소·중 |
| Swarm | 피어 handoff | 높음 | 어려움 | 중 |
| Hierarchical | 다층 계층 | 중간 | 중간 | 대 |

---

## 추가 정리

### 핵심 요약

패턴 선택의 기본 원칙은 단순한 구조부터 시작하는 것이다. 비용과 latency, 예측 가능성을 기준으로 Workflow, Orchestrator, Autonomous Agent, Human-in-the-Loop를 선택해야 한다.

### 보충 해설

Multi-Agent 토폴로지는 멋있어 보이지만 운영 난도가 높다. Supervisor는 통제가 쉽고, Swarm은 유연하지만 디버깅이 어렵고, Hierarchical은 규모 확장에 유리하다. 선택 기준은 자율성보다 관측 가능성과 실패 복구 가능성이다.
