---
title: "Agentic AI 패턴 가이드 1: Workflow vs Agent"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- agentic-workflow
- ai-agent
- workflow
toc: true
date: 2026-04-17 22:00:00 +0900
comments: false
mermaid: true
math: true
---
# Agentic AI 패턴 가이드 1: Workflow vs Agent

> **한줄 정의**
> Workflow는 LLM을 정해진 경로로 오케스트레이션하고, Agent는 LLM이 경로와 도구를 동적으로 결정한다.

## 핵심 구분

| 관점 | Workflow | Agent |
| --- | --- | --- |
| 경로 | 미리 정의 | 실행 중 결정 |
| 예측 가능성 | 높음 | 낮음 |
| 디버깅 | 쉬움 | trace 필요 |
| 비용 | 대체로 예측 가능 | 반복과 도구 호출로 가변 |
| 실패 위험 | 제한적 | 커짐 |
| 적합한 문제 | 절차가 명확한 업무 | open-ended task |

원본 노트의 핵심 원칙은 단순하다. 대부분의 문제는 단일 LLM 호출, RAG, 잘 쓴 prompt로 해결된다. 복잡도는 측정 가능한 이득이 있을 때만 추가한다.

## Augmented LLM이 기본 블록

모든 agentic pattern은 `LLM + Tools + Memory + Retrieval`을 어떻게 엮느냐의 문제다.

```text
LLM
  + Tools
  + Memory
  + Retrieval
  = Augmented LLM
```

| 구성 | 역할 |
| --- | --- |
| LLM | reasoning core |
| Tools | function call, web, DB, API |
| Memory | state, short-term, long-term |
| Retrieval | RAG, search, vector, BM25 |

## 복잡도 추가 원칙

```text
Single LLM
  -> RAG
  -> Prompt Chaining
  -> Routing
  -> Parallelization
  -> Orchestrator
  -> Evaluator loop
  -> Autonomous Agent
  -> HITL
```

아래로 갈수록 강력하지만 비용, latency, debugging 부담이 커진다.

## Workflow가 맞는 경우

| 조건 | 예 |
| --- | --- |
| 단계가 명확함 | outline -> draft -> review |
| 실패 조건을 정의할 수 있음 | 길이, tone, 금칙어 검사 |
| 입력 유형이 제한적 | FAQ, 고객지원 분류 |
| 품질 gate를 넣기 쉬움 | 테스트, schema validation |

Workflow는 덜 멋있지만 운영에 강하다.

## Agent가 맞는 경우

| 조건 | 예 |
| --- | --- |
| 다음 행동이 중간 결과에 의존 | debugging, research |
| tool 선택이 동적 | 검색, DB, code 실행 중 선택 |
| 실패 후 재계획 필요 | data source가 없을 때 대체 경로 |
| 완료 조건이 탐색 중 결정 | open-ended investigation |

Agent를 쓰려면 반드시 max steps, 비용 한도, tool permission, human checkpoint가 필요하다.

## 내 기준

Workflow와 Agent의 차이는 "AI를 쓰냐"가 아니다. "누가 흐름을 결정하냐"다.

```text
흐름을 코드가 결정하면 Workflow
흐름을 모델이 결정하면 Agent
```

기본 선택은 Workflow다. Agent는 필요한 지점에만 부분적으로 넣는다.

## 다음 글

- [Agentic AI 패턴 가이드 2: 8가지 패턴]({% post_url 2026-04-17-study-agentic-patterns-core %})
- [Agentic AI 패턴 가이드 3: 선택 기준, 비용, 토폴로지]({% post_url 2026-04-17-study-agentic-pattern-selection-topology %})
