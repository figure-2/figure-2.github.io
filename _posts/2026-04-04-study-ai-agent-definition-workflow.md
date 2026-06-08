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
# AI Agent 완벽 가이드 1: 정의와 Workflow 구분

> **한줄 정의**
> AI Agent는 LLM에 memory, planning, tools를 붙여 목표 달성을 위해 판단하고 실행하는 시스템이다.

## Agent의 4요소

원본 학습 노트는 Agent를 다음 조합으로 정리한다.

```text
Agent = LLM + Memory + Planning + Tools
```

![AI Agent components](/assets/images/study/diagrams/study-ai-agent-components.svg){: width="100%"}

| 구성요소 | 역할 | 설계 질문 |
| --- | --- | --- |
| LLM | 추론, 계획, 생성, 도구 선택 | 어떤 판단을 모델에게 맡길 것인가 |
| Memory | 현재 context와 장기 정보 유지 | 무엇을 기억하고 무엇을 버릴 것인가 |
| Planning | 목표를 단계로 분해하고 재계획 | 계획은 고정할 것인가, 동적으로 만들 것인가 |
| Tools | API, DB, 검색, 코드 실행 등 외부 행동 | 어떤 도구를 어떤 권한으로 허용할 것인가 |

LLM만 있으면 chatbot에 가깝다. Agent가 되려면 외부 상태를 읽고, 도구를 호출하고, 그 결과에 따라 다음 행동을 바꿀 수 있어야 한다.

## Workflow와 Agent의 차이

Agent를 이해할 때 가장 중요한 구분은 workflow와 agent다.

| 관점 | Workflow | Agent |
| --- | --- | --- |
| 실행 흐름 | 코드로 미리 정의 | LLM이 동적으로 결정 |
| 같은 입력의 경로 | 대체로 동일 | 달라질 수 있음 |
| 예측 가능성 | 높음 | 낮음 |
| 디버깅 | 쉬움 | trace와 observability 필요 |
| 비용 | 예측 가능 | 반복과 도구 호출로 가변 |
| 적합한 문제 | 명확한 비즈니스 프로세스 | open-ended 문제 |
| 예시 | 문서 번역 pipeline, 이메일 분류 | 코드 디버깅 agent, research agent |

대부분의 업무는 Agent가 아니라 Workflow로 충분하다. 동적 판단이 꼭 필요할 때만 Agent로 올리는 것이 안전하다.

## Workflow 예시

문서 번역 workflow는 고정된 순서를 갖는다.

```text
Input Document
  -> Analyze
  -> Draft Translation
  -> Term Check
  -> Polish
  -> Output
```

각 단계에서 LLM을 쓸 수 있지만, 전체 흐름은 코드가 결정한다. 실패 위치도 상대적으로 명확하다.

## Agent 예시

Research Agent는 같은 질문이라도 검색 결과에 따라 다른 경로를 선택한다.

```text
Goal
  -> Think
  -> Search
  -> Observe
  -> Decide next tool
  -> Search again or summarize
  -> Final answer
```

여기서 핵심은 `Observe` 이후 다음 행동이 바뀐다는 점이다.

## 언제 Agent가 필요한가

| 상황 | Agent 필요도 | 이유 |
| --- | --- | --- |
| 단일 질문 답변 | 낮음 | 일반 LLM 호출 또는 RAG로 충분 |
| 정해진 절차 처리 | 낮음 | Workflow가 더 예측 가능 |
| 입력 유형별 분기 | 중간 | Router workflow로 해결 가능 |
| 긴 작업 중 재계획 | 높음 | 중간 결과에 따라 다음 행동이 달라짐 |
| 여러 도구를 선택 실행 | 높음 | tool selection과 observation loop 필요 |
| 실패 후 복구 | 높음 | retry, fallback, replanning이 필요 |

## Agent 도입 전 질문

Agent는 강력하지만 기본값이 되어서는 안 된다.

| 질문 | No라면 |
| --- | --- |
| 작업이 open-ended인가 | Workflow로 시작 |
| 중간 결과에 따라 경로가 달라지는가 | Prompt chaining 또는 routing 사용 |
| 도구 호출 실패를 복구해야 하는가 | 단순 tool call로 충분 |
| 사람 승인 경계가 정의됐는가 | Agent 자동 실행 금지 |
| trace와 cost를 기록할 수 있는가 | production agent 금지 |

## 내 기준

Agent는 자율성이 아니라 책임 경계의 문제다.

```text
Model decides
  -> Code constrains
  -> Tool enforces
  -> Human approves
  -> Log explains
```

이 다섯 가지가 없으면 agent는 실행 시스템이 아니라 예측하기 어려운 loop다.

## 다음 글

- [AI Agent 완벽 가이드 2: Agent 성숙도 7단계]({% post_url 2026-04-04-study-ai-agent-maturity-levels %})
- [AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost]({% post_url 2026-04-04-study-ai-agent-architecture-operations %})
