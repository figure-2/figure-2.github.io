---
title: "Agentic AI 패턴 가이드 2: 8가지 패턴"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- agentic-patterns
- prompt-chaining
- routing
- multi-agent
toc: true
date: 2026-04-17 22:10:00 +0900
comments: false
mermaid: true
math: true
---
# Agentic AI 패턴 가이드 2: 8가지 패턴

> **한줄 정의**
> Agentic pattern은 LLM을 한 번 호출할지, 순차로 엮을지, 분기할지, 병렬화할지, 평가 loop를 둘지, agent에게 맡길지를 고르는 선택지다.

## 전체 목록

| 번호 | 패턴 | 핵심 |
| --- | --- | --- |
| 00 | Augmented LLM | LLM + Tools + Memory + Retrieval |
| 01 | Prompt Chaining | 순차 pipeline |
| 02 | Routing | 입력 분류 후 전문 handler로 분기 |
| 03 | Parallelization | 독립 작업 병렬 실행 또는 voting |
| 04 | Orchestrator-Workers | 중앙 orchestrator가 worker에게 동적 위임 |
| 05 | Evaluator-Optimizer | 생성과 평가를 반복 |
| 06 | Autonomous Agent | LLM이 경로와 도구를 자율 선택 |
| 07 | Human-in-the-Loop | 위험 행동 직전 사람 승인 |

## 00. Augmented LLM

모든 패턴의 기본 블록이다.

```text
LLM
  -> Tools
  -> Memory
  -> Retrieval
```

LLM이 tool, memory, retrieval을 언제 어떻게 쓸지 결정한다. 단일 call이지만 외부 세계와 연결된다.

## 01. Prompt Chaining

작업을 순차 단계로 나눈다.

```text
Brief
  -> Outline
  -> Gate
  -> Draft
  -> Refine
  -> Final
```

| 관점 | 내용 |
| --- | --- |
| 언제 쓰나 | 작업을 고정 단계로 나눌 수 있을 때 |
| 예 | 마케팅 카피, 번역, 보고서 초안 |
| 장점 | 단계별 검증 가능 |
| 실패 모드 | 앞 단계 오류가 뒤로 전파 |

Gate를 넣어 조건 미달 시 retry하거나 종료한다.

## 02. Routing

입력을 분류해 전문 handler로 보낸다.

```text
Ticket
  -> Router
  -> Billing Handler
  -> Tech Handler
  -> FAQ Handler
```

| 관점 | 내용 |
| --- | --- |
| 언제 쓰나 | 입력 유형이 명확히 나뉠 때 |
| 예 | 고객지원, 문서 분류, 모델 routing |
| 장점 | handler별 prompt/tool/model 최적화 |
| 비용 전략 | router는 작은 모델, handler는 필요한 모델 |

## 03. Parallelization

작업을 병렬로 나눠 실행한다.

| 변형 | 설명 | 예 |
| --- | --- | --- |
| Sectioning | 서로 다른 관점을 나눠 병렬 처리 | 보안/성능/스타일 code review |
| Voting | 같은 작업을 여러 번 실행 후 다수결 | fact-check, 분류 신뢰도 향상 |

장점은 latency를 줄이고 관점별 집중도를 높이는 것이다. 단점은 비용이 늘어난다.

## 04. Orchestrator-Workers

중앙 orchestrator가 runtime에 하위 작업을 분해하고 worker에게 위임한다.

```text
User Goal
  -> Orchestrator
  -> Grep Worker
  -> Read Worker
  -> Edit Worker
  -> Bash Worker
  -> Verify
```

| 관점 | 내용 |
| --- | --- |
| 언제 쓰나 | subtask가 실행 중 결정될 때 |
| 예 | coding agent, research agent |
| 핵심 | shared context와 재계획 |
| 차이 | Parallelization은 고정 분할, Orchestrator는 동적 분할 |

## 04+. Multi-Agent Topologies

Orchestrator-Workers는 Supervisor topology의 한 형태다.

| topology | 제어 구조 | 특징 |
| --- | --- | --- |
| Supervisor | 중앙 집중 | 명확한 역할 분담, 결과 통합 |
| Swarm | peer handoff | 유연하지만 debugging 어려움 |
| Hierarchical | 다층 구조 | 대규모 책임 격리 |

## 05. Evaluator-Optimizer

Generator가 만들고 Evaluator가 평가해 반복 개선한다.

```text
Generate
  -> Evaluate
  -> Feedback
  -> Improve
  -> Stop when pass
```

| 언제 쓰나 | 예 |
| --- | --- |
| 품질 기준이 명확할 때 | 번역, 코드 리뷰, 답변 품질 개선 |
| 반복 개선이 가치 있을 때 | 고품질 문서, 고객 응대 |
| 자동 평가가 가능할 때 | schema, test, rubric |

평가 기준이 없으면 loop가 비용만 쓰고 좋아졌는지 알 수 없다.

## 06. Autonomous Agent

목표만 주고 LLM이 경로, 도구, 종료를 판단한다.

| 장점 | 위험 |
| --- | --- |
| open-ended task에 강함 | 비용과 latency 예측 어려움 |
| 사람의 단계 설계 부담 감소 | tool misuse, infinite loop |
| 복잡한 탐색 가능 | debugging 어려움 |

필수 안전장치:

- max steps
- cost limit
- sandbox
- permission gate
- trace

## 07. Human-in-the-Loop

실패 비용이 큰 행동 직전 사람을 넣는다.

| 작업 | 승인 필요 이유 |
| --- | --- |
| 결제 | 금전 피해 |
| 발송 | 외부 커뮤니케이션 사고 |
| 삭제 | 복구 불가능성 |
| 권한 변경 | 보안 사고 |
| 배포 | 사용자 영향 |

HITL은 agent 성능 부족의 보완책이 아니라 production 표준 안전장치다.

## 내 기준

패턴 선택은 멋이 아니라 실패 비용의 문제다.

```text
단순하면 단일 호출
절차가 있으면 chain
유형이 갈리면 routing
관점이 나뉘면 parallel
동적으로 쪼개야 하면 orchestrator
품질 기준이 있으면 evaluator
실패 비용이 크면 HITL
```

## 다음 글

- [Agentic AI 패턴 가이드 3: 선택 기준, 비용, 토폴로지]({% post_url 2026-04-17-study-agentic-pattern-selection-topology %})
