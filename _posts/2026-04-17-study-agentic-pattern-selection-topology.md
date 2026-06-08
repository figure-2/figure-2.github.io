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
# Agentic AI 패턴 가이드 3: 선택 기준, 비용, 토폴로지

> **한줄 정의**
> Agentic pattern 선택은 단순한 것부터 시작해, 필요한 경우에만 비용과 자율성을 올리는 의사결정 문제다.

## 선택 트리

![Agentic pattern selection tree](/assets/images/study/diagrams/study-agentic-pattern-selection-tree.svg){: width="100%"}

```text
Q1. 단일 LLM + RAG로 충분한가?
  -> Yes: Single LLM + RAG
  -> No

Q2. 작업 흐름이 예측 가능한가?
  -> Yes
     -> 순차면 Prompt Chaining
     -> 분기면 Routing
     -> 병렬이면 Parallelization
     -> 반복 품질 개선이면 Evaluator-Optimizer
  -> No
     -> 동적 분해면 Orchestrator-Workers
     -> 완전 자율이면 Autonomous Agent
```

## 비용과 latency 비교

| 패턴 | 복잡도 | 비용 | latency | 예측 가능성 |
| --- | --- | --- | --- | --- |
| Single LLM | 낮음 | 낮음 | 매우 낮음 | 매우 높음 |
| Prompt Chaining | 중하 | 중 | 중 | 매우 높음 |
| Routing | 중하 | 중 | 중 | 높음 |
| Parallelization | 중하 | 중상 | 낮음 | 높음 |
| Orchestrator-Workers | 중상 | 중상 | 높음 | 중간 |
| Evaluator-Optimizer | 중상 | 중상 | 높음 | 높음 |
| Autonomous Agent | 높음 | 높음 | 높음 | 낮음 |
| Human-in-the-Loop | 중상 | 중 | 사람 대기 | 매우 높음 |

Parallelization은 비용은 늘지만 wall-clock latency는 줄일 수 있다. Evaluator loop는 latency가 늘지만 품질 기준이 명확하면 가치가 있다.

## Multi-Agent Topology 선택

| topology | 제어 구조 | 자율성 | debugging | 적합한 규모 |
| --- | --- | --- | --- | --- |
| Supervisor | 중앙 집중 | 낮음 | 쉬움 | 소, 중 |
| Swarm | peer handoff | 높음 | 어려움 | 중 |
| Hierarchical | 다층 계층 | 중간 | 중간 | 대 |

Supervisor는 가장 운영하기 쉽다. Swarm은 유연하지만 trace와 책임 분리가 어렵다. Hierarchical은 대규모 조직형 agent에 적합하다.

## 실전 유즈케이스 매핑

| 유즈케이스 | 적합한 패턴 | 이유 |
| --- | --- | --- |
| 마케팅 카피 생성 | Prompt Chaining | outline, draft, refine 단계가 명확 |
| 고객지원 자동화 | Routing | 환불, 기술, 일반 문의 분기 |
| 코드 리뷰 자동화 | Parallelization | 보안, 성능, 스타일 관점 병렬 |
| coding agent | Orchestrator-Workers | grep, read, edit, bash worker가 동적 필요 |
| 고품질 번역 | Evaluator-Optimizer | 용어, tone, 자연스러움 평가 가능 |
| browser automation | Autonomous Agent | 클릭, 입력, 관찰을 상황별 판단 |
| 결제/발송 승인 | Human-in-the-Loop | 실패 비용이 크고 승인 필요 |

## 하이브리드 조합

![Hybrid agentic flow](/assets/images/study/diagrams/study-agentic-pattern-hybrid-flow.svg){: width="100%"}

| 조합 | 설명 |
| --- | --- |
| Router -> Orchestrator | 입력 유형별 routing 후 동적 분해 |
| Chain + Evaluator | 각 단계마다 품질 gate |
| Agent + Parallelization | agent가 병렬 sub-agent 호출 |
| Agent + HITL | 자율 실행하되 irreversible action 직전 승인 |
| Hierarchical + Swarm | 상위는 계층형, 팀 내부는 swarm |
| Orchestrator + Evaluator | worker 결과를 evaluator가 검증 후 재위임 |

## 설계 원칙

| 원칙 | 의미 |
| --- | --- |
| Start simple | 단일 LLM 호출로 풀리면 멈춘다 |
| Measure before complexity | 복잡도 추가 전 평가 지표를 만든다 |
| Augmented LLM is the block | 모든 pattern은 LLM + tools + memory + retrieval 조합 |
| Guard autonomous agents | max steps, cost limit, sandbox, human check-in |
| Transparency builds trust | agent가 무엇을 왜 했는지 log로 보여준다 |
| Evaluation-driven development | agent를 만들기 전에 평가셋부터 만든다 |
| Context Engineering | long-horizon agent는 context 선택이 병목 |
| HITL for irreversible actions | 결제, 발송, 삭제, 권한 변경은 승인 경계 |
| State persistence | checkpoint가 undo와 debugging을 가능하게 한다 |

## 내 기준

패턴 선택은 아래 질문으로 끝난다.

```text
이 작업은
  고정 절차인가
  분기 문제인가
  병렬 관점 문제인가
  품질 반복 문제인가
  동적 탐색 문제인가
  사람 승인 문제인가
```

이 질문에 답하지 않고 agent framework를 고르면, 구현은 빠르지만 운영 판단은 비게 된다.

## 관련 글

- [Agentic AI 패턴 가이드 2: 8가지 패턴]({% post_url 2026-04-17-study-agentic-patterns-core %})
- [AI Agent 완벽 가이드 1: 정의와 Workflow 구분]({% post_url 2026-04-04-study-ai-agent-definition-workflow %})
