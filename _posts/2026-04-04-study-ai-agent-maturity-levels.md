---
title: "AI Agent 완벽 가이드 2: Agent 성숙도 7단계"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- maturity-model
- react
- multi-agent
toc: true
date: 2026-04-04 01:10:00 +0900
comments: false
mermaid: true
math: true
---
# AI Agent 완벽 가이드 2: Agent 성숙도 7단계

> **한줄 정의**
> Agent 성숙도는 LLM 호출이 얼마나 자율적으로 도구를 쓰고, 흐름을 결정하고, 실패를 복구하는지에 따라 올라간다.

## 단계 요약

| 레벨 | 이름 | 자율성 | 도구 사용 | 흐름 결정 | 대표 기술 |
| --- | --- | --- | --- | --- | --- |
| L0 | Simple LLM Call | 없음 | 없음 | 코드 | 기본 ChatGPT/Claude 호출 |
| L1 | Augmented LLM | 낮음 | 단일 턴 | 코드 | RAG, Function Calling |
| L2 | Chained / Sequential Agent | 낮음 | 순차 | 코드 | LangChain Chains |
| L3 | Router / Branching Agent | 중간 | 분기 | 코드 + LLM | Semantic Router |
| L4 | ReAct / Loop Agent | 높음 | loop | LLM | ReAct, Claude Tool Use |
| L5 | Planning Agent | 높음 | 계획 + loop | LLM | Plan-and-Execute, ADK |
| L6 | Multi-Agent System | 매우 높음 | 분산 | 다수 LLM | CrewAI, AutoGen |

## L0. Simple LLM Call

```text
User Prompt -> LLM -> Response
```

외부 도구, memory, loop가 없다. 빠르고 단순하지만 최신 정보나 사내 데이터에는 접근하지 못한다.

| 사용 예시 | 한계 |
| --- | --- |
| 이메일 초안 | 최신 정보 접근 불가 |
| 마케팅 카피 | 사내 데이터 참조 불가 |
| 코드 리뷰 초안 | 근거 없는 판단 가능 |

## L1. Augmented LLM

```text
User Query
  -> LLM
  -> Tool Call
  -> Tool Result
  -> Final Answer
```

LLM이 필요할 때 외부 도구를 한 번 호출한다. RAG와 function calling이 여기에 속한다.

| 특징 | 설명 |
| --- | --- |
| 장점 | 외부 데이터 접근 가능 |
| 한계 | 결과가 부족해도 재시도하지 않음 |
| 예시 | 날씨 API, DB query, naive RAG |

## L2. Chained / Sequential Agent

순서가 고정된 pipeline이다.

```text
Analyze -> Draft -> Validate -> Polish -> Output
```

각 단계에서 LLM을 쓸 수 있지만, 단계 순서는 코드가 정한다.

| 장점 | 한계 |
| --- | --- |
| 예측 가능하고 디버깅 쉬움 | 분기와 재계획이 약함 |
| gate로 품질 관리 가능 | 간단한 요청도 전체 pipeline 통과 |

## L3. Router / Branching Agent

입력을 분류해 적절한 workflow로 보낸다.

```text
Input
  -> Router
  -> Billing Handler
  -> Tech Handler
  -> FAQ Handler
```

| 적합한 경우 | 예시 |
| --- | --- |
| 입력 유형이 명확히 나뉨 | 고객 문의 분류 |
| handler별 prompt와 도구가 다름 | 환불, 기술지원, 일반 문의 |
| 비용 최적화가 필요 | router는 작은 모델, handler는 적절한 모델 |

## L4. ReAct / Loop Agent

여기서부터 진짜 agent에 가깝다.

![ReAct loop](/assets/images/study/diagrams/study-ai-agent-react-loop.svg){: width="100%"}

```text
Thought -> Action -> Observe -> Thought -> ... -> Answer
```

| 구성 | 의미 |
| --- | --- |
| Thought | 상황 분석과 다음 행동 판단 |
| Action | 도구 선택과 실행 |
| Observe | 실행 결과 확인 |
| Answer | 충분하면 종료 |

필수 안전장치는 `max_iterations`다. 종료 조건이 없으면 loop가 비용과 시간을 계속 소비한다.

## L5. Planning Agent

Planning Agent는 실행 전에 전체 계획을 세우고, 중간 실패에 따라 계획을 수정한다.

```text
Plan
  -> Execute Task 1
  -> Execute Task 2
  -> Reflect
  -> Re-plan
  -> Continue
```

| 적합한 작업 | 이유 |
| --- | --- |
| 경쟁사 분석 보고서 | 정보 수집, 비교, 작성이 순차적이지만 중간 재검색 필요 |
| 코드베이스 migration | 많은 파일을 나누어 처리하고 실패 시 재계획 필요 |
| 긴 research task | 하위 질문이 실행 중 바뀜 |

## L6. Multi-Agent System

여러 agent가 역할별로 협업한다.

```text
Orchestrator
  -> Researcher
  -> Coder
  -> Reviewer
  -> Writer
```

| 패턴 | 설명 |
| --- | --- |
| Orchestrator-Worker | 중앙 agent가 작업을 나누고 결과를 통합 |
| Evaluator-Optimizer | generator가 만들고 evaluator가 평가 후 수정 |
| Debate / Adversarial | 서로 다른 agent가 주장과 반론을 만들고 moderator가 판정 |

Multi-Agent는 context window 한계를 줄이고 역할 전문화를 줄 수 있지만, debugging과 cost가 크게 늘어난다.

## 단계 선택 기준

| 상황 | 권장 단계 |
| --- | --- |
| 단순 생성 | L0 |
| 외부 데이터 한 번 조회 | L1 |
| 고정 절차 | L2 |
| 입력별 분기 | L3 |
| 도구 결과에 따라 재시도 | L4 |
| 장기 목표와 재계획 | L5 |
| 역할별 병렬 협업 | L6 |

## 내 기준

성숙도가 높다는 것은 무조건 좋은 것이 아니다.

```text
낮은 단계
  -> 싸고 빠르고 예측 가능

높은 단계
  -> 강하지만 비싸고 관찰이 필요
```

문제가 L2로 풀리면 L4를 쓰지 않는다. 문제 해결에 필요한 최소 자율성만 추가한다.

## 다음 글

- [AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost]({% post_url 2026-04-04-study-ai-agent-architecture-operations %})
