---
title: "Agent Engineering"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- agent-engineering
- tool-permission
- evaluation
toc: true
date: 2026-05-23 14:00:00 +0900
comments: false
mermaid: true
math: true
---
# Agent Engineering

> **한줄 정의**
> Agent Engineering은 LLM 앱 개발이 아니라 model 바깥의 실행 시스템, 권한, 상태, 평가, 관측성을 설계하는 일이다.

## 대화에서 실행으로

Agent system은 단일 model call보다 많은 구성요소를 가진다.

| 구성요소 | 역할 |
| --- | --- |
| Model | 추론, 계획, 생성 |
| Tools | API, DB, browser, code execution, SaaS connector |
| Context | 파일, 대화, 사용자 상태, 조직 데이터 |
| Memory | 단기/장기 기억, 선호, 작업 이력 |
| State | 진행 중인 작업, 실패, 재시도 상태 |
| Policy | 권한, 금지 행동, 승인 조건 |
| Evaluation | 결과 검증과 품질 측정 |
| Observability | log, trace, cost, latency |
| Human-in-the-loop | 승인, 수정, 중단, rollback |

Agent Engineering의 중심은 model 성능이 아니라 이 구성요소 사이의 계약이다.

## 중심 객체가 모든 것을 결정한다

| 조직/제품 유형 | 중심 객체 | 엔지니어링 차별점 |
| --- | --- | --- |
| Google형 | 사용자, 일정, 문서, 검색 결과 | multi-app context와 개인화 |
| Claude형 | 전문 업무 문서, 분석 task | 고신뢰 vertical workflow |
| Cursor형 | codebase, diff, branch, PR | 코드 변경 실행과 검증 loop |
| OpenAI형 | tool, runtime, agent step | 범용 runtime과 tool abstraction |
| Microsoft형 | agent identity, policy, tenant | governance와 control plane |
| ServiceNow형 | ticket, workflow, incident | 상태 기반 업무 process orchestration |
| Salesforce형 | customer, account, case | domain object 기반 CRM agent |

중심 객체가 다르면 context, action, state, permission, evaluation이 모두 달라진다.

## 설계 질문

| 질문 | 답해야 하는 것 |
| --- | --- |
| Context는 어디서 오는가 | repo, CRM, 문서, mail, ticket |
| Action은 무엇인가 | PR 생성, mail 발송, case update, API 호출 |
| State는 어떻게 변하는가 | pending, running, blocked, approved, done |
| Permission은 어디에 붙는가 | user, agent, tool, object |
| Evaluation은 무엇으로 하는가 | test, grounding, SLA, 매출, CSAT |
| Human review는 어디서 필요한가 | 발송 전, 배포 전, 고객 영향 전 |

## 6가지 핵심 설계 패턴

### 1. Plan -> Act -> Observe -> Verify

```text
Plan
  -> Act
  -> Observe
  -> Verify
  -> Replan or Done
```

실행 전 계획, 실행 후 관찰, 종료 전 검증을 분리한다.

### 2. Human-gated Action

되돌릴 수 없는 행동은 agent가 초안을 만들고 사람이 승인한다.

| 작업 | 승인 기준 |
| --- | --- |
| 외부 발송 | 수신자, 내용, 첨부 확인 |
| 결제/환불 | 금액, 계정, 정책 확인 |
| 배포 | test, diff, rollback 확인 |
| 권한 변경 | 대상, 범위, 만료 확인 |

### 3. Tool Permission Matrix

| Agent 유형 | Read | Write | External 위험도 |
| --- | --- | --- | --- |
| Research Agent | 문서, 웹 | 없음 | 낮음 |
| Coding Agent | repo read | branch write | 중간 |
| Support Agent | 고객/정책 read | ticket draft | 높음 |
| Admin Agent | config read | 권한 변경 | 매우 높음 |

도구 권한은 agent 이름이 아니라 task와 object에 붙여야 한다.

### 4. Agent State Machine

```text
created
  -> planning
  -> running
  -> waiting_approval
  -> blocked
  -> completed
  -> failed
```

상태가 없으면 재시도, 중단, rollback, monitoring이 불가능하다.

### 5. Evidence-first Response

Agent의 답변은 결론보다 근거가 먼저 검증되어야 한다.

```text
Evidence
  -> Reasoning
  -> Decision
  -> Action
```

RAG, code review, 고객지원에서는 특히 evidence ID와 action log가 함께 남아야 한다.

### 6. Evaluator as First-class Component

| Evaluator | 평가 대상 |
| --- | --- |
| Grounding evaluator | 근거와 답변 일치 여부 |
| Tool result evaluator | 도구 호출 결과 해석 정확도 |
| Safety evaluator | 금지 행동과 민감정보 노출 |
| Task completion evaluator | 목표 달성 여부 |
| Cost evaluator | token, API, 실행 시간 |
| Regression evaluator | 새 버전이 기존 성능을 깨지 않았는지 |

Evaluator는 나중에 붙이는 보조 장치가 아니라 agent system의 기본 구성요소다.

## 기술 스택

| 계층 | 예 |
| --- | --- |
| Model | general model, reasoning model, small router model |
| Runtime | graph workflow, agent SDK, custom orchestrator |
| Tooling | function calling, MCP, browser, shell, DB |
| State | checkpoint, task store, event log |
| Memory | vector DB, knowledge graph, summary memory |
| Evaluation | LLM-as-Judge, rule evaluator, test suite |
| Observability | tracing, cost dashboard, latency dashboard |
| Governance | policy engine, approval workflow, audit log |

## 차별화 축

| 축 | 핵심 질문 |
| --- | --- |
| Context depth | 얼마나 깊고 정확한 맥락을 가져오는가 |
| Tool reach | 몇 개 시스템을 안전하게 조작할 수 있는가 |
| Statefulness | 장기 작업을 안정적으로 수행하는가 |
| Trust | 근거, 검증, 승인, log가 있는가 |
| UX | 사용자가 통제감을 느끼는가 |
| Governance | 조직이 관리 가능한가 |
| Evaluation | 품질을 지속적으로 측정하는가 |
| Distribution | 이미 사용자가 있는 표면에 들어가 있는가 |

## 성숙도 자가 진단

- agent가 어떤 상태인지 조회할 수 있는가
- tool별 권한과 승인 조건이 분리되어 있는가
- 실패한 run을 재현할 수 있는가
- 비용과 latency가 task별로 남는가
- evaluator가 agent output을 자동으로 검사하는가
- 사람이 승인해야 하는 경계가 명확한가
- rollback 또는 중단 절차가 있는가

## 내 기준

Agent Engineering은 prompt engineering의 확장이 아니다.

```text
Prompt
  -> Tool contract
  -> State machine
  -> Permission
  -> Evaluation
  -> Observability
```

이 구조를 설계하지 않으면 agent는 대화는 잘하지만 운영은 못한다.

## 관련 글

- [AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost]({% post_url 2026-04-04-study-ai-agent-architecture-operations %})
- [Hermes Agent vs OpenClaw]({% post_url 2026-05-23-study-hermes-vs-openclaw %})
