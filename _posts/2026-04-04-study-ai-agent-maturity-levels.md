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
Levels 0-6

## Agent 성숙도 7단계

단순 LLM 호출부터 멀티 에이전트 시스템까지, 각 레벨의 구조와 특징을 살펴봅니다

🧩

함께 보기

어떤 패턴을 내 상황에 써야 할지

선택이 고민이라면 — 패턴별 실전 예시·비용/레이턴시 비교·조합 레시피가 담긴

Agentic AI 패턴 가이드

를 보세요.

패턴 가이드 →

Simple

Autonomous

Multi-Agent

L0

### Simple LLM Call

No Tools

No Memory

Single Turn

자율성

가장 기본적인 형태입니다. 프롬프트를 넣으면 응답이 나오는 단순 호출로, LLM의 학습된 지식만으로 답변합니다. 외부 도구 접근이 없어 할루시네이션 리스크가 가장 높습니다.

U

User

Prompt

L

LLM

Response

R

Output

#### 사용 예시

- "이메일 초안 써줘"

- "이 코드 리뷰해줘"

- "마케팅 카피 만들어줘"

#### 한계

- 최신 정보 접근 불가

- 사내 데이터 참조 불가

- 정보를 지어낼 수 있음

#### 관련 기술

- Chain-of-Thought (Wei et al., 2022)

- Zero-shot / Few-shot Prompting

- 기본 ChatGPT / Claude 호출

L1

### Augmented LLM (Tool Use)

Function Calling

RAG

Single Cycle

자율성

LLM이 필요할 때 외부 도구를 한 번 호출할 수 있습니다. 도구를 쓸지 말지를 LLM이 판단하고, 결과를 받아 최종 응답을 생성합니다. 하지만 한 번의 사이클로 끝나며, 결과가 부족해도 재시도하지 않습니다.

U

User

Query

L

LLM

Tool Call

API / DB / Search

Response

R

Output

#### 도구 유형

- 검색: RAG, 웹 검색

- API: 날씨, 주가, DB 쿼리

- 실행: 코드 인터프리터, 계산기

#### 사용 예시

- "오늘 서울 날씨 알려줘" → weather API

- "Q3 매출 데이터 찾아줘" → DB 쿼리

- Naive RAG: 벡터 검색 → 답변 생성

#### 관련 기술

- Toolformer (Schick et al., 2023)

- MRKL Systems (Karpas et al., 2022)

- Function Calling / Structured Output

L2

### Chained / Sequential Agent

Pipeline

Deterministic

Multi-Step

자율성

여러 단계를 미리 정의된 순서대로 실행합니다. 앞 단계의 출력이 뒷 단계의 입력이 됩니다. Gate(검증 단계)를 넣어 품질을 체크할 수 있지만, 전체 흐름은 코드로 고정되어 있습니다.

1

분석

2

처리

G

검증

3

출력

##### 문서 번역 파이프라인

원문 분석

→

초벌 번역

→

용어 검증

→

최종 다듬기

##### 코드 생성 파이프라인

요구사항

→

코드 생성

→

린트/테스트

→

리뷰/수정

#### 특징

- 실행 순서가 코드로 고정 (deterministic)

- 각 단계에서 LLM이 동작하지만 전체 흐름은 LLM이 결정하지 않음

- Latency = 각 단계의 합산

#### 한계

- 분기(branching)가 없음

- 간단한 요청도 전체 파이프라인 통과 필요

- 한 단계 실패 시 전체 중단

L3

### Router / Branching Agent

Dynamic Routing

Classifier

Parallel

자율성

입력을 분석해서 적절한 경로로 분기합니다. LLM이 분류기(classifier) 역할을 수행하며, 각 브랜치는 독립적인 워크플로우입니다. 리소스 효율적이고 병렬 실행도 가능합니다.

U

Input

R

Router

Simple

직접 답변

Data

검색 파이프라인

Code

코드 생성

##### 고객 지원 시스템

고객 메시지

"환불 요청" → 환불 처리 워크플로우

"기술 문의" → 기술 지원 워크플로우

"일반 문의" → FAQ 기반 응답

L4

### ReAct / Loop Agent

True Agent

Self-Correction

Autonomous Loop

자율성

여기서부터 진정한 Agent입니다

LLM이 루프를 돌면서 스스로 판단하고, 도구를 선택하고, 결과를 평가하고, 필요하면 다시 시도합니다. ReAct(Reasoning + Acting) 패턴이 핵심이며, 종료 조건도 LLM이 판단합니다.

반복

T

Thought

상황 분석 & 판단

A

Action

도구 선택 & 실행

O

Observe

결과 확인 & 평가

!

Answer

#### ReAct 실행 예시: 리서치 에이전트

Thought

사용자가 2024년 한국 GDP를 묻고 있다. 최신 데이터니까 검색이 필요하다.

Action

web_search("2024 한국 GDP")

Observe

검색 결과가 2023년 데이터뿐이다. 더 최신 쿼리가 필요.

Thought

쿼리를 변경해서 재검색하자.

Action

web_search("South Korea GDP 2024 IMF estimate")

Observe

IMF 기준 1.7조 달러라는 결과 확인. 신뢰할 만한 출처.

Answer

2024년 한국 GDP는 IMF 추정 기준 약 1.7조 달러입니다...

#### 핵심 특징

- LLM이 실행 흐름을 제어

- 자기 수정(self-correction) 가능

- 종료 조건을 LLM이 판단

- max_iterations 설정 필수

#### 관련 논문

- ReAct (Yao et al., 2022) - ICLR 2023

- Reflexion (Shinn et al., 2023)

- Tree of Thoughts (Yao et al., 2023)

L5

### Planning Agent

Task Decomposition

Adaptive Replanning

Reflection

자율성

실행 전에 먼저 전체 계획을 세우고, 계획에 따라 단계별로 실행하며, 상황에 따라 계획을 동적으로 수정합니다. 장기 목표(long-horizon task) 처리가 가능합니다.

P

##### Plan

작업 분해

Task 1

Task 2

Task 3

Task 4

E

##### Execute

단계별 ReAct 실행

R

##### Reflect

결과 평가 & 계획 수정

Re-plan

##### "경쟁사 분석 보고서 만들어줘"

✓

1. 경쟁사 리스트 확정

✗

2. 재무 데이터 수집

회사 B 데이터 못 찾음

↻

2'. Re-plan: 대안 소스에서 재검색

✓

2'. 대안 소스에서 수집 완료

●

3. SWOT 분석

●

4. 보고서 작성

L6

### Multi-Agent System

Collaboration

Specialization

Distributed

자율성

여러 에이전트가 각자의 역할, 도구, 프롬프트를 가지고 협업합니다. Orchestrator가 작업을 분배하고 결과를 통합하며, 각 에이전트가 독립된 컨텍스트를 가져 컨텍스트 윈도우 한계를 완화합니다.

O

Orchestrator

R

Researcher

Search, Fetch

C

Coder

IDE, Terminal

Q

Reviewer

Lint, Test

W

Writer

Docs, Format

##### Orchestrator-Worker

중앙 관리자가 작업을 분배하고 결과를 통합. 가장 일반적인 패턴.

Claude Code의 sub-agents

##### Evaluator-Optimizer

Generator가 생성, Evaluator가 평가 후 피드백. 반복적 개선.

코드 리뷰 자동화

##### Debate / Adversarial

Agent A가 주장, Agent B가 반론. Moderator가 최종 판정.

의사결정 지원 시스템

### 단계별 비교 요약

| 레벨 | 자율성 | 도구 사용 | 흐름 결정 | 대표 기술 |
| --- | --- | --- | --- | --- |
| L0 | 없음 | 없음 | 코드 | ChatGPT 기본 호출 |
| L1 | 낮음 | 단일 턴 | 코드 | RAG, Function Calling |
| L2 | 낮음 | 순차 | 코드 | LangChain Chains |
| L3 | 중간 | 분기 | 코드+LLM | Semantic Router |
| L4 | 높음 | 루프 | LLM | ReAct, Claude Tool Use |
| L5 | 높음 | 계획+루프 | LLM | Plan-and-Execute, ADK |
| L6 | 매우 높음 | 분산 | 다수 LLM | CrewAI, AutoGen |

---

## 추가 정리

### 핵심 요약

성숙도 7단계는 에이전트를 한 번에 복잡하게 만들지 않기 위한 기준이다. Simple LLM Call에서 시작해 Tool Use, Routing, ReAct, Planning, Multi-Agent로 올라갈수록 자율성은 커지지만 비용과 실패 가능성도 함께 커진다.

### 보충 해설

각 단계의 핵심 질문은 "이 단계의 복잡도가 실제 문제를 해결하는가"다. 도구 호출이 필요 없는데 Tool Use를 넣거나, 고정 절차로 충분한데 Planning Agent를 쓰면 디버깅 지점만 늘어난다. 성숙도는 목표가 아니라 선택 기준이다.
