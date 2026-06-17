---
title: "Agentic Workflow"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- agentic-workflow
- tech-note
toc: true
date: 2026-01-28 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Agentic Workflow

> **한줄 정의**
> LLM 에이전트가 자율적으로 작업을 계획하고 실행하는 워크플로우. 단순 체인을 넘어 동적 의사결정과 반복 실행이 가능하다.

## 학습 맥락

Agentic Workflow는 업스테이지 과정에서 Prompt/RAG 다음 단계로 등장했다. 앞선 단계가 "LLM에게 잘 묻고, 필요한 문서를 검색해서 답하게 만드는 방법"이었다면, 이 단계는 LLM이 여러 행동 중 무엇을 할지 선택하고, 도구를 호출하고, 결과를 보고 다음 행동을 결정하게 만드는 방법을 다룬다.

초기 강의 계획서 기준으로는 AI Product Engineering의 `Agentic Workflow` 영역에 해당한다. AI Service & Agent Design, Function Calling, Advanced Tool Use, Memory Systems & RAG, Context Engineering, Safety, Evaluation으로 이어지는 흐름의 중심 개념이다.

## 핵심 개념

Agentic Workflow는 단일 에이전트와 멀티 에이전트 두 가지 형태로 구성된다. 단일 에이전트는 ReAct 루프로 작업을 처리하고, 멀티 에이전트는 Supervisor-Worker 계층으로 복잡한 작업을 분산 처리한다. 각 에이전트는 전문화된 역할을 가지며 메시지 패싱으로 소통한다.

핵심 패턴으로는 **Reflection**(자기 검토 및 개선), **Planning**(단계별 계획 수립), **Tool Use**(외부 도구 활용), **Orchestration**(멀티 에이전트 조율)이 있다. LangGraph는 이러한 패턴들을 그래프 구조로 구현하는 데 최적화되어 있다.

## Workflow와 Agent의 차이

워크플로우는 미리 정해진 순서대로 실행된다. 입력 검증, 검색, 답변 생성, 후처리처럼 단계가 고정되어 있으면 워크플로우가 적합하다.

에이전트는 상황에 따라 다음 행동을 선택한다. 검색이 필요한지, 어떤 도구를 호출할지, 결과가 부족하면 다시 검색할지 같은 결정을 실행 중에 내린다.

실제 서비스에서는 둘 중 하나만 쓰기보다 섞어서 쓴다. 핵심 흐름은 워크플로우로 안정화하고, 판단이 필요한 부분만 에이전트에게 맡기는 방식이 운영하기 쉽다.

## 워크플로우 패턴

![Agentic Workflow 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-agentic-workflow-diagram-1.svg)

| 패턴 | 핵심 질문 | 쓰는 상황 |
| --- | --- | --- |
| ReAct | 생각한 뒤 어떤 행동을 할 것인가 | 도구 호출과 관찰을 반복해야 할 때 |
| Plan-and-Execute | 먼저 계획을 세우고 단계별로 실행할 것인가 | 복잡한 작업을 여러 단계로 나눌 때 |
| Reflection | 결과를 다시 검토하고 고칠 것인가 | 답변 품질을 스스로 점검해야 할 때 |
| Router | 어떤 경로 또는 도구를 선택할 것인가 | 질문 유형별 처리 경로가 다를 때 |
| Supervisor-Worker | 누가 어떤 하위 작업을 맡을 것인가 | 역할 분리가 필요한 멀티 에이전트 구조 |

## 구현 관점

Agentic Workflow를 구현할 때는 처음부터 모든 것을 자율 에이전트로 만들지 않는 편이 좋다. 자율성이 커질수록 디버깅, 비용, 보안, 평가가 어려워진다.

기본 순서는 다음처럼 잡을 수 있다.

```text
사용자 요청
  -> 요청 분류
  -> 필요한 컨텍스트 확인
  -> 도구 호출 또는 검색
  -> 결과 검증
  -> 답변 생성
  -> 필요 시 재시도 또는 사람에게 넘김
```

여기서 에이전트가 맡는 부분은 "분류", "도구 선택", "재시도 여부 판단"이다. 반대로 인증, 권한, 결제, 데이터 삭제처럼 실패 비용이 큰 작업은 명시적인 규칙과 승인 절차를 둬야 한다.

## 프로젝트 연결

LangGraph 기반 프로젝트에서는 Agentic Workflow를 노드와 엣지로 표현한다. 노드는 작업 단위이고, 엣지는 다음 단계로 이동하는 조건이다. 예를 들어 검색 노드, 답변 생성 노드, 평가 노드, 재검색 노드를 나누면 에이전트의 판단 과정을 추적하기 쉬워진다.

idol-agent 같은 프로젝트에서는 사용자 요청을 받아 모델 응답만 생성하는 구조에서, 상태관리와 도구 호출을 포함한 구조로 확장할 수 있다. 이때 Agentic Workflow는 "어떤 순서로 실행할지"보다 "어떤 상태에서 어떤 다음 행동을 선택할지"를 설계하는 기준이 된다.

## 주의점

- 모든 단계를 에이전트 판단에 맡기면 결과가 불안정해진다.
- 도구 호출 결과를 검증하지 않으면 잘못된 외부 결과를 그대로 답변에 반영할 수 있다.
- 실패 시 재시도 기준이 없으면 비용이 빠르게 증가한다.
- 사용자 입력, 검색 문서, 도구 결과를 같은 신뢰도로 다루면 prompt injection 위험이 커진다.
- 운영 단계에서는 실행 경로, 토큰, 비용, 실패 이유를 추적해야 한다.

## 관련 글

- [AI 서비스와 에이전트 설계]({% post_url 2026-01-28-upstage-course-w06d01-ai-service-design %})
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %})
- [Agentic RAG]({% post_url 2026-01-30-upstage-tech-agentic-rag %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [Context Engineering]({% post_url 2026-02-02-upstage-tech-context-engineering %})
- [AgentOps]({% post_url 2026-02-03-upstage-tech-agentops %})

## 참고 자료

- [Agentic AI Course (DeepLearning.AI)](https://www.deeplearning.ai/courses/agentic-ai)
