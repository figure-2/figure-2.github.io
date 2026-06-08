---
title: "AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- memory
- guardrails
- mcp
- a2a
toc: true
date: 2026-04-04 01:20:00 +0900
comments: false
mermaid: true
math: true
---
# AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost

> **한줄 정의**
> Agent 운영의 핵심은 memory, retrieval, guardrails, protocol, evaluation, cost를 하나의 실행 시스템으로 묶는 것이다.

## Memory 시스템

Agent memory는 세 가지로 나눠 본다.

| 유형 | 의미 | 구현 예 |
| --- | --- | --- |
| Short-term Memory | 현재 대화와 즉시 필요한 작업 기억 | context window, working memory |
| Long-term Memory | 세션을 넘어 지속되는 사실, 규칙, 선호 | vector DB, knowledge graph |
| Episodic Memory | 과거 경험과 상황별 episode | semantic retrieval, event log |

Memory는 많이 넣는 것이 목표가 아니다. 어떤 정보를 언제 검색하고 언제 폐기할지 정하는 것이 핵심이다.

## Traditional RAG vs Agentic RAG

| 관점 | Traditional RAG | Agentic RAG |
| --- | --- | --- |
| 흐름 | Query -> Search -> Generate | Plan -> Retrieve -> Evaluate -> Re-retrieve -> Synthesize |
| 검색 횟수 | 보통 1회 | 필요하면 반복 |
| 결과 평가 | 별도 단계가 없을 수 있음 | 검색 결과를 평가하고 재검색 |
| 비유 | 책 한 권 빌리기 | 연구 조교가 여러 자료 교차 검증 |

Agentic RAG는 복잡 질문에 강하지만, 비용과 지연시간을 늘린다.

## Guardrails 아키텍처

가드레일은 한 계층으로 충분하지 않다.

```text
Input Guardrails
  -> Agent Core
  -> Tool Guardrails
  -> Output Guardrails
  -> Human-in-the-Loop
```

| 계층 | 방어 대상 |
| --- | --- |
| Input Guardrails | prompt injection, PII, 유해 입력 |
| Tool Guardrails | 권한 확인, 실행 전 승인, 파괴적 작업 차단 |
| Output Guardrails | hallucination, PII 노출, 정책 위반 |
| Human-in-the-Loop | 결제, 발송, 삭제, 권한 변경 같은 irreversible action |

Agent는 도구를 실행할 수 있으므로, chatbot보다 권한 경계가 더 중요하다.

## 비용 최적화

원본 학습 노트 기준으로 agent loop는 단일 호출 대비 10~100배 더 많은 token을 쓸 수 있다.

| 전략 | 원본 기준 효과 | 설명 |
| --- | --- | --- |
| Prompt Caching | 60~80% 절감 | system prompt와 tool schema 재사용 |
| Multi-Model Routing | 30~60% 절감 | 단순 작업은 저렴한 모델, 복잡 추론은 고급 모델 |
| Batch Processing | 약 50% 절감 | 비동기 batch로 할인 활용 |
| Prompt Engineering | 15~40% 절감 | 간결한 prompt, JSON 출력, 불필요 tool 제거 |

비용은 token만이 아니다. 도구 호출, 검색, reranking, code execution, human review 비용도 포함된다.

## MCP와 A2A

| 프로토콜 | 연결 방향 | 역할 |
| --- | --- | --- |
| MCP | Agent -> Tools & Data | tools, resources, prompts를 표준 방식으로 제공 |
| A2A | Agent -> Agent | agent card, task, message, artifact 교환 |

MCP는 vertical integration이다. Agent가 외부 도구와 데이터에 접근하는 방법을 표준화한다.

A2A는 horizontal integration이다. Agent가 다른 agent에게 작업을 위임하고 결과를 받는 방법을 표준화한다.

## Framework 비교

| Framework | 중심 구조 | 적합한 경우 |
| --- | --- | --- |
| LangGraph | graph-based workflow, durable execution | 복잡한 production workflow |
| CrewAI | role-based multi-agent | 역할 전문화와 팀 시뮬레이션 |
| AutoGen | event-driven, actor model | enterprise 분산 agent |
| Google ADK | code-first, runner 중심 | Google Cloud와 streaming |
| OpenAI Agents SDK | minimal Python primitives | 빠른 prototype, 단순 agent |
| Claude Code | terminal agent, sub-agent, permission gate | codebase 작업과 multi-file 변경 |

프레임워크 선택은 기능 수가 아니라 상태 관리, human gate, trace, evaluation 지원으로 판단한다.

## 흔한 실수 Top 10

| 번호 | 실수 | 문제 |
| --- | --- | --- |
| 1 | 과도한 engineering | 단순 LLM이나 workflow로 충분한 문제에 multi-agent 도입 |
| 2 | 데이터 품질 무시 | 부실한 pipeline 위에 agent를 구축 |
| 3 | 평가 framework 부재 | 원본 기준 AI team의 15%만 포괄 평가 수행 |
| 4 | 관찰 도구 누락 | 원본 기준 production agent의 5%만 성숙한 monitoring 보유 |
| 5 | RPA처럼 취급 | 구축, 배포, 방치 방식은 실패 |
| 6 | 도구 과다 등록 | 모든 tool schema가 token을 소비 |
| 7 | Human-in-the-Loop 부재 | 중요한 결정을 완전 자동화 |
| 8 | 비용 관리 실패 | agent loop는 단일 호출보다 10~100배 token 소비 가능 |
| 9 | 부실한 tool 문서화 | tool 설명이 약하면 tool 선택이 흔들림 |
| 10 | 종료 조건 미설정 | exit criteria 없는 agent는 무한 loop 위험 |

## 주요 통계

| 수치 | 의미 |
| --- | --- |
| 88% | 원본 기준 agent project가 production 전 실패 |
| 1,445% | 원본 기준 multi-agent 문의 증가율 |
| 85% | 원본 기준 개발자의 AI coding tool 사용 |
| $2.1M | 원본 기준 AI 보안 통제 적용 시 평균 비용 절감 |
| 80.9% | 원본 기준 SWE-bench Verified 최고 점수 |
| 33% | 원본 기준 2028년까지 agent AI 포함 예측 |

이 수치는 원본 학습 노트 기준이며, 최신 시장 수치로 단정하지 않는다.

## 핵심 논문과 벤치마크

| 항목 | 핵심 |
| --- | --- |
| ReAct | Thought, Action, Observation loop |
| Chain-of-Thought | 단계별 추론 |
| Toolformer | LLM의 tool use 학습 |
| Generative Agents | observation, reflection, retrieval 기반 social agent |
| Reflexion | 언어적 자기 반성 |
| Tree of Thoughts | 사고 트리 탐색 |
| MRKL | router가 symbolic tool과 LLM을 연결 |
| HuggingGPT | LLM이 전문 모델을 orchestrate |
| SWE-bench Verified | 실제 GitHub issue 해결 |
| GAIA | multimodal tool use와 reasoning |
| AgentBench | OS 환경 등 multi-environment agent 평가 |

## 내 기준

Agent는 다음 순서로 설계한다.

```text
Task boundary
  -> Tool boundary
  -> Memory policy
  -> Evaluation
  -> Observability
  -> Cost limit
  -> Human approval
```

이 순서를 빼고 framework부터 고르면, agent는 금방 demo는 되지만 production system이 되지 않는다.

## 관련 글

- [AI Agent 완벽 가이드 1: 정의와 Workflow 구분]({% post_url 2026-04-04-study-ai-agent-definition-workflow %})
- [Agent Engineering]({% post_url 2026-05-23-study-agent-engineering %})
