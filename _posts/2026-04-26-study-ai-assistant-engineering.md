---
title: "AI Assistant Engineering"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-assistant
- llm
- rag
- agent
- fine-tuning
toc: true
date: 2026-04-26 00:00:00 +0900
comments: false
mermaid: true
math: true
---
# AI Assistant Engineering

> **한줄 정의**
> AI Assistant Engineering은 LLM 기초, RAG, Agent, fine-tuning을 따로 배우는 것이 아니라 하나의 assistant를 만들기 위한 단계로 묶어 학습하는 것이다.

## 정리 범위

이 항목은 상세 구현보다 AI Assistant를 만들기 위한 학습 경로를 정리하는 성격이다. 핵심 범위는 다음과 같다.

```text
LLM
  -> RAG
  -> Agent
  -> Fine-tuning
  -> Assistant
```

따라서 이 글은 특정 구현 코드보다 학습 경로와 산출물 기준을 정리한다.

## 학습 축

| 축 | 학습 질문 | 산출물 |
| --- | --- | --- |
| LLM 기초 | token, context, prompt, structured output을 이해하는가 | 작은 prompt 실험과 출력 계약 |
| RAG | 외부 문서를 검색해 근거 기반 답변을 만들 수 있는가 | retrieval pipeline, 평가셋 |
| Agent | 도구 호출과 재시도 loop를 설계할 수 있는가 | tool-calling assistant |
| Fine-tuning | 모델 행동을 데이터로 조정할 수 있는가 | instruction dataset, eval 결과 |
| 운영 | assistant 품질을 추적할 수 있는가 | trace, cost, latency, feedback |

## 1단계. LLM 기초

Assistant를 만들기 전에 model call의 기본 계약을 알아야 한다.

| 항목 | 확인할 것 |
| --- | --- |
| Token | 입력과 출력 비용이 어떻게 계산되는가 |
| Context Window | 무엇을 넣고 무엇을 빼야 하는가 |
| Prompt | 목표, 제약, 출력 형식을 분리했는가 |
| Structured Output | JSON schema나 field contract가 있는가 |
| Evaluation | 같은 입력에 대한 기대 출력이 정의됐는가 |

## 2단계. RAG

RAG는 assistant가 외부 지식을 쓰게 하는 방법이다.

```text
Documents
  -> Parse
  -> Chunk
  -> Embed
  -> Retrieve
  -> Generate with citation
```

학습 산출물은 단순 demo가 아니라 검색 실패를 볼 수 있는 평가셋이어야 한다.

| 체크 | 질문 |
| --- | --- |
| 문서 품질 | 원본에 답이 있는가 |
| 검색 품질 | 필요한 chunk가 Top-K에 들어오는가 |
| 생성 품질 | 답변이 context에 충실한가 |
| 권한 | 사용자별로 검색 가능한 문서가 다른가 |

## 3단계. Agent

Agent는 assistant가 도구를 쓰고 중간 결과에 따라 다음 행동을 바꾸는 단계다.

| 구성 | 예 |
| --- | --- |
| Tools | search, DB, calendar, code execution |
| Planning | task decomposition |
| Memory | user preference, session state |
| Guardrails | tool permission, human approval |
| Trace | step별 action과 result |

Agent는 자율성을 늘리는 작업이 아니라, 자율성이 필요한 경계와 금지할 경계를 함께 정하는 작업이다.

## 4단계. Fine-tuning

Fine-tuning은 RAG와 대체 관계가 아니다.

| 문제 | 우선 해법 |
| --- | --- |
| 최신 지식 부족 | RAG |
| 특정 말투와 형식 | prompt 또는 fine-tuning |
| 반복되는 domain behavior | fine-tuning |
| private knowledge | RAG 또는 tool |
| 긴 workflow | agent |

Fine-tuning은 "지식을 넣는 것"보다 "행동 양식과 출력 형식 안정화"에 더 적합하다.

## 5단계. Assistant 통합

최종 assistant는 다음 구성으로 본다.

```text
User
  -> Intent classification
  -> Retrieval or Tool
  -> Generation
  -> Evaluation
  -> Feedback
```

| 계층 | 운영 기준 |
| --- | --- |
| Intent | 질문 유형을 분류할 수 있는가 |
| Retrieval | 근거가 있는가 |
| Tool | 실행 권한과 실패 처리가 있는가 |
| Generation | 출력 계약을 지키는가 |
| Evaluation | 품질과 비용을 측정하는가 |
| Feedback | 사용자 반응이 다음 개선으로 이어지는가 |

## 학습 순서

| 순서 | 목표 | 완료 기준 |
| --- | --- | --- |
| 1 | LLM 호출과 structured output | 같은 입력에 안정된 format으로 답변 |
| 2 | RAG 기본 pipeline | 문서 기반 QA와 citation |
| 3 | RAG 평가 | context precision, faithfulness 측정 |
| 4 | Tool calling | 외부 API 호출과 error handling |
| 5 | Agent loop | observe 후 retry 또는 stop |
| 6 | Fine-tuning 실험 | 특정 출력 양식의 안정화 비교 |
| 7 | Assistant 운영 | trace, cost, latency, feedback dashboard |

## 내 기준

AI Assistant는 기술 묶음이 아니라 사용자 과업을 끝내는 interface다.

```text
LLM은 말하게 하고
RAG는 근거를 주고
Tool은 행동하게 하고
Agent는 판단하게 하고
Evaluation은 믿을 수 있게 만든다.
```

이 다섯 가지가 이어질 때 assistant engineering이 된다.

## 관련 글

- [RAG 완전 가이드 1: 필요성과 기본 구조]({% post_url 2026-04-04-study-rag-why-and-pipeline %})
- [AI Agent 완벽 가이드 1: 정의와 Workflow 구분]({% post_url 2026-04-04-study-ai-agent-definition-workflow %})
