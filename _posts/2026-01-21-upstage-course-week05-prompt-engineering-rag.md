---
title: "Week 05 - Prompt Engineering & RAG"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- prompt-engineering
- rag
- course-note
- prompt-engineering-rag
toc: true
date: 2026-01-21 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Week 05 - Prompt Engineering & RAG

> **요약**
> LLM을 더 잘 쓰기 위한 입력 설계에서 시작해, 외부 지식 결합(RAG), RAG 고도화, 프롬프트 기반 보안, 멀티모달/개인화, Agentic AI로 이어지는 흐름을 학습한 주차다.

## 주간 목표

Week 05의 핵심은 “LLM에게 잘 질문하는 법”에서 끝나지 않는다. 프롬프트는 단순한 질문 문장이 아니라, 역할, 지시사항, 예시, 검증, 외부 지식, 후처리까지 포함하는 입력 설계다.

이번 주차의 흐름은 다음처럼 이어진다.

```text
LLM과 Alignment 이해
  -> Basic Prompting
  -> Advanced Prompting
  -> RAG
  -> Advanced RAG / Knowledge Conflict
  -> LLM Security
  -> Multimodal / Personalization / Agentic AI
```

노트북 실습에서는 MMLU 문제를 활용해 prompt를 바꿔가며 성능을 비교하고, RAG 실습에서는 PDF 문서를 로드해 chunking, embedding, vector store, retrieval, generation으로 이어지는 기본 파이프라인을 직접 구성했다.

## 강의 목록

| Day | 제목 | 핵심 주제 | 실습 |
|-----|------|---------|------|
| 1 | [W05D01-프롬프팅-기초]({% post_url 2026-01-21-upstage-course-w05d01-prompting-basics %}) | LLM, Alignment, Basic Prompting, Persona, ICL | [기본 프롬프팅 튜토리얼]({% post_url 2026-01-21-upstage-practice-prompting-practice %}) |
| 2 | [W05D02-고급-프롬프팅]({% post_url 2026-01-22-upstage-course-w05d02-advanced-prompting %}) | Zero-shot CoT, Plan-and-Solve, CoT, Self-Ask, Self-Consistency, ToT | [프롬프트 설계 연습]({% post_url 2026-01-21-upstage-practice-w05-prompt-design %}) |
| 3 | [W05D03-RAG-기초]({% post_url 2026-01-23-upstage-course-w05d03-rag-basics %}) | Parametric/Non-parametric Knowledge, Chunking, Sparse/Dense Retrieval | [Naive RAG 파이프라인]({% post_url 2026-01-23-upstage-practice-rag-practice %}) |
| 4 | [W05D04-Advanced-RAG-보안]({% post_url 2026-01-26-upstage-course-w05d04-advanced-rag-security %}) | HyDE, Reranking, Modular RAG, Knowledge Conflict, Prompt Injection | [RAG Knowledge Conflict]({% post_url 2026-01-26-upstage-practice-rag-conflict-practice %}) |
| 5 | [W05D05-트렌드-Agentic-AI]({% post_url 2026-01-27-upstage-course-w05d05-trend-agentic-ai %}) | Multimodal Prompting, Persistent Memory, Agentic AI | [Visual-Text Embedding Alignment]({% post_url 2026-01-27-upstage-practice-embedding-practice %}) |

## 핵심 개념

- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %}) - 역할, 지시사항, 예시, 검증, 외부 지식 사용을 설계하는 방법
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %}) - 모델 내부 지식의 한계를 외부 지식 검색으로 보완하는 구조
- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %}) - 쿼리 변환, chunking, reranking, graph/memory 기반 고도화
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %}) - prompt injection, jailbreak, guardrail, red team/blue team
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %}) - 프롬프트 기반 LLM 사용에서 도구와 계획을 가진 AI Agent로 확장

## 정리 기준

이 주차의 글을 읽을 때는 “기법 이름을 외우는 것”보다 “언제 어떤 문제가 생기고, 어떤 기법이 그 문제를 줄이는가”를 기준으로 보면 좋다.

- 의도와 형식이 불명확하면 Role, Instruction, Example을 보강한다.
- 복잡한 추론에서 단계 누락이 생기면 CoT, Plan-and-Solve, Self-Ask를 검토한다.
- 답변이 불안정하면 Self-Consistency, Self-Verification, Self-Refine을 검토한다.
- 모델 내부 지식이 낡았거나 부족하면 RAG를 붙인다.
- 검색 결과가 부정확하면 Query Transformation, Chunking, Reranking을 조정한다.
- 외부 문서와 모델 내부 지식이 충돌하면 Knowledge Conflict 전략이 필요하다.
- 외부 입력이 명령처럼 작동하면 Prompt Injection 방어가 필요하다.
