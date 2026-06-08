---
title: "개인 학습 정리 운영 기준"
categories:
- 3.STUDY
tags:
- study
- personal-study
- reference-note
toc: true
date: 2026-06-08 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 개인 학습 정리 운영 기준

## 목적

`3.STUDY`는 교육 과정이나 프로젝트에 직접 묶이지 않는 개인 학습 기록을 정리하는 공간이다.

`1.TIL`은 수업/과정 기반 기록, `2.PROJECT`는 프로젝트 결과와 회고, `3.STUDY`는 외부 가이드와 개인 탐구를 기준으로 나눈다.

## 카테고리 구조

```text
3.STUDY
  3-1.PYTHON
  3-2.RAG
  3-3.AI_AGENT
  3-4.AGENTIC_WORKFLOW
  3-5.KNOWLEDGE_GRAPH
  3-7.AI_ENGINEERING
```

## 1차 작성 범위

1차 작성은 개인 학습 정리에 맞는 주제만 다룬다. 프로젝트 소개, 단순 프롬프트 예시, 채용/태도/일반 커리어 글은 이번 범위에서 제외한다.

| 중분류 | 게시글 단위 | 상태 |
| --- | --- | --- |
| `3-2.RAG` | [RAG 완전 가이드 1: 필요성과 기본 구조]({% post_url 2026-04-04-study-rag-why-and-pipeline %}) | 작성 |
| `3-2.RAG` | [RAG 완전 가이드 2: Naive, Advanced, Modular, Agentic RAG]({% post_url 2026-04-04-study-rag-evolution-patterns %}) | 작성 |
| `3-2.RAG` | [RAG 완전 가이드 3: 평가, 도입 로드맵, 논문 타임라인]({% post_url 2026-04-04-study-rag-evaluation-roadmap-papers %}) | 작성 |
| `3-2.RAG` | [Production RAG Engineering 1: 아키텍처와 설계 지점]({% post_url 2026-04-04-study-production-rag-architecture %}) | 작성 |
| `3-2.RAG` | [Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking]({% post_url 2026-04-04-study-production-rag-retrieval %}) | 작성 |
| `3-2.RAG` | [Production RAG Engineering 3: Evaluation, Operations, Checklist]({% post_url 2026-04-04-study-production-rag-evaluation-operations %}) | 작성 |
| `3-3.AI_AGENT` | [AI Agent 완벽 가이드 1: 정의와 Workflow 구분]({% post_url 2026-04-04-study-ai-agent-definition-workflow %}) | 작성 |
| `3-3.AI_AGENT` | [AI Agent 완벽 가이드 2: Agent 성숙도 7단계]({% post_url 2026-04-04-study-ai-agent-maturity-levels %}) | 작성 |
| `3-3.AI_AGENT` | [AI Agent 완벽 가이드 3: Memory, RAG, Guardrails, Cost]({% post_url 2026-04-04-study-ai-agent-architecture-operations %}) | 작성 |
| `3-3.AI_AGENT` | [Agent Engineering]({% post_url 2026-05-23-study-agent-engineering %}) | 작성 |
| `3-3.AI_AGENT` | [Hermes Agent vs OpenClaw]({% post_url 2026-05-23-study-hermes-vs-openclaw %}) | 작성 |
| `3-3.AI_AGENT` | [AI Assistant Engineering]({% post_url 2026-04-26-study-ai-assistant-engineering %}) | 작성 |
| `3-4.AGENTIC_WORKFLOW` | [Agentic AI 패턴 가이드 1: Workflow vs Agent]({% post_url 2026-04-17-study-agentic-workflow-vs-agent %}) | 작성 |
| `3-4.AGENTIC_WORKFLOW` | [Agentic AI 패턴 가이드 2: 8가지 패턴]({% post_url 2026-04-17-study-agentic-patterns-core %}) | 작성 |
| `3-4.AGENTIC_WORKFLOW` | [Agentic AI 패턴 가이드 3: 선택 기준, 비용, 토폴로지]({% post_url 2026-04-17-study-agentic-pattern-selection-topology %}) | 작성 |
| `3-4.AGENTIC_WORKFLOW` | [바이브코딩 & Claude Code 교육 자료]({% post_url 2026-03-28-study-vibe-coding-claude-code %}) | 작성 |
| `3-4.AGENTIC_WORKFLOW` | [AI 코딩의 Cognitive Debt]({% post_url 2026-05-05-study-ai-coding-cognitive-debt %}) | 작성 |
| `3-5.KNOWLEDGE_GRAPH` | [온톨로지 & 지식 그래프 가이드 1: 개념, 스펙트럼, 트리플]({% post_url 2026-05-16-study-kg-ontology-triple %}) | 작성 |
| `3-5.KNOWLEDGE_GRAPH` | [온톨로지 & 지식 그래프 가이드 2: GraphRAG, 도구, 시작법]({% post_url 2026-05-16-study-kg-graphrag-tools %}) | 작성 |
| `3-7.AI_ENGINEERING` | [AI Native 팀 운영 가이드]({% post_url 2026-04-10-study-ai-native-team %}) | 작성 |
| `3-7.AI_ENGINEERING` | [AI-Ready Data 가이드]({% post_url 2026-05-16-study-ai-ready-data %}) | 작성 |
| `3-7.AI_ENGINEERING` | [AX 시대를 위한 DX]({% post_url 2026-05-16-study-dx-for-ax %}) | 작성 |
| `3-7.AI_ENGINEERING` | [모델 변경과 프롬프트 변화]({% post_url 2026-05-31-study-model-prompt-change %}) | 작성 |
| `3-7.AI_ENGINEERING` | [Tiny LLM from Scratch]({% post_url 2026-04-26-study-tiny-llm-from-scratch %}) | 작성 |

## 분할 기준

긴 가이드는 하나의 큰 글로 몰아넣지 않고, 대주제, 중주제, 소주제로 나눠 작성한다.

| 수준 | 의미 | 예 |
| --- | --- | --- |
| 대주제 | `3.STUDY` 안의 중분류 | `3-2.RAG`, `3-3.AI_AGENT` |
| 중주제 | 하나의 게시글 단위 | `Production RAG Engineering`, `Agent Engineering` |
| 소주제 | 글 안의 핵심 섹션 또는 후속 분리 글 | `Reranking`, `Tool Permission Matrix`, `Cognitive Debt` |

## 이번 정리 커버리지

| 담당 관점 | 확인 기준 | 현재 상태 |
| --- | --- | --- |
| 구조 담당 | 대주제와 중주제가 블로그 카테고리로 분리됐는가 | 1차 범위를 `RAG`, `AI_AGENT`, `AGENTIC_WORKFLOW`, `KNOWLEDGE_GRAPH`, `AI_ENGINEERING`으로 확정 |
| 원문 담당 | 핵심 개념이 원자 노트로 재작성됐는가 | 1차 범위 글 작성 완료 |
| 시각자료 담당 | 표, 파이프라인, 선택 트리, 운영 흐름이 남아 있는가 | 표와 핵심 다이어그램 반영. 원본 inline SVG는 새 표/다이어그램으로 재구성 |
| 블로그 담당 | 공개 글에 맞게 출처 나열보다 개인 정리 관점으로 바뀌었는가 | 외부 링크 나열 없이 내 기준과 체크리스트 중심으로 작성 |
| 검증 담당 | 내부 링크와 front matter가 깨지지 않았는가 | 정적 검증 완료. Jekyll 렌더링은 로컬 런타임 부재로 별도 확인 필요 |

## 문서 유형

| 유형 | 용도 |
| --- | --- |
| `TECH_NOTE` | 개념을 내 언어로 다시 정리 |
| `GUIDE_REVIEW` | 외부 가이드를 읽고 구조화 |
| `PRACTICE` | 직접 실습한 결과 정리 |
| `PROJECT_NOTE` | 프로젝트에 적용한 내용 정리 |
| `REFERENCE_NOTE` | 로드맵, 자료 목록, 체크리스트 |

## 작성 원칙

- 원문 요약보다 내가 이해한 구조와 판단 기준을 우선한다.
- 기존 `1.TIL` 글과 겹치면 반복하지 않고 관련 글로 연결한다.
- 최신 도구, SDK, 논문, 벤치마크는 발행 전에 원문을 확인한다.
- 코드, 로그, 설정 예시는 공개 가능한 형태로만 남긴다.
- API key, DB URL, 계정 ID, raw payload, 개인 로그는 게시하지 않는다.

## 발행 흐름

```text
외부 자료 읽기
  -> 대주제/중주제/소주제 분리
  -> 원자 노트 작성
  -> 기존 블로그 글과 연결
  -> 공개 가능 정보 검수
  -> _posts에 게시
```
