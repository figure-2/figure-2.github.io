---
title: "Production RAG Engineering 3: Evaluation, Operations, Checklist"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- production-rag
- evaluation
- operations
- guardrails
toc: true
date: 2026-04-04 12:33:03 +0900
comments: false
mermaid: true
math: true
---
# Production RAG Engineering 3: Evaluation, Operations, Checklist

> **한줄 정의**
> 운영 가능한 RAG는 답변 생성뿐 아니라 citation, evaluation, latency, cost, guardrails, fallback을 함께 설계해야 한다.

## Generation & Prompting

검색된 context를 LLM에 넣는 것만으로는 부족하다. 어떻게 답하게 할지 계약을 정해야 한다.

| 전략 | 핵심 | 적합한 경우 |
| --- | --- | --- |
| Citation Prompting | 각 주장에 `[1]`, `[2]` 형식 근거 표시 | 사내 QA, 고객 응대, 검증 가능한 답변 |
| Lost-in-the-Middle 대응 | 관련도 높은 문서를 앞과 끝에 배치 | context가 긴 경우 |
| Context Compression | 관련 문장만 추출해 token과 noise 감소 | 검색 결과가 길거나 비용이 큰 경우 |
| Chain-of-Note | 각 문서가 질문에 관련 있는지 메모 후 종합 | noisy retrieval 상황 |

답변 정책에는 `모르겠습니다`가 있어야 한다. 검색 결과에 답이 없는데도 생성하면 RAG의 신뢰성이 깨진다.

## Agentic RAG Patterns

| 패턴 | 판단 구조 | 보존 내용 |
| --- | --- | --- |
| Self-RAG | 검색 필요성, 문서 관련성, 근거성, 유용성을 LLM이 판단 | `[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]` |
| CRAG | 검색 결과를 Correct, Incorrect, Ambiguous로 분류 | Correct는 refinement, Incorrect는 web search, Ambiguous는 결합 |
| Adaptive RAG | 질문 복잡도에 따라 no retrieval, single-step, multi-step 선택 | simple, moderate, complex 분기 |
| RAFT | RAG와 fine-tuning 결합 | 관련 문서와 distractor를 함께 학습 |
| SimRAG | synthetic QA와 self-training | 원본 기준 도메인 적응 1.2~8.6% 향상 |
| Context Engineering | RAG, memory, MCP를 통합 | Knowledge Runtime 방향 |
| MCP + Agentic RAG | tool/data 연결을 표준화 | vector DB, SQL, web, API를 공통 interface로 연결 |

Agentic RAG는 검색의 자율성을 높이지만 비용과 실패 면적도 키운다. 그래서 iteration 상한, tool 권한, 평가 로그가 필요하다.

## 핵심 평가 지표

| 영역 | Metric | 의미 |
| --- | --- | --- |
| Retrieval | Context Precision | 검색된 문서 중 관련 문서 비율 |
| Retrieval | Context Recall | 필요한 정보가 검색 결과에 포함됐는가 |
| Retrieval | MRR | 첫 관련 문서의 평균 순위 |
| Generation | Faithfulness | 답변이 문서에 근거하는가 |
| Generation | Answer Relevancy | 답변이 질문에 맞는가 |
| Generation | Answer Correctness | 정답과 비교한 사실 정확도 |

검색 metric과 생성 metric을 섞으면 안 된다. 검색이 틀렸는지, 생성이 틀렸는지 분리해야 개선 방향이 나온다.

## 평가 도구 스택

| 도구 | 위치 | 역할 |
| --- | --- | --- |
| RAGAS | 개발 | ground truth 없이도 LLM 기반 자동 평가 가능 |
| DeepEval | CI/CD | pytest 스타일 RAG regression test |
| LLM-as-Judge | 비교 평가 | pairwise, pointwise, reference-based |
| TruLens / Langfuse | 운영 | trace, feedback, 품질 추적 |

운영에서는 개별 답변만 보지 않는다. query type, retrieval source, latency, cost, faithfulness score를 함께 봐야 한다.

## 레이턴시 예산

원본 기준 latency budget은 다음과 같이 잡는다.

| 단계 | P50 목표 | P99 목표 | 최적화 방법 |
| --- | --- | --- | --- |
| Query Processing | 50ms | 150ms | 경량 모델, caching |
| Embedding | 20ms | 50ms | batch 처리, GPU |
| Vector Search | 10ms | 30ms | HNSW, in-memory index |
| Reranker | 60ms | 120ms | FlashRank 또는 caching |
| LLM Generation | 500ms | 2000ms | streaming, prompt 최적화 |
| Total E2E | 약 700ms | 약 2.5s | 단계별 병목 추적 |

P50만 보면 운영 병목을 놓친다. Reranker와 generation은 P99를 같이 봐야 한다.

## 비용 최적화

| 전략 | 방식 | 원본 기준 효과 |
| --- | --- | --- |
| Semantic Cache | 유사 질문 embedding 비교 후 cache hit 시 검색/LLM skip | 20~40% 비용 절감 |
| Matryoshka Embedding | 3072d에서 256d로 차원 축소 | storage 12배 절약, 검색 3배 향상, 정확도 약 5% 감소 |
| Router | 단순 질문은 검색 없이 직접 답변 | 검색 비용 30~50% 절감 |
| Prompt Compression | context를 압축해 token 감소 | token 50~70% 절약 |

비용 절감은 품질 저하와 같이 평가해야 한다. cache hit 기준이 느슨하면 오래된 답변이나 권한이 다른 답변을 재사용할 수 있다.

## Production Checklist

### 검색 품질

- 자체 데이터셋 benchmark를 만든다. 최소 100개 이상의 QA 쌍이 필요하다.
- Hybrid Search, Dense + BM25를 적용한다.
- Reranker를 도입하고 A/B test한다.
- Chunking 전략 비교 실험을 완료한다.

### 안전성

- hallucination detection pipeline을 둔다.
- PII filtering과 masking을 적용한다.
- citation을 강제한다.
- fallback 응답을 허용한다.

### 모니터링

- latency dashboard를 P50, P95, P99로 본다.
- faithfulness 자동 평가를 붙인다.
- 사용자 feedback을 수집한다.
- per-query cost를 추적한다.

### 운영

- 문서 업데이트 자동화 pipeline을 둔다.
- index incremental update 전략을 둔다.
- embedding model 교체 migration 계획을 둔다.
- 장애 시 graceful degradation 경로를 둔다.

## 운영 로그 필드

RAG run을 재현하려면 다음 필드가 필요하다.

```text
run_id
query
query_type
retrieval_strategy
retrieved_doc_ids
reranked_doc_ids
prompt_version
model
latency_ms
cost
faithfulness_score
error_type
```

raw prompt와 raw document를 무조건 저장하면 안 된다. 민감 정보는 masking하고, 재현에 필요한 구조화 정보만 남겨야 한다.

## 내 기준

Production RAG의 운영 기준은 다음 문장으로 정리된다.

```text
답변을 만들 수 있는가보다
틀렸을 때 왜 틀렸는지 찾을 수 있는가가 중요하다.
```

운영 로그, 평가셋, citation, fallback이 없으면 RAG는 개선 가능한 시스템이 아니라 운에 맡기는 답변 생성기가 된다.

## 관련 글

- [Production RAG Engineering 1: 아키텍처와 설계 지점]({% post_url 2026-04-04-study-production-rag-architecture %})
- [Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking]({% post_url 2026-04-04-study-production-rag-retrieval %})
