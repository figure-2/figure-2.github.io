---
title: "Production RAG Engineering 1: 아키텍처와 설계 지점"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- production-rag
- rag-engineering
- architecture
toc: true
date: 2026-04-04 12:31:03 +0900
comments: false
mermaid: true
math: true
---
# Production RAG Engineering 1: 아키텍처와 설계 지점

> **한줄 정의**
> Production RAG는 offline ingestion과 online query pipeline을 분리하고, retrieval, reranking, generation, evaluation, safety를 한 실행 흐름으로 설계하는 시스템이다.

## 기준

이 글의 모델명, 수치, 비용, latency는 원본 학습 노트 기준이다. 최신 가격이나 공식 benchmark로 단정하지 않는다.

## 전체 구조

![Production RAG engineering architecture](/assets/images/study/diagrams/study-rag-engineering-architecture.svg){: width="100%"}

Production RAG는 두 개의 파이프라인과 두 개의 횡단 계층으로 본다.

```text
Offline Ingestion
  -> Data Sources
  -> Parser
  -> Chunker
  -> Context Enrichment
  -> Embed Model
  -> Index Store

Online Query
  -> User Query
  -> Semantic Cache
  -> Query Processing
  -> Router
  -> Hybrid Search
  -> Reranker
  -> LLM Generation + Citation
  -> Response

Cross-cutting
  -> Evaluation & Monitoring
  -> Guardrails
```

## Offline Ingestion

| 단계 | 역할 | 설계 질문 |
| --- | --- | --- |
| Data Sources | PDF, DB, API, Web 등 원본 수집 | 어떤 원본이 실제 질문에 답을 갖는가 |
| Parser | 문서를 구조화된 text/table로 변환 | 표, 섹션, 제목이 보존되는가 |
| Chunker | 검색 단위 생성 | 의미 단위가 깨지지 않는가 |
| Context Enrichment | contextual prefix, metadata tagging | chunk만 봐도 문서 맥락을 알 수 있는가 |
| Embed Model | dense/sparse representation 생성 | 도메인과 언어에 맞는가 |
| Index Store | dense, sparse, metadata 저장 | 권한과 version filter가 가능한가 |

Ingestion의 품질이 낮으면 online pipeline에서 비용을 써도 복구가 어렵다.

## Online Query

| 단계 | 역할 | 설계 질문 |
| --- | --- | --- |
| Semantic Cache | 유사 질문의 기존 답변 재사용 | cache hit 기준이 안전한가 |
| Query Processing | classification, rewriting, HyDE, decomposition | 복잡 질문을 검색 가능한 단위로 바꿨는가 |
| Router | 검색 경로 분기 | vector, SQL, web, direct LLM 중 무엇을 쓸 것인가 |
| Hybrid Search | dense + sparse 검색 | 의미와 키워드를 함께 잡는가 |
| Reranker | Top-K를 Top-N으로 재정렬 | latency 대비 품질 이득이 있는가 |
| Generation | 근거 기반 답변 생성 | citation과 no-answer 정책이 있는가 |
| Evaluation | 품질 추적 | faithfulness, relevancy, latency, cost를 남기는가 |

## 핵심 설계 결정

| 결정 | 선택지 | 원본 기준 판단 |
| --- | --- | --- |
| Indexing | Dense-only | 구현은 단순하지만 keyword와 고유명사에 약함 |
| Indexing | Hybrid, Dense + BM25 | 대부분의 production 기본 선택. RRF 결합. 검색 실패율 약 35% 감소 |
| Indexing | Hybrid + Late Interaction | 정확도는 높지만 ColBERT/ColPali 계열은 storage 5~10배 증가 |
| Reranker | Yes | 원본 기준 MAP +52%, 100ms 이하 추가 latency, Top-100에서 Top-5로 좁히는 데 적합 |
| Reranker | Skip | P99 200ms 미만의 초저지연 서비스에서만 고려 |
| Query Processing | None | 빠르지만 복합 질문에 약함 |
| Query Processing | Rewrite + Classification | 대부분 충분. 50~100ms 추가 |
| Query Processing | HyDE + Decomposition | 복잡 분석 질문에 적합. 300~500ms 이상 추가 |

## 최소 production 기준

내 기준에서 production RAG의 최소 기준은 다음이다.

```text
Hybrid Search
  + Reranker
  + Evaluation Dataset
  + Latency/Cost Tracking
  + Permission Filter
  + Citation Policy
```

이 중 하나라도 없으면 운영 중 품질 저하를 설명하기 어렵다.

## 설계 판단 순서

RAG 아키텍처를 설계할 때는 모델보다 다음 순서가 먼저다.

| 순서 | 질문 |
| --- | --- |
| 1 | 원본 문서는 신뢰 가능한가 |
| 2 | parser가 구조를 보존하는가 |
| 3 | chunk가 의미 단위인가 |
| 4 | metadata와 권한이 index에 들어가는가 |
| 5 | dense와 sparse를 결합해야 하는가 |
| 6 | reranker latency를 감당할 수 있는가 |
| 7 | 실패를 평가 데이터셋으로 재현할 수 있는가 |

## 다음 글

- [Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking]({% post_url 2026-04-04-study-production-rag-retrieval %})
- [Production RAG Engineering 3: Evaluation, Operations, Checklist]({% post_url 2026-04-04-study-production-rag-evaluation-operations %})
