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

Production RAG는 검색 파이프라인 하나로 끝나지 않는다. ingestion, query processing, hybrid retrieval, reranking, generation, evaluation, cache, monitoring이 함께 설계되어야 운영 가능한 시스템이 된다.

이 글은 프로덕션 RAG의 전체 아키텍처와 핵심 설계 지점을 정리한다.

## Production Architecture

프로덕션 RAG 시스템의 전체 흐름은 offline ingestion pipeline과 online query pipeline으로 나눌 수 있다.

```text
OFFLINE — INGESTION PIPELINE
Data Sources
PDF, DB, API, Web
Parser
Unstructured
Chunker
Recursive/Semantic
Context Enrichment
Contextual Prefix
Metadata Tagging
Embed Model
Dense + Sparse
Index Store
Dense Vector Index
Sparse BM25 Index
Metadata Store
ONLINE — QUERY PIPELINE
User
Query 입력
Semantic Cache
Hit → 바로 응답
Query Processing
Classification
Rewriting / HyDE
Decomposition
Router
경로 분기
Hybrid Search
Dense: ANN (HNSW/IVF)
Sparse: BM25 / SPLADE
index lookup
RRF / alpha weighting
Reranker
Cross-Encoder
Top-K → Top-N
LLM
Generation
+ Citation
Response
Evaluation & Monitoring
Faithfulness · Relevancy · Latency · Cost Tracking
RAGAS · LLM-as-Judge · User Feedback Loop
Guardrails
Hallucination Detection
PII Filter · Toxicity Check
no-retrieval path
Offline Ingestion
Retrieval
Post-processing
Evaluation
Safety
```

### 핵심 설계 결정 포인트

Indexing 전략은?

Dense-only

단순 구현, 시맨틱 매칭 우수. 키워드에 약함.

Hybrid (Dense + BM25) ✓

대부분의 프로덕션 권장. RRF로 결합. 검색 실패율 ~35% 감소.

Hybrid + Late Interaction

최고 정확도. ColBERT/ColPali. 스토리지 5~10x 증가.

Reranker를 도입해야 할까?

Yes — 거의 항상 ✓

MAP +52% 향상. 100ms 이하 추가 레이턴시. Top-100 → Top-5 재순위.

Skip — 초저지연 요구시

P99 < 200ms 필요한 실시간 서비스. Embedding 품질로 보완.

Query Processing은 얼마나?

None (Naive)

원본 쿼리 그대로. 빠르지만 복합 질문에 약함.

Rewrite + Classification ✓

LLM으로 쿼리 최적화. 50~100ms 추가. 대부분 충분.

Full (HyDE + Decomposition)

최고 재현율. 300~500ms 추가. 복잡한 분석 질문에 적합.

---

## 추가 정리

### 핵심 요약

Production RAG는 offline ingestion pipeline과 online query pipeline을 분리해서 봐야 한다. 문서를 잘 넣는 문제와 질문에 맞게 잘 꺼내는 문제는 서로 다른 설계 영역이다.

### 보충 해설

Offline 단계에서는 parsing, chunking, metadata, embedding, index 품질이 중요하다. Online 단계에서는 query rewriting, routing, hybrid search, reranking, generation, monitoring이 중요하다. 장애가 났을 때 어느 단계의 문제인지 분리할 수 있어야 운영 가능한 RAG가 된다.
