---
title: "RAG 완전 가이드 2: Naive, Advanced, Modular, Agentic RAG"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- rag
- advanced-rag
- modular-rag
- agentic-rag
toc: true
date: 2026-04-04 02:10:00 +0900
comments: false
mermaid: true
math: true
---
# RAG 완전 가이드 2: Naive, Advanced, Modular, Agentic RAG

> **한줄 정의**
> RAG의 진화는 단순 검색-생성 파이프라인에서, 검색 전후를 최적화하고, 라우터와 평가기를 붙이고, 에이전트가 검색을 반복 판단하는 방향으로 진행됐다.

![RAG evolution](/assets/images/study/diagrams/study-rag-as-product-evolution.svg){: width="100%"}

## 1. Naive RAG

Naive RAG는 가장 단순한 구조다.

```text
Indexing Phase
  Documents
  -> Chunking
  -> Embedding
  -> Vector DB

Query Phase
  User Query
  -> Query Embedding
  -> Similarity Search
  -> Prompt + Context
  -> LLM
  -> Response
```

장점은 구현이 빠르다는 것이다. 단점은 검색 결과가 나쁘면 답변도 그대로 망가진다는 점이다.

| 한계 | 설명 |
| --- | --- |
| Garbage In, Garbage Out | 관련 없는 문서가 들어오면 답변도 흔들림 |
| 단순 청킹 | 고정 크기로 자르면 문맥 단절과 주제 혼합이 생김 |
| 무분별한 전달 | 검색 결과 품질 평가 없이 LLM에 전달 |
| 중복/상충 미처리 | 중복 문서나 모순 문서를 구분하지 못함 |

Naive RAG는 MVP에는 적합하지만, 프로덕션 품질을 기대하기 어렵다.

## 2. Advanced RAG

Advanced RAG는 검색 전, 검색 중, 검색 후를 각각 개선한다.

```text
Pre-Retrieval
  -> Query Rewriting
  -> HyDE
  -> Multi-Query

Retrieval
  -> Hybrid Search
  -> Multi-Index
  -> Metadata Filtering

Post-Retrieval
  -> Reranker
  -> Compression
  -> Deduplication
```

| 구간 | 기법 | 목적 |
| --- | --- | --- |
| Pre-Retrieval | Query Rewriting | 구어체 질문을 검색 친화적 질문으로 변환 |
| Pre-Retrieval | HyDE | 가상 답변을 만든 뒤 그 답변으로 검색 |
| Pre-Retrieval | Multi-Query | 여러 관점의 질문으로 검색 재현율 향상 |
| Retrieval | Hybrid Search | dense semantic과 sparse keyword를 결합 |
| Retrieval | Fine-tuned Embedding | 도메인 특화 검색 품질 향상 |
| Retrieval | Multi-Index | 요약, 원문, 메타데이터 인덱스를 계층화 |
| Post-Retrieval | Reranker | 질문-문서 쌍을 다시 평가해 순위 조정 |
| Post-Retrieval | Compression | 관련 문장만 남겨 context noise와 비용 감소 |
| Post-Retrieval | MMR/Dedup | 중복 문서 제거와 다양성 확보 |

원본 학습 노트 기준으로 검색만 개선해도 같은 LLM에서 50% 이상 품질 향상이 가능하다고 정리되어 있다. 모델 교체보다 검색 품질이 먼저다.

## 3. Modular RAG

Modular RAG는 RAG를 고정 파이프라인이 아니라 모듈 조합으로 본다.

```text
User Query
  -> Router
  -> Retrieval / Direct LLM / SQL / Web
  -> Judge
  -> Retry or Generate
  -> Self Evaluation
  -> Response
```

| 모듈 | 역할 |
| --- | --- |
| Router | 검색 필요 여부와 검색 소스 결정 |
| Judge/Critic | 검색 결과가 충분한지 평가 |
| Adaptive Retrieval | 필요할 때만 검색 |
| Multi-Source | Vector DB, Web, SQL, API를 선택 |
| Iterative Retrieval | 여러 차례 검색과 생성을 반복 |
| Memory | 이전 대화와 중간 결과를 유지 |

Modular RAG의 핵심은 "모든 질문을 같은 경로로 처리하지 않는다"는 점이다.

## 4. 대표 구현 패턴

| 패턴 | 핵심 아이디어 | 보존할 수치/특징 |
| --- | --- | --- |
| Self-RAG | LLM이 reflection token으로 검색 필요성, 문서 관련성, 근거성, 유용성을 판단 | `[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]` |
| CRAG | 검색 결과를 Correct, Incorrect, Ambiguous로 분류 후 보정 경로 선택 | 원본 기준 정확도 19~37% 향상 |
| GraphRAG | 지식 그래프와 계층적 커뮤니티 요약으로 글로벌 질문 처리 | 여러 문서에 흩어진 관계 질문에 강점 |
| RAPTOR | 청크를 클러스터링하고 요약해 다단계 추상화 트리 생성 | 원본 기준 QuALITY +20% 향상 |

Self-RAG와 CRAG는 "검색 결과를 그대로 믿지 않는다"는 방향이다. GraphRAG와 RAPTOR는 "문서 조각을 더 큰 구조로 묶는다"는 방향이다.

## 5. Agentic RAG

Agentic RAG는 AI Agent가 검색을 도구로 사용한다.

```text
Query
  -> Plan
  -> Search / SQL / Web / API / Code
  -> Observe
  -> Reflect
  -> Replan if needed
  -> Final Answer
```

| 구성 | 설명 |
| --- | --- |
| Plan | 질문을 하위 작업으로 분해 |
| Reason | 어떤 도구를 쓸지 판단 |
| Observe | 도구 결과를 읽고 품질 평가 |
| Reflect | 부족하면 재검색 또는 다른 도구 선택 |
| Memory | 검색 이력과 중간 결과 유지 |

Agentic RAG가 필요한 질문은 단일 검색으로 답하기 어려운 질문이다.

| 질문 유형 | 적합한 방식 |
| --- | --- |
| 단순 정의 | Direct LLM 또는 Naive RAG |
| 사내 정책 조회 | Single-step RAG |
| 여러 문서 비교 | Advanced/Modular RAG |
| 수치 계산과 근거 추적 | Agentic RAG 또는 GraphRAG |
| 도구 실행이 필요한 분석 | Agentic RAG |

## 6. Speculative RAG

원본 학습 노트에는 Speculative RAG도 별도 기법으로 정리되어 있다. 작은 전문가 모델이 여러 초안을 병렬로 만들고, 큰 모델이 검증하는 draft-then-verify 패턴이다.

핵심은 정확도와 지연시간을 동시에 개선하려는 것이다. 다만 구현 복잡도와 평가 체계가 필요하므로 기본 RAG 단계에서 바로 도입할 대상은 아니다.

## 내 기준

RAG를 고도화하는 순서는 다음이 합리적이다.

```text
Naive RAG
  -> Hybrid Search
  -> Reranker
  -> Evaluation
  -> Router/Judge
  -> Agentic Loop
```

검색 품질을 측정하지 않은 상태에서 agentic loop를 붙이면, 더 비싸고 더 느린 실패가 된다.

## 관련 글

- [RAG 완전 가이드 1: 필요성과 기본 구조]({% post_url 2026-04-04-study-rag-why-and-pipeline %})
- [Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking]({% post_url 2026-04-04-study-production-rag-retrieval %})
