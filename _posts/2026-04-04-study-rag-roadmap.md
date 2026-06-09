---
title: "RAG 개인 학습 로드맵"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- rag
- advanced-rag
- agentic-rag
- guide-review
- reference-note
toc: true
date: 2026-04-04 01:50:00 +0900
comments: false
mermaid: true
math: true
---

RAG를 실무에 적용할 때는 최신 기법을 한 번에 넣는 것보다 작은 baseline을 만들고, 검색 품질과 평가 체계를 순서대로 붙이는 편이 안전하다.

## 실전 가이드

RAG 시스템을 실무에 적용할 때의 권장 사항과 로드맵이다.

### 추천 프로덕션 스택

Application

LangChain / LlamaIndex / Custom

LLM

Claude / GPT-4 / Gemini / Open-source

Reranker

Cohere Rerank / bge-reranker / FlashRank

Embedding

Cohere embed-v4 / OpenAI text-embedding-3 / BGE / E5

Search

Hybrid (Dense + BM25)

Vector DB

Pinecone / Weaviate / Qdrant / pgvector / Chroma

### 단계별 도입 로드맵

1

#### MVP: Naive RAG

기본 파이프라인 구축. 고정 크기 청킹 + 단순 벡터 검색 + LLM 생성. 빠르게 가치를 증명하는 것이 목표.

2

#### 품질 향상: Hybrid + Reranker

BM25 + Dense 하이브리드 검색 도입. Reranker 추가. 이것만으로도 상당한 품질 향상을 볼 수 있습니다.

3

#### 청킹 최적화

Recursive/Semantic 청킹 적용. Contextual Retrieval로 청크에 맥락 추가. 메타데이터 태깅.

4

#### 평가 체계 구축

RAGAS 등으로 자동 평가 파이프라인 구축. 정량적 메트릭으로 개선 효과를 측정합니다.

5

#### 고도화: Modular / Agentic

멀티스텝 질문, 멀티소스 시나리오가 필요할 때. 라우터, 반복 검색, 에이전트 패턴 도입.

### 청킹 전략 비교

| 전략 | 방식 | 장점 | 단점 | 추천 상황 |
| --- | --- | --- | --- | --- |
| Fixed-size | 고정 토큰 수로 분할 | 구현 간단, 예측 가능 | 문맥 단절 | 빠른 프로토타입 |
| Recursive | 구분자 계층으로 재귀 분할 | 안정적 (69% 승률) | 구분자 설정 필요 | 범용 (기본 추천) |
| Semantic | 의미 변화 지점에서 분할 | 높은 재현율 | 조각이 너무 작을 수 있음 | 다주제 문서 |
| Parent-Child | 작은 청크로 검색, 큰 청크 반환 | 정밀 검색 + 풍부한 컨텍스트 | 인덱스 복잡도 증가 | 긴 문서, 보고서 |
| Sentence Window | 문장 단위 + 주변 문장 | 문장 수준 정밀도 | 짧은 문서엔 비효율 | FAQ, 매뉴얼 |

#### 256~512 토큰이 최적

청크 크기는 256~512 토큰이 가장 안정적. 너무 작으면 맥락 부족, 너무 크면 노이즈 증가.

#### 검색 최적화가 먼저

LLM을 바꾸기 전에 검색 품질을 올리세요. 같은 모델에서 검색만 개선해도 50%+ 정확도 향상 가능.

#### 측정 없이 개선 없음

평가 파이프라인을 먼저 만들고, 변경할 때마다 메트릭을 비교하세요. 감이 아닌 데이터로 결정.

#### RAG는 제품이다

한 번 만들고 끝이 아닙니다. KPI를 설정하고 지속적으로 개선하세요. 문서 업데이트, 모델 교체, 파이프라인 조정.

---

## 추가 정리

### 핵심 요약

RAG 학습은 개념보다 실패 유형을 기준으로 잡는 편이 좋다. 검색 실패, chunk 실패, context noise, hallucination, citation 오류를 하나씩 줄이는 방식으로 학습하면 실무 감각이 생긴다.

### 보충 해설

로드맵을 따라갈 때는 먼저 작은 문서셋으로 baseline RAG를 만들고, 그다음 chunking, embedding, hybrid search, reranking, evaluation을 순서대로 추가하는 것이 좋다. 한 번에 모든 기법을 넣으면 어떤 개선이 실제로 효과가 있었는지 알기 어렵다.
