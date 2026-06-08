---
title: "Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- production-rag
- chunking
- embedding
- retrieval
- reranking
toc: true
date: 2026-04-04 12:32:03 +0900
comments: false
mermaid: true
math: true
---
# Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking

> **한줄 정의**
> 검색 품질은 chunking, embedding, query translation, hybrid search, reranking이 함께 결정한다.

## 기준

이 글의 모델명, 수치, 비용, latency는 원본 학습 노트 기준이다. 최신 순위나 가격으로 단정하지 않는다.

## Chunking Strategy

원본 노트 기준으로 청킹 전략에 따라 검색 정확도는 20~40% 달라질 수 있다. Vectara benchmark 요약에서는 Recursive 청킹이 69% 승률로 가장 안정적인 전략으로 정리되어 있다.

| 전략 | 작동 방식 | 최적 크기 | 정밀도 | 재현율 | 구현 복잡도 | 추천 용도 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed-size | 고정 token 수 + overlap | 256~512 tok | 중 | 중 | 낮음 | MVP, 빠른 prototype |
| Recursive | 구분자 계층으로 재귀 분할 | 512 tok | 높음 | 높음 | 낮음 | 범용 기본값 |
| Semantic | embedding 유사도 변화점에서 분할 | 가변 | 중 | 높음 | 중간 | 다주제 문서, 대화록 |
| Parent-Child | 작은 chunk 검색 후 큰 chunk 반환 | 128 / 512 | 높음 | 높음 | 높음 | 긴 보고서, 기술 문서 |
| Document Summary | 문서 요약으로 검색 후 원문 반환 | 요약 200 tok | 높음 | 중 | 높음 | 글로벌 질문, 요약형 답변 |

## Contextual Retrieval

Contextual Retrieval은 각 chunk에 문서 수준 맥락을 접두사로 붙인 뒤 embedding한다.

```text
Before:
"매출은 전년 대비 15% 증가했다."

After:
"[2024년 3분기 실적보고서, 영업이익 섹션]
매출은 전년 대비 15% 증가했다."
```

원본 기준 보존 수치:

| 항목 | 수치 |
| --- | --- |
| Contextual Embedding 단독 | 검색 실패율 49% 감소 |
| BM25 결합 | 검색 실패율 67% 감소 |
| 10K chunk 처리 비용 | 원본 기준 약 $1.02 |

장점은 embedding model 종류와 무관하게 적용 가능하다는 점이다. 단점은 ingestion 시 LLM 호출 비용이 든다는 점이다.

## Late Chunking

Late Chunking은 먼저 문서를 자르지 않는다. 전체 문서를 먼저 embedding하고 token vector에서 chunk representation을 만든다.

```text
Naive Chunking
문서 -> [chunk1, chunk2, chunk3] -> embed(chunk)

Late Chunking
문서 -> embed(전체 문서) -> token vector -> pool(token range)
```

| 관점 | Late Chunking |
| --- | --- |
| 추가 LLM 비용 | 없음 |
| 필요 조건 | 긴 context embedding model |
| 장점 | 주변 문맥을 자연스럽게 반영 |
| 제약 | 호환 model과 API 지원 필요 |

Contextual Retrieval은 비용이 들지만 범용적이고, Late Chunking은 비용이 낮지만 호환성이 필요하다. 둘 다 가능하면 결합하는 것이 가장 좋다.

## Agentic Chunking

Agentic Chunking은 문서 유형에 따라 chunking 전략 자체를 동적으로 선택한다.

| 문서 유형 | 선택 전략 |
| --- | --- |
| 연구 논문 | Semantic Chunking |
| 재무 보고서 | 페이지 또는 섹션 단위 |
| 코드 파일 | 함수/클래스 단위 |
| 표 중심 문서 | 표 전체 보존 |

고정 전략의 한계를 줄일 수 있지만 ingestion 비용과 평가 복잡도가 증가한다.

## 실무 청킹 규칙

| 규칙 | 기준 |
| --- | --- |
| Overlap | 10~20% 권장, 50% 이상은 중복 noise 위험 |
| Code | AST 기반으로 함수/클래스 단위 분할 |
| Table | Markdown/HTML로 변환 후 통째로 하나의 chunk로 보존 |
| Metadata | 날짜, 섹션, 권한, version을 반드시 포함 |

## Embedding Models

| Model | Dim | Max Tokens | MTEB Avg | 다국어 | 특징 |
| --- | --- | --- | --- | --- | --- |
| Cohere embed-v4 | 1024 | 128K | 65.2 | 100+언어 | 원본 기준 MTEB 1위, multimodal, 압축 지원 |
| OpenAI text-embedding-3-large | 3072 | 8191 | 64.6 | 다국어 | Matryoshka, 차원 축소 가능 |
| Voyage-3-large | 1024 | 32K | 64.8 | 다국어 | 코드 특화 모델 별도 제공 |
| BGE-M3 | 1024 | 8192 | 62.1 | 100+언어 | Dense, Sparse, ColBERT 동시 출력 |
| Jina-embeddings-v3 | 1024 | 8192 | 61.5 | 89언어 | Task-specific LoRA, open source |
| E5-Mistral-7B | 4096 | 32K | 61.8 | 영어 중심 | LLM 기반, instruction 지원 |

MTEB만 보면 안 된다. 도메인 데이터셋으로 직접 평가해야 한다. 원본 노트 기준으로 도메인 특화 fine-tuning은 12~30% 추가 성능 향상이 가능하다고 정리되어 있다.

## 모델 선택 기준

| 상황 | 선택 기준 |
| --- | --- |
| 한국어 사내 문서 | Cohere embed-v4 또는 BGE-M3 계열 |
| 코드 검색 | code-specialized embedding 필요 |
| 이미지/차트 포함 문서 | multimodal embedding 또는 ColPali 계열 |
| 비용 최적화 | Matryoshka 또는 binary quantization |

원본 기준 OpenAI Matryoshka는 3072d에서 256d로 줄여도 성능 감소를 약 5% 수준으로 보고, storage는 12배 절약할 수 있다고 정리되어 있다.

## Query Translation

| 패턴 | 방식 | 적합한 경우 | 비용/위험 |
| --- | --- | --- | --- |
| Multi-Query | 질문을 3~5개 관점으로 재작성 | 모호한 질문, 높은 recall 필요 | LLM 1회 추가, 약 +200ms |
| HyDE | 가상 답변을 생성해 그 embedding으로 검색 | 짧은 질문, 개념 질문 | hallucination 전파 위험, 약 +300ms |
| Step-Back | 구체 질문을 추상 질문으로 변환 | 배경 지식이 필요한 질문 | 과도한 일반화 위험 |
| Decomposition | 복합 질문을 하위 질문으로 분해 | 비교, 다중 조건, 분석 요청 | +500ms~2s, 다수 검색 |
| RAG Fusion | Multi-Query 결과를 RRF로 결합 | 다양성과 recall 동시 필요 | 중복 제거 필요 |

## Hybrid Search 공식

RRF는 여러 검색 결과의 순위를 결합한다.

$$
score(d) = \sum_i \frac{1}{k + rank_i(d)}
$$

원본 기준 일반적인 `k`는 60이다.

Weighted Hybrid는 dense와 sparse 점수를 가중합한다.

$$
score(d) = \alpha \cdot dense\_score(d) + (1-\alpha) \cdot sparse\_score(d)
$$

| alpha | 의미 | 추천 상황 |
| --- | --- | --- |
| 0.3 | keyword 중심 | 제품명, 코드, 고유명사가 많을 때 |
| 0.5 | 균형 | 범용 QA |
| 0.7 | semantic 중심 | 자연어 질문, 개념 검색, 유사 문서 탐색 |

## Late Interaction

ColBERT 계열 late interaction은 bi-encoder 속도와 cross-encoder 정확도 사이의 절충이다.

| 장점 | 단점 |
| --- | --- |
| 기존 bi-encoder 대비 정확도 5~15% 향상 가능 | token별 vector 저장으로 storage 5~10배 증가 |
| index를 미리 만들면 query 시 빠름 | indexing 비용 증가 |
| MaxSim으로 token-level matching 가능 | 지원 DB와 운영 복잡도 제한 |

## Reranking

Reranking은 검색 후 재순위화다. 원본 노트 기준으로 가장 적은 노력으로 큰 품질 향상을 주는 단계로 정리되어 있다.

| Model | 방식 | 레이턴시 | NDCG@10 | 비용 | 특징 |
| --- | --- | --- | --- | --- | --- |
| Cohere Rerank 3.5 | Cross-Encoder | 약 80ms | 높음 | $2/1K req | API, 다국어, production 검증 |
| bge-reranker-v2-m3 | Cross-Encoder | 약 60ms | 높음 | Self-host | open source, 다국어, GPU 필요 |
| Jina-reranker-v2 | Cross-Encoder | 약 50ms | 중상 | API/$0.02 | 경량, 다국어 |
| FlashRank | Cross-Encoder | 약 15ms | 중상 | Free/Self | 초경량, CPU 가능 |
| RankGPT / LLM Reranker | LLM Listwise | 500ms+ | 높음 | LLM API 비용 | 정확도 높지만 비용과 latency 큼 |

## 2단계 재순위화

```text
Top-100  Vector Search
  -> Top-20  FlashRank or BM25 filter
  -> Top-5   Strong Reranker
```

넓게 검색하고 좁게 재정렬하는 방식이다. 비용이 중요하면 FlashRank로 1차 축소 후 고성능 reranker를 적용한다.

## 내 기준

검색 품질 개선 순서는 다음이 가장 안전하다.

```text
좋은 parsing
  -> Recursive chunking
  -> Metadata
  -> Hybrid search
  -> Reranker
  -> Query translation
  -> Late interaction
```

초기부터 복잡한 query decomposition이나 ColBERT를 넣기보다, 평가셋과 reranker를 먼저 붙이는 편이 낫다.

## 관련 글

- [Production RAG Engineering 1: 아키텍처와 설계 지점]({% post_url 2026-04-04-study-production-rag-architecture %})
- [Production RAG Engineering 3: Evaluation, Operations, Checklist]({% post_url 2026-04-04-study-production-rag-evaluation-operations %})
