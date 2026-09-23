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

Production RAG의 검색 품질은 chunking, embedding, retrieval, reranking이 함께 결정합니다. 이 글에서는 각 단계의 선택지가 만드는 trade-off를 정리합니다.

## Chunking Strategy

청킹은 RAG 품질의 기초입니다. 전략에 따라 검색 정확도가 20~40% 달라질 수 있습니다.

Vectara 벤치마크 핵심 결과:

Recursive 청킹은 69% 승률로 가장 안정적입니다. Semantic 청킹은 높은 재현율을 보이지만 조각이 작아 end-to-end 정확도에서는 불리합니다. 대부분의 태스크에서는 청크 크기 256~512 토큰이 최적입니다.

| 전략 | 작동 방식 | 최적 크기 | 정밀도 | 재현율 | 구현 복잡도 | 추천 용도 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed-size | 고정 토큰 수 + overlap | 256~512 tok | ●●●○ | ●●●○ | 낮음 | MVP, 빠른 프로토타입 |
| Recursive | 구분자 계층으로 재귀 분할 | 512 tok | ●●●● | ●●●● | 낮음 | 범용 추천 (기본값) |
| Semantic | 임베딩 유사도 변화점에서 분할 | 가변 | ●●●○ | ●●●● | 중간 | 다주제 문서, 대화록 |
| Parent-Child | 작은 청크 검색 → 큰 청크 반환 | 128 / 512 | ●●●● | ●●●● | 높음 | 긴 보고서, 기술 문서 |
| Document Summary | 문서 요약으로 검색 → 원문 반환 | 요약 200 tok | ●●●● | ●●●○ | 높음 | 글로벌 질문, 요약형 답변 |

#### Contextual Retrieval Anthropic, 2024

검색 실패율 67% 감소

각 청크에 문서 수준의 맥락을 접두사로 추가한 뒤 임베딩합니다. 이때 LLM이 맥락 문장을 자동으로 생성합니다.

Before (일반 청크)

```
"매출은 전년 대비 15% 증가했다."
```

After (Contextual 청크)

```
"[2024년 3분기 실적보고서, 영업이익 섹션] 매출은 전년 대비 15% 증가했다."
```

49%↓

Contextual Embedding 단독

67%↓

+ BM25 결합 시

$1.02

10K 청크 처리 비용 (Claude Haiku)

#### Late Chunking Jina AI, 2024 — 2025~2026 주류화

추가 학습 없이 청크 품질 향상

기존 방식은 먼저 문서를 자른 뒤 각각 임베딩하므로 문맥이 손실될 수 있습니다. Late Chunking은 전체 문서를 먼저 임베딩한 뒤 토큰 벡터에서 청크를 추출합니다. 긴 컨텍스트 모델(8K+ tokens)의 어텐션이 문서 전체를 보기 때문에 각 청크가 주변 맥락을 자연스럽게 반영합니다.

Naive Chunking

```
문서 → [청크1, 청크2, 청크3] → [embed(청크1), embed(청크2), embed(청크3)]
```

Late Chunking

```
문서 → embed(전체 문서) → 토큰 벡터 → [pool(토큰1~50), pool(토큰51~100), ...]
```

0 cost

추가 LLM 호출 불필요 (vs Contextual)

8K+

토큰 컨텍스트 모델 필요

jina-v3

API에서 late_chunking 파라미터 지원

#### Agentic Chunking 2025~2026 트렌드

문서 유형별 최적 전략 자동 선택

LLM Agent가 문서 특성을 분석해 청킹 전략을 동적으로 결정합니다. 예를 들어 연구 논문에는 Semantic Chunking, 재무 보고서에는 페이지 단위, 코드 파일에는 함수 단위를 적용합니다. 고정 전략의 한계를 해결할 수 있지만 인제스팅 비용은 증가합니다.

#### Overlap 전략

청크 간 overlap은 10~20%를 권장합니다. 이는 문맥 단절을 방지하지만, 50% 이상에서는 중복 노이즈가 발생합니다.

#### 코드 청킹

코드는 AST 기반 분할이 가장 효과적입니다. 함수·클래스 단위로 자른 뒤 시그니처를 접두사로 추가합니다.

#### 테이블 처리

표는 Markdown/HTML로 변환한 뒤 하나의 청크로 처리합니다. 행 단위 분할은 의미를 훼손할 수 있습니다.

#### Contextual vs Late

Contextual Retrieval은 LLM 비용이 발생하지만 모든 임베딩 모델에 적용할 수 있습니다. Late Chunking은 추가 비용이 없지만 호환 모델이 필요합니다. 둘 다 적용할 수 있다면 결합이 가장 적합합니다.

## Embedding Models

임베딩 모델 선택은 검색 품질의 50% 이상을 결정합니다. 따라서 도메인 벤치마크가 핵심입니다.

| Model | Dim | Max Tokens | MTEB Avg | 다국어 | 특징 |
| --- | --- | --- | --- | --- | --- |
| Cohere embed-v4 | 1024 | 128K | 65.2 | 100+언어 | MTEB 1위, 멀티모달, 압축 지원 |
| OpenAI text-embedding-3-large | 3072 | 8191 | 64.6 | 다국어 | Matryoshka (차원 축소 가능) |
| Voyage-3-large | 1024 | 32K | 64.8 | 다국어 | 코드 특화 모델 별도 제공 |
| BGE-M3 | 1024 | 8192 | 62.1 | 100+언어 | Dense+Sparse+ColBERT 동시 출력 |
| Jina-embeddings-v3 | 1024 | 8192 | 61.5 | 89언어 | Task-specific LoRA, 오픈소스 |
| E5-Mistral-7B | 4096 | 32K | 61.8 | 영어 중심 | LLM 기반, 지시문 지원 |

#### MTEB만 보지 않기

범용 벤치마크 점수와 실제 도메인 성능은 다릅니다. 따라서 반드시

자체 데이터셋에서 벤치마크

하세요. 도메인 특화 파인튜닝 시 12~30% 추가 성능 향상 가능.

#### 사내 문서 검색 (한국어)

Cohere embed-v4 또는 BGE-M3

한국어 성능이 중요합니다. BGE-M3는 오픈소스로 self-host할 수 있고, Cohere는 API 품질이 높습니다.

#### 코드 검색

Voyage-code-3 또는 CodeBERT 계열

코드 특화 모델이 필요합니다. 일반 임베딩 모델은 코드 의미를 이해하는 데 취약합니다.

#### 멀티모달 (이미지 + 텍스트)

Cohere embed-v4 또는 ColPali

PDF 스캔, 차트, 다이어그램이 포함된 문서에 적합합니다. 텍스트를 추출하지 않고 직접 임베딩합니다.

#### 비용 최적화

Matryoshka (OpenAI 256d) 또는 Binary Quantization

차원을 3072에서 256으로 축소해도 성능은 5%만 감소합니다. 스토리지는 12x 절약할 수 있습니다.

## Retrieval Engineering

검색 품질은 전체 RAG 성능의 상한선을 결정합니다. 같은 LLM에서도 검색만 개선해 50% 이상 향상할 수 있습니다.

### Query Translation 패턴

#### Multi-Query

원본 질문을 3~5개 관점에서 재작성한 뒤 각각 검색하고, 결과를 RRF로 결합합니다.

재현율이 중요할 때. 쿼리가 모호할 때.

추가 비용: 약 +200ms, LLM 호출 1회 추가.

#### HyDE (Hypothetical Document)

LLM이 가상 답변을 생성한 뒤 답변의 임베딩으로 검색합니다. 질문보다 답변이 문서와 더 유사하다는 통찰에 기반합니다.

짧은 질문, 개념적 질문. Zero-shot 검색.

추가 비용은 약 +300ms와 LLM 호출 1회입니다. 가상 답변이 틀리면 할루시네이션이 검색으로 전파될 수 있습니다.

#### Step-Back Prompting

구체적 질문에서 한 단계 추상화된 질문을 생성해 더 넓은 범위로 검색합니다.

"X의 Y는?" → "X의 전반적 특성은?"

추가 비용은 약 +200ms입니다. 질문이 과도하게 일반화될 위험이 있습니다.

#### Decomposition

복합 질문을 하위 질문으로 분해하고, 각각을 독립적으로 검색한 뒤 종합합니다.

비교 질문, 다중 조건 질문, 분석 요청.

추가 비용은 약 +500ms~2s와 다수의 검색입니다. 복합 질문에서는 가장 높은 정확도를 기대할 수 있습니다.

#### RAG Fusion

Multi-Query와 Reciprocal Rank Fusion을 결합합니다. 다중 쿼리 결과는 순위 기반으로 결합합니다.

높은 재현율과 다양성이 필요할 때 사용합니다. 검색 결과의 중복을 방지합니다.

추가 비용은 약 +300ms입니다. RRF 상수로는 일반적으로 `k=60`을 사용합니다.

### Hybrid Search 설계

RRF (Reciprocal Rank Fusion)

score(d) = Σ 1 / (k + rank_i(d)) // k = 60 (일반적)

Weighted Hybrid (Convex Combination)

score(d) = α · dense_score(d) + (1-α) · sparse_score(d) // α ∈ [0, 1]

#### Alpha 가이드

| alpha | 검색 성향 | 적합한 상황 |
| --- | --- | --- |
| 0.3 | 키워드 중심 | 제품명, 코드, 고유명사가 많을 때 |
| 0.5 | 균형 | 대부분의 QA에 적합한 기본값 |
| 0.7 | 시맨틱 중심 | 자연어 질문, 개념 검색, 유사 문서 탐색 |

#### Late Interaction: ColBERT Optional — 고급

Bi-Encoder 속도 + Cross-Encoder 정확도

토큰별 임베딩을 저장하고 MaxSim(Maximum Similarity) 연산으로 유사도를 계산합니다. 각 쿼리 토큰과 가장 유사한 문서 토큰의 점수를 합산합니다.

장점:

기존 Bi-Encoder 대비 정확도가 5~15% 향상됩니다. 인덱스를 미리 구축하면 검색 시 빠릅니다.

단점:

토큰별 벡터를 저장하므로 스토리지는 5~10x 증가하고 인덱싱 비용도 증가합니다. Qdrant, Vespa 등 제한된 DB만 지원합니다.

## Reranking

검색 후 결과를 재순위화하는 단계입니다. 가장 적은 노력으로 가장 큰 품질 향상을 얻을 수 있습니다.

| Model | 방식 | 레이턴시 | NDCG@10 | 비용 | 특징 |
| --- | --- | --- | --- | --- | --- |
| Cohere Rerank 3.5 | Cross-Encoder | ~80ms | ●●●● | $2/1K req | API, 다국어, 프로덕션 검증 |
| bge-reranker-v2-m3 | Cross-Encoder | ~60ms | ●●●● | Self-host | 오픈소스, 다국어, GPU 필요 |
| Jina-reranker-v2 | Cross-Encoder | ~50ms | ●●●○ | API/$0.02 | 경량, 다국어 |
| FlashRank | Cross-Encoder | ~15ms | ●●●○ | Free/Self | 초경량, CPU 구동, 빠른 추론 |
| RankGPT / LLM Reranker | LLM Listwise | ~500ms+ | ●●●● | LLM API 비용 | 최고 정확도, 높은 비용/레이턴시 |

#### 2단계 파이프라인

Top-100(벡터 검색) → Top-20(BM25 필터) → Top-5(Reranker) 순으로 처리합니다. 넓게 검색한 뒤 결과 범위를 점차 좁힙니다.

#### 비용 절약 패턴

FlashRank로 Top-100에서 Top-20을 고르고, Cohere로 Top-20에서 Top-5를 고릅니다. 2단계 리랭킹으로 비용과 정확도를 최적화합니다.

---

## 추가 정리

### 핵심 요약

이 글에서는 Production RAG 품질의 핵심 구간인 Chunking, Embedding, Retrieval, Reranking을 다룹니다. 네 단계 중 하나만 약해도 최종 답변 품질이 크게 흔들립니다.

### 보충 해설

Chunking은 검색 가능한 최소 단위를 정하는 작업이고, Embedding은 의미 공간을 만드는 작업입니다. Retrieval은 후보를 넓게 가져오는 단계이며, Reranking은 그 후보를 답변에 쓸 순서로 다시 정렬하는 단계입니다. 실무에서는 먼저 recall을 확보하고, 그다음 reranking으로 precision을 높이는 순서가 안정적입니다.
