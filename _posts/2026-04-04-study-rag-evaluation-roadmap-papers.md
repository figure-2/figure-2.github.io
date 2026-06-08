---
title: "RAG 완전 가이드 3: 평가, 도입 로드맵, 논문 타임라인"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- rag
- evaluation
- roadmap
- papers
toc: true
date: 2026-04-04 02:20:00 +0900
comments: false
mermaid: true
math: true
---
# RAG 완전 가이드 3: 평가, 도입 로드맵, 논문 타임라인

> **한줄 정의**
> RAG 평가는 검색 품질과 생성 품질을 분리해서 봐야 하고, 도입은 MVP에서 Hybrid/Reranker, 평가, Modular/Agentic 순서로 진행하는 편이 안전하다.

## 검색과 생성을 분리해서 평가한다

RAG 답변이 틀렸을 때 원인은 둘 중 하나다.

```text
검색 실패
  -> 필요한 문서가 context에 없음

생성 실패
  -> 문서는 들어왔지만 답변이 근거를 잘못 사용
```

그래서 metric도 나눠야 한다.

| 영역 | 지표 | 의미 | 계산 관점 |
| --- | --- | --- | --- |
| Retrieval | Context Precision | 검색된 문서 중 관련 문서 비율 | 관련 문서 / 검색 문서 |
| Retrieval | Context Recall | 필요한 정보가 검색 결과에 들어왔는가 | 검색된 관련 문서 / 전체 관련 문서 |
| Retrieval | MRR | 첫 관련 문서가 얼마나 위에 있는가 | reciprocal rank 평균 |
| Generation | Faithfulness | 답변이 검색 문서에 충실한가 | 문서로 뒷받침되는 주장 비율 |
| Generation | Answer Relevancy | 답변이 질문에 적절한가 | 질문과 답변의 관련성 |
| Generation | Answer Correctness | 정답과 비교한 사실 정확도 | ground truth 필요 |

원본 노트에서 가장 중요하게 본 지표는 `Faithfulness`다. 검색 문서가 들어왔는데도 답변이 없는 내용을 만들면 RAG의 신뢰성이 무너진다.

## 평가 도구

| 도구 | 쓰는 위치 | 핵심 용도 |
| --- | --- | --- |
| RAGAS | 개발 단계 | faithfulness, relevancy, context precision/recall 자동 평가 |
| DeepEval | CI/CD | pytest 스타일 RAG 유닛 테스트와 회귀 방지 |
| LLM-as-Judge | 유연한 평가 | pairwise, pointwise, reference-based 평가 |
| TruLens / Langfuse | 운영 | trace, feedback, 품질 모니터링 |
| RAGBench | 벤치마크 | 원본 기준 12개 도메인, 100K+ 예제 |

평가 도구는 하나만 고르는 문제가 아니다. 개발 단계의 자동 평가, CI의 회귀 테스트, 운영 trace가 서로 연결되어야 한다.

## 추천 프로덕션 스택

| 계층 | 후보 |
| --- | --- |
| Application | LangChain, LlamaIndex, Custom |
| LLM | Claude, GPT, Gemini, open-source model |
| Reranker | Cohere Rerank, bge-reranker, FlashRank |
| Embedding | Cohere embed, OpenAI text-embedding, BGE, E5 |
| Search | Hybrid, Dense + BM25 |
| Vector DB | Pinecone, Weaviate, Qdrant, pgvector, Chroma |

스택 선택보다 중요한 것은 평가 데이터셋이다. 평가셋이 없으면 어느 선택이 나아졌는지 판단할 수 없다.

## 단계별 도입 로드맵

| 단계 | 목표 | 구현 |
| --- | --- | --- |
| 1. MVP | 빠른 가치 검증 | 고정 크기 청킹, 단순 벡터 검색, LLM 생성 |
| 2. 품질 향상 | 검색 품질 개선 | BM25 + Dense hybrid, reranker 추가 |
| 3. 청킹 최적화 | 검색 단위 개선 | Recursive/Semantic chunking, contextual retrieval, metadata tagging |
| 4. 평가 체계 | 개선 효과 측정 | RAGAS, DeepEval, regression set |
| 5. 고도화 | 복잡 질문 처리 | Modular RAG, Agentic RAG, router, judge, iterative retrieval |

처음부터 Agentic RAG로 가면 실패 원인 분리가 어렵다. 먼저 검색과 생성 평가를 나눠야 한다.

## 청킹 전략 비교

| 전략 | 방식 | 장점 | 단점 | 추천 상황 |
| --- | --- | --- | --- | --- |
| Fixed-size | 고정 token 수로 분할 | 구현 간단, 예측 가능 | 문맥 단절 | 빠른 prototype |
| Recursive | 구분자 계층으로 재귀 분할 | 원본 기준 69% 승률, 안정적 | 구분자 설정 필요 | 범용 기본값 |
| Semantic | 의미 변화 지점에서 분할 | 높은 재현율 | 조각이 너무 작을 수 있음 | 다주제 문서 |
| Parent-Child | 작은 chunk로 검색, 큰 chunk 반환 | 정밀 검색과 풍부한 context | 인덱스 복잡도 증가 | 긴 문서, 보고서 |
| Sentence Window | 문장 단위 + 주변 문장 | 문장 수준 정밀도 | 짧은 문서에는 비효율 | FAQ, 매뉴얼 |

원본 기준으로 chunk 크기는 256~512 token이 가장 안정적인 범위로 정리되어 있다.

## 주요 논문 타임라인

| 연도 | 항목 | 핵심 기여 |
| --- | --- | --- |
| 2020 | RAG | DPR + BART 결합, RAG-Sequence와 RAG-Token 제안 |
| 2020 | DPR | dense vector 기반 passage retrieval 표준화 |
| 2022 | HyDE | 질문 대신 가상 답변을 생성해 검색 |
| 2023 | Self-RAG | reflection token으로 검색 필요성과 근거성을 자체 판단 |
| 2024 | CRAG | 검색 결과를 Correct, Incorrect, Ambiguous로 분류 후 보정 |
| 2024 | RAPTOR | recursive clustering과 summary tree로 다단계 추상화 |
| 2024 | GraphRAG | 지식 그래프와 계층적 community summary 결합 |
| 2024 | Contextual Retrieval | chunk에 문서 수준 맥락을 접두사로 추가 |
| 2024 | Adaptive RAG | 질문 복잡도에 따라 retrieval strategy를 동적으로 선택 |
| 2024 | Late Chunking | 전체 문서 임베딩 후 token vector에서 chunk representation 추출 |
| 2024 | RAFT | RAG와 fine-tuning 결합, distractor 무시 학습 |
| 2025 | SimRAG | unlabeled corpus에서 synthetic QA 생성과 self-training |
| 2025 | MCP | 외부 도구와 데이터 연결 표준화 |
| 2026 | Context Engineering | RAG, memory, tool context를 통합하는 방향 |

이 표의 연도와 수치는 원본 학습 노트 기준이다. 발행 시 최신 논문 상태로 단정하지 않는다.

## 내 기준

RAG 평가의 목적은 점수 만들기가 아니다. 실패 원인을 분리하는 것이다.

```text
질문
  -> 검색 결과
  -> 답변
  -> 근거 검증
  -> 사용자 피드백
  -> 회귀 평가
```

이 루프가 없으면 RAG는 제품이 아니라 demo다.

## 관련 글

- [RAG 완전 가이드 2: Naive, Advanced, Modular, Agentic RAG]({% post_url 2026-04-04-study-rag-evolution-patterns %})
- [Production RAG Engineering 3: Evaluation, Operations, Checklist]({% post_url 2026-04-04-study-production-rag-evaluation-operations %})
