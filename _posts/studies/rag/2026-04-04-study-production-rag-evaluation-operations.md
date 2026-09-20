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

Production RAG의 품질은 검색 단계에서 끝나지 않는다. 검색된 컨텍스트를 어떻게 생성 단계에 전달하고, 응답을 어떻게 평가하며, 운영 중 어떤 지표를 볼지가 함께 설계되어야 한다.

## Generation & Prompting

검색된 context를 LLM에 전달할 때는 "많이 넣기"보다 "필요한 근거를 읽기 좋은 구조로 넣기"가 중요하다.

| 전략 | 목적 | 적합한 상황 |
| --- | --- | --- |
| Citation Prompting | 응답의 각 주장에 출처 표기 강제 | 사내 QA, 고객 응대, 감사 가능한 답변 |
| Lost-in-the-Middle 대응 | 중요한 문서를 context 앞쪽에 배치 | 5개 이상 문서를 넣는 긴 context |
| Context Compression | 관련 문장만 추출해 token 절약 | 비용 최적화, context window 제한 |
| Chain-of-Note | 문서별 관련성 메모 후 종합 | noisy retrieval 상황 |

## Agentic RAG Patterns

Agentic RAG는 단순 pipeline을 넘어 검색, 판단, 반복을 동적으로 수행하는 패턴이다. 강력하지만 비용과 latency가 늘기 때문에 문제 복잡도에 맞춰 써야 한다.

| 패턴 | 핵심 아이디어 | 주의점 |
| --- | --- | --- |
| Self-RAG | LLM이 검색 필요 여부와 응답 근거성을 스스로 판단 | reflection token 설계가 필요 |
| Corrective RAG | 검색 결과를 correct / incorrect / ambiguous로 평가 후 보정 | 평가기 품질이 전체 품질을 좌우 |
| Adaptive RAG | 질문 복잡도에 따라 no retrieval, single-step, multi-step을 선택 | router와 평가셋 필요 |
| RAFT | RAG와 fine-tuning 결합 | 도메인 데이터가 충분할 때만 적합 |
| SimRAG | LLM이 QA 쌍을 생성해 self-training | 생성 데이터 품질 필터링 필요 |
| MCP + Agentic RAG | MCP로 도구와 data source 연결을 표준화 | tool permission과 관측이 중요 |

Adaptive RAG의 예시는 다음처럼 볼 수 있다.

| 질문 복잡도 | 예 | 추천 경로 |
| --- | --- | --- |
| Simple | "Python 버전 확인 방법?" | No retrieval |
| Moderate | "우리 회사 휴가 정책?" | Single-step RAG |
| Complex | "3Q 매출을 경쟁사 대비 분석해줘" | Multi-step Agentic RAG |

## Context Engineering

2025~2026 흐름에서 RAG는 더 넓은 context engineering으로 확장된다. 정적 문서 검색만 다루는 것이 아니라 memory, tool, service 연결까지 포함한다.

| 구성 | 의미 |
| --- | --- |
| RAG | 정적 도메인 지식 검색 |
| Memory | 대화 이력, 사용자 상태, 작업 이력 |
| MCP | 외부 도구, DB, 파일 시스템, API 연결 |

이 관점에서는 RAG가 단일 기능이 아니라 knowledge runtime의 일부가 된다.

## Evaluation Framework

RAG 평가는 retrieval과 generation을 분리해야 한다. 검색이 틀렸는지, 생성이 틀렸는지 구분하지 못하면 개선 방향도 잡기 어렵다.

| 영역 | 메트릭 | 의미 |
| --- | --- | --- |
| Retrieval | Context Precision | 검색된 문서 중 관련 문서 비율 |
| Retrieval | Context Recall | 필요한 정보가 검색 결과에 포함된 비율 |
| Retrieval | MRR | 첫 번째 관련 문서가 얼마나 앞에 있는지 |
| Generation | Faithfulness | 응답이 검색 문서에 충실한지 |
| Generation | Answer Relevancy | 응답이 질문과 관련 있는지 |
| Generation | Answer Correctness | 정답과 비교한 사실 정확도 |

## 평가 도구

| 도구 | 용도 |
| --- | --- |
| RAGAS | faithfulness, relevancy, context precision/recall 자동 평가 |
| DeepEval | pytest 스타일의 RAG regression test |
| LLM-as-Judge | pairwise, pointwise, reference-based 평가 |
| TruLens / Langfuse | production tracing과 feedback loop |

## Production Operations

운영 환경에서는 품질뿐 아니라 latency, 비용, 장애 대응도 함께 본다.

### 레이턴시 예산

| 단계 | P50 목표 | P99 목표 | 최적화 방법 |
| --- | --- | --- | --- |
| Query Processing | 50ms | 150ms | 경량 모델, cache |
| Embedding | 20ms | 50ms | batch 처리, GPU |
| Vector Search | 10ms | 30ms | HNSW, in-memory index |
| Reranker | 60ms | 120ms | FlashRank, cache |
| LLM Generation | 500ms | 2000ms | streaming, prompt 최적화 |
| Total | 약 700ms | 약 2.5s | end-to-end budget 관리 |

### 비용 최적화

| 전략 | 설명 |
| --- | --- |
| Semantic Cache | 유사 질문은 검색과 생성 단계를 건너뜀 |
| Matryoshka Embedding | 차원 축소로 storage와 검색 비용 절감 |
| Router 분기 | 단순 질문은 검색 없이 직접 답변 |
| Prompt Compression | context를 압축해 token 비용 절감 |

## 프로덕션 체크리스트

| 영역 | 체크 항목 |
| --- | --- |
| 검색 품질 | 자체 QA set, hybrid search, reranker, chunking 비교 |
| 안전성 | hallucination 감지, PII filter, citation, fallback |
| 모니터링 | P50/P95/P99 latency, faithfulness, 사용자 feedback, query cost |
| 운영 | 문서 업데이트, incremental indexing, embedding migration, graceful degradation |

## 정리

Production RAG의 마지막 품질은 evaluation과 operations에서 결정된다. 검색이 맞았는지, 답변이 근거를 따랐는지, 비용과 latency가 허용 범위 안인지 계속 측정해야 한다.
