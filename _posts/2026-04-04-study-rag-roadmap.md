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
# RAG 개인 학습 로드맵

## 학습 목적

RAG를 "검색해서 LLM에 넣는 방식"이 아니라, 검색 품질과 운영 품질을 함께 설계해야 하는 제품 구조로 정리한다.

## 정리 범위

| 소주제 | 정리 관점 |
| --- | --- |
| Naive RAG | 기본 파이프라인과 한계 |
| Advanced RAG | query rewriting, hybrid search, reranking |
| Modular RAG | router, judge, adaptive retrieval |
| Agentic RAG | 검색 판단, 재검색, 도구 선택 |
| RAG Evaluation | faithfulness, context precision, context recall |
| Production RAG | latency, cost, monitoring, safety |

## 작성할 글

| 순서 | 게시글 | 소주제 | 상태 |
| --- | --- | --- | --- |
| 1 | [RAG 완전 가이드 1: 필요성과 기본 구조]({% post_url 2026-04-04-study-rag-why-and-pipeline %}) | RAG 필요성, 기본 구조, embedding, vector DB, semantic search | 작성 |
| 2 | [RAG 완전 가이드 2: Naive, Advanced, Modular, Agentic RAG]({% post_url 2026-04-04-study-rag-evolution-patterns %}) | RAG 진화, 구현 패턴, agentic RAG | 작성 |
| 3 | [RAG 완전 가이드 3: 평가, 도입 로드맵, 논문 타임라인]({% post_url 2026-04-04-study-rag-evaluation-roadmap-papers %}) | 평가 metric, 도입 순서, chunking table, 논문 timeline | 작성 |
| 4 | [Production RAG Engineering 1: 아키텍처와 설계 지점]({% post_url 2026-04-04-study-production-rag-architecture %}) | ingestion, query pipeline, 설계 결정 | 작성 |
| 5 | [Production RAG Engineering 2: Chunking, Embedding, Retrieval, Reranking]({% post_url 2026-04-04-study-production-rag-retrieval %}) | chunking, embedding model, hybrid search, reranker | 작성 |
| 6 | [Production RAG Engineering 3: Evaluation, Operations, Checklist]({% post_url 2026-04-04-study-production-rag-evaluation-operations %}) | generation, agentic RAG, evaluation, latency, cost, checklist | 작성 |

## 후속 분리 후보

| 후보 글 | 분리 이유 |
| --- | --- |
| RAG 논문 타임라인 | 1차 글에 반영 완료. 추후 논문별 상세 리뷰 가능 |
| Chunking 전략 비교 | 1차 글에 반영 완료. 추후 실험 로그로 분리 가능 |
| Hybrid Search와 Reranker 적용 기준 | 1차 글에 반영 완료. 추후 구현 실험으로 분리 가능 |
| Agentic RAG의 검색 판단 흐름 | 1차 글에 반영 완료. Agent 글과 연결 가능 |

## 작성 기준

RAG 글은 기법 목록으로 끝내지 않는다. 각 기법을 "어떤 실패를 줄이기 위한 선택인가"로 정리한다.
