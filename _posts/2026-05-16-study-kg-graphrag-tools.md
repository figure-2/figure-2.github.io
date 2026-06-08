---
title: "온톨로지 & 지식 그래프 가이드 2: GraphRAG, 도구, 시작법"
categories:
- 3.STUDY
- 3-5.KNOWLEDGE_GRAPH
tags:
- study
- graphrag
- knowledge-graph
- vector-rag
- ontology
toc: true
date: 2026-05-16 15:10:00 +0900
comments: false
mermaid: true
math: true
---
# 온톨로지 & 지식 그래프 가이드 2: GraphRAG, 도구, 시작법

> **한줄 정의**
> GraphRAG는 vector 검색으로 약한 관계 추론, multi-hop 질문, 집계형 질문을 지식 그래프로 보완하는 접근이다.

## GraphRAG가 필요한 질문

원본 학습 노트 기준으로 GraphRAG는 복잡한 관계 질문에서 강점을 가진다.

| 질문 유형 | Vector RAG | GraphRAG | 추천 |
| --- | --- | --- | --- |
| "X에 대한 문서 찾아줘" | 적합 | 과잉 | Vector |
| "A와 B의 관계가 뭐야?" | 부족 | 적합 | Graph |
| "지난달 X의 총 매출은?" | 불가 | 적합 | Graph |
| "A가 B에게 미친 영향의 경로?" | 불가 | 적합 | Graph |
| "이 주제의 최신 논문 요약해줘" | 적합 | 불필요 | Vector |

원본 보존 수치:

| 항목 | GraphRAG | Vector |
| --- | --- | --- |
| multi-hop reasoning | 86% | 32% |
| 수치 추론 | 100% | 50% |
| 집계 query | 90% | 0% |
| 단순 의미 검색 | 유사 | 유사 |

이 수치는 원본 학습 노트 기준이며 최신 benchmark로 단정하지 않는다.

## 80/15/5 법칙

원본 노트는 기업 query를 다음처럼 나눈다.

| 비율 | 처리 방식 | 의미 |
| --- | --- | --- |
| 80% | Vector Search | 단순 의미 검색 |
| 15% | Graph | 구조화된 관계 추론 |
| 5% | Agent | 완전한 tool 사용과 복합 처리 |

따라서 답은 vector와 graph 중 하나가 아니다. router가 질문 유형에 따라 경로를 나누는 것이다.

![Knowledge graph router](/assets/images/study/diagrams/study-knowledge-graph-router.svg){: width="100%"}

## GraphRAG 구조

![GraphRAG flow](/assets/images/study/diagrams/study-knowledge-graph-graphrag.svg){: width="100%"}

```text
Documents
  -> Entity Extraction
  -> Relation Extraction
  -> Ontology Mapping
  -> Knowledge Graph
  -> Community Summary
  -> Query Router
  -> Graph Retrieval + Vector Retrieval
  -> Answer
```

GraphRAG에서 가장 어려운 부분은 graph DB 설치가 아니라 entity resolution이다.

## Entity Resolution

초기 GraphRAG 구현의 큰 실패 원인은 같은 entity를 여러 개로 인식하는 것이다.

| 표현 1 | 표현 2 | 문제 |
| --- | --- | --- |
| John Doe, 45 | John Doe, age 45 | 같은 사람을 다른 node로 생성 |
| Type 2 Diabetes | T2D | 같은 질병을 다른 entity로 인식 |
| OpenAI Inc. | OpenAI | 조직명 alias 미처리 |

동의어 사전, canonical name, ID mapping, 중복 병합 규칙이 필요하다.

## 도구 생태계

| 도구 | 유형 | 특징 |
| --- | --- | --- |
| Neo4j | Graph DB | 가장 널리 사용, Cypher, desktop app |
| FalkorDB | Graph DB | Redis module, low-latency traversal, GraphRAG SDK |
| Graphiti | Framework | 시간 인식 KG, agent memory 특화 |
| LangChain + LangGraph | Framework | vector + graph hybrid pipeline 구성 |
| TrustGraph | Platform | ontology 기반 context graph 관리 |
| GraphRAG SDK | Platform/SDK | 비정형 데이터에서 ontology와 KG 자동 생성 |

도구를 고르기 전 ontology 범위를 먼저 정해야 한다. 도구는 schema 혼란을 해결해주지 않는다.

## 시작 순서

| 단계 | 설명 |
| --- | --- |
| 1. DB schema에서 시작 | DDL이나 기존 table 구조에서 class, property, relation 후보 추출 |
| 2. 작게 시작 | node type 3~7개, relation type 5~15개 |
| 3. Hybrid로 간다 | 80% vector, 15% graph, 5% agent로 routing |
| 4. Entity Resolution 투자 | alias, synonym, canonical ID 관리 |
| 5. 평가 질문셋 작성 | 관계 질문, 집계 질문, multi-hop 질문을 별도 평가 |

## ROI 해석

원본 노트에는 2024~2025년 production 사례 기준으로 KG 도입 조직이 300~320% ROI를 달성했다는 수치가 있다. 이 수치는 데이터가 준비된 조직의 사례로 봐야 한다.

AI-Ready Data가 없으면 KG는 ROI가 아니라 추가 정리 비용이 된다.

## 내 기준

GraphRAG는 vector RAG의 대체제가 아니다.

```text
Vector:
  문서와 의미 검색

Graph:
  entity와 relation 추론

Agent:
  tool 실행과 다단계 판단
```

좋은 구조는 셋을 하나의 router 아래에 두는 것이다.

## 관련 글

- [온톨로지 & 지식 그래프 가이드 1: 개념, 스펙트럼, 트리플]({% post_url 2026-05-16-study-kg-ontology-triple %})
- [AI-Ready Data 가이드]({% post_url 2026-05-16-study-ai-ready-data %})
