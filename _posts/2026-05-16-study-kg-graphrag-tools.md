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
GraphRAG

## GraphRAG: 지식 그래프 + RAG

벡터 검색만으로 한계에 부딪혔다면 — 그래프가 답일 수 있다

멀티홉 추론 (복잡한 질문)

GraphRAG 86% vs Vector 32%

GraphRAG 86%

Vector 32%

수치 추론

GraphRAG 100% vs Vector 50%

GraphRAG 100%

Vector 50%

집계 쿼리 (스키마 기반)

GraphRAG 90% vs Vector 0%

GraphRAG 90%

Vector 0%

단순 의미 검색 (문서 찾기)

비슷 — 그래프 오버헤드만 추가

GraphRAG ~동등

Vector ~동등

출처: FalkorDB, TianPan, Lettria

| 질문 유형 | Vector RAG | GraphRAG | 추천 |
| --- | --- | --- | --- |
| "X에 대한 문서 찾아줘" | 적합 | 과잉 | Vector |
| "A와 B의 관계가 뭐야?" | 부족 | 적합 | Graph |
| "지난달 X의 총 매출은?" | 불가 | 적합 | Graph |
| "A가 B에게 미친 영향의 경로?" | 불가 | 적합 | Graph |
| "이 주제의 최신 논문 요약해줘" | 적합 | 불필요 | Vector |

80/15/5 법칙:

2026년 벤치마크 합의

에 따르면, 기업 쿼리의 약 80%는 단순 의미 검색(Vector), 15%는 구조화된 추론(Graph), 5%는 완전한 에이전트 처리가 필요하다. 둘 중 하나를 고르는 것이 아니라

하이브리드 라우터

가 답이다.

Tools

## 실전 도구 생태계

지식 그래프를 직접 만들어보기 위한 도구들

Graph DB

#### Neo4j

가장 널리 사용되는 그래프 DB. Cypher 쿼리 언어, 데스크톱 앱으로 빠른 시작 가능. "Ontologies as a First-Class Citizen" 로드맵 (2026).

Cypher · Java · 커뮤니티 최대

Graph DB

#### FalkorDB

실시간 AI 특화 그래프 DB. 희소 행렬 곱셈 기반 순회로 초저지연. Redis 모듈로 동작. GraphRAG SDK로 자동 온톨로지 생성 지원.

C · Redis Module · Docker 한 줄 시작

Framework

#### Graphiti (by Zep)

시간 인식 지식 그래프 프레임워크. AI 에이전트 메모리 특화. Neo4j, FalkorDB, Amazon Neptune 등 다양한 DB 지원. GitHub 45k+ 스타.

Python · 멀티에이전트 · 실시간

Framework

#### LangChain + LangGraph

LangChain 생태계에서 GraphRAG 파이프라인 구축. Neo4j, FalkorDB 통합. 벡터 + 그래프 하이브리드 검색 지원.

Python/JS · 가장 넓은 통합

Platform

#### TrustGraph

Context Operating System. OntologyRAG 지원 — 온톨로지 기반 컨텍스트 그래프를 자동 구축하고 관리.

오픈소스 · OntologyRAG

Platform

#### GraphRAG SDK (FalkorDB)

비정형 데이터에서 자동으로 온톨로지를 감지하고 지식 그래프를 생성. 수동/자동 온톨로지 관리 모두 지원.

Python · 자동 온톨로지 · 프로덕션급

---

## 추가 정리

### 핵심 요약

GraphRAG는 문서를 벡터로만 찾는 방식의 한계를 보완하기 위해 지식 그래프를 함께 사용하는 접근이다. 관계, 엔티티, 경로, 커뮤니티 구조를 검색과 추론에 활용한다.

### 보충 해설

Vector RAG는 의미적으로 가까운 chunk를 찾는 데 강하고, Knowledge Graph는 명시적 관계를 따라가는 데 강하다. 둘 중 하나가 항상 우위인 것이 아니라 질문 유형에 따라 적합성이 달라진다. 도구 선택은 데이터의 관계 밀도와 운영 복잡도를 기준으로 해야 한다.
