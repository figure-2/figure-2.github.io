---
title: "Knowledge Graph 개인 학습 로드맵"
categories:
- 3.STUDY
- 3-5.KNOWLEDGE_GRAPH
tags:
- study
- ontology
- knowledge-graph
- graphrag
- rag
- guide-review
- reference-note
toc: true
date: 2026-05-16 14:50:00 +0900
comments: false
mermaid: true
math: true
---
# Knowledge Graph 개인 학습 로드맵

## 학습 목적

온톨로지와 지식 그래프를 데이터 구조화 관점에서 이해하고, RAG와 GraphRAG로 연결되는 지점을 정리한다.

## 정리 범위

| 소주제 | 정리 관점 |
| --- | --- |
| Glossary / Taxonomy | 용어와 분류 체계 |
| Ontology | 개념, 관계, 제약을 명시하는 구조 |
| Knowledge Graph | entity와 relation 기반 지식 표현 |
| Triple | subject, predicate, object 단위 |
| GraphRAG | 그래프 기반 검색과 생성의 결합 |
| Entity Resolution | 같은 대상을 하나로 묶는 문제 |
| Tool Ecosystem | Neo4j, Graphiti, LangGraph 등 |

## 작성할 글

| 순서 | 게시글 | 소주제 | 상태 |
| --- | --- | --- | --- |
| 1 | [온톨로지 & 지식 그래프 가이드 1: 개념, 스펙트럼, 트리플]({% post_url 2026-05-16-study-kg-ontology-triple %}) | ontology, knowledge graph, relational DB 차이, ontology spectrum, triple | 작성 |
| 2 | [온톨로지 & 지식 그래프 가이드 2: GraphRAG, 도구, 시작법]({% post_url 2026-05-16-study-kg-graphrag-tools %}) | GraphRAG, vector DB 비교, 80/15/5, 도구 생태계, 시작 순서 | 작성 |

## 후속 분리 후보

| 후보 글 | 분리 이유 |
| --- | --- |
| GraphRAG benchmark 읽는 법 | 1차 글에 반영 완료. 최신 benchmark 검증은 후속 가능 |
| GraphRAG 도구 생태계 비교 | 1차 글에 반영 완료. 실제 사용 비교는 후속 가능 |
| Entity Resolution 체크리스트 | 1차 글에 반영 완료. 운영 체크리스트로 확장 가능 |

## 작성 기준

Knowledge Graph 글은 용어 설명에서 멈추지 않는다. "이 구조가 검색 품질과 추론 가능성에 어떤 영향을 주는가"를 기준으로 쓴다.
