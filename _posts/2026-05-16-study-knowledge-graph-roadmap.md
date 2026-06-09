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

Knowledge Graph를 처음 만들 때는 거대한 ontology부터 설계하지 않는 편이 좋다. 이미 있는 DB schema나 업무 문서에서 entity와 relation을 뽑고, 작은 graph로 효과를 확인하는 순서가 더 현실적이다.

## 실전: 어디서부터 시작할까

온톨로지를 처음 만드는 사람을 위한 단계별 접근이다.

#### DB 스키마에서 시작하라

연구에 따르면, DB 스키마에서 온톨로지를 추출하면 텍스트에서 추출한 것과 성능이 비슷하면서 비용은 훨씬 낮다. DDL(테이블 정의)을 LLM에게 주면 클래스, 속성, 관계를 자동 추출할 수 있다. 이미 있는 데이터의 구조를 활용하라.

#### 작게 시작하라

노드 타입 3~7개, 관계 타입 5~15개로 시작. 50개 클래스의 완벽한 온톨로지보다 5개 클래스의 정확한 온톨로지가 낫다. 필요에 따라 점진적으로 확장.

#### 하이브리드로 가라

벡터 검색을 버리고 그래프로 갈 필요 없다. 80%의 쿼리는 벡터 검색으로 충분하다. 복잡한 관계 추론이 필요한 15%에 그래프를 쓰고, 나머지는 벡터에 맡겨라.

#### 엔티티 해소(Entity Resolution)에 투자하라

초기 GraphRAG 구현에서 가장 큰 문제: "John Doe, 45" vs "John Doe, age 45", "Type 2 Diabetes" vs "T2D". 같은 엔티티를 다른 이름으로 인식하면 그래프가 무너진다. 동의어 사전과 정규화가 핵심.

ROI 참고 사례에서는 지식 그래프 도입 조직이 300~320% ROI를 달성한 것으로 정리되어 있다. 단, 이 수치는 데이터가 준비된 조직의 사례로 봐야 한다. 데이터 구조화와 거버넌스가 먼저 갖춰져야 한다.

---

## 추가 정리

### 핵심 요약

Knowledge Graph 학습은 개념 정의, triple 모델링, schema 설계, graph query, GraphRAG 순서로 진행하면 된다. 처음부터 대규모 그래프를 만들기보다 작은 도메인으로 시작하는 것이 좋다.

### 보충 해설

가장 좋은 출발점은 이미 존재하는 DB schema나 업무 문서다. 테이블, 컬럼, 문서 제목, 사람, 조직, 제품, 사건 같은 엔티티를 뽑고 관계를 정의하면 작은 지식 그래프를 만들 수 있다. 이후 검색과 RAG에 연결하며 효과를 확인한다.
