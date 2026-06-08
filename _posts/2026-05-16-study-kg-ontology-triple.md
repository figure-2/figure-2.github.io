---
title: "온톨로지 & 지식 그래프 가이드 1: 개념, 스펙트럼, 트리플"
categories:
- 3.STUDY
- 3-5.KNOWLEDGE_GRAPH
tags:
- study
- ontology
- knowledge-graph
- triple
toc: true
date: 2026-05-16 15:00:00 +0900
comments: false
mermaid: true
math: true
---
# 온톨로지 & 지식 그래프 가이드 1: 개념, 스펙트럼, 트리플

> **한줄 정의**
> 온톨로지는 지식의 schema이고, 지식 그래프는 그 schema 위에 실제 entity와 relation을 연결한 데이터다.

## 온톨로지와 지식 그래프

| 개념 | 의미 | 비유 |
| --- | --- | --- |
| Ontology | 도메인의 개념, 관계, 속성, 제약을 정의한 체계 | 빈 Excel의 열 제목과 규칙 |
| Knowledge Graph | ontology를 기반으로 실제 데이터를 node와 edge로 연결한 graph | Excel에 채워진 실제 데이터 |

온톨로지는 "고객은 주문을 하고, 주문은 제품을 포함한다"는 규칙이다. 지식 그래프는 "김철수가 주문#1234를 했고, 주문#1234는 MacBook Pro를 포함한다"는 실제 사실이다.

## 관계형 DB와의 차이

관계형 DB는 table과 join으로 관계를 표현한다. 관계가 깊어지면 join이 많아진다.

지식 그래프는 관계 자체를 일급 시민으로 저장한다.

```text
김철수
  -> 주문했다
  -> 주문#1234
  -> 포함한다
  -> MacBook Pro
  -> 속한다
  -> 노트북 카테고리
```

깊은 관계를 따라가야 하는 질문에서는 graph traversal이 자연스럽다.

## AI에 중요한 이유

원본 학습 노트 기준 보존 수치:

| 조건 | 정확도 |
| --- | --- |
| 지식 그래프 없이 LLM 사용 | 16.7% |
| 지식 그래프 연결 | 56.2% |
| entity 10개 이상 복잡 질문에서 vector 검색 | 0%까지 하락 |
| entity 10개 이상 복잡 질문에서 KG 기반 | 70% 이상 유지 |

이 수치는 원본 학습 노트 기준이며 최신 benchmark로 단정하지 않는다. 중요한 해석은 복잡한 관계 질문일수록 단순 vector 검색의 한계가 커진다는 점이다.

## 온톨로지 스펙트럼

처음부터 복잡한 OWL ontology를 만들 필요는 없다.

| 단계 | 설명 | 용도 |
| --- | --- | --- |
| Glossary | 단어와 정의 목록 | 시작점 |
| Taxonomy | is-a 계층 구조 | 분류 |
| Thesaurus | 동의어, 관련어, 상위어/하위어 | 검색 확장 |
| Ontology | 속성, 제약, 논리 규칙, 추론 | 기계 추론 |

원본 노트의 실무 기준은 작게 시작하는 것이다. 노드 타입 3~7개, 관계 타입 5~15개가 실전 시작점으로 적합하다. 50개 class의 깊은 상속 구조보다 5개 class와 10개 속성의 단순한 ontology가 더 정확하게 추출될 수 있다.

## 트리플

지식 그래프의 기본 단위는 triple이다.

```text
Subject -> Predicate -> Object
```

예:

```text
김철수 -> 주문했다 -> 주문#1234
주문#1234 -> 포함한다 -> MacBook Pro
MacBook Pro -> 속한다 -> 노트북 카테고리
```

![Knowledge graph triple](/assets/images/study/diagrams/study-knowledge-graph-triple.svg){: width="100%"}

## 트리플이 중요한 이유

트리플은 단순한 문장 분해가 아니다. entity와 relation을 기계가 따라갈 수 있는 구조로 만든다.

| 질문 | 필요한 traversal |
| --- | --- |
| 김철수가 주문한 제품은? | 고객 -> 주문 -> 제품 |
| 그 제품의 카테고리는? | 고객 -> 주문 -> 제품 -> 카테고리 |
| 같은 카테고리를 산 고객은? | 제품 -> 카테고리 -> 다른 제품 -> 다른 고객 |

관계가 깊어질수록 단순 keyword나 vector similarity보다 graph 구조가 강해진다.

## ontology 설계 시작 기준

| 질문 | 예 |
| --- | --- |
| 핵심 entity는 무엇인가 | 고객, 주문, 제품, 카테고리 |
| 관계는 무엇인가 | 주문했다, 포함한다, 속한다 |
| 속성은 무엇인가 | 주문일, 금액, 수량 |
| 제약은 무엇인가 | 주문은 최소 1개 제품을 포함 |
| 동의어는 무엇인가 | 고객, 사용자, 회원 |

처음에는 완벽한 ontology보다 정확한 작은 ontology가 낫다.

## 내 기준

Knowledge Graph는 "더 똑똑한 RAG"가 아니다. 다른 질문 유형을 처리하기 위한 다른 데이터 구조다.

```text
문서 찾기
  -> Vector

관계 따라가기
  -> Graph

둘 다 필요
  -> Hybrid Router
```

## 다음 글

- [온톨로지 & 지식 그래프 가이드 2: GraphRAG, 도구, 시작법]({% post_url 2026-05-16-study-kg-graphrag-tools %})
