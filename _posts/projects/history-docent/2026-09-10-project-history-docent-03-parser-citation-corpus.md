---
title: "03. 책 한 권을 citation 가능한 검색 데이터로 바꾸기"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- Document AI
- Data Pipeline
- chunking
- Citation
toc: true
date: 2026-09-10 09:00:00 +0900
comments: true
mermaid: false
math: false
---

PDF에서 텍스트를 추출했다고 바로 RAG에 넣을 수 있는 것은 아니었다. 검색 결과가 맞더라도 답변에서 원문 페이지를 되찾지 못하면 citation을 만들 수 없었다. 반대로 페이지 번호만 남기고 문서 구조를 잃으면 같은 인물과 사건이 여러 장에 걸쳐 등장할 때 검색 단위가 흐려졌다.

그래서 전처리의 목표를 "텍스트를 많이 추출하는 것"이 아니라 "검색한 문장을 원천 block까지 되짚을 수 있게 만드는 것"으로 정했다.

## Parser 출력 형식을 하나의 schema로 고정했다

Upstage Parser 결과에는 제목, 본문, 표 같은 element가 섞여 있다. 먼저 각 element를 공통 schema로 정규화했다.

정규화 단계에서 남긴 정보는 다음과 같다.

- source document 식별자
- parser element와 normalized block 식별자
- page와 page span
- heading과 section 경로
- element type
- 정제된 text

검색 단계에서는 이 정보를 모두 사용하지 않는다. 하지만 citation을 만들거나 실패 사례를 조사할 때는 필요하다. 특정 child chunk가 검색됐을 때 어느 parent와 source block에서 왔는지 복구할 수 있어야 했다.

## 장소 이름을 문서 밖의 별도 catalog로 관리했다

도서의 목차는 관광지 중심으로 쓰이지 않았다. 한 장 안에 인물, 사건, 제도와 여러 장소가 같이 등장한다. 사용자는 "정조의 정책"보다 "수원 화성에서 정조가 뭘 했어?"처럼 장소를 먼저 말할 가능성이 높다.

장소 catalog에는 대표 이름과 별칭, 상위 지역, 관련 역사 맥락을 분리해 두었다. 이 catalog는 원문을 대신하지 않는다. 질문에 포함된 장소를 찾고 query rewrite에 맥락을 제공하는 역할만 한다.

이 분리를 해두면 장소명이 검색 문서에 정확히 등장하지 않더라도 관련 인물이나 사건을 검색어에 보완할 수 있다. 다만 모든 질문에 장소 정보를 강제로 붙이면 오히려 순위가 나빠지는 사례가 있었다. 그래서 전체 place rewrite가 아니라 짧은 음성형 질문을 대상으로 한 제한적 rewrite가 최종 후보가 됐다.

## 검색 단위와 설명 단위를 나눴다

한 덩어리가 너무 크면 검색 점수가 희석된다. 너무 작으면 앞뒤 문맥과 citation 범위를 잃는다. 현재 구조에서는 child를 검색하고 parent를 문맥 확장 단위로 사용한다.

| 단위 | 역할 |
| --- | --- |
| normalized block | parser 원천과 page provenance 보존 |
| child chunk | Retriever가 비교하는 검색 단위 |
| parent chunk | 선택된 child 주변의 설명 문맥 |
| evidence item | Generation에 전달할 순위와 citation 정보 |

Parent boundary는 heading1 구조를 사용한다. Child는 parent 안의 block을 병합해 만들고, 한 block overlap을 둔다. Child는 원천 block ID와 page span을 계속 보존한다.

이 구조가 필요한 이유는 검색과 답변의 요구가 다르기 때문이다. Retriever는 짧고 구별되는 단위를 선호한다. 생성 모델은 답을 설명할 수 있는 주변 문맥이 필요하다. 두 요구를 한 크기의 chunk로 해결하려 하지 않았다.

## 초기 조사와 현재 기준선은 구분했다

프로젝트 초기에 Parser API 후보를 비교한 기록이 있다. 당시에는 Parser의 기능과 출력 형식을 넓게 살펴보는 단계였다.

- 관련 글: [PJ Parser API 비교 분석]({% post_url projects/history-docent/2025-10-02-pj-parer-comparison %})

현재 corpus는 그 비교 글의 추천 순위를 그대로 구현한 결과가 아니다. 실제 Upstage Parser 출력에 맞춰 normalization과 provenance recovery를 다시 설계했다. 초기 조사는 후보를 이해하기 위한 자료이고, 현재 기준선은 후속 코드와 평가 보고서에 근거한다.

## 공개 저장소에는 원문을 넣지 않았다

원본 PDF, 전체 Parser JSON, 전체 chunk text, vector index, private evaluation payload는 공개 저장소에서 제외했다. 공개할 수 있는 것은 schema, 전처리 코드, aggregate metric, redacted sample, public-safe report다.

이 때문에 블로그에서도 원문 문장을 길게 인용하거나 private query와 evidence를 재현하지 않는다. 대신 어떤 schema를 보존했고, 어떤 지표로 후보를 비교했으며, 왜 한 후보를 채택하지 않았는지를 설명한다.

전처리 결과가 검색 품질을 보장하지는 않는다. Corpus에서 target chunk가 존재하는지, Retriever가 그 chunk를 찾는지, Generation이 근거를 빠뜨리지 않는지는 각각 별도 평가가 필요하다. 다음 글에서 청킹 기준선을 따로 비교한 이유다.

- 이전 글: [음성 관광 앱보다 RAG 백엔드를 먼저 만든 이유]({% post_url projects/history-docent/2026-09-09-project-history-docent-02-scope-architecture %})
- 다음 글: [청킹을 계속 바꾸기 전에 기준선을 고정한 이유]({% post_url projects/history-docent/2026-09-11-project-history-docent-04-chunking-baseline %})
