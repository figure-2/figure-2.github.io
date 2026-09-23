---
title: "01. HistoryDocent: 관광지 질문에 출처가 남는 역사 답변 만들기"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- rag
- Citation
- retrieval
- fastapi
toc: true
date: 2026-09-08 09:00:00 +0900
comments: true
mermaid: false
math: false
---

처음에는 서울 관광지에서 질문하면 바로 역사 이야기를 들려주는 음성 도슨트를 만들려고 했습니다. 경복궁 앞에서 "여기는 왜 지었어?"라고 물으면 장소를 알아보고, 짧은 설명을 음성으로 돌려주는 서비스였습니다.

막상 만들기 시작하니 음성보다 먼저 해결해야 할 문제가 보였습니다. 생성된 설명이 그럴듯하더라도 어느 책의 어느 부분을 근거로 했는지 찾을 수 없다면 관광 해설에 쓰기 어려웠습니다. 그래서 첫 범위를 음성 앱이 아니라 citation을 복구할 수 있는 RAG 백엔드로 좁혔습니다.

HistoryDocent는 한국사 도서의 Parser 결과를 검색 가능한 corpus로 정리하고, 서울의 장소와 연결된 질문에 근거를 찾아 짧은 답변을 만드는 개인 프로젝트입니다.

## 현장에서 들어오는 질문은 검색어처럼 생기지 않았다

일반적인 문서 검색은 비교적 완성된 질문을 가정합니다. 관광지에서는 다릅니다.

"이 건물은 왜 여기 있어?", "아까 말한 왕은 누구야?", "여기서 무슨 일이 있었어?"처럼 짧은 질문이 많습니다. 장소명이 생략되기도 하고, 지시어가 앞선 대화를 가리키기도 합니다. 검색 결과가 맞아도 설명이 길면 현장에서 듣기 어렵습니다.

흐름은 장소 선택과 질문, place-aware rewrite, evidence retrieval, evidence packing, citation answer 순서로 나눴습니다. 마지막 답변은 화면용 설명과 짧은 음성용 설명을 함께 반환합니다.

장소 catalog가 현재 맥락을 제공하고, query rewrite가 생략된 장소명이나 음성형 표현을 보정합니다. Retriever는 도서에서 근거 후보를 찾고, 생성 단계는 검색 근거 안에서만 답하며 citation을 함께 반환합니다.

## 내가 맡은 범위

프로젝트에서는 다음 작업을 직접 설계하고 구현했습니다.

- Upstage Parser 결과를 공통 element schema로 정규화했습니다.
- page, section, block provenance를 보존하는 parent-child chunk를 만들었습니다.
- 서울과 한양의 장소 catalog를 정리했습니다.
- BM25, Dense, Hybrid, Reranker, Query Rewrite를 같은 dev set에서 비교했습니다.
- answer, spoken_answer, citation을 분리한 응답 계약을 만들었습니다.
- FastAPI 채팅 계약과 검색 연결 smoke를 구성했습니다.
- 실패 사례를 query type과 pipeline stage별로 나누고, 후보를 채택·보류·기각했습니다.

구현 기술을 나열하는 것보다 이 선택들이 왜 바뀌었는지를 남기는 데 더 많은 시간을 썼습니다. 점수가 높아도 응답시간이 길거나 citation recall이 내려가면 기본 경로에서 제외했습니다.

## 현재 기본선이 정해진 과정

제출용 기본선은 다음 조합입니다.

| 단계 | 현재 선택 | 판단 근거 |
| --- | --- | --- |
| 청킹 | C0 parent-child | BM25 고정 조건의 C0-C6 비교에서 gate와 검색 지표를 함께 충족 |
| 검색 | multilingual-e5-small + voice rewrite | dev 70에서 Recall@5 0.85, nDCG@5 0.615293 |
| Evidence packing | P0 rank order | citation recoverability 1.0 |
| 생성 | Solar Pro 3 generation v1 | v2 repaired는 citation recall이 하락해 기본값에서 제외 |
| API | FastAPI chat contract | contract와 retrieval-backed smoke까지 확인 |

BGE-M3 Dense는 Recall@5 0.80으로 품질 상한을 보여줬지만, 지연시간을 함께 고려해야 했습니다. BGE Reranker는 검색 지표가 좋았어도 CPU p95가 약 13.1초였습니다. GraphRAG-lite, RAPTOR-lite, HyDE도 구현 여부가 아니라 현재 기준선을 넘었는지로 판단했습니다.

결과적으로 사용한 기술보다 기본 경로에서 뺀 기술이 더 많았습니다.

## 이 프로젝트가 증명하지 않는 것

이 결과는 production 서비스 성능을 의미하지 않습니다. 주요 검색 비교는 dev 70개, advanced RAG 일부는 더 작은 subset, locked retrieval 비교는 35개 조건에서 수행했습니다. 실제 관광객을 대상으로 음성 품질이나 답변 만족도를 검증하지도 않았습니다.

말할 수 있는 범위는 명확합니다. citation 가능한 corpus와 평가 harness를 만들었고, 같은 기준으로 여러 검색·생성 후보를 비교했습니다. 그리고 작은 실험에서 좋아 보였던 후보를 확대 평가나 locked 결과 때문에 기각한 기록이 남아 있습니다.

## 시리즈에서 다룰 내용

이후 글에서는 서비스 범위, 데이터 구조, 청킹, 검색 실험, advanced RAG, 음성 데모와 회고를 나누어 정리합니다.

1. 이 글
2. [음성 관광 앱보다 RAG 백엔드를 먼저 만든 이유]({% post_url projects/history-docent/2026-09-09-project-history-docent-02-scope-architecture %})
3. [책 한 권을 citation 가능한 검색 데이터로 바꾸기]({% post_url projects/history-docent/2026-09-10-project-history-docent-03-parser-citation-corpus %})
4. [청킹을 계속 바꾸기 전에 기준선을 고정한 이유]({% post_url projects/history-docent/2026-09-11-project-history-docent-04-chunking-baseline %})
5. [검색 점수와 응답시간 사이에서 기본 경로 고르기]({% post_url projects/history-docent/2026-09-12-project-history-docent-05-retrieval-comparison %})
6. [GraphRAG와 HyDE를 기본 경로에서 뺀 이유]({% post_url projects/history-docent/2026-09-13-project-history-docent-06-advanced-rag-routing %})
7. [Citation과 음성 데모를 연결하며 정한 경계]({% post_url projects/history-docent/2026-09-14-project-history-docent-07-citation-voice-retrospective %})
