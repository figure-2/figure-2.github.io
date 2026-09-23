---
title: "02. 음성 관광 앱보다 RAG 백엔드를 먼저 만든 이유"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- architecture
- Product Scope
- fastapi
- Voice Interface
toc: true
date: 2026-09-09 09:00:00 +0900
comments: true
mermaid: false
math: false
---

초기 기록에는 한국사 도서 7권과 YouTube 자막을 함께 쓰고, BGE 계열 임베딩과 Hybrid Retrieval, 도메인 Reranker, Gemini 생성 모델까지 연결하는 구상이 남아 있습니다. 당시에는 파이프라인에 많은 기능을 넣는 것이 완성도라고 생각했습니다.

현재 저장소의 방향은 다릅니다. 음성 관광 앱 전체를 한 번에 만들지 않고, 답변의 근거를 추적할 수 있는 text-first RAG 계약부터 고정했습니다. 초기 기록은 출발점이고, 현재 코드는 그 뒤에 범위를 다시 정한 결과입니다.

## 제품보다 검증 단위를 먼저 줄였다

관광 도슨트 서비스 전체를 만들려면 위치, 지도, 다국어, STT, TTS, 모바일 UI가 필요합니다. 이 기능들을 동시에 다루면 검색이 틀렸는지, 생성이 근거를 놓쳤는지, 음성 인식이 질문을 바꿨는지 구분하기 어렵습니다.

첫 MVP에서는 다음 기능만 포함했습니다.

| 포함한 범위 | 뒤로 미룬 범위 |
| --- | --- |
| 장소 catalog | 실시간 위치 기반 경로 추천 |
| 장소를 반영한 query rewrite | 완성형 모바일 앱 |
| citation RAG | 다국어 TTS |
| 짧은 관광 해설 답변 | 사용자 계정과 결제 |
| retrieval·generation 평가 harness | AR 안내 |

이렇게 범위를 줄이니 실패 지점을 단계별로 볼 수 있었습니다. 장소가 빠졌다면 rewrite를 확인하고, 정답 문서가 후보에 없다면 retrieval을 확인합니다. 근거가 있는데 답변이 틀렸다면 generation contract 문제입니다.

## Text-first 계약이 중심이 됐다

현재 흐름은 사용자 질문, 장소와 대화 맥락 결합, query rewrite, query type 분류, retrieval, evidence packing, citation generation 순서입니다. 응답에는 answer, spoken_answer, citations가 구분되어 들어갑니다.

화면용 answer와 음성용 spoken_answer를 같은 문자열로 쓰지 않았습니다. 화면에서는 citation과 세부 설명을 볼 수 있지만, 음성 응답은 길어지면 듣기 어렵습니다. 대신 두 답변이 같은 evidence를 사용하도록 계약을 분리했습니다.

no-answer도 별도 query type으로 뒀습니다. 검색기는 관련성이 낮아도 후보를 반환할 수 있습니다. 후보가 있다는 이유만으로 답을 생성하면 근거가 없는 질문에서도 그럴듯한 설명이 나옵니다. 검색 후보와 답변 가능 여부를 같은 판단으로 다루지 않은 이유입니다.

## Router는 만들었지만 실제 검색 경로는 바꾸지 않았다

Query type classifier는 dev 70개에서 macro F1 0.956818을 기록했습니다. relationship 질문에는 Hybrid, no-answer에는 abstain-first, 나머지에는 Dense voice rewrite를 연결하는 router skeleton도 만들었습니다.

하지만 분류 점수가 높다는 이유만으로 active routing을 켜지 않았습니다. 초기 failure analysis에서 false hybrid route가 2건 발견됐습니다. Guard를 추가해 dev 조건에서는 0건으로 줄였지만, locked relationship subset에서 Hybrid의 MRR과 nDCG@5가 하락했습니다.

그래서 API에는 분류 결과와 route 후보를 관찰할 수 있는 field만 남겼습니다. 실제 검색 경로를 바꾸는 active route 적용 건수는 0으로 유지했습니다. 구현 완료와 운영 적용을 구분하기 위한 선택이었습니다.

## 음성은 Adapter로 뒤에 붙였다

음성 기능은 RAG 계약을 바꾸지 않고 바깥에서 감싸도록 구성했습니다.

> audio input → local STT → existing chat contract → spoken_answer → local TTS

이 구조에서는 STT나 TTS 후보를 바꿔도 검색·생성 평가를 다시 정의할 필요가 없습니다. 반대로 RAG 검색 경로를 바꿔도 음성 UI가 기대하는 응답 형식은 유지됩니다.

현재 음성 범위는 local demo 후보와 API route smoke입니다. 마이크 입력, 자동 스피커 재생, 실제 관광 환경의 소음 평가, production provider 선정은 완료 범위가 아닙니다.

## 범위를 줄인 뒤 프로젝트 설명이 쉬워졌다

초기 구상은 기능이 많았지만 각 기능이 어느 정도 검증됐는지 설명하기 어려웠습니다. text-first 계약으로 바꾼 뒤에는 단계마다 질문이 하나씩 생겼습니다.

- 이 질문은 어느 장소와 연결되는가.
- 정답 근거가 Top-K 안에 들어오는가.
- 선택한 근거를 citation으로 복구할 수 있는가.
- 근거가 없을 때 답변을 멈추는가.
- 음성 계층이 기존 계약을 훼손하지 않는가.

이 질문에 답할 수 있는 범위까지만 구현 완료로 기록했습니다.

- 이전 글: [HistoryDocent 프로젝트 개요]({% post_url projects/history-docent/2026-09-08-project-history-docent-01-case-study %})
- 다음 글: [책 한 권을 citation 가능한 검색 데이터로 바꾸기]({% post_url projects/history-docent/2026-09-10-project-history-docent-03-parser-citation-corpus %})
