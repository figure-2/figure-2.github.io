---
title: "06. GraphRAG와 HyDE를 기본 경로에서 뺀 이유"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- graphrag
- RAPTOR
- HyDE
- routing
- evaluation
toc: true
date: 2026-09-13 09:00:00 +0900
comments: true
mermaid: false
math: false
---

HyDE를 5개 질문에 적용했을 때 Recall@5가 0.25 올랐다. 이 결과만 남겼다면 "HyDE로 검색 성능을 개선했다"고 쓸 수 있었다.

평가 범위를 40개로 늘리자 결과가 달라졌다. Recall@5는 0.033333 올랐지만 MRR은 0.035, nDCG@5는 0.018384 내려갔다. 생성 호출 때문에 지연시간도 늘었다. 작은 subset에서 좋아 보였던 후보를 기본 경로에서 뺀 이유다.

## 질문 유형마다 다른 검색기를 쓰려 했다

HistoryDocent 질문은 place_fact, place_story, relationship, overview, route_context, voice_followup, no_answer로 나뉜다. 한 검색기가 모든 유형에서 가장 좋지 않았다.

Relationship 질문은 두 인물이나 사건의 연결을 찾아야 해서 Hybrid나 Graph 구조가 유리할 가능성이 있었다. Overview 질문은 여러 section에 흩어진 정보를 모아야 했다. 짧은 voice_followup에는 query rewrite가 더 직접적이었다.

이 차이를 처리하기 위해 deterministic query type classifier와 router skeleton을 만들었다. Classifier는 dev 70개에서 macro F1 0.956818을 기록했다. 하지만 분류기의 평균 점수와 잘못된 route가 만드는 비용은 다른 문제였다.

## GraphRAG-lite는 relationship 질문에서 기준선을 넘지 못했다

GraphRAG-lite는 entity와 relationship을 가볍게 연결한 input-only 후보로 만들었다. 모든 질문에 적용하지 않고 relationship dev 10개에 한정했다.

Hybrid reference와 비교한 nDCG@5 delta는 -0.002056이었다. 개선 폭이 없었고 entity extraction과 canonicalization 오류가 새로운 실패 지점을 만들 수 있었다. 기본 검색 경로로 승격하지 않았다.

여기서 말할 수 있는 것은 "GraphRAG를 적용해 성능을 높였다"가 아니다. Relationship 질문에 필요한 연결 정보를 별도 구조로 만들어 비교했고, 현재 평가에서는 기존 Hybrid보다 나은 근거가 없었다는 사실이다.

## RAPTOR-lite도 overview 문제를 해결하지 못했다

RAPTOR-lite는 summary node를 만들어 overview와 place_story 질문에 보조 context를 주는 후보였다. Dev 20개에서 nDCG@5 delta는 -0.029969였다.

요약 node가 넓은 맥락을 담아도 target evidence의 상위 순위를 올려주지는 않았다. Citation 가능한 원문 grain과 summary node 사이를 다시 연결해야 하는 비용도 있었다. 이 후보 역시 기본값에서 제외했다.

## HyDE는 확대 평가에서 방향이 바뀌었다

HyDE는 질문에 대한 가상 답변을 먼저 만들고, 그 답변으로 문서를 검색한다. 짧거나 표현이 다른 질문을 문서 문체에 가깝게 바꿀 수 있다는 점을 기대했다.

첫 live-dev-subset은 5개였고, no-answer 한 건은 생성과 검색을 모두 차단했다. 나머지 4건에서 Recall@5 delta가 0.25였다. 다만 MRR 하락과 생성 지연이 같이 나타났다.

이 결과를 확인하기 위해 질문을 40개로 확대했다. Solar 호출은 no-answer 10개를 제외한 30회로 고정했다. 확대 결과에서는 Recall의 작은 상승보다 MRR, nDCG, latency 악화가 더 컸다. HyDE는 기본 retrieval route에서 기각했다.

## Shadow에서 좋아도 locked 결과가 우선했다

Relationship Hybrid route는 dev shadow 평가에서 전체 MRR delta 0.013888, relationship Recall@5 delta 0.20을 기록했다. Guard를 적용한 뒤 false hybrid route도 0건이었다.

API에는 active_route_mode를 shadow로 넣어 후보 경로와 판단을 관찰할 수 있게 했다. 실제 검색 경로는 바꾸지 않았다.

그 뒤 locked test 35개로 paired comparison을 실행했다. Relationship subset 5개에서 MRR delta는 -0.10, nDCG@5 delta는 -0.073814였다. 95% bootstrap confidence interval도 개선을 뒷받침하지 않았다.

Dev 결과보다 locked 결과를 우선해 active route 개선 주장을 보류했다. Router 코드는 남아 있지만 production route 적용 완료라고 쓰지 않는 이유다.

## 더 복잡한 기술이 더 좋은 기본값은 아니었다

Advanced RAG 실험은 다음 결정으로 끝났다.

| 후보 | 최종 상태 |
| --- | --- |
| GraphRAG-lite | relationship 기본값 기각 |
| RAPTOR-lite | overview·place_story 기본값 기각 |
| HyDE | 40개 확대 비교 후 기본값 기각 |
| Relationship Hybrid | shadow 후보 유지, active 적용 보류 |
| Query type router | contract와 관찰 field 유지 |

구현한 기술 수는 포트폴리오의 결과가 아니었다. 같은 평가 계약을 적용했을 때 기존 기준선을 넘었는지, 그리고 그 결과가 확대 평가와 locked split에서도 유지됐는지가 결정 기준이었다.

- 이전 글: [검색 점수와 응답시간 사이에서 기본 경로 고르기]({% post_url projects/history-docent/2026-09-12-project-history-docent-05-retrieval-comparison %})
- 다음 글: [Citation과 음성 데모를 연결하며 정한 경계]({% post_url projects/history-docent/2026-09-14-project-history-docent-07-citation-voice-retrospective %})
