---
title: "05. 검색 점수와 응답시간 사이에서 기본 경로 고르기"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- Dense Retrieval
- Hybrid Retrieval
- reranker
- Query Rewrite
toc: true
date: 2026-09-12 09:00:00 +0900
comments: true
mermaid: false
math: false
---

BGE-M3 Dense의 Recall@5는 0.80이었습니다. multilingual-e5-small의 0.733333보다 높았습니다. 점수만 보면 BGE-M3를 고르면 됐습니다.

그렇게 결정하지 않았습니다. 이 프로젝트의 검색기는 관광지에서 짧은 질문을 처리하는 API의 앞단에 놓입니다. 검색 품질뿐 아니라 상위 순위, 응답시간, no-answer 처리, 음성형 질문 보정까지 같이 봐야 했습니다.

## BM25 다음에 Dense를 다시 비교했다

청킹 ablation에서는 변수를 줄이기 위해 BM25를 사용했습니다. 청킹을 고정한 뒤 Dense 후보를 비교했습니다.

multilingual-e5-small은 dev 70개에서 Recall@5 0.733333, MRR 0.675556, nDCG@5 0.533797을 기록했습니다. BGE-M3의 Recall@5는 0.80으로 더 높았습니다. BGE-M3는 품질 상한 후보로 남겼지만 기본 검색기에는 E5-small을 사용했습니다.

모델 하나의 최대 점수보다 전체 파이프라인에서 반복 실행 가능한 비용과 지연을 우선한 결정입니다. 이 선택은 BGE-M3가 나쁜 모델이라는 판단이 아닙니다. 현재 장비와 API 조건에서 기본값으로 두지 않았다는 뜻입니다.

## Hybrid는 Recall을 올렸지만 전역 기본값이 되지 못했다

BM25와 E5-small을 weighted 방식으로 합친 후보는 Recall@5 0.783333을 기록했습니다. Dense 단독보다 높았습니다.

대신 MRR과 nDCG@5, latency가 나빠졌습니다. 더 많은 정답을 Top-5 안에 넣었지만 첫 번째와 두 번째 순위의 품질이 항상 좋아진 것은 아니었습니다. 관광 해설은 제한된 evidence만 생성 모델에 전달하기 때문에 상위 순위가 흔들리면 답변 입력도 바뀝니다.

전체 질문에 Hybrid를 적용하지 않고 relationship 질문 전용 route 후보로 남긴 이유입니다.

## Reranker의 품질은 좋았고 지연시간은 받아들이기 어려웠다

BGE Reranker top20은 Recall@5 0.833333, MRR 0.761667, nDCG@5 0.635787을 기록했습니다. 비교한 후보 중 검색 품질 상한은 가장 높았습니다.

CPU p95 지연시간은 13,140.6903ms였습니다. 검색 한 단계에서 13초가 걸리는 경로를 관광지용 실시간 API의 기본값으로 두기는 어려웠습니다.

이 실험은 실패가 아니었습니다. Reranker를 붙이면 어디까지 품질을 올릴 수 있는지 확인했고, 현재 환경에서 지연시간이 채택을 막는다는 사실도 확인했습니다. GPU 서빙이나 더 작은 Reranker를 쓸 수 있는 환경이라면 다시 비교할 후보로 남습니다.

## 모든 질문을 고쳐 쓰자 일부 순위가 나빠졌다

짧은 질문을 위해 장소 정보를 query에 넣는 rewrite를 만들었습니다. 전체 질문에 적용하면 평균 Recall@5는 올랐지만 place_fact와 place_story 일부에서 원래 query의 핵심어보다 장소 표현이 강해졌습니다. Target document는 찾더라도 target child의 순위가 내려가는 사례가 나왔습니다.

적용 범위를 voice_followup으로 줄였습니다. 장소명이나 주어가 생략된 음성형 질문에만 rewrite를 적용한 후보는 다음 결과를 기록했습니다.

| 지표 | E5-small 기본 | Voice rewrite |
| --- | ---: | ---: |
| Recall@5 | 0.733333 | 0.850000 |
| MRR | 0.675556 | 0.758056 |
| nDCG@5 | 0.533797 | 0.615293 |
| p95 latency | 15.6385ms | 19.5602ms |

약 4ms의 p95 증가로 세 검색 지표가 함께 올라갔습니다. 현재 non-rerank 기본 후보를 E5-small voice rewrite로 정한 근거입니다.

## 기본 경로는 가장 높은 단일 점수가 아니었다

최종 선택은 다음 순서로 정리됐습니다.

1. C0 parent-child corpus를 유지합니다.
2. E5-small을 기본 Dense 모델로 사용합니다.
3. voice_followup에만 제한적인 rewrite를 적용합니다.
4. Hybrid는 relationship shadow 후보로 남깁니다.
5. BGE-M3와 Reranker는 품질 상한 또는 후속 후보로 보류합니다.

이 결과는 dev set 선택 결과입니다. Production latency나 실제 관광객 질문 분포를 검증한 것은 아닙니다. 다음 단계에서는 더 복잡한 RAG 후보를 같은 기준선 위에서 비교했습니다.

- 이전 글: [청킹을 계속 바꾸기 전에 기준선을 고정한 이유]({% post_url projects/history-docent/2026-09-11-project-history-docent-04-chunking-baseline %})
- 다음 글: [GraphRAG와 HyDE를 기본 경로에서 뺀 이유]({% post_url projects/history-docent/2026-09-13-project-history-docent-06-advanced-rag-routing %})
