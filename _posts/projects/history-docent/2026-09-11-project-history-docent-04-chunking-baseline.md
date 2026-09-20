---
title: "04. 청킹을 계속 바꾸기 전에 기준선을 고정한 이유"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- chunking
- evaluation
- Parent Child
- Citation
toc: true
date: 2026-09-11 09:00:00 +0900
comments: true
mermaid: false
math: false
---

검색 실패가 나오면 청킹부터 의심하기 쉽다. 나도 처음에는 child 크기나 overlap을 바꾸면 실패 사례가 줄어들 것으로 생각했다. 하지만 청킹을 바꿀 때마다 corpus와 검색 기준선이 함께 달라졌다. 이후 Dense, Hybrid, Reranker 결과도 같은 조건으로 비교하기 어려워졌다.

그래서 Retriever를 BM25로 고정하고 청킹 후보만 바꾸는 ablation을 먼저 진행했다.

## C0부터 C6까지 한 변수씩 비교했다

현재 기준선 C0는 heading1을 parent 경계로 사용하고, parent 내부 block을 합쳐 child를 만든다. 검색은 child에서 수행하고 필요한 문맥은 parent로 확장한다. Child 최대 길이는 1,100이고 한 block overlap을 둔다.

비교 후보는 child 크기, overlap, parent 병합, fixed-size 방식을 각각 바꿨다.

| 후보 | 변경한 조건 | Recall@5 | MRR | nDCG@5 | 판단 |
| --- | --- | ---: | ---: | ---: | --- |
| C0 | 현재 parent-child | 0.566667 | 0.471389 | 0.344203 | 유지 |
| C1 | child max 800 | 0.083333 | 0.044444 | 0.026033 | 기각 |
| C2 | child max 1,400 | 0.533333 | 0.446389 | 0.272112 | 기각 |
| C3 | 작은 parent 병합 | 0.533333 | 0.453333 | 0.330712 | 기각 |
| C4 | overlap 0 | 0.483333 | 0.384722 | 0.241390 | gate 실패 |
| C5 | overlap 2 | 0.533333 | 0.368611 | 0.247787 | 기각 |
| C6 | fixed-size block | 0.316667 | 0.254167 | 0.145937 | gate 실패 |

이 결과가 parent-child 방식이 모든 문서에서 가장 좋다는 뜻은 아니다. dev 70개, BM25 고정, 현재 Parser 출력이라는 조건에서 C0가 가장 방어 가능한 기준선이라는 뜻이다.

## 작은 chunk는 예상보다 크게 무너졌다

C1은 child를 더 작게 만들어 검색어와 직접 겹치는 부분을 선명하게 하려는 후보였다. 결과는 Recall@5 0.083333이었다. 이야기를 설명하는 문장이 여러 block에 걸쳐 있어, child가 작아질수록 target grain을 온전히 담지 못했다.

반대로 C2처럼 크게 만든다고 순위가 좋아지지도 않았다. Recall@5는 C0에 가까웠지만 nDCG@5가 0.272112로 내려갔다. 관련 내용이 포함돼도 상위 순위에서 구별하기 어려워진 것으로 해석했다.

Overlap을 없앤 C4와 fixed-size C6은 selection gate 자체를 통과하지 못했다. Citation provenance와 구조 보존까지 고려하면 검색 수치만으로 다시 채택할 이유가 없었다.

## 실패 한 건을 보고 전체 corpus를 다시 만들지 않았다

Failure analysis에서 place_story 질문 한 건은 target document를 찾았지만 필요한 child와 parent grain이 검색 결과에 나타나지 않았다. 처음에는 chunk boundary 문제로 분류했다.

여기서 바로 전역 청킹을 바꾸지 않았다. 해당 사례의 normalized block, child, parent artifact를 따로 추적했다. Target child와 parent는 corpus에 존재했다. 경계에서 원문이 사라진 것이 아니라 Retriever가 그 단위를 상위에 올리지 못한 문제였다.

이 확인으로 실패 분류가 달라졌다. 청킹 재설계보다 query rewrite와 retrieval 후보 비교가 먼저였다.

## Semantic Chunking을 후순위로 둔 이유

초기 Notion 기록에서는 문장 분리와 semantic 연결을 조합한 청킹을 구상했다. 현재 저장소에서는 그 방식을 기본선으로 사용하지 않는다.

Semantic boundary는 의미가 비슷한 문장을 묶는 데 도움이 될 수 있다. 그러나 이 프로젝트는 normalized block과 page span을 citation으로 되찾아야 한다. Embedding 기준으로 경계를 다시 만들면 provenance recovery가 복잡해지고, 이미 누적한 검색·생성 실험도 새 corpus에서 다시 수행해야 한다.

따라서 semantic chunking은 최신 기법이라서 여는 실험이 아니다. 실패 분석에서 같은 boundary 손실이 반복될 때만 hard subset을 대상으로 비교할 후보로 남겼다.

## 청킹을 다시 여는 조건

현재 기준은 다음과 같다.

- 실패 사례 가운데 같은 chunk boundary 손실이 세 건 이상 반복된다.
- 선택된 evidence에서 source block이나 page citation을 복구하지 못한다.
- 특정 query type에서 같은 target grain 누락이 반복된다.
- Parser 버전이나 도서 범위가 달라져 corpus 자체가 변한다.

조건이 생겨도 전체 교체보다 sentence-window metadata나 query-type별 context expansion부터 시험할 계획이다. 기준선을 고정한 덕분에 다음 검색 실험은 같은 corpus에서 비교할 수 있었다.

- 이전 글: [책 한 권을 citation 가능한 검색 데이터로 바꾸기]({% post_url projects/history-docent/2026-09-10-project-history-docent-03-parser-citation-corpus %})
- 다음 글: [검색 점수와 응답시간 사이에서 기본 경로 고르기]({% post_url projects/history-docent/2026-09-12-project-history-docent-05-retrieval-comparison %})
