---
title: "03. 보험 용어 검색에서는 어떤 토크나이저가 유리했나"
categories:
- 2.PROJECT
- 2-8. Insurance_PF
tags:
- Insurance PF
- BM25
- Mecab
- Dense Retrieval
- Hybrid Retrieval
toc: true
date: 2026-09-18 09:00:00 +0900
comments: true
mermaid: false
math: false
---

처음에는 Kiwi가 보험 용어를 가장 잘 처리할 것으로 예상했습니다. 긴 복합명사를 보존하고 사용자 사전을 붙이기 쉬웠기 때문입니다. 30개 FAQ를 검색해보니 Mecab이 더 높은 순위에 정답을 올렸고 인덱싱도 빨랐습니다.

예상과 결과가 달랐던 첫 실험이었습니다.

## 세 토크나이저는 서로 다른 표현에서 강했다

Kiwi, Mecab, Okt를 법률 용어, 복합명사, 숫자, 구어체 문장으로 비교했습니다.

Kiwi는 "자동차손해배상보장법"을 하나의 단위로 유지했다. Mecab은 "무보험자동차상해담보"를 비교적 자연스럽게 나눴다. Okt는 "20만원"과 "어떡해요" 같은 금액·구어체 표현을 잘 보존했다.

약점도 분명했다. Mecab은 0.03 같은 숫자를 여러 token으로 나눴고, Okt는 "기명피보험자"를 부자연스럽게 분리했다. Kiwi는 "비상급유서비스"를 잘못 쪼개는 사례가 있었다.

형태소 분석 예시만 보고 결정하지 않고 동일한 BM25 평가로 이어갔다.

## 30개 FAQ에서는 Mecab BM25가 가장 앞섰다

첫 비교 결과는 다음과 같다.

| 토크나이저 | Recall@5 | MRR@5 | Index time |
| --- | ---: | ---: | ---: |
| Mecab | 1.0000 | 0.9833 | 19.29초 |
| Kiwi | 1.0000 | 0.9667 | 122.95초 |
| Okt | 1.0000 | 0.9389 | 120.47초 |

세 후보 모두 30개 정답을 Top-5 안에 넣었다. 차이는 순위와 인덱싱 시간에서 났다. Mecab은 MRR@5가 가장 높았고 Kiwi보다 약 6배 빠르게 인덱스를 만들었다.

이 결과로 BM25 토크나이저를 Mecab으로 정했다. 모든 한국어 보험 문장에서 Mecab이 낫다는 결론은 아니다. 이 30개 FAQ와 현재 전처리 corpus에 대한 선택이다.

## Dense와 합쳤지만 Sparse 단독 순위가 더 좋았다

같은 30개 질문에서 Dense, Sparse, Hybrid를 비교했다.

| 검색 방식 | Recall@5 | MRR@5 | 평균 시간 |
| --- | ---: | ---: | ---: |
| Dense | 0.7000 | 0.5483 | 0.0474초 |
| Sparse | 1.0000 | 0.9833 | 0.0246초 |
| Hybrid | 1.0000 | 0.9444 | 0.0431초 |

Hybrid는 Dense의 누락을 보완했지만 Sparse의 MRR을 넘지 못했다. 약관 제목과 전문 용어가 query에 직접 등장하는 평가셋 특성상 정확한 token matching이 강했다.

50개 비교에서도 Dense는 Recall@5 0.96, MRR@5 0.82로 올라갔지만 Sparse의 1.0과 0.97보다 낮았다. Hybrid MRR@5는 0.9667이었다.

## 로그 사이의 50개 결과는 그대로 합치지 않았다

별도 failure analysis에는 50개 가운데 48개 성공, 2개 Top-5 실패로 기록돼 있다. 반면 후속 retrieval 로그의 50개 표는 Sparse Recall@5 1.0으로 적혀 있다.

두 결과는 성공 판정 방식이나 평가 대상 artifact가 달랐을 가능성이 있지만, 현재 문서만으로 차이를 확정할 수 없다. 그래서 "50개에서 100% 정확했다"고 쓰지 않는다. 30개 공통 비교 결과와 50개 후속 비교 결과, failure report를 각각의 기록으로 남긴다.

이 불일치는 평가 코드가 있어도 dataset version, target ID, 판정 함수를 함께 고정해야 한다는 교훈이 됐다.

## BM25가 놓친 질문은 표현이 달랐다

Failure report의 두 사례는 절차형 질문이었다. "가지급금 받을 수 있나요?"와 "주소 변경하려면 어떻게?"라는 표현이 정답 문서의 핵심어와 충분히 겹치지 않았다.

최종 RAG 평가에서는 문제가 더 선명해졌다.

- 사용자는 "노트북 액정 파손"이라고 물었지만 문서는 "휴대품 손해"로 표현했다.
- "아내 차에 치임"은 배우자 면책 조항과 직접적인 단어가 겹치지 않았다.
- "전업주부 휴업손해"는 가사종사자 관련 조항을 찾지 못했다.

Sparse가 초기 평가에서 가장 좋았다는 사실과 semantic mismatch에 약하다는 사실은 동시에 성립한다. 더 큰 paraphrase 평가셋을 만들고 Hybrid나 Reranker를 다시 비교해야 production 후보를 논할 수 있다.

- 이전 글: [보험 약관을 일정한 글자 수로 자르면 안 되는 이유]({% post_url projects/insurance-pf/2026-09-17-project-insurance-pf-02-hierarchical-chunking %})
- 다음 글: [파인튜닝은 말투를 바꿨지만 사실을 보장하지 않았다]({% post_url projects/insurance-pf/2026-09-19-project-insurance-pf-04-finetuning-rag %})
