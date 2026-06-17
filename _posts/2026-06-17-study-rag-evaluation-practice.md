---
title: "RAG 평가 방법 정리: Top-K, G-Eval, Groundedness"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- rag
- evaluation
- top-k
- groundedness
toc: true
date: 2026-06-17 09:30:00 +0900
comments: false
mermaid: true
math: true
---

RAG 평가에서 가장 많이 헷갈린 부분은 “답변이 틀렸다”를 하나의 실패로 보는 습관이었다.

답변이 틀린 이유는 두 가지로 나뉜다. 검색이 근거를 못 찾았거나, 검색은 맞았는데 생성이 근거를 잘못 썼거나. 이 둘을 나누지 않으면 개선 방향도 흐려진다.

![RAG 평가 분리](/assets/images/study/diagrams/study-practice-rag-evaluation-split.svg)

## Retrieval 평가와 Generation 평가

RAG 평가는 최소한 두 층으로 나눠야 한다.

| 평가 | 질문 | 예 |
| --- | --- | --- |
| Retrieval 평가 | 정답 근거가 검색 결과 안에 들어왔는가 | Top-K Accuracy, Recall |
| Generation 평가 | 답변이 검색 근거를 잘 사용했는가 | G-Eval, Groundedness, Faithfulness |

검색이 실패했는데 생성 모델을 바꿔도 해결되지 않는다. 반대로 검색은 맞았는데 답변이 과장된다면 prompt, context 구성, judge 기준을 봐야 한다.

## Top-K를 보는 이유

Top-K 평가는 검색기가 정답 후보를 가져오는지 보는 지표다.

| 지표 | 의미 |
| --- | --- |
| Top-1 | 첫 번째 검색 결과가 맞는가 |
| Top-5 | 상위 5개 안에 근거가 있는가 |
| Top-10 | 상위 10개 안에 근거가 있는가 |
| Top-50 | 넓게 가져오면 근거를 찾을 수 있는가 |

실습 자료 중에는 Top-10과 Top-50을 함께 기록한 경우가 있었다. 이때 Top-50은 높은데 Top-10이 낮으면 검색 후보는 찾지만 순위가 좋지 않다는 뜻이다. 이 경우 reranker가 의미가 있다.

## G-Eval을 쓸 때의 기준

G-Eval은 LLM을 judge로 사용해 답변 품질을 평가하는 방식이다. 편하지만 기준을 잘못 잡으면 점수가 그럴듯한 숫자처럼 보일 뿐이다.

| 항목 | 확인할 것 |
| --- | --- |
| 평가 rubric | 정확성, 충실성, 완결성 등 기준이 분리됐는가 |
| 입력 근거 | judge가 원문 context를 같이 보는가 |
| 점수 해석 | 절대 점수인지, 조건 비교용 점수인지 구분했는가 |
| judge 일관성 | 같은 답변에 점수가 크게 흔들리지 않는가 |

G-Eval은 “좋은 답변인가”를 빠르게 비교하는 데 도움이 된다. 하지만 데이터셋과 rubric이 없으면 품질을 증명하는 숫자로 쓰기 어렵다.

## Groundedness

Groundedness는 답변이 근거에 붙어 있는지를 보는 관점이다.

예를 들어 검색 context에 없는 매출 전망을 답변이 말한다면 문장은 자연스러워도 grounded하지 않다.

| 상태 | 판단 |
| --- | --- |
| 근거 문장과 답변이 직접 연결됨 | 좋음 |
| 근거 일부를 과하게 일반화함 | 주의 |
| 근거에 없는 수치를 생성함 | 실패 |
| 최신 뉴스와 과거 리포트를 섞어 단정함 | 실패 |

내가 본 여러 실습 흐름에서 groundedness check는 특히 금융/리포트 문서에서 중요했다.

## 수치를 읽는 방식

학습 자료에는 다양한 수치가 있었다. 예를 들어 특정 조건에서 retrieval 점수가 높아지거나, reranker와 query rewrite를 붙인 뒤 G-Eval 점수가 개선된 사례가 있었다.

하지만 이런 수치는 항상 조건을 붙여 읽어야 한다.

| 수치 | 해석 기준 |
| --- | --- |
| Top-K 점수 | 어떤 평가셋, 어떤 정답 근거 기준인가 |
| G-Eval 점수 | judge 모델과 rubric이 무엇인가 |
| 개선율 | baseline이 무엇이고 변경점이 하나인가 |
| latency | 직렬 workflow인지 병렬 workflow인지 |

숫자는 글에 생동감을 주지만, 조건 없이 쓰면 과장 claim이 된다.

## 코드 조각으로 다시 보기

![RAG evaluation split code note](/assets/images/study/code-notes/study-code-rag-evaluation-split.png)

이 코드 조각에서 좋았던 점은 평가 함수 이름부터 분리되어 있다는 점이었다. `evaluate_rag`는 검색된 문서 안에 기준 근거가 있는지 보고, `evaluate_generation`은 생성 답변이 기준 답변과 얼마나 맞는지 본다. 하나의 점수로 합치기 전에 실패 위치를 먼저 나눈다.

내가 압축해서 가져간 평가 구조는 다음과 같다.

```python
@dataclass
class RagEvalResult:
    retrieval_hit: bool
    retrieval_score: float
    answer_score: float
    grounded: bool


def evaluate_rag_case(case: EvalCase) -> RagEvalResult:
    docs = retriever.search(case.question)
    answer = generator.answer(case.question, docs)

    return RagEvalResult(
        retrieval_hit=case.gold_source in [doc.source for doc in docs],
        retrieval_score=score_retrieval(case.gold_source, docs),
        answer_score=score_answer(case.reference_answer, answer),
        grounded=check_groundedness(answer, docs),
    )
```

이렇게 나누면 “답변이 틀렸다”를 바로 prompt 문제로 몰지 않아도 된다. 먼저 검색 결과 안에 정답 근거가 있었는지 확인하고, 그다음 생성 답변을 본다.

## 정리

RAG 평가는 하나의 “정확도”로 끝나지 않는다. 검색 평가와 생성 평가를 나누고, 그 사이에 reranker, groundedness, judge rubric을 둬야 한다.

내가 가져간 결론은 간단하다. RAG 답변이 틀렸을 때는 먼저 검색 실패인지 생성 실패인지부터 분리한다.
