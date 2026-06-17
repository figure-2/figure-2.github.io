---
title: "LLM 서비스 평가 실습 정리: ROUGE, BERTScore, G-Eval"
categories:
- 3.STUDY
- 3-7.AI_ENGINEERING
tags:
- study
- ai-engineering
- llm-evaluation
- rouge
- bertscore
- g-eval
toc: true
date: 2026-06-17 09:40:00 +0900
comments: false
mermaid: true
math: true
---

LLM 서비스 평가를 정리하면서 가장 먼저 든 생각은 “평가 지표가 많다”가 아니었다. 같은 출력도 무엇을 보려는지에 따라 지표가 달라진다는 점이 더 중요했다.

요약 품질을 볼 때와 업무 리포트 품질을 볼 때, RAG 답변의 근거성을 볼 때는 평가 질문이 다르다.

![LLM 서비스 평가 맵](/assets/images/study/diagrams/study-practice-llm-evaluation-map.svg)

## ROUGE

ROUGE는 생성 결과와 기준 답변의 단어 overlap을 본다.

| 장점 | 한계 |
| --- | --- |
| 계산이 빠르다 | 의미가 같아도 표현이 다르면 낮게 나올 수 있다 |
| 요약 baseline 비교에 좋다 | 사실성 검증은 어렵다 |
| 정답 요약문이 있을 때 쓰기 쉽다 | 좋은 글인지까지 보진 못한다 |

메일 요약이나 문서 요약처럼 기준 요약문이 있는 경우에는 빠른 비교 지표로 쓸 수 있다. 하지만 ROUGE가 높다고 답변이 항상 좋은 것은 아니다.

## BERTScore

BERTScore는 embedding 기반으로 의미 유사도를 본다.

| 장점 | 한계 |
| --- | --- |
| 표현이 달라도 의미가 비슷하면 반영된다 | 근거 기반 사실성은 따로 봐야 한다 |
| 요약의 semantic similarity를 보기 좋다 | 도메인 특화 용어에는 모델 영향이 있다 |
| ROUGE보다 유연하다 | 해석이 직관적이지 않을 수 있다 |

내가 정리한 기준은 이렇다. ROUGE는 표면 비교, BERTScore는 의미 비교에 가깝다.

## G-Eval

G-Eval은 LLM을 평가자로 쓰는 방식이다. 요약, 리포트, QA처럼 기준이 복합적인 결과를 볼 때 유용하다.

| 평가 축 | 예 |
| --- | --- |
| 정확성 | 중요한 내용이 맞는가 |
| 충실성 | 근거 밖의 내용을 만들지 않는가 |
| 완결성 | 사용자가 이해할 만큼 충분한가 |
| 형식 준수 | 요구한 포맷을 지켰는가 |
| 실행 가능성 | 리포트나 액션 아이템으로 쓸 수 있는가 |

G-Eval은 사람이 채점하듯 rubric을 만들 수 있다는 점이 장점이다. 대신 judge 모델이 항상 공정하다고 믿으면 안 된다.

## 지표를 함께 읽기

실습 자료 중에는 개별 요약은 ROUGE와 BERTScore로 보고, 최종 리포트는 G-Eval로 비교한 흐름이 있었다. 이 방식이 꽤 설득력 있었다.

출력의 성격이 다르기 때문이다.

| 출력 | 적합한 평가 |
| --- | --- |
| 짧은 요약 | ROUGE, BERTScore |
| 긴 리포트 | G-Eval, rubric 기반 평가 |
| RAG 답변 | Groundedness, Faithfulness, G-Eval |
| Agent 실행 결과 | tool call 성공 여부, task completion |

지표는 하나만 고르는 것이 아니라 출력 유형에 맞게 조합해야 한다.

## 평가 수치를 쓸 때의 주의

평가 수치는 글에 넣기 좋지만, 조심해서 써야 한다.

| 체크 | 이유 |
| --- | --- |
| 평가셋 크기 | 작은 데이터셋은 우연 영향이 크다 |
| baseline | 무엇과 비교했는지 없으면 의미가 약하다 |
| judge 기준 | rubric이 바뀌면 점수도 바뀐다 |
| 실패 사례 | 평균 점수만 보면 위험한 실패가 가려질 수 있다 |

예를 들어 어떤 조건에서 G-Eval 점수가 가장 높았다는 기록이 있어도, 그것은 해당 데이터와 rubric 안에서의 결과다. 일반적인 서비스 품질로 바로 확장하면 안 된다.

## 코드 조각으로 다시 보기

![LLM evaluation metrics code note](/assets/images/study/code-notes/study-code-llm-evaluation-metrics.png)

이 코드 조각에서 배운 점은 평가 지표를 함수 안에 고정하지 않고 config에서 선택한다는 것이다. 같은 요약 결과라도 ROUGE, BERTScore, G-Eval은 보는 관점이 다르다. 그래서 평가 코드는 “하나의 점수 계산기”보다 “여러 평가 질문을 실행하는 runner”에 가깝다.

내가 압축해서 가져간 형태는 다음과 같다.

```python
def run_evaluation(outputs: list[str], references: list[str], metrics: list[str]):
    results = {}

    if "rouge" in metrics:
        results["surface_overlap"] = rouge_score(outputs, references)

    if "bertscore" in metrics:
        results["semantic_similarity"] = bert_score(outputs, references)

    if "g_eval" in metrics:
        results["rubric_judgement"] = g_eval(
            outputs=outputs,
            references=references,
            rubric=["accuracy", "coverage", "faithfulness"],
        )

    return results
```

평가 지표를 이렇게 나누면 결과 해석도 같이 달라진다. ROUGE가 낮아도 의미가 맞을 수 있고, G-Eval이 높아도 judge 기준이 느슨하면 위험하다.

## 정리

LLM 서비스 평가는 “정답률” 하나로 설명하기 어렵다. 요약은 ROUGE/BERTScore, 리포트는 G-Eval, RAG는 groundedness를 함께 봐야 한다.

내가 가져간 결론은 이것이다. 지표를 고르기 전에 먼저 평가 질문을 정해야 한다. “무엇을 좋은 출력이라고 볼 것인가”가 먼저다.
