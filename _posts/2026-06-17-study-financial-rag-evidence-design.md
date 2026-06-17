---
title: "금융 RAG 근거 설계 정리: 리포트, 뉴스, 표 데이터"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- rag
- financial-rag
- evidence
- retrieval
toc: true
date: 2026-06-17 09:20:00 +0900
comments: false
mermaid: true
math: true
---

금융 RAG를 보면서 가장 먼저 든 생각은 “근거의 종류가 너무 다르다”는 것이었다.

증권 리포트는 길고 정적이다. 뉴스는 짧고 최신성이 중요하다. 표 데이터는 숫자와 단위가 중요하고, 그래프는 추세를 담는다. 이 네 가지를 같은 chunk로 취급하면 답변은 자연스러워도 근거가 흐려진다.

![금융 RAG 근거 설계](/assets/images/study/diagrams/study-practice-financial-rag-evidence.png)

## 금융 RAG에서 먼저 나눌 것

금융 질의응답에서는 “무엇을 검색할 것인가”보다 “근거를 어떻게 분리할 것인가”가 먼저였다.

| 근거 | 특징 | 주의점 |
| --- | --- | --- |
| 리포트 본문 | 분석 관점과 장기 근거 | 발행일과 증권사 기준을 남겨야 한다 |
| 뉴스 | 최신 이벤트와 감성 | 과거 리포트와 시간 기준이 다르다 |
| 표 데이터 | 매출, 이익, 비율 등 수치 | 단위, 기간, 행/열 관계가 중요하다 |
| 그래프/차트 | 추세와 변동 | 이미지 설명과 주변 문맥이 필요하다 |

금융 RAG는 근거가 하나가 아니다. 그래서 retrieval 단계에서 source type을 metadata로 남겨야 한다.

## 리포트와 뉴스는 분리한다

리포트 기반 답변과 뉴스 기반 답변을 섞으면 시간 기준이 흔들릴 수 있다.

예를 들어 리포트는 3개월 전 데이터를 기준으로 기업을 평가했는데, 뉴스는 어제 발생한 이벤트를 말할 수 있다. 둘 다 맞는 정보지만 같은 근거처럼 쓰면 답변이 애매해진다.

| 질문 유형 | 우선 근거 |
| --- | --- |
| 기업의 장기 경쟁력 | 리포트 본문 |
| 최근 주가 변동 이유 | 뉴스 |
| 재무 수치 확인 | 표 데이터 |
| 추세 설명 | 그래프/차트 설명 |

내가 정리한 기준은 단순하다. 답변에 “리포트 기준”과 “최근 뉴스 기준”을 분리해서 말할 수 있어야 한다.

## 표 데이터는 RAG의 별도 입력이다

금융 문서에서 표는 부가 자료가 아니다. 핵심 수치가 표에 들어 있다.

표를 일반 텍스트처럼 chunking하면 다음 문제가 생긴다.

| 문제 | 예 |
| --- | --- |
| 헤더 손실 | 숫자가 매출인지 영업이익인지 알 수 없음 |
| 기간 손실 | 2023년 값인지 2024년 값인지 불명확 |
| 단위 손실 | 억 원, %, 배수 같은 단위 누락 |
| 행/열 관계 손실 | 셀 값만 남고 비교 구조가 사라짐 |

그래서 표는 `table_id`, `page`, `company`, `period`, `metric`, `unit` 같은 metadata를 붙여 저장하는 편이 낫다.

## Hallucination을 줄이는 방식

금융 RAG에서 hallucination을 줄인다는 말은 조심해야 한다. 완전히 제거한다고 말할 수는 없다. 대신 줄일 수 있는 방향은 있다.

| 방법 | 기대 효과 |
| --- | --- |
| 근거 source type 분리 | 리포트, 뉴스, 표 근거가 섞이는 문제 감소 |
| Retrieval / Generation 평가 분리 | 검색 실패와 생성 실패를 구분 |
| Groundedness check | 답변이 근거에 묶여 있는지 확인 |
| Query rewrite | 질문과 문서 표현 차이 완화 |
| Reranker | 관련 근거를 앞쪽으로 재정렬 |

중요한 것은 “답변이 그럴듯한가”보다 “근거 문서로 돌아갈 수 있는가”다.

## 실습 체크리스트

금융 RAG를 설계한다면 다음을 먼저 확인할 것이다.

| 체크 | 질문 |
| --- | --- |
| 근거 종류 | 리포트, 뉴스, 표, 그래프를 분리했는가 |
| 시간 기준 | 발행일과 최신성을 답변에 반영했는가 |
| 수치 근거 | 단위와 기간을 보존했는가 |
| 검색 평가 | 정답 근거가 Top-K 안에 들어오는가 |
| 생성 평가 | 답변이 근거 밖으로 나가지 않는가 |
| 사용자 표현 | 투자 조언처럼 단정하지 않는가 |

## 코드 조각으로 다시 보기

![Financial RAG evidence table code note](/assets/images/study/code-notes/study-code-financial-evidence-table.png)

이 코드 조각에서 중요하게 본 부분은 retrieval 결과를 바로 문자열로 합치지 않는다는 점이었다. `content`, `metadata`, `score`, `source`를 묶어두고, `category == "table"`인 경우에는 별도 경로로 처리한다. 금융 문서에서 표를 본문과 같은 방식으로 다루면 수치와 단위가 쉽게 깨진다.

내가 압축해서 가져간 구조는 다음과 같다.

```python
from dataclasses import dataclass


@dataclass
class Evidence:
    content: str
    source_type: str
    page: int | None
    published_at: str | None
    score: float


@dataclass
class GroundedAnswer:
    answer: str
    evidences: list[Evidence]
    missing_context: bool
```

금융 RAG에서는 답변 문장보다 `Evidence`가 먼저다. 답변이 맞아 보여도 어떤 리포트, 어떤 날짜, 어떤 표에서 온 근거인지 남아 있지 않으면 나중에 검증할 수 없다.

## 정리

금융 RAG는 일반 문서 RAG보다 근거 경계가 더 중요하다. 리포트, 뉴스, 표, 그래프가 서로 다른 시간성과 의미를 갖기 때문이다.

내가 가져간 결론은 이것이다. 금융 RAG의 품질은 좋은 답변 문장보다 근거를 분리하고 추적하는 설계에서 시작된다.
