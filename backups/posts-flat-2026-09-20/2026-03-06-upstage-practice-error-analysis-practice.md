---
title: "Error Analysis 실습"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- PRACTICE
tags:
- upstage
- sesac
- ai-agent
- error-analysis
- evaluation
- practice
- llmops
toc: true
date: 2026-03-06 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Error Analysis 실습

> **실습 정보**
> - **주차**: Week 09, Day 04
> - **유형**: 분석 리포트 작성
> - **상태**: 보강 정리

## 실습 목표

Lumi 챗봇을 여러 케이스로 사용해 trace 데이터를 만들고, Langfuse 또는 export 데이터를 기준으로 오류 유형을 분류한다. 최종 목표는 “좋아 보인다” 수준의 감상이 아니라, 다음 수정 작업으로 이어질 수 있는 오류 분석 리포트를 작성하는 것이다.

## 실습 흐름

```text
대화 데이터 생성
  -> trace 확인
  -> 오류 메모
  -> 카테고리 분류
  -> 빈도 집계
  -> 개선 우선순위 결정
  -> 액션 플랜 작성
```

실습에서는 RAG 질문, tool 호출 질문, 일반 대화, 일부러 실패를 유도하는 질문을 섞어야 한다. 그래야 router, RAG, tool, response node의 문제를 고르게 볼 수 있다.

## 분석 기준

| 기준 | 확인할 내용 |
|---|---|
| Router | 의도 분류가 맞았는가 |
| RAG | 검색된 문서가 질문과 관련 있었는가 |
| Tool | 필요한 tool을 호출했는가, 잘못된 tool을 호출하지 않았는가 |
| Response | 루미 페르소나와 답변 형식이 유지되었는가 |
| Safety | 모르는 내용을 만들어내거나 민감 정보를 노출하지 않았는가 |

오류는 가능한 한 구체적으로 적는다. 예를 들어 “응답이 별로임”은 개선할 수 없지만, “스케줄 질문인데 tool을 호출하지 않고 일반 답변으로 처리함”은 router prompt나 tool schema 개선으로 이어질 수 있다.

## 산출물

- 테스트 대화 수
- RAG 질문 수
- Tool 호출 질문 수
- 오류 유형 목록
- 오류 유형별 빈도
- 우선 수정할 항목
- 다음 액션 플랜

가능하면 스프레드시트로 정리하고, 사람이 판단한 분류와 LLM-as-a-Judge 분류를 비교한다. 두 판단이 많이 다르면 자동 평가 기준을 바로 믿지 말고, 평가 프롬프트와 카테고리 정의를 다시 다듬는다.

## 체크포인트

- [ ] 충분한 수의 대화를 생성했다.
- [ ] RAG, Tool, 일반 대화 케이스를 나눠서 확인했다.
- [ ] 오류 메모가 구체적인 원인 후보를 포함한다.
- [ ] 카테고리별 빈도를 집계했다.
- [ ] 우선순위를 정하고 다음 수정 작업을 적었다.
- [ ] LLM-as-a-Judge 결과를 사람 판단과 비교했다.

## 관련 글

- [Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
- [Agent Evaluation]({% post_url 2026-02-03-upstage-tech-agent-evaluation %})
