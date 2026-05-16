---
title: "Error Analysis"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- error-analysis
- evaluation
- course-note
- llmops
toc: true
date: 2026-03-06 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Error Analysis

## 수업 위치

이 수업은 Observability 다음 단계다. Observability를 통해 trace, latency, token usage, node 실행 흐름을 볼 수 있게 되면 다음 질문이 생긴다.

> 데이터는 쌓였는데, 이걸 어떻게 개선으로 연결할 것인가?

Error Analysis는 이 질문에 답하는 과정이다. 단순히 평가 도구를 붙이는 것이 아니라, 실제 trace를 보면서 어떤 문제가 반복되는지 찾고, 문제 유형을 분류하고, 그 분류를 평가 기준과 개선 작업으로 연결한다.

## 핵심 개념

> **요약**
> Error Analysis는 에이전트가 만든 잘못된 응답이나 불안정한 행동을 trace 단위로 살펴보고, 원인을 분류한 뒤, 평가 기준과 개선 작업으로 연결하는 과정이다.

LLM 서비스에서는 “서버 오류가 없다”와 “좋은 답변을 했다”가 다르다. HTTP 200으로 응답했어도 persona가 깨졌거나, RAG 근거를 잘못 썼거나, tool을 호출하지 않아야 할 때 호출했거나, 날짜 형식이 서비스 맥락과 맞지 않을 수 있다.

그래서 일반적인 평가 지표만으로는 부족하다. Correctness, Faithfulness, Relevance 같은 지표는 출발점이지만, Lumi 같은 에이전트에서는 다음 기준도 필요하다.

- 루미 페르소나가 유지되었는가?
- 스케줄 질문에서 tool을 제대로 사용했는가?
- RAG 질문에서 없는 정보를 만들어내지 않았는가?
- 사용자가 원하는 톤과 형식을 지켰는가?
- 오류가 났을 때 안전한 메시지로 안내했는가?

## 분석 흐름

Error Analysis는 다음 순서로 진행한다.

```text
서비스 사용 데이터 수집
  -> Langfuse trace 확인
  -> 이상한 응답 메모
  -> 오류 유형 분류
  -> 빈도와 심각도 확인
  -> 평가 기준 작성
  -> prompt / router / tool / RAG 개선
```

핵심은 사람이 먼저 trace를 본다는 점이다. 처음부터 LLM-as-a-Judge를 붙이면 그 평가가 맞는지 알 수 없다. 사람이 직접 일부 데이터를 보고, 어떤 오류가 중요한지 판단한 뒤, 그 기준을 자동화해야 한다.

## 기록할 항목

trace를 볼 때는 막연히 “이상함”이라고 적으면 개선으로 이어지지 않는다. 최소한 아래 항목을 남겨야 한다.

| 항목 | 예시 |
|---|---|
| 사용자 입력 | 사용자가 무엇을 물었는가 |
| 실행 경로 | chat, rag, tool 중 어디로 갔는가 |
| 문제 현상 | 답변이 어떤 점에서 이상했는가 |
| 추정 원인 | router, prompt, RAG, tool, 모델 중 어디가 원인인가 |
| 심각도 | 사용자 경험에 얼마나 큰 영향을 주는가 |
| 개선 액션 | prompt 수정, tool schema 수정, 평가 케이스 추가 등 |

이렇게 기록해야 나중에 피봇 테이블이나 스프레드시트로 오류 유형을 집계할 수 있다.

## LLM-as-a-Judge의 위치

LLM-as-a-Judge는 유용하지만, 처음부터 신뢰하면 안 된다. 먼저 사람이 평가한 결과를 ground truth처럼 두고, 같은 데이터를 LLM이 평가하게 한 뒤 일치율을 확인해야 한다.

일치율이 낮다면 모델이 나쁜 것이 아니라 평가 프롬프트가 애매하거나, 사람의 기준이 정리되지 않았거나, 애초에 분류 체계가 너무 넓을 수 있다. 따라서 Error Analysis는 한 번에 끝나는 작업이 아니라 반복 작업이다.

## 체크포인트

- 평가 도구를 붙이기 전에 무엇을 평가할지 먼저 정해야 한다.
- trace를 직접 보지 않으면 우리 서비스만의 오류 유형을 찾기 어렵다.
- 오류 분류는 구체적이어야 한다. “답변 이상함”보다 “스케줄 질문에서 tool 미호출”처럼 적어야 한다.
- LLM-as-a-Judge 결과는 사람 평가와 비교해 검증해야 한다.
- 날짜 형식 검사처럼 rule-based로 충분한 것은 LLM 평가보다 코드 검사가 더 안정적일 수 있다.

## 관련 글

- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
- [Agent Evaluation]({% post_url 2026-02-03-upstage-tech-agent-evaluation %})
- [AgentOps]({% post_url 2026-02-03-upstage-tech-agentops %})
- [LLMOps 패턴 정리]({% post_url 2026-03-09-upstage-course-w09d05-llmops-patterns %})
