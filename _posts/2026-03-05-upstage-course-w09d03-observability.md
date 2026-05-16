---
title: "Observability"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- agentops
- course-note
- llmops
toc: true
date: 2026-03-05 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Observability

## 수업 위치

이 수업은 LLMOps 흐름에서 API 장애 대응과 상태/비용 관리 다음에 위치한다. 앞 수업에서 primary/fallback 모델, timeout, retry 같은 방어 장치를 만들었다면, 여기서는 “실제로 어떤 일이 벌어졌는지 어떻게 볼 것인가”를 다룬다.

에이전트 서비스는 일반 API 서버보다 원인 추적이 어렵다. 사용자의 질문 하나가 router node, RAG 검색, tool 실행, LLM 응답 생성으로 이어지고, 각 단계가 성공해도 최종 답변 품질은 낮을 수 있다. 그래서 단순히 서버가 200 응답을 반환했는지만 보면 운영 문제를 놓치게 된다.

## 핵심 개념

> **요약**
> 프로덕션 에이전트 서비스의 Observability를 구현한다. 로그, 메트릭, 트레이스를 통해 에이전트 실행 흐름을 추적하고, 문제 진단과 성능 개선에 필요한 근거를 남기는 방법을 학습한다.

## 주요 내용

### 1. Observability 개요
- Observability의 세 가지 축: Logs, Metrics, Traces
- 모니터링 vs Observability
- LLM 서비스 특화 관측 항목
- 사용자 요청 하나를 여러 node의 실행 흐름으로 추적하는 방식

### 2. 로깅 (Logging)
- 구조화된 로깅 (JSON 포맷)
- 로그 레벨 전략
- LLM 호출 로깅: 모델명, token usage, latency, fallback 여부
- 로그 수집 및 중앙화

### 3. 트레이싱 (Tracing)
- 분산 트레이싱 개념
- LangSmith / LangFuse 활용
- 에이전트 실행 경로 추적
- 노드별 지연 시간 분석
- router, rag, tool, response node를 하나의 request id로 묶기

### 4. 메트릭 (Metrics)
- 핵심 메트릭: 응답 시간, 토큰 사용량, 에러율
- 비용 메트릭 추적
- 사용자 만족도 지표
- 대시보드 구성

## 에이전트에서 봐야 하는 것

`day6-mission`의 LangGraph 흐름을 기준으로 보면 관측해야 할 지점이 명확해진다.

```text
START
  -> router
  -> rag 또는 tool 또는 response
  -> response
  -> END
```

이 구조에서 사용자에게 보이는 것은 최종 답변뿐이다. 하지만 운영자는 아래 데이터를 볼 수 있어야 한다.

| 단계 | 확인할 내용 |
|---|---|
| router | intent가 `chat`, `rag`, `tool` 중 무엇으로 분류되었는가 |
| rag | 어떤 query로 검색했고 문서가 몇 개 선택되었는가 |
| tool | 어떤 tool이 실행되었고 성공했는가, timeout 되었는가 |
| response | 어떤 모델이 사용되었고 fallback이 발생했는가 |
| 전체 요청 | latency, error type, token usage, 사용자 피드백 |

이 정보가 있어야 “에이전트가 이상하게 답했다”를 더 작은 원인으로 나눌 수 있다. 예를 들어 router가 tool 질문을 chat으로 분류했는지, RAG 검색이 빈 결과였는지, fallback 모델 사용 후 출력 형식이 달라졌는지 구분할 수 있다.

## 수업에서 남길 체크포인트

- 로그는 사람이 읽는 기록이 아니라 나중에 검색하고 집계할 수 있는 구조화 데이터여야 한다.
- 트레이스는 request id 기준으로 node 실행 순서를 복원할 수 있어야 한다.
- 메트릭은 평균값만 보면 부족하다. p95 latency, 실패율, fallback 비율, token 사용량을 함께 봐야 한다.
- raw prompt와 raw response를 그대로 저장하면 보안 문제가 생길 수 있으므로 마스킹 기준이 필요하다.
- Observability는 Evaluation과 연결되어야 한다. “어디서 실패했는지”와 “답변이 왜 나쁜지”를 함께 봐야 개선할 수 있다.

## 흐름도

![Observability 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-llmops-ops-w09d03-observability-diagram-1.svg)

## 관련 글

- [Observability 기술 노트]({% post_url 2026-03-05-upstage-tech-observability %})
- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %})
- [AgentOps]({% post_url 2026-02-03-upstage-tech-agentops %})
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
- [Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %})
