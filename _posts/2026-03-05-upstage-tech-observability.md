---
title: "Observability (관측 가능성)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- agentops
- tech-note
- llmops
toc: true
date: 2026-03-05 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Observability (관측 가능성)

> **한줄 정의**
> 시스템의 내부 상태를 로그, 메트릭, 트레이스로 추적해 “무슨 일이 왜 일어났는지” 설명할 수 있게 만드는 운영 능력.

## 핵심 이해

Observability는 단순히 로그를 많이 남기는 것이 아니다. 장애가 났을 때 원인을 좁히고, 성능이 느려졌을 때 병목을 찾고, LLM 답변 품질이 흔들릴 때 어떤 단계가 문제였는지 추적할 수 있어야 한다.

LLM 에이전트에서는 이 중요도가 더 커진다. 사용자 질문 하나가 router, RAG 검색, tool call, 모델 응답 생성, 후처리 node를 거칠 수 있기 때문이다. 겉으로는 “답변이 이상하다”로 보이지만 실제 원인은 router 오분류, 검색 문서 부족, tool timeout, prompt 누락, fallback 모델 사용 중 하나일 수 있다.

Observability는 이 흐름을 하나의 trace로 묶어 본다.

```text
user request
  -> router node
  -> rag/tool/response node
  -> llm call
  -> final response
```

각 단계의 입력, 출력, latency, error, token usage, selected document, selected tool을 기록해야 문제를 재현하고 개선할 수 있다.

## 언제 쓰는지

Observability는 다음 질문에 답해야 할 때 필요하다.

- 왜 응답이 느렸는가?
- 어떤 node에서 실패했는가?
- router가 사용자의 의도를 올바르게 분류했는가?
- RAG가 실제로 도움이 되는 문서를 가져왔는가?
- tool call이 성공했는가, timeout 되었는가?
- fallback 모델이 사용되었는가?
- 비용 증가는 특정 사용자, 특정 기능, 특정 모델 호출에서 발생했는가?

일반 웹 서비스에서는 HTTP status, latency, DB query 정도로도 많은 문제를 찾을 수 있다. 하지만 LLM 서비스는 “정상 응답인데 품질이 낮은 경우”가 있다. 그래서 성공/실패만으로는 부족하고, 중간 추론 경로와 입력 데이터를 함께 봐야 한다.

## 구현 관점

- 로그는 사건의 기록, 메트릭은 수치 변화, 트레이스는 요청의 흐름을 담당한다.
- LLM 호출에는 모델명, prompt 버전, token usage, latency, finish reason을 남긴다.
- RAG 파이프라인은 검색 query, 검색된 문서, score, reranking 결과를 함께 기록한다.
- 배포 이후에는 오류율, 평균 지연시간, 비용, 사용자 피드백을 지속적으로 확인한다.

`day6-mission` 구조에 적용하면 다음 항목이 최소 관측 대상이 된다.

| 위치 | 남길 데이터 |
|---|---|
| router node | 분류된 intent, 선택된 tool, 분류 실패 여부 |
| rag node | 검색 query, 선택 문서 수, score, 빈 검색 결과 여부 |
| tool node | tool 이름, 실행 시간, 성공 여부, 에러 유형 |
| response node | 사용 모델, latency, token usage, fallback 여부 |
| API layer | request id, status code, 전체 처리 시간 |

여기서 중요한 것은 request id다. 하나의 사용자 요청이 여러 node를 지나가므로 같은 request id로 묶어야 나중에 흐름을 복원할 수 있다.

## 주의점

- 관측 데이터를 많이 남길수록 개인정보와 비용 문제가 커진다.
- raw prompt와 raw response를 그대로 저장할 때는 마스킹 기준을 먼저 정해야 한다.
- 대시보드는 문제를 보여줄 뿐이므로, 알림 기준과 장애 대응 절차가 함께 있어야 한다.

특히 LLM 서비스의 prompt에는 사용자 입력, 검색된 문서, 시스템 지시문이 함께 들어갈 수 있다. 따라서 관측성을 높인다는 이유로 raw prompt를 그대로 저장하면 보안 리스크가 커진다. 운영 로그에는 가능한 한 요약된 metadata를 남기고, 원문 저장이 필요하다면 마스킹과 접근 제어를 먼저 정해야 한다.

## Observability와 Evaluation의 차이

Observability는 실행 중인 시스템을 보는 일이고, Evaluation은 결과 품질을 평가하는 일이다. 둘은 연결되지만 같은 것은 아니다.

- Observability: 어느 node에서 어떤 일이 일어났는가?
- Evaluation: 최종 답변이 기준에 비추어 좋은가?
- Error Analysis: 실패 사례를 모아 원인 유형을 분류할 수 있는가?

예를 들어 RAG 답변이 틀렸다면 Observability는 “검색된 문서가 0개였다” 또는 “검색은 되었지만 response node가 문서를 사용하지 않았다”를 보여준다. Evaluation은 그 답변이 왜 틀렸는지를 점수나 기준으로 판단한다. Error Analysis는 같은 실패가 반복되는 패턴인지 확인한다.

## 관련 강의

- [Observability 수업 기록]({% post_url 2026-03-05-upstage-course-w09d03-observability %})
- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %})
- [AgentOps]({% post_url 2026-02-03-upstage-tech-agentops %})
- [Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %})

## 세 기둥

![Observability (관측 가능성) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-observability-diagram-1.svg)
