---
title: "LiteLLM"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- litellm
- tech-note
- llmops
toc: true
date: 2026-03-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## LiteLLM

> **한줄 정의**
> 다양한 LLM API provider를 하나의 호출 계층으로 묶고, retry, fallback, timeout, routing을 관리하기 위한 LLMOps 도구.

## 핵심 이해

LiteLLM을 단순히 “여러 모델을 같은 문법으로 호출하는 라이브러리”로만 보면 활용 범위가 좁아진다. 수업에서 중요한 지점은 모델 호출을 운영 계층으로 분리하는 것이다.

LLM 서비스의 모델 호출은 일반적인 내부 함수 호출과 다르다. 외부 provider의 장애, rate limit, timeout, quota, 비용, 응답 형식 차이가 모두 서비스 품질에 영향을 준다. LiteLLM은 이 문제를 Router 중심으로 다룰 수 있게 한다.

```text
application code
  -> internal LLM interface
  -> LiteLLM Router
  -> primary provider
  -> fallback provider
```

이 구조를 만들면 Graph node나 API handler는 특정 provider SDK를 직접 알 필요가 없다. 서비스 코드는 내부 인터페이스만 호출하고, provider 교체와 fallback 정책은 설정과 Router 계층에서 관리한다.

## 언제 쓰는지

LiteLLM은 다음 상황에서 유용하다.

- primary 모델 장애 시 fallback 모델로 전환해야 할 때
- 모델별 비용, 속도, 품질을 비교하면서 운영 정책을 정해야 할 때
- 개발 환경과 운영 환경에서 다른 모델을 써야 할 때
- LangGraph node가 provider별 SDK 예외 처리에 묶이는 것을 피하고 싶을 때
- streaming, retry, timeout을 모델 호출 계층에서 일관되게 처리해야 할 때

수업의 에이전트 프로젝트에서는 모델 호출부를 한 파일에 모으는 것이 핵심이었다. 이렇게 해야 router node, rag node, response node가 각각 provider 장애 대응 코드를 중복해서 갖지 않는다.

## 구현 관점

`day6-mission` 기준 구현은 `app/core/llm.py`의 `LiteLLMChatModel` 같은 wrapper를 중심으로 잡을 수 있다. 목표는 LangChain의 ChatModel처럼 사용할 수 있으면서 내부에서는 LiteLLM Router를 호출하게 만드는 것이다.

```text
LangGraph node
  -> get_llm()
  -> LiteLLMChatModel
  -> router.acompletion(model="primary", messages=...)
  -> primary 실패 시 fallback
```

구현할 때는 다음 항목을 분리해서 관리해야 한다.

- `use_litellm`: LiteLLM 사용 여부
- `litellm_num_retries`: 재시도 횟수
- `litellm_timeout`: 모델 호출 timeout
- `litellm_fallback_model`: fallback 모델 이름
- provider API key: 환경변수 또는 secret으로만 주입

중요한 점은 “fallback 모델이 호출되었다”는 사실도 운영 데이터라는 것이다. fallback이 자주 발생하면 primary provider 장애일 수도 있고, timeout 설정이 너무 짧을 수도 있고, prompt가 과도하게 길어졌을 수도 있다.

## 운영 체크리스트

LiteLLM을 서비스에 붙일 때는 아래 항목을 같이 확인해야 한다.

| 항목 | 확인할 질문 |
|---|---|
| timeout | 사용자가 기다릴 수 있는 최대 시간은 얼마인가? |
| retry | 같은 요청을 몇 번까지 재시도할 것인가? |
| fallback | fallback 모델이 같은 출력 형식을 보장하는가? |
| logging | model, latency, token usage, error type을 남기는가? |
| cost | 실패한 호출과 retry 비용도 계산하는가? |
| security | API key, raw prompt, 개인정보가 로그에 남지 않는가? |

## 주의점

provider마다 지원 파라미터와 응답 형식이 조금씩 다르므로 완전한 호환을 가정하면 안 된다. 특히 structured output, tool calling, streaming은 provider별 차이가 크게 드러날 수 있다.

fallback은 장애 대응 수단이지만 품질 리스크도 만든다. 예를 들어 primary 모델은 한국어 답변을 안정적으로 만들지만 fallback 모델은 출력 형식이 달라질 수 있다. 따라서 fallback 후에도 downstream parser, graph node, frontend가 깨지지 않는지 확인해야 한다.

비용 추적은 성공한 호출만 보면 부족하다. timeout 직전까지 토큰을 사용한 요청, 여러 번 retry된 요청, fallback까지 이어진 요청이 모두 실제 비용에 영향을 준다.

## 관련 강의

- [API 이슈 & LiteLLM]({% post_url 2026-03-03-upstage-course-w09d01-api-issues-litellm %})
- [LiteLLM 통합 + Docker 실습]({% post_url 2026-03-03-upstage-practice-litellm-practice %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})

## 구조/흐름

![LiteLLM 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-litellm-diagram-1.svg)
