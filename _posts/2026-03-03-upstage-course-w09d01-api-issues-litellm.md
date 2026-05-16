---
title: "API 이슈 & LiteLLM"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- litellm
- course-note
- llmops
toc: true
date: 2026-03-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# API 이슈 & LiteLLM

## 수업 위치

이 수업은 Service Deployment 이후 LLMOps로 넘어가는 첫 번째 수업이다. 앞 단계까지는 FastAPI, LangGraph, Docker, CI/CD를 통해 에이전트 서비스를 실행하고 배포하는 흐름을 만들었다면, 여기서는 “운영 중인 LLM API가 실패하거나 느려지거나 비용이 튀면 어떻게 대응할 것인가”를 다룬다.

핵심은 모델 호출을 단순 함수 호출로 보지 않는 것이다. LLM API 호출은 외부 네트워크, 모델 제공자 상태, rate limit, 토큰 비용, 응답 품질, 보안 이슈가 동시에 얽힌 운영 지점이다. 그래서 서비스 코드 곳곳에서 모델을 직접 호출하면 장애 대응이 어려워진다. 모델 호출부를 한 계층으로 모으고, 그 계층에서 timeout, retry, fallback, logging을 관리해야 한다.

## 핵심 문제

> **요약**
> LLM API 사용 시 발생하는 rate limit, timeout, provider 장애, 비용 증가 문제를 다루고, LiteLLM Router를 통해 primary 모델과 fallback 모델을 운영 가능한 형태로 묶는 방법을 학습했다.

수업에서 본 문제는 크게 세 가지다.

- 모델 API는 실패할 수 있다. 네트워크 지연, 제공자 장애, rate limit, quota 초과가 모두 사용자 응답 실패로 이어질 수 있다.
- 모델 API는 비싸질 수 있다. 같은 요청도 retry, 긴 prompt, 불필요한 tool call이 쌓이면 비용이 증가한다.
- 모델 API는 품질이 흔들릴 수 있다. fallback 모델을 쓰면 응답 형식이나 품질이 달라질 수 있으므로, fallback을 “무조건 좋은 것”으로 보면 안 된다.

따라서 LLMOps의 첫 단계는 “모델을 하나 더 붙인다”가 아니라 “실패를 전제로 호출 계층을 설계한다”에 가깝다.

## LiteLLM로 잡은 구조

LiteLLM은 여러 LLM provider를 하나의 인터페이스로 호출할 수 있게 해주는 라이브러리다. 수업에서는 LiteLLM Router를 이용해 primary 모델과 fallback 모델을 구성하는 흐름을 다뤘다.

```text
사용자 요청
  -> LangGraph node
  -> LLM 호출 계층
  -> LiteLLM Router
  -> primary model
  -> 실패 시 fallback model
  -> 응답 반환
```

이 구조의 장점은 서비스 코드가 특정 provider SDK에 강하게 묶이지 않는다는 점이다. 예를 들어 응답 생성 node는 “모델 호출”만 요청하고, 실제로 어떤 provider를 먼저 쓰고 어떤 모델로 fallback할지는 Router 설정에서 관리한다.

## Mission 기준 구현 포인트

`day6-mission` 코드 기준으로 이 수업의 구현 포인트는 세 군데에 모인다.

| 위치 | 역할 | 보강할 내용 |
|---|---|---|
| `app/core/config.py` | 운영 설정 | LiteLLM 사용 여부, retry 횟수, timeout, fallback 모델을 환경변수로 분리 |
| `app/core/llm.py` | 모델 호출 계층 | LiteLLM Router 생성, primary/fallback 모델 설정, streaming 대응 |
| `app/graph/nodes.py` | 에이전트 실행 node | tool timeout, tool error, fallback 결과를 사용자 응답으로 안전하게 연결 |

이 중 가장 중요한 파일은 `app/core/llm.py`다. 모델 호출부가 이 파일로 모이면 Graph node, API handler, tool executor가 provider별 예외 처리에 직접 묶이지 않는다. 운영 관점에서는 이 분리가 retry, fallback, timeout, 비용 추적의 출발점이 된다.

## 수업에서 남길 체크포인트

LiteLLM을 붙였다고 해서 곧바로 운영 대응이 끝나는 것은 아니다. 아래 질문에 답할 수 있어야 실제 서비스 수준의 LLMOps로 이어진다.

- primary 모델이 실패했을 때 어떤 예외를 fallback 대상으로 볼 것인가?
- fallback 모델이 다른 응답 형식을 만들면 downstream node가 견딜 수 있는가?
- retry 횟수와 timeout은 사용자 경험과 비용 사이에서 어떤 기준으로 정할 것인가?
- 모델명, latency, token usage, fallback 발생 여부를 어디에 기록할 것인가?
- API key, prompt, 사용자 입력을 로그에 남길 때 마스킹 기준은 있는가?

이 질문들이 정리되어야 다음 수업인 상태 관리, 비용 추적, Observability, Error Analysis로 자연스럽게 이어진다.

## 흐름도

![API 이슈 & LiteLLM 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-llmops-ops-w09d01-api-issues-litellm-diagram-1.svg)

## 관련 글

- [LiteLLM 기술 노트]({% post_url 2026-03-03-upstage-tech-litellm %})
- [LiteLLM 통합 실습]({% post_url 2026-03-03-upstage-practice-litellm-practice %})
- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
