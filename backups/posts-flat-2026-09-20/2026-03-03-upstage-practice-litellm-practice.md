---
title: "LiteLLM 통합 + Docker 실습"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- PRACTICE
tags:
- upstage
- sesac
- ai-agent
- litellm
- docker
- practice
- llmops
toc: true
date: 2026-03-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## LiteLLM 통합 + Docker 실습

> **실습 정보**
> - **주차**: Week 09, Day 01
> - **유형**: 코드 구현 (Python)
> - **상태**: 완료

## 실습 목표

이 실습의 목표는 모델 호출 실패를 전제로 LLM 호출 계층을 만드는 것이다. 단순히 다른 모델 API를 하나 더 붙이는 것이 아니라, primary 모델 실패 시 fallback 모델을 사용하고, timeout과 retry를 설정하며, tool 실행 실패까지 사용자 응답으로 안전하게 연결하는 흐름을 만든다.

## 핵심 구현 사항

- LiteLLM 환경변수 추가: `use_litellm`, `fallback_model`, `timeout`, `num_retries`
- LiteLLM Router 생성: primary 모델과 fallback 모델 구성
- LangGraph node에서 동일한 LLM interface를 사용하도록 정리
- tool 실행에 `asyncio.wait_for` 기반 timeout 적용
- tool error를 graph state에 남기고 응답 node에서 설명 가능한 메시지로 변환

## 시작점

`day6-mission` 프로젝트의 실습 포인트는 세 파일에 집중된다.

| 파일 | 실습 내용 |
|---|---|
| `app/core/config.py` | LiteLLM 관련 환경변수와 기본값 추가 |
| `app/core/llm.py` | LiteLLM Router wrapper 구현 |
| `app/graph/nodes.py` | tool timeout, exception, 실패 결과 처리 |

이 구조를 잡으면 나중에 모델 provider를 바꾸더라도 Graph node를 크게 고치지 않아도 된다. 모델 호출 정책은 `core/llm.py`로 모이고, node는 “질문을 받아 모델 또는 tool에 전달한다”는 역할에 집중할 수 있다.

## 진행 순서

1. 설정 추가

   `.env.example`과 config 객체에 LiteLLM 사용 여부, retry 횟수, timeout, fallback 모델명을 추가한다. API key는 코드에 넣지 않고 환경변수로만 주입한다.

2. Router 생성

   `app/core/llm.py`에서 LiteLLM Router를 만들고, 서비스 내부에서는 `primary` 같은 alias로 호출한다. 실제 provider와 모델명은 Router 설정으로 분리한다.

3. LangGraph 연결

   router node, rag node, response node가 같은 LLM factory를 사용하도록 맞춘다. 이렇게 해야 node마다 provider별 예외 처리가 흩어지지 않는다.

4. Tool 에러 처리

   tool call은 외부 I/O가 포함될 수 있으므로 timeout을 둔다. tool이 실패했을 때는 예외를 그대로 터뜨리지 않고 graph state에 실패 사유를 남긴 뒤 response node가 사용자에게 안전한 메시지로 바꾼다.

5. Docker 환경 확인

   로컬 실행과 컨테이너 실행에서 같은 환경변수가 필요한지 확인한다. 운영에서는 GitHub Actions나 서버 secret에 들어갈 값과 `.env.example`에 남길 placeholder를 구분해야 한다.

## 체크포인트

- [ ] primary 모델 장애 시 fallback 모델이 호출되는 경로를 설명할 수 있다.
- [ ] retry와 timeout 값이 어디에서 설정되는지 찾을 수 있다.
- [ ] fallback 모델이 다른 출력 형식을 만들 때의 위험을 알고 있다.
- [ ] tool timeout과 model timeout을 구분할 수 있다.
- [ ] API key와 raw prompt가 로그나 코드에 남지 않는지 확인했다.

## 회고 질문

- 모델 호출을 각 node에 직접 넣는 방식과 `core/llm.py`로 모으는 방식의 차이는 무엇인가?
- fallback은 장애 대응에 도움이 되지만 어떤 품질 리스크를 만드는가?
- timeout을 너무 짧게 잡으면 어떤 문제가 생기고, 너무 길게 잡으면 어떤 문제가 생기는가?
- 운영 로그에 남겨야 하는 정보와 남기면 안 되는 정보는 어떻게 구분할 것인가?

## 관련 글

- [API 이슈 & LiteLLM]({% post_url 2026-03-03-upstage-course-w09d01-api-issues-litellm %})
- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %})
- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
