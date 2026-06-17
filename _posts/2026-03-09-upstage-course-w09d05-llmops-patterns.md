---
title: "LLMOps 패턴 정리"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- llmops
- course-note
- operations
toc: true
date: 2026-03-09 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## LLMOps 패턴 정리

## 수업 위치

이 수업은 Service Deployment와 LLMOps 주차를 마무리하는 정리 강의다. 빈 프로젝트 폴더에서 시작해 FastAPI, LangGraph, RAG, Tool Calling, Streaming, CI/CD, LiteLLM, 상태 관리, Observability, Error Analysis까지 붙인 흐름을 운영 패턴으로 정리한다.

핵심은 특정 도구 이름을 외우는 것이 아니다. 도구는 바뀔 수 있지만 문제 상황은 남는다. Langfuse가 아니어도 Observability는 필요하고, LiteLLM이 아니어도 Gateway 패턴은 필요하다.

## 핵심 개념

> **요약**
> LLMOps는 LLM 서비스의 비결정성, 비용, 품질 측정, prompt 관리, 보안 문제를 DevOps/MLOps 관점으로 다루는 운영 체계다. 모든 패턴을 한 번에 도입하기보다 서비스 단계와 문제 상황에 맞춰 점진적으로 적용한다.

## 주요 패턴

| 패턴 | 목적 | 이번 과정과의 연결 |
|---|---|---|
| Observability | 요청, 응답, latency, token, cost를 추적 | Langfuse, trace, node별 실행 흐름 |
| Cost Control | 비용 폭증 방지 | token counter, daily limit, Discord alert |
| Rate Limiting | 호출 횟수 제한 | 악의적 호출, provider limit 초과 방지 |
| Evaluation | 품질 기준 정의 | Error Analysis, LLM-as-a-Judge |
| Prompt Management | prompt 버전과 실험 관리 | prompt 수정이 기능에 미치는 영향 추적 |
| Caching | 반복 호출 비용 절감 | 동일 질문, semantic cache |
| Gateway | retry, fallback, timeout | LiteLLM Router |
| Guardrails | 안전성 확보 | prompt injection, hallucination, 개인정보 노출 대응 |
| Feedback Loop | 사용자 데이터 기반 개선 | trace 분석, 좋아요/싫어요, Data Flywheel |

## 점진적 도입 순서

모든 패턴을 처음부터 구현하려고 하면 작업량이 커지고 실제 서비스 검증이 늦어진다. 학습 프로젝트나 초기 서비스에서는 다음 순서가 현실적이다.

```text
Phase 1: Observability + Cost Control + Rate Limiting
Phase 2: Evaluation + Caching + Prompt Management
Phase 3: Gateway + Guardrails
Phase 4: Feedback Loop
```

이번 과정에서는 Phase 1과 Gateway의 일부를 먼저 경험했다. 특히 LiteLLM을 통한 retry/fallback, token usage 기반 비용 추적, Langfuse 기반 trace 확인은 LLMOps의 기본 골격에 해당한다.

## 안티패턴

- 한 번에 모든 LLMOps 패턴을 구현하려고 한다.
- 평가 없이 “좋아 보이니까” 배포한다.
- 비용 추적을 나중으로 미룬다.
- 자율 에이전트나 멀티 에이전트로 바로 넘어가 디버깅 불가능한 구조를 만든다.
- prompt를 코드 안의 문자열로만 두고 변경 이력을 남기지 않는다.

이런 실수는 기능이 많아질수록 더 크게 돌아온다. 특히 에이전트는 router, RAG, tool, model, prompt가 모두 얽히기 때문에 문제가 발생했을 때 원인을 찾을 수 있는 구조가 먼저 필요하다.

## 앞으로의 학습 방향

LLMOps를 계속 발전시키려면 도구 사용법만 보면 부족하다. Python, Linux, shell command, 코드 품질, 아키텍처 설계, 데이터 엔지니어링, 대규모 시스템 설계가 함께 필요하다.

AI 엔지니어링은 모델 하나를 잘 호출하는 일이 아니라, 사용자 요청이 모델, 도구, 데이터, 배포 환경을 거쳐 안정적으로 돌아오게 만드는 일이다. 그래서 Service Deployment와 LLMOps를 묶어서 보면 전체 과정의 의미가 선명해진다.

## 관련 글

- [Week 09: LLMOps]({% post_url 2026-03-03-upstage-course-week09-llmops %})
- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %})
- [상태관리]({% post_url 2026-03-04-upstage-tech-state-management %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
- [Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %})
