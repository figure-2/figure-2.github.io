---
title: "Week 09: LLMOps"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- week-09
- type-index
- course-note
- llmops
toc: true
date: 2026-03-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Week 09: LLMOps

> **개요**
> LLM 기반 서비스를 “만드는 단계”에서 “운영하는 단계”로 확장한 주차다. API 장애 대응, 모델 fallback, 상태 관리, 비용 추적, Observability를 다루고, idol-agent v0.6~v0.7 프로젝트로 연결한다.

## 주간 목표

Week 09의 핵심은 LLM API 호출을 운영 가능한 시스템 요소로 보는 것이다. 이전 주차까지는 LangGraph, RAG, Tool Calling, FastAPI, Docker, CI/CD를 통해 에이전트 서비스를 구현하고 배포하는 데 초점을 맞췄다. 이 주차부터는 서비스가 운영 중 실패하거나 비용이 증가하거나 답변 품질이 흔들릴 때 원인을 추적하고 대응하는 방법을 다룬다.

LLMOps에서 다룬 문제는 다음처럼 연결된다.

```text
API 장애 대응
  -> retry / timeout / fallback
상태 관리
  -> session / checkpoint / context trimming
비용 추적
  -> token usage / daily limit / alert
Observability
  -> log / metric / trace / error analysis
```

즉, Week 09는 기능 추가 주차라기보다 운영 안정성을 붙이는 주차에 가깝다.

## 일별 강의

| Day | 주제 | 강의 노트 | 미션 |
|-----|------|-----------|------|
| Day01 | API 이슈 & LiteLLM | [W09D01-API-이슈-LiteLLM]({% post_url 2026-03-03-upstage-course-w09d01-api-issues-litellm %}) | [idol-agent v0.6]({% post_url 2026-03-03-upstage-project-idol-agent-v06 %}) |
| Day02 | 상태 관리 | [W09D02-상태관리]({% post_url 2026-03-04-upstage-course-w09d02-state-management %}) | [idol-agent v0.7]({% post_url 2026-03-04-upstage-project-idol-agent-v07 %}) |
| Day03 | Observability | [W09D03-Observability]({% post_url 2026-03-05-upstage-course-w09d03-observability %}) | 모니터링 구현 |
| Day04 | Error Analysis | [W09D04-Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %}) | [오류 분석 실습]({% post_url 2026-03-06-upstage-practice-error-analysis-practice %}) |
| Day05 | LLMOps 패턴 정리 | [W09D05-LLMOps 패턴]({% post_url 2026-03-09-upstage-course-w09d05-llmops-patterns %}) | 추가 패턴 학습 |

## 핵심 개념

- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %}) - 멀티 provider 호출, retry, fallback
- [상태관리]({% post_url 2026-03-04-upstage-tech-state-management %}) - 세션, checkpoint, 대화 이력 영속화
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %}) - 로그, 메트릭, 트레이스 기반 운영 추적
- [Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %}) - trace 기반 오류 분류와 평가 기준 정의
- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %}) - 배포 이후 운영 자동화의 기반

## 관련 프로젝트

- [idol-agent v0.6]({% post_url 2026-03-03-upstage-project-idol-agent-v06 %}) - LiteLLM, Docker, CI/CD, tool error handling
- [idol-agent v0.7]({% post_url 2026-03-04-upstage-project-idol-agent-v07 %}) - 상태 관리, 비용 추적, 메시지 트리밍

## Wrap Up

이번 주차를 정리할 때는 “무엇을 만들었는가”보다 “어떤 운영 리스크를 줄였는가”를 기준으로 보는 편이 좋다.

- LiteLLM을 통해 provider 장애와 rate limit에 대응할 수 있는 호출 계층을 만들었다.
- 상태 관리를 통해 에이전트가 세션과 대화 흐름을 잃지 않도록 했다.
- 비용 추적과 메시지 트리밍을 통해 운영 비용과 context window 초과를 관리했다.
- Observability를 통해 에이전트가 왜 느렸는지, 어디서 실패했는지, 어떤 모델이 쓰였는지 추적할 수 있게 했다.
- Error Analysis를 통해 쌓인 trace를 평가 기준과 개선 작업으로 연결했다.
