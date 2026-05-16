---
title: "AgentOps"
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
date: 2026-02-03 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# AgentOps

> **한줄 정의**
> AI 에이전트의 운영, 모니터링, 비용 관리를 위한 운영 프레임워크.

## 학습 맥락

AgentOps는 W06D05의 Evaluation & AgentOps에서 처음 다뤘다. 앞선 주차에서 프롬프트, RAG, Tool Calling, LangGraph 같은 "에이전트를 만드는 방법"을 배웠다면, 이 주제는 만든 에이전트를 실제 서비스처럼 운영하려면 무엇을 봐야 하는지에 초점을 둔다.

에이전트는 일반 API보다 실행 흐름이 복잡하다. 한 번의 사용자 요청 안에서 모델 호출, 도구 호출, 검색, 라우팅, 재시도, fallback이 여러 번 발생할 수 있다. 그래서 단순히 에러 로그만 남기면 "왜 이상한 답변이 나왔는지"를 추적하기 어렵다. AgentOps는 이 흐름을 세션 단위로 추적하고, 품질과 비용을 함께 관리하기 위한 운영 관점이다.

## 핵심 개념

AgentOps는 MLOps/LLMOps의 에이전트 특화 버전이다. 에이전트의 실행 추적, 비용 모니터링, 성능 분석, 장애 대응 등 프로덕션 운영에 필요한 기능을 제공한다.

주요 기능으로는 세션 추적(에이전트 실행 흐름 기록), 비용 추적(API 호출 비용 모니터링), 성능 분석(지연시간, 처리량 측정), 그리고 알림(임계값 초과 시 알림)이 있다.

![Evaluation & AgentOps 흐름도](/assets/images/upstage-ai-agent/diagrams/01-modules-agentic-workflow-w06d05-evaluation-agentops-diagram-1.svg)

## 언제 쓰는지

AgentOps는 에이전트를 로컬 실습이 아니라 서비스로 운영할 때 필요하다. 사용자의 요청이 어떤 노드를 거쳤는지, 어떤 도구를 호출했는지, 어느 모델에서 비용이 많이 발생했는지 확인해야 할 때 운영 기준점이 된다.

특히 LangGraph처럼 여러 노드와 조건부 라우팅을 가진 구조에서는 단순 로그만으로 문제를 추적하기 어렵다. 세션 단위 실행 기록을 남기면 실패한 단계, 과도한 토큰 사용, 응답 품질 저하 지점을 다시 확인할 수 있다.

## 구현 관점

- 실행 단위마다 `run_id`, 사용자 세션, 모델명, 토큰 사용량, 지연시간을 함께 기록한다.
- 노드별 입력/출력, 도구 호출 결과, fallback 발생 여부를 추적한다.
- 비용 한도, 실패율, 응답 지연시간에 대한 알림 기준을 정한다.
- 평가 결과와 운영 로그를 연결해 품질 문제와 시스템 문제를 분리한다.

## 무엇을 기록할까

운영 로그는 많을수록 좋은 것이 아니라, 나중에 문제를 설명할 수 있을 만큼 구조화되어야 한다. 최소한 다음 정보는 분리해서 남기는 것이 좋다.

| 항목 | 목적 |
| --- | --- |
| `run_id` / `session_id` | 한 사용자 요청의 전체 흐름 추적 |
| model name | 어떤 모델에서 문제가 났는지 확인 |
| prompt version | 프롬프트 변경 전후 품질 비교 |
| tool calls | 외부 도구 호출 성공/실패 확인 |
| latency | 응답 지연 원인 분석 |
| token usage / cost | 비용 증가 지점 확인 |
| evaluation score | 운영 로그와 품질 평가 연결 |

## Evaluation과의 관계

Evaluation은 "답변이 좋은가"를 묻고, AgentOps는 "운영 중 어떤 일이 일어났는가"를 묻는다. 둘 중 하나만 있으면 원인을 찾기 어렵다.

예를 들어 답변 품질이 낮아졌을 때 평가 점수만 보면 품질 저하는 알 수 있지만, 원인은 모른다. 반대로 로그만 보면 어떤 도구가 실패했는지는 알 수 있지만, 그 결과 답변 품질이 실제로 나빠졌는지는 판단하기 어렵다. 그래서 에이전트 서비스에서는 평가 결과와 실행 로그를 연결해서 봐야 한다.

## 프로젝트 적용 예시

idol-agent 같은 LangGraph 기반 프로젝트라면 다음 기준으로 적용할 수 있다.

1. 사용자 요청마다 `run_id`를 생성한다.
2. 각 노드의 입력, 출력, 소요 시간, 모델명을 기록한다.
3. Tool Calling 실패와 fallback 발생 여부를 별도 필드로 남긴다.
4. 요청당 토큰 사용량과 비용을 집계한다.
5. 주요 시나리오별 평가 데이터를 만들어 변경 전후를 비교한다.

## 주의점

- 전체 프롬프트와 사용자 입력을 그대로 저장하면 개인정보나 민감정보가 남을 수 있다.
- 비용 모니터링만 있고 품질 평가가 없으면 싼 답변과 좋은 답변을 구분하기 어렵다.
- 대시보드만 만드는 것으로 끝내지 말고, 임계값 초과 시 대응 절차까지 함께 정해야 한다.

## 관련 글

- [Evaluation & AgentOps]({% post_url 2026-02-03-upstage-course-w06d05-evaluation-agentops %})
- [Agent Evaluation]({% post_url 2026-02-03-upstage-tech-agent-evaluation %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %})
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
