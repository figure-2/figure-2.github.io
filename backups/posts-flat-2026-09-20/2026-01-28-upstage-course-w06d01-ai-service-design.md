---
title: "AI 서비스와 에이전트 설계"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- agent-architecture
- course-note
- agentic-workflow
toc: true
date: 2026-01-28 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## AI 서비스와 에이전트 설계

## 수업 위치

이 수업은 Agentic Workflow 주차의 시작점이다. 앞선 Prompt/RAG 단계에서 LLM을 잘 쓰는 방법을 배웠다면, 여기서는 그것을 실제 AI 서비스 구조로 묶는 방법을 다룬다.

초기 강의 계획서 기준으로는 `AI Product Engineering > Agentic Workflow`의 `AI Service & Agent Design`, `AI Agent Architecture Design`에 해당한다. 이후 Tool Calling, Memory, Context Engineering, Safety, Evaluation으로 이어지는 기준선을 잡는 수업이다.

## 핵심 개념

> **요약**
> AI 서비스 설계의 기본 원칙과 에이전트 아키텍처 패턴을 학습한다. Agentic AI의 핵심 구성요소(계획, 도구 사용, 메모리, 반성)를 이해하고, 워크플로우 기반 에이전트 설계 방법론을 다룬다.

## 주요 내용

### 1. AI 서비스 설계
- AI 서비스의 전체 아키텍처
- 사용자 요구사항에서 서비스 설계까지
- 프로덕션 환경에서의 고려사항
- 관련: [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})

### 2. Agentic AI
- **에이전트**: 자율적으로 환경을 인식하고 행동하는 AI 시스템
- 핵심 구성요소:
  - **Planning**: 목표를 세우고 단계를 계획
  - **Tool Use**: 외부 도구를 활용하여 작업 수행
  - **Memory**: 과거 경험과 컨텍스트 유지
  - **Reflection**: 결과를 평가하고 개선
- 관련: [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})

### 3. 에이전트 패턴
- **ReAct**: Reasoning + Acting의 결합
- **Plan-and-Execute**: 먼저 계획 수립 후 실행
- **Reflexion**: 결과 반성을 통한 자기 개선
- 관련: [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})

### 4. Workflow와 Agent
- **워크플로우**: 미리 정의된 경로를 따르는 구조화된 실행
- **에이전트**: 상황에 따라 동적으로 경로를 결정
- 워크플로우와 에이전트의 하이브리드 접근
- 관련: [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})

## 설계 관점

AI 서비스를 설계할 때는 모델을 먼저 고르기보다 사용자의 작업 흐름을 먼저 나눠야 한다. 사용자가 무엇을 입력하고, 시스템이 어떤 정보를 찾아야 하며, 어떤 판단을 자동화하고, 어떤 지점은 사람에게 확인받아야 하는지를 분리해야 한다.

```text
사용자 문제
  -> 입력/출력 정의
  -> 필요한 지식과 데이터 확인
  -> LLM이 맡을 판단 구간 정의
  -> 도구/API/DB 연결 정의
  -> 실패와 예외 처리 기준 정의
  -> 평가와 운영 지표 정의
```

이 흐름에서 LLM은 전체 서비스의 일부다. 프롬프트만 잘 쓰는 것으로 끝나지 않고, 데이터, 도구, 권한, 로그, 평가가 함께 설계되어야 한다.

## Workflow와 Agent를 나누는 기준

| 기준 | Workflow | Agent |
| --- | --- | --- |
| 실행 경로 | 미리 정해짐 | 실행 중 선택 |
| 장점 | 안정적이고 디버깅 쉬움 | 복잡한 상황에 유연함 |
| 단점 | 예외 처리에 약함 | 비용과 예측 불가능성 증가 |
| 예시 | RAG 검색 후 답변 생성 | 검색 필요성 판단, 도구 선택, 재시도 |

서비스 초반에는 워크플로우로 안정적인 baseline을 만들고, 판단이 필요한 구간만 에이전트화하는 것이 현실적이다.

## 체크포인트

- 사용자의 문제를 한 문장으로 정의했는가
- LLM이 필요한 이유가 명확한가
- 검색, 도구 호출, 메모리 중 어떤 기능이 필요한가
- 실패했을 때 재시도, 중단, 사람 확인 중 어떤 경로로 갈 것인가
- 평가 기준을 기능 구현 전에 정했는가
- 운영 단계에서 비용과 로그를 추적할 수 있는가

## 흐름도

![AI 서비스와 에이전트 설계 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agentic-workflow-w06d01-ai-service-design-diagram-1.svg)

## 연결된 개념
- [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [Agent Evaluation]({% post_url 2026-02-03-upstage-tech-agent-evaluation %})
