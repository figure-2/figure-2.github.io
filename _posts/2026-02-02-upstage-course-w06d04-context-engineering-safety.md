---
title: "Context Engineering & Safety"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- context-engineering
- llm-security
- course-note
- agentic-workflow
toc: true
date: 2026-02-02 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Context Engineering & Safety

## 수업 위치

이 수업은 Agentic Workflow 주차에서 Tool Calling, Agentic RAG, Memory를 배운 뒤 이어지는 보안/품질 관리 단계다. 에이전트가 도구를 쓰고 장기 메모리를 갖게 되면, 컨텍스트에 어떤 정보를 넣을지와 어떤 정보를 믿으면 안 되는지가 서비스 품질과 안전성을 결정한다.

초기 강의 계획서 기준으로는 `Context Engineering 2.0 & DeepAgent`, `Safety & Security`에 해당한다. 단순 프롬프트 작성이 아니라 컨텍스트의 생성, 저장, 검색, 활용, 갱신을 설계 대상으로 본다.

## 핵심 개념

> **요약**
> 에이전틱 워크플로우의 핵심인 Context Engineering 2.0 개념을 학습한다. 단순한 프롬프트 엔지니어링을 넘어 컨텍스트의 생애주기를 체계적으로 관리하는 방법론을 다루고, DeepAgent를 통한 성능 극대화 기법을 살펴본다. 또한 에이전트 시스템의 보안 취약점과 안전한 서비스 구축을 위한 가드레일 전략을 학습한다.

## 주요 내용

### 1. Context Engineering 2.0
- 프롬프트 엔지니어링에서 컨텍스트 엔지니어링으로의 진화
- 컨텍스트 생애주기 관리: 생성 → 저장 → 검색 → 활용 → 갱신
- 데이터 중심의 컨텍스트 최적화 전략
- 에이전트의 행동 품질은 컨텍스트 품질에 의존
- 관련: [Context Engineering]({% post_url 2026-02-02-upstage-tech-context-engineering %})

### 2. DeepAgent
- 에이전트 성능을 극대화하는 구축 기법
- 컨텍스트를 체계적으로 구조화하여 에이전트 잠재력 극대화
- 그동안 학습한 도구 사용, MCP, 메모리 관리의 통합 적용
- 관련: [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})

### 3. Safety & Security
- 강력한 도구 사용 능력을 가진 에이전트의 내재적 위험
- 에이전트 시스템 보안 취약점:
  - **Prompt Injection**: 악의적 입력을 통한 에이전트 조작
  - **Tool Misuse**: 도구의 의도치 않은 사용
  - **Data Leakage**: 민감 정보 노출
- 관련: [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})

### 4. 가드레일 전략
- 입력 검증 및 필터링
- 출력 모니터링 및 제한
- 도구 접근 권한 제어
- 안전한 컨텍스트 설계 패턴
- 관련: [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})

## 컨텍스트를 설계한다는 것

컨텍스트는 LLM에게 전달되는 모든 정보다. 시스템 프롬프트, 사용자 입력, 대화 이력, RAG 검색 결과, 도구 실행 결과, 메모리, 정책 문서가 모두 컨텍스트에 포함된다.

문제는 컨텍스트가 많을수록 좋은 것이 아니라는 점이다. 불필요한 정보가 많으면 핵심 지시가 흐려지고, 오래된 대화나 관련 없는 검색 결과가 답변 품질을 낮출 수 있다. 따라서 Context Engineering은 어떤 정보를 넣을지뿐 아니라 무엇을 빼야 하는지도 함께 다룬다.

## 컨텍스트 생애주기

| 단계 | 질문 |
| --- | --- |
| 생성 | 어떤 정보가 새로 들어오는가 |
| 저장 | 이 정보를 나중에 다시 쓸 가치가 있는가 |
| 검색 | 현재 요청에 어떤 정보가 필요한가 |
| 활용 | 검색 결과를 어떤 우선순위로 넣을 것인가 |
| 갱신 | 오래되거나 틀린 정보를 어떻게 수정할 것인가 |

Agentic RAG와 Memory를 쓸수록 이 생애주기가 중요해진다. 에이전트가 과거 정보를 잘못 기억하거나, 검색된 문서의 지시문을 그대로 따르면 품질과 보안이 동시에 흔들릴 수 있다.

## Safety 관점

에이전트 보안은 일반 챗봇보다 어렵다. 이유는 에이전트가 도구를 호출하고, 외부 데이터를 읽고, 때로는 쓰기 작업까지 수행할 수 있기 때문이다.

특히 주의할 위험은 다음과 같다.

- Prompt Injection: 사용자나 문서가 시스템 지시를 무시하게 유도
- Tool Misuse: 의도하지 않은 도구 호출이나 잘못된 인자 사용
- Data Leakage: 대화 이력, 검색 문서, secret, 내부 정책 노출
- Permission Bypass: 사용자가 접근 권한이 없는 데이터에 우회 접근
- Unsafe Autonomy: 위험한 작업을 승인 없이 실행

## 가드레일 설계 기준

가드레일은 모델 답변을 막는 후처리만 의미하지 않는다. 입력, 검색, 도구 실행, 출력, 로그 저장 전 단계에 나눠서 둬야 한다.

```text
입력 검증
  -> 권한 기반 검색
  -> 도구 호출 schema 검증
  -> 위험 액션 승인
  -> 출력 필터링
  -> 로그 마스킹
```

이 구조를 두면 에이전트가 강력한 도구를 쓰더라도, 실행 권한은 애플리케이션이 통제할 수 있다.

## 실습/코드

- Practice07: Context Engineering 실습 ()
- Practice08: Safety Guardrails 실습 ()

## 흐름도

![Context Engineering & Safety 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agentic-workflow-w06d04-context-engineering-safety-diagram-1.svg)

## 연결된 개념
- [Context Engineering]({% post_url 2026-02-02-upstage-tech-context-engineering %})
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})
- [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})
- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %})
