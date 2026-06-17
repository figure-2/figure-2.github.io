---
title: "Tool Calling"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- tool-calling
- tech-note
- agentic-workflow
toc: true
date: 2026-01-29 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Tool Calling

> **한줄 정의**
> LLM이 외부 도구(API, 함수)를 호출하여 작업을 수행하는 기법. LLM의 텍스트 생성 능력을 실제 행동 능력으로 확장한다.

## 학습 맥락

Tool Calling은 Agentic Workflow의 실행 능력을 만드는 핵심 개념이다. 프롬프트만 사용하는 LLM은 텍스트를 생성할 수 있지만, 파일을 읽거나 API를 호출하거나 데이터베이스를 조회하는 일은 직접 할 수 없다. Tool Calling은 모델이 "어떤 도구를 어떤 인자로 호출해야 하는지"를 구조화된 형태로 내보내고, 실제 실행은 애플리케이션 코드가 담당하게 만드는 방식이다.

강의 계획서 기준으로는 `Function Calling Fundamentals`, `Advanced Tool Use & Error Handling`, `MCP` 흐름과 연결된다. 단순 함수 호출에서 시작해 도구 실패 처리, 권한 제어, 표준화된 도구 연결로 확장된다.

## 핵심 개념

Tool Calling(Function Calling)은 LLM이 특정 형식의 JSON 출력을 생성하면, 이를 파싱하여 실제 함수를 실행하는 메커니즘이다. 도구 스키마(Tool Schema)는 함수 이름, 파라미터, 설명을 JSON Schema로 정의한다. LLM은 이 스키마를 보고 적절한 도구를 선택한다.

실행 루프는 LLM → 도구 선택 → 함수 실행 → 결과 관찰 → LLM 순서로 반복된다. 에러 핸들링이 중요하며, 도구 실행 실패 시 LLM이 재시도하거나 대안을 선택할 수 있어야 한다. 병렬 도구 호출(Parallel Function Calling)로 여러 도구를 동시에 실행할 수도 있다.

## LLM-도구 상호작용

![Tool Calling 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-tool-calling-diagram-1.svg)

## 기본 흐름

```text
사용자 요청
  -> LLM이 도구 필요 여부 판단
  -> 도구 이름과 인자 생성
  -> 애플리케이션이 스키마 검증
  -> 실제 함수/API 실행
  -> 실행 결과를 LLM에 전달
  -> 최종 답변 생성
```

중요한 점은 LLM이 직접 함수를 실행하는 것이 아니라는 점이다. LLM은 호출 의도를 구조화해 제안하고, 실행 권한은 애플리케이션이 가진다. 이 분리가 보안과 디버깅에서 중요하다.

## 스키마 설계 기준

좋은 도구 스키마는 모델이 헷갈리지 않게 만든다. 이름은 구체적이어야 하고, 설명은 언제 써야 하는지와 언제 쓰면 안 되는지를 함께 알려야 한다.

```json
{
  "name": "search_course_notes",
  "description": "업스테이지 과정 노트에서 관련 개념이나 수업 기록을 검색한다.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "검색할 개념, 수업명, 키워드"
      }
    },
    "required": ["query"]
  }
}
```

도구가 많아질수록 설명의 품질이 중요해진다. 비슷한 도구가 여러 개 있으면 모델이 잘못 선택할 수 있으므로, 도구의 책임을 작게 나누고 이름을 명확히 해야 한다.

## 에러 처리

Tool Calling은 성공 경로보다 실패 경로가 더 중요하다. API timeout, 잘못된 인자, 권한 없음, 빈 검색 결과 같은 상황을 모델에게 어떻게 돌려줄지 정해야 한다.

| 실패 유형 | 처리 기준 |
| --- | --- |
| schema validation 실패 | 모델에게 인자 형식을 다시 생성하게 한다 |
| 권한 없음 | 재시도하지 않고 사용자 승인 또는 중단으로 보낸다 |
| timeout | 제한된 횟수만 재시도한다 |
| 빈 결과 | 쿼리 재작성 또는 다른 검색 소스로 전환한다 |
| 위험한 액션 | 실행 전 명시적 확인 단계를 둔다 |

## MCP와의 관계

Tool Calling이 "모델이 도구 호출을 표현하는 방식"이라면, MCP는 도구와 리소스를 표준화된 인터페이스로 연결하는 방식이다. 도구가 몇 개 없을 때는 애플리케이션 내부 함수로 충분하지만, 파일, DB, 브라우저, 외부 SaaS처럼 연결 대상이 많아지면 MCP 같은 표준 인터페이스가 유리하다.

## 주의점

- 도구 설명이 애매하면 모델이 잘못된 도구를 선택한다.
- LLM이 만든 인자를 신뢰하지 말고 실행 전 검증해야 한다.
- 쓰기, 삭제, 결제, 외부 전송 같은 작업은 승인 단계를 둬야 한다.
- 도구 결과에 포함된 지시문을 시스템 지시처럼 따르면 prompt injection 위험이 생긴다.
- 호출 로그에는 secret, token, raw payload를 남기지 않아야 한다.

## 관련 글

- [Tool Calling Fundamentals & MCP]({% post_url 2026-01-29-upstage-course-w06d02-tool-calling-mcp %})
- [MCP]({% post_url 2026-01-29-upstage-tech-mcp %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})
- [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})
- [Context Engineering & Safety]({% post_url 2026-02-02-upstage-course-w06d04-context-engineering-safety %})

## 참고 자료

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
