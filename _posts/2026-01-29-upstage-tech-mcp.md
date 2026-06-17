---
title: "MCP (Model Context Protocol)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- mcp
- tech-note
- agentic-workflow
toc: true
date: 2026-01-29 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## MCP (Model Context Protocol)

> **한줄 정의**
> Model Context Protocol — LLM과 외부 도구/데이터를 연결하는 표준 프로토콜. Anthropic이 제안한 개방형 표준으로 도구 통합을 표준화한다.

## 학습 맥락

MCP는 W06D02 `Tool Calling Fundamentals & MCP`에서 Tool Calling 다음 단계로 다뤘다. Tool Calling이 LLM이 어떤 함수나 API를 호출할지 표현하는 방식이라면, MCP는 여러 도구와 데이터 소스를 일관된 방식으로 연결하기 위한 인터페이스다.

에이전트 서비스가 커지면 연결 대상이 많아진다. 파일 시스템, 데이터베이스, 검색 API, 브라우저, 사내 문서, 배포 도구가 각각 다른 방식으로 붙으면 에이전트 코드는 복잡해진다. MCP는 이런 연결을 서버 단위로 분리해 LLM 애플리케이션에서 재사용할 수 있게 만드는 접근이다.

## 핵심 개념

MCP는 **Host**, **Client**, **Server** 세 계층으로 구성된다. Host는 Claude Desktop 같은 LLM 애플리케이션이고, Client는 MCP 서버와 통신하는 컴포넌트며, Server는 실제 도구와 리소스를 제공하는 프로세스다. JSON-RPC 2.0 프로토콜로 통신한다.

MCP Server는 세 가지 주요 기능을 제공한다. **Tools**는 LLM이 호출할 수 있는 함수, **Resources**는 파일이나 데이터 소스에 대한 접근, **Prompts**는 재사용 가능한 프롬프트 템플릿이다. 표준화된 인터페이스 덕분에 한 번 만든 MCP 서버를 여러 LLM 애플리케이션에서 재사용할 수 있다.

## MCP 아키텍처

![MCP (Model Context Protocol) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-mcp-diagram-1.svg)

## 구성 요소

| 구성 요소 | 역할 |
| --- | --- |
| Host | 사용자가 대화하는 LLM 애플리케이션 |
| Client | Host 안에서 MCP Server와 통신하는 연결 계층 |
| Server | 도구, 리소스, 프롬프트를 제공하는 독립 프로세스 |
| Tools | 모델이 호출할 수 있는 실행 기능 |
| Resources | 모델이 읽을 수 있는 파일, 문서, 데이터 |
| Prompts | 재사용 가능한 프롬프트 템플릿 |

이 구조의 장점은 도구 연결 책임을 애플리케이션 본체에서 분리할 수 있다는 점이다. 예를 들어 GitHub, 파일 시스템, 데이터베이스를 각각 MCP 서버로 분리하면 에이전트는 표준화된 방식으로 도구를 발견하고 호출할 수 있다.

## Tool Calling과의 차이

Tool Calling은 LLM이 "이 함수를 호출하고 싶다"는 의도를 구조화하는 방식이다. MCP는 그 도구를 어디서 찾고, 어떻게 연결하고, 어떤 리소스를 읽을 수 있는지까지 포함하는 연결 프로토콜이다.

```text
Tool Calling
  -> 모델이 호출할 함수 이름과 인자를 생성

MCP
  -> 도구와 리소스를 표준 서버 형태로 제공
  -> Host/Client가 서버와 통신
  -> 여러 애플리케이션에서 같은 도구 재사용
```

## 구현 관점

MCP를 적용할 때는 "무엇을 도구로 만들 것인가"보다 "어떤 권한 경계를 둘 것인가"를 먼저 정해야 한다. 파일 읽기, DB 조회, 외부 API 호출은 모두 권한과 감사 로그가 필요하다.

MCP 서버를 설계할 때 확인할 기준은 다음과 같다.

- 도구의 책임이 하나로 분리되어 있는가
- 입력 schema가 명확한가
- 읽기 도구와 쓰기 도구가 분리되어 있는가
- 실패 응답이 모델이 이해할 수 있는 형태인가
- secret, token, raw payload가 응답이나 로그에 섞이지 않는가
- 사용자가 접근할 수 없는 리소스가 노출되지 않는가

## 언제 유용한가

MCP는 도구가 많고 재사용이 필요한 상황에서 유리하다. 단일 프로젝트 내부 함수 몇 개만 호출한다면 일반 Tool Calling으로 충분할 수 있다. 반대로 여러 데이터 소스와 외부 시스템을 붙이고, 같은 도구를 여러 LLM 클라이언트에서 써야 한다면 MCP 구조가 더 적합하다.

## 주의점

- 표준 프로토콜을 쓴다고 보안이 자동으로 해결되지는 않는다.
- 도구 검색이 쉬워질수록 잘못된 도구 호출 가능성도 커진다.
- 쓰기 권한이 있는 도구는 승인 단계와 감사 로그가 필요하다.
- MCP 서버가 반환하는 리소스 내용도 prompt injection 입력이 될 수 있다.
- 서버별 권한, 네트워크 접근 범위, 로그 정책을 분리해야 한다.

## 관련 글

- [Tool Calling Fundamentals & MCP]({% post_url 2026-01-29-upstage-course-w06d02-tool-calling-mcp %})
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})
- [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %})
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})

## 참고 자료

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)
