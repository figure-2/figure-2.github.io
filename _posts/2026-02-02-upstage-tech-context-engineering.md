---
title: "Context Engineering"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- context-engineering
- tech-note
- agentic-workflow
toc: true
date: 2026-02-02 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Context Engineering

> **한줄 정의**
> LLM의 컨텍스트 윈도우를 최적으로 활용하기 위한 정보 설계 기법. 제한된 토큰 예산 내에서 최대 품질의 출력을 이끌어낸다.

## 학습 맥락

Context Engineering은 W06D04 `Context Engineering & Safety`에서 다뤘다. 프롬프트 엔지니어링이 "어떻게 지시할 것인가"에 가깝다면, Context Engineering은 "모델이 판단에 사용할 정보를 어떻게 구성할 것인가"에 가깝다.

Agentic Workflow에서는 컨텍스트가 더 복잡해진다. 사용자 입력뿐 아니라 시스템 지시, 대화 이력, RAG 검색 결과, 도구 호출 결과, 메모리, 정책 문서가 모두 모델 입력에 들어올 수 있다. 이 정보들을 무작정 많이 넣는 것이 아니라, 목적에 맞게 선별하고 순서를 정하고 신뢰도를 구분해야 한다.

## 핵심 개념

컨텍스트 윈도우는 LLM이 한 번에 처리할 수 있는 입력 범위다. 컨텍스트가 길어지면 더 많은 정보를 넣을 수 있지만, 중요 정보가 중간에 묻히거나 불필요한 정보 때문에 답변 품질이 떨어질 수 있다. 이를 흔히 **Lost in the Middle** 문제로 설명한다.

**정보 밀도(Information Density)** 최적화는 중복되거나 불필요한 정보를 제거하여 토큰을 효율적으로 사용한다. 시스템 프롬프트는 간결하되 완전해야 하며, RAG로 검색된 컨텍스트는 질의와 관련성 높은 청크만 포함해야 한다. 대화 이력은 요약하거나 슬라이딩 윈도우로 관리한다.

## 컨텍스트 구성 요소

| 구성 요소 | 예시 | 주의점 |
| --- | --- | --- |
| System instruction | 역할, 정책, 출력 형식 | 너무 길면 핵심 지시가 흐려짐 |
| User input | 현재 질문, 요구사항 | prompt injection 가능성 |
| Conversation history | 이전 대화 | 오래된 맥락과 충돌 가능 |
| Retrieved context | RAG 검색 문서 | 관련성, 권한, 오염 여부 확인 |
| Tool result | API/DB/파일 조회 결과 | 결과 검증과 에러 처리 필요 |
| Memory | 사용자 선호, 과거 작업 | 저장/삭제/만료 정책 필요 |

## 설계 기준

Context Engineering은 컨텍스트를 많이 넣는 기술이 아니라, 필요한 정보를 올바른 위치와 형식으로 넣는 기술이다. 다음 기준을 적용할 수 있다.

- 현재 질문과 직접 관련 없는 정보는 제거한다.
- 권위가 다른 정보는 구분한다. 시스템 지시, 사용자 입력, 검색 문서는 같은 신뢰도가 아니다.
- 검색 문서는 원문 전체보다 필요한 chunk와 출처 중심으로 넣는다.
- 대화 이력은 최신성과 중요도를 기준으로 요약하거나 잘라낸다.
- 도구 결과는 성공/실패 상태와 함께 전달한다.
- 민감정보와 secret은 컨텍스트에 넣지 않는다.

## Agent 서비스에서의 흐름

```text
사용자 요청
  -> 의도와 필요한 정보 판단
  -> 대화 이력 압축
  -> 필요한 메모리 검색
  -> RAG 문서 검색
  -> 도구 결과 병합
  -> 신뢰도와 우선순위 정리
  -> 모델 호출
```

이 흐름에서 중요한 것은 정보의 출처를 섞지 않는 것이다. 사용자 입력은 요청이고, 검색 문서는 참고 자료이며, 시스템 지시는 정책이다. 세 가지를 같은 위치에 같은 형식으로 넣으면 모델이 잘못된 지시를 따를 수 있다.

## Safety와의 관계

Context Engineering은 보안과 직접 연결된다. RAG 문서나 웹페이지 안에 "이전 지시를 무시하라" 같은 문장이 들어 있어도, 그것은 명령이 아니라 데이터로 취급해야 한다. 컨텍스트를 구성할 때 외부 데이터와 시스템 지시를 분리하는 이유가 여기에 있다.

## 주의점

- 긴 컨텍스트가 항상 더 좋은 답변을 보장하지 않는다.
- 검색 결과를 많이 넣으면 관련 없는 문서가 답변을 오염시킬 수 있다.
- 대화 이력을 그대로 넣으면 개인정보와 비용 문제가 생긴다.
- 요약된 이력은 누락과 왜곡 가능성이 있다.
- 외부 문서의 지시문은 명령이 아니라 데이터로 다뤄야 한다.

## 관련 글

- [Context Engineering & Safety]({% post_url 2026-02-02-upstage-course-w06d04-context-engineering-safety %})
- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %})

## 참고 자료

- [Context Engineering (Simon Willison)](https://simonwillison.net/2025/Jun/27/context-engineering/)
