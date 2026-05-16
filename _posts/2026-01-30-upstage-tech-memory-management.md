---
title: "Memory Management"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- agent-architecture
- tech-note
- agentic-workflow
toc: true
date: 2026-01-30 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Memory Management

> **한줄 정의**
> LLM 에이전트가 대화 맥락과 과거 경험을 저장·검색하는 메커니즘. 상태 없는 LLM에 지속성을 부여한다.

## 학습 맥락

Memory Management는 W06D03 `Agentic RAG & Memory Management`에서 Agentic RAG와 함께 다뤘다. 기본 LLM 호출은 상태가 없다. 이전 대화를 다시 넣지 않으면 모델은 과거 맥락을 기억하지 못한다. 에이전트 서비스에서 메모리는 이 한계를 보완하기 위한 설계 요소다.

다만 메모리는 많이 저장할수록 좋은 기능이 아니다. 무엇을 단기 컨텍스트에 둘지, 무엇을 장기 저장소에 남길지, 어떤 정보를 다시 검색할지 결정해야 한다. 잘못된 메모리는 답변 품질을 떨어뜨리고, 민감 정보 저장 위험도 만든다.

## 핵심 개념

에이전트 메모리는 네 가지 유형으로 구분된다. **단기 메모리(Short-term)**는 현재 대화의 컨텍스트 윈도우 내 이력이고, **장기 메모리(Long-term)**는 벡터 DB에 저장된 과거 대화 및 지식이다. **에피소드 메모리(Episodic)**는 특정 경험과 사건의 기록이고, **의미 메모리(Semantic)**는 도메인 지식과 사실 정보다.

**Conversation Buffer Memory**는 모든 대화를 그대로 저장하여 간단하지만 토큰 소모가 크다. **Summary Memory**는 대화를 주기적으로 요약하여 토큰을 절약한다. **벡터 메모리**는 임베딩으로 저장하고 관련성 기반으로 검색하여 장기 메모리를 효율적으로 관리한다. idol-agent에서 Supabase를 장기 메모리 저장소로 활용한다.

## 메모리 유형 분류

![Memory Management 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-memory-management-diagram-1.svg)

| 메모리 유형 | 설명 | 예시 |
| --- | --- | --- |
| Short-term | 현재 대화에서 유지하는 최근 맥락 | 최근 사용자 질문, 직전 답변 |
| Working Memory | 현재 작업을 수행하기 위한 중간 상태 | 검색 결과, 계획, 실행 상태 |
| Long-term | 이후에도 재사용할 수 있는 정보 | 사용자 선호, 프로젝트 지식 |
| Episodic | 특정 사건이나 실행 경험 | 과거 실패 사례, 선택한 해결 방법 |
| Semantic | 일반화된 지식과 사실 | 도메인 개념, 문서화된 규칙 |

## 저장 방식

메모리는 저장 방식에 따라 장단점이 다르다.

```text
Conversation Buffer
  - 장점: 구현이 쉽고 맥락 손실이 적음
  - 단점: 토큰 비용 증가

Summary Memory
  - 장점: 긴 대화를 압축 가능
  - 단점: 요약 과정에서 정보 손실 가능

Vector Memory
  - 장점: 장기 기억을 관련성 기반으로 검색 가능
  - 단점: 잘못된 검색 결과가 컨텍스트에 들어올 수 있음

Checkpoint / State
  - 장점: 그래프 실행 상태를 복원 가능
  - 단점: 상태 schema와 version 관리 필요
```

## 구현 관점

에이전트에 메모리를 넣을 때는 먼저 저장 정책을 정해야 한다. 모든 대화를 저장하면 간단하지만, 비용과 개인정보 문제가 생긴다. 반대로 아무것도 저장하지 않으면 사용자는 매번 같은 정보를 다시 말해야 한다.

실제 구현 기준은 다음처럼 나눌 수 있다.

1. 현재 요청 처리에 필요한 정보는 working memory에 둔다.
2. 다음 턴에서 바로 필요한 대화는 short-term memory로 유지한다.
3. 반복적으로 쓸 사용자 선호나 프로젝트 정보만 long-term memory로 저장한다.
4. 실패 원인이나 해결 전략은 episodic memory로 남길 수 있다.
5. 저장 전 민감 정보와 일회성 정보를 필터링한다.

## Agentic RAG와의 관계

Agentic RAG는 검색 전략을 에이전트가 판단하고, Memory Management는 그 검색 결과와 대화 맥락을 어떻게 유지할지 결정한다. 둘을 함께 쓰면 에이전트가 과거 대화와 외부 지식을 모두 참고할 수 있다.

하지만 메모리는 검색 결과보다 더 조심해야 한다. 사용자가 예전에 말한 내용이 지금도 맞는지, 저장해도 되는 정보인지, 다른 사용자에게 노출될 위험은 없는지 확인해야 한다.

## 주의점

- 사용자의 민감정보를 장기 메모리에 저장하지 않는다.
- 오래된 메모리는 현재 사실과 충돌할 수 있다.
- 요약 메모리는 정보 손실과 왜곡이 발생할 수 있다.
- 벡터 메모리는 관련 없는 기억을 검색할 수 있으므로 근거 확인이 필요하다.
- 메모리 삭제, 수정, 만료 정책이 없으면 운영 리스크가 커진다.

## 관련 글

- [Agentic RAG & Memory Management]({% post_url 2026-01-30-upstage-course-w06d03-agentic-rag-memory %})
- [Agentic RAG + Memory 실습]({% post_url 2026-01-30-upstage-practice-agentic-rag-practice %})
- [Agentic RAG]({% post_url 2026-01-30-upstage-tech-agentic-rag %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [Supabase]({% post_url 2026-02-10-upstage-tech-supabase %})
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %})
