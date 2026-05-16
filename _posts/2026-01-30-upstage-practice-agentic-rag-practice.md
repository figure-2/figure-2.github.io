---
title: "Agentic RAG + Memory 실습"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- PRACTICE
tags:
- upstage
- sesac
- ai-agent
- rag
- agent-architecture
- practice
- agentic-workflow
toc: true
date: 2026-01-30 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Agentic RAG + Memory 실습

> **실습 정보**
> - **주차**: Week 06, Day 03
> - **유형**: Jupyter Notebook
> - **상태**: 완료

## 실습 목표
Agentic RAG와 Memory Management를 실습한다. 고정된 RAG 파이프라인이 아니라, 에이전트가 검색 필요성, 검색 전략, 재검색 여부를 판단하는 흐름을 이해하는 것이 목표다. 또한 검색 결과와 대화 상태를 메모리로 관리하고, Reflexion 패턴으로 답변 품질을 점검하는 관점을 연결한다.

## 핵심 학습 포인트
- Agentic RAG: 에이전트가 자율적으로 검색 결정
- Memory Management: 단기/장기 메모리 구현
- Reflexion 패턴 활용

## 실습 맥락

이 실습은 W06D03 `Agentic RAG & Memory Management`와 연결된다. W05에서 배운 기본 RAG는 검색과 생성 흐름이 고정되어 있었고, W06에서는 이 흐름에 에이전트의 판단을 추가한다.

예를 들어 기본 RAG는 사용자가 질문하면 항상 검색을 수행한다. Agentic RAG는 질문의 성격을 먼저 보고, 검색이 필요한지, 어떤 쿼리로 검색할지, 결과가 부족하면 다시 검색할지 결정한다. 이 차이가 실제 AI 서비스에서 중요하다.

## 진행 순서

1. 사용자 질문을 입력으로 받는다.
2. 검색이 필요한 질문인지 판단한다.
3. 검색이 필요하면 쿼리를 생성하거나 재작성한다.
4. 검색 결과를 평가한다.
5. 결과가 부족하면 재검색하거나 다른 전략을 선택한다.
6. 답변을 생성한다.
7. 답변 품질을 점검하고 필요한 경우 Reflexion 결과를 메모리에 남긴다.

```text
Question
  -> Need retrieval?
  -> Retrieve
  -> Evaluate evidence
  -> Generate answer
  -> Reflect
  -> Update memory
```

## 구현 관점

실습 코드를 작성할 때는 에이전트가 판단하는 지점을 명시적으로 나누는 것이 좋다. 그래야 나중에 왜 검색했는지, 왜 재검색했는지, 어떤 메모리를 사용했는지 추적할 수 있다.

| 단계 | 확인할 것 |
| --- | --- |
| Retrieval decision | 검색이 필요한 질문인지 |
| Query generation | 검색어가 충분히 구체적인지 |
| Evidence check | 검색 결과가 답변 근거로 충분한지 |
| Memory read | 이전 대화나 장기 메모리가 필요한지 |
| Memory write | 새로 저장할 가치가 있는 정보인지 |
| Reflection | 답변이 질문에 맞고 근거가 있는지 |

## 실패 사례 관점

Agentic RAG 실습에서 중요한 것은 성공 경로만 보는 것이 아니다. 검색 결과가 없거나, 검색 결과가 질문과 맞지 않거나, 메모리에 오래된 정보가 남아 있을 때 어떻게 처리할지까지 확인해야 한다.

- 검색 결과가 없으면 답변을 억지로 만들지 않는다.
- 검색 결과가 애매하면 재검색하거나 사용자에게 범위를 다시 묻는다.
- 메모리에 저장할 정보와 저장하지 않을 정보를 구분한다.
- Reflexion은 "답변을 길게 고치는 기능"이 아니라 근거와 품질을 점검하는 단계로 둔다.

## 체크포인트

- [ ] 기본 RAG와 Agentic RAG의 차이를 설명할 수 있다.
- [ ] 검색 필요성 판단 기준을 코드나 흐름으로 분리했다.
- [ ] 검색 결과가 부족할 때의 처리 방식을 정했다.
- [ ] 단기 메모리와 장기 메모리의 역할을 구분했다.
- [ ] Reflexion 결과가 다음 실행에 어떻게 영향을 주는지 설명할 수 있다.
- [ ] 실행 결과, 답안 근거, 회고 중 하나 이상을 남겼다.

## 회고 질문

- 이번 실습에서 가장 헷갈린 개념은 무엇이었나?
- 수업 노트만 읽을 때와 직접 실습할 때 다르게 느껴진 점은 무엇인가?
- 같은 유형의 문제를 다시 만났을 때 먼저 확인할 기준은 무엇인가?
- 검색을 항상 수행하는 구조와 필요할 때만 수행하는 구조 중 어떤 차이가 있었나?
- 메모리를 저장하면 좋아지는 부분과 위험해지는 부분은 무엇인가?

## 관련 글
- [Agentic RAG & Memory Management]({% post_url 2026-01-30-upstage-course-w06d03-agentic-rag-memory %})
- [Agentic RAG]({% post_url 2026-01-30-upstage-tech-agentic-rag %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})
