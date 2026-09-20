---
title: "Agentic RAG"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- rag
- tech-note
- agentic-workflow
toc: true
date: 2026-01-30 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Agentic RAG

> **한줄 정의**
> 에이전트가 자율적으로 검색 전략을 결정하고 실행하는 RAG. 단순 파이프라인을 넘어 동적이고 반복적인 검색이 가능하다.

## 학습 맥락

Agentic RAG는 W05의 기본 RAG와 Advanced RAG를 배운 뒤, W06D03에서 Agentic Workflow와 결합해 다룬 주제다. 기본 RAG가 "질문이 들어오면 정해진 방식으로 검색하고 답변한다"는 파이프라인이라면, Agentic RAG는 검색 자체를 에이전트의 판단 대상으로 둔다.

AI Agent 서비스에서는 모든 질문에 같은 검색 전략을 쓰기 어렵다. 어떤 질문은 바로 답할 수 있고, 어떤 질문은 벡터 DB를 찾아야 하며, 어떤 질문은 웹 검색이나 SQL 조회가 필요할 수 있다. Agentic RAG는 이 판단을 에이전트 워크플로우 안에 넣는 방식이다.

## 핵심 개념

Agentic RAG에서 에이전트는 사용자 질의를 분석하여 언제, 무엇을, 어떻게 검색할지를 스스로 결정한다. 단일 검색으로 부족한 경우 추가 검색을 수행하고, 여러 소스를 조합하며, 검색 결과를 평가하여 재검색 여부를 판단한다.

**Reflexion** 패턴은 에이전트가 자신의 답변을 자기 평가하고 개선하는 메커니즘이다. 검색 결과가 불충분하면 질의를 수정하여 재검색하고, 답변의 품질이 낮으면 추가 컨텍스트를 수집한다. **도구 통합**으로 벡터 DB, 웹 검색, SQL 쿼리 등 다양한 검색 소스를 동적으로 선택할 수 있다.

## 기본 RAG와의 차이

| 기준 | 기본 RAG | Agentic RAG |
| --- | --- | --- |
| 검색 여부 | 보통 항상 검색 | 질문에 따라 판단 |
| 검색 전략 | 고정된 검색 방식 | 쿼리 재작성, 소스 선택, 재검색 가능 |
| 흐름 | 검색 -> 생성 | 판단 -> 검색 -> 평가 -> 재검색/생성 |
| 장점 | 단순하고 안정적 | 복잡한 질문에 유연 |
| 단점 | 예외 처리에 약함 | 비용, 지연, 디버깅 부담 증가 |

## 대표 패턴

### Router 패턴

질문 유형에 따라 검색 소스를 나눈다. 예를 들어 과정 노트 질문은 로컬 노트에서 찾고, 최신 API 문서는 공식 문서에서 찾는 방식이다.

### Multi-step 검색

복잡한 질문을 여러 하위 질문으로 나눈 뒤 순차적으로 검색한다. "AgentOps와 Observability를 비교하고 프로젝트에 적용 기준을 정리해줘" 같은 질문은 한 번의 검색보다 단계적 검색이 적합하다.

### Adaptive 검색

검색 결과가 부족하거나 관련성이 낮으면 쿼리를 바꾸거나 다른 소스로 넘어간다. 이때 검색 결과 평가 기준이 없으면 재검색이 무한히 늘어날 수 있으므로 종료 조건이 필요하다.

## 구현 흐름

```text
사용자 질문
  -> 검색 필요성 판단
  -> 검색 소스 선택
  -> 쿼리 생성 또는 재작성
  -> 검색 실행
  -> 근거 충분성 평가
  -> 답변 생성 또는 재검색
```

실제 구현에서는 각 단계를 노드로 분리하면 추적하기 쉽다. LangGraph를 쓰는 경우 `decide_retrieval`, `retrieve`, `grade_documents`, `rewrite_query`, `generate_answer` 같은 노드로 나눌 수 있다.

## 평가 기준

Agentic RAG의 품질은 답변만 보고 판단하기 어렵다. 검색을 왜 했는지, 어떤 문서를 근거로 선택했는지, 재검색이 필요한 상황이었는지를 함께 봐야 한다.

- 검색이 필요한 질문에서만 검색했는가
- 검색된 문서에 답변 근거가 포함되어 있는가
- 관련 없는 검색 결과를 걸러냈는가
- 재검색 횟수와 비용이 제한되어 있는가
- 답변이 검색 문서에 근거하고 있는가

## 주의점

- 검색 판단을 LLM에만 맡기면 중요한 질문에서 검색을 놓칠 수 있다.
- 재검색 루프에는 최대 횟수와 종료 조건이 필요하다.
- 외부 문서의 지시문을 그대로 따르면 indirect prompt injection 위험이 있다.
- 검색 결과를 메모리에 저장할 때는 오래된 정보와 민감 정보를 구분해야 한다.
- Agentic RAG는 기본 RAG보다 비용과 지연시간이 커질 수 있다.

## 관련 글

- [Agentic RAG & Memory Management]({% post_url 2026-01-30-upstage-course-w06d03-agentic-rag-memory %})
- [Agentic RAG + Memory 실습]({% post_url 2026-01-30-upstage-practice-agentic-rag-practice %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %})
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})

## 참고 자료

- [Agentic RAG (LangChain Blog)](https://blog.langchain.dev/agentic-rag-with-langgraph/)
