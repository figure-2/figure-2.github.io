---
title: "Advanced RAG 2 & LLM 보안"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- rag
- llm-security
- course-note
- prompt-engineering-rag
toc: true
date: 2026-01-26 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Advanced RAG 2 & LLM 보안

# Naive RAG 다음에 바로 막히는 지점

지난 수업에서 Naive RAG를 만들었을 때는 흐름이 꽤 단순해 보였다.

```text
문서 자르기 -> embedding -> 가까운 chunk 찾기 -> LLM에 넣기
```

그런데 오늘 수업은 이 방식이 실제 서비스에서는 금방 한계가 온다는 내용이었다.
질문이 애매하면 검색이 안 되고, chunk가 이상하면 근거가 잘리고, 검색 결과가 여러 개면 서로 충돌할 수 있다. 여기에 prompt injection까지 들어오면 RAG 문서가 "근거"가 아니라 "공격 입력"이 될 수도 있다.

이날 수업을 들으면서 가장 크게 바뀐 생각은 이것이다.

```text
RAG를 붙였다고 신뢰성이 생기는 게 아니라,
RAG의 각 단계를 검증해야 신뢰성이 생긴다.
```

## Advanced RAG는 무엇을 고치는가

Advanced RAG라고 해서 완전히 새로운 구조라기보다는, Naive RAG에서 실패하는 지점을 하나씩 보정하는 방법에 가깝다.

- 질문이 검색에 안 맞으면 query transformation
- chunk가 너무 작거나 크면 Small2Big, recursive retrieval
- 검색 결과 순서가 애매하면 reranking
- 질문 유형이 다르면 RAGRouter
- 문서끼리 말이 다르면 knowledge conflict handling
- 외부 문서가 공격 입력일 수 있으면 LLM security

수업에서 나온 기법 이름은 많았지만, 결국은 "어느 단계가 약한가"를 보고 붙이는 도구라고 이해했다.

## 수업에서 나온 주요 기법

### 1. Query Transformation

HyDE는 처음 들었을 때 이름보다 아이디어가 더 중요했다. 사용자의 질문은 짧고 애매한데, 문서는 보통 설명문 형태다. 그러면 질문과 문서의 모양이 너무 다르다.

HyDE는 질문을 바로 embedding하지 않고, 먼저 "이 질문에 대한 가상의 답변 문서"를 만든 뒤 그 문서로 검색한다. 질문을 문서에 더 가까운 형태로 바꾸는 셈이다.

Reverse HyDE는 반대로 문서 쪽에서 가상의 질문을 만들어 검색에 활용한다. 결국 둘 다 질문과 문서 사이의 표현 차이를 줄이려는 방법이다.

### 2. Chunking / Retrieval 고도화

Small2Big은 꽤 직관적이었다. 검색할 때는 작은 chunk가 유리한데, 답변할 때는 큰 문맥이 필요하다. 그래서 작은 chunk로 정확히 찾고, 실제 답변에는 그 chunk가 속한 더 큰 parent context를 넣는다.

Reranking은 1차 검색 결과를 다시 정렬하는 단계다. 처음부터 완벽한 검색을 기대하지 않고, 후보를 넉넉히 가져온 뒤 더 비싼 방식으로 다시 고르는 느낌이다.

RAGRouter는 질문 유형에 따라 검색 경로나 모델을 고르는 방식이다. 정의 질문, 표 기반 질문, 코드 에러 질문을 같은 방식으로 처리하면 당연히 한계가 생긴다.

### 3. Advanced RAG 심화

Self-RAG와 CRAG는 "검색 결과를 믿어도 되는지"를 한 번 더 보는 쪽에 가깝다.
Graph RAG와 RAPTOR는 문서가 길거나 관계가 복잡할 때 필요해지는 방식으로 이해했다.

다만 이 부분은 아직 이름을 외웠다기보다, 어떤 상황에서 필요한지 정도만 잡았다. 실제 프로젝트에서 바로 다 넣기보다는 baseline RAG가 어디서 실패하는지 먼저 봐야 할 것 같다.

### 4. Knowledge Conflict

오늘 제일 중요하게 느낀 부분이다.

RAG를 쓰면 외부 문서를 넣으니까 더 정확해질 것 같지만, 외부 문서가 틀렸거나 오래됐거나 서로 충돌하면 문제가 더 복잡해진다.

- Context-Memory Conflict: 모델 내부 지식과 검색 문서가 다름
- Inter-Context Conflict: 검색된 문서끼리 서로 다름

이때는 "검색된 문서를 무조건 따른다"도 위험하고, "모델이 아는 대로 답한다"도 위험하다. 출처, 최신성, 공식성 같은 우선순위가 필요하다.

### 5. RAG 평가

평가는 검색과 생성을 나눠서 봐야 한다.

- 검색이 잘 됐는가: 관련 chunk가 상위에 있는가
- 생성이 잘 됐는가: 답변이 검색 근거를 따르는가
- 충돌이 있을 때: 출처 우선순위를 제대로 적용했는가

### 6. Prompt Injection 공격

보안 파트에서는 prompt injection이 RAG와 바로 연결됐다. 사용자가 직접 "이전 지시를 무시해"라고 입력하는 것도 문제지만, 더 위험한 건 문서 안에 그런 지시가 숨어 있는 경우다.

RAG 문서는 외부 입력이다. 그러니까 문서 안의 문장을 모델이 명령처럼 받아들이면 안 된다.

### 7. Jailbreaking

Jailbreaking은 모델의 안전 가드레일을 우회하려는 시도다. 역할극이나 가상 상황을 이용해 제한을 풀게 만드는 방식들이 여기에 들어간다.

이 부분은 결국 "프롬프트만 강하게 쓰면 막을 수 있다"가 아니라, 입력 검증과 출력 필터링, 권한 제한까지 같이 가야 한다고 이해했다.

### 8. 안전성 검증과 방어 전략

방어는 한 줄짜리 시스템 프롬프트로 끝나지 않는다.

```text
입력 검증
  -> 검색 문서 권한 확인
  -> context와 system instruction 분리
  -> 출력 검토
  -> 로그 마스킹
```

LLM 보안은 애플리케이션 구조와 같이 봐야 한다.

## 실습 연결

Day4 노트북은 Knowledge Conflict를 직접 다룬다.

```text
LLM 내부 답변 생성
  -> 외부 문서와 비교
  -> 상충 문서 세트 구성
  -> chunking / embedding / vector store
  -> conflict detection
  -> conflict-aware answer 생성
  -> source priority 정책 검토
```

이 실습에서 중요한 것은 "RAG를 붙이면 항상 맞아진다"는 가정을 버리는 것이다. 검색된 문서가 틀렸거나 서로 충돌하면, LLM은 내부 지식과 외부 근거 사이에서 흔들릴 수 있다.

## 오늘 정리

Advanced RAG는 기능 이름을 많이 외우는 수업처럼 보일 수 있지만, 실제 핵심은 실패 지점을 분리하는 것이다.

Naive RAG가 실패했을 때 "Advanced RAG를 붙이자"가 아니라,

```text
질문 문제인가?
chunk 문제인가?
검색 문제인가?
정렬 문제인가?
문서 충돌 문제인가?
보안 문제인가?
```

이렇게 먼저 나눠 봐야 한다. 이 기준이 있어야 HyDE든 reranking이든 Graph RAG든 필요한 곳에 쓸 수 있다.

## 관련 글

- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %})
- [RAG Knowledge Conflict 실습]({% post_url 2026-01-26-upstage-practice-rag-conflict-practice %})
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})
- [Agentic RAG]({% post_url 2026-01-30-upstage-tech-agentic-rag %})
