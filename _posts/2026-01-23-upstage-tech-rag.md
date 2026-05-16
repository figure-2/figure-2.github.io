---
title: "RAG (Retrieval-Augmented Generation)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- rag
- tech-note
- prompt-engineering-rag
toc: true
date: 2026-01-23 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# RAG (Retrieval-Augmented Generation)

RAG는 처음 보면 "검색해서 LLM에 넣는 것"으로 끝나는 개념처럼 보인다. 실제로 큰 흐름은 맞다. 그런데 공부하면서 느낀 건, RAG의 어려움은 생성이 아니라 **검색된 근거를 믿을 수 있게 만드는 과정**에 있다는 점이다.

## 내가 이해한 RAG

RAG는 LLM이 원래 알고 있는 지식만 쓰지 않고, 외부 문서를 찾아서 같이 읽게 만드는 방식이다.

```text
질문
-> 관련 문서 검색
-> 검색된 문서를 context로 구성
-> LLM 답변 생성
```

LLM이 학습하면서 가진 지식은 parametric knowledge라고 볼 수 있고, RAG가 가져오는 외부 문서는 non-parametric knowledge라고 볼 수 있다. 회사 내부 문서나 최신 자료처럼 모델이 모르는 내용을 다루려면 이 외부 지식이 필요하다.

벡터 데이터베이스(Supabase pgvector, Pinecone, Chroma 등)는 RAG의 핵심 저장소다. 문서를 청크(Chunk) 단위로 분할하고 임베딩 모델로 벡터화하여 저장한다. 검색 시에는 코사인 유사도나 내적으로 가장 관련성 높은 청크를 반환한다.

여기서 주의할 점은 "vector DB에 넣었으니 알아서 잘 찾겠지"라고 생각하면 안 된다는 것이다. 청크 크기, overlap, embedding 모델, query 표현에 따라 검색 결과가 꽤 달라진다.

## 기본 파이프라인

```text
문서 수집
  -> 문서 파싱
  -> chunking
  -> embedding
  -> vector store 저장
  -> query embedding
  -> retrieval
  -> context formatting
  -> LLM answer generation
```

이 중 하나만 실패해도 최종 답변은 흔들린다. 특히 RAG를 디버깅할 때는 답변 문장보다 검색된 chunk를 먼저 봐야 한다.

## 검색 방식

### Sparse Retrieval

BM25 같은 키워드 기반 검색이다. 정확한 용어, 코드명, 에러 메시지, 정책 번호처럼 문자열 일치가 중요한 경우에 강하다. 대신 표현이 바뀌면 관련 문서를 놓칠 수 있다.

### Dense Retrieval

Embedding 기반 검색이다. 표현이 달라도 의미가 가까운 문서를 찾을 수 있다. 예를 들어 "비용 추적"과 "사용량 과금 관리"처럼 단어가 달라도 의미가 비슷하면 찾을 수 있다. 반면 정확한 문자열 매칭에는 약할 수 있다.

### Hybrid Retrieval

그래서 실서비스에서는 둘을 섞는 경우가 많다. 키워드 검색으로 정확한 후보를 확보하고, embedding 검색으로 의미적 후보를 보완한 뒤, 필요하면 reranker로 다시 정렬한다.

## chunking에서 봐야 할 것

chunking은 단순 전처리가 아니다. "모델에게 어떤 단위의 근거를 보여줄지"를 정하는 일이다.

| 판단 기준 | 설명 |
| --- | --- |
| 의미 단위 | 문단, 섹션, 표처럼 자연스러운 단위로 잘리는가 |
| chunk size | 답변에 필요한 맥락을 담을 만큼 충분한가 |
| overlap | 경계에서 정보가 끊기지 않는가 |
| metadata | 출처, 날짜, 섹션, 권한 정보를 함께 저장하는가 |

문서가 작을 때는 단순 문자 수 기반 chunking으로도 시작할 수 있지만, 표와 정책 문서가 많아지면 구조 기반 chunking이 필요하다.

## 구현할 때 남겨야 할 로그

Day3 실습에서 좋았던 점은 검색 결과를 눈으로 확인했다는 점이다. RAG는 최종 답변만 저장하면 디버깅이 어렵다. 최소한 아래 정도는 같이 남기는 게 좋다.

- 질문 원문
- 검색된 chunk id와 source metadata
- 상위 chunk의 score
- LLM에 실제로 들어간 context
- 최종 답변과 근거 문서

## 실패할 때 먼저 볼 것

| 실패 지점 | 증상 | 먼저 확인할 것 |
| --- | --- | --- |
| parsing 실패 | 문서 텍스트가 비어 있거나 깨짐 | PDF 추출 결과 |
| chunking 실패 | 답변 근거가 여러 chunk에 흩어짐 | chunk size, overlap |
| retrieval 실패 | 관련 없는 문서가 상위에 나옴 | embedding 모델, query |
| generation 실패 | 좋은 근거가 있는데 답이 틀림 | prompt, context order |
| conflict 실패 | 문서끼리 다른 값을 말함 | 출처 우선순위, 최신성 |

## 내가 가져간 기준

RAG는 환각을 줄일 수 있지만, 자동으로 정확성을 보장하지는 않는다. 검색 문서 자체가 오래됐거나 틀릴 수 있고, 문서 안의 지시문이 prompt injection처럼 작동할 수도 있다.

그래서 앞으로 RAG를 만들 때는 아래 순서로 볼 생각이다.

```text
문서가 제대로 읽혔는가
-> 검색 결과에 근거가 있는가
-> 답변이 그 근거를 따르는가
-> 근거끼리 충돌하지 않는가
```

## 관련 강의

- W05D03-RAG-기초
- W05D04-Advanced-RAG-보안
- W06D03-Agentic-RAG-Memory

## 아키텍처 다이어그램

![RAG (Retrieval-Augmented Generation) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-rag-diagram-1.svg)

## 관련 개념

- Advanced-RAG - 쿼리 재작성, 재순위화 등 고급 기법
- Agentic-RAG - 에이전트 기반 자율적 RAG
- Prompt-Engineering - RAG 컨텍스트 주입 설계
- Supabase - pgvector 기반 벡터 저장소
- LangGraph - RAG 파이프라인 오케스트레이션

## 관련 글

- [RAG Fundamentals & Advanced RAG 1]({% post_url 2026-01-23-upstage-course-w05d03-rag-basics %})
- [Naive RAG 파이프라인 구축 실습]({% post_url 2026-01-23-upstage-practice-rag-practice %})
- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %})
- [RAG Knowledge Conflict 실습]({% post_url 2026-01-26-upstage-practice-rag-conflict-practice %})

## 참고 자료

- [Retrieval-Augmented Generation for Large Language Models (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
