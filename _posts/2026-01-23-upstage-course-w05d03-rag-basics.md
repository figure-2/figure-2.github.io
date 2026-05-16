---
title: "RAG Fundamentals & Advanced RAG 1"
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
- course-note
- prompt-engineering-rag
toc: true
date: 2026-01-23 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# RAG Fundamentals & Advanced RAG 1

# 수업에서 처음 잡은 감

프롬프트 수업을 들을 때까지만 해도 "질문을 잘 쓰면 모델이 더 잘 답한다" 정도로 생각했다. 그런데 RAG 수업은 관점이 조금 달랐다.
모델에게 더 잘 물어보는 문제가 아니라, **모델이 모르는 자료를 어떻게 가져와서 읽게 할 것인가**의 문제였다.

처음에는 RAG를 "문서 검색해서 프롬프트에 붙이는 것" 정도로 이해했는데, 수업을 듣다 보니 그렇게 단순하게 보면 안 됐다. 검색된 문서가 이상하면 답변도 이상해지고, chunk를 잘못 자르면 정답 근거가 잘려 나가고, embedding 검색이 좋아 보여도 정확한 키워드는 놓칠 수 있다.

내가 이날 가져간 핵심은 이것이다.

```text
좋은 RAG = 좋은 LLM 답변 생성기가 아니라
좋은 근거를 찾고, 그 근거로만 답하게 만드는 구조
```

## RAG를 왜 쓰는가

LLM은 학습된 지식을 갖고 있지만, 그 지식은 학습 시점에 묶여 있다. 회사 내부 문서, 최신 정책, 특정 서비스 매뉴얼, 수업 자료 같은 것은 모델이 원래 알 수 없다.

그래서 RAG는 모델을 다시 학습시키는 대신 외부 자료를 검색해서 답변에 넣는다.

```text
질문
-> 관련 문서 검색
-> 검색된 문서를 context로 넣기
-> LLM이 그 context를 보고 답변 생성
```

여기까지는 쉬운데, 실제로 어려운 부분은 "관련 문서 검색"이다. 답변이 틀리면 보통 LLM이 이상하다고 생각하기 쉬운데, RAG에서는 먼저 검색 결과를 봐야 한다.

## 파이프라인을 쪼개서 봐야 함

수업에서는 RAG를 한 덩어리로 보지 않고 단계별로 나눠 봤다.

```text
문서 로드
  -> chunking
  -> embedding
  -> vector store 저장
  -> 질문 embedding
  -> 유사도 검색
  -> 검색 결과를 context로 넣기
  -> 답변 생성
```

이렇게 쪼개 놓으니까 어디서 문제가 생기는지 보기 쉬웠다.

예를 들어 답변이 틀렸을 때 바로 프롬프트를 고치기 전에 먼저 확인할 것이 있다.

- PDF에서 텍스트가 제대로 추출됐는가
- chunk가 너무 작거나 크지 않은가
- 질문과 관련 있는 chunk가 실제로 검색됐는가
- 검색된 chunk 안에 정답 근거가 있는가
- LLM이 검색된 근거를 무시하고 내부 지식으로 답하지 않았는가

이 순서로 보면 RAG 디버깅이 조금 현실적으로 느껴졌다.

## chunking에서 헷갈렸던 부분

처음에는 chunk를 작게 자르면 검색이 더 정밀해질 것 같았다. 그런데 너무 작게 자르면 문맥이 잘린다.

예를 들어 어떤 정책 문서에서 조건은 앞 문단에 있고 예외는 다음 문단에 있을 수 있다. chunk를 너무 작게 자르면 조건만 검색되고 예외는 빠질 수 있다. 반대로 chunk가 너무 크면 관련 없는 내용까지 같이 들어가서 LLM이 헷갈릴 수 있다.

그래서 chunking은 그냥 전처리가 아니라 RAG 품질을 정하는 중요한 설정이었다.

| 설정 | 내가 이해한 느낌 |
|---|---|
| 작은 chunk | 잘 맞는 문장을 찾기 쉽지만 문맥이 끊길 수 있음 |
| 큰 chunk | 문맥은 넓게 들어오지만 불필요한 내용도 같이 들어옴 |
| overlap | 경계에서 잘리는 문제를 줄이지만 중복이 늘어남 |

## sparse와 dense 검색

검색 방식도 단순히 "벡터 검색이 좋다"로 끝나지 않았다.

Sparse retrieval은 키워드 검색에 가깝다. 에러 메시지, 함수 이름, 정책 번호처럼 정확히 맞아야 하는 검색에 강하다.
Dense retrieval은 embedding 기반 검색이다. 표현이 달라도 의미가 비슷하면 찾을 수 있다.

예를 들어 `Docker port mapping error`처럼 정확한 문자열이 중요하면 sparse가 유리할 수 있고, "컨테이너가 외부에서 접속이 안 되는 문제"처럼 표현이 달라지는 질문은 dense가 유리할 수 있다.

그래서 나중에는 둘을 섞는 hybrid search가 필요해진다.

## 검색 방식 비교

| 방식 | 강점 | 약점 |
|---|---|---|
| Sparse Retrieval | 정확한 키워드, 에러 메시지, 고유명사 검색에 강함 | 표현이 다르면 못 찾을 수 있음 |
| Dense Retrieval | 의미가 비슷한 문서를 찾는 데 강함 | 정확한 문자열 매칭에는 약할 수 있음 |
| Hybrid Search | 키워드와 의미 검색을 함께 활용 | 구현과 튜닝 복잡도 증가 |

RAG 품질은 LLM 자체보다 retrieval 단계에서 먼저 갈리는 경우가 많다. 좋은 답변을 생성하려면 먼저 좋은 근거 청크가 검색되어야 한다.

## 실습에서 확인한 흐름

Day3 노트북은 Naive RAG 파이프라인을 직접 구성한다.

```text
PDF 문서 로드
  -> 페이지 단위 텍스트 추출
  -> chunking
  -> embedding
  -> Chroma vector store 저장
  -> manual cosine search / LangChain retriever 비교
  -> retrieved context로 답변 생성
```

여기서 좋았던 점은 LangChain retriever만 쓰고 끝내지 않고, manual cosine search도 같이 본다는 점이었다. 직접 유사도를 계산해보면 vector store가 내부에서 대충 어떤 일을 하는지 감이 온다.

실습의 핵심은 답변만 보는 것이 아니라 검색된 chunk를 같이 보는 것이다. 답변이 틀렸다면 생성 문제가 아니라 검색 문제일 수 있다. 앞으로 RAG 실습을 할 때는 질문마다 검색된 근거 chunk와 score를 같이 남기는 습관이 필요하다고 느꼈다.

## 오늘 정리

RAG는 "LLM에 문서를 붙이면 정확해진다"가 아니다.

오히려 확인할 것이 더 많아진다. 문서가 잘 읽혔는지, chunk가 적절한지, 검색 결과가 맞는지, 모델이 근거를 잘 따르는지까지 봐야 한다. 그래도 이 구조가 필요한 이유는 명확하다. 모델 내부 지식만으로는 최신 정보나 내부 자료를 다룰 수 없기 때문이다.

## 흐름도

![RAG Fundamentals & Advanced RAG 1 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-prompt-engineering-rag-w05d03-rag-basics-diagram-1.svg)

## 관련 글

- [RAG 기술 노트]({% post_url 2026-01-23-upstage-tech-rag %})
- [Naive RAG 파이프라인 구축 실습]({% post_url 2026-01-23-upstage-practice-rag-practice %})
- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %})
