---
title: "Naive RAG 파이프라인 구축 실습"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- PRACTICE
tags:
- upstage
- sesac
- ai-agent
- rag
- practice
- prompt-engineering-rag
toc: true
date: 2026-01-23 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Naive RAG 파이프라인 구축 실습

Day3 실습은 PDF 하나를 가지고 직접 RAG 파이프라인을 만들어보는 내용이었다. 처음에는 LangChain 코드만 따라가면 되는 실습이라고 생각했는데, 막상 정리해보니 핵심은 코드보다 **중간 결과를 계속 확인하는 습관**이었다.

## 실습에서 한 일

PDF 문서를 읽고, chunk로 나누고, embedding을 만들고, Chroma에 넣은 다음 질문과 가까운 chunk를 검색했다. 마지막에는 검색된 내용을 context로 넣어 LLM이 답하게 했다.

흐름만 보면 간단하다.

```text
PDF
-> text
-> chunk
-> embedding
-> vector store
-> retrieval
-> answer
```

그런데 이 중 하나만 틀어져도 최종 답변은 틀어진다. 그래서 실습하면서 "답변이 맞는가"보다 "검색된 근거가 맞는가"를 먼저 봐야 한다는 생각이 들었다.

## 1. PDF 로딩

먼저 PDF를 읽어서 페이지 단위 텍스트로 바꿨다. 이 단계는 별것 아닌 것처럼 보이지만, 실제로는 제일 먼저 확인해야 하는 부분이다.

PDF에서 텍스트가 제대로 안 뽑히면 embedding도 검색도 전부 의미가 없다. 특히 표나 이미지가 많은 문서는 그냥 loader로 읽었을 때 깨질 수 있다. 그래서 문서를 불러온 뒤에는 바로 다음 단계로 가지 말고 일부 텍스트를 직접 찍어보는 것이 필요하다.

```text
PDF 파일 -> page documents -> text preview
```

## 2. chunking

문서를 어느 크기로 자를지가 생각보다 중요했다. 처음에는 chunk size를 적당히 정하면 된다고 생각했는데, 이 값이 검색 품질에 바로 영향을 준다.

작게 자르면 검색은 정밀해질 수 있지만 문맥이 끊긴다. 크게 자르면 문맥은 남지만 관련 없는 내용이 같이 들어온다.

| 설정 | 영향 |
| --- | --- |
| chunk size 큼 | 문맥은 유지되지만 noise가 늘 수 있음 |
| chunk size 작음 | 검색 단위는 정밀하지만 맥락이 끊길 수 있음 |
| overlap 큼 | 경계 손실은 줄지만 중복 검색 가능성이 커짐 |

결국 chunking은 "문서를 잘라 저장하는 작업"이 아니라, 모델에게 어떤 단위의 근거를 보여줄지 정하는 작업이었다.

## 3. embedding과 vector store

각 chunk를 embedding으로 바꾸고 Chroma에 저장했다. 여기서 좋았던 점은 retriever만 쓰고 넘어가지 않고, cosine similarity를 직접 계산하는 흐름도 같이 봤다는 점이다.

이걸 해보면 vector store가 완전히 새로운 마법이 아니라, 결국 질문 벡터와 문서 벡터의 거리를 비교하는 구조라는 게 보인다.

## 4. retrieval 결과 먼저 보기

질문을 넣으면 관련 chunk가 검색된다. 그런데 이때 바로 LLM에 넘기면 안 된다. 사람이 먼저 봐야 한다.

```text
질문 -> 검색된 chunk 목록 -> 정답 근거 포함 여부 확인 -> 프롬프트 구성
```

검색 결과에 정답 근거가 없는데 LLM이 답을 맞히면, 그건 RAG가 잘된 것이 아니라 모델 내부 지식으로 맞힌 것일 수 있다. 반대로 검색 결과는 맞는데 답변이 틀리면 그때는 프롬프트나 생성 단계 문제를 보면 된다.

## 5. context 넣고 답변 생성

검색된 chunk를 context로 묶고, 질문과 함께 LLM에 전달한다. 프롬프트에는 "제공된 context에 근거해서 답하라"는 제약과 "근거가 부족하면 모른다고 말하라"는 조건을 넣는 것이 좋다.

이 부분에서 조심할 점은 context를 명령처럼 취급하지 않는 것이다. RAG 문서는 참고 자료이지 시스템 지시가 아니다. 이 구분은 나중에 LLM 보안에서도 이어진다.

## 다음에 다시 할 때 볼 것

이 실습을 다시 한다면 아래 순서로 확인할 것 같다.

- PDF 텍스트가 제대로 추출됐는지
- chunk 크기가 너무 작거나 크지 않은지
- 검색된 상위 chunk에 실제 근거가 있는지
- retriever 결과와 manual cosine search 결과가 크게 다르지 않은지
- LLM 답변이 context에 근거하는지

## 정리

Naive RAG는 이름처럼 단순한 구조지만, 단순하다고 대충 봐도 되는 것은 아니었다. 오히려 이 단계에서 검색 결과를 확인하는 습관을 들여야 나중에 Advanced RAG를 붙였을 때도 어디가 좋아졌는지 판단할 수 있다.

## 관련 글

- [RAG Fundamentals & Advanced RAG 1]({% post_url 2026-01-23-upstage-course-w05d03-rag-basics %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %})
