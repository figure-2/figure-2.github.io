---
title: "Advanced RAG"
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
date: 2026-01-26 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Advanced RAG

Advanced RAG는 처음 들으면 기법 이름이 너무 많다. HyDE, RAGRouter, Small-to-Big, Self-RAG, GraphRAG, RAPTOR처럼 이름만 보면 다 따로 외워야 할 것 같다.

그런데 수업을 정리하면서 생각을 바꿨다. 이건 기법 이름 목록이 아니라 **Naive RAG가 실패하는 지점을 고치는 방법들**이다.

## 먼저 Naive RAG의 한계부터 보기

W05D03에서 기본 RAG 파이프라인을 먼저 다뤘고, W05D04에서는 그 다음 단계로 Advanced RAG와 LLM 보안을 함께 다뤘다. 기본 RAG는 "문서를 잘라서 임베딩하고, 질문과 가까운 chunk를 찾아 LLM에 넣는다"는 흐름이다. 하지만 실제 서비스에서는 이 정도만으로 답변 품질이 안정적으로 나오지 않는다.

내가 이해한 실패 지점은 대략 이렇다.

- 질문이 검색에 적합하지 않다.
- 관련 문서가 검색되지 않는다.
- 문서는 검색됐지만 순서가 별로다.
- 답변에 필요한 주변 문맥이 빠져 있다.
- 문서들끼리 서로 다른 말을 한다.
- LLM이 검색 결과보다 자기 내부 지식으로 답한다.

Advanced RAG는 이 문제들을 한 번에 해결하는 하나의 기술이 아니라, 단계별 보정 도구들의 묶음이다.

## query를 고치는 방법

사용자 질문은 보통 검색하기 좋은 문장이 아니다. 짧고, 맥락이 빠져 있고, "이거", "저번 방식" 같은 표현이 들어간다.

그래서 query rewriting이 필요하다. 사용자의 질문을 검색에 맞게 다시 쓰는 것이다.

```text
사용자 질문: "이거 배포할 때 주의할 점 알려줘"
검색용 질문: "FastAPI Docker 애플리케이션 클라우드 배포 시 환경변수, 포트, health check, secret 관리 주의점"
```

HyDE는 여기서 한 단계 더 간다. 질문을 바로 embedding하지 않고, 먼저 가상의 답변 문서를 만든 뒤 그 문서로 검색한다. 질문보다 문서에 가까운 형태로 바꿔 검색하는 방식이다.

Reverse HyDE는 반대로 문서 쪽에 가상의 질문을 만들어두는 식으로 이해했다.

## 검색 결과를 고치는 방법

검색 자체도 한 가지 방식만 쓰면 한계가 있다.

- 벡터 검색은 의미가 비슷한 문서를 잘 찾는다.
- 키워드 검색은 정확한 문자열에 강하다.
- reranking은 1차 후보를 다시 정렬한다.

예를 들어 에러 메시지나 설정 키를 찾을 때는 dense retrieval보다 BM25가 더 나을 수 있다. 반대로 사용자가 자연어로 질문하면 dense retrieval이 더 유리할 수 있다. 그래서 hybrid search가 필요해진다.

## Small-to-Big

Small-to-Big는 수업에서 제일 직관적으로 이해된 기법이다.

검색할 때는 작은 chunk가 좋다. 정확히 맞는 부분을 찾기 쉽기 때문이다. 그런데 답변할 때는 작은 chunk만 있으면 문맥이 부족하다. 그래서 작은 chunk로 검색하고, 답변에는 그 chunk가 속한 더 큰 문맥을 넣는다.

```text
small chunk로 검색
-> parent document 또는 주변 문단 가져오기
-> LLM에 더 넓은 context 제공
```

검색 단위와 생성 단위를 분리한다는 점이 핵심이다.

## RAGRouter

모든 질문을 같은 RAG 파이프라인으로 처리하기 어렵다. 정의를 묻는 질문, 정책 날짜를 묻는 질문, 코드 에러를 묻는 질문은 검색 방식이 달라야 한다.

RAGRouter는 질문을 보고 어느 검색 경로를 탈지 고르는 방식이다. 이건 나중에 에이전트 구조와도 연결될 것 같다.

## Self-RAG / CRAG

Self-RAG와 CRAG는 검색 결과가 충분한지 확인하고, 부족하면 다시 검색하거나 답변을 보류하는 쪽에 가깝다.

이 부분은 특히 "모르면 모른다고 말하기"와 연결된다. 검색 결과가 약한데도 무조건 답변하게 만들면 RAG를 붙인 의미가 줄어든다.

## GraphRAG / RAPTOR

GraphRAG는 문서 안의 entity와 relation을 그래프로 구성해 관계형 질문에 대응한다. RAPTOR는 문서를 계층적으로 요약해 긴 문서 검색에 활용한다.

이 둘은 바로 모든 프로젝트에 넣을 기법이라기보다는, 문서가 길고 관계가 복잡해졌을 때 고려할 카드로 보는 게 맞을 것 같다.

## Knowledge Conflict

검색 결과가 여러 개라고 해서 답이 명확해지는 것은 아니다. 모델 내부 지식과 검색 문서가 다를 수 있고, 검색 문서끼리 서로 다른 정보를 말할 수도 있다.

이때는 답변 전에 충돌 여부를 봐야 한다.

```text
검색 결과 있음
-> 근거가 충분한가
-> 근거끼리 충돌하지 않는가
-> 어떤 출처를 우선할 것인가
```

## 구현할 때의 순서

Advanced RAG를 처음부터 다 넣으면 오히려 디버깅이 어려워질 것 같다. 그래서 순서는 이렇게 잡는 게 맞다고 본다.

```text
1. Naive RAG baseline 만들기
2. 실패 질문 모으기
3. 실패 원인을 분류하기
4. 필요한 기법만 하나씩 붙이기
5. 지연시간과 비용 같이 보기
```

## 볼 기준

- 검색 결과에 정답 근거가 포함되는가
- 상위 문서가 질문과 실제로 관련 있는가
- 답변이 검색 문서에 근거하는가
- 모르는 질문에 억지로 답하지 않는가
- prompt injection이나 오염된 문서에 의해 시스템 지시가 흔들리지 않는가
- 문서끼리 충돌할 때 출처 우선순위가 적용되는가
- 검색 전략 변경이 지연시간과 비용을 과하게 늘리지 않는가

## 정리

Advanced RAG는 "좋아 보이는 기법을 많이 붙이는 것"이 아니다.
Naive RAG가 어디서 실패하는지 보고, 그 지점에 맞는 보정 단계를 붙이는 것이다.

그리고 RAG 문서 자체도 외부 입력이라는 점을 잊으면 안 된다. 검색 결과는 참고 자료이지 명령이 아니다. 이 기준은 LLM 보안으로 그대로 이어진다.

## 관련 글

- [RAG Fundamentals & Advanced RAG 1]({% post_url 2026-01-23-upstage-course-w05d03-rag-basics %})
- [Advanced RAG 2 & LLM 보안]({% post_url 2026-01-26-upstage-course-w05d04-advanced-rag-security %})
- [RAG]({% post_url 2026-01-23-upstage-tech-rag %})
- [Agentic RAG]({% post_url 2026-01-30-upstage-tech-agentic-rag %})
- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %})

## 참고 자료

- [Advanced RAG Techniques (Llamaindex)](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/)
