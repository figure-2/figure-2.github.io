---
title: "AI-Ready Data 가이드"
categories:
- 3.STUDY
- 3-7.AI_ENGINEERING
tags:
- study
- ai-engineering
- ai-ready-data
- data-quality
- rag
- knowledge-graph
toc: true
date: 2026-05-16 14:00:00 +0900
comments: false
mermaid: true
math: true
---
# AI-Ready Data 가이드

> **한줄 정의**
> AI-Ready Data는 모델이 바로 사용할 수 있도록 접근 가능하고, 품질이 관리되고, 구조와 권한이 명확하며, 실제 유스케이스에 맞게 정리된 데이터다.

## 모델 문제가 아니라 데이터 문제일 수 있다

AI 프로젝트가 기대만큼 동작하지 않을 때 바로 모델이나 프롬프트를 바꾸면 원인을 놓치기 쉽다. RAG도 마찬가지다. 검색 품질은 embedding model만이 아니라 원본 문서의 구조, metadata, chunking, 권한, 최신성에 크게 의존한다.

학습 노트는 AI 프로젝트가 중단되거나 기대만큼 작동하지 않는 원인을 모델보다 데이터 준비 상태에서 먼저 찾는다. 요약 문구의 `60%` 수치는 정리 당시 노트 기준이며, 최신 통계로 단정하지 않는다.

| 증상 | 먼저 의심할 데이터 문제 |
| --- | --- |
| 관련 문서가 검색되지 않음 | 의미 단위 청킹 실패, metadata 부재 |
| 검색은 되지만 답이 틀림 | 원본 문서가 오래됐거나 서로 충돌 |
| 답변이 매번 달라짐 | 용어와 entity가 일관되지 않음 |
| 근거 없는 답이 나옴 | 핵심 필드 누락, 정의 부재 |
| 보안 위험이 생김 | 권한 없는 문서까지 색인 |

데이터가 지저분하면 AI는 조용히 실패한다. 전통적인 시스템은 잘못된 입력에서 에러를 내지만, LLM은 틀린 입력에서도 그럴듯한 답을 만들 수 있다.

## 5가지 기준

AI-Ready Data는 단순히 데이터가 많다는 뜻이 아니다.

| 기준 | 확인 질문 |
| --- | --- |
| Availability | AI가 필요한 데이터에 안정적으로 접근할 수 있는가 |
| Quality | 정확성, 일관성, 완전성, 중복 관리가 되는가 |
| Structure | schema, metadata, 관계, 문서 구획이 명시되어 있는가 |
| Governance | 출처, 권한, 보존 기간, 삭제 기준이 있는가 |
| Use-case Alignment | 실제 AI 작업에 맞는 단위와 품질로 준비됐는가 |

가장 자주 빠지는 것은 `Use-case Alignment`다. 전사 데이터를 모두 넣는 것보다, 실제 질문에 답할 수 있는 데이터 단위를 먼저 정하는 편이 낫다.

## 진단 순서

AI 결과가 이상할 때는 모델보다 앞단을 먼저 본다.

![AI-Ready Data diagnosis flow](/assets/images/study/diagrams/study-ai-ready-data-diagnosis.svg){: width="100%"}

RAG에서는 특히 `원본 데이터에 답이 있는가`와 `검색된 context에 답이 들어왔는가`를 분리해서 봐야 한다.

## 구조화 단계

데이터 정리는 한 번에 끝나는 작업이 아니라 점진적으로 높아지는 구조화 과정이다.

| 단계 | 작업 | 산출물 |
| --- | --- | --- |
| 1. Inventory | 데이터 소스, 소유자, 형식, 업데이트 주기 파악 | 데이터 목록 |
| 2. Schema | entity, relation, 필수 필드, 제약 정의 | schema / metadata |
| 3. Normalize | 용어 통일, 형식 표준화, 중복 제거 | 정규화 규칙 |
| 4. Prepare | RAG, agent, fine-tuning 목적별 변환 | chunk, API, dataset |
| 5. Evaluate | 질문셋과 실패 케이스로 품질 확인 | 평가셋 / 회귀 기준 |

![AI-Ready Data structuring loop](/assets/images/study/diagrams/study-ai-ready-data-structure-loop.svg){: width="100%"}

## 목적별 준비 방식

같은 데이터라도 AI가 쓰는 방식에 따라 준비 형태가 다르다.

| 목적 | 필요한 준비 |
| --- | --- |
| RAG | 의미 단위 chunking, metadata, version, access control |
| Agent | tool schema, API 문서, 권한 경계, 실행 로그 |
| Fine-tuning | 입력-출력 쌍, 품질 검수, 라벨 기준 |
| Knowledge Graph | entity, relation, ontology, entity resolution |

RAG용 데이터는 문서 chunk가 중요하고, agent용 데이터는 도구가 읽을 수 있는 schema와 권한이 중요하다. Knowledge Graph로 갈수록 같은 entity를 하나로 묶는 규칙이 중요해진다.

## RAG와 연결되는 지점

RAG 실패는 대개 검색 단계에서 드러나지만, 원인은 더 앞에 있을 수 있다.

![RAG failure points caused by data quality, metadata, and chunking](/assets/images/study/diagrams/study-ai-ready-data-rag-failure.svg){: width="100%"}

그래서 RAG 개선은 reranker를 붙이는 것만으로 끝나지 않는다. 원본 파싱, 문서 버전, section tag, 권한 필터, entity alias까지 같이 봐야 한다.

## 내 기준

AI-Ready Data의 핵심은 `AI가 읽을 수 있음`이 아니라 `AI가 틀렸을 때 원인을 추적할 수 있음`이다.

```text
데이터 접근 가능
  -> 품질과 최신성 관리
  -> 구조와 관계 명시
  -> 권한과 생명주기 관리
  -> 실제 질문으로 평가
```

이 순서가 없으면 모델을 바꿔도 같은 문제가 반복된다.

## 관련 글

- [RAG 완전 가이드 1: 필요성과 기본 구조]({% post_url 2026-04-04-study-rag-why-and-pipeline %})
- [온톨로지 & 지식 그래프 가이드 1: 개념, 스펙트럼, 트리플]({% post_url 2026-05-16-study-kg-ontology-triple %})
- [AI Native 팀 운영 가이드]({% post_url 2026-04-10-study-ai-native-team %})
