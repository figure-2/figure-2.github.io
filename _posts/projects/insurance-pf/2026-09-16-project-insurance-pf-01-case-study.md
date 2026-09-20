---
title: "01. Insurance_PF: 자동차 보험 약관을 근거로 답하는 RAG 만들기"
categories:
- 2.PROJECT
- 2-8. Insurance_PF
tags:
- Insurance PF
- rag
- Insurance
- BM25
- QLoRA
toc: true
date: 2026-09-16 09:00:00 +0900
comments: true
mermaid: false
math: false
---

검색 평가에서는 50개 질문 중 대부분의 정답 문서를 찾았다. 그런데 최종 답변에서는 "보상하지 않는다"는 조항을 놓쳤다. 아내가 운전한 차에 치인 상황에서도 Base Model, Fine-Tuned Model, RAG 조합이 모두 틀린 답을 만들었다.

Insurance_PF는 자동차 보험 약관을 검색하고 일반 사용자의 질문에 답하는 개인 프로젝트다. 시작할 때는 도메인 파인튜닝이 보험 답변의 정확도를 높일 것으로 기대했다. 실험을 진행하면서 답변 말투, 근거 검색, 조건 해석은 서로 다른 문제라는 것을 확인했다.

## 보험 질문은 키워드 하나로 끝나지 않았다

보험 약관에는 숫자, 기간, 보상 조건, 면책 조건이 함께 들어 있다. 한 문장의 부정 표현을 놓치면 답의 방향이 반대로 바뀐다.

사용자의 표현과 약관의 표현도 다르다. 사용자는 "노트북 액정이 깨졌어요"라고 묻지만 약관은 "휴대품 손해"로 적을 수 있다. 사용자는 "아내 차에 치였어요"라고 말하고, 약관은 배우자와 피보험자의 관계를 기준으로 보상 제외 조건을 설명한다.

이 문제를 다음 단계로 나눴다.

> 보험사 PDF → Parser JSON → 약관 구조 기반 chunk → Dense·BM25 검색 → Top-K context → Fine-Tuned LLM → 답변

검색과 생성이 모두 맞아야 한다. 정답 조항을 찾지 못하면 생성 모델이 고칠 근거가 없다. 조항을 찾았더라도 복잡한 조건이나 계산을 잘못 읽으면 답은 틀린다.

## 내가 수행한 작업

- 11개 보험사의 Parser JSON을 분석하고 계층적 청킹 파이프라인을 만들었다.
- 제목 경로, 보험사, 약관 유형, 페이지 범위를 metadata로 남겼다.
- HTML 표를 Markdown 표로 변환했다.
- 세 임베딩 모델과 세 형태소 분석기를 비교했다.
- Dense, Sparse, Hybrid Retrieval을 30개와 50개 질문 세트에서 비교했다.
- Llama-3-Open-Ko-8B에 QLoRA를 적용했다.
- Base, Fine-Tuned, RAG + Fine-Tuned 응답을 정성 비교했다.
- 실패 사례를 검색 실패, 부정 조항 해석, 계산 오류로 나눴다.

## 실험에서 확인한 범위

| 단계 | 확인한 결과 | 해석 범위 |
| --- | --- | --- |
| 전처리 | 11개 보험사, 6,402개 chunk | metadata와 breadcrumb 누락 여부를 검사한 corpus |
| Embedding | 30개 hard-negative 질문 | ko-sroberta가 후보 중 separation 0.0721 |
| Tokenizer | 30개 FAQ | Mecab BM25 MRR@5 0.9833 |
| Retrieval | 30개 및 별도 50개 평가 | 현재 로그 조건에서는 Sparse가 가장 높은 순위 품질 |
| Fine-Tuning | hold-out 중 50개 fast check | 정량 정확도가 아니라 대표 사례 중심 정성 평가 |
| RAG 통합 | 30개 샘플 | 구체적 수치 성공과 검색·계산 실패를 함께 기록 |

이 표의 수치를 하나의 최종 정확도로 합치지 않는다. 단계마다 데이터셋과 성공 조건이 다르기 때문이다.

## 처음 예상과 달라진 결정

최신 다국어 임베딩 모델이 가장 좋을 것으로 예상했다. 30개 hard-negative 비교에서는 ko-sroberta가 정답과 오답을 구분하는 separation에서 가장 나았다.

Dense와 Sparse를 합치면 검색이 좋아질 것으로 예상했다. 초기 30개 FAQ에서는 Mecab BM25 단독의 MRR@5가 0.9833으로 Hybrid의 0.9444보다 높았다.

Fine-Tuning을 하면 보험 지식이 정확해질 것으로 예상했다. 실제로 답변 말투와 예시 생성은 좋아졌지만, 약관 조항 번호와 부정 조건은 여전히 틀렸다. RAG를 결합하자 일부 숫자와 기간은 정확해졌으나, 검색이 실패한 질문은 그대로 오답이 됐다.

프로젝트 후반에는 "어떤 모델이 가장 좋은가"보다 "어느 단계에서 틀렸는가"가 더 중요한 질문이 됐다.

## 제출할 때 지키는 경계

이 프로젝트는 보험 상담 서비스의 정확성을 입증하지 않았다. Retrieval 평가는 30개와 50개 규모이고, 생성 평가는 대표 사례에 대한 정성 분석이 중심이다. Ragas 기반 Faithfulness와 Answer Relevance 평가는 후속 계획으로 남아 있다.

Fine-Tuned 모델이 Base보다 설명을 잘한 사례는 있다. 그러나 이를 전체 정확도 향상으로 표현하지 않는다. 보험과 법률에 가까운 질문에서는 답변을 잘 쓰는 능력보다 틀린 조건을 감지하고 답변을 보류하는 계약이 더 필요했다.

## 시리즈 구성

1. 이 글
2. [보험 약관을 일정한 글자 수로 자르면 안 되는 이유]({% post_url projects/insurance-pf/2026-09-17-project-insurance-pf-02-hierarchical-chunking %})
3. [보험 용어 검색에서는 어떤 토크나이저가 유리했나]({% post_url projects/insurance-pf/2026-09-18-project-insurance-pf-03-tokenizer-bm25 %})
4. [파인튜닝은 말투를 바꿨지만 사실을 보장하지 않았다]({% post_url projects/insurance-pf/2026-09-19-project-insurance-pf-04-finetuning-rag %})
5. [보상하지 않는다는 문장을 놓친 보험 RAG 회고]({% post_url projects/insurance-pf/2026-09-20-project-insurance-pf-05-failure-retrospective %})
