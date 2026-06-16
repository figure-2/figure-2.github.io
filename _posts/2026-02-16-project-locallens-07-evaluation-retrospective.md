---
title: "07. 정량 평가, 테스트 한계, 그리고 LocalLens 회고"
categories:
- 2.PROJECT
- 2-6. LocalLens
tags:
- LocalLens
- Evaluation
- Testing
- Retrospective
- Retrieval
toc: true
date: 2026-02-16 09:00:00 +0900
comments: true
mermaid: true
math: true
---

LocalLens의 평가는 두 층으로 나눠 봐야 한다. 하나는 발표자료 기준의 모델 선택 근거이고, 다른 하나는 코드와 commit history에서 확인되는 실험/테스트 흐름이다. 두 근거를 섞으면 숫자의 의미가 흐려진다.

## 발표자료 기준 정량 근거

텍스트 인코더 선택에서는 후보 모델의 점수와 처리 시간이 함께 비교되었다.

![텍스트 인코더 후보 비교](/assets/images/locallens/01-text-model-benchmark.png)

발표자료의 MIRACL 기준 후보 비교는 다음과 같다.

| 모델 | 모델 크기 | MIRACL 점수 |
| --- | ---: | ---: |
| BAAI/bge-m3 | 0.57B | 69.59 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.57B | 66.53 |
| google/embeddinggemma-300m | 0.31B | 66.20 |
| intfloat/multilingual-e5-small | 0.12B | 60.09 |
| ibm-granite/granite-embedding-107m-multilingual | 0.11B | 57.25 |

점수만 보면 더 높은 후보가 있었지만, 로컬 검색기에서는 인덱싱 시간이 중요했다.

![텍스트 인코더 처리 시간 비교](/assets/images/locallens/02-text-model-latency.png)

발표자료의 500개 문서 처리 실험 기준 처리 시간은 다음과 같다.

| 모델 | 500개 총 소요 시간 | 개당 평균 |
| --- | ---: | ---: |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 962.07초 | 1.92초 |
| intfloat/multilingual-e5-small | 62.07초 | 0.12초 |

이 수치는 “로컬 검색기에서 왜 경량 모델을 선택했는가”를 설명하는 근거다. 다만 특정 하드웨어, 데이터, 실행 조건에 묶인 발표자료 기준이므로 일반 성능 보장처럼 쓰지 않는다.

## PDF 평가의 경계

PDF 처리에는 text-only 방식과 text+VLM caption 방식의 비교가 포함되어 있다. 과거 평가 스크립트 기준으로는 Recall@K, MRR, 처리 시간을 비교하는 구조가 있었고, 발표에서는 text+VLM 방식이 소폭이지만 일관된 개선을 보였다고 설명한다.

다만 결과 표나 재실행 결과가 명확히 남아 있지 않은 수치는 별도로 만들거나 추정하지 않았다.

정리할 수 있는 범위는 다음 수준이다.

| 항목 | 정리 범위 |
| --- | --- |
| 평가 방식 | text-only, OCR, text+OCR, text+VLM 비교 구조가 있었다 |
| 평가 지표 | Recall@K, MRR을 비교하는 스크립트가 있었다 |
| 결과 표현 | 발표 기준 text+VLM이 소폭이지만 일관된 개선을 보였다 |
| 해석 경계 | 구체 수치, 대폭 개선, 모든 PDF 일반화는 하지 않음 |

## 테스트 코드에서 확인한 것

현재 repo에는 encoder, search engine, PDF processor, VLM client 확인용 테스트 코드가 남아 있다. 다만 일부 테스트는 현재 코드 구조와 맞지 않는 오래된 클래스명을 참조한다.

따라서 현재 상태에서는 테스트 코드의 존재와 최신 코드 기준 통과 여부를 분리해서 읽어야 한다.

| 구분 | 판단 |
| --- | --- |
| PDF processor 확인 코드 | 텍스트/이미지 추출, combined text, VLM client 확인 흐름 존재 |
| search flow 확인 코드 | sample directory 검색 흐름 확인용 코드 존재 |
| encoder 테스트 | 일부 현재 코드와 맞지 않는 오래된 참조가 있어 재정비 필요 |
| 현재 해석 | 테스트 코드가 있었지만, 최신 코드 기준 통과 결과로 단정하지 않음 |

## 자체 한계 정리

![프로젝트 한계 정리](/assets/images/locallens/04-retrospective-limits.png)

한계는 실패 목록이 아니라 설계 경계를 드러내는 정보다. 이 프로젝트에서는 다음 항목을 후속 개선 지점으로 남겼다.

| 한계 | 해석 | 개선 방향 |
| --- | --- | --- |
| 외부 VLM 호출 | PDF 내부 이미지 captioning에서 외부 호출 사용 | caption caching, local VLM 검토 |
| 모델 최적화 부족 | 양자화나 가속 엔진 적용은 다루지 못함 | ONNX, TensorRT, batching 검토 |
| 모달리티 확장 부족 | 텍스트, 이미지, PDF 중심 구현 | audio/video 검색은 후속 과제 |
| 평가 데이터 부족 | PDF 평가 수치 일반화 어려움 | 더 큰 평가셋과 재현 가능한 실험 로그 필요 |

## 회고

LocalLens에서 가장 중요한 배움은 검색 품질이 모델 하나로 결정되지 않는다는 점이다. 로컬 파일 검색에서는 파일 상태 동기화, metadata 관리, 타입별 인코더 분리, 사용자가 결과 파일을 바로 열 수 있는 UX가 함께 필요했다.

특히 VectorStore 설계가 핵심이었다. FAISS만 있으면 벡터는 찾을 수 있지만, 사용자가 열 파일 경로와 수정 상태를 관리하기 어렵다. SQLite만 있으면 파일 정보는 관리할 수 있지만, 의미 기반 유사도 검색은 어렵다. 두 저장소를 나누고 ID로 연결한 것이 이 프로젝트의 가장 설명력 있는 구현 축이다.

또 하나의 배움은 local-first라는 표현의 경계다. 로컬 파일을 대상으로 인덱싱하고 검색하는 구조는 맞지만, PDF 내부 이미지 captioning에는 외부 VLM 호출이 들어간다. 그래서 정확한 표현은 “local-first를 지향한 멀티모달 검색 엔진”이다.

## 시리즈 마무리

| 순서 | 글 |
| --- | --- |
| 01 | [LocalLens 프로젝트 개요]({% post_url 2026-01-29-project-locallens-01-case-study %}) |
| 02 | [파일명 검색의 한계를 의미 기반 검색으로 바꾸기]({% post_url 2026-02-02-project-locallens-02-problem-and-demo %}) |
| 03 | [LocalLens 검색 파이프라인 구조]({% post_url 2026-02-04-project-locallens-03-architecture %}) |
| 04 | [FAISS와 SQLite로 VectorStore를 나눈 이유]({% post_url 2026-02-04-project-locallens-04-vectorstore-sync %}) |
| 05 | [Text, Image, PDF Encoder를 하나의 검색 흐름으로 묶기]({% post_url 2026-02-07-project-locallens-05-encoder-model-selection %}) |
| 06 | [PDF 안의 표와 그래프를 검색 맥락으로 바꾸기]({% post_url 2026-02-13-project-locallens-06-pdf-vlm-retrieval %}) |
| 07 | 이 글 |
