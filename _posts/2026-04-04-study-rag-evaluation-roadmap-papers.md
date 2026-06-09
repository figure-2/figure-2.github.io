---
title: "RAG 완전 가이드 3: 평가, 도입 로드맵, 논문 타임라인"
categories:
- 3.STUDY
- 3-2.RAG
tags:
- study
- rag
- evaluation
- roadmap
- papers
toc: true
date: 2026-04-04 02:20:00 +0900
comments: false
mermaid: true
math: true
---

RAG 평가는 검색과 생성을 분리해서 봐야 한다. 좋은 답변이 나오지 않았을 때 검색 실패인지, 컨텍스트 구성 실패인지, 생성 실패인지 구분할 수 있어야 개선이 가능하다.

## RAG 평가하기

RAG 시스템의 품질은 faithfulness, answer relevancy, context precision, context recall로 나눠 측정할 수 있다.

#### Faithfulness

응답이 검색된 문서에 충실한가? 지어낸 내용 없이 문서 내용만으로 답변하는지 측정합니다.

충실도 = 문서로 뒷받침되는 주장 수 / 전체 주장 수

#### Answer Relevancy

응답이 질문에 적절한가? 질문과 관련 없는 내용이 답변에 포함되지 않았는지 평가합니다.

관련도 = 응답에서 생성된 질문과 원래 질문의 유사도 평균

#### Context Precision

검색된 문서가 정밀한가? 관련 없는 문서가 너무 많이 포함되지 않았는지 확인합니다.

정밀도 = 관련 문서 수 / 검색된 전체 문서 수

#### Context Recall

필요한 문서를 빠짐없이 검색했는가? 정답에 필요한 정보가 모두 검색 결과에 포함되었는지 봅니다.

재현율 = 검색된 관련 문서 수 / 전체 관련 문서 수

### 평가 도구

#### RAGAS

가장 대표적인 오픈소스 평가 프레임워크. Faithfulness, Relevancy, Context Precision/Recall 등 핵심 메트릭을 자동으로 계산합니다.

오픈소스

#### DeepEval

CI/CD 파이프라인에 통합 가능한 유닛 테스트 스타일 평가. pytest처럼 RAG 평가를 자동화합니다.

CI/CD 통합

#### TruLens

프로덕션 환경에서 실시간 모니터링. 피드백 함수로 지속적인 품질 추적이 가능합니다.

프로덕션 모니터링

#### RAGBench

12개 도메인에 걸친 100K+ 예제 벤치마크. 산업별 RAG 성능 비교에 활용됩니다.

벤치마크

## 주요 논문 타임라인

RAG 연구의 핵심 논문들을 시간순으로 정리했습니다.

2020

#### RAG: Retrieval-Augmented Generation

Lewis et al. — NeurIPS 2020

DPR + BART 결합. RAG-Sequence, RAG-Token 두 변형 제안. RAG 연구의 시작점.

2020

#### Dense Passage Retrieval (DPR)

Karpukhin et al. — EMNLP 2020

밀집 벡터 기반 문서 검색의 표준. 기존 TF-IDF/BM25를 뛰어넘는 시맨틱 검색 실현.

2022

#### HyDE: Hypothetical Document Embeddings

Gao et al., 2022

질문 대신 가상 답변을 생성해서 검색. Zero-shot에서도 Fine-tuned 모델 수준의 검색 성능 달성.

2023

#### Self-RAG: Self-Reflective RAG

Asai et al. — ICLR 2024 (Oral)

Reflection Token으로 검색 필요 여부, 문서 관련성, 응답 품질을 LLM이 자체 판단.

2024

#### CRAG: Corrective RAG

Yan et al., 2024

검색 결과를 Correct/Incorrect/Ambiguous로 분류 후 보정 경로 선택. 19~37% 정확도 향상.

2024

#### RAPTOR: Recursive Abstractive Processing

Sarthi et al. — ICLR 2024

재귀적 클러스터링 + 요약으로 다단계 추상화 트리 구축. QuALITY에서 +20% 향상.

2024

#### GraphRAG

Microsoft Research, 2024

지식 그래프 + 계층적 커뮤니티 요약. 여러 문서에 걸친 글로벌 질문에 강점.

2024

#### Contextual Retrieval

Anthropic, 2024

청크에 문서 수준 맥락을 접두사로 추가. BM25와 결합 시 검색 실패율 67% 감소.

2024

#### Adaptive RAG

Jeong et al., 2024

질문 복잡도에 따라 No Retrieval / Single-step / Multi-step을 동적으로 선택.

2024

#### Late Chunking

Jina AI, 2024

전체 문서를 먼저 임베딩 → 토큰 벡터에서 청크 추출. 추가 LLM 비용 없이 문맥 보존. jina-v3에서 API 지원.

2024

#### RAFT (Retrieval Augmented Fine-Tuning)

UC Berkeley, 2024

RAG + Fine-tuning 결합. 방해 문서를 무시하도록 학습. 전문 도메인에서 순수 RAG보다 높은 성능.

2025

#### SimRAG (Self-Improving RAG)

NAACL 2025

비라벨 코퍼스에서 자체 QA 쌍 생성 + self-training. 라벨링 비용 없이 도메인 적응. 11개 데이터셋 검증.

2025

#### MCP (Model Context Protocol)

Anthropic, 2025

AI 모델과 외부 도구/데이터를 연결하는 오픈 표준 프로토콜. Agentic RAG의 도구 통합을 표준화.

2026

#### Context Engineering & Knowledge Runtime

Industry Paradigm Shift

RAG → Context Engineering으로 패러다임 확장. RAG(정적 지식) + Memory(동적 이력) + MCP(도구 연결)를 통합하는 Knowledge Runtime 개념 등장.

---

## 추가 정리

### 핵심 요약

RAG 평가는 answer quality만 보면 부족하다. retrieval quality, context quality, faithfulness, citation accuracy를 나누어 봐야 한다.

### 보충 해설

답변이 틀렸을 때 원인은 여러 가지일 수 있다. 검색이 틀렸는지, 맞는 문서를 찾았지만 context에 못 넣었는지, context는 맞지만 모델이 무시했는지, citation이 잘못 붙었는지 분리해야 한다. 논문 타임라인은 이 문제들이 어떤 순서로 발전했는지 보는 기준으로 활용하면 된다.
