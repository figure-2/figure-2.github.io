---
title: "07. Citation과 음성 데모를 연결하며 정한 경계"
categories:
- 2.PROJECT
- 2-2. History_Docent
tags:
- HistoryDocent
- Citation
- Voice AI
- evaluation
- retrospective
toc: true
date: 2026-09-14 09:00:00 +0900
comments: true
mermaid: false
math: false
---

생성 프롬프트를 고쳐 citation precision을 올렸는데 citation recall이 내려갔다. 더 엄격하게 인용하도록 만들면 근거를 잘못 붙이는 경우는 줄었지만, 답변을 지지하는 근거 일부가 citation에서 빠졌다.

Solar Pro 3 generation v2 repaired를 기본값에서 뺀 이유다. 한 지표가 좋아졌다는 사실보다 관광 해설에서 필요한 근거를 놓쳤다는 사실을 더 크게 봤다.

## 검색과 생성을 분리해서 평가했다

Retriever 평가는 target evidence가 Top-K에 들어오는지 확인한다. Generation 평가는 주어진 evidence로 답을 만들고 citation을 정확히 연결하는지 확인한다.

두 평가를 섞으면 원인을 잘못 찾기 쉽다. 정답 문서가 검색되지 않았는데 프롬프트만 수정하거나, 검색 근거는 맞는데 생성 모델이 조건을 잘못 해석한 문제를 Retriever 탓으로 돌릴 수 있다.

HistoryDocent에서는 다음 항목을 분리했다.

| 단계 | 확인한 질문 |
| --- | --- |
| Retrieval | target child와 parent가 Top-K에 있는가 |
| Evidence packing | 선택한 evidence와 source provenance를 복구할 수 있는가 |
| Generation | evidence 안에서 답했는가 |
| Citation | 답변의 주장과 citation이 연결되는가 |
| No-answer | 근거가 없을 때 생성을 멈추는가 |

P0 rank-order packing의 citation recoverability는 1.0이었다. MMR 방식인 P3는 중복 parent 비율을 조금 줄였지만 생성 품질 개선으로 이어졌다는 근거가 부족했다. 단순한 P0를 유지했다.

## 음성 데모는 기존 계약을 바꾸지 않았다

로컬 음성 실험에서는 faster-whisper small CUDA를 STT demo primary 후보로 두었다. TTS는 sherpa-onnx Supertonic 3 Korean을 demo review 후보로 수락했다.

TTS 근거는 private wav 5개, 자동 proxy 4/5, 한 명의 청취자가 입력한 30개 점수의 평균 5.0이다. 높은 점수지만 청취자는 한 명이다. 실제 관광지 소음, 여러 연령대, 여러 화자의 발음에서 품질을 검증하지 않았다.

따라서 "무료 로컬 TTS 최종 선정"이라고 쓰지 않는다. 현재 표현은 "포트폴리오 데모 후보"다.

외부 음성 provider 호출과 외부 audio 전송은 0으로 유지했다. Azure, Google, AWS는 비용과 credential, 음성 데이터 전송 정책을 확인한 뒤 비교할 선택지로만 남겼다.

## API route smoke가 확인한 범위

로컬 음성 route는 기본 비활성화 상태로 만들었다. Explicit flag가 있을 때만 contract 응답을 반환한다.

Smoke에서는 기본 비활성화 요청이 403인지, explicit flag 요청이 200인지, path traversal과 public audio 경로가 거절되는지 확인했다. 이 테스트에서 실제 STT, TTS, Solar 호출은 수행하지 않았다.

계약이 연결됐다는 사실과 음성 앱이 완성됐다는 주장을 분리했다. 마이크 capture, 자동 speaker playback, production voice provider는 아직 완료되지 않았다.

## 초기 기록과 현재 결과가 달라진 이유

초기 Notion 기록에는 BGE-M3 계열 임베딩, RRF Hybrid, 파인튜닝 Reranker, Gemini 생성 모델을 중심으로 한 파이프라인이 정리돼 있다. 당시 실험은 프로젝트의 첫 방향을 정하는 데 사용됐다.

현재 제출용 저장소는 parent-child citation corpus, E5-small voice rewrite, Solar Pro 3 answer contract, query type별 evaluation gate를 기준으로 다시 정리됐다. 두 구조를 하나의 현재 아키텍처처럼 섞지 않았다.

이 차이는 기록 오류가 아니라 프로젝트가 바뀐 흔적이다. 다만 포트폴리오에서는 가장 최신이며 재검증 가능한 기준선을 먼저 보여주고, 초기 기록은 왜 방향을 바꿨는지 설명할 때만 사용한다.

## 다시 시작한다면 평가셋을 먼저 만든다

이 프로젝트에서는 기능을 붙인 뒤 평가 기준을 정리한 구간이 길었다. 그러다 보니 작은 subset의 좋은 결과를 어디까지 믿을 수 있는지 매번 다시 판단해야 했다.

다시 시작한다면 먼저 query type, target grain, no-answer, split policy를 고정할 것이다. 그다음 baseline을 만들고 한 번에 변수 하나만 바꾼다. Voice UI와 advanced RAG는 그 뒤에 둔다.

HistoryDocent에서 남은 결과는 "최신 RAG 기술을 모두 적용했다"가 아니다. 후보를 비교할 기준을 만들었고, 좋은 숫자가 나와도 latency와 citation, locked split이 맞지 않으면 적용하지 않았다. 실제 서비스 검증은 남아 있지만, 어떤 조건이 충족돼야 다음 단계로 갈 수 있는지는 이전보다 분명해졌다.

- 이전 글: [GraphRAG와 HyDE를 기본 경로에서 뺀 이유]({% post_url projects/history-docent/2026-09-13-project-history-docent-06-advanced-rag-routing %})
- 시리즈 처음: [HistoryDocent 프로젝트 개요]({% post_url projects/history-docent/2026-09-08-project-history-docent-01-case-study %})
