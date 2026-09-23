---
title: "13. SeSAC:Note Summarizer 배치 크기 — 실험과 결과 기록"
description: "Summarizer의 배치 크기와 모델별 처리 결과를 비교합니다. 배치 비교와 모델 비교의 조건을 구분했으며, 가격은 실험 당시의 가정입니다."
categories:
- 2.PROJECT
- 2-5. SeSAC-Note
tags:
- 프로젝트 자료
toc: true
date: 2026-09-24 00:00:00 +0900
comments: true
mermaid: true
math: false
---

Summarizer의 배치 크기와 모델별 처리 결과를 비교합니다. 배치 비교와 모델 비교의 조건을 구분했으며, 가격은 실험 당시의 가정입니다.

이 글은 같은 입력에서 배치 크기를 바꿨을 때의 전체 요약 실행 시간과 토큰 사용량을 다룹니다. 첫 요약을 사용자에게 보여주기까지의 시간과는 다른 지표입니다. 결과표의 50.52초·입력 26,889토큰은 **배치 크기 2**의 값이며, 배치 크기 4의 결과는 76.62초·입력 18,575토큰입니다.

worker 분리·상태 추적·결과 전달 구조는 [07. 비동기 처리]({% post_url projects/sesac-note/2026-01-06-project-sesac-note-07-async-pipeline %})에서, 첫 응답을 앞당긴 구현 과정은 [15. OCR·요약·RAG 개선 기록]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-optimization-record %})에서 다룹니다. 아래에는 실험 계획과 결과를 순서대로 남겼습니다. 계획 단계의 문장은 당시 상태를 설명하며, 후속 구현의 현재 상태를 뜻하지 않습니다.

{% raw %}

# [Experiment] Summarizer Batch Size 성능 벤치마크

## 목적
Summarizer의 배치 크기(batch size)에 따른 성능(시간, 토큰 사용량)과 출력 품질을 비교하여 향후 Progressive Pipeline 설계에 참고할 데이터를 확보한다.

## 배경
현재 Summarizer는 모든 세그먼트를 하나의 거대한 프롬프트에 담아 **Single Call**로 처리하고 있음.
- **문제점:** 영상이 길어지면 입력 토큰이 급증, Latency 악화, 단일 실패 시 전체 실패 위험.
- **향후 계획:** Progressive Pipeline (예: 2 segments씩 VLM → Sync → Summary → Judge 반복)으로 전환 예정.
- **실험 목적:** 배치 크기별 성능 특성을 파악하여 Progressive 설계 시 최적 chunk size 결정에 활용.

## 실험 설계

### 테스트 데이터
- `segments_units.jsonl` (8 segments 기준)

### 배치 크기 조건
| Batch Size | 호출 횟수 (8 seg 기준) |
|:----------:|:----------------------:|
| 1          | 8                      |
| 2          | 4                      |
| 4          | 2                      |
| 8 (전체)   | 1                      |

### 측정 지표
1. **시간 (Latency)**
   - 전체 실행 시간 (초)
   - 호출당 평균 시간
2. **토큰 사용량**
   - 입력 토큰 (Input Tokens)
   - 출력 토큰 (Output Tokens, 가능하면)
   - 총 토큰
3. **출력 품질**
   - 생성된 `segment_summaries.jsonl` 저장
   - Judge 점수 비교 (선택적)

## 구현 방향
1. **벤치마크 스크립트 생성** (`scripts/benchmark_summarizer.py` 또는 유사 경로)
   - 기존 `summarizer.py` 로직을 재활용하되, 배치 크기를 파라미터로 받음.
   - 각 조건별 결과를 별도 디렉토리/파일로 저장.
2. **결과 리포트 생성**
   - JSON 또는 Markdown 형태로 비교표 출력.

## Acceptance Criteria
- [x] 벤치마크 스크립트 작성 완료
- [x] Batch Size 1, 2, 4, 8 조건별 실행 완료
- [x] 시간/토큰 사용량 비교표 작성
- [x] 결과 요약 및 향후 권장 사항 문서화

## 비고
- 프로덕션 코드(`summarizer.py`)는 변경하지 않음 (추후 Progressive Pipeline 전환 시 별도 작업).
- 이 실험은 아키텍처 의사결정을 위한 데이터 수집 목적.

---

## 추가 실험 기록 1

## Summarizer Batch Size Benchmark 결과 보고

**실험 조건**
- **데이터**: 8 segments (`test4_Diffusion`)
- **모델**: `gemini-3-flash-preview`
- **환경**: `max-workers=4`, `request-interval=2.0s` (API Rate Limit 방지를 위한 Throttling 적용)

### 1. 결과 요약 (Baseline: Batch 8)

| Batch Size | Latency (s) | vs Batch 8 (%) | Input Tokens | vs Batch 8 (%) | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 341.00 | +310% 🔺 | 43,517 | +202% 🔺 | ⚠️ **비권장** (API 불안정, 비용↑) |
| **2** | **50.52** | **-39.2%** ⬇️ | 26,889 | +86.5% 🔺 | 🏆 **최고 속도** |
| 4 | 76.62 | -7.8% ⬇️ | 18,575 | +28.8% 🔺 | ✅ **비용/속도 균형** |
| 8 | 83.13 | - | 14,418 | - | Baseline |

### 2. 결론 및 제안

1.  **Batch 1 (세그먼트 단위 병렬화) 지양**
    - **Context 손실 위험**: 개별 세그먼트 처리 시 앞뒤 문맥 파악이 어려워 품질 저하 우려.
    - **API 불안정성**: 과도한 호출로 Rate Limit(429) 및 재시도 지연 발생.
    - **토큰 비용 폭증**: 기본 프롬프트 중복으로 Input Token이 3배(+202%) 증가.

2.  **프로그레시브 파이프라인 전략**
    - **Batch 2~4 권장**: 2~4개 세그먼트를 묶어서 처리하는 것이 속도와 안정성 면에서 최적.
      - **속도 중시**: Batch 2 (Latency 39% 감소)
      - **비용 효율 중시**: Batch 4 (Latency 8% 감소, 토큰 증가폭 합리적)
    - **프롬프트 최적화 필요**: 배치를 쪼갤수록 시스템 프롬프트(System Instruction) 비용이 중복 발생하므로, 프롬프트 경량화가 시급함.

---

## 추가 실험 기록 2

# Benchmark Comparison: Gemini 3 Flash vs Gemini 2.5 Flash

### 실험 조건
- **데이터**: 8 segments (test4_Diffusion)
- **환경**: max-workers=4, request-interval=2.0s (Throttling 적용)

## 1. 모델별 성능 요약 (Batch 2 기준)

| Metric | Gemini 3 Flash | Gemini 2.5 Flash | 차이 |
| :--- | :--- | :--- | :--- |
| **Latency (s)** | 50.52s | 54.35s | 3 Flash가 소폭 빠름 (-7%) |
| **Output Tokens** | 10,811 | 17,631 | 2.5 Flash가 훨씬 수다스러움 (+63%) |
| **TPS (Tokens/sec)** | ~214 | ~324 | 2.5 Flash가 생성 속도는 압도적 (+51%) |

## 2. 비용 분석 (8 Segments 추정치)

| Model | Token Pricing (In/Out) | Total Cost | 비율 |
| :--- | :--- | :--- | :--- |
| **Gemini 3 Flash** | $0.50 / $3.00 | $0.0458 | 100% |
| **Gemini 2.5 Flash** | $0.10 / $0.40 | $0.0096 | ~21% (1/5 수준) |

## 3. 결론: Trade-off 분석

두 모델은 명확한 장단점을 가지고 있습니다.

### Cost & Throughput (Gemini 2.5 Flash)
*   **장점**: 압도적인 가성비(비용 80% 절감)와 빠른 생성 속도(TPS).
*   **특성**: 요약문이 매우 길고 상세하게 생성됨(Verbose).

### Latency & Conciseness (Gemini 3 Flash)
*   **장점**: 응답 완료 시간(Latency)이 가장 빠르고, 결과물이 간결함.
*   **단점**: 상대적으로 높은 비용.

> (관련 논의: #31)

---

## 추가 실험 기록 3

30분 영상 정도면 잘 안돼서 필요했는데 좋습니당

{% endraw %}

## 관련 글

- [11. SeSAC:Note 프로젝트 소개와 서비스 구성]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-source-introduction %})
- [12. SeSAC:Note 슬라이드 캡처 중복 제거 — 비교 실험]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-source-capture-dedup %})
