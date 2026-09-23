---
title: "11. SeSAC:Note 프로젝트 소개와 서비스 구성"
description: "SeSAC:Note의 문제 정의, 멀티모달 분석 파이프라인과 학습 기능을 정리합니다."
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

SeSAC:Note의 문제 정의, 멀티모달 분석 파이프라인과 학습 기능을 정리합니다.

{% raw %}


### 1. 문제 정의

#### 시각 정보의 부재와 정보 손실

기존의 강의 요약 서비스들은 대부분 음성(STT) 데이터만을 활용합니다. 하지만 실제 강의에서 핵심 수식, 복잡한 도표, 실시간 판서 등은 화면을 통해 전달됩니다. 음성만으로 요약할 경우 이러한 시각적 맥락이 완전히 유실되어, 결과물만으로는 강의 내용을 완벽히 이해하기 어려운 한계가 있습니다.

#### 학습자의 높은 인지적 부담

온라인 강의 학습자는 영상 시청, 내용 이해, 필기를 동시에 수행해야 합니다. 특히 전문 용어가 많이 등장하는 전공 강의나 기술 강연의 경우, 필기에 집중하다 강의 맥락을 놓치거나 특정 부분을 찾기 위해 영상을 반복해서 돌려보는 등 학습 효율이 저하되는 고질적인 문제가 존재합니다.

### 2. 프로젝트 목표

#### 멀티모달 기반의 고품질 요약 노트 생성

음성(STT)과 화면(VLM) 정보를 통합 분석하여, 영상 없이 노트만으로도 완벽한 학습이 가능한 구조화된 강의 노트를 자동 생성합니다.

#### 요약 노트 기반의 1:1 AI 튜터 구현

생성된 노트를 학습 데이터로 활용하여, 강의 맥락을 완벽히 이해하고 사용자의 질문에 정확한 근거를 바탕으로 답변하는 맞춤형 AI 튜터링 서비스를 제공합니다.

### 3. 서비스 파이프라인

![SeSAC:Note 프로젝트 소개 이미지 2](/assets/images/source-archives/sesac-note/intro-02.webp)

#### 데이터 통합 파이프라인

- Audio Engine (STT): 한국어 인식이 뛰어난 Clova Speech를 채택했습니다. 단순히 텍스트를 추출하는 데 그치지 않고, 스테레오 역상 상쇄 문제를 해결하기 위한 오디오 채널 자동 보정 로직을 적용하여 인식 정확도를 극대화했습니다.

- Smart Capture: 모든 프레임을 분석하는 대신 ORB 특징점 추출 및 pHash 알고리즘을 활용합니다. 강연자의 움직임이나 레이저 포인터 같은 미세한 변화는 무시하고, 실제 슬라이드 내용이 전환되는 유의미한 시점만을 정확히 포착하여 연산 효율을 높였습니다.

![SeSAC:Note 프로젝트 소개 이미지 3](/assets/images/source-archives/sesac-note/intro-03.webp)

- VLM Engine: 캡처된 화면은 Qwen3-VL 모델을 통해 분석됩니다. 이미지 내의 단순 텍스트뿐만 아니라 수식, 도표, 이미지 간의 관계를 논리적으로 해석하여 구조화된 데이터(JSON) 형태로 정보를 추출합니다.

- Multimodal Fusion: 파편화된 이미지 정보와 연속적인 음성 텍스트를 시간 축(Timestamp) 기준으로 동기화합니다. 침묵 구간이나 화면 전환 시점을 기준으로 문맥을 나누어 정보의 공백 없는 통합 컨텍스트를 구성합니다.

#### 품질 보증 및 최적화

- Judge Agent: 생성된 학습 노트의 신뢰성을 검증합니다. Groundedness(근거성)와 요약 품질을 평가하며, 기준 점수(7.0) 미달 시 구체적인 피드백을 생성 엔진에 전달하여 즉시 내용을 보완하거나 재생성하는 자가 교정 루프를 수행합니다.

- 비동기 병렬 처리: 기존의 순차적인 처리 방식에서 모든 과정을 동시에 실행하는 비동기 파이프라인을 구축했습니다.

#### 지능형 질의응답

- LangGraph 기반 추론: 단순한 챗봇을 넘어 복합적인 추론이 가능하도록 상태 기반의 LangGraph를 도입했습니다.

  - Flash Mode: 생성된 노트를 즉시 검색하여 핵심 정보를 빠르게 답변합니다.

  - Thinking Mode: 답변의 근거가 부족할 경우 원본 데이터(STT/VLM)를 다시 탐색하고 교차 검증하여 논리적이고 자세한 답변을 생성합니다.

- Hallucination 방지: 모든 답변은 생성된 학습 노트를 근거로 활용하며, 답변 시 참고한 슬라이드나 음성 구간 정보를 함께 제시하여 신뢰도를 확보했습니다.

### 4. 서비스 아키텍쳐

![SeSAC:Note 프로젝트 소개 이미지 4](/assets/images/source-archives/sesac-note/intro-04.webp)

### 5. 서비스 시연

#### 서비스 링크

[![link icon](/assets/images/source-archives/sesac-note/intro-05.svg)SeSAC:Note](https://re-view-ten.vercel.app/login)

#### 사진

로그인 → 영상 업로드 → 요약 페이지 → 상세 요약 순서로 이용합니다.

### 데모 영상

[시연 영상](https://www.youtube.com/watch?v=QtGnGrilkV4)

### 발표 영상

[발표 영상](https://www.youtube.com/watch?v=2bJcU1skByw)

{% endraw %}

## 관련 글

- [12. SeSAC:Note 슬라이드 캡처 중복 제거 — 비교 실험]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-source-capture-dedup %})
- [13. SeSAC:Note Summarizer 배치 크기 — 실험과 결과 기록]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-source-summarizer-batch %})

- [구간별 요약과 AI 튜터 시연]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-demo-walkthrough %})
