---
title: "Visual-Text Embedding Alignment 실습"
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
date: 2026-01-27 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Visual-Text Embedding Alignment 실습

> **실습 정보**
> - **주차**: Week 05, Day 05
> - **유형**: Jupyter Notebook
> - **상태**: 완료
> - **원본 자료**: `[Daily Mission] Day5 Visual-Text Embedding Alignment`, `09-Related_Works_Trends.pptx.pdf`

## 실습 목표

이미지와 텍스트가 각각 embedding으로 변환된 뒤 같은 의미 공간에서 비교될 수 있음을 실험한다. 목표는 Visual Prompt Engineering을 "이미지를 잘 설명하는 문장 작성"으로만 보지 않고, **이미지 embedding과 텍스트 embedding의 의미적 정렬 문제**로 이해하는 것이다.

## 핵심 학습 포인트

- 멀티모달 모델은 이미지를 텍스트로 단순 변환하는 것이 아니라, 이미지와 텍스트를 비교 가능한 embedding 공간에 배치한다.
- 좋은 설명은 이미지의 객체, 관계, 수치, 맥락을 더 정확히 담아 이미지 embedding과 가까워진다.
- CLIP 실습에서는 이미지 embedding과 여러 텍스트 설명 embedding 사이의 cosine similarity를 비교한다.
- Visual Prompting은 문장 길이보다 의미 정확성이 중요하다.

## 실습 흐름

### 1. 테스트 이미지 준비

노트북에서는 연도별 카테고리 판매량을 나타내는 선 그래프 이미지를 생성한다. 중요한 것은 이미지가 하나의 시각 객체이며, 이후 모든 텍스트 설명이 같은 이미지를 기준으로 비교된다는 점이다.

### 2. 텍스트 설명 후보 만들기

같은 이미지에 대해 좋은 설명, 부분적으로 맞는 설명, 틀린 설명을 준비한다.

| 후보 | 특징 |
| --- | --- |
| Good | 차트 유형, 축, 카테고리, 증가/감소 추세를 구체적으로 설명 |
| Medium | 차트라는 점은 맞지만 세부 정보가 부족 |
| Bad | 차트 유형이나 내용이 실제 이미지와 다름 |

이 비교는 프롬프트의 품질을 감이 아니라 유사도 지표로 확인하기 위한 장치다.

### 3. 이미지 embedding 생성

CLIP 모델에 이미지를 넣기 전에 전처리를 수행하고, 단일 이미지를 batch 입력 형태로 만들기 위해 `unsqueeze(0)`를 사용한다.

```text
PIL Image -> preprocess -> (channels, height, width)
unsqueeze(0) -> (1, channels, height, width)
encode_image -> image embedding
```

딥러닝 모델은 보통 batch 단위 입력을 기대하므로, 이미지가 한 장이어도 batch 차원을 추가해야 한다.

### 4. 텍스트 embedding 생성

텍스트 설명은 tokenization을 거친 뒤 text encoder를 통해 embedding으로 변환된다. 이미지 embedding과 텍스트 embedding은 같은 의미 공간에 놓이므로 cosine similarity로 비교할 수 있다.

### 5. 유사도 비교와 해석

Good 설명의 유사도가 높게 나오고 Bad 설명의 유사도가 낮게 나온다면, 모델이 이미지와 텍스트의 의미 대응을 어느 정도 학습했다는 뜻이다. 다만 이 점수는 절대적 진실이 아니라 모델이 학습한 embedding 공간 안에서의 상대적 거리다.

## 진행 순서

1. 기준 이미지를 준비하고 시각적 정보를 사람이 먼저 요약한다.
2. Good, Medium, Bad 수준의 텍스트 설명을 만든다.
3. CLIP으로 이미지 embedding을 생성한다.
4. 각 텍스트 설명을 embedding으로 변환한다.
5. 이미지 embedding과 텍스트 embedding의 cosine similarity를 계산한다.
6. 유사도 순위가 설명 품질과 어떻게 연결되는지 해석한다.

## 체크포인트

- [ ] 이미지와 텍스트가 각각 embedding으로 변환된다는 점을 설명할 수 있다.
- [ ] `unsqueeze(0)`가 batch 차원을 추가하기 위한 작업임을 이해했다.
- [ ] 좋은 설명과 나쁜 설명의 차이를 유사도 결과와 연결했다.
- [ ] 멀티모달 prompting에서 문장 길이보다 의미 정확성이 중요함을 확인했다.

## 회고 질문

- 어떤 정보가 빠졌을 때 이미지와 텍스트의 의미 거리가 가장 커졌나?
- 차트 설명에서는 객체 이름, 수치, 추세 중 무엇이 가장 중요했나?
- VLM 기반 서비스에서 "이미지 이해"를 어떻게 평가할 수 있을까?

## 관련 글

- [Related Works & Trends / Agentic AI Summary]({% post_url 2026-01-27-upstage-course-w05d05-trend-agentic-ai %})
- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})
