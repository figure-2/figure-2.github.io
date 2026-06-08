---
title: "Tiny LLM from Scratch"
categories:
- 3.STUDY
- 3-7.AI_ENGINEERING
tags:
- study
- llm
- transformer
- tokenizer
- quantization
toc: true
date: 2026-04-26 00:10:00 +0900
comments: false
mermaid: true
math: true
---
> 노트북에서 직접 만드는 작은 언어 모델 — 데이터 · 토크나이저 · 트랜스포머 · 훈련 · 양자화 · 배포.

데이터 수집부터 토크나이저, 트랜스포머 구현, 훈련, 양자화, 배포까지 — 노트북 환경에서 작은 언어 모델을 직접 만들어보는 스터디입니다.

---

## 추가 정리

### 핵심 요약

Tiny LLM from Scratch는 작은 언어 모델을 직접 만들어보면서 LLM의 내부 구조를 이해하는 학습 주제다. 목표는 큰 모델을 대체하는 것이 아니라 tokenizer, transformer, training loop, inference 흐름을 손으로 확인하는 것이다.

### 보충 해설

작은 모델을 직접 구현하면 대형 LLM API를 사용할 때도 입력 토큰, context length, attention, loss, sampling, quantization의 의미를 더 정확히 이해할 수 있다. 실무 응용보다 구조 이해를 위한 학습으로 보는 것이 맞다.
