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

Tiny LLM from Scratch는 작은 언어 모델을 직접 만들어보면서 LLM의 내부 구조를 이해하는 학습 주제다. 목표는 대형 모델을 대체하는 것이 아니라 tokenizer, transformer, training loop, inference 흐름을 손으로 확인하는 것이다.

## 왜 작은 모델을 직접 만드는가

LLM API만 사용하면 모델 내부에서 어떤 일이 일어나는지 감각이 흐려지기 쉽다. 작은 모델을 직접 구현하면 다음 개념을 더 정확히 이해할 수 있다.

| 개념 | 확인할 수 있는 것 |
| --- | --- |
| Tokenizer | 텍스트가 token id로 바뀌는 방식 |
| Embedding | token이 vector로 표현되는 방식 |
| Attention | context 안에서 token이 서로를 참조하는 방식 |
| Loss | 모델이 다음 token 예측을 어떻게 학습하는지 |
| Sampling | temperature, top-k, top-p가 출력에 미치는 영향 |
| Quantization | 정확도와 메모리 사용량의 trade-off |

## 학습 단계

```text
데이터 준비
  -> tokenizer 학습
  -> transformer 구현
  -> training loop 작성
  -> inference와 sampling
  -> quantization
  -> 간단한 배포
```

처음부터 큰 corpus와 큰 모델을 목표로 잡으면 학습 비용이 커진다. 작은 데이터와 작은 parameter 수로 전체 흐름을 끝까지 통과하는 것이 더 중요하다.

## 실무에서 얻는 이점

작은 모델을 직접 만들어보면 대형 LLM API를 사용할 때도 입력 token, context length, attention, latency, memory 사용량을 더 구체적으로 이해할 수 있다. 특히 RAG나 Agent를 설계할 때 "context에 무엇을 넣을지"와 "출력 길이를 어떻게 제한할지"를 더 현실적으로 판단하게 된다.

## 정리

Tiny LLM은 실무용 대체 모델이라기보다 구조 이해용 실험 장치다. 모델을 크게 만드는 것보다, 데이터가 token이 되고, token이 attention을 거쳐 다음 token 확률이 되는 과정을 끝까지 보는 것이 핵심이다.
