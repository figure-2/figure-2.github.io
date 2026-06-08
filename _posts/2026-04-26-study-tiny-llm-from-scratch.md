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
# Tiny LLM from Scratch

> **한줄 정의**
> Tiny LLM from Scratch는 데이터, tokenizer, transformer, training, quantization, deployment를 작은 규모로 직접 이어보는 학습 경로다.

## 정리 범위

이 항목은 상세 구현보다 작은 언어 모델을 직접 만들어 보는 학습 흐름을 정리하는 성격이다.

```text
Data
  -> Tokenizer
  -> Transformer
  -> Training
  -> Quantization
  -> Deployment
```

이 글은 해당 흐름을 학습 체크리스트로 재구성한다.

## 1. Data

| 질문 | 확인할 것 |
| --- | --- |
| 어떤 text를 학습할 것인가 | domain, license, 품질 |
| 중복을 제거했는가 | near-duplicate와 boilerplate |
| train/validation을 나눴는가 | leakage 방지 |
| token 분포를 봤는가 | 긴 문장, 특수문자, 언어 혼합 |

작은 LLM에서는 데이터 품질이 model 크기보다 더 크게 드러난다.

## 2. Tokenizer

Tokenizer는 text를 model이 처리할 token ID로 바꾼다.

| 항목 | 의미 |
| --- | --- |
| Vocabulary | model이 구분하는 token 목록 |
| BPE/Unigram | subword를 만드는 방식 |
| Special Token | BOS, EOS, PAD, UNK 등 |
| Encoding | text -> token ID |
| Decoding | token ID -> text |

Tokenizer가 나쁘면 model은 같은 parameter로도 더 어려운 문제를 풀게 된다.

## 3. Transformer

작은 LLM 구현의 핵심 block은 다음이다.

```text
Token Embedding
  -> Positional Encoding
  -> Multi-Head Attention
  -> Feed Forward
  -> Layer Norm
  -> LM Head
```

| 구성 | 역할 |
| --- | --- |
| Embedding | token ID를 vector로 변환 |
| Attention | 이전 token들과의 관계 계산 |
| Feed Forward | token별 비선형 변환 |
| Layer Norm | 학습 안정화 |
| LM Head | 다음 token 확률 계산 |

## 4. Training

언어 모델의 기본 목표는 다음 token 예측이다.

```text
input:  오늘 날씨가
target: 좋다
```

| 항목 | 확인할 것 |
| --- | --- |
| Loss | cross entropy |
| Batch | GPU memory와 안정성 |
| Learning Rate | warmup, decay |
| Validation | overfitting 확인 |
| Checkpoint | 재시작과 비교 실험 |

작은 model에서는 overfitting을 빨리 확인할 수 있다. validation loss와 sample output을 같이 본다.

## 5. Quantization

Quantization은 weight precision을 낮춰 memory와 inference cost를 줄인다.

| 방식 | 의미 |
| --- | --- |
| FP16/BF16 | training/inference 표준 저정밀 |
| INT8 | memory와 속도 절감 |
| INT4 | 더 강한 압축, 품질 손실 가능 |
| KV Cache 최적화 | 긴 generation에서 memory 병목 완화 |

작은 LLM 실습에서는 quantization 자체보다 precision 변화가 output과 latency에 미치는 영향을 관찰하는 것이 중요하다.

## 6. Deployment

| 방식 | 확인할 것 |
| --- | --- |
| Local inference | model load, generation speed |
| API server | request/response schema |
| Streaming | token 단위 응답 |
| Monitoring | latency, error, memory |
| Safety | 입력 제한, output filter |

배포까지 해봐야 training artifact가 실제 assistant나 tool에서 어떻게 쓰이는지 알 수 있다.

## 학습 산출물

| 단계 | 산출물 |
| --- | --- |
| Data | 작은 corpus와 preprocessing script |
| Tokenizer | vocab, encode/decode test |
| Model | transformer block 구현 |
| Training | loss curve, checkpoint |
| Evaluation | sample generation과 validation loss |
| Quantization | size/latency 비교 |
| Deployment | local API 또는 notebook demo |

## 내 기준

Tiny LLM의 목적은 큰 모델을 이기는 것이 아니다.

```text
데이터가 token이 되고
token이 embedding이 되고
attention이 다음 token을 예측하고
훈련과 추론 비용이 어떻게 생기는지
손으로 연결해보는 것
```

이 과정을 한 번 밟으면 RAG, fine-tuning, prompt cost, context window를 훨씬 덜 추상적으로 이해하게 된다.

## 관련 글

- [AI Assistant Engineering]({% post_url 2026-04-26-study-ai-assistant-engineering %})
- [모델 변경과 프롬프트 변화]({% post_url 2026-05-31-study-model-prompt-change %})
