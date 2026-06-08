---
title: "AI Engineering 개인 학습 로드맵"
categories:
- 3.STUDY
- 3-7.AI_ENGINEERING
tags:
- study
- ai-engineering
- ai-native
- ai-ready-data
- dx
- guide-review
- reference-note
toc: true
date: 2026-04-10 11:50:00 +0900
comments: false
mermaid: true
math: true
---
# AI Engineering 개인 학습 로드맵

## 학습 목적

AI 도구 사용법을 넘어서, AI를 전제로 한 개발 방식, 데이터 준비, 팀 운영, 개발자 역량을 정리한다.

## 정리 범위

| 소주제 | 정리 관점 |
| --- | --- |
| AI Native | AI를 쓰는 팀과 AI로 설계된 팀의 차이 |
| AI-Ready Data | 모델보다 데이터 구조가 병목이 되는 경우 |
| DX for AX | AI 활용 능력과 개발자 경험 |
| Prompting Change | 모델 변경에 따라 prompt 제어면이 달라지는 문제 |
| Tiny LLM | tokenizer, transformer, training 흐름 기초 |

## 작성할 글

| 순서 | 게시글 | 소주제 | 상태 |
| --- | --- | --- | --- |
| 1 | [AI Native 팀 운영 가이드]({% post_url 2026-04-10-study-ai-native-team %}) | 6가지 운영 원칙, 4가지 역할, anti-pattern, 도입 로드맵 | 작성 |
| 2 | [AI-Ready Data 가이드]({% post_url 2026-05-16-study-ai-ready-data %}) | 5가지 기준, 진단 flow, 구조화 단계, RAG 실패 연결 | 작성 |
| 3 | [AX 시대를 위한 DX]({% post_url 2026-05-16-study-dx-for-ax %}) | DX 없이 AX를 쓰면 생기는 문제, AX 시대 DX 10가지, 자가 진단 | 작성 |
| 4 | [모델 변경과 프롬프트 변화]({% post_url 2026-05-31-study-model-prompt-change %}) | 모델별 control surface, prompt rewriting 기준, tool boundary, 성공 기준 | 작성 |
| 5 | [Tiny LLM from Scratch]({% post_url 2026-04-26-study-tiny-llm-from-scratch %}) | data, tokenizer, transformer, training, quantization, deployment | 작성 |

## 후속 분리 후보

| 후보 글 | 분리 이유 |
| --- | --- |
| AI Native 운영 원칙 상세 | 1차 글에 반영 완료. 실제 팀 적용 사례는 후속 가능 |
| Tiny LLM 실습 로그 | 1차 글은 학습 경로. 실제 구현/훈련/양자화는 단계별 실습 글로 후속 가능 |
| Prompting 변화 체크리스트 | 1차 글에 반영 완료. 모델 버전과 공식 가이드 변경 가능성이 커서 별도 업데이트 단위로 관리 |

## 작성 기준

AI Engineering 글은 생산성 주장보다 검증 비용, 운영 책임, 팀의 작업 방식 변화를 중심으로 쓴다.
