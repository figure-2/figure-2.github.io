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

AI Engineering은 모델 사용법만 배우는 분야가 아니다. LLM 기능을 실제 서비스에 넣으려면 데이터, 백엔드, 평가, 관측, 비용 관리, 배포까지 함께 다뤄야 한다.

이 로드맵은 AI 기능을 제품 안에 안정적으로 넣기 위해 어떤 역량을 순서대로 쌓아야 하는지 정리한 것이다.

## 학습 축

AI Engineering 학습은 네 축으로 나눌 수 있다.

| 축 | 필요한 역량 |
| --- | --- |
| AI / LLM Agent | prompt, RAG, tool use, memory, evaluation |
| Backend | API 설계, queue, worker, DB, auth |
| DevOps / Infra | 배포, 로그, 모니터링, 비용 추적, 장애 대응 |
| Frontend | AI 상태 표시, streaming UI, 오류 복구 UX |

LLM API를 호출하는 코드는 시작점일 뿐이다. 사용자가 실제로 쓰는 기능이 되려면 느린 응답, 실패한 tool call, 잘못된 답변, 비용 폭증, 데이터 권한 문제를 처리해야 한다.

## 단계별 학습 순서

1. LLM API와 prompt 구조를 익힌다.
2. RAG로 외부 지식을 연결한다.
3. tool calling과 workflow를 붙인다.
4. 평가 기준과 테스트셋을 만든다.
5. 로그, 비용, latency를 관측한다.
6. 배포와 장애 대응 흐름을 정리한다.

이 순서가 중요한 이유는 간단하다. 평가와 관측 없이 기능만 붙이면, 개선했는지 망가뜨렸는지 알 수 없다.

## 실전 증명 방법

AI Engineering 역량은 "무엇을 공부했다"보다 "무엇을 안정적으로 만들었는가"로 증명하는 편이 좋다.

| 증명 방식 | 예 |
| --- | --- |
| 기능 구현 | RAG QA, 요약 pipeline, tool agent |
| 품질 평가 | golden set, LLM-as-judge, regression test |
| 운영 기록 | latency, token cost, 실패율, fallback |
| 보안/권한 | 사용자별 data scope, secret 관리, audit log |

## 정리

AI Engineering 로드맵은 모델 학습 순서가 아니라 제품화 순서에 가깝다. 모델을 호출하는 능력보다, 그 호출이 실패했을 때 시스템이 어떻게 버티는지를 설계하는 능력이 더 중요해진다.
