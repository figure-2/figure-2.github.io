---
title: "AI Assistant Engineering"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-assistant
- llm
- rag
- agent
- fine-tuning
toc: true
date: 2026-04-26 00:00:00 +0900
comments: false
mermaid: true
math: true
---

AI Assistant Engineering은 챗봇을 만드는 공부가 아니다. 사용자의 작업을 안정적으로 도와주는 assistant system을 설계하는 학습 주제다.

assistant는 겉으로는 대화 인터페이스처럼 보이지만, 내부에는 LLM, context 설계, RAG, tool use, memory, evaluation, guardrail이 함께 들어간다.

## 학습 범위

| 범위 | 핵심 질문 |
| --- | --- |
| LLM 기본 | 모델은 어떤 입력을 받고 어떤 출력을 만드는가 |
| Context 설계 | 대화, 파일, 사용자 상태를 어떻게 넣을 것인가 |
| RAG | 외부 지식을 어떻게 검색하고 근거로 붙일 것인가 |
| Tool Use | 모델이 어떤 도구를 언제 호출하게 할 것인가 |
| Memory | 세션 밖의 정보를 어디까지 기억할 것인가 |
| Evaluation | 답변 품질과 실패를 어떻게 측정할 것인가 |

## Assistant와 Agent의 차이

assistant는 사용자의 작업을 돕는 제품 인터페이스에 가깝고, agent는 목표를 받아 실행 흐름을 스스로 결정하는 구조에 가깝다. 둘은 겹치지만 같은 말은 아니다.

| 구분 | Assistant | Agent |
| --- | --- | --- |
| 중심 | 사용자 지원 | 목표 실행 |
| 흐름 | 대화와 요청 처리 | 계획, 도구 실행, 반복 |
| 위험 | 잘못된 답변 | 잘못된 행동 |
| 핵심 설계 | context, grounding, UX | tool permission, loop, state |

## 학습 순서

1. 기본 LLM 호출과 prompt 구조를 이해한다.
2. 대화 context와 파일 context를 분리한다.
3. RAG로 외부 지식을 연결한다.
4. tool calling을 붙인다.
5. memory와 user preference를 다룬다.
6. evaluation과 guardrail을 만든다.

## 정리

좋은 assistant는 똑똑한 모델 하나로 만들어지지 않는다. 사용자가 무엇을 하려는지, 어떤 근거를 써야 하는지, 언제 도구를 호출해야 하는지, 어떤 경우에는 답하지 말아야 하는지를 시스템으로 설계해야 한다.
