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
> LLM · RAG · Agent · 파인튜닝을 책처럼 읽고 실습으로 완성하는 스터디 가이드.

LLM 기초부터 RAG, Agent, 파인튜닝까지 직접 실습하며 배우는 AI 어시스턴트 엔지니어링 스터디입니다.

---

## 추가 정리

### 핵심 요약

AI Assistant Engineering은 LLM, RAG, Agent, fine-tuning을 하나의 제품 관점에서 묶어 보는 학습 주제다. 목적은 챗봇을 만드는 것이 아니라, 사용자의 작업을 안정적으로 도와주는 assistant system을 설계하는 것이다.

### 보충 해설

학습 순서는 LLM 기본 입출력, prompt/context 설계, RAG, tool use, memory, evaluation 순서가 적절하다. assistant는 사용자의 요청을 받아 답변하는 인터페이스이지만, 내부적으로는 검색, 도구 호출, 상태 관리, 안전장치가 함께 필요하다.
