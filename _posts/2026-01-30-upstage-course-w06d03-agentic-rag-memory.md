---
title: "Agentic RAG & Memory Management"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-6.AGENTIC_WORKFLOW
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- rag
- agent-architecture
- course-note
- agentic-workflow
toc: true
date: 2026-01-30 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Agentic RAG & Memory Management

## 핵심 개념

> **요약**
> 기존 RAG에 에이전트의 자율성을 결합한 Agentic RAG를 학습한다. 에이전트가 검색 전략을 스스로 결정하고, 결과를 평가하며, 필요시 재검색하는 능동적 RAG 방식이다. 또한 에이전트의 메모리 관리(단기/장기 메모리)와 Reflexion 패턴을 통한 자기 개선 메커니즘을 다룬다.

## 주요 내용

### 1. RAG에서 Agentic RAG로
- 기존 RAG: 고정된 파이프라인으로 검색 -> 생성
- **Agentic RAG**: 에이전트가 **능동적으로 검색 전략 결정**
  - 검색 필요성 판단
  - 쿼리 재작성
  - 다중 소스 검색
  - 결과 품질 평가 및 재검색
- 관련: Agentic RAG, RAG

### 2. Agentic RAG 패턴
- **Router 패턴**: 질문 유형에 따라 적절한 검색 소스 선택
- **Multi-step 패턴**: 복잡한 질문을 분해하여 순차적 검색
- **Adaptive 패턴**: 검색 결과에 따라 전략 동적 조정
- 관련: Agentic RAG

### 3. 메모리 관리
- **단기 메모리(Short-term)**: 현재 대화 컨텍스트
- **장기 메모리(Long-term)**: 과거 상호작용, 학습된 정보
- **작업 메모리(Working Memory)**: 현재 태스크의 중간 상태
- 메모리 저장소: 벡터 DB, 키-값 저장소, 그래프 DB
- 관련: 에이전트 메모리

### 4. Reflexion 패턴
- 에이전트가 자신의 행동을 **평가하고 반성**하는 메커니즘
- 실패로부터 학습하여 다음 시도 개선
- 반성 결과를 메모리에 저장하여 장기적 개선
- 관련: Reflexion

## 흐름도

![Agentic RAG & Memory Management 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agentic-workflow-w06d03-agentic-rag-memory-diagram-1.svg)

## 연결된 개념
- Agentic RAG
- RAG
- 에이전트 메모리
- Reflexion
- Agentic AI
