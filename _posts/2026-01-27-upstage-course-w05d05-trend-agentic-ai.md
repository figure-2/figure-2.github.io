---
title: "Related Works & Agentic AI Summary"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- agentic-workflow
- course-note
- prompt-engineering-rag
toc: true
date: 2026-01-27 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Related Works & Agentic AI Summary

## 수업 위치

이 수업은 Prompt Engineering 주차를 마무리하면서 두 방향을 정리한다. 하나는 텍스트 중심 prompt가 이미지, 오디오, 비디오까지 확장되는 멀티모달 프롬프팅이고, 다른 하나는 prompt, RAG, memory, tool을 결합한 Agentic AI로의 확장이다.

## 핵심 개념

> **요약**
> 프롬프트 엔지니어링의 핵심 기법을 정리하고, 멀티모달 프롬프팅, 개인화와 장기 지속 메모리, Agentic AI의 발전 방향을 연결한다. 단일 입력-출력 모델에서 도구와 메모리를 가진 에이전트로 확장되는 흐름을 이해한다.

## 주요 내용

### 1. Multimodal Prompting
- 텍스트뿐 아니라 이미지, 오디오, 비디오도 prompt의 일부가 될 수 있음
- 각 modality를 encoder로 벡터화하고 공통 의미 공간에서 정렬
- 이미지 질의응답, 이미지 캡션, 문서 이해, 오디오 요약, 비디오 시점 검색으로 확장

### 2. 개인화와 장기 지속 메모리
- Stateless interaction의 한계를 넘기 위한 persistent memory
- 모든 정보를 저장하는 것이 아니라 discard/update/retain 기준으로 선별
- RAG는 외부 문서 검색이고, persistent memory는 사용자 고유 정보와 선호를 다룸
- 개인화는 보안, 사용자 통제권, 삭제 가능성과 함께 설계해야 함

### 3. 프롬프트에서 에이전트로
- 프롬프트 엔지니어링 -> RAG -> Agentic AI 발전 경로
- 각 단계가 이전 단계를 포함하며 확장
- LLM & Prompt Engineering: Input -> Model -> Output
- AI Agent: 목표 지향 작업을 수행하는 도구 결합 단위
- Agentic AI: 여러 agent를 상황에 맞게 동적으로 조합하는 orchestration

### 4. 향후 발전 방향
- 자율 에이전트의 신뢰성과 안전성
- 인간-AI 협업 패턴의 진화
- 에이전트 생태계의 형성

## 실습 연결

Day5 노트북은 Visual-Text Embedding Alignment를 다룬다. 이미지와 텍스트 설명 후보를 각각 embedding하고, cosine similarity로 어떤 설명이 이미지와 가장 잘 맞는지 확인한다.

```text
이미지 준비
  -> 텍스트 설명 후보 작성
  -> 이미지 embedding
  -> 텍스트 embedding
  -> cosine similarity 계산
  -> 좋은 prompt / 나쁜 prompt 비교
```

이 실습은 멀티모달 모델의 핵심 아이디어를 작게 확인하는 작업이다. 모델이 이미지를 “그림 그대로” 이해하는 것이 아니라, 이미지 특징과 텍스트 의미를 비교 가능한 embedding 공간에 올린 뒤 정렬한다는 점을 확인한다.

## 흐름도

![Related Works & Agentic AI Summary 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-prompt-engineering-rag-w05d05-trend-agentic-ai-diagram-1.svg)

## 관련 글

- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [Visual-Text Embedding Alignment 실습]({% post_url 2026-01-27-upstage-practice-embedding-practice %})
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %})
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %})
