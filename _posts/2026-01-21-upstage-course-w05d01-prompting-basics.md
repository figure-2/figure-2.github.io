---
title: "Introduction & Basic Prompting"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- prompt-engineering
- course-note
- prompt-engineering-rag
toc: true
date: 2026-01-21 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Introduction & Basic Prompting

## 수업 위치

이 수업은 Prompt Engineering 주차의 출발점이다. LLM을 “질문하면 답하는 모델”로만 보지 않고, 입력에 어떤 역할, 맥락, 예시, 제약을 넣느냐에 따라 출력이 달라지는 확률 기반 시스템으로 이해하는 것이 목표다.

강의자료에서는 LLM의 한계, Alignment, Prompt Engineering의 역할을 먼저 설명한 뒤, Basic Prompting의 구성요소인 Role, Instruction/Content, Persona Injection, In-context Learning으로 넘어갔다.

## 핵심 개념

> **요약**
> LLM은 다음 토큰을 예측하는 확률 기반 모델이다. 사람의 의도와 모델의 표면적 생성 사이에는 간극이 있으므로, Prompt Engineering은 이 간극을 줄이기 위해 역할, 지시사항, 예시, 출력 형식, 외부 맥락을 설계하는 활동이다.

## 주요 내용

### 1. Large Language Models
- LLM: 수많은 파라미터를 통해 방대한 텍스트를 학습하여 인간 언어 패턴을 이해/생성하는 모델
- 본질적으로 **다음에 올 단어를 맞히는 확률 기반 모델**
- ChatGPT, Solar LLM, LLaMA, Gemini, Claude, DeepSeek 등
- 관련: LLM

### 2. 기계와 사람의 간극
- 인간: 질문의 **의도와 맥락**을 파악하여 적절한 대응 선택
- LLM: 주어진 문맥에서 가장 **그럴듯한 다음 응답** 생성
- 개연성, 잘못된 정보, 표면적 요구 우선 처리 등 한계
- 관련: Alignment

### 3. 정렬(Alignment)
- 기계가 인간처럼 행동하도록 **간극을 줄이는 행위**
- AI 결과물이 인간의 기대치, 윤리적 기준, 의도와 일치하도록 방향 맞춤
- 사람 -> 모델 방향: **프롬프트 엔지니어링**으로 유도
- 모델 -> 사람 방향: 추가적인 학습을 통한 정렬
- 관련: Alignment, 프롬프트 엔지니어링

### 4. Basic Prompting
- 프롬프트 구성 요소:
  - **Role**: 모델의 관점과 정체성 정의 (System, User, Assistant)
  - **Instruction/Content**: 수행할 구체적 행동
- **시스템 프롬프트**: 최상위 레벨 명령어 집합, 고정된 컨텍스트
- **인풋 프롬프트**: 사용자가 직접 입력하는 가변적 요청
- 출력 형식, 제약 조건, 평가 기준을 명시해야 재현성이 높아짐

### 5. Persona Injection
- 모델에 특정 역할/성격을 부여하는 기법
- 단순 역할명보다 “그 역할이 어떤 기준으로 판단하고 어떤 말투를 쓰는지”까지 설명할수록 효과적

### 6. In-context Learning
- 프롬프트 내 예시를 통해 모델이 패턴을 학습하도록 유도
- 예시는 정답뿐 아니라 출력 형식과 판단 기준을 전달하는 역할도 함

## 실습 연결

Day1 노트북에서는 MMLU 데이터셋을 사용해 baseline prompt와 custom prompt를 비교했다. 실습 흐름은 다음과 같다.

```text
MMLU 데이터 로드
  -> subject / sample 구조 확인
  -> baseline prompt 작성
  -> Solar API 호출
  -> 정답 추출 함수 작성
  -> 채점
  -> custom prompt / few-shot prompt 비교
```

이 실습에서 중요한 부분은 “프롬프트를 바꿨다”가 아니라 “성능을 비교할 기준을 만들었다”는 점이다. MMLU처럼 정답이 있는 문제에서는 모델 출력에서 A/B/C/D를 안정적으로 추출하고, gold answer와 비교해야 prompt 변경 효과를 판단할 수 있다.

## 수업에서 남길 체크포인트

- 시스템 프롬프트와 사용자 프롬프트의 역할을 구분할 수 있다.
- Role, Instruction, Content, Example이 각각 어떤 문제를 줄이는지 설명할 수 있다.
- Persona Injection은 말투뿐 아니라 판단 기준에도 영향을 준다.
- Few-shot 예시는 출력 형식을 고정하는 데도 효과적이지만 token 비용이 증가한다.
- prompt 개선은 느낌이 아니라 baseline 대비 평가 결과로 확인해야 한다.

## 흐름도

![Introduction & Basic Prompting 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-prompt-engineering-rag-w05d01-prompting-basics-diagram-1.svg)

## 관련 글

- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [기본 프롬프팅 튜토리얼 실습]({% post_url 2026-01-21-upstage-practice-prompting-practice %})
- [고급 LLM 프롬프팅 전략]({% post_url 2026-01-22-upstage-course-w05d02-advanced-prompting %})
