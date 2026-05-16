---
title: "고급 LLM 프롬프팅 전략"
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
date: 2026-01-22 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 고급 LLM 프롬프팅 전략

## 수업 위치

이 수업은 Basic Prompting 이후 “어떤 유형의 문제에 어떤 프롬프팅 전략을 쓸 것인가”를 다룬다. 강의자료에서는 고급 프롬프팅을 크게 지시사항 기반, 예시 기반, 검증 기반으로 나누었다.

기본 프롬프트가 역할과 지시를 명확히 하는 단계라면, 고급 프롬프팅은 추론 과정, 하위 질문, 여러 후보 답변, 자기 검증을 설계하는 단계다.

## 핵심 개념

> **요약**
> 고급 프롬프팅은 모델이 바로 답하지 않고 계획, 추론, 예시 모방, 자기 검증을 거치도록 유도하는 입력 설계다. Zero-shot CoT, Plan-and-Solve, CoT, Self-Ask, Self-Consistency, Self-Verification, Self-Refine, Tree-of-Thoughts를 다룬다.

## 주요 내용

### 1. 지시사항 기반 기법
- **Zero-shot CoT**: 예시 없이 “단계별로 생각하라”는 지시로 추론 과정을 유도
- **Plan-and-Solve**: 먼저 계획을 세우고, 그 계획에 따라 하위 과제를 수행하도록 유도
- 장점: 예시 없이 적용 가능
- 한계: 단계 누락, 형식 불안정, 그럴듯한 오답 가능성

### 2. 예시 기반 기법
- **Few-shot**: 입력/출력 예시를 제공해 패턴을 학습시킴
- 모델에게 **단계별 추론 과정**을 유도하는 기법
- **Chain-of-Thought**: 예시에 중간 추론 과정을 포함
- **Self-Ask**: 후속 질문과 중간 답변을 구조화해서 최종 답변을 도출
- **Least-to-most**: 문제를 하위 질문으로 분해하고 순차적으로 해결

### 3. 출력 형식 제어
- JSON, 마크다운, 표 등 **구조화된 출력** 유도
- 출력 스키마 명시적 정의
- 평가나 후처리가 필요한 경우 형식을 명확히 고정

### 4. 검증 기반 기법
- 동일 질문에 여러 번 추론 후 **다수결** 방식으로 최종 답 선택
- **Self-Verification**: 정방향 추론 결과를 역방향으로 검산
- **Self-Refine**: 초기 답변에 피드백을 만들고 다시 수정
- 하나의 추론 경로가 아닌 **여러 사고 경로를 탐색**
- 각 경로를 평가하고 가장 유망한 경로 선택
- 복잡한 의사결정 문제에 효과적

## 선택 기준

| 상황 | 우선 검토할 기법 |
|---|---|
| 단순 분류/추출 | Zero-shot, 출력 형식 제어 |
| 예시가 있어야 형식이 안정되는 작업 | Few-shot, In-context Learning |
| 계산/논리 추론 | CoT, Plan-and-Solve |
| 복합 질문 | Self-Ask, Least-to-most |
| 답변 변동성이 큰 문제 | Self-Consistency |
| 초안 품질 개선 | Self-Refine |
| 여러 해결 경로 탐색 | Tree-of-Thoughts |

## 실습 연결

Day2 노트북은 Day1의 MMLU 평가 코드를 재사용해 고급 프롬프팅 기법을 비교한다. 핵심은 prompt template을 여러 개 만들고, 같은 평가 함수로 baseline, custom prompt, advanced prompt 결과를 비교하는 것이다.

temperature를 조절해 다양한 답변을 생성하는 부분은 Self-Consistency와 직접 연결된다. 낮은 temperature는 안정적인 답변에 유리하고, 높은 temperature는 다양한 후보를 얻는 데 유리하다. 다만 다양성은 비용과 평가 복잡도를 증가시킨다.

## 흐름도

![고급 LLM 프롬프팅 전략 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-prompt-engineering-rag-w05d02-advanced-prompting-diagram-1.svg)

## 관련 글

- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %})
- [프롬프트 설계 연습]({% post_url 2026-01-21-upstage-practice-w05-prompt-design %})
- [RAG Fundamentals]({% post_url 2026-01-23-upstage-course-w05d03-rag-basics %})
