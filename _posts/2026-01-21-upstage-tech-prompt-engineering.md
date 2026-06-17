---
title: "Prompt Engineering"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-5.PROMPT_ENGINEERING_RAG
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- prompt-engineering
- tech-note
- prompt-engineering-rag
toc: true
date: 2026-01-21 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Prompt Engineering

> **한줄 정의**
> LLM에 최적의 출력을 유도하기 위한 입력 설계 기법. 프롬프트의 구조, 예시, 지시 방식에 따라 모델 성능이 크게 달라진다.
>
> **수업 근거**
> `02-Basic_Prompting.pptx.pdf`, `03-Advanced LLM Prompting Strategies_1.pdf`, `2-04-Advanced LLM Prompting Strategies_2.pdf`, `SeSAC_2강_Tutorial.ipynb`, Day1/Day2 Daily Mission 노트북을 바탕으로 정리했다.

## 핵심 이해

프롬프트 엔지니어링은 "질문을 잘 쓰는 법"이라기보다 **LLM이 수행할 작업의 입력 계약을 설계하는 일**에 가깝다. 모델에게 역할을 부여하고, 문제를 구조화하고, 예시를 제공하고, 출력 형식을 제한하고, 결과를 검증하는 전체 과정이 포함된다.

기본 단위는 Zero-shot, One-shot, Few-shot이다. Zero-shot은 예시 없이 지시만으로 수행하고, Few-shot은 몇 가지 예시를 제공하여 모델이 답변 패턴을 따라오게 만든다. 시스템 프롬프트는 모델의 역할, 톤, 금지 조건, 모르는 경우의 행동을 정하는 상위 지시 레이어다.

MMLU 실습에서 확인한 것처럼 프롬프트의 품질은 최종 답변 문장만 보고 판단하면 안 된다. 객관식 문제에서는 정답률뿐 아니라 출력 형식 준수, 선택지 추출 가능성, temperature에 따른 안정성까지 함께 봐야 한다.

## 프롬프트 구성 요소

| 요소 | 역할 | 예시 |
| --- | --- | --- |
| Role | 모델이 맡을 관점과 전문성 정의 | "너는 초등학생 과학 튜터다" |
| Task | 수행할 작업을 명확히 지정 | "리뷰 감성을 긍정/부정/중립으로 분류" |
| Context | 판단에 필요한 배경 정보 제공 | 문서, 표, 사용자 조건 |
| Examples | 원하는 입출력 패턴을 보여줌 | Few-shot 예시 |
| Constraints | 금지 사항과 출력 형식 제한 | "A/B/C/D 중 하나만 출력" |
| Verification | 결과 검토 기준 제공 | "계산 오류가 있는지 확인" |

이 요소를 한 문장에 섞어 쓰면 수정하기 어렵다. 실습에서는 역할, 입력, 출력 형식, 평가 기준을 분리해 둬야 어떤 변화가 결과를 바꿨는지 추적할 수 있다.

## 주요 기법

### Zero-shot / Few-shot

Zero-shot은 baseline을 만들 때 적합하다. 빠르고 비용이 낮지만 작업 기준이 모호하면 결과가 흔들릴 수 있다. Few-shot은 분류 기준이나 출력 양식을 예시로 보여줄 수 있어 안정성을 높인다. 다만 예시가 편향되면 모델도 그 편향을 따라갈 수 있다.

### Chain of Thought와 변형 기법

**Chain of Thought(CoT)** 프롬프팅은 복잡한 추론 문제에 효과적이다. "단계별로 생각하세요(Think step by step)"와 같은 지시로 모델이 중간 추론 과정을 명시하게 하여 정확도를 높인다. Self-Consistency는 여러 CoT 경로를 생성하고 다수결로 최종 답을 선택하는 방법이다.

고급 프롬프팅에서는 문제를 먼저 계획한 뒤 풀이하는 Plan-and-Solve, 질문을 하위 질문으로 쪼개는 Self-Ask, 쉬운 문제부터 해결하는 Least-to-Most, 여러 추론 경로를 검토하는 Tree of Thoughts 같은 기법을 다룬다. 핵심은 "모델에게 더 길게 말하게 한다"가 아니라 **복잡한 문제를 검증 가능한 중간 단계로 나누는 것**이다.

### Self-Verification / Self-Refine

Self-Verification은 모델이 생성한 답을 다시 검토하게 하는 방식이다. Self-Refine은 초안 생성, 피드백, 수정의 반복 구조를 만든다. 두 방법 모두 유용하지만, 검증자도 LLM이라는 한계가 있다. 수치 계산, 코드 실행, 정책 판단처럼 정확성이 중요한 작업은 외부 테스트나 규칙 기반 검증과 함께 써야 한다.

### Prompt Chaining

프롬프트 체이닝(Prompt Chaining)은 복잡한 작업을 여러 단계의 프롬프트로 분해하여 순차적으로 처리한다. 각 단계의 출력이 다음 단계의 입력이 되며, RAG 시스템과 결합하면 강력한 파이프라인을 구성할 수 있다. 프롬프트 인젝션 공격에 대한 방어도 설계 단계부터 고려해야 한다.

```text
입력 정리
  -> 정보 추출
  -> 근거 검색
  -> 초안 생성
  -> 검증
  -> 최종 답변
```

## 언제 쓰는가

프롬프트 엔지니어링은 다음 상황에서 특히 중요하다.

- 모델 출력이 평가나 후처리에 연결되는 경우
- 같은 작업을 반복적으로 안정화해야 하는 경우
- RAG처럼 외부 문서를 context로 넣어야 하는 경우
- Tool Calling이나 Agent Workflow에서 다음 행동을 결정해야 하는 경우
- 보안, 정책, 개인정보처럼 답변 제한이 필요한 경우

## 구현 관점

프롬프트 실험은 한 번에 여러 요소를 바꾸면 안 된다. 먼저 baseline을 만들고, 변경한 요소와 평가 결과를 같이 기록해야 한다.

```text
baseline prompt
  -> 출력 형식 제약 추가
  -> few-shot 예시 추가
  -> CoT 또는 plan 단계 추가
  -> self-verification 추가
  -> 평가 결과 비교
```

실서비스에서는 프롬프트 문자열만 관리하지 말고 버전, 평가셋, 실패 사례, 금지 패턴을 함께 관리해야 한다. 그렇지 않으면 "어제는 잘 됐는데 오늘은 왜 안 되지" 같은 문제가 반복된다.

## 주의점

- CoT는 추론 문제에 유용하지만 모든 작업에 필요한 것은 아니다.
- Few-shot 예시는 품질이 낮으면 오히려 성능을 떨어뜨린다.
- 프롬프트에 secret, API key, 내부 정책 원문을 넣으면 안 된다.
- RAG 문서는 명령이 아니라 참고 자료로 취급해야 한다.
- 모델이 확신 있게 말해도 검증 가능한 작업은 별도 평가가 필요하다.

## 관련 강의

- W05D01-프롬프팅-기초
- W05D02-고급-프롬프팅

## 기법 계층도

![Prompt Engineering 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-prompt-engineering-diagram-1.svg)

## 관련 개념

- RAG - 외부 지식 주입을 위한 프롬프트 설계
- Context-Engineering - 컨텍스트 윈도우 최적화
- LLM-보안 - 프롬프트 인젝션 방어
- Agent-Architecture - 에이전트 시스템 프롬프트 설계

## 관련 글

- [Prompting & Context Engineering 기초]({% post_url 2026-01-21-upstage-course-w05d01-prompting-basics %})
- [Advanced Prompting Strategies]({% post_url 2026-01-22-upstage-course-w05d02-advanced-prompting %})
- [MMLU로 확인한 기본 프롬프팅 튜토리얼]({% post_url 2026-01-21-upstage-practice-prompting-practice %})
- [Week05 실습: 프롬프트 설계 연습]({% post_url 2026-01-21-upstage-practice-w05-prompt-design %})

## 참고 자료

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
