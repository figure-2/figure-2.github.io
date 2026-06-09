---
title: "AI Agent 완벽 가이드 1: 정의와 Workflow 구분"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- workflow
- tools
- memory
toc: true
date: 2026-04-04 01:00:00 +0900
comments: false
mermaid: true
math: true
---
AI Agent를 설계할 때 가장 먼저 구분해야 하는 것은 `Workflow`와 `Agent`다. 둘 다 LLM을 쓰지만, 실행 흐름을 누가 결정하는지가 다르다. Workflow는 사람이 정한 경로를 따르고, Agent는 실행 중에 다음 행동을 선택한다.

이 글은 AI Agent의 기본 정의와 구성 요소, 그리고 Workflow와 Agent의 차이를 정리한다.

## AI Agent란 무엇인가?

Lilian Weng의 정리를 빌리면, AI Agent는 대략 다음 네 요소의 조합으로 볼 수 있다.

```text
Agent = LLM + Memory + Planning + Tools
```

### LLM (두뇌)

추론과 의사결정의 핵심 엔진이다. 자연어를 이해하고, 계획을 세우고, 도구 사용 여부를 결정한다.

### Memory (기억)

단기 기억은 현재 대화와 작업 맥락을 유지하고, 장기 기억은 과거 경험이나 외부 지식을 다시 참조하게 만든다.

### Planning (계획)

복잡한 목표를 실행 가능한 단계로 나누고, 필요하면 중간 결과를 보고 계획을 수정한다.

### Tools (도구)

검색, API, DB, 코드 실행기처럼 모델 밖의 세계에 접근하는 수단이다. 도구가 붙는 순간 LLM은 답변 생성기를 넘어 작업 실행기로 확장된다.

### Workflow vs Agent: 핵심 구분

Anthropic의 "Building Effective Agents"에서는 Workflow와 Agent를 명확히 구분한다.

| 구분 | Workflow | Agent |
| --- | --- | --- |
| 실행 흐름 | 코드로 미리 정의 | LLM이 실행 중 선택 |
| 재현성 | 같은 입력이면 대체로 같은 경로 | 같은 입력이라도 다른 경로 가능 |
| 장점 | 예측 가능, 디버깅 쉬움, 비용 관리 쉬움 | 열린 문제, 도구 선택, 재시도에 강함 |
| 리스크 | 유연성이 낮음 | 비용과 latency가 가변적 |
| 예시 | 문서 번역 파이프라인, 이메일 분류 | 코드 디버깅 에이전트, 리서치 에이전트 |

---

## 추가 정리

### 핵심 요약

Workflow는 사람이 정한 경로를 LLM이 따라가는 구조이고, Agent는 실행 중에 다음 행동을 스스로 선택하는 구조다. 두 개념을 섞어 쓰면 설계 복잡도가 불필요하게 올라간다.

### 보충 해설

실무에서는 먼저 Workflow로 충분한지 판단해야 한다. 입력 유형이 명확하고 단계가 고정되어 있으면 Workflow가 더 안전하고 저렴하다. Agent가 필요한 경우는 도구 선택, 경로 선택, 재시도 판단이 실행 중에 달라지는 문제다.
