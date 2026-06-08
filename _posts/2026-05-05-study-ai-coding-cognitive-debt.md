---
title: "AI 코딩의 Cognitive Debt"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- ai-coding
- cognitive-debt
- agentic-coding
- developer-skill
toc: true
date: 2026-05-05 22:00:00 +0900
comments: false
mermaid: true
math: true
---
# AI 코딩의 Cognitive Debt

> **한줄 정의**
> Cognitive Debt는 AI가 만든 결과를 이해하지 못한 채 수락하면서, 나중에 검증, 수정, 운영 비용으로 돌아오는 부채다.

## 무엇이 위험한가

원본 글은 네 가지 경고를 중심으로 정리되어 있다.

| 위험 | 의미 |
| --- | --- |
| Supervision Paradox | AI를 감독하려면 전문성이 필요한데, AI를 많이 쓰면 그 전문성이 약해질 수 있음 |
| Skill Atrophy | 직접 설계하고 디버깅하는 능력이 줄어듦 |
| Token Cost | 모든 context를 붙이고 반복시키며 비용이 과열 |
| Vendor Lock-in | 특정 도구 없이는 팀이 멈추는 상태 |

핵심은 AI를 쓰지 말자는 것이 아니다. 무엇을 위임하고 무엇을 직접 해야 하는지 경계를 정하자는 것이다.

## 위임해도 좋은 것

| 영역 | 예시 | 이유 |
| --- | --- | --- |
| Boilerplate | CRUD endpoint, config, type definition | 패턴이 명확하고 검증이 쉬움 |
| 탐색적 prototype | library 가능성 확인 | 버릴 코드라 부채가 작음 |
| Refactoring execution | 변수명 변경, 함수 추출, 파일 분리 | 사람이 intent를 정하고 실행만 위임 |
| Test generation | 이미 이해한 코드의 test 작성 | 사람이 logic을 알고 검증 가능 |

## 직접 해야 하는 것

| 영역 | 예시 | 이유 |
| --- | --- | --- |
| Architecture decision | service boundary, data model, API contract | 되돌리기 어렵고 blast radius가 큼 |
| Debugging | reproduce, hypothesis, verify loop | 시스템 이해의 핵심 과정 |
| Code review | AI 생성 코드 포함 모든 코드 | merge 순간부터 사람 책임 |
| Core business logic | payment, auth, data integrity | 틀리면 직접 피해 발생 |

## Token 비용 전략

나쁜 패턴은 "일단 다 붙여"다. context가 많다고 답이 좋아지는 것이 아니다.

| 전략 | 설명 |
| --- | --- |
| Context diet | 필요한 파일과 로그만 제공 |
| Summarize before attach | 긴 문서는 요약 후 핵심만 첨부 |
| Tool schema pruning | 쓰지 않는 tool 정의 제거 |
| Small model routing | 단순 분류와 추출은 작은 모델 사용 |
| Cache reusable context | 반복되는 project rule과 schema cache |

매 요청마다 system prompt, tool schema, repo summary가 선불 token으로 나간다. 이 비용을 보지 않으면 agentic coding 비용은 금방 불어난다.

## 기술 위축을 막는 훈련

| 훈련 | 방식 |
| --- | --- |
| AI 없는 30분 | 먼저 직접 가설과 설계를 써본다 |
| "왜?" 습관 | AI 출력의 이유를 설명할 수 있을 때만 수락 |
| 주간 deep dive | AI가 만든 변경 중 하나를 끝까지 분석 |
| 설명 가능 기준 | 팀원에게 3분 안에 변경을 설명할 수 있어야 merge |

## Vendor Lock-in

AI 없으면 팀이 멈추는 상태는 위험하다.

| 증상 | 리스크 |
| --- | --- |
| 특정 tool 없이는 build/test를 못 함 | 운영 지속성 문제 |
| prompt가 개인에게만 있음 | 팀 지식 손실 |
| AI가 만든 구조를 아무도 설명 못 함 | 유지보수 불가 |
| tool 변경 시 workflow 전체가 멈춤 | vendor 종속 |

AI 도구는 없어도 불편해야지, 없으면 개발이 불가능하면 안 된다.

## 팀 규칙

- AI가 만든 코드는 사람이 소유한다.
- 핵심 설계는 AI에게 초안을 받을 수 있지만 결정은 사람이 한다.
- AI 출력은 test와 review를 통과해야 한다.
- 변경 설명을 못 하면 merge하지 않는다.
- secret, credential, private log는 AI context에 넣지 않는다.
- 비용과 token 사용량을 관찰한다.

## 내 기준

AI coding의 목표는 생각을 생략하는 것이 아니다.

```text
AI에게 실행을 맡기되
문제 이해와 책임은 사람에게 남긴다.
```

Cognitive Debt는 AI 사용량이 아니라 이해하지 못한 변경량에 비례한다.

## 관련 글

- [바이브코딩 & Claude Code 교육 자료]({% post_url 2026-03-28-study-vibe-coding-claude-code %})
- [AX 시대를 위한 DX]({% post_url 2026-05-16-study-dx-for-ax %})
