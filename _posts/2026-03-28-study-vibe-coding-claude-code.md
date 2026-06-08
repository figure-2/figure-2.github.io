---
title: "바이브코딩 & Claude Code 교육 자료"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- vibe-coding
- claude-code
- ai-coding
- education
toc: true
date: 2026-03-28 01:00:00 +0900
comments: false
mermaid: true
math: true
---
# 바이브코딩 & Claude Code 교육 자료

> **한줄 정의**
> 바이브코딩 교육의 핵심은 AI에게 코드를 맡기는 법이 아니라, 작업을 작게 쪼개고 검증 가능한 계약으로 실행하는 법을 배우는 것이다.

## 정리 범위

이 글은 PM 팀 1~2시간 교육용 자료를 개인 학습용 흐름으로 재구성한 것이다. 자료 목록을 나열하지 않고, 교육 설계와 실습 흐름 중심으로 정리한다.

## 교육 목표

| 목표 | 설명 |
| --- | --- |
| AI coding 이해 | AI가 잘하는 일과 위험한 일을 구분 |
| Claude Code 흐름 이해 | repo 읽기, 수정, 검증, diff 확인 흐름 이해 |
| 작업 계약 작성 | 모호한 요청을 실행 가능한 단위로 변경 |
| 검증 루프 습득 | test, lint, diff, review로 결과 확인 |
| 위험 경계 인식 | secret, 삭제, 외부 전송, 권한 변경 차단 |

## 바이브코딩의 오해

| 오해 | 정정 |
| --- | --- |
| 그냥 느낌대로 말하면 된다 | 요구사항, 제약, 완료 기준이 필요 |
| AI가 코드를 다 알아서 고친다 | context를 잘 줘야 하고 결과는 검증해야 함 |
| 빠르면 좋은 것이다 | 빠르게 틀린 코드를 만들 수도 있음 |
| prompt가 전부다 | git, test, architecture, debugging이 더 중요할 때가 많음 |

## Claude Code 작업 루프

```text
Goal
  -> Context 제공
  -> Plan
  -> Edit
  -> Test
  -> Diff Review
  -> Iterate
```

| 단계 | 사람이 해야 할 일 |
| --- | --- |
| Goal | 결과물과 제외 범위 정의 |
| Context | 관련 파일, 제약, 기존 규칙 제공 |
| Plan | AI가 잡은 접근이 위험하지 않은지 확인 |
| Edit | 변경 범위가 너무 넓어지지 않게 제한 |
| Test | 자동 검증과 수동 확인 병행 |
| Diff Review | 내가 이해할 수 있는 변경만 수락 |
| Iterate | 실패 원인을 좁혀 재요청 |

## 좋은 작업 지시 형식

```text
목표:
  무엇을 바꿀 것인가

범위:
  어떤 파일/기능까지만 볼 것인가

제약:
  바꾸면 안 되는 것, 보안 경계, 호환성

완료 기준:
  어떤 테스트/상태면 완료인가

출력:
  diff 요약, 검증 결과, 남은 리스크
```

## 실습 구성

| 시간 | 내용 | 산출물 |
| --- | --- | --- |
| 10분 | AI coding 개념과 위험 | 좋은/나쁜 요청 예시 |
| 20분 | 작은 수정 실습 | diff와 test 결과 |
| 20분 | 기능 추가 실습 | 작업 계약과 검증 루프 |
| 20분 | 실패 복구 실습 | error log, 원인 가설, 재시도 |
| 10분 | review | 배운 점과 팀 적용 규칙 |

## PM에게 중요한 포인트

PM이 꼭 코드를 직접 다 알아야 하는 것은 아니다. 하지만 AI에게 줄 작업을 검증 가능한 단위로 쪼개야 한다.

| PM 역할 | AI coding에서 의미 |
| --- | --- |
| 요구사항 정의 | ambiguity를 줄임 |
| acceptance criteria | AI output 검증 기준 |
| edge case 정리 | 놓치기 쉬운 실패 조건 |
| 우선순위 | AI가 과도한 변경을 하지 않게 제한 |
| 리뷰 관점 | 사용자 영향과 product risk 확인 |

## 금지해야 할 요청

| 요청 | 문제 |
| --- | --- |
| "전체 코드 정리해줘" | 변경 범위가 너무 넓음 |
| "에러 알아서 고쳐줘" | 원인 분리 없이 patch만 반복 |
| "테스트는 나중에" | 검증 기준 부재 |
| "배포까지 해줘" | 외부 상태 변경 위험 |
| secret 포함 로그 전달 | 보안 사고 |

## 내 기준

바이브코딩은 무계획 coding이 아니다.

```text
작게 요청
  -> 명확히 제한
  -> 자동 검증
  -> diff 이해
  -> 위험 작업은 사람 승인
```

이 루프를 지키면 AI는 속도를 높인다. 이 루프를 빼면 AI는 이해하지 못한 변경을 빠르게 쌓는다.

## 관련 글

- [AI 코딩의 Cognitive Debt]({% post_url 2026-05-05-study-ai-coding-cognitive-debt %})
- [AX 시대를 위한 DX]({% post_url 2026-05-16-study-dx-for-ax %})
