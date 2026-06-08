---
title: "AX 시대를 위한 DX"
categories:
- 3.STUDY
- 3-7.AI_ENGINEERING
tags:
- study
- dx
- ax
- ai-coding
- developer-experience
toc: true
date: 2026-05-16 10:00:00 +0900
comments: false
mermaid: true
math: true
---
# AX 시대를 위한 DX

> **한줄 정의**
> AX를 잘 쓰려면 DX가 필요하다. AI가 만든 결과를 검증하고 운영하려면 Git, test, debugging, documentation, architecture 같은 개발 기반이 더 중요해진다.

## DX 없이 AX에 뛰어들면 생기는 문제

| 문제 | 핵심 |
| --- | --- |
| AI 코드를 관리할 수 없음 | git diff, branch, rollback 기반 부족 |
| 결과물이 맞는지 모름 | test culture 부재 |
| 깨진 코드를 고칠 수 없음 | debugging 능력 부재 |
| 뭘 시켜야 할지 모름 | system design 감각 부재 |
| prompt가 안 먹힘 | context engineering 부재 |
| 도구 설정에서 막힘 | CLI, env, package, permission 운영 능력 부재 |

이 문제들은 AI 문제가 아니라 DX 문제다.

## AX 시대의 DX 10가지

| DX 기반 | 기존 의미 | AX 시대 의미 |
| --- | --- | --- |
| Git | version control | AI 변경 diff 검토, 실험 격리, rollback |
| Test | correctness 확인 | AI output 자동 검증과 회귀 방지 |
| Code Review | 동료 코드 검토 | AI 생성 코드의 보안, 성능, 유지보수성 검토 |
| CI/CD | build/test/deploy 자동화 | AI code 품질 gate |
| Documentation | 사람을 위한 README | AI가 읽는 context asset |
| Architecture | 유지보수 구조 | AI가 이해하고 수정하기 쉬운 boundary |
| CLI/Terminal | 개발 편의 | agent interface, MCP, tool 설정 |
| Debugging | log와 stack trace | AI 협력 디버깅, 원인 검증은 사람 |
| Environment | Docker, env var | API key, model 설정, sandbox, 권한 |
| API Design | REST/GraphQL/SDK | tool/function schema, structured output |

## 과거 DX와 현재 DX

| 영역 | 2020년 DX | AX 시대 DX |
| --- | --- | --- |
| 문서화 소비자 | 사람 | 사람 + AI |
| 코드 리뷰 대상 | 동료 코드 | 동료 코드 + AI 생성 코드 |
| 테스트 목적 | 내 코드 correctness | 내 코드 + AI output correctness |
| CLI 활용 | 개발 편의 | agent 운영의 필수 조건 |
| 아키텍처 기준 | 사람이 유지보수하기 좋은 구조 | 사람과 AI가 이해, 수정하기 좋은 구조 |

## 병렬 학습 전략

| 전략 | 방식 |
| --- | --- |
| AI로 project를 만들며 Git 학습 | 생성할 때마다 commit, diff, branch 확인 |
| AI code를 검증하며 test 학습 | test를 먼저 작성하고 AI에게 구현 요청 |
| AI와 debugging하며 system 이해 | 원인 후보를 받고 사람이 검증 |
| context 파일 작성 | CLAUDE.md, AGENTS.md, project rule 정리 |

클래식 DX를 다 끝내고 AX로 가는 것이 아니다. AX를 하면서 DX를 학습하되, 검증 책임은 사람에게 둔다.

## 자가 진단

- `git diff`를 읽고 AI 변경을 판단할 수 있는가
- AI 생성 코드를 검증할 test를 작성할 수 있는가
- error message와 stack trace를 읽을 수 있는가
- project rule을 AI가 읽을 수 있는 형태로 정리했는가
- terminal에서 AI tool을 설정하고 운영할 수 있는가
- AI 없이도 핵심 업무를 수행할 수 있는가
- 문제를 AI에게 던지기 전에 적절한 단위로 분해하는가
- CI/CD로 AI code 품질을 자동 검증하는가
- API/tool schema를 AI가 정확히 쓰도록 설계하는가
- AI가 만든 코드를 3분 안에 설명할 수 있는가

## 내 기준

DX는 AX 이전 시대의 낡은 개념이 아니다.

```text
DX는 사람이 일하기 위한 기반이고
AX는 AI와 함께 일하기 위한 확장이다.
```

DX가 약하면 AI를 많이 쓸수록 통제할 수 없는 변경이 늘어난다.

## 관련 글

- [바이브코딩 & Claude Code 교육 자료]({% post_url 2026-03-28-study-vibe-coding-claude-code %})
- [AI 코딩의 Cognitive Debt]({% post_url 2026-05-05-study-ai-coding-cognitive-debt %})
