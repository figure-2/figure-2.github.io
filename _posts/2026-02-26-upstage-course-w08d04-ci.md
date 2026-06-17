---
title: "GitHub Actions CI"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-8.AGENT_ARCHITECTURE
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- ci-cd
- course-note
- agent-architecture
toc: true
date: 2026-02-26 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## GitHub Actions CI

## 수업 위치

이 수업은 코드가 “내 컴퓨터에서 돌아간다”를 넘어서 “변경할 때마다 깨지지 않는지 자동으로 확인한다”로 넘어가는 단계다. 강의자료의 문제 상황은 수동 테스트의 한계다. 프롬프트나 tool 호출 로직을 조금 수정했는데, 일부 기능만 손으로 확인하고 머지하면 운영에서 스케줄 조회나 RAG 경로가 깨질 수 있다.

## 핵심 개념

> **요약**
> GitHub Actions를 활용한 CI 파이프라인을 구축한다. 코드 변경 시 lint, format, test, PR comment, AI code review를 자동 실행하고, branch protection을 통해 품질 확인 전 merge를 막는 흐름을 학습한다.

## 주요 내용

### 1. CI 개념
- Continuous Integration의 목적과 가치
- CI 파이프라인 구성 요소: 린트, 테스트, 빌드
- CI를 통한 코드 품질 자동 관리
- 수동 테스트의 한계와 자동화의 필요성

### 2. GitHub Actions 기초
- Workflow, Job, Step 구조
- YAML 기반 워크플로우 정의
- 트리거 이벤트: push, pull_request
- 환경변수와 시크릿 관리
- `paths`, `concurrency`, `workflow_dispatch` 같은 실행 조건

### 3. 에이전트 프로젝트 CI
- Python 프로젝트 CI 구성
- pytest 자동 실행
- Docker 이미지 빌드 자동화
- 코드 품질 검사 (ruff, mypy)
- PR comment와 AI 기반 코드 리뷰

## CI 파이프라인 구조

강의자료와 `day6-mission` workflow 기준으로 CI는 다음 단계로 구성된다.

```text
pull_request / push
  -> lint / format check
  -> test
  -> PR status comment
  -> AI code review
  -> branch protection
```

CI는 개발자를 귀찮게 만드는 절차가 아니라, 반복 검증을 자동화하는 안전장치다. 특히 에이전트 프로젝트에서는 프롬프트, router, tool schema, RAG 검색 로직이 서로 영향을 주기 때문에 “한 기능만 손으로 눌러보기”로는 부족하다.

## 실습에서 중요한 지점

| 항목 | 의미 |
|---|---|
| Ruff | Python lint/format을 자동 검사 |
| pytest | 핵심 기능이 깨졌는지 확인 |
| GitHub Secrets | API key, DB URL 같은 민감값을 안전하게 주입 |
| Branch protection | CI 통과 전 merge 방지 |
| AI code review | 사람이 놓칠 수 있는 변경 위험을 보조 검토 |

pre-commit은 GitHub Actions보다 앞단의 안전장치다. GitHub CI는 최종 수문장이고, pre-commit은 로컬에서 빠르게 실수를 줄이는 장치로 보는 편이 좋다.

## 체크포인트

- CI에는 secret 실제 값이 출력되면 안 된다.
- lint 실패와 test 실패는 의미가 다르므로 로그를 구분해서 봐야 한다.
- unsafe fix는 도구가 자동 수정하지 않는 이유가 있으므로 사람이 판단해야 한다.
- branch protection을 켜야 CI가 실제 merge gate 역할을 한다.
- AI code review는 보조 수단이지 테스트를 대체하지 않는다.

## 흐름도

![GitHub Actions CI 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agent-architecture-w08d04-ci-diagram-1.svg)

## 관련 글

- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %})
- [Docker]({% post_url 2026-01-12-upstage-tech-docker %})
- [Git]({% post_url 2026-01-08-upstage-tech-git %})
