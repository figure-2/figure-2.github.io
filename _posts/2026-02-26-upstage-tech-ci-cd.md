---
title: "CI/CD"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-3.DEV_ENV_GIT_DOCKER
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- ci-cd
- tech-note
- dev-env-git-docker
toc: true
date: 2026-02-26 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# CI/CD

> **한줄 정의**
> 지속적 통합(Continuous Integration)과 지속적 배포(Continuous Delivery/Deployment). 코드 변경을 자동으로 빌드·테스트·배포하는 파이프라인이다.

## 학습 맥락

CI/CD는 개발환경 주차의 Git, Docker, 클라우드 학습이 실제 서비스 운영으로 이어지는 지점이다. 업스테이지 과정에서는 GitHub Actions CI와 CD를 통해 에이전트 프로젝트를 반복 가능하게 테스트하고 배포하는 흐름을 다뤘다.

LLM/Agent 서비스는 "로컬에서 한 번 실행됨"만으로는 충분하지 않다. 프롬프트, 모델 설정, API 호출, Docker 이미지, secret, 배포 환경이 조금만 달라져도 결과가 바뀔 수 있다. CI/CD는 이 변동성을 줄이고, 변경사항이 서비스에 들어가기 전에 최소한의 품질 검사를 자동화하는 장치다.

## 핵심 개념

**CI(지속적 통합)**는 개발자가 코드를 병합할 때마다 자동으로 빌드와 테스트를 실행한다. 빠른 피드백으로 버그를 조기에 발견하고 통합 문제를 줄인다. GitHub Actions, GitLab CI, Jenkins가 대표적인 CI 도구다.

**CD(지속적 배포)**는 CI를 통과한 코드를 자동으로 스테이징 또는 프로덕션 환경에 배포한다. Docker 이미지를 빌드하고 컨테이너 레지스트리에 푸시한 뒤 클라우드 서비스에 배포하는 흐름이 일반적이다. idol-agent 프로젝트(Week09)에서 GitHub Actions 기반 CI/CD 파이프라인을 구성했다.

## 파이프라인 흐름

![CI/CD 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-ci-cd-diagram-1.svg)

## GitHub Actions 기준 구조

GitHub Actions는 보통 Workflow, Job, Step으로 나눠 생각한다.

- Workflow: 언제 실행할지 정의하는 전체 자동화 파일
- Job: 독립적으로 실행되는 작업 단위
- Step: checkout, Python 설치, 의존성 설치, 테스트 실행 같은 실제 명령 단위

```yaml
name: ci

on:
  pull_request:
  push:
    branches: ["main"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest
```

이 예시는 최소 구조다. 실제 프로젝트에서는 `ruff`, `mypy`, Docker build, 이미지 push, 배포 step이 추가될 수 있다.

## Agent 서비스에서 필요한 검사

일반 웹 서비스와 달리 Agent 서비스는 테스트 기준을 더 나눠야 한다. API 서버가 켜지는지만 보는 테스트, 에이전트 그래프가 올바르게 라우팅되는지 보는 테스트, 모델 응답 품질을 보는 평가가 서로 다르기 때문이다.

| 검사 | 목적 |
| --- | --- |
| lint / format | 코드 스타일과 기본 오류 확인 |
| unit test | 함수, parser, utility 단위 검증 |
| integration test | API, DB, vector store, tool 호출 연결 확인 |
| smoke test | 배포 후 기본 endpoint 동작 확인 |
| evaluation | 프롬프트/모델 변경 후 답변 품질 회귀 확인 |

## Secret 관리

CI/CD에서 가장 조심해야 할 부분은 secret이다. OpenAI API key, DB URL, 클라우드 credential 같은 값은 repository에 직접 넣지 않고 GitHub Actions Secrets나 배포 플랫폼의 secret storage에 둔다.

로그에도 secret이 찍히지 않아야 한다. 특히 디버깅 목적으로 환경변수 전체를 출력하는 습관은 위험하다. 배포 자동화가 편해질수록 실수도 자동으로 퍼질 수 있으므로, secret 출력 금지와 권한 최소화를 기본값으로 둬야 한다.

## 적용 순서

1. 로컬에서 실행 가능한 테스트 명령을 먼저 정한다.
2. PR마다 lint와 test가 실행되도록 CI를 구성한다.
3. Docker 이미지가 필요한 경우 build 단계만 먼저 자동화한다.
4. main branch merge 후 staging 배포를 자동화한다.
5. 운영 배포는 smoke test와 rollback 기준을 함께 둔다.

## 주의점

- 테스트가 없는데 CI만 있으면 자동화된 형식 검사에 그친다.
- 배포 자동화 전에 환경변수, secret, 포트, health check 기준을 먼저 정리해야 한다.
- LLM 서비스는 "빌드 성공"과 "답변 품질 유지"가 다르므로 evaluation을 별도 축으로 봐야 한다.
- 자동 배포는 rollback 절차가 있을 때 의미가 있다.

## 관련 글

- [GitHub Actions CI]({% post_url 2026-02-26-upstage-course-w08d04-ci %})
- [CD (Continuous Deployment)]({% post_url 2026-02-27-upstage-course-w08d05-cd %})
- [Git]({% post_url 2026-01-08-upstage-tech-git %})
- [Docker]({% post_url 2026-01-12-upstage-tech-docker %})
- [클라우드 컴퓨팅]({% post_url 2026-01-15-upstage-tech-cloud-computing %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})

## 참고 자료

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
