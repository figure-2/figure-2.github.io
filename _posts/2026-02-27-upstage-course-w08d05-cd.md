---
title: "CD (Continuous Deployment)"
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
date: 2026-02-27 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# CD (Continuous Deployment)

## 수업 위치

이 수업은 CI를 통과한 코드를 실제 서버 환경으로 배포하는 단계다. CI가 “코드가 깨지지 않았는가”를 확인한다면, CD는 “다른 사람이 접속할 수 있는 환경에 안전하게 반영되는가”를 다룬다.

강의자료 기준 핵심은 로컬 실행에서 서버 실행으로 옮기는 것이다. 로컬에서 `uvicorn`으로 띄운 서비스는 나만 접속할 수 있지만, 배포를 하면 외부 사용자가 접근할 수 있다. 이때 Docker, Docker Compose, GHCR, EC2, GitHub Actions가 연결된다.

## 핵심 개념

> **요약**
> CI에 이어 CD 파이프라인을 구축하여 에이전트 서비스를 서버에 배포한다. Docker 이미지 빌드, GHCR push, EC2 pull, Docker Compose 재시작, health check, rollback 흐름을 학습한다.

## 주요 내용

### 1. CD 개념
- Continuous Delivery vs Continuous Deployment
- 배포 파이프라인 설계
- 배포 전략: Rolling, Blue-Green, Canary
- main merge 이후 배포 자동화

### 2. Docker 배포
- Docker 이미지 빌드 및 태깅
- 컨테이너 레지스트리 (GHCR, Docker Hub)
- docker-compose를 통한 멀티 컨테이너 관리
- `.dockerignore`, non-root user, health check

### 3. 클라우드 배포
- 클라우드 환경에서의 에이전트 서비스 배포
- 환경변수 및 시크릿 관리
- 모니터링 및 로깅 설정
- 스케일링 전략

## CD 흐름

강의자료의 배포 흐름은 다음처럼 정리할 수 있다.

```text
main merge
  -> GitHub Actions build
  -> Docker image push to GHCR
  -> SSH to EC2
  -> docker pull
  -> docker compose up --force-recreate
  -> health check
```

이 방식은 EC2에서 직접 빌드하지 않고, GitHub Actions에서 이미지를 빌드한 뒤 EC2에서는 pull만 수행한다. 서버의 작업을 줄이고 배포 시간을 짧게 만들 수 있다.

## Docker에서 중요한 지점

| 항목 | 이유 |
|---|---|
| Multi-stage build | runtime 이미지 크기를 줄이고 불필요한 빌드 도구를 제거 |
| non-root user | 컨테이너가 root 권한으로 실행되는 위험 감소 |
| health check | 프로세스 생존 여부가 아니라 앱 응답 가능 여부 확인 |
| `.dockerignore` | `.venv`, `.git`, `.env` 같은 불필요하거나 민감한 파일 제외 |
| Docker Compose | 컨테이너 실행 옵션과 환경변수를 파일로 관리 |

특히 `.env`는 이미지 안에 복사되면 안 된다. 배포 환경에서는 GitHub Secrets, EC2 환경변수, 서버의 별도 설정 파일처럼 노출 범위를 통제할 수 있는 경로로 주입해야 한다.

## 배포 방식의 한계

이번 강의에서 다룬 방식은 단일 EC2와 Docker Compose 기반이다. 학습 프로젝트나 초기 프로토타입에는 적합하지만, 트래픽이 급증하거나 고가용성이 필요해지면 Auto Scaling Group, 관리형 컨테이너 서비스, Kubernetes 같은 선택지를 검토해야 한다.

다만 처음부터 Kubernetes로 가는 것은 과할 수 있다. 현재 단계에서는 “이미지를 만들고, 서버에서 실행하고, health check로 확인한다”는 기본 배포 흐름을 정확히 이해하는 것이 더 중요하다.

## 흐름도

![CD (Continuous Deployment) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-agent-architecture-w08d05-cd-diagram-1.svg)

## 관련 글

- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %})
- [Docker]({% post_url 2026-01-12-upstage-tech-docker %})
- [Cloud Computing]({% post_url 2026-01-15-upstage-tech-cloud-computing %})
- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %})
