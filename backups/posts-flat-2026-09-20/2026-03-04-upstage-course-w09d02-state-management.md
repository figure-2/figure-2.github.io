---
title: "상태 관리"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-9.LLMOPS
- COURSE_NOTE
tags:
- upstage
- sesac
- ai-agent
- state-management
- course-note
- llmops
toc: true
date: 2026-03-04 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## 상태 관리

## 수업 위치

이 수업은 LiteLLM을 통한 API 장애 대응 다음 단계다. Day01에서 모델 호출 계층의 안정성을 다뤘다면, Day02에서는 에이전트가 대화와 실행 상태를 어떻게 유지할지 다룬다.

에이전트 서비스는 stateless API처럼 매 요청을 독립적으로 처리하기 어렵다. 사용자의 이전 발화, 선택된 tool, 검색된 문서, 모델 응답, 비용 정보가 다음 응답에 영향을 줄 수 있다. 따라서 상태를 어디까지 유지하고, 어디에 저장하고, 언제 줄일지 기준이 필요하다.

## 핵심 개념

> **요약**
> 에이전트 서비스의 상태 관리 패턴을 학습한다. LangGraph의 checkpoint를 활용한 대화 이력 관리, 세션 기반 상태 분리, 외부 저장소 기반 영속화, 비용 추적과 context trimming을 함께 다룬다.

## 주요 내용

### 1. 에이전트 상태 관리의 필요성
- 대화 이력 유지
- 세션별 컨텍스트 분리
- 에이전트 실행 상태 추적
- 서비스 재시작 시 상태 복구
- context window 초과 방지

### 2. LangGraph 체크포인터
- MemorySaver: 인메모리 체크포인터
- PostgresSaver: Supabase 기반 영속 체크포인터
- 체크포인트 기반 대화 분기 및 되감기
- 개발 환경과 운영 환경의 저장소 분리

### 3. 세션 관리
- 사용자별 세션 식별
- 세션 생성, 조회, 삭제 API
- 세션 만료 정책
- 멀티 턴 대화 관리

### 4. 비용 추적과 메시지 트리밍
- 모델 호출별 token usage 기록
- 일일 비용 한도와 알림 기준 설정
- 오래된 메시지 또는 낮은 우선순위 context 제거
- trimming 이후에도 답변 품질이 유지되는지 확인
- 사용 로그를 별도 테이블에 저장해 세션별, 모델별 비용을 분석

### 5. idol-agent v0.7
- Supabase 기반 영속 체크포인터 적용
- 세션 관리 API 구현
- 대화 이력 조회 기능

## 구현 관점

상태 관리는 “대화 내용을 모두 저장한다”가 아니다. 운영 가능한 상태 관리는 저장할 것과 버릴 것을 구분하는 일이다.

| 상태 | 저장 이유 | 주의점 |
|---|---|---|
| messages | 멀티 턴 대화 유지 | context window 초과 가능 |
| intent | router 판단 추적 | 잘못된 분류가 누적될 수 있음 |
| retrieved_docs | RAG 근거 확인 | 원문 저장 시 보안 검토 필요 |
| tool_result | tool call 결과 재사용 | 실패 결과도 기록해야 분석 가능 |
| token_usage | 비용 추적 | retry/fallback 비용 포함 필요 |

LangGraph checkpoint는 그래프 실행 상태를 저장하고 복원하는 데 유용하다. 하지만 checkpoint만으로 운영 문제가 모두 해결되지는 않는다. 세션 만료, 개인정보 보관 기간, token budget, 비용 한도, 알림 기준을 함께 설계해야 한다.

강의자료에서는 새로고침과 서버 재시작을 분리해서 봤다. 브라우저 새로고침으로 세션 ID가 바뀌면 체크포인터가 있어도 이전 대화를 찾지 못한다. 서버 재시작으로 메모리가 사라지면 MemorySaver만으로는 대화를 복구할 수 없다. 그래서 프론트의 session id 유지와 백엔드의 영속 checkpoint가 함께 필요하다.

## 수업에서 남길 체크포인트

- 개발 환경에서는 MemorySaver로 충분하지만 운영에서는 프로세스 재시작 후에도 상태가 유지되어야 한다.
- 상태 저장소는 모델 품질과 사용자 개인정보 리스크를 동시에 가진다.
- context trimming은 비용 절감뿐 아니라 모델 입력 품질 관리 문제다.
- 비용 추적은 성공한 호출뿐 아니라 retry, fallback, 실패 호출까지 포함해야 한다.
- 상태 관리 데이터는 Observability와 연결되어야 나중에 장애 분석이 가능하다.
- Discord 같은 알림 채널에는 비용 요약만 보내고 raw prompt, API key, 사용자 원문은 보내지 않는다.

## 흐름도

![상태 관리 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/01-modules-llmops-ops-w09d02-state-management-diagram-1.svg)

## 관련 글

- [상태관리 기술 노트]({% post_url 2026-03-04-upstage-tech-state-management %})
- [상태관리 + 비용 추적 실습]({% post_url 2026-03-04-upstage-practice-state-management-practice %})
- [idol-agent v0.7]({% post_url 2026-03-04-upstage-project-idol-agent-v07 %})
- [Observability]({% post_url 2026-03-05-upstage-tech-observability %})
