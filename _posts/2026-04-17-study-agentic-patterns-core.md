---
title: "Agentic AI 패턴 가이드 2: 8가지 패턴"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- agentic-patterns
- prompt-chaining
- routing
- multi-agent
toc: true
date: 2026-04-17 22:10:00 +0900
comments: false
mermaid: true
math: true
---
02 · Patterns

## 8가지 핵심 패턴

Input / User

LLM Call

Decision / Check

Tool / Worker

Memory / Storage

Success Output

Error / Exit

00

Augmented LLM

— 모든 패턴의 기본 블록

Foundation

에이전틱 시스템의 가장 작은 단위. LLM + Tools + Memory + Retrieval의 조합입니다. 아래 모든 패턴은 이 기본 블록을 어떻게 여러 개 엮느냐의 문제로 환원됩니다.

핵심 원리

LLM이 자기 스스로

언제/어떻게

tools/memory/retrieval을 쓸지 결정

인터페이스

Function calling · Tool use API · Retrieval augmentation

왜 중요한가

아래 모든 패턴은 이 블록을

어떻게 조합

하느냐의 문제

01

Prompt Chaining

Sequential

Simple

작업을 순차 단계로 쪼개서 이전 LLM의 출력이 다음 LLM의 입력이 됩니다. 중간에 게이트(검증)를 넣어 조건에 맞지 않으면 중단하거나 재시도할 수 있습니다.

언제 쓰나

작업을

고정된 단계

로 명확히 쪼갤 수 있을 때

대표 예시

마케팅 카피: 아웃라인 → 초안 → 번역 → 검수

실패 모드

앞 단계 에러가 뒷 단계로 전파 → 게이트 필수

02

Routing

Classifier

Simple

입력을 분류해서 적절한 전문 핸들러로 라우팅합니다. 각 핸들러가 자기 작업에만 특화되어 품질이 올라갑니다.

언제 쓰나

입력 타입이

명확히 구분

되고 각각 다른 처리가 필요할 때

대표 예시

고객지원: 환불/기술지원/일반 분류 → 전문 에이전트로

팁

Router는 작은 모델(Haiku)로, Handler는 적절한 크기 모델로 → 비용 절감

03

Parallelization

Parallel

Sectioning · Voting

작업을 병렬로 쪼개 실행 후 집계합니다. Sectioning은 독립 하위 작업 분할, Voting은 같은 작업을 여러 번 실행해 다수결.

언제 쓰나

작업이

독립적

이거나

신뢰도

를 높여야 할 때

대표 예시

코드 리뷰: 보안/성능/스타일 관점을 동시에 검토

장점

레이턴시 단축 + 컨텍스트 분리로 각 관점 집중

04

Orchestrator-Workers

Dynamic

Advanced

중앙 Orchestrator LLM이 런타임에 하위 작업을 동적으로 분해하고 워커들에게 위임합니다. 서브태스크를 미리 정해둘 수 없는 문제에 적합합니다.

언제 쓰나

서브태스크가

런타임에 결정

되고 병렬화 가능할 때

대표 예시

Claude Code의 sub-agents, 검색 에이전트

핵심 차이

Parallelization은 고정 분할, Orchestrator는

동적 분할

04+

Multi-agent Topologies

— Orchestrator 확장

LangGraph

Extension

Orchestrator-Workers는 Supervisor 토폴로지의 한 형태입니다. 실무에서는 제어 구조에 따라 세 가지로 나뉩니다: Supervisor(중앙 조정) · Swarm(피어 handoff) · Hierarchical(다층 구조).

Supervisor

현재 우리의 Orchestrator-Workers. 코드 에이전트, Claude Code의 sub-agents

Swarm

고객지원 multi-specialist, 다중 역할 롤플레잉, OpenAI Swarm SDK

Hierarchical

엔터프라이즈 에이전트 플랫폼 (회계·법무·고객 팀 병렬 운영)

05

Evaluator-Optimizer

Iterative

Feedback Loop

한 LLM이 생성(Generator), 다른 LLM이 피드백(Evaluator). 기준을 충족할 때까지 반복해 품질을 끌어올립니다.

언제 쓰나

정답은 없지만

품질 기준이 명확

할 때

대표 예시

번역, 에세이 개선, 코드 리팩토링

주의

최대 반복 횟수(예: 3회) 제한 필수 → 무한루프 방지

06

Autonomous Agent

Autonomous

High Complexity

LLM이 스스로 도구 사용과 경로를 결정합니다. 환경에서 피드백을 받고 계획을 수정하며 종료 조건까지 자율 실행. 유연하지만 비용과 실패 위험이 큽니다.

언제 쓰나

경로가

완전히 열려있고

자율성이 핵심 가치일 때

대표 예시

Claude Code, Computer Use, 리서치 에이전트

필수 가드레일

Max steps · 비용 한도 · Human check-in · 테스트 샌드박스

07

Human-in-the-Loop

— Pause · Approve · Resume

Production-grade

Safety

에이전트가 중요한 결정 직전에 일시정지하고 사람의 검토/승인/수정을 기다린 뒤 재개합니다. 자율성과 안전성의 균형을 잡는 프로덕션 1순위 패턴. LangGraph는 이를 checkpoint + interrupt로 구현합니다.

언제 쓰나

실수의 비용이

크고 되돌릴 수 없는

모든 상황

대표 예시

코드 PR 머지 · 송금 · 이메일 발송 · 회의 예약 · 리소스 삭제

구현 핵심

상태 직렬화(checkpoint) · 비동기 대기 · 알림 · 재개 가능 메시지 큐

---

## 추가 정리

### 핵심 요약

8가지 패턴은 서로 경쟁하는 기술이 아니라 조합 가능한 설계 블록이다. Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer, Autonomous Agent, Human-in-the-Loop는 문제의 불확실성 수준에 따라 선택된다.

### 보충 해설

고정 절차는 Prompt Chaining이 적합하고, 입력 유형이 갈리면 Routing이 적합하다. 병렬 검토가 필요하면 Parallelization을 쓰고, 하위 작업이 실행 중에 정해지면 Orchestrator-Workers가 필요하다. 위험한 행동은 Human-in-the-Loop로 멈춤 지점을 만들어야 한다.
