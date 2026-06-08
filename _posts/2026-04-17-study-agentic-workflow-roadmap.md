---
title: "Agentic Workflow 개인 학습 로드맵"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- ai-agent
- agentic-workflow
- agentic-patterns
- workflow
- guide-review
- reference-note
toc: true
date: 2026-04-17 21:50:00 +0900
comments: false
mermaid: true
math: true
---
04 · Use Cases

## 실전 유즈케이스 매핑

Prompt Chaining

마케팅 카피 생성

브리프 → 아웃라인 → 초안 → 편집 → 번역. 각 단계 게이트로 품질 보장.

Routing

고객지원 자동화

환불/기술/일반 문의 분류 후 전문 핸들러로. Router는 Haiku, Handler는 Sonnet.

Parallelization

코드 리뷰 자동화

보안/성능/스타일 관점을 병렬 검토 후 통합 리포트. 관점별 전문 프롬프트.

Orchestrator-Workers

코딩 에이전트

Claude Code처럼 런타임에 Grep/Read/Edit/Bash 워커를 동적 분배.

Evaluator-Optimizer

고품질 번역

Generator가 번역, Evaluator가 용어·톤·자연스러움 평가. 최대 3회 루프.

Autonomous Agent

브라우저 자동화

Computer Use: 목표만 주고 클릭·입력·관찰을 스스로 판단.

🐝 Swarm Topology

다역할 고객지원

결제 → 기술 → 법무 상담원끼리 필요시 자율 handoff. 중앙 감독 없음.

🏛️ Hierarchical Topology

엔터프라이즈 에이전트 플랫폼

Top supervisor → 회계/법무/고객 팀 supervisor → 각 팀 워커들. 대규모 책임 격리.

Human-in-the-Loop

결제·발송 승인

에이전트가 이메일 50명 발송 직전 일시정지 → 사람 검토 → 재개. 실수 비용이 큰 모든 곳.

### 하이브리드 조합 예시

실무에서는 단일 패턴보다 조합이 많습니다:

- Router → Orchestrator: 입력 타입별 라우팅 후 동적 분해

- Chain + Evaluator: 각 단계마다 평가자로 품질 게이트

- Agent + Parallelization: 에이전트가 병렬 서브 에이전트 호출

- Agent + HITL: 자율 실행하되 되돌릴 수 없는 작업 직전만 승인 — 프로덕션 표준 조합

- Hierarchical + Swarm: 상위는 계층 구조, 각 팀 내부는 swarm — 대규모 기업형 멀티에이전트

- Orchestrator + Evaluator: 워커 결과를 평가자가 검증, 기준 미달 시 재위임

05 · Principles

## 설계 원칙

🎯 Start simple. 단일 LLM 호출로 풀 수 있으면 거기서 멈추세요. 패턴을 쌓을수록 디버깅이 어려워집니다.

📏 Measure before adding complexity. 복잡도 추가가 성능 이득을 정량적으로 만드는지 확인 후 결정.

🧱 Augmented LLM이 기본 블록. 모든 패턴은 "LLM + Tools + Memory + Retrieval"을 어떻게 조합하느냐의 문제.

🚨 Autonomous Agent는 가드레일 필수. Max steps, 비용 한도, Human check-in, 샌드박스 — 네 가지를 반드시.

💡 투명성이 곧 신뢰. 에이전트가 무엇을 왜 하는지 로깅하고 보여주세요. 블랙박스는 프로덕션에서 살아남지 못합니다.

🔄 평가 주도 개발. 에이전트를 만들기 전에 평가셋부터. 이게 없으면 개선도 회귀도 탐지할 수 없습니다.

🧠 Context Engineering. 컨텍스트 윈도우에 무엇을 넣을지가 long-horizon 에이전트의 최대 병목. 어떤 메모리를 압축·요약·폐기할지 설계하세요. (Harrison Chase, 2026)

🙋 되돌릴 수 없는 작업엔 HITL. 결제·발송·삭제·권한 변경 — 자율 실행이 유혹이지만 Pause → Approve → Resume이 프로덕션 표준. 사람이 개입할 지점을 미리 설계하세요.

💾 State persistence = 에이전트의 undo. 체크포인트(checkpoint)로 모든 스텝 상태를 저장하면 실패 시 되감기, 디버깅 시 time-travel이 가능합니다.

---

## 추가 정리

### 핵심 요약

Agentic Workflow 학습은 패턴을 암기하는 것이 아니라, 문제를 어떤 실행 구조로 바꿀지 판단하는 훈련이다. 유즈케이스를 패턴에 매핑하는 연습이 중요하다.

### 보충 해설

좋은 연습 방법은 하나의 문제를 여러 패턴으로 다시 설계해 보는 것이다. 예를 들어 코드 리뷰는 Parallelization으로, 고객지원은 Routing으로, 장기 리서치는 Orchestrator-Workers로, 배포 승인 흐름은 Human-in-the-Loop로 모델링할 수 있다.
