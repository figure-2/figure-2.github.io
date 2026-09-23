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

Agentic Workflow는 패턴 이름을 외우는 것보다 유즈케이스를 어떤 실행 구조로 바꿀지 판단하는 것이 중요합니다. 같은 문제도 비용, 품질 기준, 실패 비용에 따라 다른 패턴을 선택할 수 있습니다.

## 실전 유즈케이스 매핑

Prompt Chaining

마케팅 카피 생성

브리프 → 아웃라인 → 초안 → 편집 → 번역 순으로 진행합니다. 각 단계의 게이트로 품질을 보장합니다.

Routing

고객지원 자동화

환불·기술·일반 문의를 분류한 뒤 전문 핸들러로 보냅니다. Router는 Haiku, Handler는 Sonnet을 사용합니다.

Parallelization

코드 리뷰 자동화

보안·성능·스타일 관점을 병렬로 검토한 뒤 통합 리포트를 만듭니다. 관점별 전문 프롬프트를 사용합니다.

Orchestrator-Workers

코딩 에이전트

Claude Code처럼 런타임에 Grep/Read/Edit/Bash 워커를 동적으로 분배합니다.

Evaluator-Optimizer

고품질 번역

Generator가 번역하고 Evaluator가 용어·톤·자연스러움을 평가합니다. 최대 3회 반복합니다.

Autonomous Agent

브라우저 자동화

Computer Use는 목표만 주고 클릭·입력·관찰을 스스로 판단합니다.

Swarm Topology

다역할 고객지원

결제 → 기술 → 법무 상담원이 필요할 때 자율적으로 handoff합니다. 중앙 감독은 없습니다.

Hierarchical Topology

엔터프라이즈 에이전트 플랫폼

Top supervisor → 회계·법무·고객 팀 supervisor → 각 팀 워커로 구성합니다. 대규모 책임을 격리합니다.

Human-in-the-Loop

결제·발송 승인

에이전트가 이메일 50명 발송 직전에 일시 정지하고, 사람이 검토한 뒤 재개합니다. 실수 비용이 큰 모든 상황에 적용할 수 있습니다.

### 하이브리드 조합 예시

실무에서는 단일 패턴보다 조합이 많습니다:

- Router → Orchestrator: 입력 타입별 라우팅 후 동적 분해

- Chain + Evaluator: 각 단계마다 평가자로 품질 게이트

- Agent + Parallelization: 에이전트가 병렬 서브 에이전트 호출

- Agent + HITL: 자율 실행하되 되돌릴 수 없는 작업 직전만 승인 — 프로덕션 표준 조합

- Hierarchical + Swarm: 상위는 계층 구조, 각 팀 내부는 swarm — 대규모 기업형 멀티에이전트

- Orchestrator + Evaluator: 워커 결과를 평가자가 검증, 기준 미달 시 재위임

## 설계 원칙

- Start simple. 단일 LLM 호출로 풀 수 있으면 거기서 멈춘다. 패턴을 쌓을수록 디버깅이 어려워진다.

- Measure before adding complexity. 복잡도 추가가 성능 이득을 정량적으로 만드는지 확인한 뒤 결정한다.

- Augmented LLM이 기본 블록이다. 모든 패턴은 `LLM + Tools + Memory + Retrieval`을 어떻게 조합하느냐의 문제다.

- Autonomous Agent에는 가드레일이 필수다. Max steps, 비용 한도, Human check-in, sandbox를 먼저 둔다.

- 투명성이 곧 신뢰다. 에이전트가 무엇을 왜 하는지 로깅하고 보여줘야 한다.

- 평가 주도 개발이 필요하다. 에이전트를 만들기 전에 평가셋부터 준비해야 개선과 회귀를 구분할 수 있다.

- Context Engineering이 long-horizon agent의 병목이다. 어떤 memory를 압축, 요약, 폐기할지 설계해야 한다.

- 되돌릴 수 없는 작업에는 Human-in-the-Loop가 필요하다. 결제, 발송, 삭제, 권한 변경은 사람이 개입할 지점을 미리 둔다.

- State persistence는 agent의 undo 역할을 한다. checkpoint로 step 상태를 저장하면 실패 복구와 디버깅이 쉬워진다.

---

## 추가 정리

### 핵심 요약

Agentic Workflow 학습은 패턴을 암기하는 것이 아니라, 문제를 어떤 실행 구조로 바꿀지 판단하는 훈련이다. 유즈케이스를 패턴에 매핑하는 연습이 중요하다.

### 보충 해설

좋은 연습 방법은 하나의 문제를 여러 패턴으로 다시 설계해 보는 것입니다. 예를 들어 코드 리뷰는 Parallelization으로, 고객지원은 Routing으로, 장기 리서치는 Orchestrator-Workers로, 배포 승인 흐름은 Human-in-the-Loop로 모델링할 수 있습니다.
