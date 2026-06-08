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
# Agentic Workflow 개인 학습 로드맵

## 학습 목적

Agentic Workflow를 패턴 이름 암기가 아니라, 작업 복잡도와 실패 비용에 따라 선택하는 설계 도구로 정리한다.

## 정리 범위

| 소주제 | 정리 관점 |
| --- | --- |
| Prompt Chaining | 단계를 나누면 안정성이 올라가는 경우 |
| Routing | 입력 유형에 따라 경로를 나눌 기준 |
| Parallelization | 병렬 실행이 비용보다 이득인 경우 |
| Orchestrator-Workers | 복잡한 작업을 분배하는 기준 |
| Evaluator-Optimizer | 생성과 평가를 반복하는 구조 |
| Human-in-the-Loop | 사람이 개입해야 하는 경계 |
| Multi-Agent Topology | 협업 구조 선택 기준 |
| Vibe Coding | AI와 함께 구현할 때의 생산성과 검증 비용 |
| Cognitive Debt | AI 코딩이 이해, 검증, 비용, 종속성에 남기는 부채 |

## 작성할 글

| 순서 | 게시글 | 소주제 | 상태 |
| --- | --- | --- | --- |
| 1 | [Agentic AI 패턴 가이드 1: Workflow vs Agent]({% post_url 2026-04-17-study-agentic-workflow-vs-agent %}) | workflow와 agent 구분, augmented LLM, 복잡도 원칙 | 작성 |
| 2 | [Agentic AI 패턴 가이드 2: 8가지 패턴]({% post_url 2026-04-17-study-agentic-patterns-core %}) | prompt chaining, routing, parallelization, orchestrator, evaluator, autonomous, HITL | 작성 |
| 3 | [Agentic AI 패턴 가이드 3: 선택 기준, 비용, 토폴로지]({% post_url 2026-04-17-study-agentic-pattern-selection-topology %}) | 선택 트리, 비용/레이턴시, topology, use case, hybrid 조합 | 작성 |
| 4 | [바이브코딩 & Claude Code 교육 자료]({% post_url 2026-03-28-study-vibe-coding-claude-code %}) | 바이브코딩 정의, Claude Code 활용, 작업 계약, 검증 루프, 교육용 실습 구성 | 작성 |
| 5 | [AI 코딩의 Cognitive Debt]({% post_url 2026-05-05-study-ai-coding-cognitive-debt %}) | supervision paradox, skill atrophy, token cost, vendor lock-in, 위임/직접 수행 기준 | 작성 |

## 후속 분리 후보

| 후보 글 | 분리 이유 |
| --- | --- |
| Multi-Agent 토폴로지 비교 | 1차 글에 반영 완료. 추후 사례별 설계로 분리 가능 |
| HITL 설계 기준 | 1차 글에 반영 완료. 권한/승인 정책 상세는 후속 가능 |
| Evaluator-Optimizer 패턴 | 1차 글에 반영 완료. 평가셋 설계 글로 확장 가능 |

## 작성 기준

패턴은 기술 이름이 아니라 선택지다. 각 글은 "언제 쓰지 말아야 하는가"를 반드시 포함한다.
