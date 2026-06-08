---
title: "Agentic AI 패턴 가이드 1: Workflow vs Agent"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- agentic-workflow
- ai-agent
- workflow
toc: true
date: 2026-04-17 22:00:00 +0900
comments: false
mermaid: true
math: true
---
Building Effective Agents · 2026.04

# Agentic AI 패턴 가이드

LLM으로 실제로 동작하는 시스템을 설계할 때 쓰이는 6가지 핵심 패턴. 언제 어느 패턴을 써야 하는지, 어떻게 조합하는지, 무엇이 실패 모드인지.

8

핵심 패턴

3

Multi-agent 토폴로지

10+

실전 다이어그램

8가지 패턴 보러가기 →

어떤 걸 써야 할까?

01 · Introduction

## Workflow vs Agent

Workflows는 미리 정의된 경로로 LLM을 오케스트레이션합니다. 예측 가능하고 디버깅하기 쉽습니다. Agents는 LLM이 스스로 도구와 경로를 결정합니다. 유연하지만 비용과 실패 위험이 높습니다.

대부분의 문제는 단일 LLM 호출 + RAG + 잘 쓴 프롬프트로 해결됩니다. 복잡도는 측정 가능한 성능 이득이 있을 때만 추가하세요. 이 가이드는 Anthropic의 "Building effective agents" 분류를 기반으로 합니다.

🤖

함께 보기

에이전트가 무엇이며 어떻게 진화해왔는지

개념부터 알고 싶다면 — 7단계 성숙도·Memory/RAG/Guardrails 아키텍처가 담긴

AI Agent 완벽 가이드

를 보세요.

Agent 가이드 →

목차

- 00Augmented LLM (foundation)

- 01Prompt Chaining

- 02Routing

- 03Parallelization

- 04Orchestrator-Workers

- 04+Multi-agent Topologies

- 05Evaluator-Optimizer

- 06Autonomous Agent

- 07Human-in-the-Loop

---

## 추가 정리

### 핵심 요약

Workflow와 Agent의 차이는 자율성의 위치다. Workflow는 사람이 경로를 정하고, Agent는 모델이 실행 중에 경로를 선택한다.

### 보충 해설

실무에서는 Agent라는 이름을 붙이기 전에 제어 흐름이 정말 동적인지 확인해야 한다. 고정된 단계를 자동화하는 문제라면 Workflow가 더 적합하다. Agent는 불확실성과 탐색이 필요한 문제에서 가치가 생긴다.
