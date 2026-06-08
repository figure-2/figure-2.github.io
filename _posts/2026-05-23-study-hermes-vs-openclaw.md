---
title: "Hermes Agent vs OpenClaw"
categories:
- 3.STUDY
- 3-3.AI_AGENT
tags:
- study
- ai-agent
- hermes
- openclaw
- agent-ops
toc: true
date: 2026-05-23 18:00:00 +0900
comments: false
mermaid: true
math: true
---
# Hermes Agent vs OpenClaw

> **한줄 정의**
> Hermes는 agent-first 설계이고, OpenClaw는 gateway-first 설계다. 하나는 agent가 성장하는 방식에, 다른 하나는 agent가 항상 연결되는 방식에 무게를 둔다.

## 핵심 비교

| 관점 | Hermes Agent | OpenClaw |
| --- | --- | --- |
| 한 줄 정의 | 스스로 성장하는 agent | 항상 켜진 gateway assistant |
| 중심 설계 | Agent-first | Gateway-first |
| 핵심 질문 | agent가 어떻게 더 나아지는가 | agent가 어떻게 항상 연결되는가 |

## 아키텍처

Hermes는 agent가 주인이다. memory, tool, skill, delegate가 agent 내부 성장 루프를 중심으로 구성된다.

OpenClaw는 gateway가 주인이다. channel, session, plugin, boundary가 여러 표면에서 agent를 운영하는 구조를 만든다.

```text
Hermes
  Agent Core
    -> Memory
    -> Tools
    -> Skills
    -> Delegate

OpenClaw
  Gateway
    -> Channel
    -> Session
    -> Plugin
    -> Boundary
```

## Memory

| 관점 | Hermes | OpenClaw |
| --- | --- | --- |
| 저장 형식 | markdown, section delimiter | markdown, file 분리 |
| load 전략 | session 시작 시 frozen snapshot | session 시작 시 MEMORY와 오늘/어제 직접 load |
| 용량 제한 | 명시적, 2,200자 / 1,375자 | 암묵적, token 예산 내 |
| cache 최적화 | prefix cache 보존 중심 | context engine이 token 예산 관리 |
| 과거 검색 | SQLite FTS5 + 8개 plugin | vector + BM25 hybrid |
| 자동 정리 | curator가 skill 정리, memory는 수동 | dreaming pass가 자동 승격 |

Hermes의 memory는 안정성과 cache 효율에 강하고, OpenClaw의 memory는 투명성과 파일 기반 조작에 강하다.

## Tool System

| 관점 | Hermes | OpenClaw |
| --- | --- | --- |
| 등록 방식 | AST 자동 발견 | manifest 선언 + SDK |
| 언어 | Python | TypeScript |
| 도구 수 | 95+ built-in | core tools + plugin |
| 확장성 | 파일 하나 추가 | plugin package 구조 |
| 정책 관리 | Toolset grouping | Slot + Boundary system |

Hermes는 도구를 많이 내장하고 자동 발견한다. OpenClaw는 plugin 경계와 manifest를 통해 운영 가능성을 높인다.

## Skill System

Hermes가 "성장하는 agent"로 보이는 이유는 skill system에 있다.

```text
Experience
  -> Skill candidate
  -> Curator
  -> Skill registry
  -> Reuse
```

OpenClaw는 같은 의미의 skill system을 중심 기능으로 두지 않는다. 대신 gateway, channel, plugin 운영에 초점을 둔다.

## Multi-Agent

| 관점 | Hermes | OpenClaw |
| --- | --- | --- |
| 병렬 실행 | MoA + Delegate sub-agent | per-session 직렬, session spawn 가능 |
| cross-model 합성 | MoA로 여러 model 결과 합성 | 단일 model 중심 |
| 적합한 패턴 | orchestrator -> specialist agent | 단일 agent가 분해하며 실행 |

Hermes는 복수 agent와 복수 model을 합성하는 구조에 가깝고, OpenClaw는 연결된 session을 안정적으로 운용하는 쪽에 가깝다.

## Channel & Gateway

| 관점 | Hermes | OpenClaw |
| --- | --- | --- |
| platform 수 | 35+ | 40+ |
| architecture | platform별 adapter | WebSocket protocol + channel plugin |
| protocol | platform 직접 연결 | typed WebSocket, version negotiation, nonce auth |
| 역할 분리 | 없음, agent 중심 | operator vs node, scope 기반 권한 |
| DM 보안 | platform별 | pairing code 기반 |

Gateway-first 설계에서는 agent의 지능보다 연결성과 권한 경계가 더 중요한 문제가 된다.

## MCP

| 관점 | Hermes | OpenClaw |
| --- | --- | --- |
| MCP 역할 | client | server + client |
| 연결 방향 | 외부 tool 사용 | 외부 tool 사용과 제공 모두 고려 |
| 설계 무게 | agent 확장 | gateway 확장 |

## 배포와 운영

| 관점 | Hermes | OpenClaw |
| --- | --- | --- |
| 언어 | Python 3.11+ | TypeScript, Node 24 |
| 설치 | curl one-line, Windows installer | npm install + onboard |
| Docker | 지원 | 권장, multi-stage build, Tini init |
| 실행 backend | 7가지, local, Docker, SSH, Singularity, Modal, Daytona, Vercel | 3가지, Docker, SSH, OpenShell |
| 보안 | memory injection scan, MCP package malware scan | cap_drop, no-new-privileges, sandbox policy |
| 관측성 | built-in tracing | OpenTelemetry + Prometheus |
| migration | OpenClaw -> Hermes 지원 | 해당 없음 |

## 종합 비교

| 구분 | Hermes Agent | OpenClaw |
| --- | --- | --- |
| 제작 | Nous Research | OpenClaw community |
| license | MIT | MIT |
| 중심 설계 | Agent-first, 자기 개선 loop | Gateway-first, channel 운영 |
| memory | frozen snapshot + plugin | plain markdown + hybrid search |
| skill system | 자율 생성 + curator | 없음 |
| tool 등록 | AST 자동 발견, 95+ module | manifest 기반 plugin SDK |
| multi-agent | MoA + Delegate | per-session serial |
| channel | 35+ platform adapter | 40+ channel plugin |
| protocol | platform 직접 연결 | typed WebSocket + scope |
| observability | built-in tracing | OpenTelemetry + Prometheus |

## 선택 기준

| 원하는 것 | 더 맞는 방향 |
| --- | --- |
| agent가 경험을 축적하고 skill을 재사용 | Hermes |
| 여러 channel에서 항상 켜진 assistant 운영 | OpenClaw |
| 많은 built-in tool과 delegate | Hermes |
| plugin boundary와 gateway governance | OpenClaw |
| agent 연구/실험 | Hermes |
| 운영형 assistant platform | OpenClaw |

## 내 기준

두 시스템은 같은 꿈을 꾸지만 중심 객체가 다르다.

```text
Hermes:
  agent가 더 잘하게 만드는 구조

OpenClaw:
  agent가 더 많은 표면에서 안전하게 연결되는 구조
```

Agent 제품을 볼 때 먼저 물어볼 질문은 "어떤 모델을 쓰나"가 아니다. "중심 객체가 agent인가, gateway인가, workflow인가"다.

## 관련 글

- [Agent Engineering]({% post_url 2026-05-23-study-agent-engineering %})
