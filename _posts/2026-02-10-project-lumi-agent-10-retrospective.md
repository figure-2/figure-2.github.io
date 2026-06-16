---
title: "10. Tool Call 불안정성, 평가 한계, 그리고 Lumi_agent 회고"
categories:
- 2.PROJECT
- 2-7. Lumi_agent
tags:
- Lumi_agent
- Retrospective
- Tool Calling
- Evaluation
- Agent
toc: true
date: 2026-02-10 09:00:00 +0900
comments: true
mermaid: true
math: true
---

Lumi_agent에서 가장 많이 배운 점은 Agent 프로젝트가 “모델이 똑똑하면 끝”이 아니라는 것이다.

외부 도구 실행, 사용자 승인, 기억 검색, GUI 이벤트 루프, 페르소나 유지가 모두 같이 맞아야 사용자가 하나의 비서처럼 느낀다.

## 자체 평가

당시 회고에는 목표 기능의 약 85% 정도를 수행했다고 판단했다는 내용이 남아 있다. 이 숫자는 성능 점수가 아니다. Tool Call이 가끔 불안정하고 애니메이션 동작이 고정적이었다는 한계를 함께 언급한 자체 평가다.

따라서 이 숫자는 다음처럼 읽어야 한다.

| 항목 | 해석 |
| --- | --- |
| 약 85% | 당시 자체 완성도 판단 |
| Tool Call 불안정성 | 개선이 필요한 핵심 한계 |
| 애니메이션 고정성 | UX 개선 여지 |
| 평가 결과표 | 확인 가능한 정량 근거로 쓰지 않음 |

## 잘된 점

첫 번째는 Agent 실행 흐름을 노드로 분리한 점이다. Analyzer, Context Builder, Agent, Tool, Memory Manager를 나누면서 각 단계의 책임이 명확해졌다.

두 번째는 Safe/Sensitive Tool을 분리한 점이다. 메시지 전송과 일정 변경처럼 외부 상태를 바꾸는 작업에 사용자 승인을 둔 것은 Agent UX에서 중요한 경계다.

세 번째는 Memory 구조를 별도로 둔 점이다. 단기 대화 흐름과 장기 기억을 분리하면서 개인 비서에 필요한 맥락 유지 구조를 만들었다.

## 아쉬운 점

Tool Call 안정성은 더 다뤄야 했다. 잘못된 도구 선택, 반복 호출, schema 오류, 외부 API 실패 같은 문제는 Agent에서 흔히 발생한다.

평가 결과도 더 남겼어야 했다. 평가 설계는 있었지만, 재현 가능한 결과표와 로그가 부족하면 성능 주장으로 연결하기 어렵다.

GUI도 더 다듬을 여지가 있었다. 데스크톱 캐릭터와 Agent가 연결된 것은 의미 있지만, 애니메이션과 반응 다양성은 후속 개선 과제로 남았다.

## 후속 개선 방향

| 영역 | 개선 방향 |
| --- | --- |
| Tool Call | schema 검증, retry 정책, 실행 로그, 반복 호출 제한 |
| HITL | 승인 화면에서 변경될 외부 상태를 더 명확히 표시 |
| Memory | 잘못 저장된 기억 삭제/수정, threshold 실험 |
| Evaluation | 33개 Tool Call 시나리오 결과표와 실패 유형 기록 |
| Desktop UX | 상태별 animation 다양화, 응답 중 busy 상태 개선 |

## 주장 경계

Lumi_agent를 설명할 때 가장 조심해야 할 부분은 자동화와 안전성이다.

| 말할 수 있는 것 | 넘겨 말하지 않는 것 |
| --- | --- |
| MCP 기반 외부 도구 연동 구조 | 외부 API 실패 처리의 충분한 검증 |
| 민감 도구 사용자 승인 흐름 | 승인 UI만으로 안전성을 단정하는 것 |
| Memory/RAG 기반 대화 맥락 구조 | 장기 기억 결과가 항상 맞다는 주장 |
| Agent 평가 설계 | 평가 결과가 남아 있다는 주장 |
| 데스크톱 Agent prototype | 배포 환경에서 검증됐다는 주장 |

이 경계를 지키면 프로젝트는 더 설득력 있어진다. 과장된 완성도가 아니라, Agent 시스템을 만들 때 필요한 실행 흐름과 위험 경계를 직접 다뤘다는 점이 남기 때문이다.

## 시리즈 마무리

| 순서 | 글 |
| --- | --- |
| 01 | [Lumi_agent 프로젝트 개요]({% post_url 2026-01-21-project-lumi-agent-01-case-study %}) |
| 02 | [웹/모바일 AI 비서의 한계를 PC Agent 문제로 바꾸기]({% post_url 2026-01-23-project-lumi-agent-02-problem-demo %}) |
| 03 | [MCP prototype에서 LangGraph Agent까지의 개발 흐름]({% post_url 2026-01-29-project-lumi-agent-03-development-timeline %}) |
| 04 | [Lumi_agent 전체 아키텍처]({% post_url 2026-02-02-project-lumi-agent-04-architecture %}) |
| 05 | [LangGraph StateGraph로 Agent 실행 흐름을 분리한 이유]({% post_url 2026-02-03-project-lumi-agent-05-langgraph-workflow %}) |
| 06 | [MCP Tool Calling과 HITL 승인 경계 설계]({% post_url 2026-02-04-project-lumi-agent-06-mcp-hitl %}) |
| 07 | [단기 기억, 요약, ChromaDB로 대화 맥락 유지하기]({% post_url 2026-02-05-project-lumi-agent-07-memory-rag %}) |
| 08 | [PySide6와 qasync로 데스크톱 Agent UX를 연결하기]({% post_url 2026-02-06-project-lumi-agent-08-desktop-persona-ux %}) |
| 09 | [Agent 품질을 어떻게 평가하려 했는가]({% post_url 2026-02-07-project-lumi-agent-09-evaluation-design %}) |
| 10 | 이 글 |
