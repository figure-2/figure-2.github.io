---
title: "Agent Workflow 실습 정리: Tool Calling, Reflection, Multi-Agent"
categories:
- 3.STUDY
- 3-4.AGENTIC_WORKFLOW
tags:
- study
- agentic-workflow
- tool-calling
- multi-agent
- reflection
toc: true
date: 2026-06-17 09:50:00 +0900
comments: false
mermaid: true
math: true
---

Agent를 공부하면서 처음에는 “LLM이 도구를 쓰면 Agent”라고 생각했다. 그런데 여러 구현 흐름을 비교해보니 도구 호출은 시작일 뿐이었다.

실제로 중요한 것은 도구를 언제 부를지, 결과를 어떻게 검증할지, 여러 역할을 나눌 때 latency와 비용을 어떻게 감당할지였다.

![Agent Workflow 실습 정리](/assets/images/study/diagrams/study-practice-agent-workflow.png)

## Tool Calling은 실행 경계다

Tool Calling은 모델이 외부 기능을 호출하는 구조다. 검색, 메일, 일정, DB 조회, 메시지 전송 같은 작업이 여기에 들어간다.

하지만 모든 도구 호출이 같은 위험도를 갖지는 않는다.

| 도구 유형 | 위험 |
| --- | --- |
| 검색 | 결과가 틀려도 다시 검색할 수 있다 |
| DB 조회 | 권한과 범위가 중요하다 |
| 메시지 전송 | 외부 상태를 바꾼다 |
| 일정 생성/삭제 | 사용자 생활에 직접 영향을 준다 |
| 결제/주문 | 반드시 승인 경계가 필요하다 |

그래서 Agent workflow를 설계할 때는 `도구를 호출할 수 있는가`보다 `호출해도 되는가`를 먼저 봐야 한다.

## Reflection과 Self-Refine

Reflection은 모델이 자기 출력이나 중간 결과를 다시 점검하는 흐름이다.

예를 들어 메일 요약에서는 개별 요약이 빠뜨린 내용을 다시 보정하거나, 최종 리포트가 요구 형식을 지키는지 확인할 수 있다. RAG에서는 검색 근거가 부족할 때 질문을 다시 쓰거나, groundedness를 확인한 뒤 재생성할 수 있다.

| 패턴 | 역할 |
| --- | --- |
| Self-Refine | 초안 답변을 다시 고친다 |
| Groundedness Check | 근거에 없는 내용을 줄인다 |
| Evaluator-Optimizer | 평가자가 피드백하고 생성자가 다시 만든다 |
| Query Rewrite Loop | 검색 실패 시 질문을 바꿔 다시 검색한다 |

이 패턴은 품질을 올릴 수 있지만 비용과 latency를 늘린다. 그래서 모든 요청에 붙이기보다 실패 가능성이 큰 구간에 제한적으로 붙이는 편이 낫다.

## Multi-Agent는 역할 분리다

Multi-Agent는 여러 모델 인스턴스를 붙이는 것이 아니라 역할을 분리하는 방식으로 이해하는 편이 좋다.

| 역할 | 예 |
| --- | --- |
| Planner | 작업 순서 설계 |
| Retriever | 필요한 근거 검색 |
| Domain Agent | 특정 영역 분석 |
| Evaluator | 결과 품질 점검 |
| Reporter | 최종 보고서 생성 |

복잡한 리포트 생성에서는 역할 분리가 자연스럽다. 기업 분석, 뉴스, 재무, 거시경제, 차트처럼 근거 축이 다르면 하나의 prompt에 모두 넣는 것보다 역할을 나누는 편이 읽기 쉽다.

다만 역할을 나누면 workflow가 길어진다. 한 실습 흐름에서는 직렬 Agent 구조가 2~3분 latency 한계로 언급되기도 했다. 그래서 Multi-Agent는 품질 이득과 대기시간을 함께 봐야 한다.

## Workflow와 Agent를 구분한다

모든 자동화가 Agent일 필요는 없다.

| 상황 | 적합한 구조 |
| --- | --- |
| 순서가 고정된 요약 파이프라인 | Workflow |
| 질문에 따라 도구가 달라짐 | Agent |
| 품질 점검 후 재생성 필요 | Evaluator-Optimizer |
| 여러 도메인 분석을 합침 | Multi-Agent |
| 민감한 외부 실행 포함 | Human-in-the-Loop |

실습 관점에서는 먼저 workflow로 만들고, 동적으로 판단해야 하는 부분만 Agent로 바꾸는 방식이 안전하다.

## 체크리스트

Agent workflow를 설계할 때 다음 질문을 남겨두면 좋다.

| 체크 | 질문 |
| --- | --- |
| 도구 선택 | 모델이 왜 이 도구를 골랐는지 추적 가능한가 |
| 권한 경계 | 외부 상태 변경 전에 승인 단계가 있는가 |
| 실패 처리 | tool call 실패 시 재시도와 fallback이 있는가 |
| 평가 루프 | 결과를 다시 점검하는 기준이 있는가 |
| latency | 사용자가 기다릴 수 있는 시간 안에 끝나는가 |
| 로그 | 중간 tool call과 결과가 기록되는가 |

## 코드 조각으로 다시 보기

![Agent reflection loop code note](/assets/images/study/code-notes/study-code-agent-reflection-loop.png)

이 코드 조각에서 Agent workflow를 다시 보게 됐다. 핵심은 모델 호출이 아니라 반복 상태다. 생성 결과를 만들고, 평가하고, 기준을 넘지 못하면 reflection을 저장한 뒤 다시 생성한다. 이 구조는 단순한 prompt chaining보다 실패 원인을 남기기 좋다.

내가 압축해서 가져간 형태는 다음과 같다.

```python
def run_reflection_loop(task: str, max_iter: int = 3) -> str:
    memory = []
    best_answer = ""
    best_score = 0.0

    for _ in range(max_iter):
        draft = generator.run(task, reflections=memory)
        score, feedback = evaluator.score(task, draft)

        if score > best_score:
            best_score = score
            best_answer = draft

        if score >= PASS_THRESHOLD:
            break

        memory.append(reflector.summarize(feedback))

    return best_answer
```

여기서 중요한 것은 반복 횟수와 통과 기준이다. reflection loop는 품질을 올릴 수 있지만, 무제한으로 돌리면 비용과 latency가 커진다. 그래서 Agent workflow에는 항상 budget이 같이 있어야 한다.

## 정리

Agent Workflow는 모델이 똑똑한지보다 실행 흐름이 통제되는지가 중요하다.

내가 가져간 결론은 이것이다. Agent를 붙이기 전에 tool boundary, reflection 기준, latency budget을 먼저 설계해야 한다.
