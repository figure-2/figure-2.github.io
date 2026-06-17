---
title: "재귀 (Recursion)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-2.DATA_STRUCTURE_ALGORITHM
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- algorithms
- tech-note
- data-structure-algorithm
toc: true
date: 2026-01-05 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## 재귀 (Recursion)

> **한줄 정의**
> 함수가 자기 자신을 호출하여 문제를 더 작은 부분 문제로 분해하여 해결하는 알고리즘 기법.

## 핵심 이해

재귀의 필수 요소는 **기저 조건(Base Case)**과 **재귀 호출(Recursive Call)**이다. 기저 조건 없이 재귀를 실행하면 스택 오버플로우(Stack Overflow)가 발생한다. Python의 기본 재귀 깊이 제한은 1000이며 `sys.setrecursionlimit()`으로 변경 가능하다.

**분할 정복(Divide and Conquer)**은 재귀의 대표 패턴이다. 문제를 같은 형태의 부분 문제로 나누고(Divide), 각각 해결하여(Conquer), 결합(Combine)한다. 피보나치, 팩토리얼, 하노이 탑이 고전적 예시다. **메모이제이션(Memoization)**으로 중복 계산을 캐싱하면 지수 복잡도를 다항식으로 줄일 수 있다.

## 실행 흐름

![재귀 (Recursion) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-recursion-diagram-1.svg)

## 관련 개념

- 스택 - 재귀는 내부적으로 콜 스택 사용
- BFS-DFS - DFS의 재귀 구현
- 시간복잡도 - 재귀 알고리즘의 복잡도 분석
- 정렬-알고리즘 - 병합 정렬, 퀵 정렬의 재귀 구조
