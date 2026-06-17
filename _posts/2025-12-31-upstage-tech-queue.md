---
title: "큐 (Queue)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-2.DATA_STRUCTURE_ALGORITHM
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- data-structures
- tech-note
- data-structure-algorithm
toc: true
date: 2025-12-31 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## 큐 (Queue)

> **한줄 정의**
> FIFO(First In, First Out) 구조의 선형 자료구조. 먼저 삽입된 원소가 가장 먼저 삭제된다.

## 핵심 이해

큐의 핵심 연산은 **enqueue**(뒤에 삽입)와 **dequeue**(앞에서 삭제)다. Python에서는 `collections.deque`를 사용하면 양쪽 끝 O(1) 연산이 가능하다. 리스트의 `pop(0)`은 O(n)이므로 큐 구현에 부적합하다.

원형 큐(Circular Queue)는 배열 기반 큐의 메모리 낭비를 해결한다. 우선순위 큐(Priority Queue)는 원소마다 우선순위를 부여하여 높은 우선순위 순서로 dequeue한다(힙으로 구현). BFS(너비 우선 탐색)의 핵심 자료구조이며, 작업 스케줄링, 프린터 스풀, 이벤트 처리에 활용된다.

## 구조 시각화

![큐 (Queue) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-queue-diagram-1.svg)

## 관련 개념

- 스택 - LIFO 구조와 비교
- 힙 - 우선순위 큐의 구현
- BFS-DFS - BFS에서 큐 활용
