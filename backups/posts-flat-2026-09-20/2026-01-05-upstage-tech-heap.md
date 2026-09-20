---
title: "힙 (Heap)"
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
date: 2026-01-05 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## 힙 (Heap)

> **한줄 정의**
> 완전 이진 트리 기반의 자료구조. 부모 노드가 항상 자식보다 크거나(최대 힙) 작은(최소 힙) 성질을 만족한다.

## 핵심 이해

**최대 힙(Max Heap)**은 루트가 최댓값, **최소 힙(Min Heap)**은 루트가 최솟값이다. 삽입과 삭제 모두 O(log n)이며, 최댓값/최솟값 조회는 O(1)이다. 배열로 구현하며 인덱스 i의 부모는 (i-1)//2, 자식은 2i+1과 2i+2다.

Python의 `heapq` 모듈은 최소 힙을 제공한다. 최대 힙이 필요하면 값에 음수를 취해 삽입한다. **우선순위 큐(Priority Queue)**의 표준 구현체이며, 힙 정렬(Heap Sort)과 Dijkstra 최단 경로 알고리즘에 활용된다.

## 구조 시각화

![힙 (Heap) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-heap-diagram-1.svg)

## 관련 개념

- 트리 - 완전 이진 트리 구조
- 큐 - 우선순위 큐의 힙 구현
- 정렬-알고리즘 - 힙 정렬(Heap Sort)
