---
title: SWEA 알고리즘 - Queue 및 BFS
categories:
- 1.TIL
- 1-4.ALGORITHM

tags:
- algorithm
- swea
- queue
- bfs
- breadth-first-search
- graph-traversal
date: 2023-08-03 09:00:00 +0900
toc: true
comments: false
mermaid: true
math: true
---

# SWEA 알고리즘 - Queue 및 BFS

> **SW Expert Academy** 파이썬 SW문제해결 기본 과정의 Queue 관련 알고리즘 학습

## Queue 알고리즘 기초

### 1. Queue 자료 구조의 개념
- FIFO (First In, First Out) 원리
- Queue의 기본 연산 (enqueue, dequeue, peek)
- Python에서 Queue 구현 방법
- Queue의 활용 사례

### 2. Queue의 활용
- BFS (너비 우선 탐색)
- 레벨별 탐색
- 최단 경로 찾기
- 프로세스 스케줄링

### 3. BFS (너비 우선 탐색)
- 그래프 탐색 알고리즘
- 레벨별로 탐색하는 특성
- 최단 경로 보장
- 큐를 이용한 구현

### 4. BFS vs DFS 비교
- **BFS**: 최단 경로, 레벨별 탐색
- **DFS**: 깊이 우선, 스택/재귀 활용
- **시간 복잡도**: O(V + E)
- **공간 복잡도**: O(V)

## 관련 문제

### Queue 문제들
- **[회전](https://github.com/figure-2/Algorithm/blob/master/swea/5097_%ED%9A%8C%EC%A0%84/sol.py)** - 큐의 기본 연산
- **[미로의거리](https://github.com/figure-2/Algorithm/blob/master/swea/5105_%EB%AF%B8%EB%A1%9C%EC%9D%98%EA%B1%B0%EB%A6%AC/sol.py)** - BFS를 이용한 최단 거리
- **[피자굽기](https://github.com/figure-2/Algorithm/blob/master/swea/5099_%ED%94%BC%EC%9E%90%EA%B5%BD%EA%B8%B0/sol.py)** - 큐의 실무 활용
- **[노드의거리](https://github.com/figure-2/Algorithm/blob/master/swea/5102_%EB%85%B8%EB%93%9C%EA%B1%B0%EB%A6%AC/sol.py)** - BFS 그래프 탐색

## 학습 포인트

### BFS 알고리즘 핵심
1. **레벨별 탐색**: 같은 거리의 노드들을 먼저 탐색
2. **최단 경로**: 가중치가 없는 그래프에서 최단 경로 보장
3. **큐 활용**: FIFO 특성을 이용한 탐색 순서 관리

### 구현 패턴
```python
from collections import deque

def bfs(graph, start):
    queue = deque([start])
    visited = set([start])
    
    while queue:
        node = queue.popleft()
        # 노드 처리
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 성능 최적화
- **방문 체크**: 중복 방문 방지
- **큐 크기 관리**: 메모리 사용량 최적화
- **조기 종료**: 목표 도달 시 탐색 중단

### 실무 적용
- **네트워크 탐색**: 최단 경로 찾기
- **게임 AI**: 레벨별 탐색
- **소셜 네트워크**: 친구 추천 시스템
- **웹 크롤링**: 레벨별 페이지 탐색

Queue와 BFS를 통해 체계적인 그래프 탐색 능력을 기를 수 있습니다.
