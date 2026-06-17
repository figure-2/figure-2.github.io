---
title: "BFS / DFS"
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
date: 2026-01-02 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## BFS / DFS

> **한줄 정의**
> 그래프와 트리를 탐색하는 두 가지 핵심 알고리즘. BFS는 너비 우선, DFS는 깊이 우선으로 노드를 방문한다.

## 학습 맥락

자료구조와 알고리즘 과정에서 BFS/DFS는 단순한 코딩 테스트 기법이 아니라, "연결된 구조를 어떤 순서로 읽을 것인가"를 판단하는 기본 도구로 다뤘다. 트리, 그래프, 큐, 스택을 한 번에 연결해서 이해해야 하므로 각각의 자료구조를 따로 외우는 것보다 탐색 흐름을 먼저 잡는 것이 중요하다.

이후 Agentic Workflow나 LangGraph를 배울 때도 이 감각이 다시 나온다. 에이전트의 노드, 조건부 라우팅, 실행 경로도 결국 그래프 구조로 볼 수 있기 때문이다. 그래서 BFS/DFS는 알고리즘 주차에서 끝나는 개념이 아니라, 뒤의 에이전트 설계까지 이어지는 기초 체력에 가깝다.

## 핵심 개념

**BFS(Breadth-First Search)**는 큐를 사용하여 같은 레벨의 노드를 먼저 탐색한다. 최단 경로 탐색(가중치 없는 그래프), 레벨 순서 순회에 적합하다. 시간/공간 복잡도 모두 O(V+E)다.

**DFS(Depth-First Search)**는 스택(또는 재귀)을 사용하여 한 방향으로 끝까지 탐색 후 백트래킹한다. 경로 존재 여부, 사이클 탐지, 위상 정렬, 연결 요소 탐색에 활용된다. 재귀 구현이 직관적이며 시간/공간 복잡도 O(V+E)다.

## 탐색 순서 비교

![BFS / DFS 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-bfs-dfs-diagram-1.svg)

## 언제 BFS를 쓰는가

BFS는 "가까운 것부터" 확인해야 할 때 적합하다. 대표적으로 무가중치 그래프에서 시작점부터 목표 지점까지의 최단 거리, 단계별 확산, 레벨 단위 탐색을 구할 때 사용한다.

예를 들어 미로에서 출발점부터 도착점까지 최소 이동 횟수를 구하거나, 소셜 그래프에서 특정 사용자와 몇 단계 떨어져 있는지 확인하는 문제는 BFS로 접근하는 것이 자연스럽다. 큐에 먼저 들어온 노드를 먼저 처리하기 때문에 같은 거리의 후보를 모두 확인한 뒤 다음 거리로 넘어간다.

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for next_node in graph[node]:
            if next_node not in visited:
                visited.add(next_node)
                queue.append(next_node)

    return order
```

## 언제 DFS를 쓰는가

DFS는 "한 경로를 끝까지" 확인해야 할 때 적합하다. 경로 존재 여부, 백트래킹, 사이클 탐지, 연결 요소 분리, 깊은 의존성 탐색에서 자주 사용한다.

재귀로 구현하면 코드가 짧고 직관적이지만, 입력 크기가 크면 재귀 깊이 제한에 걸릴 수 있다. 그런 경우에는 명시적인 스택을 사용해 반복문으로 바꾸는 편이 안전하다.

```python
def dfs(graph, start):
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for next_node in reversed(graph[node]):
            if next_node not in visited:
                stack.append(next_node)

    return order
```

## 구현 체크포인트

- `visited`를 반드시 관리한다. 그래프에 사이클이 있으면 무한 반복이 발생할 수 있다.
- BFS는 큐, DFS는 스택 또는 재귀를 사용한다.
- 그래프 표현은 보통 인접 리스트가 효율적이다.
- 최단 거리가 필요하면 BFS를 먼저 떠올린다. 단, 간선에 가중치가 있으면 다익스트라 같은 다른 알고리즘이 필요하다.
- DFS 재귀 구현은 입력 크기와 재귀 제한을 함께 확인한다.

## 관련 글

- [트리, 그래프, DFS/BFS 탐색]({% post_url 2026-01-02-upstage-course-w02d03-tree-graph-explore %})
- [그래프]({% post_url 2026-01-02-upstage-tech-graph %})
- [트리]({% post_url 2026-01-02-upstage-tech-tree %})
- [스택]({% post_url 2025-12-31-upstage-tech-stack %})
- [큐]({% post_url 2025-12-31-upstage-tech-queue %})
- [재귀]({% post_url 2026-01-05-upstage-tech-recursion %})
