---
title: "정렬 알고리즘"
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
date: 2026-01-06 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## 정렬 알고리즘

> **한줄 정의**
> 데이터를 특정 순서로 나열하는 알고리즘. 시간/공간 복잡도와 안정성(Stability)에 따라 상황에 맞는 알고리즘을 선택한다.

## 핵심 이해

**비교 기반 정렬**의 하한은 O(n log n)이다. 버블/선택/삽입 정렬은 O(n²)으로 소규모 데이터에 적합하고, 병합/퀵/힙 정렬은 O(n log n)으로 대규모에 적합하다. Python의 `sorted()`와 `list.sort()`는 Timsort(병합+삽입 정렬 하이브리드)를 사용한다.

**퀵 정렬**은 평균 O(n log n)이지만 최악(이미 정렬된 배열) O(n²)이 가능하다. **병합 정렬**은 항상 O(n log n)이며 안정 정렬이지만 O(n) 추가 공간이 필요하다. **힙 정렬**은 O(n log n)에 제자리 정렬이지만 캐시 지역성이 낮다.

## 복잡도 비교

![정렬 알고리즘 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-sorting-algorithm-diagram-1.svg)

## 관련 개념

- 시간복잡도 - 정렬 알고리즘 복잡도 분석
- 힙 - 힙 정렬의 기반 자료구조
- 재귀 - 병합 정렬, 퀵 정렬의 분할 정복
