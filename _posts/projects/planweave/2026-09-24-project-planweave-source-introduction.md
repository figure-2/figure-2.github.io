---
title: "01. PlanWeave 프로젝트 소개와 서비스 구성"
description: "PlanWeave의 문제 정의, 문서 생성 파이프라인과 서비스 기능을 정리합니다."
categories:
  - 2.PROJECT
  - 2-9. PlanWeave
tags:
  - 프로젝트 자료
toc: true
date: 2026-09-24 00:00:00 +0900
comments: true
mermaid: true
math: false
---

PlanWeave의 문제 정의, 문서 생성 파이프라인과 서비스 기능을 정리합니다.

{% raw %}

<!-- 목차는 제목 구조에서 자동 생성할 수 있습니다. -->

## PlanWeave 소개

> **“모호한 Vibe를 정교한 Design으로”**
> 아이디어의 파편을 모아 논리적인 기획서로 구체화합니다.

### PlanWeave 개발 배경

---

![image.png](/assets/images/source-archives/planweave/intro-004.png)

_[https://ch.yes24.com/Article/Details/27512](https://ch.yes24.com/Article/Details/27512?utm_source=chatgpt.com)_

![image.png](/assets/images/source-archives/planweave/intro-005.png)

_[https://www.newstap.co.kr/news/articleView.html](https://www.newstap.co.kr/news/articleView.html?idxno=309357&utm_source=chatgpt.com)_

![스크린샷 2026-02-08 231844.png](/assets/images/source-archives/planweave/intro-006.png)

_[https://www.newstap.co.kr/news/articleView.html](https://www.newstap.co.kr/news/articleView.html?idxno=309357&utm_source=chatgpt.com)_

> 💡

부정확한 요구 사항과 단순 반복 업무로 소모되는 **기획의 에너지**를 바로잡습니다.

본 프로젝트는 **'빈 화면 공포증'**을 해소하는 논리적 가이드를 제공하여, 사용자가 저부가가치 업무에서 벗어나 **창의적인 전략 설계**에만 온전히 집중할 수 있는 환경을 구축합니다.

---

### 핵심 전략 및 솔루션

1. 적응형 추론 엔진 기반의 **‘동적 기획 프레임워크’** 생성
   고정된 양식을 불러오는 것이 아닌, 입력값의 맥락에 따라 매번 다르게 생성되는 **‘맞춤형 기획 아키텍처’**입니다.

![A_clean_line_iconography_architectural_diagram_sho-1770563398757.png](/assets/images/source-archives/planweave/intro-007.png)

1. 논리적 공백 추적 및 **‘수렴형 역질문’**
   LLM의 무한한 생성을 제어하고 실질적인 결과물에 도달하기 위해, 본 프로젝트는 ‘상태 기반 공백 분석’ 메커니즘을 채택합니다.

![Line_iconography_diagram_illustrating_a_convergent-1770564148129.png](/assets/images/source-archives/planweave/intro-008.png)

---

### 유사 서비스 대비 차별점

| 비교 항목       | 일반 AI 챗봇 (GPT 등)            | 기존 기획 자동화 툴 | **PlanWeave (Our Project)** |
| --------------- | -------------------------------- | ------------------- | --------------------------- |
| **시작 방식**   | 사용자 질문 (프롬프트 설계 필요) | 고정된 템플릿       | **아이디어 기반 역제안**    |
| **기획 주도권** | AI (일방적 생성)                 | 시스템 (정해진 틀)  | **사용자 (인터랙션 중심)**  |
| **논리 구조화** | 별도 정리 필요                   | 특정 분야 한정      | **실시간 동적 구조화**      |
| **맥락 유지**   | 대화 흐름에 의존                 | 입력값 단순 반영    | **상태 기반 공백 추적**     |

## 서비스 구조

![Line_iconography_diagram_showing_a_supervisor-base-1770564525986.png](/assets/images/source-archives/planweave/intro-009.png)

**중앙 제어 Agent(Supervisor)**가 기획의 전 과정을 모니터링하며 흐름을 이끌어 나갑니다.

아이디어 구조화부터 자료조사, 초안 작성까지 모든 단계에서 하나의 맥락이 흐트러지지 않도록 관리하며, 기획의 본질에서 벗어나지 않는 일관된 결과물을 보장합니다.

---

![image (3).png](/assets/images/source-archives/planweave/intro-010.png)

---

![88E3057F-A2A7-4B79-BAA7-D6C022915D58.png](/assets/images/source-archives/planweave/intro-011.png)

---

![image.png](/assets/images/source-archives/planweave/intro-012.png)

## 서비스 주요 기능

![1CA3E0D3-B424-4E8E-925F-8FB89384DE1A.png](/assets/images/source-archives/planweave/intro-013.png)

**1️⃣ 자료 조사 내역(좌측 사이드바):**

- AI 에이전트가 상호작용 과정의 맥락을 바탕으로 웹 서치로 수집한 자료 목록이 표시됩니다.

**2️⃣ 기획서 (중앙 화면):**

- AI 에이전트와 상호작용을 통해 생성되고 수정되는 기획서의 내용이 실시간으로 표시됩니다.

**3️⃣ 대화창(우측 사이드바):**

- AI 에이전트와 상호작용하는 주요 공간입니다.
- 아래 화면과 같이 동적으로 변화하며 사용자와의 상호작용을 이끌어 나갑니다.
  ![image.png](/assets/images/source-archives/planweave/intro-014.png)

![image.png](/assets/images/source-archives/planweave/intro-015.png)

![image.png](/assets/images/source-archives/planweave/intro-016.png)

## 시연 영상

[영상 열기](https://www.youtube.com/watch?v=2hMNEXIO3zQ)

{% endraw %}

## 관련 글

- [02. PlanWeave 부분 문서 수정 에이전트 아키텍처 — 설계 기록]({% post_url projects/planweave/2026-09-24-project-planweave-source-partial-edit-design %})
- [03. PlanWeave 문서 부분 수정 배치 실험 — 결과 기록]({% post_url projects/planweave/2026-09-24-project-planweave-source-batch-edit-report %})
