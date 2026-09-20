---
title: "업스테이지 AI Agent 서비스 파이프라인 과정 학습 정리"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-11.RESOURCES
- REFERENCE_NOTE
tags:
- upstage
- sesac
- ai-agent
- llm
- rag
- prompt-engineering
- llmops
- learning-roadmap
- resources
- reference-note
toc: true
date: 2026-05-15 00:00:00 +0900
comments: false
mermaid: true
math: true
---
## 업스테이지 AI Agent 서비스 파이프라인 과정 학습 정리

## 개요

이 카테고리는 `업스테이지 - AI Agent 서비스 파이프라인 기획부터 구현까지 (새싹)` 과정에서 학습한 내용을 블로그용으로 다시 정리하는 공간이다.

원본 학습 노트는 Obsidian 볼트 형태로 정리되어 있으며, 블로그에는 수업 기록형, 기술 노트형, 실습형, 프로젝트 정리형, 참고자료형으로 나누어 이관했다.

## 교육 일정 기준

첨부한 시간표를 기준으로 블로그 날짜를 다시 맞췄다. 수업 노트와 기술 노트는 가능한 한 해당 개념을 처음 학습한 날짜에 배치하고, 로드맵/참고자료처럼 과정 전체를 정리한 글은 별도 작성일을 유지했다.

| 기간 | 과정 흐름 | 블로그 정리 범위 |
| --- | --- | --- |
| 2025-12-22 ~ 2025-12-29 | AI Literacy | AI 기초, 대화 설계, 산업 적용, AI 윤리 |
| 2025-12-30 ~ 2026-01-06 | 자료구조와 알고리즘 | 배열, 스택/큐/해시, 트리/그래프, 힙, 정렬 |
| 2026-01-07 ~ 2026-01-12 | 개발환경 구성 | Python 환경, Git, GitHub 협업, Docker/MySQL |
| 2026-01-14 ~ 2026-01-20 | 네트워크와 클라우드 | HTTP, 클라우드, 웹서버, 배포 자동화 |
| 2026-01-21 ~ 2026-01-27 | Prompt & Context Engineering | 프롬프팅, RAG, Advanced RAG, LLM 보안 |
| 2026-01-28 ~ 2026-02-03 | Agentic Workflow | Agent 설계, Tool Calling, MCP, Memory, Evaluation, AgentOps |
| 2026-02-04 ~ 2026-02-20 | LLM Product Engineering Project | 서비스 기획, 워크플로우 설계, Agent Architecture, LangGraph MVP |
| 2026-02-23 ~ 2026-02-27 | Service Deployment | CI/CD, 배포 자동화 |
| 2026-03-03 ~ 2026-03-09 | LLM Ops | LiteLLM, 상태관리, Observability, Error Analysis, LLMOps 패턴 |
| 2026-03-10 ~ 2026-05-01 | Final Project / 기업 연계 프로젝트 | 현재 원본 노트 기준 별도 post 없음 |

## 정리 기준

- 수업 기록형은 원본 주차/일차 흐름을 보존한다.
- 기술 노트형은 개념을 다시 참고하기 쉽도록 정의, 사용 맥락, 구현 관점 중심으로 정리한다.
- 실습형은 목표, 구현 흐름, 체크포인트 중심으로 정리한다.
- 프로젝트형은 문제 정의, 아키텍처, 구현 포인트, 회고 중심으로 정리한다.
- 공개하면 안 되는 키, 토큰, DB URL, 계정 정보는 제외하거나 마스킹한다.

## 2차 품질 개선 기준

1차 이관본은 Obsidian 노트를 블로그 형식으로 옮기는 데 초점을 뒀다. 2차 정리에서는 멀티캠퍼스 TIL처럼 사람이 읽었을 때 학습 흐름이 보이도록 다음 항목을 보강한다.

- 학습 맥락: 이 개념을 과정의 어느 시점에 왜 배웠는지 설명한다.
- 핵심 개념: 정의만 두지 않고 판단 기준과 용도를 함께 적는다.
- 구현 관점: 실제 프로젝트에서 어떤 순서로 적용할지 정리한다.
- 주의점: 비용, 보안, 디버깅, 운영 리스크를 분리해 기록한다.
- 관련 글: 수업 기록, 기술 노트, 실습, 프로젝트 글을 서로 연결한다.

## 카테고리 구조

```text
1.TIL
  1-2.UPSTAGE_AI_AGENT
    1-2-1.AI_LITERACY  # AI Literacy
    1-2-2.DATA_STRUCTURE_ALGORITHM  # 자료구조와 알고리즘
    1-2-3.DEV_ENV_GIT_DOCKER  # 개발환경, Git, Docker
    1-2-4.NETWORK_CLOUD  # 네트워크와 클라우드
    1-2-5.PROMPT_ENGINEERING_RAG  # Prompt Engineering & RAG
    1-2-6.AGENTIC_WORKFLOW  # Agentic Workflow
    1-2-7.AI_SERVICE_PLANNING  # AI 서비스 기획
    1-2-8.AGENT_ARCHITECTURE  # Agent Architecture
    1-2-9.LLMOPS  # LLMOps
    1-2-10.PROJECTS  # Projects
    1-2-11.RESOURCES  # Resources
```

## 전체 목차

## 1-2-1.AI_LITERACY - AI Literacy

### 수업 기록형

- [AI 대화 설계 및 컨텍스트 엔지니어링]({% post_url 2025-12-24-upstage-course-w01d03-aiconversation-design %}) — `01_Modules/AI-기초/W01D03-AI대화-설계.md`
- [산업별 AI 적용 사례 탐구]({% post_url 2025-12-26-upstage-course-w01d04-industry-ai-application %}) — `01_Modules/AI-기초/W01D04-산업별-AI-적용.md`
- [AI 윤리와 사회적 영향]({% post_url 2025-12-29-upstage-course-w01d05-ai-ethics-society %}) — `01_Modules/AI-기초/W01D05-AI-윤리와-사회.md`
- [Week 01 - AI Literacy]({% post_url 2025-12-22-upstage-course-week01-ai-literacy %}) — `01_Modules/AI-기초/Week01-AI-Literacy.md`

### 실습형

- [AI 윤리와 사회적 영향 실습]({% post_url 2025-12-29-upstage-practice-ai-ethics-practice %}) — `04_Exercises/AI-윤리-실습.md`

## 1-2-2.DATA_STRUCTURE_ALGORITHM - 자료구조와 알고리즘

### 수업 기록형

- [알고리즘 분석 기초 - Big-O, 배열, 링크드 리스트]({% post_url 2025-12-30-upstage-course-w02d01-algorithm-analysis %}) — `01_Modules/자료구조-알고리즘/W02D01-알고리즘-분석.md`
- [스택, 큐, 해시 테이블]({% post_url 2025-12-31-upstage-course-w02d02-stack-queue-hash %}) — `01_Modules/자료구조-알고리즘/W02D02-스택-큐-해시.md`
- [트리, 그래프, DFS/BFS 탐색]({% post_url 2026-01-02-upstage-course-w02d03-tree-graph-explore %}) — `01_Modules/자료구조-알고리즘/W02D03-트리-그래프-탐색.md`
- [우선순위 큐, 힙, 재귀호출, 탐색 알고리즘]({% post_url 2026-01-05-upstage-course-w02d04-heap-recursion-explore %}) — `01_Modules/자료구조-알고리즘/W02D04-힙-재귀-탐색.md`
- [정렬 알고리즘과 코딩 문제 풀이]({% post_url 2026-01-06-upstage-course-w02d05-sorting-problem-solving %}) — `01_Modules/자료구조-알고리즘/W02D05-정렬-문제풀이.md`
- [Week 02 - 자료구조와 알고리즘]({% post_url 2025-12-30-upstage-course-week02-data-structure-algorithm %}) — `01_Modules/자료구조-알고리즘/Week02-자료구조-알고리즘.md`

### 기술 노트형

- [BFS / DFS]({% post_url 2026-01-02-upstage-tech-bfs-dfs %}) — `02_Concepts/BFS-DFS.md`
- [그래프 (Graph)]({% post_url 2026-01-02-upstage-tech-graph %}) — `02_Concepts/그래프.md`
- [스택 (Stack)]({% post_url 2025-12-31-upstage-tech-stack %}) — `02_Concepts/스택.md`
- [시간복잡도 (Time Complexity)]({% post_url 2025-12-30-upstage-tech-time-complexity %}) — `02_Concepts/시간복잡도.md`
- [재귀 (Recursion)]({% post_url 2026-01-05-upstage-tech-recursion %}) — `02_Concepts/재귀.md`
- [정렬 알고리즘]({% post_url 2026-01-06-upstage-tech-sorting-algorithm %}) — `02_Concepts/정렬-알고리즘.md`
- [큐 (Queue)]({% post_url 2025-12-31-upstage-tech-queue %}) — `02_Concepts/큐.md`
- [트리 (Tree)]({% post_url 2026-01-02-upstage-tech-tree %}) — `02_Concepts/트리.md`
- [해시 (Hash)]({% post_url 2025-12-31-upstage-tech-hash %}) — `02_Concepts/해시.md`
- [힙 (Heap)]({% post_url 2026-01-05-upstage-tech-heap %}) — `02_Concepts/힙.md`

### 실습형

- [스택/큐/해시 실습]({% post_url 2025-12-31-upstage-practice-stackqueue-practice %}) — `04_Exercises/스택큐-실습.md`
- [알고리즘 분석 실습]({% post_url 2025-12-30-upstage-practice-algorithm-practice %}) — `04_Exercises/알고리즘-실습.md`
- [정렬/문제풀이 실습]({% post_url 2026-01-06-upstage-practice-sorting-practice %}) — `04_Exercises/정렬-실습.md`
- [트리/그래프 탐색 실습]({% post_url 2026-01-02-upstage-practice-treegraph-practice %}) — `04_Exercises/트리그래프-실습.md`
- [힙/재귀 실습]({% post_url 2026-01-05-upstage-practice-heaprecursion-practice %}) — `04_Exercises/힙재귀-실습.md`

## 1-2-3.DEV_ENV_GIT_DOCKER - 개발환경, Git, Docker

### 수업 기록형

- [개발환경 구성 - IDE, Python, uv]({% post_url 2026-01-07-upstage-course-w03d01-dev-env-setup %}) — `01_Modules/개발환경-버전관리/W03D01-개발환경-구성.md`
- [Git 기본과 심화 - 버전 관리의 핵심]({% post_url 2026-01-08-upstage-course-w03d02-git-basics-advanced %}) — `01_Modules/개발환경-버전관리/W03D02-Git-기본-심화.md`
- [GitHub 협업 - PR, Issue, 프로젝트 관리]({% post_url 2026-01-09-upstage-course-w03d03-github-collaboration %}) — `01_Modules/개발환경-버전관리/W03D03-GitHub-협업.md`
- [Docker 인프라와 MySQL 데이터베이스]({% post_url 2026-01-12-upstage-course-w03d04-docker-mysql %}) — `01_Modules/개발환경-버전관리/W03D04-Docker-MySQL.md`
- [Week 03 - 개발환경, Git, Docker]({% post_url 2026-01-07-upstage-course-week03-dev-env-git-docker %}) — `01_Modules/개발환경-버전관리/Week03-개발환경-Git-Docker.md`

### 기술 노트형

- [CI/CD]({% post_url 2026-02-26-upstage-tech-ci-cd %}) — `02_Concepts/CI-CD.md`
- [Docker]({% post_url 2026-01-12-upstage-tech-docker %}) — `02_Concepts/Docker.md`
- [Git]({% post_url 2026-01-08-upstage-tech-git %}) — `02_Concepts/Git.md`

### 실습형

- [Docker/MySQL 실습]({% post_url 2026-01-12-upstage-practice-docker-practice %}) — `04_Exercises/Docker-실습.md`
- [Git 기본/심화 실습]({% post_url 2026-01-08-upstage-practice-git-practice %}) — `04_Exercises/Git-실습.md`
- [GitHub 협업 실습]({% post_url 2026-01-09-upstage-practice-github-practice %}) — `04_Exercises/GitHub-실습.md`
- [개발환경 구성 실습]({% post_url 2026-01-07-upstage-practice-dev-env-practice %}) — `04_Exercises/개발환경-실습.md`

## 1-2-4.NETWORK_CLOUD - 네트워크와 클라우드

### 수업 기록형

- [HTTP와 웹 통신 이해]({% post_url 2026-01-14-upstage-course-w04d01-http-web-communication %}) — `01_Modules/네트워크-클라우드/W04D01-HTTP-웹통신.md`
- [클라우드 컴퓨팅 시작하기]({% post_url 2026-01-15-upstage-course-w04d02-cloud-computing %}) — `01_Modules/네트워크-클라우드/W04D02-클라우드-컴퓨팅.md`
- [웹서버 사용자 서비스]({% post_url 2026-01-16-upstage-course-w04d03-web-server-service %}) — `01_Modules/네트워크-클라우드/W04D03-웹서버-서비스.md`
- [클라우드 배포 자동화]({% post_url 2026-01-19-upstage-course-w04d04-deployment-automation %}) — `01_Modules/네트워크-클라우드/W04D04-배포-자동화.md`
- [네트워크와 AI]({% post_url 2026-01-20-upstage-course-w04d05-network-ai %}) — `01_Modules/네트워크-클라우드/W04D05-네트워크와-AI.md`
- [Week 04 - 네트워크와 클라우드]({% post_url 2026-01-14-upstage-course-week04-network-cloud %}) — `01_Modules/네트워크-클라우드/Week04-네트워크-클라우드.md`

### 기술 노트형

- [HTTP]({% post_url 2026-01-14-upstage-tech-http %}) — `02_Concepts/HTTP.md`
- [클라우드 컴퓨팅]({% post_url 2026-01-15-upstage-tech-cloud-computing %}) — `02_Concepts/클라우드-컴퓨팅.md`

### 실습형

- [HTTP 웹통신 실습]({% post_url 2026-01-14-upstage-practice-http-practice %}) — `04_Exercises/HTTP-실습.md`
- [네트워크와 AI 실습]({% post_url 2026-01-20-upstage-practice-networkai-practice %}) — `04_Exercises/네트워크AI-실습.md`
- [클라우드 컴퓨팅 실습]({% post_url 2026-01-15-upstage-practice-cloud-practice %}) — `04_Exercises/클라우드-실습.md`
- [웹서버 서비스 실습]({% post_url 2026-01-16-upstage-practice-web-server-practice %}) — `04_Exercises/웹서버-실습.md`

## 1-2-5.PROMPT_ENGINEERING_RAG - Prompt Engineering & RAG

### 수업 기록형

- [Introduction & Basic Prompting]({% post_url 2026-01-21-upstage-course-w05d01-prompting-basics %}) — `01_Modules/Prompt-Engineering-RAG/W05D01-프롬프팅-기초.md`
- [고급 LLM 프롬프팅 전략]({% post_url 2026-01-22-upstage-course-w05d02-advanced-prompting %}) — `01_Modules/Prompt-Engineering-RAG/W05D02-고급-프롬프팅.md`
- [RAG Fundamentals & Advanced RAG 1]({% post_url 2026-01-23-upstage-course-w05d03-rag-basics %}) — `01_Modules/Prompt-Engineering-RAG/W05D03-RAG-기초.md`
- [Advanced RAG 2 & LLM 보안]({% post_url 2026-01-26-upstage-course-w05d04-advanced-rag-security %}) — `01_Modules/Prompt-Engineering-RAG/W05D04-Advanced-RAG-보안.md`
- [Related Works & Agentic AI Summary]({% post_url 2026-01-27-upstage-course-w05d05-trend-agentic-ai %}) — `01_Modules/Prompt-Engineering-RAG/W05D05-트렌드-Agentic-AI.md`
- [Week 05 - Prompt Engineering & RAG]({% post_url 2026-01-21-upstage-course-week05-prompt-engineering-rag %}) — `01_Modules/Prompt-Engineering-RAG/Week05-Prompt-Engineering-RAG.md`

### 기술 노트형

- [Advanced RAG]({% post_url 2026-01-26-upstage-tech-advanced-rag %}) — `02_Concepts/Advanced-RAG.md`
- [LLM 보안]({% post_url 2026-01-26-upstage-tech-llm-security %}) — `02_Concepts/LLM-보안.md`
- [Prompt Engineering]({% post_url 2026-01-21-upstage-tech-prompt-engineering %}) — `02_Concepts/Prompt-Engineering.md`
- [RAG (Retrieval-Augmented Generation)]({% post_url 2026-01-23-upstage-tech-rag %}) — `02_Concepts/RAG.md`

### 실습형

- [Naive RAG 파이프라인 구축 실습]({% post_url 2026-01-23-upstage-practice-rag-practice %}) — `04_Exercises/RAG-실습.md`
- [RAG Knowledge Conflict 실습]({% post_url 2026-01-26-upstage-practice-rag-conflict-practice %}) — `04_Exercises/RAG-충돌-실습.md`
- [Visual-Text Embedding Alignment 실습]({% post_url 2026-01-27-upstage-practice-embedding-practice %}) — `04_Exercises/임베딩-실습.md`
- [Week05 실습: 프롬프트 설계 연습]({% post_url 2026-01-21-upstage-practice-w05-prompt-design %}) — `exercises/w05-prompt-design.md`
- [기본 프롬프팅 튜토리얼 실습]({% post_url 2026-01-21-upstage-practice-prompting-practice %}) — `04_Exercises/프롬프팅-실습.md`

## 1-2-6.AGENTIC_WORKFLOW - Agentic Workflow

### 수업 기록형

- [AI 서비스와 에이전트 설계]({% post_url 2026-01-28-upstage-course-w06d01-ai-service-design %}) — `01_Modules/Agentic-Workflow/W06D01-AI-서비스-에이전트-설계.md`
- [Tool Calling Fundamentals & MCP]({% post_url 2026-01-29-upstage-course-w06d02-tool-calling-mcp %}) — `01_Modules/Agentic-Workflow/W06D02-Tool-Calling-MCP.md`
- [Agentic RAG & Memory Management]({% post_url 2026-01-30-upstage-course-w06d03-agentic-rag-memory %}) — `01_Modules/Agentic-Workflow/W06D03-Agentic-RAG-Memory.md`
- [Context Engineering & Safety]({% post_url 2026-02-02-upstage-course-w06d04-context-engineering-safety %}) — `01_Modules/Agentic-Workflow/W06D04-Context-Engineering-Safety.md`
- [Evaluation & AgentOps]({% post_url 2026-02-03-upstage-course-w06d05-evaluation-agentops %}) — `01_Modules/Agentic-Workflow/W06D05-Evaluation-AgentOps.md`
- [Week 06: Agentic Workflow]({% post_url 2026-01-28-upstage-course-week06-agentic-workflow %}) — `01_Modules/Agentic-Workflow/Week06-Agentic-Workflow.md`

### 기술 노트형

- [Agent Evaluation]({% post_url 2026-02-03-upstage-tech-agent-evaluation %}) — `02_Concepts/Agent-Evaluation.md`
- [Agentic RAG]({% post_url 2026-01-30-upstage-tech-agentic-rag %}) — `02_Concepts/Agentic-RAG.md`
- [Agentic Workflow]({% post_url 2026-01-28-upstage-tech-agentic-workflow %}) — `02_Concepts/Agentic-Workflow.md`
- [Context Engineering]({% post_url 2026-02-02-upstage-tech-context-engineering %}) — `02_Concepts/Context-Engineering.md`
- [MCP (Model Context Protocol)]({% post_url 2026-01-29-upstage-tech-mcp %}) — `02_Concepts/MCP.md`
- [Memory Management]({% post_url 2026-01-30-upstage-tech-memory-management %}) — `02_Concepts/Memory-Management.md`
- [Tool Calling]({% post_url 2026-01-29-upstage-tech-tool-calling %}) — `02_Concepts/Tool-Calling.md`

### 실습형

- [Tool Calling 기본 과제]({% post_url 2026-01-29-upstage-practice-tool-calling-practice %}) — `04_Exercises/Tool-Calling-실습.md`
- [Agentic RAG + Memory 실습]({% post_url 2026-01-30-upstage-practice-agentic-rag-practice %}) — `04_Exercises/Agentic-RAG-실습.md`
- [Context Engineering + Safety 실습]({% post_url 2026-02-02-upstage-practice-context-safety-practice %}) — `04_Exercises/Context-Safety-실습.md`
- [Evaluation + AgentOps 실습]({% post_url 2026-02-03-upstage-practice-evaluation-practice %}) — `04_Exercises/Evaluation-실습.md`

## 1-2-7.AI_SERVICE_PLANNING - AI 서비스 기획

### 수업 기록형

- [AI 서비스 기획 가이드]({% post_url 2026-02-04-upstage-course-w07-ai-service-planning %}) — `01_Modules/서비스-기획/W07-AI-서비스-기획.md`
- [Agent 개발 팁, 효과적인 프로젝트 운영 전략]({% post_url 2026-02-06-upstage-course-w07-agent-tips %}) — `01_Modules/서비스-기획/W07-Agent-개발-팁.md`
- [AI Workflow Design]({% post_url 2026-02-05-upstage-course-w07-workflow-design %}) — `01_Modules/서비스-기획/W07-Workflow-Design.md`
- [Week 07 - AI 서비스 프로젝트]({% post_url 2026-02-04-upstage-course-week07 %}) — `01_Modules/서비스-기획/Week07-프로젝트.md`

## 1-2-8.AGENT_ARCHITECTURE - Agent Architecture

### 수업 기록형

- [Agent Architecture]({% post_url 2026-02-09-upstage-course-w08d01-agent-architecture %}) — `01_Modules/Agent-아키텍처/W08D01-Agent-Architecture.md`
- [LangGraph MVP]({% post_url 2026-02-10-upstage-course-w08d02-langgraph-mvp %}) — `01_Modules/Agent-아키텍처/W08D02-LangGraph-MVP.md`
- [Streaming 구현]({% post_url 2026-02-11-upstage-course-w08d03-streaming-implementation %}) — `01_Modules/Agent-아키텍처/W08D03-Streaming-구현.md`
- [GitHub Actions CI]({% post_url 2026-02-26-upstage-course-w08d04-ci %}) — `01_Modules/Agent-아키텍처/W08D04-CI.md`
- [CD (Continuous Deployment)]({% post_url 2026-02-27-upstage-course-w08d05-cd %}) — `01_Modules/Agent-아키텍처/W08D05-CD.md`
- [Week 08: Agent 아키텍처 / 서비스 배포]({% post_url 2026-02-09-upstage-course-week08-agent-architecture %}) — `01_Modules/Agent-아키텍처/Week08-Agent-아키텍처.md`

### 기술 노트형

- [Agent Architecture]({% post_url 2026-01-28-upstage-tech-agent-architecture %}) — `02_Concepts/Agent-Architecture.md`
- [FastAPI]({% post_url 2026-01-07-upstage-tech-fastapi %}) — `02_Concepts/FastAPI.md`
- [Gradio]({% post_url 2026-02-11-upstage-tech-gradio %}) — `02_Concepts/Gradio.md`
- [LangGraph]({% post_url 2026-01-28-upstage-tech-langgraph %}) — `02_Concepts/LangGraph.md`
- [Supabase]({% post_url 2026-02-10-upstage-tech-supabase %}) — `02_Concepts/Supabase.md`

### 실습형

- [LangGraph 에이전트 구현 실습]({% post_url 2026-02-10-upstage-practice-langgraph-implementation-practice %}) — `04_Exercises/LangGraph-구현-실습.md`
- [프로젝트 아키텍처 설계 실습]({% post_url 2026-02-09-upstage-practice-architecture-design-practice %}) — `04_Exercises/아키텍처-설계-실습.md`

## 1-2-9.LLMOPS - LLMOps

### 수업 기록형

- [API 이슈 & LiteLLM]({% post_url 2026-03-03-upstage-course-w09d01-api-issues-litellm %}) — `01_Modules/LLMOps-운영/W09D01-API-이슈-LiteLLM.md`
- [상태 관리]({% post_url 2026-03-04-upstage-course-w09d02-state-management %}) — `01_Modules/LLMOps-운영/W09D02-상태관리.md`
- [Observability]({% post_url 2026-03-05-upstage-course-w09d03-observability %}) — `01_Modules/LLMOps-운영/W09D03-Observability.md`
- [Error Analysis]({% post_url 2026-03-06-upstage-course-w09d04-error-analysis %}) — `업스테이지/AI Backend Engineering - LLMOps/09_Error Analysis.pdf`
- [LLMOps 패턴 정리]({% post_url 2026-03-09-upstage-course-w09d05-llmops-patterns %}) — `업스테이지/AI Backend Engineering - LLMOps/10_LLMOps_pattern.pdf`
- [Week 09: LLMOps]({% post_url 2026-03-03-upstage-course-week09-llmops %}) — `01_Modules/LLMOps-운영/Week09-LLMOps.md`

### 기술 노트형

- [AgentOps]({% post_url 2026-02-03-upstage-tech-agentops %}) — `02_Concepts/AgentOps.md`
- [LiteLLM]({% post_url 2026-03-03-upstage-tech-litellm %}) — `02_Concepts/LiteLLM.md`
- [상태관리 (State Management)]({% post_url 2026-03-04-upstage-tech-state-management %}) — `02_Concepts/상태관리.md`
- [Observability (관측 가능성)]({% post_url 2026-03-05-upstage-tech-observability %}) — `02_Concepts/Observability.md`

### 실습형

- [LiteLLM 통합 + Docker 실습]({% post_url 2026-03-03-upstage-practice-litellm-practice %}) — `04_Exercises/LiteLLM-실습.md`
- [배포 자동화 실습]({% post_url 2026-01-19-upstage-practice-deployment-practice %}) — `04_Exercises/배포-실습.md`
- [상태관리 + 비용 추적 실습]({% post_url 2026-03-04-upstage-practice-state-management-practice %}) — `04_Exercises/상태관리-실습.md`
- [Error Analysis 실습]({% post_url 2026-03-06-upstage-practice-error-analysis-practice %}) — `업스테이지/AI Backend Engineering - LLMOps/(EXT) [SeSAC] [LLMOps] daily mission.pdf`

## 1-2-10.PROJECTS - Projects

### 프로젝트 정리형

- [Fitness Buddy - AI 운동 버디 서비스 기획]({% post_url 2026-02-04-upstage-project-fitness-buddy-planning %}) — `03_Projects/fitness-buddy-기획.md`
- [idol-agent v0.2 - LangGraph MVP]({% post_url 2026-02-10-upstage-project-idol-agent-v02 %}) — `03_Projects/idol-agent-v02.md`
- [idol-agent v0.6 - LiteLLM + Docker + CI/CD]({% post_url 2026-03-03-upstage-project-idol-agent-v06 %}) — `03_Projects/idol-agent-v06.md`
- [idol-agent v0.7 - 상태관리 + 비용 추적]({% post_url 2026-03-04-upstage-project-idol-agent-v07 %}) — `03_Projects/idol-agent-v07.md`
- [Solar FastAPI 앱]({% post_url 2026-01-07-upstage-project-llm-api-server %}) — `03_Projects/llm-api-server.md`
- [PPT Workspace - HTML to PPTX 변환]({% post_url 2025-12-22-upstage-project-ppt-workspace %}) — `03_Projects/ppt-workspace.md`

## 1-2-11.RESOURCES - Resources

### 실습형

- [Exercises - 자기주도 실습]({% post_url 2026-05-15-upstage-practice-readme %}) — `exercises/README.md`

### 참고자료형

- [Ask Mode - 질문하기]({% post_url 2026-05-15-upstage-reference-claude-command-ask %}) — `.claude/commands/ask.md`
- [Explore Mode - 개념 탐색]({% post_url 2026-05-15-upstage-reference-claude-command-explore %}) — `.claude/commands/explore.md`
- [Learning Mode - 주차별 순서 학습]({% post_url 2026-05-15-upstage-reference-claude-command-learn %}) — `.claude/commands/learn.md`
- [Exercise Mode - 실습 도전]({% post_url 2026-05-15-upstage-reference-claude-command-mission %}) — `.claude/commands/mission.md`
- [Progress - 학습 진도 확인]({% post_url 2026-05-15-upstage-reference-claude-command-progress %}) — `.claude/commands/progress.md`
- [Review Mode - 복습 퀴즈]({% post_url 2026-05-15-upstage-reference-claude-command-review %}) — `.claude/commands/review.md`
- [Roadmap - 커리큘럼 전체 보기]({% post_url 2026-05-15-upstage-reference-claude-command-roadmap %}) — `.claude/commands/roadmap.md`
- [학습 로드맵]({% post_url 2025-12-22-upstage-reference-00-learning-roadmap %}) — `00_Home/00_학습로드맵.md`
- [Agents Companion Guide]({% post_url 2026-05-15-upstage-reference-agents-companion %}) — `05_Resources/Agents-Companion.md`
- [GPT-2: Language Models are Unsupervised Multitask Learners]({% post_url 2026-01-21-upstage-reference-gpt-2-paper %}) — `05_Resources/GPT-2-논문.md`
- [GPT-3: Language Models are Few-Shot Learners]({% post_url 2026-01-21-upstage-reference-gpt-3-paper %}) — `05_Resources/GPT-3-논문.md`
- [AI Product Engineer - Interactive Learning System]({% post_url 2026-05-15-upstage-reference-claude-learning-system %}) — `CLAUDE.md`

## 원칙

이 카테고리의 목적은 수료 사실 기록이 아니라, AI Agent 서비스를 다시 만들 때 참고할 수 있는 개인 기술 문서를 남기는 것이다.
