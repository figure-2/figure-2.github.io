---
title: "Gradio"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-8.AGENT_ARCHITECTURE
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- streaming
- tech-note
- agent-architecture
toc: true
date: 2026-02-11 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## Gradio

> **한줄 정의**
> Python 기반의 ML 데모 및 인터랙티브 UI 프레임워크.

## 핵심 이해

Gradio는 머신러닝 모델의 데모 UI를 몇 줄의 코드로 구축할 수 있는 프레임워크다. 챗봇 인터페이스, 파일 업로드, 스트리밍 출력 등을 쉽게 구현할 수 있어 LLM 기반 에이전트의 프론트엔드로 활용된다.

## 언제 쓰는지

Gradio는 빠르게 데모를 만들고 사용자 흐름을 확인할 때 유용하다. 모델이나 에이전트 로직을 만들었지만 별도 프론트엔드를 붙이기 전, 입력 폼과 응답 화면을 짧은 코드로 검증할 수 있다.

수업 맥락에서는 스트리밍 챗봇, 파일 업로드 기반 질의응답, 모델 응답 비교 화면을 빠르게 만들어보는 용도에 잘 맞는다. 다만 운영용 제품 화면이라기보다는 프로토타입과 내부 검증 도구에 가깝다.

## 구현 관점

- `gr.ChatInterface`는 챗봇 형태의 입출력을 빠르게 만들 때 사용한다.
- `gr.Blocks`는 여러 입력 컴포넌트와 출력 영역을 조합해야 할 때 사용한다.
- 스트리밍 응답은 generator를 사용해 토큰이 생성되는 대로 화면에 흘려보낸다.
- 에이전트 로직은 UI 코드와 분리하고, UI는 API 또는 함수 호출 계층만 사용하게 두는 편이 유지보수에 좋다.

## 주의점

- 인증, 권한, 배포 안정성이 필요한 서비스라면 FastAPI/프론트엔드 구조를 별도로 고려해야 한다.
- 장시간 실행 작업은 timeout, 중복 요청, 상태 초기화 문제를 함께 확인해야 한다.
- 데모 UI에 API 키나 내부 설정값이 노출되지 않도록 환경변수와 서버 설정을 분리해야 한다.

## 관련 강의
- W08D03-Streaming-구현 - 스트리밍 UI 구현

## 관련 개념
- FastAPI - 백엔드 API 서버
- LangGraph - 에이전트 로직
