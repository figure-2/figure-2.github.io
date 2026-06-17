---
title: "HTTP"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-4.NETWORK_CLOUD
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- http
- tech-note
- network-cloud
toc: true
date: 2026-01-14 09:00:00 +0900
comments: false
mermaid: true
math: true
---
## HTTP

> **한줄 정의**
> HyperText Transfer Protocol — 웹에서 클라이언트와 서버 간 데이터를 주고받는 통신 프로토콜.

## 핵심 이해

HTTP는 **요청(Request)**과 **응답(Response)** 구조로 동작한다. 요청은 메서드, URL, 헤더, 바디로 구성되고, 응답은 상태코드, 헤더, 바디로 구성된다. REST API는 HTTP 메서드를 의미에 맞게 사용하는 설계 원칙이다.

주요 메서드: **GET**(조회), **POST**(생성), **PUT**(전체 수정), **PATCH**(부분 수정), **DELETE**(삭제). 상태코드: 2xx(성공), 3xx(리다이렉션), 4xx(클라이언트 오류: 400 Bad Request, 401 Unauthorized, 404 Not Found), 5xx(서버 오류). HTTP/2는 멀티플렉싱으로 성능을 개선하고, HTTPS는 TLS로 암호화한다.

## 요청-응답 흐름

![HTTP 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-http-diagram-1.svg)

## 관련 개념

- FastAPI - HTTP 기반 REST API 구현
- 클라우드-컴퓨팅 - HTTP 서버 인프라
- CI-CD - HTTP API 자동 테스트

## 참고 자료

- [MDN HTTP Guide](https://developer.mozilla.org/ko/docs/Web/HTTP)
