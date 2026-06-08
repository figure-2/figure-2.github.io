---
title: "검증 범위, 보안 보강, 그리고 프로젝트 회고"
categories:
- 2.PROJECT
- 2-5. SeSAC-Note
tags:
- Security
- Validation
- AI Service
- Retrospective
- Claim Boundary
toc: true
date: 2026-02-10 10:10:00 +0900
comments: true
mermaid: true
math: true
---

# 검증 범위, 보안 보강, 그리고 프로젝트 회고

SeSAC:Note를 공개 프로젝트 글로 정리할 때 가장 중요한 것은 "무엇을 만들었는가"만이 아니었다. "무엇을 어디까지 검증했는가"를 같은 비중으로 적어야 했다.

원본 `Re:View`는 팀 프로젝트였고, `SeSAC:Note`는 이를 공개용으로 정리한 이름이다. 일부 보안 보강과 테스트 기록은 SeSAC:Note clone 기준이다. 이 기준을 섞으면 실제보다 넓은 claim처럼 보일 수 있다.

```mermaid
flowchart TB
    A[프로젝트 공개 정리] --> B[무엇을 만들었는가]
    A --> C[어디까지 검증했는가]
    B --> D[멀티모달 노트<br/>영상 근거 QA]
    C --> E[문서상 sample]
    C --> F[제한된 benchmark]
    C --> G[clone 기준 테스트]
    E --> H[claim boundary]
    F --> H
    G --> H
```

## 검증을 기준별로 나누는 이유

검증에는 종류가 있다. 문서상 sample 실행 기록, Judge benchmark, 보안 단위 테스트, frontend build는 서로 다른 의미를 가진다.

| 검증 기준 | 의미 |
| --- | --- |
| 문서상 sample pipeline | 특정 sample 기준으로 파이프라인 결과가 기록됨 |
| 제한된 Judge benchmark | Judge prompt 버전별 평가 시간, 토큰, 통과 여부 비교 |
| SeSAC:Note clone 보안 테스트 | clone 기준 일부 보안 흐름의 부정 경로 확인 |
| frontend build | 프론트엔드 빌드 가능성 확인 |

이것들을 하나로 묶어 "전체 서비스가 검증됐다"고 말하면 안 된다. 각 검증은 범위와 조건이 다르다.

## 문서상 sample pipeline 검증

프로젝트 기록에는 sample4 기준 pipeline 결과가 남아 있다.

| 항목 | 문서상 sample4 기준 기록 |
| --- | --- |
| Video 상태 | DONE |
| STT 결과 | 42개 |
| Batch | 2/2 |
| Segment | 8개 |
| Judge Scores | 8.26 / 8.96 |

이 결과는 pipeline이 어떤 식으로 결과를 남기는지 보여주는 근거로 사용할 수 있다. 다만 현재 시점에 전체 cloud 환경에서 같은 과정을 다시 실행했다는 의미는 아니다. 따라서 표현은 "문서상 sample4 기준으로 기록됐다"로 제한한다.

## Judge benchmark 검증

Judge benchmark는 요약 품질 점검 prompt를 비교하기 위한 자료다. 제한된 benchmark 기준으로 Judge v3는 평균 평가 시간 14.7초, 평균 토큰 14,734 tokens, 통과율 100%로 기록되어 있다. 이때 100%는 benchmark 조건에서 5개 설정이 모두 통과했다는 뜻이다.

또한 v1 대비 Judge 평가 단계 시간이 53.5% 줄고, token 사용량이 10.0% 절감된 것으로 정리되어 있다. 이 수치는 전체 서비스 처리 시간 개선이 아니라 Judge 평가 단계의 비교 결과로만 해석한다.

## SeSAC:Note clone 기준 보안 테스트

보안 보강은 원본 전체에 소급해서 말하지 않는다. SeSAC:Note clone 기준으로 media ticket과 upload security 관련 테스트가 12 passed로 기록되어 있다.

| 보안 흐름 | 목적 |
| --- | --- |
| media ticket scope | video_id와 purpose를 제한해 접근 범위를 좁힘 |
| upload validation | 확장자, MIME, 크기, 빈 파일 같은 부정 경로 확인 |
| storage object check | 업로드 완료 시 실제 object 상태 확인 |
| owner check | 사용자별 접근 범위 확인 |

여기서도 표현은 신중해야 한다. "보안이 완성됐다"가 아니라 "노출 시 피해 범위를 줄이도록 scope와 검증을 보강했다"가 맞다.

## frontend build와 재실행 보류

SeSAC:Note clone 기준 frontend build 성공 기록은 있다. 하지만 build 성공은 배포 성공이나 전체 서비스 검증과 다르다. 또한 공개 글 작성 시점에서 동일 cloud 환경의 upload, process, summary, chat 전체 구간을 재실행했다고 주장하지 않는다.

검증 범위는 다음처럼 정리한다.

| 말할 수 있는 것 | 말하지 않는 것 |
| --- | --- |
| sample pipeline 기록이 있다 | 모든 영상에서 동일하게 동작한다 |
| Judge benchmark 기록이 있다 | Judge 결과를 품질 보증으로 해석한다 |
| clone 기준 보안 테스트 12 passed | 원본 전체 보안 범위로 확대 해석 |
| frontend build 기록이 있다 | 배포 환경 검증이 끝났다 |

## 회고: AI 기능보다 연결 구조가 중요했다

이 프로젝트에서 가장 크게 배운 점은 AI 서비스가 모델 호출만으로 완성되지 않는다는 것이다.

STT, VLM, Summarizer, Judge는 각각 중요하다. 하지만 실제 서비스에서는 그 사이의 연결이 더 중요해진다. 화면과 음성을 어떻게 같은 segment로 묶을지, 중간 결과를 어디에 저장할지, 사용자가 긴 처리 상태를 어떻게 볼지, 질문이 영상 근거를 벗어나지 않게 할지, 보안 claim을 어디까지 말할지 모두 설계 대상이었다.

특히 멀티모달 AI 서비스에서는 앞단 입력 품질이 뒤쪽 생성 품질을 좌우한다. 캡처 중복이 줄어야 VLM 비용이 줄고, VLM 출력이 구조화되어야 Summarizer가 안정적으로 쓰며, segment가 정리되어야 Judge와 QA도 근거를 따라갈 수 있다.

## 전체 시리즈

```mermaid
flowchart LR
    A[Sample 기록] --> D[문서상 검증]
    B[Judge benchmark] --> E[평가 단계 기준]
    C[보안 테스트] --> F[clone 기준]
    D --> G[공개 가능한 claim]
    E --> G
    F --> G
```

이번 시리즈는 SeSAC:Note를 하나의 긴 개발 흐름으로 정리했다.

1. [SeSAC:Note - 멀티모달 AI 강의 노트 서비스 개발기]({% post_url 2026-02-10-project-sesac-note-case-study %})
2. [SeSAC:Note Visual Portfolio - 6장으로 보는 프로젝트 요약]({% post_url 2026-02-10-project-sesac-note-visual-portfolio %})
3. [강의 영상 요약 문제를 어떻게 재정의했는가]({% post_url 2026-02-10-project-sesac-note-problem-definition %})
4. [STT, VLM, Fusion으로 강의 영상을 노트로 바꾸는 구조]({% post_url 2026-02-10-project-sesac-note-architecture %})
5. [슬라이드 중복 캡처와 VLM 입력 품질을 줄이는 과정]({% post_url 2026-02-10-project-sesac-note-capture-vlm %})
6. [긴 영상 처리에서 상태 추적과 체감 대기시간을 다룬 방법]({% post_url 2026-02-10-project-sesac-note-async-pipeline %})
7. [단순 챗봇이 아니라 영상 근거 안에서 답하는 QA 흐름 만들기]({% post_url 2026-02-10-project-sesac-note-evidence-qa %})
8. [LLM Judge로 요약 품질을 점검할 때 조심해야 할 것]({% post_url 2026-02-10-project-sesac-note-judge-evaluation %})
9. [검증 범위, 보안 보강, 그리고 프로젝트 회고]({% post_url 2026-02-10-project-sesac-note-validation-retrospective %})

- 이전 글: [LLM Judge로 요약 품질을 점검할 때 조심해야 할 것]({% post_url 2026-02-10-project-sesac-note-judge-evaluation %})
