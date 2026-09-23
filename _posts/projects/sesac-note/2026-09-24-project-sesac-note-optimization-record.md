---
title: "15. SeSAC:Note OCR·요약·RAG 개선 기록"
excerpt: "OCR-first 처리, VLM 프롬프트, 배치 스트리밍과 RAG 개선 기록입니다."
categories:
- 2.PROJECT
- 2-5. SeSAC-Note
tags:
- SeSAC-Note
- 개발 기록
toc: true
date: 2026-09-24 00:00:00 +0900
comments: true
mermaid: false
math: false
---

{% raw %}

<p><img src="/assets/images/notion-records/sesac-note/optimization-01.png" alt="프로젝트 구성 1"></p>
<h3 id="section-1"><strong>1) 프로젝트 개요</strong></h3>
<ul>
<li>소속: 네이버 커넥트재단</li>
<li>유형: 팀 프로젝트 (5인)</li>
<li>기간: 2025.12 ~ 2026.02 (5주)</li>
</ul>
<h3 id="section-2"><strong>SeSAC:Note​</strong></h3>
<ul>
</ul>
<p>기존의 강의 요약 서비스들은 대부분 음성 데이터만을 활용합니다. 하지만 실제 강의에서 핵심 수식, 복잡한 도표, 실시간 판서 등은 화면을 통해 전달됩니다. 음성만으로 요약할 경우 이러한 시각적 맥락이 완전히 유실되어, 결과물만으로는 강의 내용을 완벽히 이해하기 어려운 한계가 있었습니다. SeSAC:Note는 STT + 화면 정보를 결합해 학습자가 강의 내용을 더 정확하게 복습할 수 있는 요약/노트 경험을 제공하는 것을 목표로 서비스를 개발했습니다.</p>
<h3 id="section-3"><strong>2) 파이프라인</strong></h3>
<p><img src="/assets/images/notion-records/sesac-note/optimization-02.png" alt="프로젝트 구성 2"></p>
<h3 id="section-4"><strong>3) 문제 인식</strong></h3>
<p>이런 요약 서비스는 품질도 중요하지만, 사용자가 결과를 받아보는 시간이 매우 중요하다고 생각했습니다. 그래서 저는 파이프라인 경량화에 가장 큰 우선순위를 두게 되었습니다.</p>
<h3 id="section-5"><strong>4) 프로젝트 기여 </strong></h3>
<ul>
<li>VLM 병목을 해소하기 위해 Layout Detection + OCR 기반의 VLM 사전 필터링을 설계/적용하여 텍스트 화면 처리시간을 7s → 162ms(약 43.2배, 97.7% 감소)로 경량화</li>
<li>VLM 프롬프트 구조화 및 환각 억제 튜닝. 텍스트 추출을 main/aux로 분리하고 레이아웃·강조 정보를 함께 추출하는 방식으로 토큰 사용량 절감 및 요약 노트 품질 및 신뢰도 개선</li>
<li>전체 완료를 기다리지 않도록 배치 단위 처리 + 결과 스트리밍 파이프라인을 구현해 체감 대기시간(UX)을 개선하는 방향으로 구조 전환</li>
<li>챗봇 Q&amp;A에서 전체 요약을 매번 컨텍스트로 주입하지 않도록, 배치별 요약을 임베딩하고 Supabase RPC(SQL Function)로 서버에서 유사도 검색 후 top-k만 retrieve하는 배치 단위 RAG 구성</li>
</ul>
<hr>
<h3 id="section-6"><strong>VLM 사전 필터링(OCR-first)</strong></h3>
<h4 id="section-7"><strong>문제 정의</strong></h4>
<p>캡처 이미지 1장을 VLM이 처리하는데 평균 7초가 소요되어,  영상 길이가 길어질수록 전체 처리시간이 선형적으로 증가했습니다. 30분 영상 기준 캡쳐 장수가 30~60장, VLM 처리에만 3분에서 7분까지 걸리는 문제가 발생했습니다.</p>
<h4 id="section-8"><strong>접근 방법</strong></h4>
<p>이미 VLM 모델 변경, 프롬프트 경량화 등 최적화를 진행한 상태였기 때문에 VLM에 입력되는 양 자체를 줄이자는 전략을 세웠습니다. 텍스트 화면은 OCR로 대체하고 이미지/도표가 있는 화면만 VLM으로 처리하는 파이프라인을 생각했습니다.</p>
<h4 id="section-9"><strong>해결</strong></h4>
<ul>
<li>1차 분기: Paddle Layout Detection<ul>
<li>목적: 텍스트만 있는 화면 vs 이미지가 포함된 화면 판별</li>
<li>처리시간: 53ms/장</li>
</ul>
</li>
<li>텍스트 화면 처리: Paddle OCR<ul>
<li>Text Detection (90ms) + Text Recognition (8ms)</li>
</ul>
</li>
<li>최종 처리시간(텍스트 화면 기준)<ul>
<li>VLM 7초(7000ms) → Layout+OCR 162ms</li>
</ul>
</li>
</ul>
<h4 id="section-10"><strong>결과</strong></h4>
<ul>
<li>텍스트만 있는 슬라이드 처리 시간 7000ms → 162ms로 감소</li>
</ul>
<hr>
<h3 id="section-11"><strong>VLM 프롬프트 튜닝(구조화 + 환각 억제)</strong></h3>
<h4 id="section-12"><strong>문제 정의</strong></h4>
<p>화면 정보를 추출하기 위해 VLM을 사용하면서 다음과 같은 문제가 발생했습니다.</p>
<ul>
<li>노이즈 텍스트로 인한 토큰 낭비<ul>
<li>화면 내 모든 글씨를 그대로 추출하면 페이지 번호, copyright 로고, 소속/워터마크 등이 함께 포함되어 요약 입력 컨텍스트가 불필요하게 커졌습니다. 이런 요소는 제외해라 같은 지시를 프롬프트에 추가해봤지만 성공적으로 제거하지 못했습니다.</li>
</ul>
</li>
<li>OCR과의 차별성 부족<ul>
<li>글씨만 뽑으면 결국 OCR과 유사해져, 화면의 구성(레이아웃), 요소 간 연결, 강조(하이라이트) 같은 화면의 맥락이 손실되었습니다.</li>
</ul>
</li>
<li>환각이 요약 환각으로 직결<ul>
<li>VLM 출력은 요약 노트의 source 데이터이므로, 여기서의 환각(객체명/수량/정확한 표현 오류)이 그대로 요약 노트 환각으로 이어져 UX에 직접적인 악영향이 있었습니다.</li>
</ul>
</li>
</ul>
<h4 id="section-13"><strong>접근 방법</strong></h4>
<ul>
<li>main/aux 분리 전략<ul>
<li>텍스트를 강의 내용에 필요한 본문(main)과 메타/노이즈(aux)로 나누어 둘 다 추출하되, one-shot 예시로 어떤 것이 main에 들어가야 하는지 학습시키는 방식으로 안정성을 확보했습니다.</li>
<li>결과적으로 노이즈를 aux 카테고리로 격리함으로써 요약 LLM에 불필요한 정보가 들어가지 않도록 했습니다.</li>
</ul>
</li>
<li>화면 구조(레이아웃/관계/강조) 추출<ul>
<li>레이아웃 블록/요소 간 연결/하이라이트를 별도 필드로 추출하도록 schema를 설계했습니다.</li>
</ul>
</li>
<li>환각 억제<ul>
<li>불확실한 경우 추정하지 말고 일반화(예: 강아지 → 동물)하거나 unknown 처리, 고유명사/개체명은 확실하지 않으면 치환, 수량/개체 수는 단정 금지 등의 원칙을 프롬프트에 명시했습니다.</li>
</ul>
</li>
</ul>
<h4 id="section-14"><strong>해결</strong></h4>
<ul>
<li>출력 포맷을 구조화(JSON 등)하고, 필드를 다음처럼 분리해 문제를 해결했습니다.<ul>
<li>main_text: 강의 요약에 필요한 핵심 텍스트</li>
<li>aux_text: 페이지 번호/로고/소속/워터마크 등 메타/노이즈 텍스트</li>
<li>layout: 섹션/블록 단위의 배치 정보(제목/본문/캡션/도표 영역 등)</li>
<li>relations: 요소 간 연결(예: 도표 ↔ 설명 문장, 항목 ↔ 하위 항목)</li>
<li>highlights: 강조 표시(색/밑줄/박스 등)와 강조 대상 텍스트</li>
</ul>
</li>
</ul>
<h4 id="section-15"><strong>결과</strong></h4>
<ul>
<li>노이즈 텍스트가 aux로 격리되어 요약 LLM에 전달되는 컨텍스트 정제<ul>
<li>불필요 토큰을 줄이고, 요약 모델이 핵심 내용에 더 집중할 수 있어 노트 품질이 개선되었습니다.</li>
</ul>
</li>
<li>레이아웃/연결/강조 정보까지 함께 전달되며 요약 노트의 품질 상승</li>
<li>요약 노트에서 객체명, 수량 설명의 환각이 감소하여 사용자 UX 개선</li>
</ul>
<h3 id="section-16"><strong>배치 단위 처리 + 실시간 결과 출력으로 체감 대기시간(UX) 개선</strong></h3>
<h4 id="section-17"><strong>문제 정의</strong></h4>
<p>VLM 필터링 도입으로 텍스트 중심의 케이스는 처리 시간이 크게 줄었지만 이미지와 도표 비중이 높은 실제 강의 PPT의 특성상 전체 파이프라인의 평균 처리 시간은 기대만큼 줄어들지 않았습니다. 모델 변경, 병렬 처리, 프롬프트 최적화 등 가용 가능한 모든 수단을 동원했음에도 6분 영상 기준 약 2분 30초의 대기 시간으로 사용자 이탈이 우려되었습니다.</p>
<h4 id="section-18"><strong>접근 방법</strong></h4>
<p>배치 단위로 처리하고 결과를 바로 출력해준다면 사용자가 첫 응답을 받는데까지 걸리는 시간을 줄임으로써 체감 대기시간을 크게 개선할 수 있다고 판단했습니다. 따라서 전체 요약 방식에서 배치 단위 처리 + 스트리밍 방식으로 파이프라인을 변경했습니다.</p>
<h4 id="section-19"><strong>해결</strong></h4>
<ul>
<li>입력을 배치로 분할</li>
<li>배치가 들어오는 즉시 배치 단위로 파이프라인 실행:<ul>
<li>VLM → 요약 → Judge</li>
</ul>
</li>
<li>배치 결과가 생성되는 즉시 사용자에게 전달</li>
</ul>
<h4 id="section-20"><strong>결과</strong></h4>
<ul>
<li>6분 영상 기준 첫 응답까지 지연시간 2분 30초 → 30초로 80% 감소</li>
<li>전체 결과가 나오기 전에 앞부분 결과를 먼저 확인할 수 있어, 사용자의 체감 대기시간을 줄이는 방향으로 UX 개선</li>
</ul>
<hr>
<h3 id="section-21"><strong>배치 단위 RAG로 Q&amp;A 비용/레이턴시 감소</strong></h3>
<h4 id="section-22"><strong>문제 정의</strong></h4>
<p>사용자가 챗봇으로 질문할 때마다 전체 요약을 컨텍스트로 넣으면 API 호출 비용과 응답 레이턴시가 증가한다고 판단했습니다. 또한 전체 요약을 넣으면 답변에 사용해야 할 중요한 내용이 희석될 위험이 있었습니다.</p>
<h4 id="section-23"><strong>접근 방법</strong></h4>
<p>배치별로 요약 결과 임베딩을 만들어두고 질문이 들어오면 관련 배치만 검색(top-k)해서 컨텍스트를 구성하는 방식으로 비용(LLM 토큰, 레이턴시), 적합도 문제를 해결할 수 있다고 생각했습니다.</p>
<h4 id="section-24"><strong>해결</strong></h4>
<ul>
<li>배치별 요약 결과를 Qwen3 Embedding 8B로 임베딩</li>
<li>쿼리도 임베딩한 뒤 유사도 검색으로 관련 배치 top-k를 retrieve</li>
<li>검색 로직을 Supabase SQL Function(RPC)로 정의해 서버 측에서 벡터 유사도 검색이 가능하도록 구성하고, 챗봇이 해당 함수를 tool-call로 호출하도록 연동</li>
</ul>
<h4 id="section-25"><strong>결과</strong></h4>
<ul>
<li>전체 요약을 매번 챗봇에 주입하지 않아도 되어서 비용/레이턴시 감소</li>
<li>질문과 관련된 내용 중심으로 컨텍스트를 구성하여 답변의 적합도 상승</li>
</ul>

{% endraw %}
