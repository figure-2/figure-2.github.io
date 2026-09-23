---
title: "14. SeSAC:Note 서비스 개발 기록 — 파이프라인부터 운영까지"
excerpt: "SeSAC:Note의 STT, 화면 분석, 요약, AI 튜터와 운영 개선 기록입니다."
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

<p><strong>프로젝트 전체 기간 (6주) : 2025년 12월 30일 (화) 10:00 ~ 2026년 2월 11일 (수) 19:00</strong></p>
<h2 id="section-1"><strong><strong>1. 프로젝트 개요 (Introduction)</strong></strong></h2>
<h3 id="section-2"><strong><strong>1.1 배경 및 목적</strong></strong></h3>
<ul>
<li>강의 영상 학습은 탐색이 어렵고, 필기 부담이 크며, 결과적으로 학습 효율이 떨어지는 문제가 있음.</li>
<li>본 프로젝트는 이 문제를 해결하기 위해, 영상 없이도 학습이 가능한 독립형 강의 노트를 자동 생성하고, 이를 기반으로 한 맞춤형 챗봇 튜터 구현을 목표로 함.</li>
</ul>
<h3 id="section-3"><strong><strong>1.2 주요 기능</strong></strong></h3>
<ul>
<li>스마트 슬라이드 캡처: 중복 제거 및 전환 감지 기반 고품질 슬라이드 추출</li>
<li>멀티모달 분석: 음성(STT) + 텍스트/수식/도표(VLM) 통합</li>
<li>자동 구조화 요약 및 품질 검증: Judge 피드백 기반 재생성 파이프라인</li>
<li>강의 노트를 기반으로 한 챗봇 학습 도우미</li>
</ul>
<h2 id="section-4"><strong><strong>2. 시스템 아키텍처 (System Architecture)</strong></strong></h2>
<h3 id="section-5"><strong><strong>2.1 전체 구조도</strong></strong></h3>
<p><img src="/assets/images/notion-records/sesac-note/development-01.png" alt="서비스 개발 도식 1"></p>
<h3 id="section-6"><strong><strong>2.2 기술 스택</strong></strong></h3>
<ul>
<li><strong>Tech Stack</strong><ul>
<li><strong>Language</strong>: Python 3.10+</li>
<li><strong>Frontend</strong>: Streamlit, React (Vite)</li>
<li><strong>Backend</strong>: FastAPI, Supabase (PostgreSQL)</li>
</ul>
</li>
<li><strong>Frameworks &amp; Libraries</strong><ul>
<li><strong>Ochestration</strong>: LangGraph, ADK</li>
<li><strong>AI SDK</strong>: OpenAI SDK, Google GenAI SDK</li>
<li><strong>Capture</strong>: OpenCV, NumPy, FFmpeg</li>
</ul>
</li>
<li><strong>AI Models (API)</strong><ul>
<li><strong>VLM</strong>: Qwen3-VL-32B-Instruct (Alibaba Cloud)</li>
<li><strong>LLM (Judge/Summary)</strong>: Gemini 3 Flash</li>
<li>LLM (Chatbot): Gemini 3 Flash</li>
<li><strong>STT</strong>: Naver Clova Speech (Main), OpenAI Whisper (Fallback)</li>
</ul>
</li>
<li><strong>Library</strong><ul>
<li>FastAPI: 0.123.10 (ASGI Server: uvicorn 0.40.0)</li>
<li>React: 19.x (Frontend via Vite)</li>
<li>LangGraph: 0.3.5+ (Agent Workflow Orchestration)</li>
<li>OpenAI SDK: 2.14.0 (for OpenRouter/Qwen)</li>
<li>Google GenAI SDK: 1.56.0 (for Gemini Judge/Summarizer)</li>
<li>OpenCV (opencv-python): 4.12.0.88</li>
<li>MediaPipe: 0.10.31</li>
<li>NumPy: 2.2.6</li>
<li>OpenAI Whisper: 20250625 (Local STT Engine) </li>
<li>Supabase: 2.0.0+ (PostgreSQL Client)</li>
<li>Pydantic: 2.11.7 (Data Validation)</li>
</ul>
</li>
</ul>
<h2 id="section-7"><strong><strong>3. 핵심 파이프라인 (Core Pipeline)</strong></strong></h2>
<p><img src="/assets/images/notion-records/sesac-note/development-02.png" alt="서비스 개발 도식 2"></p>
<h3 id="section-8"><strong><strong>3.1 Preprocess Stage (데이터 추출)</strong></strong></h3>
<ul>
<li><strong>STT (Speech-to-Text)</strong>: 오디오 추출 및 텍스트 변환, 타임스탬프 동기화</li>
<li><strong>Slide Capture</strong>: 슬라이드 추출 및 변화 감지 기반 구간 분할</li>
</ul>
<h3 id="section-9"><strong><strong>3.2 Main Process Stage (분석 및 생성)</strong></strong></h3>
<ul>
<li><strong>VLM Analysis</strong>: 슬라이드 내 텍스트, 수식, 도표 등 시각 정보 추출</li>
<li><strong>Fusion Engine</strong>: 오디오(STT)와 시각 정보(VLM)의 타임라인 동기화 및 세그먼트 매칭</li>
<li><strong>Summarization</strong>: LLM 기반의 섹션별 구조화된 요약 생성</li>
<li><strong>Judge &amp; Feedback</strong>: 요약 품질 평가 및 점수 기반 재생성 로직</li>
</ul>
<h2 id="section-10"><strong><strong>4. 기능별 상세 구현 (Implementation Details)</strong></strong></h2>
<h3 id="section-11"><strong><strong>4.1 Audio Processing (</strong><code>src/audio</code><strong>)</strong></strong></h3>
<p><strong>Audio Engine: </strong><code>clova_stt.py</code><strong> &amp; </strong><code>whisper_stt.py</code></p>
<ul>
<li><strong>Clova Speech Client</strong>: 한국어 전문용어 및 긴 문장 인식률이 뛰어난 Naver Clova Speech API를 메인 엔진으로 사용</li>
<li><strong>Whisper Client</strong>: 비용 절감 및 로컬 테스트, 다국어 지원을 위해 OpenAI Whisper 모델(base/small 등)을 서브/백업 엔진으로 구현</li>
</ul>
<p><strong>Output Schema</strong>:</p>
<pre><code>{
&quot;segments&quot;: [
{
&quot;id&quot;: &quot;stt_001&quot;,
&quot;start_ms&quot;: 0,
&quot;end_ms&quot;: 4500,
&quot;text&quot;: &quot;안녕하세요, 이번 강의에서는 변분 추론에 대해 알아보겠습니다.&quot;,
&quot;confidence&quot;: 0.985
},
...
],
&quot;confidence&quot;: 0.988,
&quot;raw_response&quot;: { ... } // (Optional) Clova/Whisper 원본 응답
}
</code></pre>
<div class="table-wrapper"><table>
<tr><th><strong>지표</strong></th><th><strong>Clova</strong></th><th><strong>Whisper</strong></th><th><strong>승자</strong></th></tr>
<tr><td><strong>WER (단어 오류율)</strong></td><td>6.2%</td><td>45.4%</td><td>Clova (<strong>7배 좋음</strong>)</td></tr>
<tr><td><strong>CER (글자 오류율)</strong></td><td>4.0%</td><td>26.7%</td><td>Clova (<strong>7배 좋음</strong>)</td></tr>
<tr><td><strong>속도 (latency)</strong></td><td>10초</td><td>16초</td><td>Clova (<strong>1.6배 빠름</strong>)</td></tr>
</table></div>
<p><strong>Audio Routing </strong><code>stt_router.py</code></p>
<ul>
<li>설정 파일 기반으 코드 수정 없이 STT 엔진을 즉시 교체할 수 있는 <strong>Router Pattern</strong> 적용</li>
<li>오디오 추출부터 설정 파일을 통해서 결정된 backend(Clova/Whispeer)로 STT추출을 진행까지의 과정을 하나의 라우팅 레이어에서 관리</li>
<li>오디오 추출 로직을 분리하여, 오디오 파일만으로도 독립 실행 가능한 형태로 구성</li>
</ul>
<p><strong>Smart Audio Extraction (</strong><code>extract_audio.py</code><strong>)</strong>:</p>
<ul>
<li><strong>Phase Cancellation Fix</strong>: 스테레오 오디오의 역상 상쇄 문제를 해결하기 위해 <code>volumedetect</code> 기반으로 Left/Right/Downmix/Phase-fix 중 최적의 모노 변환 방식을 자동으로 선택</li>
<li><strong>Multi-Format Support</strong>: 스토리지 용량 최적화 및 다양한 업로드 요건 충족을 위해 WAV, FLAC 뿐만 아니라 <strong>MP3 (128k)</strong> 인코딩을 기본 지원</li>
</ul>
<p><strong>최종 정리</strong></p>
<ol>
<li><strong>유연성 (Flexibility)</strong>: Router 도입으로 Clova(고성능)와 Whisper(무료/로컬)를 자유롭게 오가며 운영 가능하며, 설정 파일을 통해 손쉽게 제어 가능</li>
<li><strong>안정성 (Stability)</strong>: 역상 문제 자동 해결 로직(<code>auto-mono</code>)과 표준화된 출력 스키마를 통해 예측 가능한 파이프라인 구축</li>
<li><strong>효율성 (Efficiency)</strong>: MP3 압축을 통한 업로드 용량 절감 및 처리 속도 최적화</li>
</ol>
<p><strong>시도해 본 것들</strong></p>
<ul>
<li><strong>Whisper 모델 최적화</strong>: V100 환경에서 Whisper Base 모델 서빙을 시도했으나, 한국어 전문용어 인식률 대비 속도(RTF)가 Clova API에 비해 현저히 떨어져 메인 서비스에는 Clova를 채택함 (벤치마크: Clova WER 6% vs Whisper 45%)</li>
<li><strong>Faster-Whisper</strong>: 속도 개선을 위해 CTranslate2 기반의 faster-whisper 도입을 검토했으나, 여전히 클라우드 API의 편의성과 정확도를 넘어서지 못해 보류</li>
</ul>
<h3 id="section-12"><strong><strong>4.2 Smart Capture (</strong><code>src/capture</code><strong>)</strong></strong></h3>
<p><strong>Slide Extraction Engine: (</strong><code>hybrid_extractor.py</code><strong> ):</strong></p>
<ul>
<li><strong>배경 정보 차단</strong> : 대부분의 픽셀을 차지하는 배경을 배제하기 위해, 모서리 등의 유의미한 시각적 특징만 추출하여 실제 정보에 집중</li>
<li><strong>유효 정보 탐지</strong>: 추출된 특징점이 <code>persistence_threshold</code>(10회, 총 5초) 이상 동일 좌표에 고정될 때만 유효 특징점으로 확정하여, 일시적인 움직임과 정적인 슬라이드 및 글자 정보를 정밀하게 구분</li>
</ul>
<p><img src="/assets/images/notion-records/sesac-note/development-03.png" alt="서비스 개발 도식 3"></p>
<p>(좌측 : orb로 추출한 특이점, 우측 : 정적으로 유지되는 특이점)</p>
<ul>
<li><strong>슬라이드 분할</strong>: 유효 특징점의 개수가 급격히 감소하는 지점을 슬라이드 전환의 임계점으로 판단. 이를 통해 화면 일부만 변하는 환경에서도 명확한 분할 경계를 식별<ul>
<li>유효 특징점 개수가 <code>min_orb_features</code>(기본 50개)를 넘어서는 순간 캡처 후보로 등록.</li>
<li>유효 특징점 개수가 기준치 대비 <code>persistence_drop_ratio</code>(기본 70%) 이상 급감하면 슬라이드 종료로 간주하고 저장 프로세스를 실행.</li>
</ul>
</li>
</ul>
<p><img src="/assets/images/notion-records/sesac-note/development-04.png" alt="서비스 개발 도식 4"></p>
<ul>
<li><strong>최적 프레임 선별</strong>: 슬라이드 구간 내에서 특징점 수가 정점에 도달하는 순간을 포착하여, 글자가 가장 많이 채워진 프레임을 추출.<ul>
<li>같은 슬라이드로 판단되는 동안, 더 선명/정보량이 많은(특징점이 더 많은) 프레임이 들어오면 후보 이미지를 교체.</li>
</ul>
</li>
<li><strong>중복 제어</strong> : 재등장하는 슬라이드는 하나의 ID로 그룹화하여 스토리지 및 후속 VLM 분석 비용을 최적화.<ol>
<li>저장 대상 프레임의 pHash와 ORB를 계산.</li>
<li>저장된 슬라이드 히스토리(<code>saved_slides_history</code>)에서 유사 슬라이드를 검색.<ul>
<li>1차 필터: pHash 거리가 <code>dedup_phash_threshold</code>(기본 12) 이하 후보 검색.</li>
<li>2차 정밀: ORB 매칭 수행. 유사도가 <code>dedup_sim_threshold</code>(기본 70%) 이상이면 중복 확정.</li>
</ul>
</li>
<li>중복 발견 시 파일 저장을 생략하고, 기존 슬라이드 ID에 현재 시간 구간(<code>time_ranges</code>)만 병합. 신규 슬라이드는 파일 저장 및 새로운 ID 부여.</li>
</ol>
</li>
<li><strong>알고리즘 성능 비교</strong>: 가장 score가 낮게 나온 영상 기준으로 비교</li>
</ul>
<div class="table-wrapper"><table>
<tr><th>알고리즘 방식</th><th>처리시간</th><th>오탐</th><th>미탐</th><th>F1-score</th><th>비고</th></tr>
<tr><td>Pixel Differency
(단순 픽셀 변화 기준)</td><td>324s</td><td>13</td><td>17</td><td>0.77</td><td>처리시간이 길고, 미탐이 많아 신뢰도가 낮음</td></tr>
<tr><td>ORB Change
(특이점 변화 기준)</td><td>156s</td><td>17</td><td>3</td><td>0.86</td><td>가장 치명적이었던 미탐 문제가 해결
움직임 등에 민감해져 오탐 증가</td></tr>
<tr><td>ORB Drop
(고정 특이점 변화 기준)</td><td>126s</td><td>4</td><td>2</td><td>0.95</td><td>움직이는 물체에 대한 강건성이 확보</td></tr>
<tr><td>최종 알고리즘
(중복제거 적용)</td><td>140s</td><td>2</td><td>2</td><td>0.97</td><td>최종 상용화 모델 선정</td></tr>
</table></div>
<p>Manifest Output (JSON)</p>
<pre><code>[
{
&quot;id&quot;: &quot;cap_001&quot;,
&quot;file_name&quot;: &quot;Video_001.jpg&quot;,
&quot;time_ranges&quot;: [
{ &quot;start_ms&quot;: 0, &quot;end_ms&quot;: 15000 },
{ &quot;start_ms&quot;: 45000, &quot;end_ms&quot;: 60000 }
]
}
]
</code></pre>
<p><strong>최종 정리</strong></p>
<ol>
<li><strong>특징점 기반 고속 연산</strong>: 640x360 (230,400개의 픽셀 데이터) 대신 2,000개 이내의 특징점 좌표 데이터만을 연산 대상으로 삼아 CPU 부하를 대폭 낮추고, 전체 분석 속도를 극대화.</li>
<li><strong>최적 정보량 프레임 포착</strong>: 슬라이드 유지 구간 내에서 유효한 특징점 수가 최대치에 도달하는 지점을 추적하여, 정보가 가장 많이 채워진 완성본 이미지를 캡쳐.</li>
<li><strong>중복 제거</strong>:2 stage(pHash, orb매칭)파이프라인, 중복 이미지 저장을 차단하고 데이터를 구조화.</li>
</ol>
<h3 id="section-13"><strong><strong>4.3 Visual Language Understanding (</strong><code>src/vlm</code><strong>)</strong></strong></h3>
<p><strong>VLM Engine: </strong><code>vlm_engine.py</code><strong> </strong></p>
<ul>
<li>Alibaba Cloud의 API를 통해 VLM 모델 qwen/qwen3-vl-32b-instruct를 사용해 이미지 분석을 수행</li>
<li>슬라이드 이미지에서 텍스트/수식/도표 정보를 추출하고, STT와 결합하여 요약 입력 데이터로 활용</li>
<li>여러 개의 API key를 환경 변수에 등록하고, 하나의 key가 실패하거나 한도에 도달하면 자동으로 다음 key로 전환되는 Round-robin 로직을 구현</li>
</ul>
<p>속도 향상을 위해 여러 이미지를 하나의 prompt context로 묶어 처리하는 <strong>Batch Mode</strong>를 지원</p>
<pre><code>def _build_batch_user_prompt(self, image_count: int) -&gt; str:
&quot;&quot;&quot;배치 처리용 사용자 프롬프트를 구성한다.&quot;&quot;&quot;
return (
&quot;여러 이미지를 순서대로 제공한다. &quot;
f&quot;이미지는 총 {image_count}장이다. &quot;
&quot;각 이미지 결과를 `## Image N` 제목으로 구분해 작성하라. &quot;
&quot;제목은 반드시 `## Image 1`, `## Image 2`처럼 숫자를 붙여 순서대로 출력하고 &quot;
&quot;이미지 수만큼 섹션을 만들어라.\n&quot;
f&quot;{self.user_prompt}&quot;
)
</code></pre>
<ul>
<li>settings.yaml, prompts.yaml을 이용하여 vlm_engine.py의 프롬프트와 설정을 제어할 수 있도록 구현</li>
</ul>
<p><strong>Prompt Engineering: </strong><code>vlm_engine.py</code><strong> </strong></p>
<ul>
<li>슬라이드 분석 정확도를 높이고 환각을 줄이기 위해 프롬프트를 여러 버전으로 실험하며 개선</li>
<li>모델 역할을 Slide OCR + Visual Evidence Extractor로 정의하여 해석/요약이 아닌 사실 추출에 집중하도록 유도</li>
<li>출력 format을 Markdown으로 제한하고 코드 블록 사용을 금지하여 후처리 시에 생기는 오류들을 방지</li>
<li>정보를 다음과 같이 분리하여 출력하도록 설계<ul>
<li>Main Text: 제목/본문/수식 등 핵심 콘텐츠</li>
<li>Auxiliary Text: 페이지 번호/저작권 문구 등 상대적으로 불필요한 정보</li>
<li>Visual Evidence: 레이아웃, 화살표 연결, 하이라이트 등 시각 증거</li>
</ul>
</li>
<li>번역을 금지하여 OCR은 원문 유지, 시각 묘사는 한국어로 작성하도록 제약</li>
<li>불확실한 객체는 단정 짓지 않고 ‘~로 보임’과 같은 표현을 쓰거나 Candidate로 표기하도록 지시해 환각을 방지</li>
<li>슬라이드에 없는 내용을 추가하지 않도록 구체적 금지 조항을 포함</li>
</ul>
<p><strong>VLM Fusion: </strong><code>vlm_fusion.py</code></p>
<ul>
<li>VLM은 캡쳐 이미지 단위로 수행되므로, 시간 정보를 부여하여 STT와의 결합이 필요</li>
<li>vlm_raw.json(VLM 출력)과 manifest.json(캡쳐 메타데이터)을 결합해 타임스탬프가 포함된 vlm.json을 생성</li>
</ul>
<p><strong>vlm.json</strong></p>
<pre><code>&quot;items&quot;: [
{
&quot;extracted_text&quot;: &quot;Main Text\ ...&quot;,
&quot;id&quot;: &quot;cap_001&quot;,
&quot;timestamp_ms&quot;: 33
},
    ...
]
</code></pre>
<ul>
<li>최종 VLM 결과가 캡쳐 정보와 함께 타임라인 순서대로 정렬되도록 구성</li>
</ul>
<p><strong>최종 정리</strong></p>
<ol>
<li><strong>안정성</strong>: 다중 key 관리 및 에러 핸들링으로 외부 API 의존성 리스크를 최소화</li>
<li><strong>성능</strong>: Batch + 병렬 요청으로 높은 품질의 VLM 응답을 빠르게 수집하도록 최적화</li>
<li><strong>확장성</strong>: 설정 파일 분리로 모델/프롬프트 변경 시 코드 수정 없이 대응 가능</li>
</ol>
<p><strong>시도해 본 것들</strong></p>
<ul>
<li>temperature가 1.0으로 높아 토큰 붕괴 현상이 있었고, temperature를 낮춰 안정성을 개선. (Issue #40)</li>
<li>PaddleOCR는 GPU 서빙이 필수인데 서버 V100의 환경 제약으로 제외함</li>
<li>GroundingDINO를 활용해 그림/그래프 위치를 찾아 전달하는 방안을 실험했으나, 모델 수 증가 대비 효과가 제한적이라 중단</li>
<li>OpenRouter 사용 시 API 병목이 있어 Alibaba Cloud API로 전환을 선택</li>
</ul>
<h3 id="section-14"><strong><strong>4.4 Context Fusion &amp; Generation (</strong><code>src/fusion</code><strong>)</strong></strong></h3>
<p>개별적으로 추출된 STT(청각)와 VLM(시각) 데이터를 시간 축으로 동기화하고, LLM을 통해 맥락을 보존하며 요약을 생성하는 단계</p>
<p><code>sync_engine.py</code><strong>: 멀티모달(Audio, Image) 간 정보 병합 로직</strong></p>
<ul>
<li>화면 전환(슬라이드 변경/장면 전환)을 주제 변화의 강한 신호로 가정하고, VLM 타임스탬프를 1차 세그먼트 경계로 사용</li>
<li>1차 세그먼트가 설정된 길이 또는 글자 수를 초과하면 강제 절단하지 않고 STT의 침묵 구간을 찾아 분할. 문장 중간 절단을 방지하고 의미 보존을 우선</li>
<li>서로 다른 주기의 STT(텍스트)와 VLM(이미지) 데이터를 공통 구조인 <strong>Unit</strong>(<code>segments_units.jsonl</code>)로 병합합니다.</li>
<li>모든 데이터에 고유 ID(stt_01, cap_01)를 부여하여 Summarizer 단계에서 근거 인용이 가능하도록 구성</li>
</ul>
<p><code>summarizer.py</code><strong>: 요약 생성 및 검증 로직</strong></p>
<ul>
<li><strong>배치 처리 및 맥락 전달: </strong>긴 영상을 배치 단위로 병렬/순차 처리. 직전 배치 요약을 다음 배치 프롬프트의 System Context로 주입하여 정의/흐름의 일관성을 유지.</li>
<li><strong>근거 기반 요약 생성</strong>: 요약 항목마다 source_type(Direct/Inferred)과 evidence_refs(STT/VLM ID)를 명시하도록 강제하여, 문장 근거를 역추적 가능하게 구현</li>
<li><strong>Judge 피드백 루프</strong>: Judge 평가 점수가 최소 점수(예: 7.0) 미만이면 재생성을 수행. 재시도 프롬프트에 이전 시도 피드백을 포함해 동일 오류 재발을 방지.</li>
<li><strong>스키마 검증</strong>: LLM이 생성한 JSON 형식을 파싱하고 정해진 스키마로 출력되었는지 검증. 형식이 깨지거나 누락되면 재생성 프롬프트로 원인 피드백 후 재생성을 수행.</li>
</ul>
<p><strong>최종 정리</strong></p>
<ul>
<li><strong>데이터 정렬</strong>: 시각적 변화(VLM)와 음성 흐름(STT)을 결합하되, 문맥이 끊기지 않도록 침묵 구간을 활용해 의미 단위(Unit)로 동기화</li>
<li><strong>요약 Context 보존</strong>: 긴 영상도 배치 처리 시 이전 맥락을 주입하여 흐름을 유지하며, 모든 요약 문장에 근거(ID)를 태깅해 신뢰성 확보</li>
<li><strong>결과 검증 루프</strong>: Judge 모델의 평가와 스키마 검증을 통해 기준 미달인 결과물은 피드백을 반영해 즉시 재생성하는 자가 교정 프로세스</li>
</ul>
<h3 id="section-15"><strong><strong>4.5 Quality Assurance (</strong><code>src/judge</code><strong>)</strong></strong></h3>
<p>Judge 모듈은 생성된 요약 콘텐츠의 품질을 보증하는 단계</p>
<ul>
<li>입력 : 생성된 요약본 <code>segment_summaries.jsonl</code>+ 원본 STT/이미지 싱크 데이터 <code>segments_units.jsonl</code></li>
<li>평가 : 정해진 항목으로 평가</li>
<li>산출 : 가중치가 적용된 종합 점수와 개선 사항이 담긴 한국어 한 줄 피드백.</li>
</ul>
<p><code>src/judge/judge.py</code><strong>: 요약 결과 평가</strong></p>
<ul>
<li>생성된 요약 콘텐츠가 신뢰 가능(groundedness, multimodal_use)하고, 규칙 준수(compliance) 및 교육적 가치(note_quality)가 있는지 평가</li>
<li>처리 로직<ol>
<li>요약본과 원본 근거 데이터를 로드하고 ID 매칭 수행</li>
<li>segments_units를 batch_size 단위로 분할</li>
<li>Gemini 모델로 평가 요청(병렬 처리 지원)</li>
<li>LLM이 반환한 3가지 항목의 점수를 가중치를 적용해 계산 후 정규화를 통해 최종 점수를 계산</li>
<li>최종 리포트 및 세그먼트별 상세 리포트를 JSON으로 저장</li>
</ol>
</li>
</ul>
<p><code>config/judge/prompts.yaml</code><strong>: 평가 기준 프롬프트</strong></p>
<ul>
<li>목적: 토큰 소모를 최소화하면서 핵심만 빠르게 평가.</li>
<li>구조 :<ol>
<li>System : &quot;엄격한 AI 평가자&quot; 페르소나 부여.</li>
<li>Criteria (채점 기준표) :</li>
</ol>
<div class="table-wrapper"><table>
<tr><th><strong>항목</strong></th><th><strong>가중치</strong></th><th><strong>설명</strong></th></tr>
<tr><td><strong>groundedness</strong></td><td>45%</td><td>주장이 근거에 의해 완벽히 지지되는가? (Hallucination 방지)</td></tr>
<tr><td><strong>note_quality</strong></td><td>35%</td><td>영상 없이도 이해 가능한 독립적인 학습 노트인가?</td></tr>
<tr><td><strong>compliance</strong></td><td>20%</td><td>JSON 스키마 및 금지어 규칙을준수했는가?</td></tr>
<tr><td><strong>multimodal_use</strong></td><td>참고</td><td>시각 정보(cap_ids)를 적절히 활용했는가?</td></tr>
</table></div>
<ol>
<li>Protocol: Validation Report 확인 -&gt; 의미론적 근거 검증 -&gt; JSON 출력 순서 강제.</li>
</ol>
</li>
</ul>
<p><strong>최종 정리</strong></p>
<pre><code>{
&quot;pass&quot;: true, // 최종 통과 여부 (boolean)
&quot;final_score&quot;: 10.0, // 최종 점수 (10점 만점)
&quot;min_score&quot;: 7.0, // 통과 기준 점수
&quot;model&quot;: &quot;gemini-3-flash-preview&quot;, // 사용된 Judge LLM 모델
&quot;prompt_version&quot;: &quot;v3&quot;, // 프롬프트 버전
&quot;generated_at_utc&quot;: &quot;...&quot;, // 생성 시간 (UTC)
&quot;feedback&quot;: [ // 세그먼트별 피드백 (선택적일 수 있음)
{
&quot;segment_id&quot;: 1,
&quot;feedback&quot;: &quot;VLM 텍스트와 레이아웃 정보를 바탕으로...&quot;
},
    ...
  ],
&quot;report&quot;: { // 상세 평가 리포트
&quot;scores_avg&quot;: {
&quot;groundedness&quot;: 10.0, // 근거 기반 점수 (환각 여부 등)
&quot;multimodal_use&quot;: 10.0, // 멀티모달 정보 활용도
&quot;note_quality&quot;: 10.0, // 노트 품질
&quot;compliance&quot;: 10.0, // 지시 이행도
&quot;final&quot;: 10.0 // 최종 평균점
},
&quot;segments&quot;: {
&quot;matched&quot;: 3,
&quot;missing_summaries&quot;: [],
&quot;missing_units&quot;: []
}
}
}
</code></pre>
<h3 id="section-16"><strong>4.6 챗봇(LangGraph)</strong></h3>
<p><strong>LangGraph</strong></p>
<p><img src="/assets/images/notion-records/sesac-note/development-05.png" alt="서비스 개발 도식 5"></p>
<p>LangGraph는 Node와 Edge로 구성된 그래프 구조를 통해 의사결정 과정을 시각화하고 제어할 수 있는 프레임워크. </p>
<p>본 프로젝트의 그래프 구조는 사진과 같음</p>
<p><strong>주요 노드</strong></p>
<ul>
<li>parse_input: 사용자 입력 파싱(시간 태그 추출, 의도 분석)</li>
<li>route_with_llm: LLM 기반 의도 파악 후 다음 단계 라우팅</li>
<li>prepare_full: 전체 요약 데이터 로드 및 응답 준비</li>
<li>decide_summary: 현재 요약만으로 답변이 충분한지 여부 판단(gemini-2.5-flash)</li>
<li>enrich_evidence: 필요 시 DB에서 STT/VLM 원본 조회로 근거 보강</li>
<li>generate_answer: 최종 컨텍스트 기반 답변 생성(gemini-3-flash-preview)</li>
</ul>
<p><strong>라우팅/처리 흐름</strong></p>
<ul>
<li>입력 처리 관문은 설정에 따라 Full/Partial 모드로 동작</li>
<li>현재는 Full mode만 구현되어 있으며, 기본 설정으로 Full mode로 동작</li>
</ul>
<p><strong>Processing Workflow</strong></p>
<ul>
<li>DB에서 Summary 데이터를 가져와 answer_records에 적재</li>
<li>LLM이 질문과 요약만으로 답변 가능한지 판단<ul>
<li>Flash mode: 증거 추가 조회 없이 요약 중심으로 빠르게 응답</li>
<li>Thinking mode: 항상 Need Evidence로 판단하여 STT/VLM 근거를 함께 조회</li>
</ul>
</li>
<li>source_refs의 STT/VLM ID 수집 후 원본 데이터 조회</li>
<li>보강된 evidence를 prompt에 넣어 최종 답변 생성</li>
</ul>
<div class="table-wrapper"><table>
<tr><th><strong>단계</strong></th><th><strong>상태 변화</strong></th><th><strong>설명</strong></th></tr>
<tr><td>초기</td><td>message</td><td>사용자 입력만 존재</td></tr>
<tr><td>Parse</td><td>cleaned_message, time_ms</td><td>시간 태그 분리 및 정제</td></tr>
<tr><td>Prepare</td><td>answer_records (Summary Only)</td><td>요약 데이터 로드</td></tr>
<tr><td>Enrich</td><td>answer_records(+Evidence)</td><td>요약 데이터 내부에 STT/VLM 근거 추가</td></tr>
<tr><td>Generate</td><td>respone, history</td><td>최종 답변 생성 및 대화 이력 갱신</td></tr>
</table></div>
<p><strong>LangGraph Prompt</strong></p>
<p><strong>Flash Mode</strong></p>
<pre><code>요약본 자체에 집중해 복잡한 추론보다는 있는 그대로 답변하는 데 초점을 둠

&quot;Use only the provided summary records to answer.&quot;

하지만 evidence도 참고할 수 있도록 함

&quot;If summaries are missing but evidence is sufficient, provide a reasonable interpretation based on the evidence.&quot;</code></pre>
<p><strong>Thinking Mode</strong></p>
<pre><code>증거를 적극적으로 활용하고 주장에 대한 근거를 설명하도록 유도함

“Use only the provided summary records and evidence to answer.&quot;

&quot;When answering, connect claims to evidence and explain the reasoning briefly.&quot;</code></pre>
<h3 id="section-17"><strong>4.7 Frontend</strong></h3>
<p>React(Vite) 기반으로 구성하였으며 예시 사진은 아래와 같음</p>
<p><img src="/assets/images/notion-records/sesac-note/development-06.png" alt="서비스 개발 도식 6"></p>
<p><img src="/assets/images/notion-records/sesac-note/development-07.png" alt="서비스 개발 도식 7"></p>
<h3 id="section-18"><strong>4.8 최적화 전략 </strong></h3>
<h4 id="section-19"><strong>4.8.1 핵심 과제</strong></h4>
<ul>
<li>Cost: VLM 호출 수, LLM 입력 토큰, 저장 용량</li>
<li>Latency/Perf: 전처리(STT+Capture) 소요 시간, VLM 병렬 처리, 배치 처리 지연</li>
<li>Quality: VLM 환각 감소, 요약의 근거성(groundedness), 노트 품질</li>
<li>Reliability: 외부 API 장애/한도 대응, 출력 스키마 안정화, 자동 재시도(Judge feedback loop)</li>
<li>DevEx: 설정 파일 분리, 모듈 독립 실행 가능, 로그/벤치마크 가시화</li>
</ul>
<hr>
<h4 id="section-20"><strong>4.8.2 Release Timeline </strong></h4>
<p><strong>v1.0 Baseline E2E + 모듈화 기반 (2025-12-30 ~ 2026-01-07)</strong></p>
<p><strong>목표</strong></p>
<ul>
<li>End-to-End 파이프라인 검증(업로드 → STT/캡처 → VLM → 요약 → 결과)</li>
<li>최적화를 위한 모듈 간 인터페이스 및 데이터 스키마 설계</li>
</ul>
<p><strong>핵심 변경</strong></p>
<ul>
<li>STT: Clova Speech 기반 인터페이스 구현 및 응답 정규화, 신뢰도 산출 로직 적용</li>
<li>Capture: 화면 전환 중심의 슬라이드 추출 로직 설계</li>
<li>Judge: 평가 지표(근거성, 품질, 준수도) 수립 및 자동화 평가를 위한 기반 마련(01/07)</li>
</ul>
<p><strong>결과</strong></p>
<ul>
<li>End-to-End 동작은 확인했지만, 토큰 낭비 및 병목 이슈가 발생한다는 개선점 확인</li>
</ul>
<hr>
<p><strong>v2.0 Batch 처리 및 QA Loop 도입 (2026-01-08 ~ 2026-01-16)</strong></p>
<p><strong>목표</strong></p>
<ul>
<li>긴 영상 처리 시 발생하는 메모리 및 토큰 제한 문제를 해결</li>
<li>자동화된 품질 관리(Judge Loop)를 통해 시스템의 재현성과 신뢰도를 확보하는 데 집중</li>
</ul>
<p><strong>핵심 변경</strong></p>
<ul>
<li>Orchestration: 전처리(Pre-ADK)와 실행(ADK Pipeline) 단계를 분리(01/08)</li>
<li>QA: Judge Judge 평가 모듈을 파이프라인에 연동, 기준에 미달하는 결과물은 피드백 루프를 통해 자동으로 재생성되는 로직 구현(01/09~01/16,  <code>c64f53c</code>)</li>
<li>Batch: 긴 영상을 N 단위로 쪼개 처리하는 Batch Mode 도입(01/12, Issue #63)</li>
<li>리팩토링: config파일로 각 모듈 프롬프트, 설정 가능하게 구조 개선</li>
</ul>
<p><strong>결과</strong></p>
<ul>
<li>Judge 도입으로 품질 미달 시 스스로 교정하는 구조를 완성</li>
<li>Batch처리 과정에서 동일 프롬프트 중복 전달로 토큰이 크게 증가하는 문제가 발생하여 v2.1에서 프롬프트 최적화로 해결</li>
</ul>
<hr>
<p><strong>v2.1 토큰 및 지연시간(Latency) 최적화 (2026-01-13 ~ 2026-01-24)</strong></p>
<p><strong>목표</strong></p>
<ul>
<li>서비스 품질은 유지하면서, 비용과 직결되는 LLM 입력 토큰 및 응답 속도를 구조적으로 개선하는 데 집중</li>
</ul>
<p><strong>핵심 변경</strong></p>
<ul>
<li>API key를 여러 개 적용하여 Round-robin 형태가 되도록 함(01/15, <code>5ae8b1d</code>)</li>
<li>Summarizer 프롬프트 영문화 및 중복 제거, 기술 용어는 영어로 출력하도록 변경. 입력 및 출력 토큰 절감(01/19, PR #97, <code>021f12c</code>)</li>
<li>출력 포맷/입력 포맷 실험: JSON → JSONL, 한글 → 영어 전환 실험(Issue #134)</li>
<li>VLM 프롬프트 안정화: 환각 문제(표 행/열, 객체 오인식 등) 개선 및 LaTeX/특수 포맷 금지로 후처리 안정성 개선(01/17, Issue #76)</li>
<li>VLM 프롬프트 최적화: VLM 프롬프트 영문화로 입력 토큰 절감(01/22~01/23, PR #115)</li>
<li>Judge 프롬프트 최적화(Issue #91)</li>
<li>Judge 최적 배치 수 실험(Issue #95)</li>
</ul>
<p><strong>결과</strong></p>
<ul>
<li>Summarizer 입력 토큰(6분 기준): 9065 → 5396 (약 -40.5%)</li>
<li>Summarizer latency: 45초 → 17.9초 (약 -60.2%)</li>
<li>VLM 토큰(장당): 1077 → 1005 (약 -6.7%)</li>
<li>토큰 절감 실험 결과(부분): JSON→JSONL(약 -4.5%), 한글→영어 단순 전환(약 -7.5%)</li>
</ul>
<hr>
<p><strong>v3.0 서비스 아키텍처 고도화 및 운영 안정화(Cloud Sync, Parallel, Provider) (2026-01-20 ~ 2026-01-29)</strong></p>
<p><strong>목표</strong></p>
<ul>
<li>로컬 실행 중심에서 벗어나 클라우드 기반의 비동기 처리 구조를 확립</li>
<li>외부 API 의존성 리스크(한도/병목)를 운영 가능 수준으로 개선</li>
<li>전처리 병렬화로 처리 시간 개선</li>
</ul>
<p><strong>핵심 변경</strong></p>
<ul>
<li>Data/Infra: Supabase Storage 연동 및 단계별 스트리밍 업로드 도입(01/20~01/22)</li>
<li>Preprocess 개별적으로 수행되던 STT와 Capture 과정을 병렬화하여 전체 전처리 시간을 단축(01/25~01/29)</li>
<li>병목 현상이 잦은 기존 Provider를 Alibaba Cloud로 전환(PR #137, 01/29)</li>
<li>Orchestration: ADK 처리 시간 병목으로 ADK → LangGraph 전환 결정(01/25, Issue #117) 및 PR merge(01/27, PR #125)</li>
</ul>
<p><strong>결과</strong></p>
<ul>
<li>전처리 시간: 15.1s → 9.0s (약 -40%)</li>
<li>캡처 중복/비용 최적화: 중복 캡처 수 11→5(-55%), VLM 호출(예시) 6→3(-50%)</li>
</ul>
<hr>
<h4 id="section-21"><strong>4.8.3 KPI 대시보드</strong></h4>
<div class="table-wrapper"><table>
<tr><th>KPI</th><th>Before</th><th>After</th><th>Delta</th><th>Evidence</th></tr>
<tr><td>Summarizer 입력 토큰(6분)</td><td>9,065</td><td>5,396</td><td>-40.5%</td><td>PR #97 / <code>021f12c</code></td></tr>
<tr><td>Summarizer latency</td><td>45s</td><td>17.9s</td><td>-60.2%</td><td>PR #97 / <code>021f12c</code></td></tr>
<tr><td>전처리 시간(STT+Capture)</td><td>15.1s</td><td>9.0s</td><td>-40%</td><td>BenchmarkTimer(01/25~01/29)</td></tr>
<tr><td>중복 캡처 수</td><td>11</td><td>5</td><td>-55%</td><td>캡처 최적화 보고서</td></tr>
<tr><td>VLM 호출 수(예시)</td><td>6</td><td>3</td><td>-50%</td><td>캡처 dedup 결과</td></tr>
</table></div>
<p><strong>추가 실험(트레이드오프)</strong></p>
{% endraw %}

배치를 작게 나누면 호출별 시스템 프롬프트가 반복되므로, 실행 시간뿐 아니라 입력 토큰도 함께 비교해야 합니다. 배치별 측정 조건·결과·권장 범위는 [13. Summarizer 배치 크기 실험]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-source-summarizer-batch %})에 정리했습니다. 프롬프트 압축·영어화의 효과는 위의 별도 최적화 기록에서 다룹니다.

{% raw %}
<h2 id="section-22"><strong><strong>5. 데이터 스키마 및 저장소 (Data &amp; Storage)</strong></strong></h2>
<h3 id="section-23"><strong><strong>5.1 Database Design (Supabase)</strong></strong></h3>
<p><img src="/assets/images/notion-records/sesac-note/development-08.jpg" alt="서비스 개발 도식 8"></p>
<ul>
<li><code>videos</code>, <code>preprocessing_jobs</code>, <code>processing_jobs</code></li>
<li><code>stt_results</code>, <code>captures</code>, <code>fs_segments</code>, <code>fs_summaries</code></li>
</ul>
<h2 id="section-24"><strong><strong>6. 결론 및 향후 과제 (Conclusion)</strong></strong></h2>
<h3 id="section-25"><strong>프로젝트 성과</strong></h3>
<p>SeSAC:Note는 강의 영상 학습에서 발생하는 “STT-only 요약의 근거 부족 문제”를 해결하기 위해, 음성(STT)과 화면 정보(VLM)를 결합한 멀티모달 노트 생성 파이프라인을 구현하였다.</p>
<p>기존 방식이 음성 정보에만 의존해 슬라이드의 핵심 용어·수식·도표 같은 시각적 근거를 놓치기 쉬웠던 반면, 본 프로젝트는 STT + VLM을 통해 동일 시간 구간의 발화와 화면 콘텐츠를 정렬하고, 이를 기반으로 구조화된 학습 노트를 생성함으로써 “영상 없이도 이해 가능한 독립 노트”에 가까운 결과물을 제공하였다. 또한 요약 결과를 기반으로 질의응답까지 확장하여, 단순 요약을 넘어 학습 보조 도구로서의 가능성을 확인하였다.</p>
<h3 id="section-26"><strong>한계점</strong></h3>
<ol>
<li>속도(Latency) 문제<p>전체 처리 시간이 길어 사용자 경험 측면에서 개선 필요성이 확인되었다.</p>
</li>
<li>품질 확보를 위한 외부 API 의존<p>요약 품질과 안정성을 높이기 위해 상용/외부 API 기반의 모델 호출을 사용했다. 이 방식은 빠른 프로토타이핑에는 유리했지만, 비용·쿼터·네트워크·키 장애 등 운영 리스크가 존재하며, 장기적으로는 자체 서빙 전환도 고려해 봐야 한다.</p>
</li>
<li>모델 서빙 경험 부재<p>이번 프로젝트에서는 모델을 직접 서빙하는 운영 경험(AWS/GCP 기반 배포, 스케일링, 모니터링, 비용 최적화 등)을 충분히 수행하지 못했다. 결과적으로 서비스 관점의 End-to-End 운영 역량까지는 검증이 제한적이었다.</p>
</li>
<li>판서 중심 강의 처리 미흡<p>현재 파이프라인은 ‘슬라이드 기반 강의’를 전제로 설계되어 있어, 판서(화이트보드/태블릿 필기) 중심 강의에서 발생하는 점진적 변화, 손글씨 인식, 지우기/덧쓰기 등 시각적 특성을 충분히 처리하지 못했다.</p>
</li>
</ol>
<h3 id="section-27"><strong>향후 개선 계획</strong></h3>
<ol>
<li>속도 개선</li>
<li>DB/스토리지 아키텍처 재설계(Supabase 의존도 축소)</li>
</ol>
<ul>
<li>Supabase 중심 구조에서 벗어나, 운영 요구사항에 맞게 DB/스토리지를 분리·이관하는 방안을 검토한다.<ul>
<li>예: PostgreSQL 자체 운영(RDS/GCP Cloud SQL), Object Storage(S3/GCS)로 분리하여 성능/비용/확장성을 최적화</li>
</ul>
</li>
<li>실행 이력(run/batch/segment) 관리와 관측성(로그/메트릭)을 강화해 대규모 처리에도 안정적으로 운영 가능한 형태로 확장한다.</li>
</ul>
<ol>
<li>모델 서빙(AWS/GCP) 실전 적용</li>
</ol>
<ul>
<li>AWS 또는 GCP에서 VLM/LLM(또는 일부 경량 모델)의 서빙 파이프라인을 직접 구축해, API 의존도를 단계적으로 낮춘다.</li>
<li>컨테이너 기반 배포(Docker), 오토 스케일링, GPU 인스턴스 운영, 모니터링/알람을 포함한 “서비스 운영 루프”를 실제로 경험하고 최적화한다.</li>
</ul>
<ol>
<li>판서 처리 기능 추가</li>
</ol>
<ul>
<li>판서 강의 특성에 맞게 “점진적 변화 감지” 및 “필기 누적 캡처” 로직을 별도로 설계한다.</li>
<li>손글씨/수식 인식(OCR/수식 인식)과 결합하여 판서 내용을 구조화하고, 슬라이드 기반 파이프라인과 동일하게 STT 시간축 정렬 → 근거 기반 요약으로 연결한다.</li>
<li>결과적으로 슬라이드/판서 혼합 강의까지 범용적으로 대응 가능한 멀티모달 노트 생성기로 확장한다.</li>
</ul>
<h2 id="section-28"><strong>Appendix</strong></h2>
<h4 id="section-29"><strong>A. 모듈별 타임라인</strong></h4>
<p>Audio (src/audio)</p>
<div class="table-wrapper"><table>
<tr><th>Date</th><th>Change Bundle</th><th>Category</th><th>Evidence</th><th>Impact</th></tr>
<tr><td>12/31</td><td>Clova STT 클라이언트 초기 구현(요청/응답 정규화, 신뢰도 계산)</td><td>DevEx/Quality</td><td><code>5e18553</code></td><td>STT 메인 엔진 기반 확보</td></tr>
<tr><td>01/01</td><td>역상 스테레오 무음 해결: volumedetect 기반 auto-mono 선택(Left/Right/Downmix/Phase-fix)</td><td>Reliability</td><td><code>0e2760f</code></td><td>무음 케이스 자동 복구</td></tr>
<tr><td>01/02</td><td>Whisper 백업 엔진 도입 및 성능 비교</td><td>Cost/Quality</td><td><code>03e4a0d</code></td><td>WER: Clova 6% vs Whisper 45% → 메인 Clova 유지</td></tr>
<tr><td>01/14~01/15</td><td>Router Pattern + settings.yaml 분리 + 리팩토링(독립 실행 가능)</td><td>DevEx</td><td><code>d906e05</code></td><td>코드 수정 없이 STT 엔진/옵션 교체</td></tr>
<tr><td>01/20</td><td>STT 세그먼트에 고유 ID(stt_XXX) 부여(추적성 강화)</td><td>Reliability/QA</td><td><code>3098f59</code></td><td>이후 Summarizer 근거 참조 가능</td></tr>
<tr><td>01/21</td><td>MP3(128k)/FLAC 등 다중 포맷 지원 + DB 직동기화 최적화</td><td>Cost/Perf</td><td><code>8e77ee6</code></td><td>저장/업로드 비용 절감, 운영 유연성 증가</td></tr>
</table></div>
<p>Capture (src/capture)</p>
<div class="table-wrapper"><table>
<tr><th>Date/Phase</th><th>Change Bundle</th><th>Category</th><th>Evidence</th><th>Impact</th></tr>
<tr><td>01/02</td><td>픽셀 차이 기반 알고리즘</td><td>Reliability</td><td><code>bc40b6a</code></td><td>초기 캡처 로직 구현</td></tr>
<tr><td>01/07</td><td>Manifest.json, 
VLM 분석용 표준 JSON 스키마 정립</td><td>DevEx</td><td><code>1aa8250</code></td><td>VLM/Judge 단계와의 데이터 계약(Contract) 수립</td></tr>
<tr><td>01/12</td><td>Delayed Save
2.5초 대기, 최적 프레임 저장</td><td> Quality </td><td><code>4263878</code></td><td> 마우스 커서, 화면 전환 노이즈로 인한 오탐지 대폭 감소</td></tr>
<tr><td>01/16</td><td>모듈 독립 및 설정 구조</td><td>DevEx</td><td><code>c12caa9</code></td><td>캡처 엔진의 독립적 테스트 및 파라미터 튜닝 용이성 확보</td></tr>
<tr><td>01/23</td><td>pHash+ORB 구조적 
특징점 매칭 알고리즘</td><td>Cost/Perf</td><td><code>bdfe9e0</code></td><td>핵심 전환점: 픽셀 단위 비교에서 의미론적 슬라이드 비교로 전환</td></tr>
<tr><td>01/23</td><td>슬라이드 중복 제거 
최적화 구현</td><td>Quality</td><td><code>bdfe9e0</code></td><td>슬라이드 내 미세 움직임 무시 및 처리 속도 향상 (비용 50% 절감)</td></tr>
<tr><td>01/24</td><td>불연속 구간 병합 로직</td><td>Accuracy </td><td><code>4a6aee4</code></td><td>동일 슬라이드가 반복 등장해도 하나의 그룹(time_ranges)으로 묶어 데이터 정합성 확보</td></tr>
<tr><td>01/29</td><td>Audio/Capture 병렬 실행 안정화</td><td>Perf</td><td><code>f844277</code></td><td>전처리 소요 시간 단축 
6분 영상(12.2s → 9.0s) </td></tr>
</table></div>
<p>대표 결과</p>
<ul>
<li>중복 캡처 수 11→5(-55%)</li>
<li>VLM 호출 수(예시) 6→3(-50%)</li>
<li>time_ranges 기반 “반복 등장 슬라이드” 무결성 확보</li>
</ul>
<p>VLM (src/vlm)</p>
<div class="table-wrapper"><table>
<tr><th>Date</th><th>Change Bundle</th><th>Category</th><th>Evidence</th><th>Impact</th></tr>
<tr><td>12/31</td><td>여러 이미지를 배치로 묶어 처리(Batch Mode)</td><td>Perf</td><td><code>66c8da9</code></td><td>API 호출 효율 개선 기반</td></tr>
<tr><td>01/10</td><td>ThreadPoolExecutor로 배치 단위 병렬 요청</td><td>Perf</td><td><code>11c4a7a</code></td><td>VLM 수집 속도 개선</td></tr>
<tr><td>01/11~01/12</td><td>병렬 로그 가시화(PR #57) + 프롬프트 버전 관리(PR #61)</td><td>DevEx/Reliability</td><td><strong>PR #57</strong><strong>
</strong><strong>PR #61</strong></td><td>운영/디버깅 용이</td></tr>
<tr><td>01/15</td><td>다중 API 키 round-robin 운영</td><td>Reliability</td><td><code>5ae8b1d</code></td><td>키 실패/한도 시 자동 전환</td></tr>
<tr><td>01/17</td><td>환각 감소 프롬프트 튜닝(표 행/열 오류, 객체 오인식 등)</td><td>Quality</td><td><strong>Issue #76 </strong><code>ee0d0c2</code></td><td>동일 속도 유지하며 결과 개선</td></tr>
<tr><td>01/22~01/23</td><td>LaTeX/Unicode/HTML 출력 금지(plain text) + 영어화(PR #115)</td><td>Reliability/Cost</td><td><strong>PR #115 </strong>
<code>84ece67</code></td><td>토큰 장당 1077→1005(-6.7%)</td></tr>
<tr><td>01/27~01/29</td><td>OpenRouter 병목 대응: Alibaba Cloud API로 전환 + RR 유지</td><td>Reliability/Perf</td><td>PR #137</td><td>API 병목 완화, 운영 안정성 상승</td></tr>
</table></div>
<p>Fusion &amp; Summarizer (src/fusion)</p>
<div class="table-wrapper"><table>
<tr><th>Date</th><th>Change Bundle</th><th>Category</th><th>Evidence</th><th>Impact</th></tr>
<tr><td>12/31</td><td>초기 Sync 및 Summarizer 구성</td><td>-</td><td><strong>PR #14</strong><strong> / </strong><code>708820a</code></td><td>-</td></tr>
<tr><td>01/05</td><td>요약이 과도하게 간결 → 프롬프트 보강 1차</td><td>Quality</td><td><strong>Issue #19</strong><strong> / </strong><code>c083ae7</code></td><td>노트 독립성 개선 방향 확보</td></tr>
<tr><td>01/12~01/13</td><td>Batch 처리 도입(긴 영상)</td><td>Perf</td><td><strong>Issue #63</strong><strong> /
</strong><strong>PR #65</strong></td><td>메모리 한계 극복</td></tr>
<tr><td>01/12~01/13</td><td>문맥 연속성(Context chaining) </td><td>Quality</td><td><strong>Issue #63</strong><strong> /
</strong><strong>PR #65</strong></td><td>배치 경계 맥락 연결성</td></tr>
<tr><td>01/13</td><td>프롬프트 압축(7000자→700자)</td><td>Cost/Perf</td><td><strong>Issue #64</strong></td><td>토큰/타임아웃 리스크 감소</td></tr>
<tr><td>01/19</td><td>Summarizer &amp; Judge 배치 기반 최적 조합 찾기/재시도 최적화</td><td>Quality/Reliability</td><td><strong>Issue #91</strong><strong>/</strong><strong>#95</strong></td><td>세그먼트 품질 안정화</td></tr>
<tr><td>01/22~01/23</td><td>영어화 + 용어 영어 유지</td><td>Cost/Perf</td><td><strong>PR #97</strong></td><td>입력 토큰 -40.5%, latency -60.2%</td></tr>
<tr><td>상시</td><td>근거 기반 요약 강제(source_type, evidence_refs) + 스키마 검증/재생성</td><td>Reliability/QA</td><td><code>cb50044</code></td><td>결과 추적성과 재현성 확보</td></tr>
</table></div>
<p>Judge (src/judge)</p>
<div class="table-wrapper"><table>
<tr><th>Date</th><th>Change Bundle</th><th>Category</th><th>Evidence</th><th>Impact</th></tr>
<tr><td>01/07</td><td>Judge 초기 구성(가중치 기반 평가 + 배치 처리)</td><td>QA</td><td><code>465c392</code><strong> </strong></td><td>품질 기준 “정의” 완료</td></tr>
<tr><td>01/08</td><td>점수 인플레이션 해결: 0~10 엄격 기준 정립</td><td>Quality</td><td><code>b5ba590</code></td><td>과대평가 방지</td></tr>
<tr><td>01/09</td><td>병렬 처리 도입(ThreadPoolExecutor)</td><td>Perf</td><td><code>7d35075</code></td><td>평가 속도 약 40% 개선</td></tr>
<tr><td>01/09</td><td>ADK UI 연동(버튼 클릭 평가)</td><td>DevEx</td><td><code>a7baa8a</code><strong> </strong></td><td>데모/운영 편의 향상</td></tr>
<tr><td>01/10</td><td>피드백을 한 줄로 압축(토큰 절감)</td><td>Cost</td><td><code>723709c</code><strong> </strong></td><td>평가 비용/지연 감소</td></tr>
<tr><td>01/15</td><td>프롬프트/설정 파일 분리(prompt.yaml)</td><td>DevEx</td><td><code>723709c</code></td><td>유지보수성 향상</td></tr>
<tr><td>01/16</td><td>Feedback loop 리팩토링(최대 2회 재시도)</td><td>Reliability</td><td><code>c64f53c</code></td><td>“자가 교정” 품질 보증 완성</td></tr>
<tr><td>01/19</td><td>Judge 프롬프트 
v2/v3 추가</td><td>Reliability</td><td><code>7be6ad1</code>
Issue #91</td><td>summarizer (v3) 조합.
v2 (34.2초, 14,131) → 
v1 (17.6초, 14,704) →
v3 (14.8초, 13,980)</td></tr>
<tr><td>02/07</td><td>배치 간 
Context Chaining </td><td>Reliability</td><td><code>c1fe261</code>
</td><td>Summarizer→ Judge 
문맥 연속성 확보, 맥락 단절 해결</td></tr>
</table></div>
<p>Orchestration / Chatbot (ADK → LangGraph)</p>
<div class="table-wrapper"><table>
<tr><th>Date</th><th>Change Bundle</th><th>Category</th><th>Evidence</th><th>Impact</th></tr>
<tr><td>01/08~01/17</td><td>ADK 기반 Root-Sub 구조, PipelineService 도입, 챗봇+Streamlit 연동</td><td>DevEx</td><td><strong>Issue #68</strong></td><td>서비스 계층화, UI-로직 분리</td></tr>
<tr><td>01/25</td><td>ADK, GenAI SDK, LangGraph 비교 후 Langgraph로 전환 결정</td><td>DevEx</td><td><strong>Issue #117</strong></td><td>그래프 기반 라우팅/제어로 확장성 확보</td></tr>
<tr><td>01/27</td><td>LangGraph PR merge</td><td>DevEx</td><td><strong>PR #125</strong></td><td>full/partial 모드 확장 기반</td></tr>
</table></div>

{% endraw %}
