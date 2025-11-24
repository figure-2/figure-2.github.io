---
title: "시각장애인을 위한 안내 서비스"
categories:
- 2.PROJECT
- 2-3. Visually_Impaired_Service
tags:
- Deep Learning
- Object Detection
- OCR
- TTS
- Streamlit
- Project
toc: true
date: 2025-09-30
comments: true
mermaid: true
math: true
---

# 시각장애인을 위한 안내 서비스 (견: 見)

> **프로젝트 기간**: 2025.07 ~ 2025.09 (3개월)  
> **목적**: 실시간 카메라 송출을 통한 이미지 인식 및 음성 안내 모델 제작  
> **역할**: Object Detection 모델 학습, OCR 데이터 파이프라인 구축, Streamlit 웹 서비스 개발

<br>

## 🎯 프로젝트 개요

사회적 약자를 위한 ‘**Social Impact 프로젝트**’를 만들어 보고자 하는 취지에서 본 프로젝트를 시작했습니다. 단순한 삶의 편의 개선도 좋지만 필수적으로 AI가 필요한 곳에 기술이 쓰이면 좋을 것이라고 판단했습니다.

기존 시각장애인 안내견은 간단한 길 안내와 위험물 탐지만이 가능하다는 한계가 있었습니다. 따라서 시각장애인에게 눈이 되어 주자는 목표로,  **시각장애인을 위한 안내 서비스**를 제작하였습니다.

<br>

## 🏗 시스템 아키텍처

### 전체 워크플로우

카메라를 통해 입력된 이미지는 실시간으로 객체 탐지(Object Detection) 모델을 거칩니다. 위험물이나 안내판이 감지되면, 글자가 있는 영역(ROI)을 잘라내어 OCR 모델로 전달하고, 최종적으로 사용자에게 음성(TTS)으로 상황을 안내합니다.

graph TB
    %% 입력
    Input[Camera Input] --> Streamlit[Streamlit App]

    %% 처리 과정
    Streamlit --> OD[Object Detection<br/>Faster-RCNN / DETR]
    
    %% 분기 처리
    OD --> Check{Detected?}
    Check -- Yes --> Filter[Threshold Filtering<br/>score > 0.3]
    Check -- No --> Streamlit
    
    %% 텍스트 인식 분기
    Filter --> IsSign{Is it Sign/Text?}
    IsSign -- Yes --> Crop[Image Cropping]
    Crop --> OCR[OCR Model<br/>TPS-ResNet-BiLSTM-Attn]
    OCR --> Merge[Text Merge]
    
    IsSign -- No --> Merge
    
    %% 결과 생성
    Merge --> GenText[Natural Language Generation]
    GenText --> TTS[Google gTTS API]
    TTS --> Speaker[Audio Output]

    %% 스타일링
    classDef input fill:#e1f5fe,stroke:#01579b
    classDef process fill:#fff3e0,stroke:#e65100
    classDef model fill:#e8f5e9,stroke:#1b5e20
    classDef output fill:#f3e5f5,stroke:#4a148c
    
    class Input,Streamlit input
    class Filter,Check,IsSign,Crop,Merge,GenText process
    class OD,OCR,TTS model
    class Speaker output<br>

## 🛠 기술 스택

### AI & Data Science
- **Object Detection**: MMDetection, Facebook DETR, Faster-RCNN
- **OCR**: NAVER Clova TRBA (TPS-ResNet-BiLSTM-Attn)
- **Deep Learning Framework**: PyTorch, TorchVision
- **Libraries**: OpenCV, Pandas, NumPy, PIL

### Application & Deployment
- **Web Framework**: Streamlit
- **API**: Google gTTS (Text-to-Speech), Naver Cloud Platform (OCR API)
- **Environment**: Python 3.8+

<br>

## 💻 주요 기능 및 코드 구현

### 1. Object Detection (객체 탐지)

MMDetection 라이브러리를 활용하여 29가지 장애물을 탐지합니다. `inference_detector`를 통해 결과를 얻고, 신뢰도(Threshold) 0.3 이상인 객체만 필터링합니다.

# Visually_Impaired_Service/Front Streamlit/Object_Detection.py

import mmcv
from mmdet.apis import (inference_detector, show_result_pyplot)

def object_detection(img):
    # 사전 학습된 Faster-RCNN 모델 로드
    model = torch.load('./model_pt/faster-rcnn_model_0.44.pt')

    # 추론 실행
    result = inference_detector(model, img)
    
    # 결과 필터링 및 후처리
    class_number = []
    result_list = []
    ocr_list = []
    
    for i, j in enumerate(result):
        if len(j) != 0:
            for k in j:
                # Threshold 0.3 이상인 물체만 선별
                if k[-1] > 0.3:
                    class_number.append(i)
                    # 안내판, 표지판 등 글자가 있는 객체는 별도 리스트(ocr_list)로 관리
                    if i in [0, 6, 10, 12]:
                        result_list.append(k)
                        ocr_list.append(i)
    
    # ... (중략) ...
    
    # 탐지된 정보를 바탕으로 안내 멘트 생성
    object_text = '앞에 ' + ', '.join(object_list) + '가 탐지되었습니다.'
    return object_text, ocr_list, cut_list### 2. OCR (광학 문자 인식)

탐지된 객체 중 텍스트 정보가 필요한 객체(안내판, 표지판 등)는 이미지를 Crop하여 OCR 모델로 전달합니다. 모델 구조는 **TRBA (TPS-ResNet-BiLSTM-Attn)** 방식을 채택하여 불규칙한 텍스트에서도 높은 인식률을 보입니다.

**Model Architecture (TRBA)**
1. **Transformation (TPS)**: 휘어진 글자를 펴줌
2. **Feature Extraction (ResNet)**: 이미지 특징 추출
3. **Sequence Modeling (BiLSTM)**: 문맥 정보 파악
4. **Prediction (Attention)**: 최종 문자 예측

# Visually_Impaired_Service/Optical Character Recognition/model.py

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        # 1. Transformation: TPS (Thin Plate Spline)
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), ...
            )

        # 2. FeatureExtraction: ResNet
        if opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)

        # 3. Sequence Modeling: BiLSTM
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))

        # 4. Prediction: Attention
        if opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)### 3. OCR API 연동 및 결과 처리

자체 학습된 모델 외에도 Naver Cloud Platform의 OCR API를 활용하여 하이브리드 방식으로 텍스트 인식을 수행합니다.

# Visually_Impaired_Service/Front Streamlit/OCR.py

def ocr(ocr_list, cut_list):
    # ... (API 설정 코드 생략) ...
    
    ocr_text_list = []
    
    for i in cut_list:
        # Crop된 이미지를 API로 전송
        response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
        res = json.loads(response.text.encode('utf8'))
        
        # 결과 파싱
        ocr_text = ""
        for field in res['images'][0]['fields']:
            ocr_text += field['inferText'] + " "
            
        # 자연스러운 문장 생성
        if len(ocr_text) != 0:
            for i in ocr_list:
                ocr_text_list.append(label_list[i] + '에는 "' + ocr_text + '" 라고 적혀져 있습니다.') 

    return ','.join(ocr_text_list)<br>

## 📊 프로젝트 결과

### 모델 성능
- **Object Detection (Faster-RCNN)**: mAP 0.44 달성 (Custom Dataset 기준)
- **OCR (TRBA)**: Word Accuracy 85% 이상 (Scene Text Dataset 기준)

### 시연 시나리오
1. **상황**: 사용자가 버스 정류장 앞에 서 있음
2. **탐지**: "전방에 버스 정류장과 사람 2명이 탐지되었습니다." (Object Detection)
3. **인식**: "버스 정류장 안내판에는 '7016번 도착 예정'이라고 적혀져 있습니다." (OCR)
4. **출력**: 위 문장을 합성하여 음성으로 안내

<br>

## 🎓 학습한 점

1. **End-to-End 파이프라인 구축**: 단순히 모델을 학습시키는 것을 넘어, 웹캠 입력부터 음성 출력까지 이어지는 전체 서비스 파이프라인을 구축해 보았습니다.
2. **Model Selection**: 다양한 Object Detection 모델(YOLO, DETR, Faster-RCNN)을 비교 실험하며, 실시간성과 정확도 사이의 Trade-off를 경험했습니다.
3. **Data Preprocessing**: OCR 성능 향상을 위해 TPS(Spatial Transformer Network) 모듈의 중요성을 깨달았으며, 다양한 Augmentation 기법을 적용해 보았습니다.

<br>

## 🔗 관련 링크
- **GitHub Repository**: [Visually_Impaired_Service](https://github.com/figure-2/Visually_Impaired_Service)
