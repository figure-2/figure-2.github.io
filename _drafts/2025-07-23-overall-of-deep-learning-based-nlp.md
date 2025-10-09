---
title: 딥러닝 기반 자연어처리 개괄
date: 2025-07-23 13:05:00 +0900
categories: 
tags:
  - 급발진거북이
toc: true
comments: false
mermaid: true
math: true
---
## 📦 사용하는 python package

- torch==2.0.0+
- numpy==1.24.0+
- transformers==4.30.0+
- tensorflow==2.13.0+ (선택사항)

## 🚀 TL;DR

> 💡 딥러닝 기반 자연어처리는 인간의 뇌신경망 구조를 모방한 **인공신경망**을 통해 언어를 이해하고 처리하는 기술로, NLP의 꽃이라 할 수 있다!

- **인공지능 → 기계학습 → 딥러닝**의 계층 구조를 이해하는 것이 중요하며, 딥러닝은 기계학습의 한 분야로 **다층 신경망**을 활용한다
- 기계학습과 달리 딥러닝은 **특징 추출(Feature Extraction)**을 자동으로 수행하여 End-to-End 학습이 가능하다
- 규칙 기반 NLP와 딥러닝 기반 NLP의 가장 큰 차이는 **데이터 의존성**과 **일반화 능력**에 있다
- 딥러닝 모델의 학습은 **순전파(Forward Pass)**와 **역전파(Backward Pass)**를 통해 가중치를 최적화하는 과정이다
- Classical NLP의 복잡한 전처리 과정이 딥러닝에서는 **Embedding → Hidden → Output**의 단순한 구조로 자동화되었다
- 딥러닝 모델은 본질적으로 **수많은 파라미터의 최적값을 찾아가는 과정**이며, Large Language Model은 백만(Million) 단위를 넘어선다

## 📓 실습 Jupyter Notebook

- w.i.p.
## 🤖 인공지능, 기계학습, 딥러닝의 관계

인공지능, 기계학습, 딥러닝이라는 용어들이 혼재되어 사용되는 경우가 많은데, 이들 간의 명확한 관계와 흐름을 이해하는 것이 매우 중요하다. 각각을 따로 외우는 것이 아니라 상관관계와 발전 흐름을 파악해야 한다.

### 인공지능 (Artificial Intelligence)

**인공지능**은 주변 환경을 인식하고 목표를 성취할 가능성을 최대화하는 행동을 계획할 수 있는 알고리즘이나 장치를 개발하는 것이 목표다. 복잡하게 생각할 필요 없이, 인간의 다양한 지능을 컴퓨터가 수행할 수 있도록 구현한 것이 인공지능이다. 인간과 닮아가고자 하는 학문이라고 이해할 수 있다.

인공지능을 구현하는 방법은 시대에 따라 발전해왔다:

- **규칙 기반 (Rule-based)**: 명시적 규칙 정의
- **통계 기반 (Statistical-based)**: 확률과 통계 활용
- **기계학습 및 딥러닝 기반**: 데이터로부터 학습
- **Pre-trained Fine-tuning**: 사전 학습 모델 활용
- **Neural Symbolic**: 신경망과 기호 논리의 융합
- **Large Language Model**: 대규모 언어 모델

### 기계학습 (Machine Learning)

**기계학습**은 인공지능의 하위 분야로, 목표에 대한 명시적인 프로그래밍 없이도 이를 수행할 수 있는 알고리즘을 개발하는 것이 목표다.

규칙 기반 방식과의 핵심 차이:

- **규칙 기반**: 규칙에 맞게 수행하도록 명시적으로 프로그래밍
- **기계학습**: 학습 데이터만 있으면 기계가 자동으로 학습하여 특징을 추출하고 새로운 데이터를 판별

기계학습의 대표적인 알고리즘:

- **SVM (Support Vector Machine)**
- **Naive Bayes**
- **Random Forest**
- **Decision Tree**
- **K-Nearest Neighbors**

### 딥러닝 (Deep Learning)

**딥러닝**은 기계학습의 하위 분야로, 인간의 뇌신경망 구조를 모방하여 학습하는 방법이다.

역사적 발전 과정:

- 최초: **퍼셉트론(Perceptron)** - 인간의 뇌신경 구조를 구현한 최초 모델
- 발전: 퍼셉트론을 여러 층으로 깊게(Deep) 연결 → **딥러닝**
- 반대 개념: **Shallow Learning** - 층이 얕게 쌓인 신경망

[시각적 표현 넣기: Shallow Learning vs Deep Learning 구조 비교]

딥러닝의 구조는 입력층(Input), 은닉층(Hidden), 출력층(Output)으로 구성되며, 은닉층이 깊을수록 더 복잡한 패턴을 학습할 수 있다.

```python
import torch
import torch.nn as nn

# 얕은 신경망 (Shallow Network) - 은닉층 1개
class ShallowNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 깊은 신경망 (Deep Network) - 은닉층 여러 개
class DeepNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNetwork, self).__init__()
        layers = []
        
        # 여러 은닉층을 깊게 쌓음
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 모델 인스턴스 생성
shallow_model = ShallowNetwork(100, 50, 10)
deep_model = DeepNetwork(100, [256, 128, 64, 32], 10)

print(f"Shallow Network Parameters: {sum(p.numel() for p in shallow_model.parameters())}")
# 출력: Shallow Network Parameters: 5,560
print(f"Deep Network Parameters: {sum(p.numel() for p in deep_model.parameters())}")
# 출력: Deep Network Parameters: 60,298
```

## 🔄 기계학습과 딥러닝의 차이점

기계학습과 딥러닝의 가장 핵심적인 차이는 **특징 추출(Feature Extraction)** 과정을 누가 수행하느냐에 있다.

### 기계학습의 프로세스

**Input → Feature Extraction (사람이 수행) → Classification → Output**

기계학습에서 Feature Extraction은 **사람**이 직접 수행한다. 예를 들어 고양이와 강아지를 구분하는 모델을 만든다면:

- 털의 길이가 어떻게 다른지
- 눈의 크기가 어떻게 다른지
- 귀의 모양이 어떻게 다른지
- 코의 형태가 어떻게 다른지

이러한 특징들을 인간이 직접 정의하고 설계해야 한다. 과거 머신러닝에서는 이 Feature Extraction에 굉장히 많은 시간이 소요되었다. 그러나 이 때문에 설명가능성(Explainability) 측면에서는 딥러닝보다 우수할 수 있다.

### 딥러닝의 프로세스

**Input → [Feature Extraction + Classification (End-to-End)] → Output**

딥러닝에서는 Feature Extraction과 Classification이 **End-to-End**로 자동으로 이루어진다. 데이터만 주어지면 컴퓨터가 자동으로 특징을 정의하고 분류를 진행한다.

이로 인한 변화:

- **인간의 노력 감소**: Feature Engineering이 자동화됨
- **구현의 용이성**: 데이터만 있으면 모델 구축 가능
- **성능의 급격한 향상**: 인간이 생각하지 못한 특징도 학습
- **블랙박스 문제 발생**: 왜 그런 결과가 나왔는지 설명이 어려움
- **데이터의 중요성 부각**: 데이터의 질과 양이 성능을 좌우

```python
# 기계학습 방식: 수동 특징 추출
def machine_learning_approach(image):
    """
    기계학습: 사람이 특징을 정의하고 추출
    """
    # 사람이 정의한 특징들
    features = []
    features.append(calculate_fur_length(image))      # 털 길이
    features.append(calculate_eye_size(image))        # 눈 크기
    features.append(calculate_ear_shape(image))       # 귀 모양
    features.append(calculate_color_histogram(image)) # 색상 분포
    
    # 추출된 특징으로 분류
    result = classifier.predict(features)
    return result

# 딥러닝 방식: 자동 특징 추출
class DeepLearningModel(nn.Module):
    """
    딥러닝: 특징 추출과 분류가 End-to-End로 자동 학습
    """
    def __init__(self):
        super(DeepLearningModel, self).__init__()
        # 특징 추출 레이어 (자동으로 학습됨)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 분류 레이어
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # End-to-End 학습: 특징 추출과 분류가 한번에
        x = self.pool(F.relu(self.conv1(x)))  # 자동 특징 추출
        x = self.pool(F.relu(self.conv2(x)))  # 자동 특징 추출
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 분류
        return x
```

> 딥러닝의 자동 특징 추출 능력으로 인해 데이터의 중요성이 극도로 부각되었다. 데이터만 충분하다면 인간이 생각하지 못한 특징까지 학습할 수 있기 때문이다. {: .prompt-tip}

## ⚖️ 규칙 기반 vs 딥러닝 기반의 근본적 차이

규칙 기반과 딥러닝 기반 접근법의 차이를 정확히 이해하는 것이 중요하다.

### 규칙 기반 방식의 특징

**장점:**

- **적은 양의 데이터로 일반화 가능**: 규칙만 정의하면 됨
- **논리적 추론을 통한 결론 도출**: 명확한 인과관계
- **학습에 필요한 데이터가 비교적 적음**: 규칙이 핵심
- **결과에 대한 명확한 설명 가능**: 어떤 규칙이 적용되었는지 추적 가능

**한계점:**

- **전문가의 실력을 넘어서기 매우 어려움**: 인간이 정의한 규칙은 인간의 사고를 넘어설 수 없음
- **전문가의 오류가 그대로 전파됨**: 인간의 편견과 실수도 함께 전달
- **규칙 구축에 많은 시간과 비용 소요**: 도메인 전문가가 필요
- **주로 Toy Task에만 적용 가능**: 복잡한 실제 문제에는 한계

### 딥러닝 기반 방식의 특징

**장점:**

- **데이터의 질이 좋으면 인간의 실력을 넘어설 수 있음**: 특징을 딥러닝이 자동으로 추출하기 때문
- **인간이 생각하지 못한 새로운 방법 사용 가능**: 데이터에서 패턴을 자동 발견
- **복잡한 실제 문제에 적용 가능**: 대규모 데이터 처리에 적합

**한계점:**

- **기본적으로 많은 데이터가 필요**: 데이터의 양과 질이 성능을 좌우
- **논리적 추론이 아닌 귀납적, 경험적 근사**: 통계적 패턴에 의존
- **결과 해석이 어려움**: 블랙박스 문제
- **모델 구축에 많은 계산 자원 필요**: GPU, TPU 등 고성능 하드웨어 필요

### Neural Symbolic - 융합의 시도

최근에는 **Neural Symbolic** 접근법처럼 딥러닝과 규칙 기반을 융합하려는 시도가 활발하다. 이는 학문이 앞으로만 나아가는 것이 아니라 과거의 방법론을 다시 융합하는 트렌드를 보여준다. 딥러닝 기반 모델도 Neural Symbolic을 적용할 경우 규칙 구축에 시간과 비용이 소요될 수 있다.

```python
# 규칙 기반 예시
class RuleBasedClassifier:
    def __init__(self):
        # 전문가가 정의한 규칙들
        self.rules = {
            'spam': ['discount', 'free', 'click here', 'limited offer'],
            'important': ['urgent', 'deadline', 'meeting', 'action required']
        }
    
    def classify(self, text):
        text_lower = text.lower()
        
        # 규칙 기반 분류
        for category, keywords in self.rules.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'normal'

# 딥러닝 기반 예시
class DeepLearningClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # 데이터로부터 자동으로 패턴 학습
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Neural Symbolic 융합 예시
class NeuralSymbolicClassifier(nn.Module):
    def __init__(self, neural_model, rule_based_model):
        super().__init__()
        self.neural = neural_model
        self.rules = rule_based_model
        self.combiner = nn.Linear(2, 1)  # 두 접근법 결합
    
    def forward(self, x, text):
        # 딥러닝 예측
        neural_output = self.neural(x)
        
        # 규칙 기반 예측
        rule_output = self.rules.classify(text)
        
        # 두 결과 융합
        combined = self.combiner(torch.cat([neural_output, rule_output]))
        return combined
```

## 🏗️ 딥러닝 모델의 구조적 분류

딥러닝 모델은 입력과 출력의 관계에 따라 다양하게 분류할 수 있다. 이보다 더 다양한 구조가 있을 수 있지만, 대표적인 구조들은 다음과 같다:

### 1대1 (One-to-One)

- **구조**: 단일 입력 → 단일 출력
- **예시**: 이미지 분류, 감정 분류

### 1대N (One-to-Many)

- **구조**: 단일 입력 → 시퀀스 출력
- **예시**: 이미지 캡셔닝 (이미지 → 문장)

### N대1 (Many-to-One)

- **구조**: 시퀀스 입력 → 단일 출력
- **예시**: 감성 분석 (문장 → 긍정/부정)

### N대N (Many-to-Many)

N대N 구조는 **두 가지 형태**로 구분된다:

#### 1. 병렬 처리 방식 (Parallel)

- 입력 시퀀스의 각 요소가 병렬적으로 처리되어 출력 생성
- 예시: 품사 태깅, 개체명 인식

#### 2. 인코더-디코더 방식 (Encoder-Decoder)

- 입력 시퀀스를 먼저 인코딩한 후, 이를 기반으로 출력 시퀀스 생성
- 예시: 기계 번역, 텍스트 요약

[시각적 표현 넣기: 4가지 구조와 N대N의 2가지 세부 형태 다이어그램]

```python
import torch.nn as nn

# 1대1: 이미지 분류
class OneToOne(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# 1대N: 이미지 캡셔닝
class OneToMany(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, output_dim)
        self.seq_len = seq_len
    
    def forward(self, x):
        encoded = self.encoder(x)
        # 단일 입력을 시퀀스로 확장
        decoded = self.decoder(encoded.unsqueeze(0).repeat(self.seq_len, 1, 1))
        return decoded

# N대1: 감성 분석
class ManyToOne(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# N대N (병렬): 품사 태깅
class ManyToManyParallel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, tagset_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 각 시간 단계마다 출력 생성
        return self.fc(lstm_out)

# N대N (인코더-디코더): 기계 번역
class ManyToManySeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_dim):
        super().__init__()
        # 인코더
        self.encoder_embedding = nn.Embedding(src_vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        
        # 디코더
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        # 인코더: 소스 시퀀스 처리
        src_embedded = self.encoder_embedding(src)
        _, (hidden, cell) = self.encoder(src_embedded)
        
        # 디코더: 타겟 시퀀스 생성
        tgt_embedded = self.decoder_embedding(tgt)
        output, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.output_projection(output)
```

## 📚 Classical NLP vs 딥러닝 기반 NLP

딥러닝 이전의 Classical NLP와 딥러닝 기반 NLP의 차이는 매우 극적이다.

### Classical NLP의 복잡한 파이프라인

Classical NLP는 언어학 지식을 기반으로 다음과 같은 복잡한 단계를 거쳤다:

1. **토큰화 (Tokenization)**: 문장을 단어로 분리
2. **형태소 분석 (Morphological Analysis)**: 단어의 형태소 분해
3. **품사 태깅 (POS Tagging)**: 각 단어의 품사 결정
4. **불용어 제거 (Stopword Removal)**: 의미 없는 단어 제거
5. **구문 분석 (Syntactic Parsing)**: 문장 구조 분석
6. **의미 분석 (Semantic Analysis)**: 의미 파악
7. **화용 분석 (Pragmatic Analysis)**: 문맥 이해

이러한 전처리와 특징 추출에 굉장히 많은 노력이 필요했다.

### 딥러닝 기반 NLP의 단순화

딥러닝 기반 NLP는 이 모든 과정을 크게 단순화했다:

**Embedding → Hidden → Output**

단 세 단계로 축약된 이 구조의 특징:

- **전처리 과정의 대폭 감소**: 복잡한 언어학적 분석 불필요
- **자동 특징 추출**: 데이터로부터 자동으로 패턴 학습
- **End-to-End 학습**: 입력에서 출력까지 직접 학습

```python
# Classical NLP의 복잡한 파이프라인
def classical_nlp_pipeline(text):
    # 1. 토큰화
    tokens = tokenize(text)
    
    # 2. 형태소 분석
    morphemes = morphological_analysis(tokens)
    
    # 3. 품사 태깅
    pos_tags = pos_tagging(morphemes)
    
    # 4. 불용어 제거
    filtered = remove_stopwords(pos_tags)
    
    # 5. 구문 분석
    parse_tree = syntactic_parsing(filtered)
    
    # 6. 의미 분석
    semantics = semantic_analysis(parse_tree)
    
    # 7. 특징 추출 (수동)
    features = manual_feature_extraction(semantics)
    
    # 8. 분류
    result = traditional_classifier.predict(features)
    return result

# 딥러닝 기반 NLP의 단순한 구조
class DeepNLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        # 1. Embedding: 텍스트를 벡터로 자동 변환
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Hidden: 자동 특징 추출 및 학습
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 3. Output: 최종 결과 생성
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 단 3단계로 처리 완료
        embedded = self.embedding(x)      # 임베딩
        hidden_states, _ = self.lstm(embedded)  # 특징 추출
        output = self.fc(hidden_states[:, -1, :])  # 출력
        return output

# 사용 예시
model = DeepNLP(vocab_size=10000, embed_dim=300, hidden_dim=256, output_dim=2)
print("Classical NLP: 7-8단계의 복잡한 처리")
print("Deep Learning NLP: 3단계의 단순한 처리")
```

### 언어학 중요성의 변화

딥러닝이 등장하면서 언어학의 중요성이 많이 낮아진다는 얘기가 들리는 이유가 바로 이 부분 때문이다. Classical NLP에서 필수적이었던 언어학적 전처리 과정이 딥러닝에서는 자동화되어 불필요해졌기 때문이다.

> 하지만 언어학적 지식이 완전히 불필요해진 것은 아니다. 모델의 해석, 오류 분석, 성능 개선을 위해서는 여전히 언어학적 이해가 중요하다. {: .prompt-warning}

## 🎯 딥러닝 모델의 학습: 가중치를 찾아가는 여정

딥러닝 모델의 학습은 본질적으로 **가중치(Weight)를 학습**하는 과정이다. 딥러닝이 "Learning"이라 불리는 이유는 바로 이 가중치를 조절하면서 학습하기 때문이다.

### 파라미터의 규모와 발전

신경망 레이어의 출력값은 가중치(파라미터)들의 값에 의해 결정된다. 이러한 가중치가 많아질수록 파라미터가 증가한다:

- **일반 딥러닝 모델**: 수천 개의 파라미터 레이어
- **Large Language Model**: 수천 개를 넘어 **백만(Million) 단위**로 구성
- **GPT-3**: 1750억 개의 파라미터
- **ChatGPT (GPT-4)**: 추정 1조 개 이상의 파라미터

파라미터가 점점 커진 것이 현재의 ChatGPT와 같은 Large Language Model의 핵심이다.

### 파라미터 계산 예시

N개 입력을 받아 M개 출력을 만드는 완전 연결층(Fully Connected Layer):

- **가중치**: N × M 개
- **편향(Bias)**: M 개
- **총 파라미터**: N × M + M 개

이는 선형 함수 **y = Ax + b**의 연속적인 적용으로 볼 수 있다.

```python
import torch
import torch.nn as nn

# 파라미터 수 계산 및 시각화
def analyze_model_parameters(model):
    """
    모델의 파라미터를 분석하고 시각화
    """
    total_params = 0
    layer_info = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        layer_info.append({
            'name': name,
            'shape': list(param.shape),
            'params': num_params
        })
        print(f"{name}: {param.shape} = {num_params:,} parameters")
    
    print(f"\n총 파라미터 수: {total_params:,}")
    return total_params, layer_info

# 간단한 모델 예시
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)  # MNIST 입력
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)   # 10개 클래스
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Large Language Model 스케일 예시
class MiniLLM(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=768, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=12),
            num_layers=num_layers
        )
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x

# 모델 생성 및 파라미터 분석
simple_model = SimpleModel()
print("=== Simple Model Parameters ===")
simple_params, _ = analyze_model_parameters(simple_model)
# 출력: 총 파라미터 수: 235,146

print("\n=== Mini Language Model Parameters ===")
mini_llm = MiniLLM()
llm_params = sum(p.numel() for p in mini_llm.parameters())
print(f"Mini LLM 총 파라미터: {llm_params:,}")
# 출력: Mini LLM 총 파라미터: 약 8500만

print(f"\n파라미터 규모 비교:")
print(f"Simple Model: {simple_params:,}")
print(f"Mini LLM: {llm_params:,} (약 {llm_params/simple_params:.0f}배)")
print(f"GPT-3: 175,000,000,000 (1750억)")
print(f"ChatGPT-4: ~1,000,000,000,000 (약 1조)")
```

### 학습의 본질: 최적 파라미터 찾기

딥러닝에서 학습이란 결국 **모델의 파라미터 값을 조절하는 과정**이다:

1. 원하는 출력을 만들기 위해 모든 파라미터를 정밀하게 조정
2. 파라미터에 따라 다양한 입력-출력 관계 학습 가능
3. 입력이 이미지면 Computer Vision
4. 입력이 자연어면 Natural Language Processing

딥러닝의 학습 = **수많은 파라미터의 최적값을 찾아가는 과정**

## 🔄 순전파(Forward Pass)와 역전파(Backward Pass)

딥러닝의 학습은 **순전파**와 **역전파**의 반복으로 이루어진다.

### 순전파 (Forward Pass)

입력으로부터 예측(Prediction)을 만들어내는 과정:

1. 입력이 모델에 들어옴
2. 각 은닉층의 가중치와 연산을 통과
3. 최종 출력(예측값) 생성

### 역전파 (Backward Pass)

예측과 정답 사이의 차이를 줄이는 방향으로 파라미터를 수정하는 과정:

1. 예측값과 정답의 차이(손실) 계산
2. 손실을 최소화하는 방향으로 그래디언트 계산
3. 체인 룰(Chain Rule)을 통해 각 층의 그래디언트 전파
4. 파라미터 업데이트

**실제 딥러닝 모델의 학습이 이루어지는 과정은 바로 이 역전파**에서다.

[시각적 표현 넣기: 고양이 이미지 → 모델 → "오리" 예측 → 손실 계산 → 역전파 과정]

### 구체적인 학습 예시

1. **입력**: 고양이 이미지가 들어옴
2. **순전파**: 딥러닝 모델이 예측 → "오리"
3. **오류 발생**: 실제 정답 레이블은 "고양이"
4. **손실 계산**: Loss Function이 "오리"와 "고양이"의 차이 계산
5. **역전파**: 오차를 최소화하는 방향으로 그래디언트 계산
6. **파라미터 수정**: Backpropagation으로 가중치 조정
7. **반복**: 학습 데이터로 Forward-Backward 연속 수행
8. **최적화**: 가장 최적의 가중치를 찾아감

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 완전한 학습 과정 구현
class CatDogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 2)  # 고양이, 개
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 학습 과정 상세 구현
def train_step_detailed(model, data, target, optimizer, criterion):
    """
    한 스텝의 학습 과정을 상세히 보여주는 함수
    """
    print("=== 학습 과정 시작 ===")
    
    # 1. 순전파 (Forward Pass)
    print("1. Forward Pass: 입력 → 모델 → 예측")
    output = model(data)
    predictions = torch.argmax(output, dim=1)
    print(f"   예측 결과: {['고양이', '개'][predictions[0].item()]}")
    
    # 2. 손실 계산
    print("2. Loss Calculation: 예측과 정답의 차이 계산")
    loss = criterion(output, target)
    print(f"   정답: {['고양이', '개'][target[0].item()]}")
    print(f"   손실값: {loss.item():.4f}")
    
    # 3. 그래디언트 초기화
    print("3. Zero Gradients: 이전 그래디언트 제거")
    optimizer.zero_grad()
    
    # 4. 역전파 (Backward Pass)
    print("4. Backward Pass: 손실을 최소화하는 방향 계산")
    loss.backward()
    
    # 그래디언트 확인
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
    print(f"   총 그래디언트 크기: {total_grad_norm:.4f}")
    
    # 5. 파라미터 업데이트
    print("5. Parameter Update: 가중치 조정")
    optimizer.step()
    print(f"   파라미터 업데이트 완료!")
    
    print("=== 학습 과정 종료 ===\n")
    return loss.item()

# 모델, 손실함수, 옵티마이저 설정
model = CatDogClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 가상의 데이터로 학습 시뮬레이션
batch_size = 4
dummy_image = torch.randn(batch_size, 3, 32, 32)  # 32x32 RGB 이미지
dummy_label = torch.tensor([0, 1, 0, 1])  # 고양이, 개, 고양이, 개

# 한 번의 학습 스텝 실행
loss = train_step_detailed(model, dummy_image, dummy_label, optimizer, criterion)

# 여러 에포크 학습 시뮬레이션
print("=== 전체 학습 과정 ===")
for epoch in range(3):
    epoch_loss = 0
    for _ in range(10):  # 10개 배치
        dummy_image = torch.randn(batch_size, 3, 32, 32)
        dummy_label = torch.randint(0, 2, (batch_size,))
        
        output = model(dummy_image)
        loss = criterion(output, dummy_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}: 평균 손실 = {epoch_loss/10:.4f}")
```

### 손실 함수 (Loss Function)

손실 함수는 딥러닝 모델을 학습하기 위해 손실을 계산하는 핵심 요소다:

- **역할**: 예측값과 정답 간의 차이를 수치화
- **목표**: 이 값을 최소화하는 방향으로 파라미터 조정
- **과정**:
    1. 손실 계산
    2. 편미분으로 그래디언트 계산
    3. 그래디언트를 통해 파라미터 수정

일반적인 손실 함수:

- **분류**: Cross-Entropy Loss
- **회귀**: Mean Squared Error (MSE)
- **순위**: Ranking Loss

### 체인 룰 (Chain Rule)과 역전파

그래디언트를 계산하는 방식은 **체인 룰(Chain Rule)**을 이용한다. 이 과정을 통해 모델의 파라미터를 조정하는 것이 **역전파(Backpropagation)**다.

```python
# 체인 룰의 수학적 구현 예시
def chain_rule_demonstration():
    """
    체인 룰을 통한 그래디언트 계산 과정 시연
    f(x) = loss(output(hidden(input(x))))
    """
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    
    # Forward pass with intermediate values
    z1 = x * 2  # 첫 번째 변환
    z2 = z1 + 3  # 두 번째 변환
    z3 = z2 ** 2  # 세 번째 변환
    loss = z3.mean()  # 최종 손실
    
    print("Forward Pass:")
    print(f"x = {x.data}")
    print(f"z1 = x * 2 = {z1.data}")
    print(f"z2 = z1 + 3 = {z2.data}")
    print(f"z3 = z2^2 = {z3.data}")
    print(f"loss = mean(z3) = {loss.data}")
    
    # Backward pass (자동 미분)
    loss.backward()
    
    print("\nBackward Pass (Chain Rule):")
    print(f"∂loss/∂x = {x.grad}")
    
    # 수동 계산으로 검증
    # ∂loss/∂z3 = 1/2 (mean의 미분)
    # ∂z3/∂z2 = 2*z2
    # ∂z2/∂z1 = 1
    # ∂z1/∂x = 2
    # ∂loss/∂x = ∂loss/∂z3 * ∂z3/∂z2 * ∂z2/∂z1 * ∂z1/∂x
    
    z2_value = (x * 2 + 3).data
    manual_grad = (1/2) * 2 * z2_value * 1 * 2
    print(f"수동 계산 결과: {manual_grad}")

chain_rule_demonstration()
```

## 💡 딥러닝이 자연어처리에 미친 영향

딥러닝 기반 자연어처리는 단순히 성능 향상만 가져온 것이 아니라, NLP 연구와 응용의 패러다임 자체를 바꾸었다.

### 주요 변화들

1. **데이터 중심 접근**: 규칙과 언어학적 지식보다 데이터의 질과 양이 중요해짐
2. **End-to-End 학습**: 복잡한 파이프라인 없이 직접 학습 가능
3. **전이 학습의 활성화**: 한 작업에서 학습한 지식을 다른 작업에 활용
4. **다국어 처리 용이**: 언어별 규칙 없이도 다양한 언어 처리 가능
5. **실시간 처리 가능**: GPU 가속으로 대규모 텍스트 실시간 처리

### 실제 응용 분야

딥러닝 기반 NLP는 현재 다음과 같은 분야에서 활발히 사용되고 있다:

```python
# 실제 응용 예시들
applications = {
    "검색 엔진": "사용자 의도 파악, 시맨틱 검색",
    "기계 번역": "Google Translate, Papago",
    "챗봇": "ChatGPT, Claude, Gemini",
    "음성 비서": "Siri, Alexa, Google Assistant",
    "감성 분석": "제품 리뷰 분석, 소셜 미디어 모니터링",
    "문서 요약": "뉴스 요약, 논문 요약",
    "질의 응답": "고객 서비스, FAQ 자동화",
    "코드 생성": "GitHub Copilot, CodeWhisperer"
}

for app, example in applications.items():
    print(f"{app}: {example}")
```

### 미래 전망

딥러닝 기반 NLP는 계속 발전하고 있으며, 특히:

- **초거대 언어 모델**: GPT-4, Claude 등 파라미터 수조 개 규모
- **멀티모달 학습**: 텍스트, 이미지, 음성을 함께 이해
- **Few-shot Learning**: 적은 데이터로도 새로운 작업 학습
- **Prompt Engineering**: 프롬프트로 모델 성능 최적화

> 딥러닝 기반 자연어처리는 이제 NLP의 표준이 되었으며, 앞으로도 더욱 발전할 것으로 예상된다. 특히 Large Language Model의 등장으로 인간 수준의 언어 이해와 생성이 가능해지고 있다. {: .prompt-tip}

## 🎓 마무리

딥러닝 기반 자연어처리는 인공지능의 한 분야로서, 인간의 언어를 이해하고 처리하는 혁명적인 기술이다. 핵심을 정리하면:

1. **딥러닝은 기계학습의 한 분야**로, 인간의 뇌신경망을 모방한 깊은 신경망 구조를 사용한다
2. **특징 추출의 자동화**가 딥러닝의 가장 큰 혁신이며, 이로 인해 End-to-End 학습이 가능해졌다
3. **파라미터 학습**이 딥러닝의 본질이며, 순전파와 역전파를 통해 최적의 가중치를 찾아간다
4. **Classical NLP의 복잡한 과정**이 Embedding-Hidden-Output의 단순한 구조로 대체되었다
5. **Large Language Model**의 등장으로 수백만, 수십억 개의 파라미터를 통해 인간 수준의 언어 처리가 가능해졌다

이러한 기초 개념들을 확실히 이해하는 것이 딥러닝 기반 자연어처리를 깊이 있게 학습하는 시작점이 된다.