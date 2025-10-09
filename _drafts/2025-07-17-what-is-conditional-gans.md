---
title: 조건부 gan
date: 2025-07-17 13:26:00 +0900
categories: [ ]
tags: [ "급발진거북이" ]
toc: true
comments: false
mermaid: true
math: true
---
# 📦 사용하는 python package

- torch==2.0.0+
- torchvision==0.15.0+
- numpy==1.24.0+
- matplotlib==3.7.0+
- Pillow==9.5.0+
- clip-by-openai==1.0

## 🚀 TL;DR

- **조건부 GANs**는 단순히 랜덤 노이즈에서 생성하는 것이 아니라 **특정 조건**을 입력받아 **원하는 의미의 데이터**를 생성하는 생성 모델
- **Pix2Pix**는 페어 데이터를 이용한 이미지 대 이미지 변환의 초기 성공 사례로 **UNet 구조**와 **적대적 학습**을 결합
- **CycleGAN**은 페어 데이터 없이도 **Cycle Consistency** 원리를 통해 두 도메인 간 변환을 가능하게 함
- **ACGAN**, **StarGAN** 등은 분류 손실과 다중 도메인 변환을 통해 조건부 생성의 품질과 범위를 확장
- **텍스트-이미지 생성**은 자연어 설명으로부터 이미지를 생성하는 기술로 높은 자유도를 제공하지만 학습 난이도가 높음
- **GigaGAN**은 대규모 모델과 피라미드 구조를 통해 고품질 텍스트-이미지 생성과 빠른 추론 속도를 동시에 달성

## 📓 실습 Jupyter Notebook

- [Conditional GANs Tutorial](https://github.com/yuiyeong/notebooks/blob/main/deep_learning/conditional_gans.ipynb)

## 🎯 조건부 생성 모델(Conditional Generation)이란?

조건부 생성 모델은 **특정 조건(condition)**을 입력받아 그 조건에 맞는 **원하는 의미를 가진 데이터**를 생성하는 생성 모델이다.

기존의 생성 모델이 단순히 데이터 분포만 학습하여 무작위로 생성물을 만들어냈다면, 조건부 생성 모델은 사용자가 **생성 결과를 제어**할 수 있게 해준다.

> 조건부 생성 모델의 핵심은 "내가 원하는 것을 만들어 달라"고 명령할 수 있다는 점이다. 예를 들어 "숫자 7을 그려줘", "말을 얼룩말로 바꿔줘", "분홍색 꽃이 있는 풍경을 그려줘"와 같은 구체적인 요구사항을 반영할 수 있다. {: .prompt-tip}

### 왜 조건부 생성이 중요한가?

- **데이터 증강**: 특정 클래스의 데이터가 부족할 때 해당 클래스만 골라서 생성할 수 있어 판별 모델 학습에 활용
- **이미지 편집**: 사용자의 의도에 맞게 이미지를 수정하거나 변환
- **창작 도구**: 텍스트 설명만으로 원하는 이미지나 콘텐츠 생성
- **실용적 응용**: 스케치를 실사로 변환, 흑백 이미지를 컬러로 변환 등

## 🏗️ 조건부 GAN의 기본 구조

조건부 GAN(Conditional GAN)은 기존 GAN에 조건 정보를 추가한 모델이다.

### 수학적 표현

기존 GAN의 목적 함수에 조건 **y**가 추가된다:

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)))] $$

여기서 **y**는 범주, 텍스트, 이미지 등 다양한 형태의 조건이 될 수 있다.

### 기본 구현 방식

가장 간단한 조건부 GAN은 조건을 **원-핫 벡터**나 **임베딩 벡터**로 변환하여 입력에 연결(concatenate)하는 방식이다.

```python
import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, embed_dim=100):
        super(ConditionalGenerator, self).__init__()
        
        # 조건(클래스)을 임베딩으로 변환
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # 노이즈와 조건을 결합한 입력 차원
        input_dim = noise_dim + embed_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512), 
            nn.ReLU(),
            nn.Linear(512, 784),  # 28x28 이미지
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # 라벨을 임베딩으로 변환
        embedded_labels = self.label_embedding(labels)
        
        # 노이즈와 조건을 결합
        combined_input = torch.cat([noise, embedded_labels], dim=1)
        
        return self.model(combined_input)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=100):
        super(ConditionalDiscriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # 이미지와 조건을 결합한 입력 차원
        input_dim = 784 + embed_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images, labels):
        # 이미지를 평면화
        flattened_images = images.view(images.size(0), -1)
        
        # 라벨을 임베딩으로 변환
        embedded_labels = self.label_embedding(labels)
        
        # 이미지와 조건을 결합
        combined_input = torch.cat([flattened_images, embedded_labels], dim=1)
        
        return self.model(combined_input)

# 사용 예시
generator = ConditionalGenerator()
discriminator = ConditionalDiscriminator()

# 배치 크기 32, 노이즈 차원 100
noise = torch.randn(32, 100)
labels = torch.randint(0, 10, (32,))  # 0-9 클래스

# 조건부 생성
fake_images = generator(noise, labels)
print(f"생성된 이미지 크기: {fake_images.shape}")  # torch.Size([32, 784])

# 판별
real_images = torch.randn(32, 784)
real_output = discriminator(real_images, labels)
fake_output = discriminator(fake_images.detach(), labels)
```

## 🎭 고급 조건부 GAN: ACGAN

ACGAN(Auxiliary Classifier GAN)은 조건부 GAN의 한계를 극복하기 위해 제안된 모델이다.

### ACGAN의 핵심 아이디어

기존 조건부 GAN에서 판별자는 "진짜/가짜" + "조건 일치 여부"를 동시에 판단했지만, ACGAN에서는 **분류 문제를 별도로 해결**한다.

```python
class ACGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(ACGANDiscriminator, self).__init__()
        
        # 공통 특징 추출 레이어
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # 진짜/가짜 판별 헤드
        self.discriminator_head = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )
        
        # 클래스 분류 헤드
        self.classifier_head = nn.Sequential(
            nn.Linear(256 * 3 * 3, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # 두 개의 출력: 진짜/가짜, 클래스 확률
        validity = self.discriminator_head(features)
        class_pred = self.classifier_head(features)
        
        return validity, class_pred

# ACGAN 손실 함수
def acgan_loss(real_images, real_labels, fake_images, fake_labels, discriminator):
    # 실제 이미지에 대한 판별과 분류
    real_validity, real_class_pred = discriminator(real_images)
    
    # 생성 이미지에 대한 판별과 분류  
    fake_validity, fake_class_pred = discriminator(fake_images)
    
    # 적대적 손실
    adversarial_loss = nn.BCELoss()
    real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
    fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
    
    # 분류 손실
    classification_loss = nn.CrossEntropyLoss()
    real_class_loss = classification_loss(real_class_pred, real_labels)
    fake_class_loss = classification_loss(fake_class_pred, fake_labels)
    
    # 총 손실
    d_loss = (real_loss + fake_loss) / 2 + (real_class_loss + fake_class_loss) / 2
    
    return d_loss
```

> ACGAN의 핵심은 판별자가 "이 이미지가 진짜인가?"와 "이 이미지는 어떤 클래스인가?"를 **동시에 학습**함으로써 클래스 정보를 더 잘 이해하게 만드는 것이다. {: .prompt-tip}

## 🖼️ Pix2Pix: 이미지 대 이미지 변환의 시작

Pix2Pix는 2017년에 발표된 **페어 데이터를 이용한 이미지 변환**의 대표적인 연구다.

### UNet 구조와 스킵 연결

Pix2Pix의 핵심은 **UNet 구조**를 생성자로 사용하는 것이다.

[시각적 표현 넣기 - UNet 구조도]

```python
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(UNetGenerator, self).__init__()
        
        # 인코더 (다운샘플링)
        self.down1 = self.down_block(input_channels, 64, normalize=False)
        self.down2 = self.down_block(64, 128)
        self.down3 = self.down_block(128, 256)
        self.down4 = self.down_block(256, 512)
        self.down5 = self.down_block(512, 512)
        
        # 디코더 (업샘플링)
        self.up1 = self.up_block(512, 512, dropout=True)
        self.up2 = self.up_block(1024, 256, dropout=True)  # 스킵 연결로 채널 수 2배
        self.up3 = self.up_block(512, 128)
        self.up4 = self.up_block(256, 64)
        
        # 최종 출력 레이어
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def down_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)
    
    def up_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 인코더 - 스킵 연결을 위해 중간 결과 저장
        d1 = self.down1(x)      # [B, 64, H/2, W/2]
        d2 = self.down2(d1)     # [B, 128, H/4, W/4]  
        d3 = self.down3(d2)     # [B, 256, H/8, W/8]
        d4 = self.down4(d3)     # [B, 512, H/16, W/16]
        d5 = self.down5(d4)     # [B, 512, H/32, W/32]
        
        # 디코더 - 스킵 연결 적용
        u1 = self.up1(d5)                              # [B, 512, H/16, W/16]
        u2 = self.up2(torch.cat([u1, d4], dim=1))      # [B, 256, H/8, W/8]
        u3 = self.up3(torch.cat([u2, d3], dim=1))      # [B, 128, H/4, W/4]
        u4 = self.up4(torch.cat([u3, d2], dim=1))      # [B, 64, H/2, W/2]
        
        output = self.final(torch.cat([u4, d1], dim=1)) # [B, 3, H, W]
        
        return output

# PatchGAN 판별자
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=6):  # 입력 + 출력 이미지 연결
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 1, 4, 1, 1)  # 패치별 판별 결과
        )
    
    def forward(self, x, y):
        # 입력과 출력 이미지를 채널 방향으로 연결
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)

# Pix2Pix 손실 함수
def pix2pix_loss(real_A, real_B, fake_B, discriminator, lambda_l1=100):
    # L1 손실 (픽셀 레벨 복원)
    l1_loss = nn.L1Loss()
    l1 = l1_loss(fake_B, real_B)
    
    # 적대적 손실
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # 실제 페어에 대한 판별
    real_output = discriminator(real_A, real_B)
    real_loss = adversarial_loss(real_output, torch.ones_like(real_output))
    
    # 생성 페어에 대한 판별
    fake_output = discriminator(real_A, fake_B)
    fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output))
    
    # 생성자 손실 (판별자를 속이려는 손실)
    gen_adversarial_loss = adversarial_loss(fake_output, torch.ones_like(fake_output))
    
    # 총 생성자 손실
    gen_loss = gen_adversarial_loss + lambda_l1 * l1
    
    # 판별자 손실
    disc_loss = (real_loss + fake_loss) / 2
    
    return gen_loss, disc_loss, l1
```

### Pix2Pix의 핵심 특징

- **UNet 스킵 연결**: 고해상도 디테일 보존
- **PatchGAN 판별자**: 전체 이미지가 아닌 패치 단위로 판별하여 지역적 디테일 향상
- **L1 + 적대적 손실**: 픽셀 레벨 정확도와 자연스러움을 동시에 추구

> Pix2Pix는 **페어 데이터가 필요**하다는 한계가 있지만, 스케치→실사, 세그멘테이션 맵→사진 등의 작업에서 뛰어난 성능을 보여줬다. {: .prompt-tip}

## 🔄 CycleGAN: 페어 데이터 없는 도메인 변환

CycleGAN은 2017년에 발표된 연구로, **페어 데이터 없이도** 두 도메인 간 변환을 가능하게 했다.

### Cycle Consistency의 핵심 아이디어

**"A → B → A로 변환했을 때 원본과 같아야 한다"**

[시각적 표현 넣기 - CycleGAN 구조도와 Cycle Consistency 설명]

### 수학적 표현

CycleGAN의 목적 함수는 다음과 같다:

$$ \mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda \mathcal{L}_{cyc}(G, F) $$

여기서 Cycle Consistency 손실은:

$$ \mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1] $$

```python
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        
        # 두 개의 생성자: X→Y, Y→X
        self.G_AB = Generator()  # A에서 B로 (예: 말 → 얼룩말)
        self.G_BA = Generator()  # B에서 A로 (예: 얼룩말 → 말)
        
        # 두 개의 판별자: A 도메인, B 도메인
        self.D_A = Discriminator()  # A 도메인 판별자
        self.D_B = Discriminator()  # B 도메인 판별자
    
    def forward(self, real_A, real_B):
        # A → B → A 사이클
        fake_B = self.G_AB(real_A)
        recovered_A = self.G_BA(fake_B)
        
        # B → A → B 사이클  
        fake_A = self.G_BA(real_B)
        recovered_B = self.G_AB(fake_A)
        
        return fake_A, fake_B, recovered_A, recovered_B

def cyclegan_loss(real_A, real_B, fake_A, fake_B, recovered_A, recovered_B, 
                  D_A, D_B, lambda_cycle=10.0, lambda_identity=0.5):
    
    # 적대적 손실
    adversarial_loss = nn.MSELoss()
    
    # D_A에 대한 손실 (실제 A vs 생성된 A)
    real_A_pred = D_A(real_A)
    fake_A_pred = D_A(fake_A.detach())
    
    loss_D_A = (adversarial_loss(real_A_pred, torch.ones_like(real_A_pred)) + 
                adversarial_loss(fake_A_pred, torch.zeros_like(fake_A_pred))) / 2
    
    # D_B에 대한 손실 (실제 B vs 생성된 B)
    real_B_pred = D_B(real_B)
    fake_B_pred = D_B(fake_B.detach())
    
    loss_D_B = (adversarial_loss(real_B_pred, torch.ones_like(real_B_pred)) + 
                adversarial_loss(fake_B_pred, torch.zeros_like(fake_B_pred))) / 2
    
    # 생성자의 적대적 손실
    loss_G_A = adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))
    loss_G_B = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))
    
    # Cycle Consistency 손실
    cycle_loss = nn.L1Loss()
    loss_cycle_A = cycle_loss(recovered_A, real_A)
    loss_cycle_B = cycle_loss(recovered_B, real_B)
    
    # Identity 손실 (선택적)
    identity_loss = nn.L1Loss()
    loss_identity_A = identity_loss(G_BA(real_A), real_A)
    loss_identity_B = identity_loss(G_AB(real_B), real_B)
    
    # 총 생성자 손실
    loss_G = (loss_G_A + loss_G_B + 
              lambda_cycle * (loss_cycle_A + loss_cycle_B) +
              lambda_identity * (loss_identity_A + loss_identity_B))
    
    # 총 판별자 손실
    loss_D = loss_D_A + loss_D_B
    
    return loss_G, loss_D

# 학습 예시
def train_cyclegan(dataloader_A, dataloader_B, num_epochs=200):
    cyclegan = CycleGAN()
    
    optimizer_G = torch.optim.Adam(
        list(cyclegan.G_AB.parameters()) + list(cyclegan.G_BA.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(cyclegan.D_A.parameters()) + list(cyclegan.D_B.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    
    for epoch in range(num_epochs):
        for batch_A, batch_B in zip(dataloader_A, dataloader_B):
            real_A, real_B = batch_A, batch_B
            
            # Forward pass
            fake_A, fake_B, recovered_A, recovered_B = cyclegan(real_A, real_B)
            
            # 손실 계산
            loss_G, loss_D = cyclegan_loss(
                real_A, real_B, fake_A, fake_B, recovered_A, recovered_B,
                cyclegan.D_A, cyclegan.D_B
            )
            
            # 생성자 업데이트
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            # 판별자 업데이트
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            
        print(f"Epoch {epoch}: G_loss={loss_G:.4f}, D_loss={loss_D:.4f}")
```

### CycleGAN의 활용 사례

- **스타일 변환**: 사진 ↔ 그림, 여름 ↔ 겨울 풍경
- **객체 변환**: 말 ↔ 얼룩말, 사과 ↔ 오렌지
- **도메인 적응**: 시뮬레이션 이미지 → 실제 이미지

> CycleGAN의 혁신은 **대칭성 가정**에 있다. "A를 B로 바꾼 후 다시 A로 바꿨을 때 원본과 같아야 한다"는 직관적인 제약으로 페어 데이터 없이도 변환을 학습할 수 있게 했다. {: .prompt-tip}

## ⭐ StarGAN: 다중 도메인 변환

StarGAN은 2018년에 발표된 연구로, **여러 도메인 간 변환**을 하나의 모델로 처리할 수 있게 했다.

### 다중 도메인 변환의 필요성

기존 방식으로 N개 도메인 간 변환을 하려면 N×(N-1)개의 모델이 필요하지만, StarGAN은 **단일 모델**로 처리한다.

```python
class StarGANGenerator(nn.Module):
    def __init__(self, conv_dim=64, num_domains=5):
        super(StarGANGenerator, self).__init__()
        
        # 다운샘플링 블록
        self.down_layers = nn.ModuleList([
            nn.Conv2d(3 + num_domains, conv_dim, 7, 1, 3),  # 이미지 + 도메인 레이블
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.InstanceNorm2d(conv_dim*4), 
            nn.ReLU()
        ])
        
        # 잔차 블록들
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(conv_dim*4) for _ in range(6)
        ])
        
        # 업샘플링 블록
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4, 2, 1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 4, 2, 1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(),
            
            nn.Conv2d(conv_dim, 3, 7, 1, 3),
            nn.Tanh()
        ])
    
    def forward(self, x, target_domain):
        # 타겟 도메인 레이블을 이미지 크기에 맞게 확장
        target_domain = target_domain.view(target_domain.size(0), target_domain.size(1), 1, 1)
        target_domain = target_domain.repeat(1, 1, x.size(2), x.size(3))
        
        # 이미지와 도메인 레이블 결합
        x = torch.cat([x, target_domain], dim=1)
        
        # 인코더
        for layer in self.down_layers:
            x = layer(x)
        
        # 잔차 블록
        for block in self.residual_blocks:
            x = block(x)
        
        # 디코더  
        for layer in self.up_layers:
            x = layer(x)
            
        return x

class StarGANDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, num_domains=5):
        super(StarGANDiscriminator, self).__init__()
        
        # 공통 특징 추출 레이어
        self.main = nn.Sequential(
            nn.Conv2d(3, conv_dim, 4, 2, 1),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(conv_dim*8, conv_dim*16, 4, 2, 1),
            nn.LeakyReLU(0.01)
        )
        
        # 진짜/가짜 판별 헤드
        self.dis_head = nn.Conv2d(conv_dim*16, 1, 3, 1, 1)
        
        # 도메인 분류 헤드
        self.cls_head = nn.Conv2d(conv_dim*16, num_domains, 2, 1, 0)
    
    def forward(self, x):
        features = self.main(x)
        
        # 진짜/가짜 점수
        validity = self.dis_head(features)
        
        # 도메인 분류 점수
        domain_pred = self.cls_head(features)
        domain_pred = domain_pred.view(domain_pred.size(0), -1)
        
        return validity, domain_pred

# StarGAN 손실 함수
def stargan_loss(real_images, real_domains, target_domains, generator, discriminator,
                lambda_cls=1.0, lambda_rec=10.0):
    
    # 가짜 이미지 생성
    fake_images = generator(real_images, target_domains)
    
    # 복원 이미지 생성 (사이클 일관성)
    reconstructed_images = generator(fake_images, real_domains)
    
    # 판별자 예측
    real_validity, real_domain_pred = discriminator(real_images)
    fake_validity, fake_domain_pred = discriminator(fake_images.detach())
    
    # 적대적 손실
    adversarial_loss = nn.MSELoss()
    d_loss_real = adversarial_loss(real_validity, torch.ones_like(real_validity))
    d_loss_fake = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
    
    # 도메인 분류 손실 (실제 이미지)
    classification_loss = nn.CrossEntropyLoss()
    d_loss_cls = classification_loss(real_domain_pred, real_domains)
    
    # 판별자 총 손실
    d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls
    
    # 생성자 적대적 손실
    fake_validity_for_g, fake_domain_pred_for_g = discriminator(fake_images)
    g_loss_fake = adversarial_loss(fake_validity_for_g, torch.ones_like(fake_validity_for_g))
    
    # 생성자 도메인 분류 손실
    g_loss_cls = classification_loss(fake_domain_pred_for_g, target_domains)
    
    # 복원 손실 (사이클 일관성)
    reconstruction_loss = nn.L1Loss()
    g_loss_rec = reconstruction_loss(reconstructed_images, real_images)
    
    # 생성자 총 손실
    g_loss = g_loss_fake + lambda_cls * g_loss_cls + lambda_rec * g_loss_rec
    
    return g_loss, d_loss

# 사용 예시
def train_stargan():
    # 5개 도메인: 성별, 나이, 머리색 등
    num_domains = 5
    
    generator = StarGANGenerator(num_domains=num_domains)
    discriminator = StarGANDiscriminator(num_domains=num_domains)
    
    # 실제 이미지와 도메인 레이블
    real_images = torch.randn(16, 3, 128, 128)
    real_domains = torch.randint(0, num_domains, (16,))
    target_domains = torch.randint(0, num_domains, (16,))
    
    # 타겟 도메인을 원-핫 벡터로 변환
    target_domains_onehot = torch.zeros(16, num_domains)
    target_domains_onehot.scatter_(1, target_domains.unsqueeze(1), 1)
    
    real_domains_onehot = torch.zeros(16, num_domains)
    real_domains_onehot.scatter_(1, real_domains.unsqueeze(1), 1)
    
    # 손실 계산
    g_loss, d_loss = stargan_loss(
        real_images, real_domains, target_domains,
        generator, discriminator
    )
    
    print(f"Generator Loss: {g_loss:.4f}")
    print(f"Discriminator Loss: {d_loss:.4f}")
```

### StarGAN의 장점

- **모델 효율성**: N개 도메인을 하나의 모델로 처리
- **다양한 속성 제어**: 나이, 성별, 머리색, 표정 등 동시 변경 가능
- **확장성**: 새로운 도메인 추가가 용이

## 💬 텍스트-이미지 생성: 자연어로 그림 그리기

텍스트에서 이미지를 생성하는 것은 **가장 높은 자유도**를 제공하지만 **학습이 가장 어려운** 조건부 생성 작업이다.

### 텍스트-이미지 생성의 어려움

- **순차적 vs 동시적**: 텍스트는 순차적으로 생성되어 이전 컨텍스트 활용 가능하지만, 이미지는 모든 픽셀을 동시에 생성해야 함
- **모달리티 차이**: 언어와 시각의 표현 방식이 근본적으로 다름
- **복잡한 의미론**: "털이 부슬부슬한 귀여운 강아지"와 같은 추상적 표현을 시각적으로 구현

### 초기 GAN 기반 접근법

```python
class TextToImageGAN(nn.Module):
    def __init__(self, text_embedding_dim=256, noise_dim=100):
        super(TextToImageGAN, self).__init__()
        
        # 텍스트 인코더 (미리 학습된 임베딩 사용 가능)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 생성자
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*64*64),  # 64x64 RGB 이미지
            nn.Tanh()
        )
        
        # 판별자 (이미지 + 텍스트를 입력으로)
        self.discriminator = nn.Sequential(
            nn.Linear(3*64*64 + 128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_embeddings, noise):
        # 텍스트 인코딩
        text_features = self.text_encoder(text_embeddings)
        
        # 노이즈와 텍스트 특징 결합
        combined_input = torch.cat([noise, text_features], dim=1)
        
        # 이미지 생성
        generated_images = self.generator(combined_input)
        generated_images = generated_images.view(-1, 3, 64, 64)
        
        return generated_images, text_features
    
    def discriminate(self, images, text_features):
        # 이미지를 평면화
        flattened_images = images.view(images.size(0), -1)
        
        # 이미지와 텍스트 특징 결합
        combined = torch.cat([flattened_images, text_features], dim=1)
        
        return self.discriminator(combined)

# 향상된 손실 함수 (잘못된 텍스트-이미지 페어 활용)
def text_to_image_loss(real_images, real_texts, wrong_texts, model):
    batch_size = real_images.size(0)
    noise = torch.randn(batch_size, 100)
    
    # 실제 텍스트로 이미지 생성
    fake_images, real_text_features = model(real_texts, noise)
    
    # 잘못된 텍스트 인코딩
    wrong_text_features = model.text_encoder(wrong_texts)
    
    # 판별자 예측
    real_real_pred = model.discriminate(real_images, real_text_features)      # 실제 이미지 + 실제 텍스트
    real_wrong_pred = model.discriminate(real_images, wrong_text_features)    # 실제 이미지 + 잘못된 텍스트  
    fake_real_pred = model.discriminate(fake_images, real_text_features)      # 생성 이미지 + 실제 텍스트
    
    # 손실 계산
    bce_loss = nn.BCELoss()
    
    # 판별자 손실
    d_loss_real_real = bce_loss(real_real_pred, torch.ones_like(real_real_pred))
    d_loss_real_wrong = bce_loss(real_wrong_pred, torch.zeros_like(real_wrong_pred))  
    d_loss_fake_real = bce_loss(fake_real_pred, torch.zeros_like(fake_real_pred))
    
    d_loss = (d_loss_real_real + d_loss_real_wrong + d_loss_fake_real) / 3
    
    # 생성자 손실 (판별자를 속이려는 손실)
    g_loss = bce_loss(fake_real_pred, torch.ones_like(fake_real_pred))
    
    return g_loss, d_loss

# 텍스트 보간법 (Interpolation) 적용
def text_interpolation_training(model, text1, text2, alpha=0.5):
    """두 텍스트 간 보간을 통한 데이터 증강"""
    
    # 텍스트 특징 추출
    text1_features = model.text_encoder(text1)
    text2_features = model.text_encoder(text2)
    
    # 선형 보간
    interpolated_features = alpha * text1_features + (1 - alpha) * text2_features
    
    # 보간된 특징으로 이미지 생성
    noise = torch.randn(text1.size(0), 100)
    combined_input = torch.cat([noise, interpolated_features], dim=1)
    interpolated_images = model.generator(combined_input)
    interpolated_images = interpolated_images.view(-1, 3, 64, 64)
    
    return interpolated_images, interpolated_features
```

## 🚀 GigaGAN: 대규모 텍스트-이미지 생성

GigaGAN은 2023년에 발표된 연구로, **대규모 모델**과 **계층적 생성**을 통해 고품질 텍스트-이미지 생성을 달성했다.

### GigaGAN의 핵심 구조

[시각적 표현 넣기 - GigaGAN 전체 구조도]

```python
class GigaGANGenerator(nn.Module):
    def __init__(self, text_embedding_dim=512, style_dim=512):
        super(GigaGANGenerator, self).__init__()
        
        # 텍스트 프로세싱 (CLIP 인코더 활용)
        self.text_encoder = nn.Linear(text_embedding_dim, 256)
        
        # 스타일 네트워크 (StyleGAN 방식)
        self.style_network = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, style_dim)
        )
        
        # 계층적 생성자 (저해상도 → 고해상도)
        self.base_generator = BaseGenerator(style_dim, 256)  # 64x64
        self.super_resolution = SuperResolutionNet(256)      # 64x64 → 512x512
        
        # 크로스 어텐션 레이어들
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(256, 256) for _ in range(6)
        ])
    
    def forward(self, text_embeddings, noise):
        batch_size = text_embeddings.size(0)
        
        # 텍스트 처리
        text_features = self.text_encoder(text_embeddings)  # [B, 256]
        
        # 스타일 벡터 생성
        style_vector = self.style_network(noise)  # [B, 512]
        
        # 기본 이미지 생성 (64x64)
        base_image = self.base_generator(style_vector)  # [B, 3, 64, 64]
        
        # 크로스 어텐션을 통한 텍스트 반영
        features = base_image
        for attention_layer in self.cross_attention_layers:
            features = attention_layer(features, text_features)
        
        # 고해상도로 업스케일
        high_res_image = self.super_resolution(features)  # [B, 3, 512, 512]
        
        return high_res_image

class CrossAttentionBlock(nn.Module):
    def __init__(self, visual_dim, text_dim, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        
        # Query: 시각적 특징, Key/Value: 텍스트 특징
        self.q_linear = nn.Linear(visual_dim, visual_dim)
        self.k_linear = nn.Linear(text_dim, visual_dim)
        self.v_linear = nn.Linear(text_dim, visual_dim)
        
        self.out_linear = nn.Linear(visual_dim, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)
        
    def forward(self, visual_features, text_features):
        B, C, H, W = visual_features.size()
        
        # 시각적 특징을 시퀀스로 변환 [B, H*W, C]
        visual_seq = visual_features.view(B, C, H*W).transpose(1, 2)
        
        # 텍스트 특징 차원 맞추기 [B, text_len, text_dim] → [B, 1, text_dim]
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # Q, K, V 계산
        Q = self.q_linear(visual_seq)    # [B, H*W, visual_dim]
        K = self.k_linear(text_features) # [B, text_len, visual_dim] 
        V = self.v_linear(text_features) # [B, text_len, visual_dim]
        
        # 멀티헤드 어텐션
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, text_len, head_dim]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, text_len, head_dim]
        
        # 어텐션 점수 계산
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 어텐션 적용
        attended = torch.matmul(attention_weights, V)  # [B, num_heads, H*W, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(B, H*W, self.visual_dim)
        
        # 출력 변환
        output = self.out_linear(attended)
        
        # 잔차 연결 및 정규화
        output = self.norm(output + visual_seq)
        
        # 원래 형태로 복원
        output = output.transpose(1, 2).view(B, C, H, W)
        
        return output

class HierarchicalDiscriminator(nn.Module):
    def __init__(self, text_dim=256):
        super(HierarchicalDiscriminator, self).__init__()
        
        # 다단계 판별자 (각 해상도별)
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(3 + text_dim, 64),   # 64x64
            PatchDiscriminator(3 + text_dim, 128),  # 128x128  
            PatchDiscriminator(3 + text_dim, 256),  # 256x256
            PatchDiscriminator(3 + text_dim, 512),  # 512x512
        ])
        
        # 텍스트 처리
        self.text_encoder = nn.Linear(text_dim, text_dim)
        
    def forward(self, images, text_features):
        batch_size = images.size(0)
        outputs = []
        
        # 텍스트 특징 처리
        text_features = self.text_encoder(text_features)
        
        # 각 해상도에서 판별
        current_images = images
        for i, discriminator in enumerate(self.discriminators):
            # 텍스트 특징을 이미지 크기에 맞게 확장
            text_map = text_features.unsqueeze(-1).unsqueeze(-1)
            text_map = text_map.repeat(1, 1, current_images.size(2), current_images.size(3))
            
            # 이미지와 텍스트 결합
            combined = torch.cat([current_images, text_map], dim=1)
            
            # 판별
            output = discriminator(combined)
            outputs.append(output)
            
            # 다음 레벨을 위해 다운샘플링
            if i < len(self.discriminators) - 1:
                current_images = F.interpolate(current_images, scale_factor=0.5, mode='bilinear')
        
        return outputs

# GigaGAN 학습 함수
def train_gigagan(dataloader, num_epochs=100):
    # 모델 초기화
    generator = GigaGANGenerator()
    discriminator = HierarchicalDiscriminator()
    
    # 옵티마이저
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.99))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.0, 0.99))
    
    # 손실 함수
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, text_embeddings) in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # 노이즈 생성
            noise = torch.randn(batch_size, 512)
            
            # === 판별자 학습 ===
            d_optimizer.zero_grad()
            
            # 실제 이미지 판별
            real_outputs = discriminator(real_images, text_embeddings)
            d_loss_real = sum([adversarial_loss(output, torch.ones_like(output)) 
                              for output in real_outputs]) / len(real_outputs)
            
            # 가짜 이미지 생성 및 판별
            fake_images = generator(text_embeddings, noise)
            fake_outputs = discriminator(fake_images.detach(), text_embeddings)
            d_loss_fake = sum([adversarial_loss(output, torch.zeros_like(output)) 
                              for output in fake_outputs]) / len(fake_outputs)
            
            # 판별자 손실
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # === 생성자 학습 ===
            g_optimizer.zero_grad()
            
            # 생성자 적대적 손실
            fake_outputs = discriminator(fake_images, text_embeddings)
            g_loss = sum([adversarial_loss(output, torch.ones_like(output)) 
                         for output in fake_outputs]) / len(fake_outputs)
            
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
```

### GigaGAN의 혁신점

- **계층적 생성**: 저해상도에서 시작하여 점진적으로 고해상도로 업스케일
- **크로스 어텐션**: 텍스트와 시각적 특징 간의 정교한 상호작용
- **빠른 추론**: 디퓨전 모델과 달리 **단일 포워드 패스**로 생성
- **스타일 제어**: 전역적 스타일과 지역적 디테일을 분리하여 제어

> GigaGAN은 GAN의 **빠른 생성 속도**와 최신 **대규모 모델**의 장점을 결합하여, 디퓨전 모델에 맞서는 고품질 텍스트-이미지 생성을 달성했다. {: .prompt-tip}

## 🎨 실제 활용 사례와 한계점

### 조건부 GANs의 실제 활용

- **콘텐츠 창작**: 게임, 영화, 광고 등에서 컨셉 아트 생성
- **패션 디자인**: 의류 디자인 프로토타이핑
- **건축 시각화**: 설계도를 실제 건물 이미지로 변환
- **의료 이미징**: 서로 다른 의료 영상 모달리티 간 변환
- **데이터 증강**: 부족한 데이터 클래스의 샘플 생성

### 현재의 한계점

- **데이터 의존성**: 고품질 페어 데이터나 대규모 데이터셋 필요
- **도메인 특화**: 특정 도메인에 최적화되면 다른 도메인에서 성능 저하
- **모드 붕괴**: 다양성 부족으로 비슷한 결과만 생성하는 경우
- **제어의 한계**: 미세한 디테일이나 복잡한 공간 관계 제어 어려움

### 미래 전망

- **멀티모달 학습**: 텍스트, 이미지, 오디오를 통합한 조건부 생성
- **실시간 편집**: 사용자 상호작용을 통한 실시간 이미지 편집
- **개인화**: 개별 사용자 선호도를 학습한 맞춤형 생성
- **윤리적 고려**: 딥페이크 방지와 안전한 AI 생성 콘텐츠

## 🔍 마무리

조건부 GANs는 **"내가 원하는 것을 만들어 달라"**는 인간의 근본적 욕구를 AI로 구현한 기술이다. Pix2Pix의 페어 데이터 방식에서 시작하여, CycleGAN의 사이클 일관성, StarGAN의 다중 도메인 변환, 그리고 최근의 대규모 텍스트-이미지 생성까지, 조건부 생성은 지속적으로 발전해왔다.

비록 디퓨전 모델이 생성 품질에서 앞서나가고 있지만, GAN의 **빠른 생성 속도**와 **직관적인 학습 과정**은 여전히 많은 실용적 가치를 제공한다. 특히 실시간 애플리케이션이나 리소스가 제한된 환경에서는 조건부 GANs가 계속해서 중요한 역할을 할 것으로 예상된다.

> 조건부 생성 모델의 진정한 가치는 단순히 "그럴듯한" 이미지를 만드는 것이 아니라, 인간의 **창의적 의도를 AI가 이해하고 구현**할 수 있게 하는 데 있다. 이는 AI가 단순한 도구를 넘어 **창작 파트너**로 발전하는 중요한 단계이다. {: .prompt-tip}