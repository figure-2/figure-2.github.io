---
title: "🎨 Pix2Pix와 CycleGAN: 조건부 GAN을 활용한 이미지 간 변환"
date: 2025-07-17 13:29:00 +0900
categories: 
tags:
  - 급발진거북이
toc: true
comments: false
mermaid: true
math: true
---
## 📦 사용하는 python package

- torch==1.4.0+
- torchvision==0.5.0+
- matplotlib==3.3.0+
- PIL==7.2.0+
- numpy==1.19.0+
- os (내장 모듈)

## 🚀 TL;DR

- **Pix2Pix**는 **paired 데이터**를 활용한 조건부 GAN으로, 입력 이미지를 조건으로 받아 대상 도메인의 이미지를 생성한다
- **CycleGAN**은 **unpaired 데이터**로도 이미지 변환이 가능하며, **cycle consistency loss**를 통해 두 도메인 간 양방향 변환을 학습한다
- Pix2Pix는 **U-Net 구조의 생성자**와 **PatchGAN 판별자**를 사용하고, **GAN loss + L1 reconstruction loss**로 학습한다
- CycleGAN은 **ResNet 기반 생성자 2개**와 **판별자 2개**를 사용하며, **adversarial loss + cycle consistency loss**로 학습한다
- **실제 응용 분야**: 스케치→사진, 흑백→컬러, 낮→밤, 말→얼룩말 등 다양한 이미지 스타일 변환에 활용된다

## 📓 실습 Jupyter Notebook

- [Pix2Pix 공식 코드](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [CycleGAN & Pix2Pix PyTorch 구현](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## 🎯 이미지 간 변환(Image-to-Image Translation)이란?

이미지 간 변환은 하나의 이미지 도메인에서 다른 이미지 도메인으로 이미지를 변환하는 기술이다. 예를 들어 스케치를 실제 사진으로, 흑백 사진을 컬러 사진으로, 또는 건물의 라벨맵을 실제 건물 사진으로 변환하는 것이 이에 해당한다.

전통적인 컴퓨터 비전에서는 각 변환 작업마다 별도의 알고리즘을 개발해야 했지만, **조건부 GAN(Conditional GAN)**의 등장으로 하나의 프레임워크로 다양한 변환 작업을 수행할 수 있게 되었다.

> 이미지 간 변환은 **조건부 생성 모델링**의 대표적인 응용 사례로, 입력 이미지를 조건으로 활용하여 원하는 스타일이나 도메인의 이미지를 생성하는 기술이다. {: .prompt-tip}

### 주요 응용 분야

- **스타일 변환**: 사진을 그림체로, 그림을 사진으로
- **시맨틱 분할**: 라벨맵을 실제 이미지로
- **이미지 복원**: 흑백을 컬러로, 저해상도를 고해상도로
- **도메인 적응**: 낮 풍경을 밤 풍경으로

## 🎨 Pix2Pix: Paired Data를 활용한 이미지 변환

Pix2Pix는 2017년 Isola et al.이 제안한 조건부 GAN 기반의 이미지 간 변환 모델이다. 핵심 아이디어는 **입력 이미지와 출력 이미지의 쌍(pair)**이 존재하는 데이터셋에서 하나의 통합된 프레임워크로 다양한 이미지 변환 작업을 수행하는 것이다.

### 수학적/이론적 표현

Pix2Pix의 목적함수는 조건부 GAN 손실과 L1 재구성 손실의 조합으로 구성된다:

$$ \mathcal{L}_{total} = \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G) $$

여기서 조건부 GAN 손실은:

$$ \mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x}[\log(1 - D(x, G(x)))] $$

L1 재구성 손실은:

$$ \mathcal{L}_{L1}(G) = \mathbb{E}_{x,y}[||y - G(x)||_1] $$

- **G**: 생성자 (입력 이미지 x를 받아 출력 이미지 생성)
- **D**: 판별자 (입력-출력 이미지 쌍의 진위 판별)
- **λ**: L1 손실의 가중치 (일반적으로 100 사용)

### 아키텍처 구조

**생성자 (U-Net)**

- 인코더-디코더 구조에 skip connection 추가
- 고해상도 세부사항 보존에 효과적
- 입력 이미지의 공간적 정보를 유지하면서 변환 수행

**판별자 (PatchGAN)**

- 70×70 패치 단위로 이미지의 진위 판별
- 전체 이미지가 아닌 로컬 패치의 사실성에 집중
- 고주파 세부사항과 텍스처 품질 개선

[시각적 표현 넣기 - Pix2Pix 아키텍처 다이어그램]

### 실제 구현 예시

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class Pix2PixModel:
    def __init__(self, direction='BtoA', lambda_L1=100.0):
        self.direction = direction
        self.lambda_L1 = lambda_L1
        
        # 네트워크 초기화
        self.netG = self._define_generator()
        self.netD = self._define_discriminator()
        
        # 손실 함수
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        
        # 옵티마이저
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    def optimize_parameters(self, real_A, real_B):
        # Forward pass
        fake_B = self.netG(real_A)
        
        # 판별자 업데이트
        self.optimizer_D.zero_grad()
        
        # Real 이미지 쌍
        pred_real = self.netD(torch.cat([real_A, real_B], 1))
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        
        # Fake 이미지 쌍
        pred_fake = self.netD(torch.cat([real_A, fake_B.detach()], 1))
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()
        
        # 생성자 업데이트  
        self.optimizer_G.zero_grad()
        
        # GAN 손실
        pred_fake = self.netD(torch.cat([real_A, fake_B], 1))
        loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
        
        # L1 손실
        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()
        
        return {'loss_G_GAN': loss_G_GAN.item(), 
                'loss_G_L1': loss_G_L1.item(),
                'loss_D': loss_D.item()}

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 학습 예시
model = Pix2PixModel()
for epoch in range(num_epochs):
    for i, (input_img, target_img) in enumerate(dataloader):
        losses = model.optimize_parameters(input_img, target_img)
        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}: {losses}")
```

### 데이터셋 준비

Pix2Pix는 **paired 데이터**를 필요로 한다. 일반적으로 하나의 이미지 파일에 입력과 출력이 가로로 연결되어 저장된다:

```python
class AlignedDataset:
    def __init__(self, dataroot, phase='train'):
        self.dataroot = dataroot
        self.dir_AB = os.path.join(dataroot, phase)
        self.AB_paths = sorted(self._make_dataset(self.dir_AB))
        
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        
        # 이미지를 A와 B로 분할
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))      # 왼쪽 절반
        B = AB.crop((w2, 0, w, h))      # 오른쪽 절반
        
        # 변환 적용
        A = self.transform(A)
        B = self.transform(B)
        
        return {'A': A, 'B': B, 'A_paths': AB_path}
```

## 🔄 CycleGAN: Unpaired Data를 활용한 이미지 변환

CycleGAN은 2017년 Zhu et al.이 제안한 모델로, **쌍으로 이루어지지 않은(unpaired) 데이터**로도 이미지 간 변환을 학습할 수 있다는 혁신적인 아이디어를 제시했다.

### 핵심 아이디어: Cycle Consistency

CycleGAN의 핵심은 **cycle consistency**라는 개념이다. 이미지를 한 도메인에서 다른 도메인으로 변환한 후, 다시 원래 도메인으로 변환했을 때 원본과 동일해야 한다는 제약 조건이다.

> **Cycle Consistency**: X → Y → X' 에서 X'가 원본 X와 동일해야 한다는 원리로, 이를 통해 unpaired 데이터로도 의미 있는 변환을 학습할 수 있다. {: .prompt-tip}

### 수학적/이론적 표현

CycleGAN의 전체 목적함수는 다음과 같다:

$$ \mathcal{L}_{total} = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda \mathcal{L}_{cyc}(G, F) $$

**Adversarial Loss**: $$ \mathcal{L}_{GAN}(G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{data}(y)}[\log D_Y(y)] + \mathbb{E}_{x \sim p_{data}(x)}[\log(1 - D_Y(G(x)))] $$

**Cycle Consistency Loss**: $$ \mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1] $$

- **G**: X → Y 변환 생성자
- **F**: Y → X 변환 생성자
- **D_X, D_Y**: 각 도메인의 판별자
- **λ**: cycle consistency loss 가중치

### 아키텍처 구조

CycleGAN은 **4개의 네트워크**로 구성된다:

1. **Generator G** (X → Y): ResNet-9 blocks 사용
2. **Generator F** (Y → X): ResNet-9 blocks 사용
3. **Discriminator D_X**: X 도메인 이미지 판별
4. **Discriminator D_Y**: Y 도메인 이미지 판별

[시각적 표현 넣기 - CycleGAN 아키텍처 다이어그램]

### 실제 구현 예시

```python
class CycleGANModel:
    def __init__(self):
        # 4개 네트워크 초기화
        self.netG_A = self._define_generator()  # A → B
        self.netG_B = self._define_generator()  # B → A
        self.netD_A = self._define_discriminator()
        self.netD_B = self._define_discriminator()
        
        # 손실 함수 (LSE GAN 사용)
        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = nn.L1Loss()
        
        # 옵티마이저
        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999))
        
        # 이전 이미지 저장용 풀
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)
    
    def forward(self, real_A, real_B):
        # Forward cycle: A → B → A
        self.fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(self.fake_B)
        
        # Backward cycle: B → A → B  
        self.fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(self.fake_A)
    
    def optimize_parameters(self, real_A, real_B):
        # Forward pass
        self.forward(real_A, real_B)
        
        # 생성자 업데이트
        self.optimizer_G.zero_grad()
        
        # GAN 손실
        loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), 
                                    torch.ones_like(self.netD_B(self.fake_B)))
        loss_G_B = self.criterionGAN(self.netD_A(self.fake_A),
                                    torch.ones_like(self.netD_A(self.fake_A)))
        
        # Cycle consistency 손실
        loss_cycle_A = self.criterionCycle(self.rec_A, real_A) * 10.0
        loss_cycle_B = self.criterionCycle(self.rec_B, real_B) * 10.0
        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        self.optimizer_G.step()
        
        # 판별자 업데이트
        self.optimizer_D.zero_grad()
        
        # 판별자 A
        pred_real_A = self.netD_A(real_A)
        loss_D_real_A = self.criterionGAN(pred_real_A, torch.ones_like(pred_real_A))
        
        fake_A = self.fake_A_pool.query(self.fake_A)
        pred_fake_A = self.netD_A(fake_A.detach())
        loss_D_fake_A = self.criterionGAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        
        # 판별자 B (동일한 방식)
        # ... 판별자 B 코드 ...
        
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        self.optimizer_D.step()
```

### Image Pool 기법

CycleGAN에서는 **Image Pool**이라는 중요한 기법을 사용한다:

```python
class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []
    
    def query(self, images):
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # 50% 확률로 풀에서 선택
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return torch.cat(return_images, 0)
```

> Image Pool은 과거에 생성된 이미지들을 저장해 두었다가 판별자 학습에 활용하는 기법으로, **판별자가 과거의 실수를 잊지 않도록** 도와준다. {: .prompt-tip}

## 🔍 Pix2Pix vs CycleGAN 비교

|특징|Pix2Pix|CycleGAN|
|---|---|---|
|**데이터 요구사항**|Paired 데이터 필요|Unpaired 데이터 가능|
|**네트워크 수**|2개 (G, D)|4개 (G_A, G_B, D_A, D_B)|
|**주요 손실**|GAN Loss + L1 Loss|GAN Loss + Cycle Consistency|
|**생성자 구조**|U-Net|ResNet with skip connections|
|**학습 안정성**|상대적으로 안정|더 복잡한 학습 과정|
|**적용 분야**|구조적 변환 (스케치→사진)|스타일 변환 (말→얼룩말)|

## 🎯 실제 활용 사례

### Pix2Pix 활용 사례

- **건축/도시계획**: 건물 라벨맵 → 실제 건물 사진
- **의료 영상**: 해부학적 구조 → MRI/CT 이미지
- **디자인 도구**: 스케치 → 제품 렌더링
- **게임 개발**: 2D 맵 → 3D 환경

### CycleGAN 활용 사례

- **예술/창작**: 사진 → 그림체 변환 (모네, 반 고흐 스타일)
- **계절 변환**: 여름 풍경 → 겨울 풍경
- **동물 변환**: 말 → 얼룩말
- **스타일 변환**: 낮 → 밤, 맑은 날씨 → 비 오는 날씨

## ⚠️ 한계점과 개선 방향

### 공통 한계점

- **모드 붕괴(Mode Collapse)**: 다양성 부족 문제
- **계산 비용**: 높은 메모리와 연산 요구량
- **불안정한 학습**: 생성자와 판별자 균형 맞추기 어려움

### 최신 개선 기법

```python
# Spectral Normalization으로 학습 안정화
import torch.nn.utils.spectral_norm as spectral_norm

class StabilizedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        # ... 추가 레이어

# Progressive Growing
class ProgressiveGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0  # fade-in 파라미터
        # 점진적으로 해상도 증가
```

## 🚀 최신 발전 동향

### 최신 연구 방향

- **Few-shot 학습**: 적은 데이터로 효과적인 변환
- **다중 도메인 변환**: StarGAN, MUNIT 등
- **고해상도 변환**: SPADE, GauGAN 등
- **실시간 변환**: 모바일/웹 환경 최적화

### 실무 적용을 위한 팁

```python
# 학습 안정화를 위한 팁
def train_with_curriculum(model, dataloader, epochs):
    # 점진적 학습률 감소
    scheduler = torch.optim.lr_scheduler.LinearLR(
        model.optimizer_G, start_factor=1.0, end_factor=0.1, total_iters=epochs//2
    )
    
    for epoch in range(epochs):
        for batch in dataloader:
            # 정기적인 검증
            if epoch % 10 == 0:
                model.eval()
                validate_model(model, test_batch)
                model.train()
            
            losses = model.optimize_parameters(batch)
            
        scheduler.step()
        
        # 모델 체크포인트 저장
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
```

> Pix2Pix와 CycleGAN은 이미지 생성 분야의 **패러다임을 바꾼 혁신적인 기술**로, 현재도 다양한 분야에서 활발히 연구되고 응용되고 있다. 특히 **데이터 요구사항과 문제 특성**에 따라 적절한 모델을 선택하는 것이 중요하다. {: .prompt-tip}